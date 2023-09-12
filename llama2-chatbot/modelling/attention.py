# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
import math

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard, repeat_axis
from popxl_addons.layers import Linear
from popxl_addons import remote

from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce

from .rotary_pos_embed import rotary_pos_embed, trig_table_constants

from config import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention as HFModel
from scipy.special import softmax


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape((micro_batch_size, sequence_length, heads, head_hidden_size))


class LlamaAttentionHeads(addons.Module):
    def __init__(self, config: LlamaConfig, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.config = config
        self.replica_grouping = replica_grouping

        if self.replica_grouping:
            n_heads_groups = self.replica_grouping.num_groups
        else:
            n_heads_groups = 1

        assert (
            self.config.model.attention.heads % n_heads_groups == 0
        ), f"{self.config.model.attention.heads} % {n_heads_groups} != 0"

        self.n_heads_groups = n_heads_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups

        # Llama Attention does not use bias
        self.qkv = Linear(3 * self.config.model.hidden_size // n_heads_groups, bias=False, replica_grouping=replica_grouping)
        # Rotary dims determined by hidden size and attention head as in Transformers Llama implementation. 
        # No rotary scaling percentage is implemented for the model.
        self.rotary_ndims = self.config.model.hidden_size // self.config.model.attention.heads

        # Rotary positional embeddings base value used as constant in Transformers Llama.
        self.rotary_pos_emb_base = 10000 

    def build(self, x: popxl.Tensor, past_kv: Optional[popxl.Tensor] = None):
        # x: [batch*seq, hidden]
        qkv_act = self.qkv(x)
        query, key, value = ops.split(qkv_act, 3, axis=-1)

        causal_mask = popxl.constant(
            # HF version 1e9 to mask. However, this model runs in float16 and 1e9 is beyond the float16 range, therefore 1e4 is used to similar effect.
            1e4 * (np.tril(np.ones((self.config.model.sequence_length, self.config.model.sequence_length))) - 1),
            query.dtype,
            name="causal_mask",
        )

        query = reshape_for_scores(query, self.config.model.sequence_length, self.n_heads)
        key = reshape_for_scores(key, self.config.model.sequence_length, self.n_heads)
        value = reshape_for_scores(value, self.config.model.sequence_length, self.n_heads)
        
        sin, cos = trig_table_constants(
            self.config.model.sequence_length,
            self.rotary_ndims,
            self.rotary_pos_emb_base,
            self.config.model.dtype,
        )

        query = rotary_pos_embed(query, sin, cos, self.rotary_ndims).transpose((0, 2, 1, 3))
        key = rotary_pos_embed(key, sin, cos, self.rotary_ndims).transpose((0, 2, 3, 1))
        value = value.transpose((0, 2, 1, 3))

        attn_output = self.attention_block(query, key, value, causal_mask)

        return attn_output.transpose((0, 2, 1, 3)).reshape(
            (self.config.execution.micro_batch_size * self.config.model.sequence_length, -1)
        )

    def attention_block(self, query: popxl.Tensor, key: popxl.Tensor, value: popxl.Tensor, mask: popxl.Tensor):
        attn_weights = query @ key

        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))
        attn_weights = attn_weights + mask

        if attn_weights.dtype == popxl.float16: 
            attn_weights = ops.cast(attn_weights, popxl.float32)

        attn_scores = ops.softmax(attn_weights, axis=-1)

        if attn_scores.dtype == popxl.float32: 
            attn_scores = ops.cast(attn_scores, popxl.float16)

        return attn_scores @ value


class LlamaSelfAttentionTP(addons.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        attn_tp = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )
        tp = attn_tp
        dp = config.execution.data_parallel * (config.execution.tensor_parallel // attn_tp)
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

        # Sharded across devices
        self.heads = LlamaAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        self.output = Linear(self.config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""

        # ----- Sharded computation -----
        z = self.heads(x)
        z = self.output(z)

        z = replicated_all_reduce(z, group=self.replica_grouping.transpose())

        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        attn_tp = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )

        hf_query_w = to_numpy(hf_model.q_proj.weight.data, dtype).T
        hf_key_w = to_numpy(hf_model.k_proj.weight.data, dtype).T
        hf_value_w = to_numpy(hf_model.v_proj.weight.data, dtype).T

        query_w = shard(hf_query_w, attn_tp, -1)
        key_w = shard(hf_key_w, attn_tp, -1)
        value_w = shard(hf_value_w, attn_tp, axis=-1)

        qkv_w = np.ascontiguousarray(
            np.concatenate(
                [np.concatenate([query_w[i], key_w[i], value_w[i]], axis=-1)[np.newaxis, ...] for i in range(attn_tp)]
            )
        )

        hf_out_proj_w = to_numpy(hf_model.o_proj.weight.data.T, dtype)
        out_proj_w = shard(hf_out_proj_w, attn_tp, axis=0)

        return {
                variables.heads.qkv.weight: qkv_w,
                variables.output.weight: out_proj_w,
        }
