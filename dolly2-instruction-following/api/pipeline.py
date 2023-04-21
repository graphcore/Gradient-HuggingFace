# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Union, List, Optional, Any

import logging
from scipy.special import softmax

import popxl
from inference import inference
from modelling.embedding import DollyEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from popxl_addons import timer
from popxl_addons.array_munging import tensor_parallel_input, repeat
from config import DollyConfig

from transformers.models.gpt_neox import GPTNeoXForCausalLM
from transformers import AutoTokenizer

import popxl
import popart
import numpy as np
import time

# Prompt format code from https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py
# Licensed under Apache 2.0 see https://github.com/databrickslabs/dolly/blob/master/LICENSE
# Instruction finetuned models need a specific prompt
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "The instruction below describes a task. Write a response that appropriately completes the request."

# This is the prompt that is used for generating responses using an already-trained model. It ends with the response
# key, where the job of the model is to provide the completion that follows it (which means the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def format_prompts(prompt: Union[str, List[str]]):
    if isinstance(prompt, str):
        prompt = [prompt]

    # iterate over prompts and apply prompt template
    return [PROMPT_FOR_GENERATION_FORMAT.format(instruction=p) for p in prompt]


def tokenize_initial(prompt: List[str], tokenizer: AutoTokenizer, config: DollyConfig):
    tokenizer.padding_side = "right"

    tokenizer_result = tokenizer(prompt, return_length=True)
    tokenized_prompt = tokenizer_result.input_ids

    # we want to obtain the real unpadded length from the tokenizer, hence we tokenize without padding, then pad later.
    tokenized_length = np.asarray(tokenizer_result.length, dtype=np.int32)

    padded_prompt = np.full(
        (
            len(prompt),
            config.model.sequence_length,
        ),
        tokenizer.pad_token_id,
    )

    # length can vary, hence we iterate over each prompt.
    for i in range(len(prompt)):
        padded_prompt[i, : tokenized_length[i]] = tokenized_prompt[i]

    return padded_prompt, tokenized_prompt, tokenized_length


class DollyPipeline:
    def __init__(
        self,
        config: DollyConfig,
        *args,
        hf_dolly_checkpoint: Union[str, GPTNeoXForCausalLM] = "databricks/dolly-v2-12b",
        sequence_length=None,
        micro_batch_size=None,
        print_live=True,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs,
    ) -> None:
        if sequence_length is not None:
            config.model.sequence_length = sequence_length
        if micro_batch_size is not None:
            config.execution.micro_batch_size = micro_batch_size

        # TODO: we really want to lazily compile session, however currently in PopXL we cannot detect globally whether IPU is already attached, making it possible to lock up the notebook kernel. Therefore, we compile in __init__ for now.
        logging.info(f"Creating session")
        session: popxl.Session = inference(config)
        if isinstance(hf_dolly_checkpoint, str):
            logging.info(f"Downloading '{hf_dolly_checkpoint}' pretrained weights")
            hf_model = GPTNeoXForCausalLM.from_pretrained(hf_dolly_checkpoint)
            if tokenizer is None:
                logging.info(f"Downloading '{hf_dolly_checkpoint}' tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(hf_dolly_checkpoint)
        else:
            hf_model = hf_dolly_checkpoint
        if tokenizer is None:
            raise ValueError(
                "A tokenizer needs to be passed to the pipeline if a custom checkpoint is being provided."
                "Use: AutoTokenizer.from_pretrained(model-name) to create the tokenizer."
            )

        with timer("Loading HF pretrained model to IPU"):
            weights = hf_mapping_lm_tp(config, session, hf_model)
            session.write_variables_data(weights)

        eos_token_id = tokenizer.encode(END_KEY)
        assert len(eos_token_id) == 1
        tokenizer.eos_token_id = eos_token_id[0]

        self.tokenizer = tokenizer
        self.pretrained = hf_model
        self.config = config
        self.session = session
        self.print_live = print_live
        self.decoded_result = None
        self.last_instruction_prompt = None

    def next_token(self, inputs, lengths, temperature, k):
        shards = self.config.execution.tensor_parallel * self.config.execution.data_parallel

        parallel_inputs = tensor_parallel_input(
            inputs, shards, shards, lambda t, i: DollyEmbeddingsTP.offset_input(t, i, self.config)
        )
        # tensor_parallel_input will squeeze out the dim if len(inputs) == 1, so we must expand_dim again.
        if inputs.shape[0] == 1:
            parallel_inputs = np.expand_dims(parallel_inputs, axis=1)
        next_token_logits = self.session.run(
            {
                self.session.inputs.words: parallel_inputs,
                self.session.inputs.last_token_indices: repeat(np.array(lengths - 1), shards, axis=0),
            }
        )[self.session.outputs.next_token_logits][
            0
        ]  # extract 0th replica as all are identical

        if k:
            # TODO: vectorize below. This was in preparation for batching, which comes in a later release.
            topk_shape = (next_token_logits.shape[0], k)
            topk_logits = np.empty(topk_shape)
            topk_idx = np.empty(topk_shape, dtype=np.int32)
            for i in range(next_token_logits.shape[0]):
                topk_idx[i] = np.argpartition(next_token_logits[i], -k)[-k:]
                topk_logits[i] = next_token_logits[i, topk_idx[i]]
            next_token_logits = topk_logits

        if temperature > 0:
            next_token_prob = softmax(next_token_logits.astype(np.float32) / temperature, axis=-1)
            # TODO: vectorize below. This was in preparation for batching, which comes in a later release.
            next_token_id = np.asarray(
                [
                    np.random.choice(next_token_logits.shape[-1], p=next_token_prob[i])
                    for i in range(next_token_prob.shape[0])
                ]
            )
        else:  # mathematically equivalent to temperature = 0
            next_token_id = next_token_logits.argmax(axis=-1)

        if k:
            # retrieve real token ids from top_k subset.
            next_token_id = topk_idx[range(next_token_logits.shape[0]), next_token_id]

        return next_token_id

    """
        Run Dolly 2.0 inference loop on a `str` prompt, or a list of prompts.

        prompt: Union[str, List[str]], prompt or list of prompts to run inference on.
        temperature: float, control sampling temperature by dividing logits by this value. For temperature = 0 where argmax sampling is used instead
        k: int, limits random sampling to top `k` most probably tokens. For `k=0` equivalent to `k=vocab_size`.
        output_length: Optional[int], maximum number of tokens to sample. Cannot exceed `sequence_length - output_length`. Defaults to maximum possible value.
        print_live: Optional[bool], whether to print the tokens one-by-one as they are decoded. `None` results in automatic behaviour depending on batch size.
    """

    def __call__(
        self,
        prompt: Union[str, List[str]],
        *args,
        temperature: float = 1.0,
        k: int = 5,
        output_length: Optional[int] = None,
        print_live: Optional[bool] = None,
    ):
        assert 0.0 <= temperature <= 1.0, "Temperature must be a float value in the range [0, 1]."
        assert (
            0 <= k <= self.config.model.embedding.vocab_size
        ), f"top k value must be in the range [0, vocab_size] (maximum = {self.config.model.embedding.vocab_size})"
        original_prompt = prompt if isinstance(prompt, list) else [prompt]
        prompt = format_prompts(prompt)
        N = len(prompt)  # record original number of prompts so we can remove padding later

        # default to class print live if batch size == 1. For batching override to false. Can override to true by passing to this fn
        if print_live is None and len(prompt) > 1:
            print_live = False
        else:
            print_live = self.print_live

        # Preprocess the data including batching it
        micro_batch = self.config.execution.micro_batch_size
        assert (
            len(prompt) <= micro_batch
        ), f"Number of prompts greater than session batch size! Got {len(prompt)} but expected no more than {self.config.execution.micro_batch_size}"

        # Create a mask to show when a specific batch entry has finished sampling.
        # Padding elements begin already complete.
        complete_mask = np.asarray([False] * len(prompt) + [True] * (micro_batch - len(prompt)), dtype=bool)

        # Apply padding to batch.
        prompt = prompt + [""] * (micro_batch - len(prompt))

        logging.info("Attach to IPUs")
        self.session.__enter__()
        logging.info("Start inference")

        padded_prompt, _, tokenized_length = tokenize_initial(prompt, self.tokenizer, self.config)
        self.last_instruction_prompt = prompt
        num_generated = 0
        result = [[] for _ in range(len(prompt))]

        if output_length is None:
            output_length = self.config.model.sequence_length - max(tokenized_length)
        assert 1 <= output_length <= self.config.model.sequence_length - max(tokenized_length)
        if print_live:
            print(f"Input prompt: {original_prompt[0]}")
            print("Response:")

        start_time = time.time()
        for _ in range(output_length):
            next_tokens = self.next_token(padded_prompt, tokenized_length, temperature, k)

            # update mask based on whether EOS was sampled and whether maximum length was exceeded
            complete_mask = complete_mask | (next_tokens == self.tokenizer.eos_token_id)
            complete_mask = complete_mask | (tokenized_length >= self.config.model.sequence_length)

            if complete_mask.all():
                break

            for i, t in enumerate(next_tokens):
                if complete_mask[i]:
                    continue
                result[i].append(t)

            padded_prompt[
                range(len(prompt)), tokenized_length
            ] = next_tokens  # update final elements in each batch element with next token
            tokenized_length[~complete_mask] += 1  # update length by one for elements that are not complete
            num_generated += len(prompt)

            # TODO: anyway to preview entire batch? For now just do first ~
            if print_live and not complete_mask[0]:
                print(self.tokenizer.decode(next_tokens[0]), end="", flush=True)
        end_time = time.time()

        self.decoded_result = self.tokenizer.batch_decode(result)[:N]  # unpad result

        if print_live:
            print("")
            print("Final output:")
            print(f"{self.decoded_result[0]}")
            print(f"Output in {end_time - start_time:.2f} seconds")
            print(f"Throughput: {num_generated / (end_time - start_time):.2f} t/s")

        return self.decoded_result

    def detach(self):
        was_attached_or_device = self.session._was_attached_stack.pop()

        # self.session.weights_to_host()
        self.session._device.detach()
        self.session._pb_session.setEngineIsLoaded(False)

        # If a DeviceInfo was stored in the stack then restore it.
        if isinstance(was_attached_or_device, popart.DeviceInfo):
            self.session._set_device(was_attached_or_device)


if __name__ == "__main__":
    from utils.setup import dolly_config_setup

    config, _, hf_model = dolly_config_setup("config/inference.yml", "release", "dolly_pod4", hf_model_setup=True)
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    pipe = DollyPipeline(config, hf_dolly_checkpoint=hf_model, tokenizer=tokenizer)

    print(pipe(["Can sheep survive on the moon?"], k=5))
