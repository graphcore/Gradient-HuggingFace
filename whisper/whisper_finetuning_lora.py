#!/usr/bin/env python
# coding: utf-8

import os

# Generic imports
import numpy as np
import torch

import argparse
import wandb

from datasets import load_dataset, Audio, Dataset, DatasetDict

import evaluate
from transformers import WhisperForConditionalGeneration
from transformers.integrations import WandbCallback
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import get_peft_model, LoraConfig


params_to_wandb = {"lora": (True, bool),
                   "lora_attention_dimension": (16, int),
                   "lora_target_q_proj": (True, bool),
                   "lora_target_k_proj": (False, bool),
                   "lora_target_v_proj": (True, bool),
                   "lora_target_out_proj": (False, bool),
                   "lora_target_fc1": (False, bool),
                   "lora_target_fc2": (False, bool),
                   "wandb": (False, bool),
                   "steps": (8000, int),
                   "learning_rate": (1e-4, float),
                   "whisper_size": ("small", str),
                   "fp16": (True, bool),
                   "gpu": (False, bool)}

parser = argparse.ArgumentParser()
for param in params_to_wandb:
    default, param_type = params_to_wandb[param]
    if param_type == bool:
        parser.add_argument('--' + param, dest=param, default=default, action='store_true')
        parser.add_argument('--no_' + param, dest=param, action='store_false')
    else:
        parser.add_argument('--' + param, default=default, type=param_type)

args = parser.parse_args()

print(args)

if not args.gpu:
    # IPU-specific imports
    from optimum.graphcore import (
        IPUConfig, 
        IPUSeq2SeqTrainer, 
        IPUSeq2SeqTrainingArguments, 
    )
    from optimum.graphcore.models.whisper import WhisperProcessorTorch

    n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 4))
    executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/whisper"
else:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor


if args.wandb:
    wandb_config = {}
    for param in params_to_wandb:
        wandb_config[param] = getattr(args, param)

    wandb.init(project="whisper_finetuning_lora", entity="pse", config=wandb_config)



dataset = DatasetDict()
split_dataset = Dataset.train_test_split(
    load_dataset("openslr", "SLR69", split="train", token=False), test_size=0.2, seed=0
)

dataset["train"] = split_dataset["train"] #.shuffle(seed=42)
dataset["eval"] = split_dataset["test"] #.shuffle(seed=42)
    
print(dataset)

dataset = dataset.remove_columns(["path"])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

MODEL_NAME = "whisper-" + args.whisper_size
MODEL_PATH = "openai/" + MODEL_NAME
LANGUAGE = "spanish"
TASK = "transcribe"
MAX_LENGTH = 224

if not args.gpu:
    processor = WhisperProcessorTorch.from_pretrained(MODEL_PATH, language=LANGUAGE, task=TASK)
else:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language=LANGUAGE, task=TASK)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.max_length = MAX_LENGTH
processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)


def prepare_dataset(batch, processor):
    inputs = processor.feature_extractor(
        raw_speech=batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"],
    )
    if args.fp16:
        batch["input_features"] = inputs.input_features[0].astype(np.float16)
    else:
        batch["input_features"] = inputs.input_features[0].astype(np.float32)

    transcription = batch["sentence"]
    batch["labels"] = processor.tokenizer(text=transcription).input_ids
    return batch

columns_to_remove = dataset.column_names["train"]
dataset = dataset.map(
    lambda elem: prepare_dataset(elem, processor),
    remove_columns=columns_to_remove,
    num_proc=1,
)

train_dataset = dataset["train"]
eval_dataset = dataset["eval"]

@dataclass
class DataCollatorSpeechSeq2SeqWithLabelProcessing:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch["input_features"] = torch.tensor([feature["input_features"] for feature in features])
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding="longest", pad_to_multiple_of=MAX_LENGTH)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


metric = evaluate.load("wer")


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    normalized_pred_str = [tokenizer._normalize(pred).strip() for pred in pred_str]
    normalized_label_str = [tokenizer._normalize(label).strip() for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    normalized_wer = 100 * metric.compute(predictions=normalized_pred_str, references=normalized_label_str)

    return {"wer": wer, "normalized_wer": normalized_wer}


model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
if args.gpu:
    model = model.to("cuda")
    if args.fp16:
        model = model.half()
print(model)

model.config.max_length = MAX_LENGTH
model.generation_config.max_length = MAX_LENGTH


if args.lora:
    target_modules = []
    if args.lora_target_q_proj:
        target_modules.append("q_proj")
    if args.lora_target_k_proj:
        target_modules.append("k_proj")
    if args.lora_target_v_proj:
        target_modules.append("v_proj")
    if args.lora_target_out_proj:
        target_modules.append("out_proj")
    if args.lora_target_fc1:
        target_modules.append("fc1")
    if args.lora_target_fc2:
        target_modules.append("fc2")

    print(f'Target modules: {target_modules}')

    config = LoraConfig(
        r=args.lora_attention_dimension,
        lora_alpha=32, 
        target_modules=target_modules, 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # layers_per_ipu depending on q k v out

    layers_per_ipu = [5, 7, 5, 7] # [6, 6, 6, 6]
    matmul_proportion = [0.2, 0.2, 0.6, 0.6] #[.25, .25, .25, .25]
    gradient_accumulation_steps = 16 #2 * len(layers_per_ipu) + 2
else:
    layers_per_ipu = [5, 7, 5, 7]
    matmul_proportion = [0.2, 0.2, 0.6, 0.6]
    gradient_accumulation_steps = 16


model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
model.generation_config.suppress_tokens = []

callbacks = []
if args.wandb:
    callbacks.append(WandbCallback())

# Common (IPU/GPU) training arguments
total_steps = args.steps
output_dir = f"./{MODEL_NAME}-ipu-checkpoints"
do_train = True
do_eval = True
warmup_steps = 0 #total_steps // 4
evaluation_strategy = "steps"
eval_steps = 1000
save_strategy = "steps"
logging_steps = 25
dataloader_num_workers = 16
dataloader_drop_last = True
remove_unused_columns = False

if not args.gpu:
    replication_factor = 1 #n_ipu // 4
    ipu_config = IPUConfig.from_dict(
        {
            "optimizer_state_offchip": True,
            "recompute_checkpoint_every_layer": True,
            "enable_half_partials": True,
            "executable_cache_dir": executable_cache_dir,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "replication_factor": replication_factor,
            "layers_per_ipu": layers_per_ipu,
            "matmul_proportion": matmul_proportion,
            "projection_serialization_factor": 5,
            "inference_replication_factor": 1,
            "inference_layers_per_ipu": [12, 12],
            "inference_parallelize_kwargs": {
                "use_cache": True,
                "use_encoder_output_buffer": True,
                "on_device_generation_steps": 16,
            }
        }
    )
    total_steps = total_steps // replication_factor

    training_args = IPUSeq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=do_train,
        do_eval=do_eval,
        predict_with_generate=True,
        learning_rate=args.learning_rate * replication_factor,
        warmup_steps=warmup_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        max_steps=total_steps,
        save_strategy=save_strategy,
        save_steps=total_steps,
        logging_steps=logging_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=dataloader_drop_last,
        remove_unused_columns=remove_unused_columns,
    )
    trainer = IPUSeq2SeqTrainer(
        model=model,
        ipu_config=ipu_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithLabelProcessing(processor),
        compute_metrics=lambda x: compute_metrics(x, processor.tokenizer),
        tokenizer=processor.feature_extractor,
        callbacks=callbacks
    )
else:
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=do_train,
        do_eval=do_eval,
        predict_with_generate=True,
        learning_rate=args.learning_rate,
        warmup_steps=total_steps // 4,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        max_steps=total_steps,
        save_strategy=save_strategy,
        save_steps=total_steps,
        logging_steps=logging_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=dataloader_drop_last,
        fp16=args.fp16,
        remove_unused_columns=remove_unused_columns,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithLabelProcessing(processor),
        compute_metrics=lambda x: compute_metrics(x, processor.tokenizer),
        tokenizer=processor.feature_extractor,
        callbacks=callbacks
    )

trainer.train()
