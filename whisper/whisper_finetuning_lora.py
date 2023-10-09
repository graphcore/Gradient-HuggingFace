#!/usr/bin/env python
# coding: utf-8

import os

n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 4))
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/whisper"

# Generic imports
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import load_dataset, Audio, Dataset, DatasetDict

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# IPU-specific imports
from optimum.graphcore import (
    IPUConfig, 
    IPUSeq2SeqTrainer, 
    IPUSeq2SeqTrainingArguments, 
)
from optimum.graphcore.models.whisper import WhisperProcessorTorch

# HF-related imports
from transformers import WhisperForConditionalGeneration

dummy_data = False
if dummy_data:
    test_size = 0.001
else:
    test_size = 0.2

dataset = DatasetDict()
split_dataset = Dataset.train_test_split(
    load_dataset("openslr", "SLR69", split="train", token=False), test_size=test_size, seed=0
)

if dummy_data:
    dataset["train"] = split_dataset["test"]
    dataset["eval"] = split_dataset["test"]
else:
    dataset["train"] = split_dataset["train"]
    dataset["eval"] = split_dataset["test"]
    
print(dataset)

dataset = dataset.remove_columns(["path"])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "spanish"
TASK = "transcribe"
MAX_LENGTH = 224

processor = WhisperProcessorTorch.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.max_length = MAX_LENGTH
processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)


def prepare_dataset(batch, processor):
    inputs = processor.feature_extractor(
        raw_speech=batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"],
    )
    batch["input_features"] = inputs.input_features[0].astype(np.float16)

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


model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.max_length = MAX_LENGTH
model.generation_config.max_length = MAX_LENGTH

config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
model.generation_config.suppress_tokens = []


replication_factor = 1 #n_ipu // 4
ipu_config = IPUConfig.from_dict(
    {
        "optimizer_state_offchip": True,
        "recompute_checkpoint_every_layer": True,
        "enable_half_partials": True,
        "executable_cache_dir": executable_cache_dir,
        "gradient_accumulation_steps": 4, #16,
        "replication_factor": replication_factor,
        "layers_per_ipu": [12, 12],# [5, 7, 5, 7],
        "matmul_proportion": [0.5, 0.5], # [0.2, 0.2, 0.6, 0.6],
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


total_steps = 5000 // replication_factor
training_args = IPUSeq2SeqTrainingArguments(
    output_dir="./whisper-small-ipu-checkpoints",
    do_train=True,
    do_eval=True,
    predict_with_generate=True,
    learning_rate=1e-5 * replication_factor,
    warmup_steps=total_steps // 4,
    evaluation_strategy="steps",
    eval_steps=total_steps,
    max_steps=total_steps,
    save_strategy="steps",
    save_steps=total_steps,
    logging_steps=25,
    dataloader_num_workers=16,
    dataloader_drop_last=True,
    remove_unused_columns=False,
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
)

trainer.evaluate()

trainer.train()

trainer.evaluate()

