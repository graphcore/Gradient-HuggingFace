if __name__ == "__main__":
    import os

    executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/language_modelling_from_scratch"

    from datasets import load_dataset
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    from datasets import ClassLabel
    import random
    import pandas as pd

    model_checkpoint = "gpt2"
    tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"

    ipu_config_name = "Graphcore/gpt2-small-ipu"
    micro_batch_size = 1
    gradient_accumulation_steps = 64
    dataloader_workers = 64

    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    from transformers import AutoConfig, AutoModelForCausalLM
    from optimum.graphcore import IPUConfig, IPUTrainer, IPUTrainingArguments

    ipu_config = IPUConfig.from_pretrained(ipu_config_name, executable_cache_dir=executable_cache_dir)

    config = AutoConfig.from_pretrained(model_checkpoint)
    config.update({'activation_function':'gelu'})
    model = AutoModelForCausalLM.from_config(config)

    training_args = IPUTrainingArguments(
        f"{model_checkpoint}-wikitext2",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=10,
        loss_scaling=16384,
        n_ipu=4,
        warmup_ratio=0.1,
        dataloader_drop_last=True,
        dataloader_num_workers=dataloader_workers,
        logging_steps=10,
        push_to_hub=False,
        # hub_model_id=f"username-or-organization/{model_checkpoint}-wikitext2",
    )

    from transformers import default_data_collator

    trainer = IPUTrainer(
        model=model,
        ipu_config=ipu_config,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    import math
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


    #####################################################################################################################################

    model_checkpoint = "bert-base-cased"
    tokenizer_checkpoint = "sgugger/bert-like-tokenizer"

    ipu_config_name = "Graphcore/bert-base-ipu"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    from transformers import AutoConfig, AutoModelForMaskedLM
    from optimum.graphcore import IPUConfig, IPUTrainer, IPUTrainingArguments

    ipu_config = IPUConfig.from_pretrained(ipu_config_name, executable_cache_dir=executable_cache_dir)

    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_config(config)

    training_args = IPUTrainingArguments(
        f"{model_checkpoint}-wikitext2-test-mlm",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=10,
        dataloader_drop_last=True,
        dataloader_num_workers=dataloader_workers,
        warmup_ratio=0.1,
        logging_steps=10,
        n_ipu=4,
        push_to_hub=False,
        # hub_model_id=f"username-or-organization/{model_checkpoint}-wikitext2-test-mlm",
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = IPUTrainer(
        model=model,
        args=training_args,
        ipu_config=ipu_config,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")