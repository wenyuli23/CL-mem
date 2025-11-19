#!/usr/bin/env python

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any

import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0
MODEL_NAME = "Qwen/Qwen2.5-3B"
CUDA_LAUNCH_BLOCKING=1

# LoRA experiments to run.  Each key corresponds to a subdirectory under
# ``--output_dir``.  The value is the list of module names to adapt.
# ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj`` correspond to the
# query/key/value/output projection matrices in the attention layers.
# ``up_proj``, ``gate_proj`` and ``down_proj`` correspond to the
# feed‑forward (MLP) layers.  See Qwen documentation for details【200665441583520†L220-L305】.
EXPERIMENTS: Dict[str, List[str]] = {
    # Adapt only the attention projections.
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Adapt only the feed‑forward network (SwiGLU) projections.
    "ffn": ["up_proj", "gate_proj", "down_proj"],
    # Adapt all projection matrices (attention + FFN).
    "all": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ],
    # A minimal configuration adapting only the query and value projections.
    "qv_only": ["q_proj", "v_proj"],
}

# Training hyperparameters.  Adjust as needed.
DEFAULT_TRAIN_ARGS = dict(
    learning_rate=1e-4,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    weight_decay=0.0,
    logging_steps=50,
    save_steps=200,
    fp16=True,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_qwen_simpleqa",
        help="Where to save the resulting LoRA adapters.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, only use this many examples for training (useful for quick experiments).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="If set, only use this many examples for evaluation.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        choices=list(EXPERIMENTS.keys()),
        default=list(EXPERIMENTS.keys()),
        help="Subset of LoRA experiments to run (default: run all).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_TRAIN_ARGS["learning_rate"],
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_TRAIN_ARGS["num_train_epochs"],
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_TRAIN_ARGS["per_device_train_batch_size"],
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=DEFAULT_TRAIN_ARGS["per_device_eval_batch_size"],
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_TRAIN_ARGS["gradient_accumulation_steps"],
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=DEFAULT_TRAIN_ARGS["fp16"],
        help="Use 16‑bit precision during training.",
    )
    return parser.parse_args()


def filter_wikipedia_examples(dataset: datasets.Dataset) -> datasets.Dataset:
    """Return only examples where at least one URL contains "wikipedia.org"."""
    def is_wikipedia(example: Dict[str, Any]) -> bool:
        urls = example.get("urls") or []
        # The ``urls`` field may be a string with comma‑separated URLs or a list.
        if isinstance(urls, str):
            urls_list = [u.strip() for u in urls.split(",")]
        else:
            urls_list = urls
        return any("wikipedia.org" in url for url in urls_list)

    return dataset.filter(is_wikipedia)


def build_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """Create a prompt–response pair from a SimpleQA example."""
    # Use a straightforward instruction format.
    prompt = f"Question: {example['problem']}\nAnswer:"
    answer = example["answer"]
    return {"prompt": prompt, "answer": answer}


def prepare_dataset() -> Dict[str, datasets.Dataset]:
    """Load and preprocess the SimpleQA Verified dataset."""
    # Load the entire dataset (simpleqa_verified split 'eval').
    raw = datasets.load_dataset("google/simpleqa-verified", split="eval")
    # Keep only examples grounded by at least one Wikipedia URL.
    wiki_subset = filter_wikipedia_examples(raw)
    # Map into prompt/answer pairs.
    processed = wiki_subset.map(build_prompt, remove_columns=wiki_subset.column_names)
    return {
        "train": processed,
        "eval": processed  # For demonstration we reuse the subset for evaluation.
    }


def tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int = 1024) -> Dict[str, Any]:
    """Tokenize the prompt and answer for language model training."""
    # Concatenate prompt and answer for supervised fine‑tuning.
    inputs = [p + " " + a for p, a in zip(examples["prompt"], examples["answer"])]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    # The labels are the same as the input_ids; mask the prompt part to ignore its loss.
    input_ids = model_inputs["input_ids"]
    labels = [seq[:] for seq in input_ids]  # <-- deep copy each inner list
    # Identify the end of the prompt and set tokens before it to -100 so that loss only applies to the answer.
    for i, (prompt, answer) in enumerate(zip(examples["prompt"], examples["answer"])):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # +1 to include the space separating question and answer.
        prompt_length = len(prompt_ids) + 1
        labels[i][:prompt_length] = [-100] * prompt_length
    model_inputs["labels"] = labels
    return model_inputs


def run_experiment(name: str, target_modules: List[str], datasets_dict: Dict[str, datasets.Dataset], args: argparse.Namespace) -> None:
    """Run one LoRA fine‑tuning experiment and save the adapter."""
    print(f"\n=== Running experiment {name} adapting modules: {target_modules} ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Ensure the model uses the padding token; if not present, assign it.
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    # Build LoRA configuration.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank of the LoRA approximation.  Increase for stronger adaptation.
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize the dataset.
    tokenized_datasets = datasets_dict.copy()
    tokenized_datasets["train"] = tokenized_datasets["train"].map(
        lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=["prompt", "answer"]
    )
    tokenized_datasets["eval"] = tokenized_datasets["eval"].map(
        lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=["prompt", "answer"]
    )

    if args.max_train_samples:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples:
        tokenized_datasets["eval"] = tokenized_datasets["eval"].select(range(args.max_eval_samples))

    # mask padding tokens in the loss.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments.  The output directory includes the experiment name.
    experiment_output = f"{args.output_dir}/{name}"
    training_args = TrainingArguments(
        output_dir=experiment_output,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        **{
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "fp16": args.fp16,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
    )

    trainer.train()
    # Save only the LoRA adapter to disk.
    model.save_pretrained(experiment_output)


def main() -> None:
    args = parse_args()
    # Prepare the dataset once and reuse across experiments.
    datasets_dict = prepare_dataset()
    print(f"Loaded {len(datasets_dict['train'])} examples with Wikipedia grounding.")
    for exp_name in args.experiments:
        target_modules = EXPERIMENTS[exp_name]
        run_experiment(exp_name, target_modules, datasets_dict, args)


if __name__ == "__main__":
    main()