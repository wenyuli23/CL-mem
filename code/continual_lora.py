import argparse
import math
import os
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
from huggingface_hub import login
login()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # optional: pick a single GPU
MODEL_NAME = "Qwen/Qwen2.5-3B"

EXPERIMENTS: Dict[str, List[str]] = {
    # Adapt only the attention projections.
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Adapt only the feed-forward network (SwiGLU) projections.
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

# Default training hyperparameters.
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
        help="Number of epochs to train per task.",
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
        help="Use 16-bit precision during training.",
    )
    # NEW: continual learning controls
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=3,
        help="Number of sequential tasks to split the training data into.",
    )
    parser.add_argument(
        "--shuffle_before_split",
        action="store_true",
        help="Shuffle training data before splitting into tasks.",
    )
    parser.add_argument(
        "--task_seed",
        type=int,
        default=42,
        help="Random seed used when shuffling before splitting into tasks.",
    )
    return parser.parse_args()


def filter_wikipedia_examples(dataset: datasets.Dataset) -> datasets.Dataset:
    """Return only examples where at least one URL contains 'wikipedia.org'."""
    def is_wikipedia(example: Dict[str, Any]) -> bool:
        urls = example.get("urls") or []
        if isinstance(urls, str):
            urls_list = [u.strip() for u in urls.split(",")]
        else:
            urls_list = urls
        return any("wikipedia.org" in url for url in urls_list)

    return dataset.filter(is_wikipedia)


def build_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """Create a prompt–response pair from a SimpleQA example."""
    prompt = f"Question: {example['problem']}\nAnswer:"
    answer = example["answer"]
    return {"prompt": prompt, "answer": answer}


def prepare_dataset() -> Dict[str, datasets.Dataset]:
    """Load and preprocess the SimpleQA Verified dataset."""
    raw = datasets.load_dataset("google/simpleqa-verified", split="eval")
    wiki_subset = filter_wikipedia_examples(raw)
    processed = wiki_subset.map(build_prompt, remove_columns=wiki_subset.column_names)
    return {
        "train": processed,
        "eval": processed,  # For demonstration we reuse the subset for evaluation.
    }


def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> Dict[str, Any]:
    """Tokenize the prompt and answer for language model training."""
    inputs = [p + " " + a for p, a in zip(examples["prompt"], examples["answer"])]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    labels = model_inputs["input_ids"].copy()
    # Mask out the prompt portion so loss is computed only on the answer.
    for i, (prompt, answer) in enumerate(zip(examples["prompt"], examples["answer"])):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # +1 to roughly account for the space between prompt and answer.
        prompt_length = len(prompt_ids) + 1
        labels[i][:prompt_length] = [-100] * prompt_length

    model_inputs["labels"] = labels
    return model_inputs


def split_dataset_into_tasks(
    dataset: datasets.Dataset,
    num_tasks: int,
    shuffle: bool = False,
    seed: int = 42,
) -> List[datasets.Dataset]:
    """
    Split a dataset into num_tasks sequential shards for continual learning.

    If shuffle=True, the dataset is shuffled once before splitting.
    """
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    total = len(dataset)
    task_size = math.ceil(total / num_tasks)
    tasks: List[datasets.Dataset] = []

    for k in range(num_tasks):
        start = k * task_size
        end = min((k + 1) * task_size, total)
        if start >= end:
            break
        # indices must be absolute indices into the original dataset
        indices = list(range(start, end))
        tasks.append(dataset.select(indices))

    return tasks


def run_experiment(
    name: str,
    target_modules: List[str],
    datasets_dict: Dict[str, datasets.Dataset],
    args: argparse.Namespace,
) -> None:
    """
    Run one LoRA fine-tuning experiment in a continual learning setting.

    The LoRA-wrapped model is trained sequentially on a list of tasks obtained
    by splitting the tokenized training set.
    """
    print(f"\n=== Running experiment {name} adapting modules: {target_modules} ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    # Build LoRA configuration (same adapter kept across tasks).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize datasets.
    tokenized_datasets: Dict[str, datasets.Dataset] = {}
    tokenized_datasets["train"] = datasets_dict["train"].map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=["prompt", "answer"],
    )
    tokenized_datasets["eval"] = datasets_dict["eval"].map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=["prompt", "answer"],
    )

    if args.max_train_samples:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(args.max_train_samples)
        )
    if args.max_eval_samples:
        tokenized_datasets["eval"] = tokenized_datasets["eval"].select(
            range(args.max_eval_samples)
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["eval"]

    # Split into continual tasks.
    task_datasets = split_dataset_into_tasks(
        train_dataset,
        num_tasks=args.num_tasks,
        shuffle=args.shuffle_before_split,
        seed=args.task_seed,
    )
    print(
        f"Continual setup: {len(task_datasets)} tasks, "
        f"total train examples = {len(train_dataset)}"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train sequentially on each task with the same LoRA adapter.
    for task_idx, task_ds in enumerate(task_datasets):
        task_id = task_idx + 1
        experiment_output = os.path.join(args.output_dir, name, f"task_{task_id}")
        os.makedirs(experiment_output, exist_ok=True)

        print(
            f"\n--- Task {task_id}/{len(task_datasets)} "
            f"({len(task_ds)} examples) -> {experiment_output} ---"
        )

        training_args = TrainingArguments(
            output_dir=experiment_output,
            evaluation_strategy="steps",
            eval_steps=200,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            remove_unused_columns=True,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            logging_steps=DEFAULT_TRAIN_ARGS["logging_steps"],
            warmup_steps=DEFAULT_TRAIN_ARGS["warmup_steps"],
            weight_decay=DEFAULT_TRAIN_ARGS["weight_decay"],
            save_steps=DEFAULT_TRAIN_ARGS["save_steps"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=task_ds,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        # Save the current state of the LoRA adapter after this task.
        model.save_pretrained(experiment_output)

    print(f"\n>>> Finished continual training for experiment '{name}'. <<<")


def main() -> None:
    args = parse_args()
    datasets_dict = prepare_dataset()
    print(f"Loaded {len(datasets_dict['train'])} examples with Wikipedia grounding.")

    for exp_name in args.experiments:
        target_modules = EXPERIMENTS[exp_name]
        run_experiment(exp_name, target_modules, datasets_dict, args)


if __name__ == "__main__":
    main()