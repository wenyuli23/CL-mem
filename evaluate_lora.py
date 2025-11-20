#!/usr/bin/env python
"""
Evaluate a trained LoRA adapter on the SimpleQA dataset (inference-only).

This script loads a LoRA adapter trained with train_qwen_lora_simpleqa.py and
evaluates it by generating answers for questions. The SimpleQA dataset only has
an 'eval' split (1000 examples), so we perform inference-only evaluation.

Metrics computed:
- Exact match accuracy (case-insensitive)
- F1 score (token-level overlap)
- Sample generation outputs

Example usage:

```
python evaluate_lora.py \
    --lora_adapter_path ./lora_qwen_simpleqa/all \
    --max_eval_samples 100 \
    --batch_size 4
```
"""

import argparse
from typing import Dict, List, Any, Set
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Same model as training
MODEL_NAME = "Qwen/Qwen3-1.7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA adapter")
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory",
    )
    parser.add_argument(
        "--baseline_adapter_path",
        type=str,
        default=None,
        help="Path to baseline adapter for comparison (e.g., before continual fine-tuning)",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Train/test split ratio (must match training script, default: 0.8)",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="Evaluate on train split instead of test split",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per answer (default: 128)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty to avoid loops (default: 1.2, higher = less repetition)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="Prevent n-gram repetitions (default: 3)",
    )
    parser.add_argument(
        "--num_samples_to_show",
        type=int,
        default=5,
        help="Number of sample outputs to display (default: 5)",
    )
    return parser.parse_args()


def filter_wikipedia_examples(dataset) -> Any:
    """Return only examples where at least one URL contains 'wikipedia.org'."""
    def is_wikipedia(example: Dict[str, Any]) -> bool:
        urls = example.get("urls") or []
        if isinstance(urls, str):
            urls_list = [u.strip() for u in urls.split(",")]
        else:
            urls_list = urls
        return any("wikipedia.org" in url for url in urls_list)
    return dataset.filter(is_wikipedia)


def prepare_test_dataset(train_split_ratio: float = 0.8, use_train: bool = False):
    """Load and split dataset the same way as training script."""
    from datasets import load_dataset
    
    raw = load_dataset("google/simpleqa-verified", split="eval")
    wiki_subset = filter_wikipedia_examples(raw)
    
    # Same split as training (with same seed for reproducibility)
    split_dict = wiki_subset.train_test_split(test_size=1.0 - train_split_ratio, seed=42)
    
    split_name = "train" if use_train else "test"
    return split_dict[split_name]


def build_prompt(example: Dict[str, Any]) -> str:
    """Create a prompt from a SimpleQA example."""
    return f"Question: {example['problem']}\nAnswer:"


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = pred_tokens & truth_tokens
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_generation(model, tokenizer, dataset, max_new_tokens: int, repetition_penalty: float = 1.2, 
                        no_repeat_ngram_size: int = 3, max_samples: int = None, num_to_show: int = 5) -> Dict[str, Any]:
    """Generate answers and compute exact match accuracy and F1 score (inference-only)."""
    model.eval()
    eval_samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    exact_matches = 0
    total_f1 = 0.0
    total = len(eval_samples)
    samples_to_display = []
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(eval_samples, desc="Generating answers")):
            prompt = build_prompt(example)
            ground_truth = example['answer'].strip().lower()
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=1.0,
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the answer part (after the prompt)
            generated_answer = generated_text[len(prompt):].strip().lower()
            
            # Check exact match (case-insensitive)
            is_match = generated_answer == ground_truth
            if is_match:
                exact_matches += 1
            
            # Compute F1 score
            f1 = compute_f1(generated_text[len(prompt):].strip(), example['answer'])
            total_f1 += f1
            
            # Save samples to display
            if i < num_to_show:
                samples_to_display.append({
                    "question": example['problem'],
                    "ground_truth": example['answer'],
                    "generated": generated_text[len(prompt):].strip(),
                    "match": is_match,
                    "f1": f1,
                })
    
    accuracy = (exact_matches / total) * 100
    avg_f1 = (total_f1 / total) * 100
    return {
        "exact_match_accuracy": accuracy,
        "exact_matches": exact_matches,
        "avg_f1": avg_f1,
        "total": total,
        "samples": samples_to_display,
    }


def main():
    args = parse_args()
    
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
    
    print(f"\nLoading test dataset (same split as training)...")
    eval_dataset = prepare_test_dataset(
        train_split_ratio=args.train_test_split,
        use_train=args.eval_on_train
    )
    split_name = "train" if args.eval_on_train else "test"
    print(f"Evaluating on {split_name} split: {len(eval_dataset)} examples")
    
    # Evaluate main adapter
    print(f"\n{'='*80}")
    print(f"EVALUATING: {args.lora_adapter_path}")
    print(f"{'='*80}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    model.eval()
    
    # Evaluate generation (inference-only)
    print("\n" + "="*80)
    print("INFERENCE-ONLY EVALUATION")
    print("="*80)
    gen_results = evaluate_generation(
        model, tokenizer, eval_dataset, 
        args.max_new_tokens, args.repetition_penalty, args.no_repeat_ngram_size,
        args.max_eval_samples, args.num_samples_to_show
    )
    
    print(f"\nExact Match Accuracy: {gen_results['exact_match_accuracy']:.2f}%")
    print(f"Exact Matches: {gen_results['exact_matches']}/{gen_results['total']}")
    print(f"Average F1 Score: {gen_results['avg_f1']:.2f}%")
    
    # Show sample outputs
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS")
    print("="*80)
    for i, sample in enumerate(gen_results['samples'], 1):
        print(f"\nExample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Ground Truth: {sample['ground_truth']}")
        print(f"Generated: {sample['generated']}")
        print(f"Match: {'✓' if sample['match'] else '✗'} | F1: {sample['f1']:.2f}")
    
    # Compare with baseline if provided
    baseline_results = None
    if args.baseline_adapter_path:
        print(f"\n{'='*80}")
        print(f"BASELINE COMPARISON: {args.baseline_adapter_path}")
        print(f"{'='*80}")
        
        # Reload base model for baseline
        baseline_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        baseline_model.resize_token_embeddings(len(tokenizer))
        baseline_model = PeftModel.from_pretrained(baseline_model, args.baseline_adapter_path)
        baseline_model.eval()
        
        baseline_results = evaluate_generation(
            baseline_model, tokenizer, eval_dataset,
            args.max_new_tokens, args.repetition_penalty, args.no_repeat_ngram_size,
            args.max_eval_samples, 0  # Don't show samples again
        )
        
        print(f"\nBaseline Exact Match: {baseline_results['exact_match_accuracy']:.2f}%")
        print(f"Baseline F1 Score: {baseline_results['avg_f1']:.2f}%")
        
        # Compute improvements
        em_improvement = gen_results['exact_match_accuracy'] - baseline_results['exact_match_accuracy']
        f1_improvement = gen_results['avg_f1'] - baseline_results['avg_f1']
        print(f"\nImprovement over baseline:")
        print(f"  Exact Match: {em_improvement:+.2f}%")
        print(f"  F1 Score: {f1_improvement:+.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA Adapter: {args.lora_adapter_path}")
    print(f"Dataset: SimpleQA ({split_name} split, Wikipedia-grounded only)")
    print(f"Exact Match Accuracy: {gen_results['exact_match_accuracy']:.2f}%")
    print(f"Average F1 Score: {gen_results['avg_f1']:.2f}%")
    print(f"Evaluated on: {gen_results['total']} examples")
    
    if baseline_results:
        em_improvement = gen_results['exact_match_accuracy'] - baseline_results['exact_match_accuracy']
        f1_improvement = gen_results['avg_f1'] - baseline_results['avg_f1']
        print(f"\nContinual Fine-Tuning Gains (vs baseline):")
        print(f"  EM Improvement: {em_improvement:+.2f}%")
        print(f"  F1 Improvement: {f1_improvement:+.2f}%")


if __name__ == "__main__":
    main()
