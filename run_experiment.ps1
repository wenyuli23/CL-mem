# Run a single LoRA fine-tuning experiment on Qwen with SimpleQA dataset
# This script trains only the "all" configuration (all attention + FFN modules)

$env:CUDA_VISIBLE_DEVICES = "0"

python train_qwen_lora_simpleqa.py `
    --output_dir ./cl_lora_qwen_simpleqa `
    --experiments all `
    --num_train_epochs 2 `
    --per_device_train_batch_size 2 `
    --per_device_eval_batch_size 2 `
    --gradient_accumulation_steps 8 `
    --learning_rate 1e-4 `
    --save_every_steps 200 `
    --save_total_limit 3 `
    --fp16

Write-Host "Training complete! LoRA adapter saved to ./cl_lora_qwen_simpleqa/all" -ForegroundColor Green
