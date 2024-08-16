#!/bin/bash
#SBATCH --job-name=finetune-Qwen
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=60-00:00:00
#export GPUS_PER_NODE=1
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9908


 torchrun $DISTRIBUTED_ARGS --master_port=$MASTER_PORT cft.py \
    --model_name_or_path /mnt/petrelfs/libozhou/sft/checkpoints/llama3-merged \
    --data_path /mnt/petrelfs/libozhou/wiki-qa/llama3/cft_data.jsonl \
    --replay_data_path /mnt/petrelfs/libozhou/wiki-qa/llama3/replay_data.jsonl \
    --replay_ratio 0.2 \
    --replay_dataset True \
    --bf16 True \
    --output_dir /mnt/petrelfs/libozhou/sft/checkpoints/llama3/checkpoints-epoch-114-merged-base-15e-5-policy6-epoch-6-15e-5-2 \
    --num_train_epochs 6 \
    --per_device_train_batch_size  32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 15e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --model_max_length 512 \
    --lazy_preprocess True \
    --use_lora True \
    --q_lora False \
    --gradient_checkpointing True \
    --deepspeed /mnt/petrelfs/libozhou/Qwen2/examples/sft/ds_config_zero2.json \
