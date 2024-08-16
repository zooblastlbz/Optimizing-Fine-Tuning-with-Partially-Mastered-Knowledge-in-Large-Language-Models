#!/bin/bash
#SBATCH --job-name=finetune-llama3
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=60-00:00:00
#export GPUS_PER_NODE=1
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_PORT=9901

bash finetune.sh -m  \
 -d  \
 --output_dir   \
 --report_to wandb \
 --num_train_epochs 15 \
 --use_lora True \
 --deepspeed  