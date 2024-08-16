#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=60-00:00:00
#export GPUS_PER_NODE=1
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_PORT=9901



MODEL_PATH=""
DATASET_PATH=""
BASE_OUTPUT_PATH=""



python ../../test_acc_fix.py --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$BASE_OUTPUT_PATH" \
    --lora_adapter_path "$LORA_ADAPTER_PATH" \
