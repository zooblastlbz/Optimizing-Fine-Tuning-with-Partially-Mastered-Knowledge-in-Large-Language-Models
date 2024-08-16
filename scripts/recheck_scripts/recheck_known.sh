#!/bin/bash
#SBATCH --partition 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=60-00:00:00
#export GPUS_PER_NODE=1
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_PORT=9901
cd 
python ../../recheck_known.py --model_path   \
    --data_source  \
    --dataset_path  \
    --output_path \
    --lora_adapter_path  \