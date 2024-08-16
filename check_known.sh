#!/bin/bash
#SBATCH --job-name=check_known
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:1
#SBATCH --time=60-00:00:00   


python path/to/dataset/dir P740.train.json