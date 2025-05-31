#!/bin/bash
#SBATCH --account=def-mheywood
#SBATCH --time=0-0:55:00                # Up to 10 hours
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --mail-user=zs549061@dal.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=cs336_train
#SBATCH --output=logs/train_%j.out       # Save stdout to logs/train_JOBID.out

module load python/3.11
cd /project/def-mheywood/zeshengj/test/CS336_assignment1/
source .venv/bin/activate
uv run cs336_basics/train.py