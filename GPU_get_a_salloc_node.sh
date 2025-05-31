#!/bin/bash
#SBATCH --account=def-mheywood
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --mail-user=zs549061@dal.ca
#SBATCH --mail-type=BEGIN

# 👇 Define your own wait time in minutes here
WAIT_MINUTES=80000

# Convert to seconds
WAIT_SECONDS=$((WAIT_MINUTES * 60))

echo "Job started on $HOSTNAME"
echo "Sleeping for $WAIT_MINUTES minutes ($WAIT_SECONDS seconds)..."

sleep "$WAIT_SECONDS"


# GPU options:
# --gres=gpu:a100_3g.20gb:1
# --gres=gpu:a100 