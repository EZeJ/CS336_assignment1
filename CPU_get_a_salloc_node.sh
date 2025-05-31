#!/bin/bash
#SBATCH --account=def-mheywood
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=50G
#SBATCH --mail-user=zs549061@dal.ca
#SBATCH --mail-type=BEGIN

# 👇 Define your own wait time in minutes here
WAIT_MINUTES=8000

# Convert to seconds
WAIT_SECONDS=$((WAIT_MINUTES * 60))

echo "Job started on $HOSTNAME"
echo "Sleeping for $WAIT_MINUTES minutes ($WAIT_SECONDS seconds)..."

sleep "$WAIT_SECONDS"