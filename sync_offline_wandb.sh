#!/bin/bash

while true; do
    echo "Syncing W&B offline run..."
    wandb sync /home/zeshengj/projects/def-mheywood/zeshengj/test/CS336_assignment1/wandb/offline-run-20250530_230252-d8unlibb
    wandb sync /home/zeshengj/projects/def-mheywood/zeshengj/test/CS336_assignment1/wandb/offline-run-20250530_235724-3augw6jv
    wandb sync /home/zeshengj/projects/def-mheywood/zeshengj/test/CS336_assignment1/wandb/offline-run-20250531_000814-kmeu382p
    wandb sync /home/zeshengj/projects/def-mheywood/zeshengj/test/CS336_assignment1/wandb/offline-run-20250531_014935-gbzjleio
    echo "Waiting 30 seconds..."
    sleep 10
done