#!/bin/bash

while true; do
    echo "Syncing W&B offline run..."
    wandb sync /home/zeshengj/projects/def-mheywood/zeshengj/test/CS336_assignment1/wandb/offline-run-20250530_230252-d8unlibb
    wandb sync /home/zeshengj/projects/def-mheywood/zeshengj/test/CS336_assignment1/wandb/offline-run-20250530_235724-3augw6jv
    echo "Waiting 30 seconds..."
    sleep 30
done