#!/bin/bash

# --- Configuration ---
STGCN_IP="192.168.1.76"
JUMP_HOST_IP="192.168.1.18"
LOCUST_IP="192.168.1.146"
REMOTE_PATH_STGCN="/home/ubuntu/Carbon-Aware-AutoScaler/DeepScaler"
REMOTE_PATH_LOCUST="~/evaluation"
SSH_KEY_PATH="~/Carbon-Aware-AutoScaler/DeepScaler/train.pem"
# TODO
APP_NAME="bookinfo"

# Define your models, IDs, and numeric restrictions
#MODEL_NAMES=("AdapGLD" "AdapGLD" "AdapGLD" "AdapGLD" "AdapGLA" "AdapGLA" "AdapGLA" "AdapGLA")
#MODEL_IDS=("AdapGLD_fresh" "AdapGLD_fresh" "AdapGLD_fresh" "AdapGLD_fresh" "AdapGLA_fresh" "AdapGLA_fresh" "AdapGLA_fresh" "AdapGLA_fresh")
#MODEL_RESTRICTIONS=(0.35 0.4 0.3 0.12 0.6 0.78 0.44 0.77)  # Numeric restrictions

# Inference model argument (name of the model)
MODEL_NAMES=("AdapGLA" "AdapGLA" "AdapGLA" "AdapGLD" "AdapGLD" "AdapGLD" "AdapGLT" "AdapGLT" "AdapGLT")
# How you name the folder
MODEL_IDS=("AdapGLA" "AdapGLA" "AdapGLA" "AdapGLD" "AdapGLD" "AdapGLD" "AdapGLT" "AdapGLT" "AdapGLT")
MODEL_RESTRICTIONS=(0.35 0.5 0.7 0.35 0.5 0.7 0.35 0.5 0.7)  # Numeric restrictions

START_ROUND=0
MAX_ROUND=9  # Set the last round number here
SAVE_DIR="./saved_images"

# Iterate over all models
for (( MODEL_IDX=0; MODEL_IDX<${#MODEL_NAMES[@]}; MODEL_IDX++ ))
do
  MODEL_NAME=${MODEL_NAMES[MODEL_IDX]}
  MODEL_ID=${MODEL_IDS[MODEL_IDX]}
  MODEL_RESTRICTION=${MODEL_RESTRICTIONS[MODEL_IDX]}

  for (( ROUND=$START_ROUND; ROUND<=MAX_ROUND; ROUND++ ))
  do
    echo "Starting round $ROUND for model $MODEL_NAME ($MODEL_ID) with restriction $MODEL_RESTRICTION..."
    MODEL_FILE="$REMOTE_PATH_STGCN/model_${APP_NAME}/${MODEL_ID}/${MODEL_ID}.pkl"
  done
done