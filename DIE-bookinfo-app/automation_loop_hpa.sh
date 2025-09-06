#!/bin/bash

# --- Configuration ---
STGCN_IP="192.168.1.76"
JUMP_HOST_IP="192.168.1.18"
LOCUST_IP="192.168.1.146"
REMOTE_PATH_STGCN="/home/ubuntu/Carbon-Aware-AutoScaler/DeepScaler"
REMOTE_PATH_LOCUST="~/evaluation"
SSH_KEY_PATH="~/Carbon-Aware-AutoScaler/DeepScaler/train.pem"
APP_NAME="bookinfo"

START_ROUND=0
MAX_ROUND=5  # Set the last round number here
SAVE_DIR="./saved_images"

for (( ROUND=$START_ROUND; ROUND<=MAX_ROUND; ROUND++ ))
do
  echo "Starting round $ROUND..."

  # --- Step 1: Local - Start Minikube and deploy Istio ---
  echo "[Local] Starting Minikube..."
  minikube start --cpus 15 --memory 50000

  echo "[Local] Deploying Istio..."
  bash deploy_istio.sh

  # --- Step 2: HPA setup ---
  bash hpa_all.sh

  # --- Step 3: Locust - Start load test ---
  echo "[Remote: Locust] Running load test..."
  ssh -i train.pem ubuntu@$LOCUST_IP << EOF
    cd $REMOTE_PATH_LOCUST

    echo "[Locust] Activating Python environment..."
    source .venv/bin/activate

    # Use only app name and round for path
    bash load_test_${APP_NAME}.sh ${APP_NAME}/hpa/round_${ROUND}
EOF

  # --- Step 5: Local - Cleanup Minikube ---
  echo "[Local] Deleting Minikube cluster..."
  minikube delete

  echo "Round $ROUND completed."
  echo ""
done

echo "All rounds completed."
