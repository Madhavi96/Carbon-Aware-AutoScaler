LOCUST_IP="192.168.1.146"
REMOTE_PATH_LOCUST="~/evaluation"
DATASET_NAME="dataset/test"

source .venv/bin/activate

minikube start --cpus 15 --memory 50000
bash /home/ubuntu/DIE-train-ticket-app/deploy_istio.sh

bash /home/ubuntu/DIE-train-ticket-app/hpa_all.sh

start_time=$(date)

ssh -i /home/ubuntu/DIE-train-ticket-app/train.pem ubuntu@$LOCUST_IP << EOF
      cd $REMOTE_PATH_LOCUST

      echo "[Locust] Activating Python environment..."
      source .venv/bin/activate

      bash load_test_train.sh train-ticket/hpa_dataset
EOF
end_time=$(date)
python main.py "$start_time" "$end_time" $DATASET_NAME
python createDatasets.py --input_csv ${DATASET_NAME}.csv

echo "[Local] Deleting Minikube cluster..."
minikube delete