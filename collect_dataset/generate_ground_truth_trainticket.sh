LOCUST_IP="192.168.1.146"
REMOTE_PATH_LOCUST="~/evaluation"
DATASET_NAME="dataset/trainticket"

source .venv/bin/activate

minikube start --cpus 15 --memory 50000
bash /home/ubuntu/carbon-aware-autoscaler/DIE-train-ticket-app/deploy_istio.sh

bash /home/ubuntu/carbon-aware-autoscaler/DIE-train-ticket-app/hpa_all.sh
sleep 120

# Get the NodePort from the "ts-ui-dashboard" service
port=$(kubectl get svc ts-ui-dashboard -o jsonpath='{.spec.ports[0].nodePort}')
echo "App is exposed on NodePort: $port"

start_time=$(date)

ssh -i /home/ubuntu/carbon-aware-autoscaler/DIE-train-ticket-app/train.pem ubuntu@$LOCUST_IP << EOF
      cd $REMOTE_PATH_LOCUST

      echo "[Locust] Activating Python environment..."
      source .venv/bin/activate

      bash load_test_trainticket.sh trainticket/hpa_dataset $port
EOF
end_time=$(date)

python main_trainticket.py "$start_time" "$end_time" $DATASET_NAME
python createDatasets_trainticket.py --input_csv ${DATASET_NAME}.csv

echo "[Local] Deleting Minikube cluster..."
minikube delete




