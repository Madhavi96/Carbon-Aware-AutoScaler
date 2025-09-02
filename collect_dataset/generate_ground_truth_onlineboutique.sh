LOCUST_IP="192.168.1.146"
REMOTE_PATH_LOCUST="~/evaluation"
DATASET_NAME="dataset/onlineboutique"

source .venv/bin/activate

minikube start --cpus 15 --memory 50000
bash /home/ubuntu/DIE-onlineboutique-app/deploy_istio.sh

bash /home/ubuntu/DIE-onlineboutique-app/hpa_all.sh
sleep 120

# Get the NodePort from the "frontend" service (in onlineboutique app, the nodeport changes dynamically)
port=$(kubectl get svc frontend -o jsonpath='{.spec.ports[0].nodePort}')
echo "App is exposed on NodePort: $port"

start_time=$(date)

ssh -i /home/ubuntu/DIE-onlineboutique-app/train.pem ubuntu@$LOCUST_IP << EOF
      cd $REMOTE_PATH_LOCUST

      echo "[Locust] Activating Python environment..."
      source .venv/bin/activate

      bash load_test_onlineboutique.sh onlineboutique/hpa_dataset $port
EOF
end_time=$(date)

python main.py "$start_time" "$end_time" $DATASET_NAME
python createDatasets_onlineboutique.py --input_csv ${DATASET_NAME}.csv

echo "[Local] Deleting Minikube cluster..."
#minikube delete




