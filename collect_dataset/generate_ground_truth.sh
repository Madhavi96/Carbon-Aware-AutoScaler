LOCUST_IP="192.168.1.146"
REMOTE_PATH_LOCUST="~/evaluation"
DATASET_NAME="dataset/test"

source .venv/bin/activate
bash /home/ubuntu/DIE-train-ticket-app/hpa_all.sh

start_time=$(date --date='-90 minutes')

ssh -i /home/ubuntu/DIE-train-ticket-app/train.pem ubuntu@$LOCUST_IP << EOF
      cd $REMOTE_PATH_LOCUST

      echo "[Locust] Activating Python environment..."
      source .venv/bin/activate

      bash load_test.sh hpa_dataset
EOF
end_time=$(date)
python main.py "$start_time" "$end_time" $DATASET_NAME
python createDatasets.py --input_csv ${DATASET_NAME}.csv
