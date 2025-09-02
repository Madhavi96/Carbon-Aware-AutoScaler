#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <output_directory_prefix> <port>"
  exit 1
fi

output_prefix="$1"
port="$2"

mkdir -p "$output_prefix"

screen -dmS port1 bash -c "ssh -i train.pem -L ${port}:192.168.49.2:${port} ubuntu@195.148.22.200"
echo "Started SSH tunnel in screen session 'port1' on port ${port}"


# prometheus port forwarding
if ! screen -list | grep -q "\.port2"; then
  screen -dmS port2 bash -c "ssh -i train.pem -L 9090:localhost:9090 ubuntu@195.148.22.200"
  echo "Started SSH tunnel in screen session 'port2'"
else
  echo "Screen session 'port2' already running."
fi

start_time=$(date)
echo "Start time: $start_time"

echo "Current time: $(date). Sleeping for 12 minutes."
sleep 720

# TODO: CHECK ALWAYS TO COMMENT WHEN RUNNING EXPERIMENTS (INFERENCE)
locust -f load_generator_onlineboutique.py --headless --host "http://localhost:${port}" --csv "${output_prefix}/data"
#sleep 720
#locust -f load_generator_onlineboutique.py --headless --host "http://localhost:${port}" --csv "${output_prefix}/data_2" --only-summary
#sleep 720
#locust -f load_generator_onlineboutique.py --headless --host "http://localhost:${port}" --csv "${output_prefix}/data_3" --only-summary
#sleep 720
#locust -f load_generator_onlineboutique.py --headless --host "http://localhost:${port}" --csv "${output_prefix}/data_4" --only-summary
#sleep 720
#locust -f load_generator_onlineboutique.py --headless --host "http://localhost:${port}" --csv "${output_prefix}/data_5" --only-summary


echo "Time after executing Locust: $(date). Sleeping for 12 minutes."
sleep 720

end_time=$(date)
echo "Script started at: $start_time"
echo "Script ended at: $end_time"

# Convert to UTC ISO 8601 format
start_time_utc=$(date -u -d "$start_time" +%Y-%m-%dT%H:%M:%SZ)
end_time_utc=$(date -u -d "$end_time" +%Y-%m-%dT%H:%M:%SZ)

# Construct Prometheus URL
url="http://localhost:9090/api/v1/query_range?query=kepler_container_joules_total%7Bnamespace=%22monitoring%22,%20mode=%22dynamic%22%7D&start=${start_time_utc}&end=${end_time_utc}&step=60s"

# Download data
curl -s "$url" -o "${output_prefix}/kepler_data.json"

echo "Prometheus data saved to ${output_prefix}/kepler_data.json"


# Get pod count metrics
# pod_url="http://localhost:9090/api/v1/query_range?query=count(container_spec_cpu_period%7Bnamespace%3D%22default%22%7D)&start=${start_time_utc}&end=${end_time_utc}&step=60s"
pod_url="http://localhost:9090/api/v1/query_range?query=count(container_spec_cpu_period%7Bnamespace%3D%22default%22%7D)%20by%20(pod)&start=${start_time_utc}&end=${end_time_utc}&step=60s"

# replicaset_url="http://localhost:9090/api/v1/query?query=count(kube_pod_owner{namespace=\"default\", owner_kind=\"ReplicaSet\"}) by (owner_name)"
curl -s "$pod_url" -o "${output_prefix}/replicaset_pod_counts.json"
echo "ReplicaSet-level pod count saved to ${output_prefix}/replicaset_pod_counts.json"