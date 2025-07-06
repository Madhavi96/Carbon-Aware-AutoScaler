Use this script to generate the ground truth CSV and NPZ files.
Simply run generate_ground_truth.sh. This will start a minikube cluster, apply HPA to all services, start the load test, and collect the metrics in both CSV and NPZ formats.

Once completed, copy the NPZ file to the STGCN machine to proceed with training and inference.