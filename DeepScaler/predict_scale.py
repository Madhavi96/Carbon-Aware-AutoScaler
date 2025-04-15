import torch
import numpy as np
import os
import sys
import yaml
import argparse
import time
import datetime
import math
from torch.utils.data import DataLoader
from utils import scaler
from models import AdapGL
from dataset import TPDataset2
from metrics_fetch import save_all_fetch_data, fetch_cpu_usage_for_hpa, fetch_pods
from prepareData import predict_read_and_generate_dataset
from k8sop import K8sOp

from codecarbon import EmissionsTracker  # Codecarbon

def load_config(data_path):
    """Load configuration from a YAML file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fetch_metrics_data(services, metrics, start_time_str, current_time_str):
    """Fetch and process metric data for all services."""
    times_original = [(str(start_time_str), str(current_time_str))]
    print(times_original)
    save_all_fetch_data(times_original, 1, root_dir='./dataForPredict/', interval=60, services=services)

    metric_data = {metric: {} for metric in metrics}

    for service in services:
        for metric in metrics:
            file_path = f'./dataForPredict/1_{service}_{metric}.log'
            data = np.genfromtxt(file_path, dtype=np.double)

            if data.ndim == 0:  # Handle single-value cases
                data = np.array([data])

            metric_data[metric][service] = data[:, np.newaxis]

    return metric_data


def prepare_input_tensor(metric_data, services, metrics, prev_data):
    """Prepare input tensor from the fetched metric data."""
    time_len = len(next(iter(metric_data['cpu'].values())))  # Get the length from any service metric
    xx = prev_data[1:, :, :] if prev_data is not None else torch.tensor([])

    for i in range(time_len):
        stacked_metrics = []
        for service in services:
            values = [metric_data[metric][service][i] for metric in metrics]
            stacked_metrics.append(np.vstack(values))

        tensor_data = torch.tensor(np.hstack(stacked_metrics), dtype=torch.float32)
        tensor_data = torch.unsqueeze(tensor_data, dim=0)
        xx = torch.cat((xx, tensor_data), dim=0)

    return xx


def predict_scaling(xx, model_config_path, train_config_path, model_name):
    """Run the prediction model to determine pod scaling."""
    model_config = load_config(model_config_path)
    train_config = load_config(train_config_path)

    torch.manual_seed(train_config['seed'])

    Scaler = getattr(sys.modules['utils.scaler'], train_config['scaler'])
    data_scaler = Scaler(axis=(0, 1, 2))

    data_config = model_config['dataset']
    device = torch.device(data_config['device'])
    data_name = 'predict_scale_r1ssj.npz'

    dataset = TPDataset2(os.path.join(data_config['data_dir'], data_name))
    dataset.data['x'] = dataset.data['x'].transpose(0, 1, 3, 2)
    data_scaler.fit(dataset.data['x'])
    dataset.fit(data_scaler)
    data_loader = DataLoader(dataset, batch_size=data_config['batch_size'])

    # Load model
    Model = getattr(AdapGL, model_name, None)
    if Model is None:
        raise ValueError(f'Model {model_name} is not valid!')

    net_pred = Model(**model_config[model_name], **data_config).to(device)
    net_pred.load_state_dict(torch.load(args.model_save_path))  # Load trained model
    best_adj_path = os.path.join(os.path.dirname(args.model_save_path), "best_adj_mx.npy")
    adj = np.load(best_adj_path)
    torch_adj = torch.from_numpy(adj)
    data_tensor = torch.from_numpy(data_loader.dataset.data['x'])
    # print(f'Data Tensor: {data_tensor}')
    # print(f'Adj Data: {torch_adj}')
    
    # print(f'Model: {net_pred}')


    pred = net_pred(data_tensor, torch_adj).detach()
    # print(pred)
    pred = data_scaler.inverse_transform(data=pred, axis=0)
    print(pred.shape)
    print(pred)
    return pred

def determine_scaling_for_hpa(services_to_scale):
    target_cpu_utilization = 60
    pods_num_to_scale = {}
    
    for svc_name in services_to_scale:
        print(f"Service : {svc_name}")

        current_cpu = fetch_cpu_usage_for_hpa(svc_name)
        current_pod_count = fetch_pods(svc_name)
      
        print(f"Current CPU: {current_cpu}")
        print(f"Current PODS: {current_pod_count}")
        # Calculate target pod count using HPA formula
        target_pod_count = math.ceil((current_cpu * current_pod_count) / target_cpu_utilization)
        print(f"Target PODS: {target_pod_count}")

        if target_pod_count > 10:
            continue
    
        pods_num_to_scale[svc_name] = target_pod_count if target_pod_count > 0 else 1 
        
    return pods_num_to_scale
    
def determine_services_to_scale(pred, services, threshold):
    services_to_scale = []
    
    for idx, service in enumerate(services):
        service_value = pred[-1, 0, idx]
        service_value = 1 if math.isnan(service_value) else service_value
        
        if service_value >= threshold:
            services_to_scale.append(service)
        
    return services_to_scale
    

def determine_scaling(pred, services, restriction=0.4):
    """Determine the number of pods to scale for each service."""
    pods_num_to_scale = {}
    # pred = pred[0]
    for idx, service in enumerate(services):
        service_value = pred[-1, 0, idx]
        service_value = 1 if math.isnan(service_value) else service_value

        fractional_part = service_value - math.floor(service_value)
        pods_num_to_scale[service] = (
            math.floor(service_value) if fractional_part < restriction else math.ceil(service_value)
        )

        if pods_num_to_scale[service] <= 0:
            pods_num_to_scale[service] = 1  # Ensure at least 1 pod

    return pods_num_to_scale


def main(args, tracker):
    """Main function for fetching data, predicting scaling, and adjusting pods."""
    k8s_op = K8sOp()
    services = ['ts-admin-basic-info-service', 'ts-admin-order-service', 'ts-admin-route-service', 'ts-admin-travel-service',
    'ts-admin-user-service', 'ts-assurance-mongo', 'ts-assurance-service', 'ts-auth-mongo', 'ts-auth-service',
    'ts-basic-service', 'ts-cancel-service', 'ts-config-mongo', 'ts-config-service', 'ts-consign-mongo',
    'ts-consign-price-mongo', 'ts-consign-price-service', 'ts-consign-service', 'ts-contacts-mongo',
    'ts-contacts-service', 'ts-execute-service', 'ts-food-map-mongo', 'ts-food-map-service', 'ts-food-mongo',
    'ts-food-service', 'ts-inside-payment-mongo', 'ts-inside-payment-service', 'ts-news-service',
    'ts-notification-service', 'ts-order-mongo', 'ts-order-other-mongo', 'ts-order-other-service', 'ts-order-service',
    'ts-payment-mongo', 'ts-payment-service', 'ts-preserve-other-service', 'ts-preserve-service', 'ts-price-mongo',
    'ts-price-service', 'ts-rebook-service', 'ts-route-mongo', 'ts-route-plan-service', 'ts-route-service',
    'ts-seat-service', 'ts-security-mongo', 'ts-security-service', 'ts-station-mongo', 'ts-station-service',
    'ts-ticket-office-mongo', 'ts-ticket-office-service', 'ts-ticketinfo-service', 'ts-train-mongo', 'ts-train-service',
    'ts-travel-mongo', 'ts-travel-plan-service', 'ts-travel-service', 'ts-travel2-mongo', 'ts-travel2-service',
    'ts-ui-dashboard', 'ts-user-mongo', 'ts-user-service', 'ts-verification-code-service', 'ts-voucher-mysql', 'ts-voucher-service']
    metrics = ["pod", "vCPU", "cpu", "mem_", "mem","res","req"]
    # metrics = ["pod", "vCPU", "cpu", "mem_", "mem", "energy_idle", "energy_dynamic", "throttled_cpu"]
    # metrics = ["pod", "cpu", "mem", "res", "req"]

    c_temp = 0
    prev_data = None

    scale_up_time_tracker = {}  # Tracks the last time a service was scaled up
    cooldown_period = datetime.timedelta(minutes=2)
    max_pods = 8

    
    try:
        while True:
            tracker.start()
            start_time = time.time()

            current_time = datetime.datetime.now()
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            start_time_str = (
                (current_time - datetime.timedelta(hours=0, minutes=13)).strftime('%Y-%m-%d %H:%M:%S')
                if c_temp == 0 else
                (current_time - datetime.timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
            )

            # Fetch and process metric data
            
            metric_data = fetch_metrics_data(services, metrics, start_time_str, current_time_str)

            # Prepare input tensor
            xx = prepare_input_tensor(metric_data, services, metrics, prev_data)
            np.savez("./data/predict_scale", xx)
            print("xx")
            print(xx.shape)

            # Generate dataset and predict scaling
            predict_read_and_generate_dataset('./data/predict_scale.npz', num_of_hours=1, num_for_predict=1, points_per_hour=12, save=True)
            print("HUE")
            pred = predict_scaling(xx, args.model_config_path, args.train_config_path, args.model_name)

            # Determine and apply scaling
            # pods_num_to_scale = determine_scaling(pred, services)
            # services_to_scale = determine_services_to_scale(pred, services, threshold=1.8)
            # pods_num_to_scale = determine_scaling_for_hpa(services_to_scale)
            if args.model_name=='AdapGLT':
                pred = pred[0]
            pods_num_to_scale = determine_scaling(pred, services)

            
            #for svc, num_pods in pods_num_to_scale.items():
            #    k8s_op.scale_deployment_by_replicas(svc, "default", num_pods)
            for svc, desired_pods in pods_num_to_scale.items():
                desired_pods = min(desired_pods, max_pods)  # Rule 1: Cap pod count at 8
                current_pods = fetch_pods(svc)

                # Only attempt scaling if current pods do not equal desired pods
                if current_pods == desired_pods:
                    # print(f"[No Change] {svc} remains at {current_pods} pods.")
                    continue

                now = datetime.datetime.now()
                last_scaled_up_time = scale_up_time_tracker.get(svc)

                if desired_pods > current_pods:
                    # Scaling up: record the time and apply new replica count
                    scale_up_time_tracker[svc] = now
                    k8s_op.scale_deployment_by_replicas(svc, "default", desired_pods)
                    print(f"[Scale Up] {svc} scaled from {current_pods} to {desired_pods} pods.")
                elif desired_pods < current_pods:
                    # Scaling down: enforce cooldown period of 2 minutes since last scale-up
                    if not last_scaled_up_time or (now - last_scaled_up_time) >= cooldown_period:
                        k8s_op.scale_deployment_by_replicas(svc, "default", desired_pods)
                        print(f"[Scale Down] {svc} scaled from {current_pods} to {desired_pods} pods.")
                    else:
                        print(f"[Cooldown] Skipping scale down for {svc} due to cooldown (last scale-up at {last_scaled_up_time}).")
            tracker.stop()

            print("After scaling:", pods_num_to_scale)
            # Manage sleep time to maintain loop timing
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 55 - elapsed_time)
            time.sleep(sleep_time)

            c_temp += 1
            prev_data = xx
            
    except KeyboardInterrupt:
        print("Training interrupted manually. Stopping tracker...")
    # except Exception as e:
    #    print(f"An error occurred: {e}")
    finally:
        # Stop emissions tracking when the program exits
        tracker.stop()
        # print(f"Stopped the carbon emission tracker")
        print(f"Prediciton process ended!")

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='./config/train_dataset_speed.yaml', help='Config path of models')
    parser.add_argument('--train_config_path', type=str, default='./config/train_config.yaml', help='Config path of Trainer')
    parser.add_argument('--model_name', type=str, default='AdapGLA', help='Model name to train')
    parser.add_argument('--num_epoch', type=int, default=5, help='Training times per epoch')
    parser.add_argument('--num_iter', type=int, default=5, help='Maximum value for iteration')
    parser.add_argument('--model_save_path', type=str, default='/home/ubuntu/carbon-aware-autoscaler/DeepScaler/model/AdapGLA_1/AdapGLA_1.pkl', help='Model save path')
    parser.add_argument('--max_graph_num', type=int, default=3, help='Volume of adjacency matrix set')

    args = parser.parse_args()
    # base_name = os.path.splitext(os.path.basename(args.model_save_path))[0]
    model_save_dir = os.path.dirname(args.model_save_path)

    args.model_save_dir = model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)
    if os.path.exists(os.path.join(model_save_dir,'emissions_inference.csv')):
        os.remove(os.path.join(model_save_dir,'emissions_inference.csv'))
    tracker = EmissionsTracker(
        output_file=os.path.join(model_save_dir,'emissions_inference.csv')
    ) #
    
    main(args, tracker)
