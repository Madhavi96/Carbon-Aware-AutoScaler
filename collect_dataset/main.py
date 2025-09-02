import os
import subprocess
import json
import datetime
import time
import requests
from requests.auth import HTTPBasicAuth

# profile = "hpa"
# if not os.path.exists(profile):
#     os.makedirs(profile)

def get_services():
    """Fetch the service names that start with 'ts-'."""
    command = ["kubectl", "get", "deployments"] 
    process = subprocess.run(command, capture_output=True, text=True)
    
    if process.returncode == 0:
        output = process.stdout
        # TODO: check if any logic change is required here for a specific application to omit unnecessary services.
        service_names = [
            line.split()[0] for line in output.splitlines()[1:]
            if not line.split()[0].startswith("redis") and 
            not line.split()[0].startswith("frontend")
        ]
        return service_names
    else:
        print("Error executing 'kubectl get svc':", process.stderr)
        return []
    
def delete_autoscale_deployment(scalable_services, service_name):
    if (service_name in scalable_services): 
        
        command = [
            "kubectl", "delete", "hpa", service_name
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode == 0:
            print(f"Autoscaling deleted from {service_name}")
        else:
            print(f"Error deleting autoscaling from {service_name}: {process.stderr}")
        

def autoscale_deployment(scalable_services, service_name):
    if (service_name in scalable_services): 

        """Apply autoscaling to a deployment corresponding to a service."""
        command = [
            "kubectl", "autoscale", "deployment", service_name,
            "--cpu-percent=80", "--min=1", "--max=10"
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode == 0:
            print(f"Autoscaling applied to {service_name}")
        else:
            print(f"Error autoscaling {service_name}: {process.stderr}")
        
def run_load_test():
    command = f"locust -f load_generator_train.py --headless --csv {profile}/locust"
    
    # Run the command and wait for it to complete
    subprocess.run(command, shell=True)
    print("\nLoad Testing Completed!\n")
    
    
def collect_metrics(services, start_time, end_time, saved_path):   
    regex_services = [f'{service}.*' for service in services] 
    # Convert the services array to a JSON string
    services_json = json.dumps(services)
    
    # Call file1.py using subprocess
    subprocess.run(['python', 'metrics_fetch.py', services_json, start_time.isoformat(), end_time.isoformat(), saved_path])
    
    print("\nMetric Fetch Completed!\n")

def login_to_wattime():
    login_url = 'https://api.watttime.org/login'
    rsp = requests.get(login_url, auth=HTTPBasicAuth('ediss2024', 'ediss@2024'))
    return rsp.json()['token']

def fetch_carbon_intensity(start_time, end_time):
    token = login_to_wattime()    
    url = "https://api.watttime.org/v3/historical"

    headers = {"Authorization": f"Bearer {token}"}
    start = start_time.strftime('%Y-%m-%dT%H:%M%z') + "+00:00"
    end = end_time.strftime('%Y-%m-%dT%H:%M%z') + "+00:00"

    params = {
        "region": "CAISO_NORTH",
        "start": start,
        "end": end,
        "signal_type": "co2_moer",
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()['data']
    # with open('carbon_intensity.log', 'w') as f:
    #     f.write(data)

    with open(f'{profile}/carbon_intensity.log', "w") as file:
        json.dump(data, file, indent=4)
    
    # response.raise_for_status()
    
    # # Extract values from the data
    # values = [entry['value'] for entry in data]

    # # Repeat each value 10 times
    # expanded_values = []
    # for value in values:
    #     expanded_values.extend([value] * 10)

    # # Write the expanded values to a file``
    # with open('carbon_intensity.log', 'w') as f:
    #     for value in expanded_values:
    #         f.write(f"{value}\n")

    print("\nCarbon Intensity Data Fetch Completed!\n") 

    
#def main():
    # fetch all service names
    # services = get_services()    
    # print(f'Fetched {len(services)} service names: {services}')
    # scalable_services = [s for s in services if 'mongo' not in s]
    # print("Scalable services")
    # print(scalable_services)
    # # scalable_services = ['ts-auth-service', 'ts-station-service', 'ts-order-service' ]
    

    # # delete HPA for services
    # for service in services:            
    #     delete_autoscale_deployment(scalable_services, service)
        
    # # apply HPA for services
    # for service in services:            
    #     autoscale_deployment(scalable_services, service)
    
    # print(f'\n****************** Starting Load Test ******************\n')
    
    # start_time = datetime.datetime.now() - datetime.timedelta(seconds=30)
        
    # print(f'Start Time: {start_time}. Sleeping for 80 minutes')
    # time.sleep(80 * 60)
    # print('Load test start')
    # # # run locust load generator
    # run_load_test()
    # print('Load test ended, sleeping for 80 minutes')
    # time.sleep(80 * 60)
    # end_time = datetime.datetime.now() + datetime.timedelta(seconds=30)
    
    # print(f'End Time: {end_time}')
    
    # print(f'\n****************** Starting Metrics Collection ******************\n')

    # collect_metrics(services, start_time, end_time)
    
    # print(f'\n****************** Starting Fetching Carbon Intensity Data ******************\n')

    # # collect carbon intensity data
    # fetch_carbon_intensity(start_time, end_time)      
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect metrics between two times.")
    parser.add_argument("start_time", type=str, help="Start time (e.g., 'Sat Jul 6 14:02:18 UTC 2025')")
    parser.add_argument("end_time", type=str, help="End time (e.g., 'Sat Jul 6 14:04:00 UTC 2025')")
    parser.add_argument("csv", type=str, help="csv path")
    args = parser.parse_args()

    start_time = datetime.datetime.strptime(args.start_time, "%a %b %d %H:%M:%S %Z %Y")
    end_time = datetime.datetime.strptime(args.end_time, "%a %b %d %H:%M:%S %Z %Y")
    saved_path = args.csv
    services = get_services()
    collect_metrics(services, start_time, end_time, saved_path)