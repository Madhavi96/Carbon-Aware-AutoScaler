import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import networkx as nx
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import torch
import os
import numpy as np
import argparse
import configparser
from sklearn.model_selection import train_test_split



# Modify Path to the Metrics Data File
# path_to_metrics_data = '/content/drive/MyDrive/stgcn_data/NEW_metrics_data.csv'
# path_to_metrics_data = '/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/FEB_metrics_data.csv'
# path_to_metrics_data = '/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/original_deepscaler_metrics.csv'
# path_to_metrics_data = '/home/ubuntu/carbon-aware-autoscaler/collect_metrics/hpa_istio_full_12/metrics_data.csv'
path_to_metrics_data = '/home/ubuntu/carbon-aware-autoscaler/collect_metrics/hpa_2m/metrics_data.csv'
path_to_save_npz = '/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/hpa_2m'


# Load the CSV file
df = pd.read_csv(path_to_metrics_data)

# Extract timestamp column
timestamps = df["timestamp"].values

timeLen = len(df)

# List of microservices and feature suffixes
microservices = [
    'ts-admin-basic-info-service', 'ts-admin-order-service', 'ts-admin-route-service', 'ts-admin-travel-service',
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
    'ts-ui-dashboard', 'ts-user-mongo', 'ts-user-service', 'ts-verification-code-service', 'ts-voucher-mysql', 'ts-voucher-service'
]

# Modify this with Istio
# feature_suffixes = ["_pod", "_vCPU", "_cpu", "_mem_", "_mem","_res","_req", "_energy_idle", "_energy_dynamic", "_throttled_cpu"]
feature_suffixes = ["_pod", "_vCPU", "_cpu", "_mem_", "_mem","_res","_req"]

# 4 metrics
# feature_suffixes = ["_pod", "_cpu", "_mem", "_res","_req"]


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + points_per_hour
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence,  num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)#0到12

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]#12到24

    return hour_sample, target

def read_and_generate_dataset(graph_signal_matrix_filename,
                                                     num_of_hours, num_for_predict,
                                                     points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['arr_0']  # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []

    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        hour_sample, target = sample

        sample = []
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0)

            sample.append(hour_sample)


        tt = np.expand_dims(target, axis=0)  # (1,N,T)

        target = tt[:, :, :, 0]


        sample.append(target)

        all_samples.append(sample)
    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:])]
    train_x=training_set[0]

    train_target = training_set[1]

    all_data = {
        'train': {
            'x': train_x,
            'target': train_target,
        },
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)

    print("Target : ", all_data['train']['target'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + 'ssj' )
        print('save file:', filename)
        np.savez_compressed(filename,
                            x=all_data['train']['x'], y=all_data['train']['target'],
                            )
    return all_data

import numpy as np
import os

def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, save=False):
    data_seq = np.load(graph_signal_matrix_filename)['arr_0']

    all_samples = []
    print(data_seq.shape[0])
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_hours, idx, num_for_predict, points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        hour_sample, target = sample

        sample = []
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0)
            sample.append(hour_sample)

        tt = np.expand_dims(target, axis=0)
        target = tt[:, :, :, 0]
        sample.append(target)

        all_samples.append(sample)

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:])]
    train_x = training_set[0]
    train_target = training_set[1]

    train_size = 138
    # valid_size = 100

    # num_of_data_samples = df.shape[0]
    # valid_size = round(num_of_data_samples * 0.1)
    # train_size = num_of_data_samples - valid_size
    
    # print(train_x.shape)

    # train_x_split, valid_x_split = train_x[:train_size], train_x[train_size:]
    # train_target_split, valid_target_split = train_target[:train_size], train_target[train_size:]

    # print(f"Train Data Samples : {len(train_x_split)}")
    # print(f"Valid Data Samples : {len(valid_x_split)}")
    
    train_x_split, valid_x_split, train_target_split, valid_target_split = train_test_split(
        train_x, train_target, test_size=20, random_state=42
    )
    
    print(f"Train set size: {train_x_split.shape}, Validation set size: {valid_x_split.shape}")

    
    
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        train_filename = os.path.join(dirpath, file + '_train.npz')
        valid_filename = os.path.join(dirpath, file + '_valid.npz')

        np.savez_compressed(train_filename, x=train_x_split, y=train_target_split)
        np.savez_compressed(valid_filename, x=valid_x_split, y=valid_target_split)

        print(f'Saved train file: {train_filename}')
        print(f'Saved valid file: {valid_filename}')

    return {"train": {"x": train_x_split, "target": train_target_split},
            "valid": {"x": valid_x_split, "target": valid_target_split}}

def predict_get_sample_indices(data_sequence,  num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    hour_sample = None

    if label_start_idx > data_sequence.shape[0]:
        return hour_sample, None

    if num_of_hours > 0:
        hour_indices = predict_search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)


    return hour_sample


def predict_search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + points_per_hour
        if end_idx==47:
            print(47)
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]




# Initialize tensor
xx = torch.tensor([])

# Iterate through time steps
for i in range(timeLen):
    feature_list = []

    for service in microservices:
        service_features = []

        for suffix in feature_suffixes:
            col_name = f"{service}.*{suffix}"
            if col_name in df.columns:
                service_features.append(df[col_name].iloc[i])
            else:
                raise Exception("NO")
                service_features.append(0)  # If column is missing, fill with 0

        feature_list.append(service_features)

    # Convert to tensor
    feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
    feature_tensor = torch.unsqueeze(feature_tensor, dim=0)  # Add batch dimension
    xx = torch.cat((xx, feature_tensor), dim=0) if xx.numel() else feature_tensor

xx = torch.where(torch.isinf(xx), torch.tensor(0.0), xx)
print("xx")
print(xx.shape)
# Convert to numpy and save
xx = xx.numpy()
# np.savez("/content/drive/MyDrive/stgcn_data/final_with_istio", xx)
np.savez(path_to_save_npz, xx)
# np.savez("/content/drive/MyDrive/stgcn_data/new_metrics", xx)


# np.savez('/content/final',xx)
print("Saved dataset shape:", xx.shape)

all_data = read_and_generate_dataset(graph_signal_matrix_filename=f'{path_to_save_npz}.npz', num_of_hours=1, num_for_predict=1, points_per_hour=12, save=True)
