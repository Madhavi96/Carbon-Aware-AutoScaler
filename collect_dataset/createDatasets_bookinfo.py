import numpy as np
import pandas as pd
import torch
import os
import argparse
from sklearn.model_selection import train_test_split

# TODO
microservices = [
    "details", "productpage", "ratings", "reviews",
    ]

feature_suffixes = ["_pod", "_vCPU", "_cpu", "_mem_", "_mem", "_res", "_req"]


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
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
    return x_idx[::-1] if len(x_idx) == num_of_depend else None


def get_sample_indices(data_sequence, num_of_hours, label_start_idx, num_for_predict, points_per_hour=12):
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return None, None
    hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour)
    if not hour_indices:
        return None, None
    hour_sample = np.concatenate([data_sequence[i:j] for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    return hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename, num_of_hours, num_for_predict, points_per_hour=12, save=False):
    data_seq = np.load(graph_signal_matrix_filename)['arr_0']
    all_samples = []
    for idx in range(data_seq.shape[0]):
        hour_sample, target = get_sample_indices(data_seq, num_of_hours, idx, num_for_predict, points_per_hour)
        if hour_sample is None or target is None:
            continue
        hour_sample = np.expand_dims(hour_sample, axis=0)
        target = np.expand_dims(target, axis=0)[:, :, :, 0]
        all_samples.append([hour_sample, target])

    train_x, train_target = [np.concatenate(i, axis=0) for i in zip(*all_samples)]
    train_x_split, valid_x_split, train_target_split, valid_target_split = train_test_split(
        train_x, train_target, test_size=30, random_state=42
    )

    if save:
        base = os.path.splitext(graph_signal_matrix_filename)[0]
        np.savez_compressed(base + "_train.npz", x=train_x_split, y=train_target_split)
        np.savez_compressed(base + "_valid.npz", x=valid_x_split, y=valid_target_split)
        print(f'Saved to: {base}_train.npz and {base}_valid.npz')

    return {"train": {"x": train_x_split, "target": train_target_split},
            "valid": {"x": valid_x_split, "target": valid_target_split}}


def prepare_data(path_to_metrics_data, output_npz_path=None):
    df = pd.read_csv(path_to_metrics_data)
    time_len = len(df)
    xx = torch.tensor([])
    for i in range(time_len):
        feature_list = []
        for service in microservices:
            service_features = []
            for suffix in feature_suffixes:
                col_name = f"{service}{suffix}"
                service_features.append(df[col_name].iloc[i] if col_name in df.columns else 0)
            feature_list.append(service_features)
        feature_tensor = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(0)
        xx = torch.cat((xx, feature_tensor), dim=0) if xx.numel() else feature_tensor

    xx = torch.where(torch.isinf(xx), torch.tensor(0.0), xx)
    xx_np = xx.numpy()

    output_npz_path = output_npz_path or os.path.splitext(path_to_metrics_data)[0]
    np.savez(output_npz_path, xx_np)
    print("Saved raw tensor to:", output_npz_path + ".npz")

    return read_and_generate_dataset(graph_signal_matrix_filename=output_npz_path + ".npz",
                                     num_of_hours=1, num_for_predict=1,
                                     points_per_hour=12, save=True)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset from metrics CSV.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_npz', type=str, help='Optional path to save intermediate .npz file.')
    args = parser.parse_args()

    prepare_data(args.input_csv, args.output_npz)


if __name__ == "__main__":
    main()
