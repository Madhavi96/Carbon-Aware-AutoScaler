import pandas as pd

# Load the CSV file
def remove_columns_with_patterns(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    # metrics = ["pod", "vCPU", "cpu", "mem_", "mem", "res", "req", "energy_idle", "energy_dynamic", "throttled_cpu"]

    # Define patterns to match
    patterns = ["energy_idle", "energy_dynamic", "throttled_cpu"]
    
    # Special case: allowlist exact column names to *keep*, even if they match a pattern
    # exact_allowlist = ["mem_"]
    
    # Filter columns that do not match the patterns
    filtered_columns = [
        col for col in df.columns
        if not any(col.endswith(pattern) for pattern in patterns)
    ]    
    
    # Create a new DataFrame with only the filtered columns
    df_filtered = df[filtered_columns]
    
    # Save the new DataFrame to a CSV file
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"Filtered CSV saved as: {output_csv}")

# Example usage
input_csv = "/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/metrics_data_hpa_no_preserve_2m.csv"   # Replace with your input CSV file name
output_csv = "/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/metrics_data_hpa_no_preserve_2m_7_features.csv"   # Output file name
remove_columns_with_patterns(input_csv, output_csv)
