import numpy as np
import json

# Mapping of microservice names to indexes
microservice_index = {
    "ts-admin-basic-info-service": 0, "ts-admin-order-service": 1, "ts-admin-route-service": 2, "ts-admin-travel-service": 3,
    "ts-admin-user-service": 4, "ts-assurance-mongo": 5, "ts-assurance-service": 6, "ts-auth-mongo": 7,
    "ts-auth-service": 8, "ts-basic-service": 9, "ts-cancel-service": 10, "ts-config-mongo": 11,
    "ts-config-service": 12, "ts-consign-mongo": 13, "ts-consign-price-mongo": 14, "ts-consign-price-service": 15,
    "ts-consign-service": 16, "ts-contacts-mongo": 17, "ts-contacts-service": 18, "ts-execute-service": 19,
    "ts-food-map-mongo": 20, "ts-food-map-service": 21, "ts-food-mongo": 22, "ts-food-service": 23,
    "ts-inside-payment-mongo": 24, "ts-inside-payment-service": 25, "ts-news-service": 26, "ts-notification-service": 27,
    "ts-order-mongo": 28, "ts-order-other-mongo": 29, "ts-order-other-service": 30, "ts-order-service": 31,
    "ts-payment-mongo": 32, "ts-payment-service": 33, "ts-preserve-other-service": 34, "ts-preserve-service": 35,
    "ts-price-mongo": 36, "ts-price-service": 37, "ts-rebook-service": 38, "ts-route-mongo": 39,
    "ts-route-plan-service": 40, "ts-route-service": 41, "ts-seat-service": 42, "ts-security-mongo": 43,
    "ts-security-service": 44, "ts-station-mongo": 45, "ts-station-service": 46, "ts-ticket-office-mongo": 47,
    "ts-ticket-office-service": 48, "ts-ticketinfo-service": 49, "ts-train-mongo": 50, "ts-train-service": 51,
    "ts-travel-mongo": 52, "ts-travel-plan-service": 53, "ts-travel-service": 54, "ts-travel2-mongo": 55,
    "ts-travel2-service": 56, "ts-ui-dashboard": 57, "ts-user-mongo": 58, "ts-user-service": 59,
    "ts-verification-code-service": 60, "ts-voucher-mysql": 61, "ts-voucher-service": 62
}

# Load JSON file
data_file = "/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/microservice_dependencies.json"
with open(data_file, "r") as file:
    connections = json.load(file)

# Number of microservices
num_services = len(microservice_index)

# Initialize adjacency matrix with zeros (as float type)
# adj_matrix = np.zeros((num_services, num_services), dtype=float)
adj_matrix = np.full((num_services, num_services), 1e-5, dtype=float)

# Populate adjacency matrix with 1.0 where there is a dependency
for connection in connections:
    parent = connection["parent"]
    child = connection["child"]
    if parent in microservice_index and child in microservice_index:
        parent_idx = microservice_index[parent]
        child_idx = microservice_index[child]
        adj_matrix[parent_idx, child_idx] = 1.0

# Save adjacency matrix as .npy file
save_path = "binary_adjacency_matrix_non_zero.npy"
np.save(save_path, adj_matrix)

print(f"Binary adjacency matrix saved to {save_path}")

# import numpy as np
# import json

# # Mapping of microservice names to indexes
# microservice_index = {
#     "ts-admin-basic-info-service": 0, "ts-admin-order-service": 1, "ts-admin-route-service": 2, "ts-admin-travel-service": 3,
#     "ts-admin-user-service": 4, "ts-assurance-mongo": 5, "ts-assurance-service": 6, "ts-auth-mongo": 7,
#     "ts-auth-service": 8, "ts-basic-service": 9, "ts-cancel-service": 10, "ts-config-mongo": 11,
#     "ts-config-service": 12, "ts-consign-mongo": 13, "ts-consign-price-mongo": 14, "ts-consign-price-service": 15,
#     "ts-consign-service": 16, "ts-contacts-mongo": 17, "ts-contacts-service": 18, "ts-execute-service": 19,
#     "ts-food-map-mongo": 20, "ts-food-map-service": 21, "ts-food-mongo": 22, "ts-food-service": 23,
#     "ts-inside-payment-mongo": 24, "ts-inside-payment-service": 25, "ts-news-service": 26, "ts-notification-service": 27,
#     "ts-order-mongo": 28, "ts-order-other-mongo": 29, "ts-order-other-service": 30, "ts-order-service": 31,
#     "ts-payment-mongo": 32, "ts-payment-service": 33, "ts-preserve-other-service": 34, "ts-preserve-service": 35,
#     "ts-price-mongo": 36, "ts-price-service": 37, "ts-rebook-service": 38, "ts-route-mongo": 39,
#     "ts-route-plan-service": 40, "ts-route-service": 41, "ts-seat-service": 42, "ts-security-mongo": 43,
#     "ts-security-service": 44, "ts-station-mongo": 45, "ts-station-service": 46, "ts-ticket-office-mongo": 47,
#     "ts-ticket-office-service": 48, "ts-ticketinfo-service": 49, "ts-train-mongo": 50, "ts-train-service": 51,
#     "ts-travel-mongo": 52, "ts-travel-plan-service": 53, "ts-travel-service": 54, "ts-travel2-mongo": 55,
#     "ts-travel2-service": 56, "ts-ui-dashboard": 57, "ts-user-mongo": 58, "ts-user-service": 59,
#     "ts-verification-code-service": 60, "ts-voucher-mysql": 61, "ts-voucher-service": 62
# }

# # Load JSON file
# data_file = "/home/ubuntu/carbon-aware-autoscaler/DeepScaler/data/microservice_dependencies.json"
# with open(data_file, "r") as file:
#     connections = json.load(file)

# # Number of microservices
# num_services = len(microservice_index)

# # Initialize adjacency matrix with zeros
# adj_matrix = np.zeros((num_services, num_services))

# # Extract all callCount values
# call_counts = []

# for connection in connections:
#     parent = connection["parent"]
#     child = connection["child"]
#     call_count = int(connection["callCount"])

#     if parent in microservice_index and child in microservice_index:
#         parent_idx = microservice_index[parent]
#         child_idx = microservice_index[child]
#         adj_matrix[parent_idx, child_idx] = call_count  # Store raw call count
#         call_counts.append(call_count)

# # Normalize call counts to range [0,1]
# if call_counts:  # Ensure there's at least one connection
#     min_call = min(call_counts)
#     max_call = max(call_counts)

#     if max_call != min_call:  # Avoid division by zero
#         adj_matrix = (adj_matrix - min_call) / (max_call - min_call)
#     else:
#         adj_matrix[adj_matrix > 0] = 1  # Set all non-zero values to 1 if they are equal

# # Replace zeros with a very small random value
# random_values = np.random.uniform(0, 0.00001, adj_matrix.shape)
# adj_matrix[adj_matrix == 0] = random_values[adj_matrix == 0]
# # Save adjacency matrix as .npy file
# save_path = "new_adjacency_matrix.npy"
# np.save(save_path, adj_matrix)
# # print(adj_matrix)
# print(f"Adjacency matrix saved to {save_path}")

# # Define a threshold for high correlation (e.g., top 10% values)
# threshold = np.percentile(adj_matrix[adj_matrix > 0], 95)

# # Find indexes where values exceed the threshold
# high_corr_indexes = np.argwhere(adj_matrix > threshold)

# # Reverse mapping from index to microservice name
# index_to_microservice = {v: k for k, v in microservice_index.items()}

# # Print high-correlation service pairs
# print("Highly correlated microservices:")
# for parent_idx, child_idx in high_corr_indexes:
#     print(f"{index_to_microservice[parent_idx]} -> {index_to_microservice[child_idx]}")
