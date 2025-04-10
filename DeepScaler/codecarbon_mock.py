import numpy as np
import time
from sklearn.linear_model import LinearRegression
from codecarbon import EmissionsTracker

# Start tracking emissions
tracker = EmissionsTracker()
tracker.start()

# Generate random training data
X = np.random.rand(1000, 10)
y = np.random.rand(1000)

# Train a simple model
model = LinearRegression()
start_time = time.time()
model.fit(X, y)
end_time = time.time()

# Stop tracking emissions
tracker.stop()

# Print training time
print(f"Training time: {end_time - start_time:.4f} seconds")
