import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. THE DATA (Realistic historical decadal averages)
# ---------------------------------------------------------
# Years: 1960, 1970, 1980, 1990, 2000, 2010, 2020
co2_ppm = np.array([[316], [325], [338], [354], [369], [389], [414]], dtype=np.float32)
temp_anomaly = np.array([[0.02], [0.05], [0.27], [0.44], [0.61], [0.71], [1.02]], dtype=np.float32)

# Scale CO2 by dividing by 100 for stable neural network training
co2_scaled = co2_ppm / 100.0

# Convert numpy arrays to PyTorch Tensors (the format PyTorch requires)
X = torch.from_numpy(co2_scaled)
y = torch.from_numpy(temp_anomaly)

# ---------------------------------------------------------
# 2. THE MODEL
# ---------------------------------------------------------
# A simple linear model with 1 input (CO2) and 1 output (Temperature)
class ClimatePredictor(nn.Module):
    def __init__(self):
        super(ClimatePredictor, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

model = ClimatePredictor()

# ---------------------------------------------------------
# 3. TRAINING SETUP
# ---------------------------------------------------------
# Mean Squared Error (MSE) measures the distance between our guess and the real data
criterion = nn.MSELoss()
# Stochastic Gradient Descent (SGD) tweaks the weights to reduce the error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---------------------------------------------------------
# 4. THE TRAINING LOOP
# ---------------------------------------------------------
epochs = 2000
print("Training the model...")

for epoch in range(epochs):
    # Make a prediction
    predictions = model(X)
    
    # Calculate how wrong the prediction is
    loss = criterion(predictions, y)
    
    # Erase the old adjustments
    optimizer.zero_grad()
    
    # Calculate the new adjustments (Backward pass)
    loss.backward()
    
    # Apply the adjustments to the model
    optimizer.step()
    
    # Print an update every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Error (Loss): {loss.item():.4f}")

print("Training Complete!")

# ---------------------------------------------------------
# 5. VISUALIZATION & FUTURE PREDICTION
# ---------------------------------------------------------
# Predict a future scenario: 450 ppm (expected around year 2050 if trends continue)
future_co2_scaled = torch.tensor([[4.50]], dtype=torch.float32)
future_prediction = model(future_co2_scaled).item()

# Get the model's final predictions for our historical data to draw the line
with torch.no_grad(): # Turn off training mode for purely graphing
    predicted_line = model(X).numpy()

print(f"\nModel Prediction: At 450 ppm CO2, the temperature anomaly will be +{future_prediction:.2f}°C")

# Plotting the data
plt.figure(figsize=(8, 5))
plt.scatter(co2_ppm, temp_anomaly, color='red', label='Historical Real Data')
plt.plot(co2_ppm, predicted_line, color='blue', label='AI Learned Trend Line')
plt.scatter([450], [future_prediction], color='purple', marker='*', s=150, label='Future Prediction (450 ppm)')

plt.title('Global Temperature Anomaly vs CO2 Levels')
plt.xlabel('CO2 Concentration (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Display the chart
plt.show()