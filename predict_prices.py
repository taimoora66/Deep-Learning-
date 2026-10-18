import numpy as np
from sklearn.linear_model import LinearRegression

#1.Prepare the training data 
#X = House sizes in square feet(reshaped to a 2D arrray for the model)
house_sizes =np.array([600,900,1200, 1500, 1800, 2100]).reshape(-1,1)
#Y = House prices in thousands of dollars($150k, $210k, etc.)
house_prices= np.array([150,210,270,330,390,450])

print("---Training the machine Learning model---")

#2. Initialize and train the Linear REgression model
#This algorithm finds the mathematical formula: Price = (slope*Size) + Intercept
model = LinearRegression()
model.fit(house_sizes,house_prices)

print("Model trained successfully!\n")

#3. Use the model to predict the prices of the complete new house
new_house_size =1400
predicted_price =model.predict([[new_house_size]])

#4. Output the results
print(f"Prediction for a {new_house_size} sq ft house:")
print(f"Estimated Price: ${predicted_price[0]:.2f}k (${predicted_price[0]*1000:,.2f})")
