import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv("tourism_revenue.csv")

# Create a numerical representation for time (0 for 2560, 1 for 2561, ...)
df['Time'] = np.arange(len(df))

# Prepare features (X) and target variable (y)
X = df[['Time']]
y = df['Revenue']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict for the next year
next_year_time = df['Time'].max() + 1
predicted_revenue = model.predict([[next_year_time]])

# Print prediction
print(f"Predicted tourism revenue for year 2567: {predicted_revenue[0]:.2f} million baht")

# Plot the data and prediction
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Revenue'], color='blue', label='Actual Data')
plt.plot(df['Year'], model.predict(X), color='red', label='Regression Line')

# Prepare future year for prediction
future_year = 2567
plt.scatter(future_year, predicted_revenue, color='green', s=100, label='Future Prediction')

plt.xlabel('Year')
plt.ylabel('Revenue (Million Baht)')
plt.title('Tourism Revenue Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
