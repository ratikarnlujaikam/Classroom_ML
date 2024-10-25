import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv("month.csv")

# Thai month mapping
thai_month_mapping = {
    'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03',
    'เม.ย.': '04', 'พ.ค.': '05', 'มิ.ย.': '06',
    'ก.ค.': '07', 'ส.ค.': '08', 'ก.ย.': '09',
    'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12'
}

# Extract month and convert to standard format
df['Month'] = df['Month'].str.extract(r'\((.*?)\)')[0]
df['Month'] = df['Month'].replace(thai_month_mapping)

# Create date column
df['Date'] = pd.to_datetime('2024-' + df['Month'] + '-01')

# Create a numerical representation for time
df['Time'] = range(len(df))

# Convert Revenue to string and remove commas
df['Revenue'] = df['Revenue'].astype(str).str.replace(',', '')
df['Revenue'] = df['Revenue'].astype(float)

# Prepare features (X) and target variable (y)
X = df[['Time']]
y = df['Revenue']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict for the next months until the current month
predictions = []
num_months_to_predict = 12  # Number of months to predict

for i in range(1, num_months_to_predict + 1):
    next_month_time = df['Time'].max() + i
    prediction = model.predict([[next_month_time]])
    predictions.append((next_month_time, prediction[0]))

# Print predictions
for time, revenue in predictions:
    print(f"Predicted tourism revenue for month {time}: {revenue:.2f} million baht")

# Plot the data and prediction
plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], df['Revenue'], color='blue', label='Actual Data')
plt.plot(df['Date'], model.predict(X), color='red', label='Regression Line')

# Prepare future dates for predictions
future_dates = [df['Date'].max() + pd.DateOffset(months=i) for i in range(1, num_months_to_predict + 1)]
future_revenues = [revenue for _, revenue in predictions]

plt.scatter(future_dates, future_revenues, color='green', s=100, label='Future Predictions')

plt.xlabel('Date')
plt.ylabel('Revenue (Million Baht)')
plt.title('Tourism Revenue Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
