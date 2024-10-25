import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Read data from the CSV file
df = pd.read_csv("revenue_data.csv", encoding='utf-8')

# Write DataFrame to a file for inspection
df.to_csv("original_dataframe_output.csv", index=False, encoding='utf-8')
print("Original DataFrame has been written to original_dataframe_output.csv")

# Convert 'Year' and 'Revenue (Million Baht)' to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Revenue (Million Baht)'] = pd.to_numeric(df['Revenue (Million Baht)'], errors='coerce')

# Continue with your existing logic...


# Check for NaN values in 'Year' and 'Revenue'
print("NaN values in Year:\n", df['Year'].isna().sum())
print("NaN values in Revenue:\n", df['Revenue (Million Baht)'].isna().sum())

# Fill missing revenue values with 0 or another appropriate method
df['Revenue (Million Baht)'] = df['Revenue (Million Baht)'].fillna(0)

# Check rows after filling NaN
print(f"Rows after filling NaN: {len(df)}")

# Fill missing year values (if any) with a default or remove those rows
df['Year'] = df['Year'].fillna(0).astype(int)  # Set a default value or handle accordingly
# Consider removing rows with invalid years
# df = df[df['Year'] > 0]  # Uncomment this line if you want to enforce a positive year

# Check the number of rows after filtering
print(f"Rows after filtering: {len(df)}")

# Convert month to numerical codes
df['Month'] = pd.Categorical(df['Month']).codes

# Prepare the data for training
X = df[['Year', 'Month']]  # Features
y = df['Revenue (Million Baht)']  # Target variable

# Check for any remaining non-numeric data
print("X data types:\n", X.dtypes)
print("y data type:\n", y.dtype)

# Check the shapes of X and y
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split the data into training and testing sets
if len(df) == 0:
    print("DataFrame is empty. Exiting.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict revenue for the next 5 years (2027 to 2031)
    future_years = []
    future_months = []

    for year in range(2027, 2032):  # From 2027 to 2031
        for month in range(12):  # For each month
            future_years.append(year)
            future_months.append(month)

    future_data = pd.DataFrame({'Year': future_years, 'Month': future_months})

    # Make predictions
    predicted_revenue = model.predict(future_data)

    # Create a DataFrame for the predictions
    future_data['Predicted_Revenue'] = predicted_revenue

    # Display the predictions
    print(future_data)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['Revenue (Million Baht)'], label='Historical Revenue', marker='o')
    plt.plot(future_data['Year'], future_data['Predicted_Revenue'], label='Predicted Revenue', marker='o', color='orange')
    plt.xlabel('Year')
    plt.ylabel('Revenue (Million Baht)')
    plt.title('Revenue Forecast for the Next 5 Years')
    plt.legend()
    plt.tight_layout()
    plt.show()
