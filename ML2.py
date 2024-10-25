from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import io
import csv

# Set console to support UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Create a WebDriver instance (using Chrome here)
driver = webdriver.Chrome()

# Open the desired URL
url = 'https://xn--42cah7d0cxcvbbb9x.com/%E0%B8%A3%E0%B8%B2%E0%B8%84%E0%B8%B2%E0%B8%97%E0%B8%AD%E0%B8%87%E0%B8%A2%E0%B9%89%E0%B8%AD%E0%B8%99%E0%B8%AB%E0%B8%A5%E0%B8%B1%E0%B8%87/'
driver.get(url)

# Wait for the table to load
try:
    table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="post-74"]/div[6]/div[2]/table'))
    )
    
    rows = table.find_elements(By.TAG_NAME, 'tr')

    # Create a list to store the data
    formatted_data = []

    # Table headers with unique names
    headers = ["วันที่/เวลา", "ครั้งที่", "รับซื้อ(บาท)", "ขายออก(บาท) 1", "ฐานภาษี(บาท)", "ขายออก(บาท) 2", "Gold spot", "เงินบาท", "ขึ้น/ลง"]
    formatted_data.append(headers)  # Add headers to the list

    # Display data in each row
    for i, row in enumerate(rows):
        # Skip the last row and the row containing "ทองแท่ง" and "ทองรูปพรรณ"
        if i == len(rows) - 1 or "ทองแท่ง" in row.text or "ทองรูปพรรณ" in row.text:
            continue
        
        columns = row.find_elements(By.TAG_NAME, 'td')
        if columns:  # Check if the row has data
            data = [column.text for column in columns]
            # Check if at least one value contains a comma
            if any(',' in value for value in data):
                # Convert numeric strings to floats, removing commas
                converted_data = []
                for value in data:
                    try:
                        # Attempt to convert to float
                        converted_value = float(value.replace(',', ''))
                    except ValueError:
                        # If conversion fails, keep original value (could be a date or non-numeric)
                        converted_value = value
                    converted_data.append(converted_value)
                formatted_data.append(converted_data)  # Add the converted data to the list

    # Save the data to a CSV file
    with open('gold_prices.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(formatted_data)  # Write all data to the CSV file

    print("ข้อมูลได้ถูกบันทึกลงใน gold_prices.csv")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")

finally:
    # Close the WebDriver
    driver.quit()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Read the CSV file while skipping rows with errors
data = pd.read_csv('gold_prices.csv', header=0, on_bad_lines='skip')

# Ensure the 'รับซื้อ(บาท)' column is treated as string before replacement
data['รับซื้อ(บาท)'] = data['รับซื้อ(บาท)'].astype(str).str.replace(',', '', regex=False)

# Convert to numeric, coerce errors to NaN
data['ราคาซื้อ'] = pd.to_numeric(data['รับซื้อ(บาท)'], errors='coerce')

# Create feature for day of the year
data['วัน'] = pd.to_datetime(data['ครั้งที่'], errors='coerce').dt.dayofyear

# Create target variable: 1 if price goes up, 0 if it goes down
data['ขึ้น'] = (data['ราคาซื้อ'].shift(-1) > data['ราคาซื้อ']).astype(int)

# Drop rows with NaN values
data.dropna(inplace=True)

# Define features and target variable
X = data[['วัน', 'ราคาซื้อ']]
y = data['ขึ้น']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"ความแม่นยำของโมเดล: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))

# Predicting the next day's movement
next_day_feature = np.array([[X['วัน'].max() + 1, data['ราคาซื้อ'].iloc[-1]]])  # วันถัดไปและราคาซื้อปัจจุบัน
next_day_prediction = model.predict(next_day_feature)

# Prepare results for display
result = {
    'วันถัดไป': [X['วัน'].max() + 1],
    'ราคาซื้อปัจจุบัน (บาท)': [data['ราคาซื้อ'].iloc[-1]],
    'คาดการณ์การเคลื่อนไหว': ['ขึ้น' if next_day_prediction[0] == 1 else 'ลง']
}

# Create a DataFrame for the results
result_df = pd.DataFrame(result)

# Display the results
print(result_df)






# ส่วนสำคัญ

import requests

# ใช้ API ของ ExchangeRate-API หรือบริการอื่น ๆ
response = requests.get('https://api.exchangerate-api.com/v4/latest/THB')
data = response.json()

# ตรวจสอบอัตราแลกเปลี่ยน
usd_to_thb = data['rates']['USD']
thb_to_usd = 1 / usd_to_thb

print(f"1 THB = {thb_to_usd:.6f} USD")

print("**********************************************************ใช้ LinearRegression******************************************************************")
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the CSV file while skipping rows with errors
data = pd.read_csv('gold_prices.csv', header=0, on_bad_lines='skip')

# Ensure the 'รับซื้อ(บาท)' column is treated as string before replacement
data['รับซื้อ(บาท)'] = data['รับซื้อ(บาท)'].astype(str).str.replace(',', '', regex=False)

# Convert to numeric, coerce errors to NaN
y = pd.to_numeric(data['รับซื้อ(บาท)'], errors='coerce')

# Create feature for day of the year
data['วัน'] = pd.to_datetime(data['ครั้งที่'], errors='coerce').dt.dayofyear

# Check if there are enough samples after dropping NaN
if len(y.dropna()) == 0 or len(data[data['วัน'].notna()]) == 0:
    print("ไม่มีข้อมูลเพียงพอสำหรับการฝึกโมเดล")
else:
    # Align X with y
    X = data.loc[y.index, ['วัน']]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y.dropna(), test_size=0.2, random_state=0)
    # 80% = X_train 20% =X_test
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the next day's price
    next_day = np.array([[X['วัน'].max() + 1]])  # วันถัดไป
    predicted_price_baht = model.predict(next_day)

    # Predict prices for the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"คำนวณ Linear Regression RMSE: {rmse:.2f}")

    predicted_price_usd = predicted_price_baht[0] / thb_to_usd
    first_row = data.iloc[0]  # แถวแรก
    date_time = first_row['วันที่/เวลา']  # วันที่และเวลา
    buy_price = first_row['รับซื้อ(บาท)']  # ราคา

    print(f"วันที่/เวลา: {date_time}, รับซื้อ(บาท): {buy_price}")

    # แสดงราคา
    print(f"ราคาทองคำที่คาดการณ์ในวันถัดไป: {predicted_price_baht[0]:,.2f} บาท")

    def thb_to_usd(predicted_price_baht):
        return predicted_price_baht / usd_to_thb

    amount_usd = thb_to_usd(predicted_price_baht)
    print(f"ราคาทองคำที่คาดการณ์ในวันถัดไป: {predicted_price_usd:.2f} USD")
    
    data_diff = float(predicted_price_baht[0]) - float(buy_price)
    print(f"ขึ้น/ลง {data_diff:.2f} บาท")

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({'วัน': X['วัน'], 'รับซื้อ(บาท)': y})
    
    # Create a new DataFrame for the predicted price
    predicted_data = pd.DataFrame({'วัน': [X['วัน'].max() + 1], 'รับซื้อ(บาท)': [predicted_price_baht[0]]})

    # Concatenate the original data with the predicted data
    plot_data = pd.concat([plot_data, predicted_data], ignore_index=True)


print("*******************************************************test RandomForestRegressor******************************************************************")

# test 2 Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import requests

# ใช้ API ของ ExchangeRate-API หรือบริการอื่น ๆ
response = requests.get('https://api.exchangerate-api.com/v4/latest/THB')
data = response.json()

# ตรวจสอบอัตราแลกเปลี่ยน
usd_to_thb = data['rates']['USD']
thb_to_usd = 1 / usd_to_thb

print(f"1 THB = {thb_to_usd:.6f} USD")
# อ่านข้อมูล
data = pd.read_csv('gold_prices.csv', header=0, on_bad_lines='skip')

# แปลงปีจากพุทธศักราชเป็นคริสต์ศักราช
data['วันที่/เวลา'] = data['วันที่/เวลา'].str.replace(r'(\d{1,2}/\d{1,2}/)(\d{4})', 
    lambda x: f"{x.group(1)}{int(x.group(2)) - 543}", 
    regex=True)

# แปลงวันที่เป็น datetime
data['วันที่/เวลา'] = pd.to_datetime(data['วันที่/เวลา'], format='%d/%m/%Y %H:%M', errors='coerce')

# สร้างฟีเจอร์วัน
data['วัน'] = (data['วันที่/เวลา'] - data['วันที่/เวลา'].min()).dt.days

# กำหนด X และ y
X = data[['วัน']]

y = data['รับซื้อ(บาท)']

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# สร้างโมเดล Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# สร้างโมเดล Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X_train, y_train)

# คาดการณ์
lin_pred = lin_reg.predict(X_test)
rf_pred = rf_reg.predict(X_test)

# ประเมินผล
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"Linear Regression RMSE: {lin_rmse:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")



first_row = data.iloc[0]  # แถวแรก
date_time = first_row['วันที่/เวลา']  # วันที่และเวลา
buy_price = first_row['รับซื้อ(บาท)']  # ราคา

print(f"วันที่/เวลา: {date_time}, รับซื้อ(บาท): {buy_price}")
# คาดการณ์ราคาสำหรับวันถัดไป
next_day = np.array([[X['วัน'].max() + 1]])

predicted_price_lin = lin_reg.predict(next_day)
predicted_price_rf = rf_reg.predict(next_day)


print(f"ราคาทองคำที่คาดการณ์ในวันถัดไป (Linear Regression): {predicted_price_lin[0]:,.2f} บาท")
print(f"ราคาทองคำที่คาดการณ์ในวันถัดไป (Random Forest): {predicted_price_rf[0]:,.2f} บาท")







# ส่วนของกราฟ
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# แปลงคอลัมน์ 'วันที่/เวลา' เป็น datetime
df['วันที่/เวลา'] = pd.to_datetime(df['วันที่/เวลา'], format='%d/%m/%Y %H:%M')
# สร้างกราฟ
plt.figure(figsize=(10, 6))

# Plot ราคาซื้อ with 'วันที่/เวลา'
plt.plot(df['วันที่/เวลา'], df['รับซื้อ(บาท)'], marker='o', linestyle='-', color='blue')

# ตั้งค่า label และ title
plt.xlabel('Date/Time')
plt.ylabel('Buying Price (Baht)')
plt.title('Gold Prices Over Time')

# ปรับรูปแบบของกริดและการแสดงผลแกน x ให้แสดงเป็นวันที่/เวลา
plt.grid(True)
plt.xticks(rotation=45)  # หมุนข้อความบนแกน x เพื่อให้ดูง่ายขึ้น

# ปรับ layout เพื่อให้ข้อความไม่ทับกัน
plt.tight_layout()

# แสดงผลกราฟ
plt.show()

