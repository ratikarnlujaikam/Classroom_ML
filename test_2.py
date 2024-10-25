from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import io
import csv

# กำหนดให้คอนโซลรองรับ UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# สร้างตัวแปรสำหรับ WebDriver (ใช้ Chrome ในที่นี้)
driver = webdriver.Chrome()

# เปิด URL ที่ต้องการ
url = 'https://xn--42cah7d0cxcvbbb9x.com/%E0%B8%A3%E0%B8%B2%E0%B8%84%E0%B8%B2%E0%B8%97%E0%B8%AD%E0%B8%87%E0%B8%A2%E0%B9%89%E0%B8%AD%E0%B8%99%E0%B8%AB%E0%B8%A5%E0%B8%B1%E0%B8%87/'
driver.get(url)

# รอให้ตารางโหลด
try:
    table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="post-74"]/div[6]/div[2]/table'))
    )
    
    rows = table.find_elements(By.TAG_NAME, 'tr')

    # สร้างรายการเพื่อเก็บข้อมูล
    formatted_data = []

    # หัวตาราง
    headers = ["วันที่/เวลา", "ครั้งที่", "รับซื้อ(บาท)", "ขายออก(บาท)", "ฐานภาษี(บาท)", "Gold spot", "เงินบาท", "ขึ้น/ลง"]
    formatted_data.append(headers)  # เพิ่มหัวตารางในรายการ

    # แสดงข้อมูลในแต่ละแถว
    for i, row in enumerate(rows):
        # ข้ามแถวสุดท้าย
        if i == len(rows) - 1:
            continue
        
        columns = row.find_elements(By.TAG_NAME, 'td')
        if columns:  # ตรวจสอบว่าแถวมีข้อมูลหรือไม่
            data = [column.text for column in columns]
            # ตรวจสอบว่าข้อมูลมีเครื่องหมายจุลภาคและไม่ใช่ "l6fmhkp"
            if any(',' in value for value in data) and "l6fmhkp" not in data:
                formatted_data.append(data)  # เพิ่มข้อมูลในแถวที่ตรงตามเงื่อนไข

    # บันทึกข้อมูลลงในไฟล์ CSV
    with open('gold_prices.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(formatted_data)  # เขียนข้อมูลทั้งหมดลงในไฟล์ CSV

    print("ข้อมูลได้ถูกบันทึกลงใน gold_prices.csv")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")

finally:
    # ปิด WebDriver
    driver.quit()
    
    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# อ่านไฟล์ CSV โดยข้ามแถวที่มีข้อผิดพลาด
data = pd.read_csv('gold_prices.csv', on_bad_lines='skip')

# แสดงข้อมูล
print(data.head())  # แสดงเฉพาะ 5 แถวแรก

# # ตรวจสอบข้อมูล
# print(data.isna().sum())  # ดูว่ามี NaN ในคอลัมน์ใดบ้าง

# # ลบแถวที่มี NaN
# data.dropna(inplace=True)

# สร้าง Feature
data['วัน'] = pd.to_datetime(data['วันที่/เวลา']).dt.dayofyear
X = data[['วัน']]
y = pd.to_numeric(data['รับซื้อ(บาท)'].str.replace(',', ''), errors='coerce')  # แปลงค่าจาก string เป็น float

# ลบ NaN ใน y
y.dropna(inplace=True)

# ตรวจสอบจำนวนตัวอย่างหลังจากลบ NaN
if len(y) == 0:
    print("ไม่มีข้อมูลเพียงพอสำหรับการฝึกโมเดล")
else:
    # แบ่งข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(X[:len(y)], y, test_size=0.2, random_state=0)

    # สร้างโมเดล
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ทำนายราคาสำหรับวันถัดไป
    next_day = np.array([[X['วัน'].max() + 1]])  # วันถัดไป
    predicted_price = model.predict(next_day)

    print(f"ราคาทองคำที่คาดการณ์ในวันถัดไป: {predicted_price[0]:,.2f} บาท")
