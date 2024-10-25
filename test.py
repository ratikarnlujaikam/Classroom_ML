import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV พร้อมระบุการเข้ารหัส
df = pd.read_csv("year.csv", encoding='utf-8')  # เปลี่ยนชื่อไฟล์ให้ตรงกับของคุณ

# แสดงข้อมูลเพื่อยืนยันการอ่าน
print(df.to_string(index=False).encode('utf-8').decode('utf-8'))

# แปลงปีเป็นตัวแปรเชิงตัวเลขในรูปแบบ พ.ศ.
df['Year_Buddhist'] = df['Year'] + 543  # เปลี่ยนปี ค.ศ. เป็น พ.ศ.

# เตรียมข้อมูลสำหรับการเรียนรู้ของเครื่อง
X = df[['Year']]  # ใช้ปี ค.ศ. เป็นตัวแปรในการพยากรณ์
y = df['Revenue']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# พยากรณ์รายได้ในปีถัดไป (2024) และ 5 ปีถัดไป
future_years = []
predicted_revenues = []

for i in range(1, 6):  # พยากรณ์ 5 ปีถัดไป
    next_year = df['Year'].max() + i
    next_year_time = [[next_year]]  # ปีถัดไปใน ค.ศ.
    predicted_revenue = model.predict(next_year_time)
    future_years.append(next_year + 543)  # แปลงเป็น พ.ศ.
    predicted_revenues.append(predicted_revenue[0])

# แสดงผลการพยากรณ์
for year, revenue in zip(future_years, predicted_revenues):
    print(f"Predicted tourism revenue for year {year}: {revenue:.2f} million baht")

# วาดกราฟ
plt.figure(figsize=(10, 6))
plt.scatter(df['Year_Buddhist'], df['Revenue'], color='blue', label='Actual Data')
plt.plot(df['Year'], model.predict(X), color='red', label='Random Forest Prediction')

# เพิ่มการพยากรณ์ในกราฟ
for year, revenue in zip(future_years, predicted_revenues):
    plt.scatter(year, revenue, color='green', s=100, label='Future Prediction' if year == future_years[0] else "")

# วาดกราฟ
plt.xlabel('Year (B.E.)')  # ปรับชื่อแกน X
plt.ylabel('Revenue (Million Baht)')
plt.title('Tourism Revenue Prediction Using Random Forest')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
