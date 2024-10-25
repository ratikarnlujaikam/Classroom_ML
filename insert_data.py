import pandas as pd

# อ่านไฟล์ Excel
file_path = 'path_to_your_excel_file.xlsx'
df = pd.read_excel(file_path, header=None)

# แบ่งข้อมูลโดยใช้เครื่องหมาย |
df_split = df[0].str.split('|', expand=True)

# แสดงผลข้อมูลที่แบ่งแล้ว
print(df_split)
