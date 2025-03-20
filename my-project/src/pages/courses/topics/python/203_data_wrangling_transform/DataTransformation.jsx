import React from "react";

const DataTransformation = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 การแปลงข้อมูล (Data Transformation)</h1>
      <p className="mt-4">
        การแปลงข้อมูลเป็นกระบวนการที่ช่วยให้สามารถปรับเปลี่ยนรูปแบบของข้อมูลให้อยู่ในรูปแบบที่เหมาะสมกับการวิเคราะห์ข้อมูลและ Machine Learning เช่น การเปลี่ยนหน่วย, การสร้างคอลัมน์ใหม่, หรือการแปลงประเภทข้อมูล
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การแปลงหน่วยข้อมูล</h2>
      <p className="mt-2">แปลงเงินเดือนจาก USD เป็น THB โดยใช้ค่าอัตราแลกเปลี่ยน 35 บาทต่อ 1 USD</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import pandas as pd

# สร้าง DataFrame ตัวอย่าง
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Salary': [50000, 60000, 55000, 65000]}

df = pd.DataFrame(data)

# แปลงหน่วยเงินจาก USD เป็น THB
df['Salary_THB'] = df['Salary'] * 35
print(df)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การสร้างคอลัมน์ใหม่จากข้อมูลเดิม</h2>
      <p className="mt-2">สร้างคอลัมน์แสดงระดับเงินเดือน โดยแบ่งตามเกณฑ์</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`def categorize_salary(salary):
    if salary > 60000:
        return 'High'
    elif salary > 50000:
        return 'Medium'
    else:
        return 'Low'

df['Salary_Level'] = df['Salary'].apply(categorize_salary)
print(df)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การแปลงประเภทข้อมูล</h2>
      <p className="mt-2">เปลี่ยนประเภทข้อมูลจากตัวเลขให้เป็นสตริง หรือแปลงค่าตัวเลขให้อยู่ในช่วงมาตรฐาน</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df['Salary'] = df['Salary'].astype(str)  # แปลงตัวเลขเป็นข้อความ

# การปรับขนาดข้อมูลให้อยู่ในช่วง 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['Salary_Normalized'] = scaler.fit_transform(df[['Salary_THB']])
print(df)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การแปลงวันที่และเวลา</h2>
      <p className="mt-2">เปลี่ยนวันที่จากสตริงให้เป็นรูปแบบ datetime</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df['Join_Date'] = pd.to_datetime(['2022-01-01', '2023-03-15', '2021-07-30', '2024-02-20'])
print(df)`}</code>
      </pre>
    </div>
  );
};

export default DataTransformation;
