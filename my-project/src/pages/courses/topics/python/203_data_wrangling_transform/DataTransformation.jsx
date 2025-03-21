import React from "react";

const DataTransformation = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 การแปลงข้อมูล (Data Transformation)</h1>
      <p className="mt-4">
        การแปลงข้อมูล (Transformation) ช่วยให้ข้อมูลพร้อมสำหรับการวิเคราะห์และการนำไปใช้ต่อ เช่น การแปลงหน่วย, การสร้างคอลัมน์ใหม่, การปรับประเภทข้อมูล และการจัดการวันที่
      </p>

      <h2 className="text-xl font-semibold mt-6">1. การแปลงหน่วย (Unit Conversion)</h2>
      <p className="mt-2">เช่น การแปลงเงินเดือนจาก USD เป็น THB โดยใช้อัตราแลกเปลี่ยน 35 บาทต่อ 1 USD</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`import pandas as pd

data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'David'],
  'Salary': [50000, 60000, 55000, 65000]
}
df = pd.DataFrame(data)

df['Salary_THB'] = df['Salary'] * 35
print(df)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. การสร้างคอลัมน์ใหม่จากคอลัมน์เดิม</h2>
      <p className="mt-2">สร้างคอลัมน์ใหม่ที่แสดงระดับเงินเดือน (Low / Medium / High)</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`def categorize_salary(salary):
    if salary > 60000:
        return 'High'
    elif salary > 50000:
        return 'Medium'
    else:
        return 'Low'

df['Salary_Level'] = df['Salary'].apply(categorize_salary)
print(df)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. การแปลงประเภทข้อมูล (Data Type Conversion)</h2>
      <p className="mt-2">เปลี่ยนประเภทข้อมูลเป็น string หรือ normalize ให้อยู่ในช่วง 0-1</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`df['Salary'] = df['Salary'].astype(str)  # แปลงเป็นข้อความ

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['Salary_Normalized'] = scaler.fit_transform(df[['Salary_THB']])
print(df)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. การแปลงวันที่และเวลา</h2>
      <p className="mt-2">เปลี่ยนวันที่จาก string เป็น datetime เพื่อใช้ใน Time Series หรือวิเคราะห์วัน</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`df['Join_Date'] = pd.to_datetime([
  '2022-01-01', '2023-03-15', '2021-07-30', '2024-02-20'
])
print(df)`}
      </pre>
    </div>
  );
};

export default DataTransformation;
