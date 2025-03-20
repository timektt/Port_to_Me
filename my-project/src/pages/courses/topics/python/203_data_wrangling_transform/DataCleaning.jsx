import React from "react";

const DataCleaning = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 การล้างข้อมูล (Data Cleaning)</h1>
      <p className="mt-4">
        การล้างข้อมูลเป็นกระบวนการที่สำคัญในการเตรียมข้อมูลให้พร้อมสำหรับการวิเคราะห์ โดยเป็นการจัดการข้อมูลที่ขาดหาย ซ้ำซ้อน หรือมีค่าผิดพลาด
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การลบค่าที่ขาดหายไป (Handling Missing Data)</h2>
      <p className="mt-2">สามารถใช้ `dropna()` เพื่อลบแถวที่มีค่าเป็น `NaN`</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import pandas as pd

# สร้าง DataFrame ตัวอย่าง
data = {'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, 30, None, 40]}

df = pd.DataFrame(data)

# ลบค่าที่หายไป
df_cleaned = df.dropna()
print(df_cleaned)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การแทนค่าที่ขาดหายไป</h2>
      <p className="mt-2">สามารถใช้ `fillna()` เพื่อแทนค่าที่หายไปด้วยค่าที่เหมาะสม เช่น ค่าเฉลี่ย</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df_filled = df.fillna({'Age': df['Age'].mean()})
print(df_filled)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การลบข้อมูลซ้ำซ้อน (Removing Duplicates)</h2>
      <p className="mt-2">สามารถใช้ `drop_duplicates()` เพื่อลบแถวที่มีค่าซ้ำกัน</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df_unique = df.drop_duplicates()
print(df_unique)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การปรับรูปแบบข้อมูล (Data Formatting)</h2>
      <p className="mt-2">การแปลงค่าของข้อมูลให้อยู่ในรูปแบบที่เหมาะสม เช่น การแปลงตัวพิมพ์เล็ก-ใหญ่ หรือการลบช่องว่าง</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df['Name'] = df['Name'].str.strip().str.title()
print(df)`}</code>
      </pre>
    </div>
  );
};

export default DataCleaning;
