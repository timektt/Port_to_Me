import React from "react";

const DataCleaning = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 การล้างข้อมูล (Data Cleaning)</h1>
      <p className="mt-4">
        การล้างข้อมูล (Data Cleaning) เป็นขั้นตอนสำคัญของกระบวนการวิเคราะห์ข้อมูล
        เพื่อทำให้ข้อมูลมีคุณภาพ น่าเชื่อถือ และพร้อมสำหรับการใช้งานต่อไป
      </p>

      <h2 className="text-xl font-semibold mt-6">1. การลบค่าที่ขาดหายไป</h2>
      <p className="mt-2">
        ใช้ <code>dropna()</code> เพื่อลบแถวที่มีค่า <code>NaN</code> ออกไป:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`import pandas as pd

data = {
  'Name': ['Alice', 'Bob', None, 'David'],
  'Age': [25, 30, None, 40]
}
df = pd.DataFrame(data)

df_cleaned = df.dropna()
print(df_cleaned)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. การแทนค่าที่ขาดหายไป</h2>
      <p className="mt-2">
        ใช้ <code>fillna()</code> เพื่อแทนที่ค่า <code>NaN</code> ด้วยค่าเฉลี่ยหรือค่าที่กำหนด:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`df_filled = df.fillna({'Age': df['Age'].mean()})
print(df_filled)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. การลบข้อมูลซ้ำซ้อน</h2>
      <p className="mt-2">
        ใช้ <code>drop_duplicates()</code> เพื่อลบแถวที่ซ้ำกันใน DataFrame:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`df_unique = df.drop_duplicates()
print(df_unique)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. การปรับรูปแบบข้อมูล</h2>
      <p className="mt-2">
        ใช้ <code>str.strip()</code> และ <code>str.title()</code> เพื่อลบช่องว่างและแปลงข้อความให้อยู่ในรูปแบบที่เหมาะสม:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`df['Name'] = df['Name'].str.strip().str.title()
print(df)`}
      </pre>
    </div>
  );
};

export default DataCleaning;
