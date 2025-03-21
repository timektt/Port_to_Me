import React from "react";

const DataFormatting = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 การจัดรูปแบบข้อมูล (Data Formatting)</h1>
      <p className="mt-4">
        การจัดรูปแบบข้อมูล (Data Formatting) เป็นขั้นตอนในการปรับปรุงข้อมูลให้อยู่ในรูปแบบที่เหมาะสมกับการวิเคราะห์ เช่น การจัดการวันที่, ข้อความ, ตัวเลข และการลบช่องว่างที่ไม่จำเป็น
      </p>

      <h2 className="text-xl font-semibold mt-6">1. การแปลงวันที่ให้เป็น DateTime</h2>
      <p className="mt-2">ใช้ <code>pd.to_datetime()</code> เพื่อให้คอลัมน์วันที่มีรูปแบบมาตรฐาน</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`import pandas as pd

data = {
  'Date': ['2025-03-01', '2025-03-02', '2025-03-03'],
  'Value': [100, 200, 150]
}
df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
print(df)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. การแปลงตัวอักษรให้เป็นมาตรฐาน</h2>
      <p className="mt-2">ใช้ <code>str.lower()</code>, <code>str.upper()</code>, หรือ <code>str.title()</code> เพื่อจัดรูปแบบข้อความ</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`df['Category'] = df['Category'].str.lower()   # แปลงเป็นตัวพิมพ์เล็ก
df['Category'] = df['Category'].str.title()   # แปลงให้ตัวอักษรต้นคำเป็นพิมพ์ใหญ่`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. การลบช่องว่างที่ไม่จำเป็น</h2>
      <p className="mt-2">ใช้ <code>str.strip()</code> เพื่อลบช่องว่างข้างหน้าหรือข้างหลังของข้อความ</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`df['Name'] = df['Name'].str.strip()`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. การแปลงประเภทข้อมูล</h2>
      <p className="mt-2">ใช้ <code>astype()</code> เพื่อแปลงชนิดข้อมูล เช่น จาก string เป็น int</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`df['Value'] = df['Value'].astype(int)`}
      </pre>
    </div>
  );
};

export default DataFormatting;
