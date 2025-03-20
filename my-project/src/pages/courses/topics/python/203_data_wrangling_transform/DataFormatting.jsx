import React from "react";

const DataFormatting = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 การจัดรูปแบบข้อมูล (Data Formatting)</h1>
      <p className="mt-4">
        การจัดรูปแบบข้อมูลช่วยทำให้ข้อมูลมีโครงสร้างที่เป็นระเบียบและสะดวกต่อการนำไปใช้งานในการวิเคราะห์ข้อมูลหรือประมวลผล
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การแปลงวันที่ให้เป็น DateTime</h2>
      <p className="mt-2">การทำให้คอลัมน์ที่เก็บข้อมูลวันที่มีรูปแบบที่ถูกต้อง</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import pandas as pd

# สร้าง DataFrame ตัวอย่าง
data = {'Date': ['2025-03-01', '2025-03-02', '2025-03-03'],
        'Value': [100, 200, 150]}

df = pd.DataFrame(data)

# แปลงคอลัมน์ Date เป็น datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
print(df)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การแปลงตัวอักษรให้เป็นมาตรฐาน</h2>
      <p className="mt-2">ทำให้ข้อมูลตัวอักษรอยู่ในรูปแบบที่เหมาะสม เช่น ตัวพิมพ์เล็กหรือตัวพิมพ์ใหญ่</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df['Category'] = df['Category'].str.lower()  # แปลงเป็นตัวพิมพ์เล็ก
df['Category'] = df['Category'].str.title()  # แปลงให้ตัวอักษรขึ้นต้นเป็นพิมพ์ใหญ่`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การลบช่องว่างที่ไม่จำเป็น</h2>
      <p className="mt-2">สามารถใช้ `.strip()` เพื่อลบช่องว่างข้างหน้าและข้างหลังของข้อความ</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df['Name'] = df['Name'].str.strip()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การแปลงประเภทข้อมูล</h2>
      <p className="mt-2">เปลี่ยนประเภทข้อมูลให้เหมาะสม เช่น การแปลงตัวเลขที่เป็น string ให้เป็น int</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`df['Value'] = df['Value'].astype(int)`}</code>
      </pre>
    </div>
  );
};

export default DataFormatting;