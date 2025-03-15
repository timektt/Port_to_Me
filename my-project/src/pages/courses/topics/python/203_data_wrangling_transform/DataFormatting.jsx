import React from "react";

const DataFormatting = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">การจัดรูปแบบข้อมูล</h1>
      <p>การจัดรูปแบบข้อมูลช่วยทำให้ข้อมูลมีโครงสร้างที่เป็นระเบียบ</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import pandas as pd

# สร้าง DataFrame ตัวอย่าง
data = {'Date': ['2025-03-01', '2025-03-02', '2025-03-03'],
        'Value': [100, 200, 150]}

df = pd.DataFrame(data)

# แปลงคอลัมน์ Date เป็นรูปแบบ datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
print(df)`}</code>
      </pre>
    </div>
  );
};

export default DataFormatting;
