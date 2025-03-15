import React from "react";

const DataCleaning = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">การล้างข้อมูล</h1>
      <p>การล้างข้อมูลเป็นกระบวนการจัดการข้อมูลที่ผิดพลาด ข้อมูลซ้ำ และข้อมูลที่ไม่สมบูรณ์</p>
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
    </div>
  );
};

export default DataCleaning;
