import React from "react";

const DataTransformation = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">การแปลงข้อมูล</h1>
      <p>การแปลงข้อมูลช่วยให้สามารถปรับเปลี่ยนรูปแบบของข้อมูลให้เหมาะสมกับการวิเคราะห์</p>
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
    </div>
  );
};

export default DataTransformation;
