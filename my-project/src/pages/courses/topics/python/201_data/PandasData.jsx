import React from "react";

const PandasData = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        การจัดการข้อมูลด้วย Pandas
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Pandas เป็นไลบรารีที่ทรงพลังสำหรับการจัดการและวิเคราะห์ข้อมูลใน Python โดยใช้โครงสร้างข้อมูลหลักคือ DataFrame และ Series
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การสร้าง DataFrame</h2>
      <p className="mt-2">DataFrame เป็นโครงสร้างข้อมูลแบบตารางที่สามารถเก็บข้อมูลหลายประเภทได้</p>
      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`import pandas as pd

data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การอ่านไฟล์ข้อมูล</h2>
      <p className="mt-2">สามารถโหลดข้อมูลจากไฟล์ CSV ได้โดยใช้ `read_csv()`</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`df = pd.read_csv("data.csv")
print(df.head())  # แสดง 5 แถวแรกของข้อมูล`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การเข้าถึงข้อมูลใน DataFrame</h2>
      <p className="mt-2">สามารถเข้าถึงข้อมูลแต่ละคอลัมน์หรือแต่ละแถวได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`print(df["Name"])  # เข้าถึงคอลัมน์ Name
print(df.iloc[0])  # เข้าถึงแถวแรกของ DataFrame`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การกรองข้อมูล</h2>
      <p className="mt-2">สามารถใช้เงื่อนไขเพื่อกรองข้อมูลที่ต้องการได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`filtered_df = df[df["Age"] > 25]
print(filtered_df)`}
        </pre>
      </div>
    </div>
  );
};

export default PandasData;
