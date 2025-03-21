import React from "react";

const PandasData = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        การจัดการข้อมูลด้วย Pandas
      </h1>

      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Pandas เป็นไลบรารีที่ทรงพลังสำหรับการจัดการและวิเคราะห์ข้อมูลใน Python โดยใช้โครงสร้างข้อมูลหลักคือ DataFrame และ Series
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การสร้าง DataFrame</h2>
      <p className="mt-2">DataFrame คือข้อมูลแบบตารางที่สามารถเก็บข้อมูลหลายประเภทในแต่ละคอลัมน์ได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre>
{`import pandas as pd

data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การอ่านไฟล์ข้อมูล</h2>
      <p className="mt-2">ใช้ `read_csv()` สำหรับอ่านข้อมูลจากไฟล์ CSV</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`df = pd.read_csv("data.csv")
print(df.head())`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การเข้าถึงข้อมูล</h2>
      <p className="mt-2">สามารถเข้าถึงข้อมูลแบบคอลัมน์ แถว หรือเซลล์ได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`df["Name"]       # คอลัมน์
df.iloc[0]        # แถวแรก
df.at[0, "Age"]   # ค่าเฉพาะจุด`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การกรองข้อมูล</h2>
      <p className="mt-2">ใช้เงื่อนไข Boolean ในการกรองแถวที่ตรงตามเงื่อนไข</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`filtered_df = df[df["Age"] > 25]
print(filtered_df)`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">5. การสรุปข้อมูลเบื้องต้น</h2>
      <p className="mt-2">ใช้ `describe()` และ `info()` เพื่อดูภาพรวมของข้อมูล</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`print(df.describe())   # ค่าสถิติเบื้องต้น
print(df.info())        # ข้อมูลคอลัมน์และชนิดข้อมูล`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">6. การจัดการค่าว่าง (Missing Data)</h2>
      <p className="mt-2">ใช้ `dropna()` และ `fillna()` เพื่อจัดการข้อมูลที่หายไป</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`df.dropna()               # ลบแถวที่มี NaN
df.fillna("ไม่มีข้อมูล")  # แทนค่าที่หายไป`}
        </pre>
      </div>
    </div>
  );
};

export default PandasData;
