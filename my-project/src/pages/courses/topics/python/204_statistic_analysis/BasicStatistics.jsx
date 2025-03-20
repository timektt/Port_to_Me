import React from "react";

const BasicStatistics = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 สถิติพื้นฐาน (Basic Statistics)</h1>
      <p className="mt-4">
        สถิติพื้นฐานเป็นเครื่องมือสำคัญในการวิเคราะห์ข้อมูล ซึ่งใช้ในการวัดแนวโน้มส่วนกลางและการกระจายของข้อมูล เช่น ค่าเฉลี่ย ค่ามัธยฐาน และค่าเบี่ยงเบนมาตรฐาน
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การคำนวณค่าเฉลี่ยและค่ามัธยฐาน</h2>
      <p className="mt-2">ค่าเฉลี่ย (Mean) และค่ามัธยฐาน (Median) เป็นค่าที่ใช้บ่งบอกแนวโน้มของข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np

# ตัวอย่างข้อมูล
data = [10, 20, 30, 40, 50]

# คำนวณค่าเฉลี่ย (Mean)
mean_value = np.mean(data)
print("Mean:", mean_value)

# คำนวณค่ามัธยฐาน (Median)
median_value = np.median(data)
print("Median:", median_value)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การคำนวณค่ามัธยฐาน (Mode)</h2>
      <p className="mt-2">ค่ามัธยฐานเป็นค่าที่เกิดขึ้นบ่อยที่สุดในข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from scipy import stats

mode_value = stats.mode(data)
print("Mode:", mode_value.mode[0])`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การวัดการกระจายของข้อมูล</h2>
      <p className="mt-2">ค่าเบี่ยงเบนมาตรฐาน (Standard Deviation) ใช้ในการวัดการกระจายของข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`std_dev = np.std(data)
print("Standard Deviation:", std_dev)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การคำนวณค่าความแปรปรวน (Variance)</h2>
      <p className="mt-2">ค่าความแปรปรวนใช้บอกความกระจายของข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`variance = np.var(data)
print("Variance:", variance)`}</code>
      </pre>
    </div>
  );
};

export default BasicStatistics;
