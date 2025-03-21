import React from "react";

const BasicStatistics = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 สถิติพื้นฐาน (Basic Statistics)</h1>
      <p className="mt-4">
        สถิติพื้นฐานเป็นพื้นฐานของการวิเคราะห์ข้อมูล ช่วยให้เข้าใจข้อมูลโดยรวมได้ชัดเจนขึ้น เช่น ค่าเฉลี่ย ค่ามัธยฐาน ค่าฐานนิยม ความแปรปรวน และค่ามาตรฐานเบี่ยงเบน
      </p>

      <h2 className="text-xl font-semibold mt-6">1. ค่าเฉลี่ย (Mean) และ ค่ามัธยฐาน (Median)</h2>
      <p className="mt-2">- ค่าเฉลี่ยแสดงค่าโดยรวมของข้อมูลทั้งหมด<br />- ค่ามัธยฐานแสดงค่าตรงกลาง (เมื่อเรียงลำดับข้อมูล)</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`import numpy as np

data = [10, 20, 30, 40, 50]

mean_value = np.mean(data)
print("Mean:", mean_value)

median_value = np.median(data)
print("Median:", median_value)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. ฐานนิยม (Mode)</h2>
      <p className="mt-2">- ฐานนิยม (Mode) คือค่าที่เกิดขึ้นบ่อยที่สุดในชุดข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`from scipy import stats

mode_value = stats.mode(data, keepdims=True)
print("Mode:", mode_value.mode[0])`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. ค่ามาตรฐานเบี่ยงเบน (Standard Deviation)</h2>
      <p className="mt-2">- ใช้ดูว่าข้อมูลกระจายห่างจากค่าเฉลี่ยมากน้อยเพียงใด</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`std_dev = np.std(data)
print("Standard Deviation:", std_dev)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. ความแปรปรวน (Variance)</h2>
      <p className="mt-2">- เป็นการวัดความกระจายที่คล้ายกับ Standard Deviation แต่ไม่ถอดรูท</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`variance = np.var(data)
print("Variance:", variance)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔍 ตัวอย่างผลลัพธ์ (ถ้ารันใน Python)</h2>
      <div className="bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-white p-4 rounded-md mt-2">
        <pre>
{`Mean: 30.0
Median: 30.0
Mode: 10
Standard Deviation: 14.14
Variance: 200.0`}
        </pre>
      </div>
    </div>
  );
};

export default BasicStatistics;
