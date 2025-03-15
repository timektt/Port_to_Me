import React from "react";

const BasicStatistics = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">สถิติพื้นฐาน</h1>
      <p>สถิติพื้นฐานเป็นเครื่องมือสำคัญในการวิเคราะห์ข้อมูล</p>
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
    </div>
  );
};

export default BasicStatistics;
