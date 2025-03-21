import React from "react";

const MatplotlibBasics = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-3xl mx-auto">
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        📊 พื้นฐาน Matplotlib
      </h1>

      <p className="mt-4 text-center sm:text-left leading-relaxed">
        Matplotlib เป็นไลบรารีสำหรับการสร้างกราฟใน Python ใช้สำหรับสร้าง Visualization ได้หลากหลาย เช่น กราฟเส้น กราฟแท่ง ฮิสโตแกรม และอื่น ๆ
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การติดตั้ง Matplotlib</h2>
      <p className="mt-2">ติดตั้งด้วยคำสั่ง:</p>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap font-mono text-sm sm:text-base">
{`pip install matplotlib`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การสร้างกราฟเส้น (Line Chart)</h2>
      <p className="mt-2 font-semibold">📌 ตัวอย่างโค้ด:</p>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap font-mono text-sm sm:text-base">
{`import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

plt.plot(x, y)
plt.xlabel('แกน X')
plt.ylabel('แกน Y')
plt.title('ตัวอย่างกราฟ Matplotlib')
plt.show()`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. เพิ่มเส้นตาราง / เปลี่ยนสี</h2>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap font-mono text-sm sm:text-base">
{`plt.plot(x, y, color='red', linestyle='dashed', marker='o')
plt.grid(True)`}
        </pre>
      </div>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-300 rounded-lg shadow-md">
        💡 <strong>Tip:</strong> เพิ่ม <code className="bg-gray-300 text-gray-900 px-1 rounded dark:bg-gray-700 dark:text-gray-200">plt.grid(True)</code> เพื่อแสดงเส้นตารางในกราฟของคุณ
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การสร้างกราฟแท่ง (Bar Chart)</h2>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap font-mono text-sm sm:text-base">
{`labels = ['A', 'B', 'C']
values = [10, 20, 15]

plt.bar(labels, values)
plt.title('กราฟแท่งตัวอย่าง')
plt.show()`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">5. การสร้างฮิสโตแกรม (Histogram)</h2>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap font-mono text-sm sm:text-base">
{`import numpy as np

data = np.random.randn(1000)
plt.hist(data, bins=30, color='skyblue')
plt.title('Histogram ตัวอย่าง')
plt.show()`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">6. การบันทึกภาพกราฟลงไฟล์</h2>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap font-mono text-sm sm:text-base">
{`plt.plot(x, y)
plt.savefig('chart.png')  # บันทึกเป็นไฟล์ภาพ`}
        </pre>
      </div>
    </div>
  );
};

export default MatplotlibBasics;
