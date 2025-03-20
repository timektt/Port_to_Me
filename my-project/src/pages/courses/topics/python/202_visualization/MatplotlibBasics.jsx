import React from "react";

const MatplotlibBasics = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        📊 พื้นฐาน Matplotlib
      </h1>

      {/* ✅ Description */}
      <p className="mt-4 text-center sm:text-left leading-relaxed">
        Matplotlib เป็นไลบรารีสำหรับการสร้างกราฟใน Python สามารถใช้เพื่อสร้าง Visualization ที่หลากหลาย เช่น กราฟเส้น กราฟแท่ง และฮิสโตแกรม
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การติดตั้ง Matplotlib</h2>
      <p className="mt-2">ก่อนใช้งาน Matplotlib ต้องติดตั้งไลบรารีก่อน โดยใช้คำสั่ง:</p>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap sm:whitespace-pre text-sm sm:text-base font-mono">
{`pip install matplotlib`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. ตัวอย่างโค้ดการสร้างกราฟเส้น</h2>
      <p className="mt-2 font-semibold">📌 ตัวอย่างโค้ด:</p>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap sm:whitespace-pre text-sm sm:text-base font-mono">
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
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การเพิ่มเส้นตารางและการเปลี่ยนสีของเส้น</h2>
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap sm:whitespace-pre text-sm sm:text-base font-mono">
{`plt.plot(x, y, color='red', linestyle='dashed', marker='o')
plt.grid(True)`}
        </pre>
      </div>
      
      {/* ✅ Info Box */}
      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-300 rounded-lg shadow-md">
        💡 <span className="font-semibold">Tip:</span>  
        Matplotlib สามารถใช้กำหนดค่าต่างๆ เช่น สีของเส้น และรูปแบบจุดบนกราฟได้ ลองเพิ่ม <code className="bg-gray-300 text-gray-900 px-1 rounded dark:bg-gray-700 dark:text-gray-200">plt.grid(True)</code> เพื่อเพิ่มเส้นตารางให้กราฟของคุณ
      </div>
    </div>
  );
};

export default MatplotlibBasics;
