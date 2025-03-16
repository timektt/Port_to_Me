import React from "react";

const MatplotlibBasics = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left text-gray-900 dark:text-white">
        📊 Matplotlib Basics
      </h1>

      {/* ✅ Description */}
      <p className="mt-4 text-gray-700 dark:text-gray-300 text-center sm:text-left leading-relaxed">
        Matplotlib เป็นไลบรารีสำหรับการสร้างกราฟใน Python สามารถใช้เพื่อสร้าง Visualization ที่หลากหลาย เช่น กราฟเส้น กราฟแท่ง และฮิสโตแกรม
      </p>

      <p className="mt-2 text-gray-700 dark:text-gray-300 text-center sm:text-left font-semibold">
        📌 ตัวอย่างโค้ด:
      </p>

      {/* ✅ Code Block */}
      <div className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg">
        <pre className="whitespace-pre-wrap sm:whitespace-pre text-sm sm:text-base font-mono">
{`import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Matplotlib Graph')
plt.show()`}
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
