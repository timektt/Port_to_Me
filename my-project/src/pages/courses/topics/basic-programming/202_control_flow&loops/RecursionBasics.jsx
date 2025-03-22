import React from "react";

const RecursionBasics = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">📌 พื้นฐานของการเรียกซ้ำ (Recursion Basics)</h1>

      <p className="mt-2 text-gray-700 dark:text-gray-300 leading-relaxed">
        <strong>Recursion</strong> หรือ “การเรียกซ้ำ” เป็นเทคนิคในการเขียนฟังก์ชันที่เรียกตัวเองซ้ำ ๆ
        โดยแต่ละรอบจะมีเงื่อนไขเพื่อหยุดการทำงานเมื่อถึงจุดที่กำหนด เรียกว่า “<strong>base case</strong>”
      </p>

      <h2 className="text-xl font-semibold mt-6">🔁 ตัวอย่างการเรียกซ้ำ</h2>
      <p className="mt-2">ตัวอย่างฟังก์ชันที่ใช้ recursion เพื่อหาค่า factorial:</p>
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm sm:text-base mt-2">
        <pre>
{`def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # Output: 120`}
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">⚠️ ความสำคัญของ Base Case</h2>
      <p className="mt-2">
        Base case ช่วยหยุดการเรียกซ้ำ ถ้าไม่มี base case โปรแกรมจะทำงานต่อไปเรื่อย ๆ จน Stack Overflow เกิดขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">🧠 การใช้ Recursion ในชีวิตจริง</h2>
      <ul className="list-disc ml-5 mt-2 text-gray-700 dark:text-gray-300">
        <li>การคำนวณฟีโบนัชชี (Fibonacci Sequence)</li>
        <li>การค้นหาในโครงสร้างต้นไม้ (Tree Traversal)</li>
        <li>การแก้ปัญหาแบบแบ่งย่อย เช่น Merge Sort</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📉 เปรียบเทียบกับ Loop</h2>
      <p className="mt-2">
        การเขียนแบบ loop มักใช้หน่วยความจำน้อยกว่า recursion แต่ recursion จะทำให้โค้ดกระชับและเข้าใจง่ายสำหรับบางปัญหา
      </p>

      <h2 className="text-xl font-semibold mt-6">📍 ข้อควรระวัง</h2>
      <ul className="list-disc ml-5 mt-2 text-red-600 dark:text-red-400">
        <li>อย่าลืม base case</li>
        <li>หลีกเลี่ยง recursion ในกรณีที่มีการเรียกซ้ำจำนวนมากโดยไม่ใช้ memoization</li>
      </ul>
    </div>
  );
};

export default RecursionBasics;
