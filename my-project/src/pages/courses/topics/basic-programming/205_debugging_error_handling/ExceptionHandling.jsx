// File: ExceptionHandling.jsx

import React from "react";

const ExceptionHandling = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">🧯 การจัดการข้อยกเว้น (Exception Handling)</h1>

      <p className="mt-4 text-base sm:text-lg">
        ในการเขียนโปรแกรม ข้อผิดพลาดเป็นสิ่งที่หลีกเลี่ยงไม่ได้ เช่น การหารด้วยศูนย์ การเข้าถึงไฟล์ที่ไม่มีอยู่ หรือการรับข้อมูลที่ไม่ถูกต้อง
        <strong className="block mt-2">Exception Handling</strong> คือกลไกที่ช่วยให้โปรแกรมจัดการกับข้อผิดพลาดเหล่านี้ได้อย่างปลอดภัยโดยไม่ทำให้โปรแกรมหยุดทำงานทันที
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 โครงสร้าง try-except</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`try:
    # โค้ดที่อาจเกิดข้อผิดพลาด
    num = int(input("กรุณาใส่ตัวเลข: "))
    print("คุณใส่:", num)
except ValueError:
    print("ข้อมูลที่ใส่ไม่ใช่ตัวเลข!")`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">📍 ใช้ else และ finally เพิ่มเติม</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`try:
    x = 10 / 2
except ZeroDivisionError:
    print("หารด้วยศูนย์ไม่ได้")
else:
    print("ผลลัพธ์ =", x)
finally:
    print("ทำงานเสร็จสิ้นแล้ว")`}
      </pre>
      <ul className="list-disc ml-6 mt-2 text-sm sm:text-base">
        <li><strong>else</strong>: จะทำงานถ้าไม่มีข้อผิดพลาดในบล็อก try</li>
        <li><strong>finally</strong>: จะทำงานทุกกรณี ไม่ว่าจะเกิดหรือไม่เกิดข้อผิดพลาด</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">🚫 ตัวอย่างข้อผิดพลาดยอดนิยม</h2>
      <ul className="list-disc ml-6 mt-2 text-sm sm:text-base">
        <li><code>ZeroDivisionError</code>: หารด้วยศูนย์</li>
        <li><code>ValueError</code>: แปลงค่าข้อมูลผิดพลาด เช่น str → int</li>
        <li><code>FileNotFoundError</code>: หาไฟล์ไม่พบ</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <p className="mt-2">
        Exception Handling ช่วยให้โปรแกรมปลอดภัย และใช้งานได้ต่อเนื่องแม้เกิดปัญหา ช่วยเพิ่มประสบการณ์ผู้ใช้และความน่าเชื่อถือของโปรแกรมอย่างมาก
      </p>
    </div>
  );
};

export default ExceptionHandling;
