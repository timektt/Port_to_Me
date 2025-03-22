import React from "react";

const BreakContinueStatements = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">🧭 คำสั่ง Break และ Continue</h1>
      <p className="mb-4">
        คำสั่ง <code>break</code> และ <code>continue</code> เป็นคำสั่งควบคุมลูปที่ช่วยให้เราสามารถควบคุมการทำงานของลูปได้ตามต้องการ โดย:
      </p>
      <ul className="list-disc ml-6 mb-6">
        <li><strong>break</strong>: ใช้สำหรับหยุดการทำงานของลูปทันที</li>
        <li><strong>continue</strong>: ข้ามการทำงานในรอบนั้น และไปทำรอบถัดไป</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้ break</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`for i in range(1, 10):
    if i == 5:
        break
    print(i)`}
      </pre>
      <p className="mt-2">ในตัวอย่างนี้ เมื่อ i มีค่าเป็น 5 ลูปจะหยุดการทำงานทันที</p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้ continue</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`for i in range(1, 6):
    if i == 3:
        continue
    print(i)`}
      </pre>
      <p className="mt-2">ในตัวอย่างนี้ เมื่อ i มีค่าเป็น 3 ลูปจะข้ามรอบนั้นและไปทำรอบถัดไปทันที</p>

      <h2 className="text-2xl font-semibold mt-6">📍 การใช้ร่วมกับเงื่อนไข</h2>
      <p className="mt-2">
        ทั้งคำสั่ง <code>break</code> และ <code>continue</code> สามารถใช้ร่วมกับเงื่อนไข <code>if</code> เพื่อควบคุมลูปให้ยืดหยุ่นได้ เช่น ตรวจสอบค่าว่าเป็นคู่หรือคี่ แล้วเลือกหยุดหรือข้ามได้ตามต้องการ
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-4">
{`for i in range(1, 11):
    if i % 2 == 0:
        continue  # ข้ามเลขคู่
    print("เลขคี่:", i)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🧠 สรุป</h2>
      <ul className="list-disc ml-6 mt-2">
        <li><code>break</code>: ใช้หยุดลูปทันทีเมื่อเข้าเงื่อนไข</li>
        <li><code>continue</code>: ใช้ข้ามการทำงานรอบปัจจุบันและไปยังรอบถัดไป</li>
        <li>ใช้สำหรับควบคุมลูปให้ทำงานตามเงื่อนไขที่ต้องการได้อย่างยืดหยุ่น</li>
      </ul>
    </div>
  );
};

export default BreakContinueStatements;
