import React from "react";

const NestedLoops = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">🌀 ลูปซ้อน (Nested Loops)</h1>

      <p className="mb-4">
        ลูปซ้อน (Nested Loop) คือการใช้ลูปภายในลูปอีกทีหนึ่ง เพื่อให้สามารถวนซ้ำหลายระดับ เช่น การวนซ้ำเพื่อสร้างตารางหรือรูปแบบซ้ำซ้อนหลายมิติ
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">📌 ตัวอย่างลูปซ้อนแบบ For-For</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
{`for i in range(3):
    for j in range(2):
        print(f"รอบนอก i={i}, รอบใน j={j}")`}
      </pre>

      <p className="mt-2">
        โค้ดด้านบนจะวนลูปชั้นนอก 3 รอบ (i = 0 ถึง 2) และในแต่ละรอบจะวนลูปชั้นในอีก 2 รอบ (j = 0 ถึง 1)
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">🧮 ตัวอย่างการสร้างตาราง</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
{`for row in range(1, 4):
    for col in range(1, 4):
        print(row * col, end="\t")
    print()  # ขึ้นบรรทัดใหม่`}
      </pre>

      <p className="mt-2">
        Output จะเป็นตารางคูณ 3x3 โดยใช้ลูปซ้อนกันทั้งแถว (row) และคอลัมน์ (col)
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">⚠️ สิ่งที่ควรระวัง</h2>
      <ul className="list-disc ml-6 mt-2 space-y-2">
        <li>ระวังการเขียนลูปซ้อนที่ไม่มีเงื่อนไขสิ้นสุด อาจทำให้เกิด Infinite Loop</li>
        <li>การซ้อนลูปหลายชั้นมากเกินไป อาจทำให้โปรแกรมทำงานช้าหรืออ่านยาก</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6 mb-2">✅ สรุป</h2>
      <p>
        ลูปซ้อนเหมาะสำหรับการจัดการข้อมูลที่อยู่ในโครงสร้างหลายมิติ เช่น ตาราง เมทริกซ์ หรือการแสดงผลรูปแบบที่มีความซับซ้อน
      </p>
    </div>
  );
};

export default NestedLoops;
