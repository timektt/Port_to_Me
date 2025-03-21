import React from "react";

const PythonControlStructure = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">โครงสร้างควบคุมใน Python</h1>
      <p className="mt-4">
        โครงสร้างควบคุม (Control Structures) เป็นหัวใจของการเขียนโปรแกรมที่ใช้ควบคุมลำดับการทำงานของโค้ด
        ทำให้สามารถสร้างเงื่อนไขและการทำซ้ำได้ ซึ่งช่วยให้โปรแกรมตอบสนองตามสถานการณ์ที่เกิดขึ้นได้หลากหลาย
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. คำสั่งเงื่อนไข (Conditional Statements)</h2>
      <p className="mt-2">
        คำสั่ง if-elif-else ใช้เพื่อตรวจสอบเงื่อนไขหลายแบบและเลือกทำเฉพาะทางที่ตรงกับเงื่อนไขนั้น
      </p>
      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: If-Else</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
{`x = 10
if x > 0:
    print("เป็นจำนวนบวก")
elif x == 0:
    print("เป็นศูนย์")
else:
    print("เป็นจำนวนลบ")`}
      </pre>
      <p className="mt-2">
        นอกจากนี้ยังสามารถใช้ if ซ้อนกัน (Nested if) เพื่อเช็คเงื่อนไขหลายระดับได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">2. ลูป (Looping Structures)</h2>
      <p className="mt-2">
        ใช้สำหรับทำซ้ำคำสั่งเดิมหลายครั้ง เช่น การวนลูปผ่านรายการหรือวนลูปจนกว่าเงื่อนไขจะเป็นเท็จ
      </p>

      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ลูป For</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
{`for i in range(1, 6):
    print("รอบที่", i)`}
      </pre>

      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ลูป While</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
{`x = 0
while x < 5:
    print("ค่าของ x:", x)
    x += 1`}
      </pre>

      <p className="mt-2">
        สามารถใช้คำสั่ง <code>break</code> และ <code>continue</code> เพื่อควบคุมการวนลูปเพิ่มเติมได้เช่นกัน
      </p>
      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ใช้ break</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
{`for i in range(10):
    if i == 5:
        break
    print("i =", i)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. การจัดการข้อผิดพลาด (Exception Handling)</h2>
      <p className="mt-2">
        ใช้สำหรับป้องกันการหยุดทำงานของโปรแกรมเมื่อเกิดข้อผิดพลาด เช่น ข้อมูลที่ป้อนไม่ถูกต้อง หรือหารด้วยศูนย์
      </p>

      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ใช้ Try-Except</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
{`try:
    num = int(input("ป้อนตัวเลข: "))
    print("คุณป้อน:", num)
except ValueError:
    print("ข้อมูลไม่ถูกต้อง! กรุณาป้อนตัวเลขเท่านั้น")`}
      </pre>

      <h3 className="text-xl font-medium mt-3">ตัวอย่างเพิ่มเติม: ZeroDivisionError</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
{`try:
    x = 10 / 0
except ZeroDivisionError:
    print("ไม่สามารถหารด้วยศูนย์ได้")`}
      </pre>
    </div>
  );
};

export default PythonControlStructure;
