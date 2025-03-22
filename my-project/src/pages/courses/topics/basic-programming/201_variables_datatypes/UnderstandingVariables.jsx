import React from "react";

const UnderstandingVariables = () => {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6">
      <h1 className="text-2xl sm:text-3xl font-bold">📘 ทำความเข้าใจกับตัวแปร (Variables)</h1>
      <p className="mt-4 text-gray-700 dark:text-gray-300">
        ตัวแปร (Variable) คือสิ่งที่ใช้ในการเก็บข้อมูลไว้ในหน่วยความจำของคอมพิวเตอร์ ซึ่งเราสามารถนำตัวแปรไปใช้งานในภายหลังหรือปรับเปลี่ยนค่าได้ตลอดระยะเวลาที่โปรแกรมทำงานอยู่
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 การตั้งชื่อตัวแปร</h2>
      <p className="mt-2">
        การตั้งชื่อตัวแปรควรใช้ชื่อที่สื่อความหมาย และเป็นไปตามกฎของภาษาโปรแกรม เช่น ห้ามขึ้นต้นด้วยตัวเลข และห้ามใช้คำสงวน (Reserved Keywords)
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 overflow-x-auto text-sm">
{`# ✅ ชื่อตัวแปรที่ถูกต้อง
name = "Alice"
age = 25

# ❌ ตัวอย่างที่ผิด
2name = "Bob"  # ขึ้นต้นด้วยตัวเลข
if = 10        # ใช้คำสงวนของภาษา`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 การกำหนดค่าให้กับตัวแปร</h2>
      <p className="mt-2">
        เราสามารถกำหนดค่าลงในตัวแปรได้โดยใช้เครื่องหมาย <code>=</code> เช่น:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 overflow-x-auto text-sm">
{`message = "Hello, world!"
number = 100
is_active = True`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 ตัวแปรสามารถเปลี่ยนค่าได้ (Dynamic Typing)</h2>
      <p className="mt-2">
        Python เป็นภาษาที่ใช้ระบบ Dynamic Typing ซึ่งหมายความว่า ตัวแปรสามารถเปลี่ยนประเภทของข้อมูลที่เก็บได้ระหว่างที่โปรแกรมทำงาน
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 overflow-x-auto text-sm">
{`x = 10       # เป็น int
x = "Ten"    # เปลี่ยนเป็น string
x = True     # เปลี่ยนเป็น boolean`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 การตรวจสอบประเภทของตัวแปร</h2>
      <p className="mt-2">สามารถใช้ฟังก์ชัน <code>type()</code> เพื่อตรวจสอบชนิดข้อมูลของตัวแปรได้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 overflow-x-auto text-sm">
{`x = 42
print(type(x))  # Output: <class 'int'>

x = "Hello"
print(type(x))  # Output: <class 'str'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 การประกาศหลายตัวแปรในบรรทัดเดียว</h2>
      <p className="mt-2">สามารถประกาศตัวแปรหลายตัวได้พร้อมกัน</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 overflow-x-auto text-sm">
{`a, b, c = 1, 2, 3
name, age = "John", 30`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">💡 สรุป</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>ตัวแปรใช้สำหรับเก็บข้อมูลที่เราต้องการใช้ซ้ำในโปรแกรม</li>
        <li>ตั้งชื่อตัวแปรให้อ่านง่ายและสื่อความหมาย</li>
        <li>สามารถเปลี่ยนชนิดของค่าที่เก็บได้ในภาษา Python</li>
        <li>ใช้ <code>type()</code> เพื่อตรวจสอบชนิดข้อมูล</li>
      </ul>
    </div>
  );
};

export default UnderstandingVariables;
