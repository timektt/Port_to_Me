import React from "react";

const PythonVariables = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">ตัวแปรใน Python (Python Variables)</h1>
      <p>
        ตัวแปร (Variable) คือค่าที่ใช้ในการเก็บข้อมูลในหน่วยความจำ และสามารถนำไปใช้งานหรือเปลี่ยนแปลงค่าได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. การประกาศตัวแปร</h2>
      <p>
        Python เป็นภาษาที่ไม่ต้องกำหนดชนิดของข้อมูลล่วงหน้า (Dynamic Typing) สามารถกำหนดค่าตัวแปรได้ทันที
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`x = 5            # ตัวแปรเก็บค่าตัวเลข
name = "Alice"    # ตัวแปรเก็บข้อความ
is_active = True  # ตัวแปรเก็บค่าบูลีน`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. การเปลี่ยนค่าของตัวแปร</h2>
      <p>
        ตัวแปรสามารถเปลี่ยนค่าหรือแม้กระทั่งเปลี่ยนชนิดข้อมูลได้ตลอดเวลา
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`x = 10
x = "Hello Python"
print(x)  # Output: Hello Python`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. การใช้ตัวแปรหลายตัว</h2>
      <p>
        สามารถกำหนดค่าหลายตัวแปรได้ในบรรทัดเดียว
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`a, b, c = 1, 2, 3
print(a, b, c)  # Output: 1 2 3`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">4. การตั้งชื่อตัวแปรที่ดี</h2>
      <p>
        ตัวแปรควรตั้งชื่อให้สื่อความหมาย เช่น `score`, `total_price` และไม่ใช้ชื่อที่ขัดกับคำสำรองของ Python เช่น `class`, `def`, `if`
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`total_price = 150
score = 95`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">5. ค่าคงที่ (Constants)</h2>
      <p>
        แม้ว่า Python จะไม่มีคำสั่งสำหรับค่าคงที่โดยตรง แต่สามารถใช้ชื่อพิมพ์ใหญ่ทั้งหมดเพื่อสื่อว่าเป็นค่าคงที่
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`PI = 3.14159
GRAVITY = 9.8`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">6. การตรวจสอบชนิดข้อมูลของตัวแปร</h2>
      <p>
        ใช้คำสั่ง `type()` เพื่อตรวจสอบชนิดข้อมูลของตัวแปร
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`x = 42
y = "Hello"
print(type(x))  # <class 'int'>
print(type(y))  # <class 'str'>`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">7. ตัวแปร Global และ Local</h2>
      <p>
        ตัวแปรที่ประกาศภายในฟังก์ชันเป็น Local แต่หากต้องการใช้งานตัวแปรจากภายนอกให้ใช้ `global`
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`x = 10

def show():
    global x
    x = x + 5

show()
print(x)  # Output: 15`}
      </pre>
    </div>
  );
};

export default PythonVariables;
