import React from "react";

const PythonInputFunction = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">การรับข้อมูล (Input) และฟังก์ชัน (Functions) ใน Python</h1>
      <p className="mt-4">
        ฟังก์ชันช่วยให้เราสามารถเขียนโค้ดที่สามารถนำกลับมาใช้ซ้ำได้
        ซึ่งช่วยให้โค้ดมีความเป็นระเบียบและง่ายต่อการจัดการ
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. การใช้ input() รับค่าจากผู้ใช้</h2>
      <p>Python มีฟังก์ชัน <code>input()</code> ที่ช่วยให้สามารถรับค่าจากผู้ใช้ผ่านทางคีย์บอร์ดได้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`name = input("กรุณาป้อนชื่อของคุณ: ")
print(f"สวัสดี {name}!")`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. ฟังก์ชันใน Python</h2>
      <p>ฟังก์ชันคือชุดคำสั่งที่สามารถเรียกใช้ซ้ำได้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`def greet(name):
    print(f"Hello, {name}")

greet("Alice")`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. ฟังก์ชันที่มีค่าเริ่มต้น (Default Arguments)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`def greet(name="ผู้ใช้"):
    print(f"สวัสดี, {name}!")

greet()         # Output: สวัสดี, ผู้ใช้!
greet("บ็อบ")   # Output: สวัสดี, บ็อบ!`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">4. Keyword Arguments</h2>
      <p>เราสามารถส่ง argument โดยระบุชื่อ parameter ได้เลย</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`def introduce(name, age):
    print(f"{name} อายุ {age} ปี")

introduce(age=25, name="ซาร่า")`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">5. ฟังก์ชันที่ส่งค่ากลับ (Return Values)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`def add(a, b):
    return a + b

result = add(3, 5)
print("ผลรวมคือ:", result)  # Output: ผลรวมคือ: 8`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">6. Lambda Functions</h2>
      <p>ฟังก์ชันแบบย่อที่ใช้คำว่า <code>lambda</code></p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`square = lambda x: x * x
print(square(4))  # Output: 16`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">7. ส่งคืนค่าหลายค่า (Multiple Return Values)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
{`def calc(a, b):
    return a + b, a * b

sum_result, product = calc(2, 3)
print("ผลบวก:", sum_result)
print("ผลคูณ:", product)`}
      </pre>
    </div>
  );
};

export default PythonInputFunction;
