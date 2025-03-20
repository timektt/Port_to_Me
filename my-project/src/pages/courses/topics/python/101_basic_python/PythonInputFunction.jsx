import React from "react";

const PythonInputFunction = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">การรับข้อมูล (Input) และฟังก์ชัน (Functions) ใน Python</h1>
      <p>
        ฟังก์ชันช่วยให้เราสามารถเขียนโค้ดที่สามารถนำกลับมาใช้ซ้ำได้ ซึ่งช่วยให้โค้ดมีความเป็นระเบียบและง่ายต่อการจัดการ
      </p>
      
      <h2 className="text-2xl font-semibold mt-6">1. การใช้ input() รับค่าจากผู้ใช้</h2>
      <p>
        Python มีฟังก์ชัน `input()` ที่ช่วยให้สามารถรับค่าจากผู้ใช้ผ่านทางคีย์บอร์ดได้
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`name = input("กรุณาป้อนชื่อของคุณ: ")
print(f"สวัสดี {name}!")`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">2. ฟังก์ชันใน Python</h2>
      <p>
        ฟังก์ชันใน Python คือชุดคำสั่งที่สามารถเรียกใช้ซ้ำได้โดยไม่ต้องเขียนโค้ดเดิมซ้ำ ๆ
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`def greet(name):
    print(f"Hello, {name}")

greet("Alice")`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">3. ฟังก์ชันที่มีค่าเริ่มต้น (Default Arguments)</h2>
      <p>
        เราสามารถกำหนดค่าเริ่มต้นให้กับพารามิเตอร์ของฟังก์ชันได้
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`def greet(name="ผู้ใช้"):
    print(f"สวัสดี, {name}!")

greet()  # Output: สวัสดี, ผู้ใช้!
greet("บ็อบ")  # Output: สวัสดี, บ็อบ!`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">4. ฟังก์ชันที่ส่งค่ากลับ (Return Values)</h2>
      <p>
        ฟังก์ชันสามารถส่งค่ากลับไปให้กับส่วนอื่นของโปรแกรมได้โดยใช้คำสั่ง `return`
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`def add(a, b):
    return a + b

result = add(3, 5)
print("ผลรวมคือ:", result)  # Output: ผลรวมคือ: 8`}
      </pre>
    </div>
  );
};

export default PythonInputFunction;