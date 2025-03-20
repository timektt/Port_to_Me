import React from "react";

const PythonVariables = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">ตัวแปรใน Python (Python Variables)</h1>
      <p>
        ตัวแปร (Variable) คือค่าที่ใช้ในการเก็บข้อมูลในหน่วยความจำ และสามารถนำไปใช้งานหรือเปลี่ยนแปลงค่าได้
      </p>
      
      <h2 className="text-2xl font-semibold mt-6">การประกาศตัวแปร</h2>
      <p>
        ใน Python เราไม่ต้องกำหนดชนิดของตัวแปรล่วงหน้า เพราะ Python เป็นภาษาที่มี Dynamic Typing
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`x = 5  # ตัวแปรเก็บค่าตัวเลข
name = "Alice"  # ตัวแปรเก็บข้อความ
is_active = True  # ตัวแปรเก็บค่าบูลีน`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">การเปลี่ยนค่าของตัวแปร</h2>
      <p>
        ตัวแปรใน Python สามารถเปลี่ยนค่าได้ง่ายโดยใช้เครื่องหมาย `=`
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`x = 10
x = "Hello Python"
print(x)  # Output: Hello Python`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">การใช้ตัวแปรหลายตัว</h2>
      <p>
        Python รองรับการกำหนดค่าหลายตัวแปรพร้อมกันได้
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`a, b, c = 1, 2, 3
print(a, b, c)  # Output: 1 2 3`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">ค่าคงที่ (Constants)</h2>
      <p>
        แม้ว่า Python จะไม่มีคำสั่งสำหรับสร้างค่าคงที่โดยตรง แต่สามารถใช้ตัวพิมพ์ใหญ่ทั้งหมดเพื่อแสดงค่าคงที่ได้
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`PI = 3.14159
GRAVITY = 9.8
print(PI, GRAVITY)`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">การตรวจสอบชนิดข้อมูลของตัวแปร</h2>
      <p>
        สามารถใช้ฟังก์ชัน `type()` เพื่อตรวจสอบชนิดของตัวแปร
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
        {`x = 42
y = "Hello"
print(type(x))  # Output: <class 'int'>
print(type(y))  # Output: <class 'str'>`}
      </pre>
    </div>
  );
};

export default PythonVariables;