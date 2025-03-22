import React from "react";

const PrimitiveDataTypes = () => {
  return (
    <div className="max-w-3xl mx-auto p-4 sm:p-6">
      <h1 className="text-2xl sm:text-3xl font-bold">ประเภทข้อมูลพื้นฐาน (Primitive Data Types)</h1>
      <p className="mt-4">
        ในการเขียนโปรแกรม การเลือกใช้ประเภทของข้อมูล (Data Type) ให้เหมาะสมเป็นสิ่งสำคัญ
        เพราะจะมีผลต่อประสิทธิภาพของโปรแกรมและการประมวลผลข้อมูลอย่างถูกต้อง
      </p>

      <h2 className="text-xl font-semibold mt-6">1. Integer (int)</h2>
      <p className="mt-2">เก็บค่าตัวเลขจำนวนเต็ม เช่น 0, 1, -10, 1000</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        {`x = 42
print(type(x))  # <class 'int'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. Floating Point (float)</h2>
      <p className="mt-2">ใช้เก็บค่าตัวเลขที่มีจุดทศนิยม เช่น 3.14, -0.5</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        {`pi = 3.14159
print(type(pi))  # <class 'float'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. String (str)</h2>
      <p className="mt-2">ใช้เก็บข้อความ เช่น ชื่อ, ที่อยู่, ประโยค</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        {`name = "Alice"
print(type(name))  # <class 'str'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. Boolean (bool)</h2>
      <p className="mt-2">ใช้เก็บค่าความจริง เช่น True หรือ False</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        {`is_active = True
print(type(is_active))  # <class 'bool'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">5. NoneType</h2>
      <p className="mt-2">ใช้สำหรับค่าที่ไม่มีการกำหนด เช่น การรีเซตตัวแปรหรือรอค่าจากการประมวลผล</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        {`result = None
print(type(result))  # <class 'NoneType'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">สรุป</h2>
      <ul className="list-disc ml-6 mt-2">
        <li><strong>int</strong>: ใช้เมื่อทำงานกับค่าตัวเลขจำนวนเต็ม</li>
        <li><strong>float</strong>: ใช้สำหรับค่าที่มีทศนิยม</li>
        <li><strong>str</strong>: ใช้เก็บข้อความหรือข้อมูลที่เป็นอักขระ</li>
        <li><strong>bool</strong>: ใช้ในเงื่อนไขทางตรรกศาสตร์</li>
        <li><strong>None</strong>: ใช้เมื่อยังไม่มีค่าหรือค่าที่ไม่มีความหมาย</li>
      </ul>
    </div>
  );
};

export default PrimitiveDataTypes;
