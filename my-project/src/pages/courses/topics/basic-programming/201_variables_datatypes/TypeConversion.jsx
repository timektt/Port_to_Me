import React from "react";

const TypeConversion = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">🌀 การแปลงชนิดข้อมูล (Type Conversion & Casting)</h1>

      <p className="mb-4">
        ในภาษา Python เราสามารถแปลงค่าจากชนิดข้อมูลหนึ่งไปยังอีกชนิดหนึ่งได้โดยใช้ฟังก์ชันสำหรับการแปลงข้อมูล เช่น <code>int()</code>, <code>float()</code>, <code>str()</code> เป็นต้น
      </p>

      <h2 className="text-xl font-semibold mt-6">1. การแปลงแบบอัตโนมัติ (Implicit Type Conversion)</h2>
      <p className="mt-2">
        Python จะทำการแปลงชนิดข้อมูลให้โดยอัตโนมัติเมื่อจำเป็น เช่น การรวมค่าจาก <code>int</code> กับ <code>float</code> จะได้ผลลัพธ์เป็น <code>float</code>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-3 overflow-x-auto text-sm">
{`x = 5
y = 2.0
result = x + y
print(result)        # Output: 7.0
print(type(result))  # Output: <class 'float'>`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. การแปลงแบบกำหนดเอง (Explicit Type Conversion)</h2>
      <p className="mt-2">
        เป็นการแปลงโดยผู้เขียนโปรแกรม เช่น จาก string เป็น int โดยใช้ <code>int()</code>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-3 overflow-x-auto text-sm">
{`num_str = "100"
num_int = int(num_str)
print(num_int + 50)  # Output: 150`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. ฟังก์ชันที่นิยมใช้ในการแปลงข้อมูล</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><code>int(x)</code>: แปลงค่า x เป็นจำนวนเต็ม</li>
        <li><code>float(x)</code>: แปลงค่า x เป็นเลขทศนิยม</li>
        <li><code>str(x)</code>: แปลงค่า x เป็น string</li>
        <li><code>bool(x)</code>: แปลงค่า x เป็นค่าบูลีน</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">4. ตัวอย่างการแปลงจากชนิดต่าง ๆ</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-3 overflow-x-auto text-sm">
{`print(int(3.9))     # Output: 3
print(float("5.2"))  # Output: 5.2
print(str(10))       # Output: "10"
print(bool(0))       # Output: False
print(bool(""))      # Output: False
print(bool("Python")) # Output: True`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">5. สิ่งที่ควรระวังในการแปลง</h2>
      <p className="mt-2">
        การแปลงที่ไม่ถูกต้องจะทำให้เกิด Error เช่น แปลง string ที่ไม่ใช่ตัวเลขด้วย <code>int()</code>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-3 overflow-x-auto text-sm">
{`s = "abc"
num = int(s)  # ❌ จะเกิด ValueError เพราะ "abc" ไม่ใช่ตัวเลข`}
      </pre>

      <div className="mt-6 p-4 bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 rounded-lg">
        💡 <strong>สรุป:</strong> การเข้าใจการแปลงชนิดข้อมูลช่วยให้เขียนโปรแกรมได้แม่นยำ ปลอดภัยจากข้อผิดพลาด และลดบั๊กที่เกิดจากชนิดข้อมูลไม่ตรงกัน
      </div>
    </div>
  );
};

export default TypeConversion;
