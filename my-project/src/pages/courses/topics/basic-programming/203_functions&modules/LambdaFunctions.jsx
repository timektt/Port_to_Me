import React from "react";

const LambdaFunctions = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">📌 Lambda Functions & Anonymous Functions</h1>

      <p className="text-lg mt-2">
        Lambda Function (หรือ Anonymous Function) คือฟังก์ชันแบบย่อที่ไม่มีชื่อ ใช้ในกรณีที่ไม่ต้องการสร้างฟังก์ชันแบบเต็มรูปแบบ เช่น ใช้ภายในฟังก์ชันอื่นหรือใช้กับเมธอดของคอลเลกชัน
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. โครงสร้างของ Lambda Function</h2>
      <p className="mt-2">รูปแบบทั่วไปของ lambda:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`lambda arguments: expression`}
      </pre>
      <p className="mt-2">ตัวอย่าง:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`square = lambda x: x * x
print(square(5))  # Output: 25`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. ใช้งานร่วมกับฟังก์ชัน built-in</h2>
      <p className="mt-2">เช่นใช้กับ <code>map()</code>, <code>filter()</code>, หรือ <code>sorted()</code></p>

      <h3 className="text-xl font-medium mt-4">ตัวอย่าง: ใช้กับ map()</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`nums = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, nums))
print(squared)  # Output: [1, 4, 9, 16]`}
      </pre>

      <h3 className="text-xl font-medium mt-4">ตัวอย่าง: ใช้กับ filter()</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`nums = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, nums))
print(evens)  # Output: [2, 4, 6]`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. ข้อจำกัดของ Lambda</h2>
      <ul className="list-disc ml-5 mt-2">
        <li>ใช้ได้เฉพาะนิพจน์เดียว (ไม่สามารถมีหลายบรรทัด)</li>
        <li>เหมาะกับงานเล็ก ๆ ที่ไม่ต้องสร้างฟังก์ชันใหม่ให้ยืดยาว</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">4. เปรียบเทียบกับฟังก์ชันทั่วไป</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`# ฟังก์ชันทั่วไป
def add(x, y):
    return x + y

# Lambda
add_lambda = lambda x, y: x + y`}
      </pre>
    </div>
  );
};

export default LambdaFunctions;
