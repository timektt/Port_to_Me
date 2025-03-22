import React from "react";

const ConditionalStatements = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold text-center sm:text-left">
        🧠 คำสั่งเงื่อนไขในภาษา Python (Conditional Statements)
      </h1>

      <p className="mt-4">
        คำสั่งเงื่อนไข (Conditional Statements) ช่วยให้โปรแกรมสามารถตัดสินใจได้ตามเงื่อนไขที่กำหนด เช่น ถ้าเงื่อนไขเป็นจริงให้ทำสิ่งหนึ่ง ถ้าไม่ใช่ให้ทำอีกอย่างหนึ่ง โดยทั่วไปใช้คำสั่ง <code>if</code>, <code>elif</code> และ <code>else</code> ในภาษา Python
      </p>

      <h2 className="text-xl font-semibold mt-6">1. โครงสร้างพื้นฐานของ if-elif-else</h2>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded-md overflow-x-auto">
{`x = 10

if x > 0:
    print("x เป็นจำนวนบวก")
elif x == 0:
    print("x เป็นศูนย์")
else:
    print("x เป็นจำนวนลบ")`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. เงื่อนไขซ้อน (Nested if)</h2>
      <p className="mt-2">สามารถใช้ if ซ้อนกันได้ เพื่อสร้างลำดับการตรวจสอบหลายระดับ:</p>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded-md overflow-x-auto">
{`x = 15

if x > 0:
    if x < 20:
        print("x อยู่ระหว่าง 0 และ 20")
    else:
        print("x มากกว่าหรือเท่ากับ 20")`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. ตัวดำเนินการเปรียบเทียบและตรรกะ</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><code>&gt;</code>, <code>&lt;</code>, <code>&gt;=</code>, <code>&lt;=</code>, <code>==</code>, <code>!=</code></li>
        <li><code>and</code>, <code>or</code>, <code>not</code></li>
      </ul>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded-md overflow-x-auto mt-3">
{`age = 18
has_permission = True

if age >= 18 and has_permission:
    print("สามารถเข้าใช้งานได้")`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. คำสั่ง if แบบย่อ (Ternary Operator)</h2>
      <p className="mt-2">สามารถเขียน if-else แบบย่อในบรรทัดเดียวได้:</p>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded-md overflow-x-auto">
{`score = 75
result = "ผ่าน" if score >= 50 else "ไม่ผ่าน"
print(result)`}
      </pre>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-200 rounded-md">
        💡 <strong>สรุป:</strong> การใช้คำสั่งเงื่อนไขช่วยให้โปรแกรมของเรามีความยืดหยุ่นและฉลาดขึ้น สามารถตัดสินใจได้ตามสถานการณ์ต่าง ๆ
      </div>
    </div>
  );
};

export default ConditionalStatements;
