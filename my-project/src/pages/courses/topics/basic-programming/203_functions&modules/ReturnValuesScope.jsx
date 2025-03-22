import React from "react";

const ReturnValuesScope = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center sm:text-left">
        การคืนค่าและขอบเขตของตัวแปรในฟังก์ชัน (Return Values & Scope)
      </h1>

      <p className="mt-2">
        ใน Python ฟังก์ชันสามารถส่งค่ากลับได้โดยใช้คำสั่ง <code>return</code> ซึ่งค่าที่ส่งกลับ
        สามารถนำไปใช้ต่อในโปรแกรมหรือเก็บไว้ในตัวแปรได้ นอกจากนี้ การเข้าใจขอบเขตของตัวแปร
        (Scope) ก็สำคัญไม่แพ้กัน เพราะมันมีผลต่อการเข้าถึงและการทำงานของตัวแปรภายในโปรแกรม
      </p>

      <h2 className="text-xl font-semibold mt-6">1. การคืนค่า (Return Values)</h2>
      <p className="mt-2">
        คำสั่ง <code>return</code> จะสิ้นสุดการทำงานของฟังก์ชันและคืนค่ากลับไปยังผู้เรียก
        ฟังก์ชันสามารถส่งคืนค่าเดียวหรือหลายค่าในรูปแบบ tuple ได้
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`def add(a, b):
    return a + b

result = add(3, 5)
print("ผลรวม:", result)  # Output: ผลรวม: 8`}
      </pre>

      <h3 className="text-lg font-medium mt-6">✅ คืนค่าหลายค่า</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`def calculate(a, b):
    return a + b, a * b

sum_val, product = calculate(4, 5)
print("ผลรวม:", sum_val)
print("ผลคูณ:", product)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. ขอบเขตของตัวแปร (Variable Scope)</h2>
      <p className="mt-2">
        ตัวแปรใน Python มีขอบเขตที่กำหนดการมองเห็นและการเข้าถึง เช่น ตัวแปรภายในฟังก์ชันจะไม่สามารถเข้าถึงจากภายนอกได้
        ซึ่งเรียกว่า <strong>Local Scope</strong> ในขณะที่ตัวแปรที่ประกาศภายนอกสามารถใช้ได้ทั่วโปรแกรม เรียกว่า <strong>Global Scope</strong>
      </p>

      <h3 className="text-lg font-medium mt-4">📌 ตัวอย่าง: Local & Global Scope</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`x = 10  # global scope

def show():
    x = 5  # local scope
    print("ภายในฟังก์ชัน x =", x)

show()
print("ภายนอกฟังก์ชัน x =", x)`}
      </pre>

      <h3 className="text-lg font-medium mt-6">🔒 Global Keyword</h3>
      <p className="mt-2">
        หากต้องการเปลี่ยนค่าตัวแปร global จากภายในฟังก์ชัน ต้องใช้คำสั่ง <code>global</code>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`count = 0

def increment():
    global count
    count += 1

increment()
print(count)  # Output: 1`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <ul className="list-disc ml-6 mt-2 text-sm sm:text-base">
        <li><code>return</code> ใช้ส่งค่ากลับจากฟังก์ชัน</li>
        <li>สามารถคืนค่าได้หลายค่าโดยใช้ tuple</li>
        <li>เข้าใจ Local และ Global Scope สำคัญมากในการเขียนโปรแกรมให้ถูกต้อง</li>
      </ul>
    </div>
  );
};

export default ReturnValuesScope;
