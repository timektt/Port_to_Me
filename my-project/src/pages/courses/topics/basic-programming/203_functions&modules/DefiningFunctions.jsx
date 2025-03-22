import React from "react";

const DefiningFunctions = () => {
  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-6">
      <h1 className="text-2xl sm:text-3xl font-bold">🧩 การสร้างฟังก์ชัน (Defining Functions)</h1>
      <p className="mt-4 text-base sm:text-lg">
        ฟังก์ชัน (Function) เป็นหน่วยของโค้ดที่ออกแบบมาเพื่อทำงานเฉพาะอย่าง โดยสามารถนำมาเรียกใช้ซ้ำได้ในหลาย ๆ ส่วนของโปรแกรม ซึ่งช่วยให้โค้ดอ่านง่าย เป็นระเบียบ และดูแลรักษาได้ง่ายขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 การสร้างฟังก์ชันใน Python</h2>
      <p className="mt-2">ใช้คำสั่ง <code className="bg-gray-200 px-1 rounded">def</code> ตามด้วยชื่อฟังก์ชันและวงเล็บเปิด-ปิด</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 text-sm overflow-x-auto">
        {`def greet():
    print("สวัสดี!")

greet()  # เรียกใช้ฟังก์ชัน`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 การส่งค่าพารามิเตอร์เข้าไปในฟังก์ชัน</h2>
      <p className="mt-2">สามารถระบุค่าที่ต้องการส่งเข้าไปในฟังก์ชันได้ เช่น ชื่อ หรือ ตัวเลข</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 text-sm overflow-x-auto">
        {`def greet(name):
    print(f"สวัสดี {name}!")

greet("ภูผา")`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 การใช้ Default Parameters</h2>
      <p className="mt-2">กำหนดค่าเริ่มต้นให้พารามิเตอร์ได้ หากไม่มีการส่งค่ามา</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 text-sm overflow-x-auto">
        {`def greet(name="ผู้ใช้"):
    print(f"สวัสดี {name}!")

greet()         # สวัสดี ผู้ใช้!
greet("ไผ่")     # สวัสดี ไผ่!`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 การคืนค่าจากฟังก์ชัน (Return Values)</h2>
      <p className="mt-2">ใช้คำสั่ง <code className="bg-gray-200 px-1 rounded">return</code> เพื่อส่งค่ากลับ</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 text-sm overflow-x-auto">
        {`def square(x):
    return x * x

result = square(5)
print("ผลลัพธ์:", result)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔹 ความสำคัญของการใช้ฟังก์ชัน</h2>
      <ul className="list-disc ml-6 mt-2 text-base">
        <li>ช่วยให้โค้ดกระชับและอ่านง่าย</li>
        <li>ลดการเขียนโค้ดซ้ำ</li>
        <li>ง่ายต่อการดูแลและแก้ไข</li>
        <li>สามารถแบ่งปันและใช้ร่วมกันได้</li>
      </ul>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-200 rounded-lg">
        💡 <strong>Tips:</strong> ควรตั้งชื่อฟังก์ชันให้สื่อความหมาย เช่น <code className="bg-gray-200 px-1 rounded">calculate_total()</code> หรือ <code className="bg-gray-200 px-1 rounded">print_summary()</code>
      </div>
    </div>
  );
};

export default DefiningFunctions;
