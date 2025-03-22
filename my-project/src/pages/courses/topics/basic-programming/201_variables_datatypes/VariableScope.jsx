import React from "react";

const VariableScope = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">📌 ขอบเขตของตัวแปร (Variable Scope)</h1>

      <p className="mb-4">
        ขอบเขตของตัวแปร (Scope) คือบริเวณในโปรแกรมที่สามารถเข้าถึงตัวแปรได้อย่างถูกต้อง การเข้าใจ Scope ช่วยให้คุณเขียนโปรแกรมได้อย่างปลอดภัยและลดปัญหาเกี่ยวกับตัวแปรซ้อนทับกัน (Variable Shadowing)
      </p>

      <h2 className="text-xl font-semibold mt-6">1. Local Scope</h2>
      <p className="mt-2">
        ตัวแปรที่ถูกประกาศภายในฟังก์ชันจะมีขอบเขตเป็นภายในฟังก์ชันเท่านั้น ไม่สามารถเข้าถึงได้จากภายนอก
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-3 overflow-auto text-sm">
{`def greet():
    message = "สวัสดี!"
    print(message)

greet()
# print(message)  ❌ Error: message is not defined`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. Global Scope</h2>
      <p className="mt-2">
        ตัวแปรที่ถูกประกาศนอกฟังก์ชัน สามารถถูกเข้าถึงได้จากภายในฟังก์ชัน หากไม่มีการสร้างตัวแปรชื่อเดียวกันซ้อนทับ
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-3 overflow-auto text-sm">
{`message = "Hello from outside!"

def show():
    print(message)

show()  # ✅ สามารถเข้าถึง message ได้`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. การใช้ global keyword</h2>
      <p className="mt-2">
        หากต้องการเปลี่ยนค่าตัวแปร global จากภายในฟังก์ชัน ต้องใช้คำสั่ง <code>global</code>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-3 overflow-auto text-sm">
{`count = 0

def increment():
    global count
    count += 1

increment()
print(count)  # Output: 1`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. Enclosing Scope (Nested Functions)</h2>
      <p className="mt-2">
        ฟังก์ชันซ้อนสามารถเข้าถึงตัวแปรจากฟังก์ชันภายนอกได้ ซึ่งเป็นหนึ่งในหลักของ <strong>Closure</strong>
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md mt-3 overflow-auto text-sm">
{`def outer():
    msg = "จาก outer"

    def inner():
        print(msg)

    inner()

outer()`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">5. สรุป Scope ทั้งหมด</h2>
      <ul className="list-disc ml-6 mt-2">
        <li><strong>Local:</strong> ภายในฟังก์ชัน</li>
        <li><strong>Global:</strong> ภายนอกฟังก์ชัน</li>
        <li><strong>Enclosing:</strong> ฟังก์ชันซ้อน</li>
        <li><strong>Built-in:</strong> คำสั่งหรือฟังก์ชันที่มาจาก Python โดยตรง เช่น <code>print()</code></li>
      </ul>

      <div className="mt-6 p-4 bg-blue-100 text-blue-900 rounded-md dark:bg-blue-900 dark:text-blue-100">
        💡 <strong>Tip:</strong> ควรใช้ตัวแปรให้เหมาะสมกับ Scope เพื่อลดบั๊ก และทำให้โค้ดอ่านง่ายขึ้น
      </div>
    </div>
  );
};

export default VariableScope;
