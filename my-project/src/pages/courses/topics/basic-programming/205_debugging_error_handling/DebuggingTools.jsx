import React from "react";

const DebuggingTools = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">🛠️ การใช้เครื่องมือ Debugging</h1>

      <p className="mb-4">
        การ Debug คือกระบวนการตรวจสอบและแก้ไขข้อผิดพลาด (bugs) ในโปรแกรม ซึ่งเป็นขั้นตอนสำคัญในการพัฒนาซอฟต์แวร์ให้ทำงานได้ถูกต้องและมีประสิทธิภาพ
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. การใช้ print() ตรวจสอบค่าตัวแปร</h2>
      <p className="mt-2">
        การใช้ <code>print()</code> เป็นวิธีที่ง่ายและพื้นฐานที่สุดในการดูค่าของตัวแปรหรือ flow ของโปรแกรม
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`x = 5
y = 0
print("ก่อนการหาร")
print("x =", x)

result = x / y  # จะเกิด ZeroDivisionError
print("ผลลัพธ์ =", result)`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. การใช้ Debugger ใน IDE</h2>
      <p className="mt-2">
        IDE อย่าง VS Code, PyCharm หรือ Thonny มีเครื่องมือ Debug ที่ช่วยให้สามารถหยุดโปรแกรมชั่วคราว (breakpoint), ดูค่าตัวแปร, และตรวจสอบ flow ได้แบบ real-time
      </p>
      <ul className="list-disc ml-6 mt-2">
        <li>ตั้ง Breakpoint ที่บรรทัดที่ต้องการ</li>
        <li>รันโปรแกรมในโหมด Debug</li>
        <li>ดูค่าตัวแปร, Stack trace และ Step over</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">3. การใช้ฟังก์ชัน assert()</h2>
      <p className="mt-2">
        ฟังก์ชัน <code>assert</code> ใช้ตรวจสอบว่าค่าเป็นจริง (True) หรือไม่ หากไม่เป็นจริงจะหยุดโปรแกรมและแจ้ง error
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`def divide(a, b):
    assert b != 0, "b ห้ามเป็น 0"
    return a / b

divide(10, 0)  # AssertionError: b ห้ามเป็น 0`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6">4. การใช้ logging แทน print()</h2>
      <p className="mt-2">
        การใช้ <code>logging</code> ช่วยให้ควบคุมระดับความสำคัญของข้อความ เช่น info, warning, error และยังสามารถเก็บ log ไว้ในไฟล์ได้อีกด้วย
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`import logging

logging.basicConfig(level=logging.INFO)
x = 10
logging.info(f"ค่า x คือ {x}")`}</code>
      </pre>

      <p className="mt-6">
        ✅ การเลือกใช้เครื่องมือ Debug ที่เหมาะสม จะช่วยให้คุณแก้ไขข้อผิดพลาดได้เร็วและแม่นยำยิ่งขึ้น
      </p>
    </div>
  );
};

export default DebuggingTools;
