import React from "react";

const SetupEnvironment = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">🧰 การตั้งค่าโปรแกรมสำหรับการเขียนโค้ด</h1>

      <p className="mt-4 text-gray-700 dark:text-gray-300">
        การเริ่มต้นเขียนโปรแกรมจำเป็นต้องมีเครื่องมือและสภาพแวดล้อมที่เหมาะสม เพื่อให้เราสามารถพัฒนาและทดสอบโค้ดได้อย่างมีประสิทธิภาพ
      </p>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">1. เลือกภาษาโปรแกรมที่ต้องการ</h2>
      <p className="mt-2">
        ก่อนอื่นควรเลือกภาษาที่จะเริ่มเรียน เช่น Python, JavaScript, Java หรือ C++ แล้วจึงไปโหลดเครื่องมือให้เหมาะสมกับภาษานั้น
      </p>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">2. ติดตั้ง Code Editor หรือ IDE</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>💻 <strong>VS Code</strong>: ฟรี, เบา, รองรับหลายภาษา</li>
        <li>🧠 <strong>PyCharm</strong>: เหมาะสำหรับ Python</li>
        <li>🧱 <strong>IntelliJ IDEA</strong>: สำหรับภาษา Java และ Kotlin</li>
      </ul>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">3. ติดตั้ง Interpreter หรือ Compiler</h2>
      <p className="mt-2">หากคุณใช้ Python ให้ติดตั้ง Python Interpreter จาก <a href="https://python.org" target="_blank" className="text-blue-500 underline">python.org</a></p>
      <p>หากคุณใช้ C++ หรือ Java อาจต้องติดตั้ง Compiler เช่น GCC หรือ JDK</p>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">4. ทดลองรันโค้ดแรกของคุณ</h2>
      <p className="mt-2">สร้างไฟล์ใหม่ เช่น <code className="bg-gray-200 px-1 rounded">hello.py</code> แล้วพิมพ์:</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2 text-sm">
        <pre>
{`print("Hello, world!")`}
        </pre>
      </div>
      <p className="mt-2">จากนั้นเปิด Terminal แล้วรันคำสั่ง:</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2 text-sm">
        <pre>
{`python hello.py`}
        </pre>
      </div>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">5. เพิ่มความสามารถด้วย Extensions</h2>
      <p className="mt-2">
        Editor อย่าง VS Code สามารถเพิ่ม Extension เช่น Python, Prettier, ESLint เพื่อช่วยให้โค้ดของคุณดูดีและทำงานได้แม่นยำมากขึ้น
      </p>

      <h2 className="text-xl sm:text-2xl font-semibold mt-6">6. เคล็ดลับเพิ่มเติม</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>💾 ตั้งค่าบันทึกอัตโนมัติ (Auto Save)</li>
        <li>🌗 ใช้โหมดมืดช่วยถนอมสายตา</li>
        <li>🧪 ติดตั้ง Extension สำหรับรันโค้ดได้ภายใน Editor</li>
      </ul>

      <div className="mt-8 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-200 rounded-lg">
        ✅ หลังจากตั้งค่าทั้งหมดแล้ว คุณก็พร้อมสำหรับการเริ่มต้นเรียนรู้การเขียนโปรแกรมอย่างเต็มรูปแบบ!
      </div>
    </div>
  );
};

export default SetupEnvironment;
