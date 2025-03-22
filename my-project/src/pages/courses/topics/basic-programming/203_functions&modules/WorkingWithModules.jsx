import React from "react";

const WorkingWithModules = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📦 การใช้งานโมดูลใน Python</h1>

      <p className="mt-4">
        ใน Python โมดูล (Module) คือไฟล์ที่ประกอบด้วยโค้ด Python ซึ่งสามารถนำกลับมาใช้ซ้ำได้
        ช่วยให้เราสามารถแยกโค้ดเป็นส่วน ๆ เพื่อให้ง่ายต่อการจัดการและดูแล
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. การนำเข้าโมดูล (Importing Modules)</h2>
      <p className="mt-2">สามารถใช้คำสั่ง <code>import</code> เพื่อนำเข้าโมดูลมาตรฐานที่มีอยู่แล้ว เช่น:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
{`import math
print(math.sqrt(16))  # Output: 4.0`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. การนำเข้าแบบเจาะจง</h2>
      <p className="mt-2">สามารถเลือกนำเข้าเฉพาะฟังก์ชันหรือคลาสจากโมดูล:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
{`from math import pi, sqrt
print(pi)
print(sqrt(25))`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. การใช้ Alias</h2>
      <p className="mt-2">เราสามารถตั้งชื่อย่อให้กับโมดูลเพื่อให้เขียนง่ายขึ้น:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
{`import numpy as np
arr = np.array([1, 2, 3])
print(arr)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">4. การสร้างโมดูลของเราเอง</h2>
      <p className="mt-2">สามารถสร้างไฟล์ Python (.py) แล้ว import มาใช้ได้ เช่น:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
{`# ไฟล์ mymodule.py
def say_hello(name):
    print(f"Hello, {name}!")

# ไฟล์ main.py
import mymodule
mymodule.say_hello("Time")`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">5. โฟลเดอร์ที่มี __init__.py จะกลายเป็นแพ็กเกจ</h2>
      <p className="mt-2">
        หากต้องการรวมหลายโมดูลเข้าด้วยกัน ให้สร้างโฟลเดอร์พร้อมไฟล์ <code>__init__.py</code>
        แล้วสามารถ import โมดูลจากโฟลเดอร์นั้นได้เหมือนเป็นแพ็กเกจ
      </p>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-300 rounded-lg">
        💡 <strong>Tip:</strong> โมดูลช่วยให้โค้ดของคุณดูดี มีโครงสร้าง และนำกลับมาใช้ซ้ำได้ง่ายขึ้นในโปรเจกต์ขนาดใหญ่
      </div>
    </div>
  );
};

export default WorkingWithModules;
