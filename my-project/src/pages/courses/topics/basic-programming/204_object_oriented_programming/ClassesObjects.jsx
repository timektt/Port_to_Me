import React from "react";

const ClassesObjects = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">🧱 Classes & Objects ใน Python</h1>

      <p className="text-lg mb-4">
        ในแนวคิดของ Object-Oriented Programming (OOP) คลาส (Class) และอ็อบเจกต์ (Object) คือหัวใจสำคัญที่ช่วยให้เราสามารถจัดกลุ่มข้อมูลและพฤติกรรมไว้ในหน่วยเดียว ซึ่งส่งผลให้โปรแกรมมีความยืดหยุ่นและสามารถนำกลับมาใช้ซ้ำได้ง่ายขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">1. Class คืออะไร?</h2>
      <p>
        Class คือแม่แบบ (blueprint) สำหรับการสร้างอ็อบเจกต์ ซึ่งภายใน class จะประกอบด้วยฟิลด์ (attributes) และเมธอด (methods) ที่บอกว่าอ็อบเจกต์นั้นควรมีอะไรบ้างและสามารถทำอะไรได้บ้าง
      </p>

      <pre className="bg-gray-800 text-white p-4 rounded-md mt-4 overflow-x-auto text-sm">
{`class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"สวัสดี ฉันชื่อ {self.name} อายุ {self.age} ปี")`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">2. การสร้าง Object</h2>
      <p>
        Object คืออินสแตนซ์ของ class ซึ่งจะใช้คำสั่งเรียก class เพื่อสร้าง object
      </p>

      <pre className="bg-gray-800 text-white p-4 rounded-md mt-4 overflow-x-auto text-sm">
{`person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

person1.introduce()  # สวัสดี ฉันชื่อ Alice อายุ 25 ปี
person2.introduce()  # สวัสดี ฉันชื่อ Bob อายุ 30 ปี`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">3. Constructor (__init__)</h2>
      <p>
        ฟังก์ชัน <code>__init__</code> เป็น constructor ที่จะถูกเรียกโดยอัตโนมัติเมื่อสร้าง object ใหม่ เพื่อใช้กำหนดค่าตั้งต้นของ object
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">4. Attribute และ Method</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Attribute:</strong> ตัวแปรที่อยู่ภายใน object เช่น <code>self.name</code></li>
        <li><strong>Method:</strong> ฟังก์ชันที่อยู่ภายใน class เช่น <code>introduce()</code></li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">5. การเข้าถึง Attribute และ Method</h2>
      <p>ใช้เครื่องหมายจุด <code>.</code> เช่น <code>person1.name</code> หรือ <code>person1.introduce()</code></p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">6. ตัวอย่างเพิ่มเติม</h2>
      <p className="mt-2">เพิ่มเมธอดที่คำนวณปีเกิดจากอายุ:</p>

      <pre className="bg-gray-800 text-white p-4 rounded-md mt-4 overflow-x-auto text-sm">
{`class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def year_of_birth(self, current_year):
        return current_year - self.age

p = Person("Sara", 20)
print(p.year_of_birth(2025))  # Output: 2005`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">สรุป</h2>
      <p>
        การใช้ Class และ Object ช่วยให้โค้ดมีโครงสร้างที่ดีขึ้นและสามารถขยายฟีเจอร์ได้ง่าย เหมาะสำหรับการพัฒนาโปรแกรมขนาดใหญ่หรือซับซ้อน
      </p>
    </div>
  );
};

export default ClassesObjects;
