import React from "react";

const EncapsulationInheritance = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">🧱 Encapsulation & Inheritance</h1>

      <p className="mt-2">
        Encapsulation (การห่อหุ้ม) และ Inheritance (การสืบทอด) เป็นคุณสมบัติสำคัญของ OOP ที่ช่วยให้โค้ดมีโครงสร้างชัดเจนและนำกลับมาใช้ใหม่ได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. Encapsulation (การห่อหุ้มข้อมูล)</h2>
      <p className="mt-2">
        Encapsulation คือกระบวนการซ่อนรายละเอียดการทำงานของคลาส และให้เข้าถึงข้อมูลผ่านเมธอดเท่านั้น เพื่อความปลอดภัยและความเป็นระเบียบ
      </p>

      <h3 className="text-xl font-medium mt-4">📌 ตัวอย่าง Encapsulation:</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        {`class Person:
    def __init__(self, name):
        self.__name = name  # Private attribute

    def get_name(self):
        return self.__name

    def set_name(self, name):
        self.__name = name

p = Person("Alice")
print(p.get_name())
        `}
      </pre>

      <p className="mt-4">
        จากตัวอย่างด้านบน ตัวแปร <code>__name</code> ถูกกำหนดให้เป็น <strong>private</strong> และสามารถเข้าถึงได้ผ่านเมธอด get/set เท่านั้น
      </p>

      <h2 className="text-2xl font-semibold mt-6">2. Inheritance (การสืบทอดคุณสมบัติ)</h2>
      <p className="mt-2">
        Inheritance คือการสร้างคลาสใหม่ (Subclass) โดยรับคุณสมบัติจากคลาสแม่ (Superclass) เพื่อประหยัดเวลาและโค้ดซ้ำซ้อน
      </p>

      <h3 className="text-xl font-medium mt-4">📌 ตัวอย่าง Inheritance:</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        {`class Animal:
    def speak(self):
        print("สัตว์ส่งเสียง")

class Dog(Animal):
    def speak(self):
        print("หมาเห่า โฮ่ง โฮ่ง")

d = Dog()
d.speak()  # Output: หมาเห่า โฮ่ง โฮ่ง
        `}
      </pre>

      <p className="mt-4">
        คลาส <code>Dog</code> ได้รับเมธอด <code>speak</code> มาจาก <code>Animal</code> และสามารถ override เมธอดนั้นได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ข้อดีของการใช้ Encapsulation & Inheritance</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>เพิ่มความปลอดภัยของข้อมูล</li>
        <li>ควบคุมการเข้าถึงข้อมูลได้</li>
        <li>ลดการเขียนโค้ดซ้ำ</li>
        <li>เพิ่มความสามารถในการขยายระบบ</li>
      </ul>
    </div>
  );
};

export default EncapsulationInheritance;
