
import React from "react";

const PolymorphismMethodOverriding = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📘 พอลิมอร์ฟิซึม และ Method Overriding</h1>

      <p className="mt-4 text-lg">
        พอลิมอร์ฟิซึม (Polymorphism) คือแนวคิดใน OOP ที่ทำให้วัตถุสามารถแสดงพฤติกรรมต่างกันได้แม้จะใช้เมธอดเดียวกัน ส่วน Method Overriding คือการเขียนเมธอดใหม่ในคลาสลูกที่มีชื่อเหมือนกับเมธอดในคลาสแม่ เพื่อเปลี่ยนพฤติกรรมให้เหมาะสมกับคลาสลูก
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. ความเข้าใจในพอลิมอร์ฟิซึม</h2>
      <p className="mt-2">
        พอลิมอร์ฟิซึมแบ่งออกได้เป็น 2 ประเภทหลัก ๆ คือ:
      </p>
      <ul className="list-disc ml-6 mt-2">
        <li>Compile-time Polymorphism (Method Overloading)</li>
        <li>Run-time Polymorphism (Method Overriding)</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">2. ตัวอย่างการใช้ Method Overriding</h2>
      <p className="mt-2">ดูตัวอย่างการเขียนเมธอดในคลาสลูกที่ override เมธอดจากคลาสแม่:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm">
{`class Animal:
    def speak(self):
        print("Animal is speaking")

class Dog(Animal):
    def speak(self):
        print("Dog is barking")

class Cat(Animal):
    def speak(self):
        print("Cat is meowing")

# ใช้งาน
animals = [Dog(), Cat()]
for a in animals:
    a.speak()`}
      </pre>
      <p className="mt-2">
        โค้ดข้างต้นใช้พอลิมอร์ฟิซึมในการเรียกเมธอด <code>speak()</code> ที่แม้จะเรียกผ่าน reference ของคลาสแม่ แต่ผลลัพธ์ที่ได้จะขึ้นกับ object จริงของคลาสลูก
      </p>

      <h2 className="text-2xl font-semibold mt-6">3. ข้อดีของการใช้ Polymorphism</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>ทำให้โค้ดอ่านง่ายและขยายได้สะดวก</li>
        <li>สามารถเขียนฟังก์ชันหรือเมธอดที่รองรับหลายชนิดของ object</li>
        <li>ช่วยเพิ่มความยืดหยุ่นในการพัฒนาโปรแกรม</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">4. เปรียบเทียบกับ Overloading</h2>
      <p className="mt-2">
        ใน Python ไม่รองรับ Method Overloading โดยตรง แต่สามารถจำลองได้ด้วยการใช้ default arguments หรือ *args
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm">
{`def greet(name=None):
    if name:
        print(f"Hello, {name}")
    else:
        print("Hello")

greet()
greet("Alice")`}
      </pre>

      <p className="mt-4 text-blue-600 dark:text-blue-300">
        ✅ สรุป: Polymorphism และ Method Overriding ช่วยให้สามารถเขียนโค้ดที่มีความยืดหยุ่นและรองรับการเปลี่ยนแปลงของพฤติกรรมตามชนิดของ object ได้อย่างมีประสิทธิภาพ
      </p>
    </div>
  );
};

export default PolymorphismMethodOverriding;
