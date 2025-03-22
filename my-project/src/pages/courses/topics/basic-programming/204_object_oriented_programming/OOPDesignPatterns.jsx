import React from "react";

const AbstractionInterfaces = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">
        🧩 Abstraction & Interfaces ใน OOP
      </h1>

      <p className="mt-2 text-lg">
        Abstraction (นามธรรม) และ Interface (อินเทอร์เฟซ) เป็นแนวคิดสำคัญใน
        Object-Oriented Programming ที่ช่วยให้เราออกแบบซอฟต์แวร์ให้มีโครงสร้างที่ดี
        แยกส่วนรายละเอียดการทำงานออกจากสิ่งที่ผู้ใช้ต้องรู้
      </p>

      <h2 className="text-xl font-semibold mt-6">1. Abstraction คืออะไร?</h2>
      <p className="mt-2">
        Abstraction คือการซ่อนรายละเอียดที่ซับซ้อน และแสดงเฉพาะสิ่งที่จำเป็นให้กับผู้ใช้
        ตัวอย่างเช่น เมื่อเราขับรถ เราไม่จำเป็นต้องรู้ว่าภายในเครื่องยนต์ทำงานอย่างไร —
        เราเพียงแค่กดคันเร่งและเบรก
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 text-sm">
{`from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        print("Woof!")

d = Dog()
d.make_sound()  # Output: Woof!`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">2. Interface คืออะไร?</h2>
      <p className="mt-2">
        Interface ใน Python จะใช้ได้ผ่านการใช้ abstract base class เช่นเดียวกับ Abstraction
        โดยเรากำหนด method ที่ต้องถูก override โดย subclass
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 text-sm">
{`class Printable(ABC):
    @abstractmethod
    def print_info(self):
        pass

class Book(Printable):
    def __init__(self, title):
        self.title = title

    def print_info(self):
        print(f"หนังสือชื่อ: {self.title}")

b = Book("Python 101")
b.print_info()  # Output: หนังสือชื่อ: Python 101`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. ประโยชน์ของ Abstraction & Interface</h2>
      <ul className="list-disc ml-6 mt-2 text-base">
        <li>ช่วยลดความซับซ้อนของระบบ</li>
        <li>ช่วยในการออกแบบระบบให้ยืดหยุ่นและสามารถขยายได้</li>
        <li>ส่งเสริมหลักการของ Encapsulation และความเป็นอิสระระหว่างโมดูล</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">4. เปรียบเทียบ</h2>
      <table className="w-full mt-4 text-left border text-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="p-2 border">Aspect</th>
            <th className="p-2 border">Abstraction</th>
            <th className="p-2 border">Interface</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="p-2 border">การใช้งาน</td>
            <td className="p-2 border">ซ่อนรายละเอียด</td>
            <td className="p-2 border">กำหนดรูปแบบ</td>
          </tr>
          <tr>
            <td className="p-2 border">มี method implementation?</td>
            <td className="p-2 border">ได้</td>
            <td className="p-2 border">ไม่ได้</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default AbstractionInterfaces;
