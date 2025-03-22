import React from "react";

const AbstractionInterfaces = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">📦 Abstraction & Interfaces</h1>

      <p className="mb-4">
        <strong>Abstraction (นามธรรม)</strong> คือแนวคิดในการซ่อนรายละเอียดที่ไม่จำเป็น
        และแสดงเฉพาะสิ่งที่จำเป็นต่อผู้ใช้งาน เป็นหนึ่งในหลักการพื้นฐานของ OOP ที่ช่วยให้โปรแกรมมีความยืดหยุ่น และง่ายต่อการดูแล
      </p>

      <h2 className="text-2xl font-semibold mt-6">💡 ทำไมต้องใช้ Abstraction?</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li>ลดความซับซ้อนของระบบ</li>
        <li>ซ่อนการทำงานภายในไว้จากผู้ใช้</li>
        <li>สามารถเปลี่ยนแปลงการทำงานภายในได้โดยไม่กระทบกับส่วนอื่น</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Abstraction (Python)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
{`from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

c = Circle(5)
print(c.area())  # Output: 78.5`}
      </pre>

      <p className="mt-4">
        ในตัวอย่างนี้ `Shape` เป็นคลาสนามธรรม (abstract class) ที่มีเมธอด `area()` ซึ่งจะต้องถูกกำหนด (implement) ในคลาสลูก (`Circle`)
      </p>

      <h2 className="text-2xl font-semibold mt-6">🔌 Interfaces คืออะไร?</h2>
      <p className="mt-2">
        <strong>Interface</strong> เป็นเหมือนสัญญาหรือข้อตกลงที่กำหนดว่าคลาสใดที่ใช้ Interface นี้
        จะต้องมีเมธอดตามที่ Interface กำหนดไว้ Interface ช่วยให้เกิดการทำงานร่วมกันโดยไม่สนใจรายละเอียดภายใน
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Interface (Java)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
{`interface Animal {
    void makeSound();
}

class Dog implements Animal {
    public void makeSound() {
        System.out.println("Woof!");
    }
}

public class Main {
    public static void main(String[] args) {
        Dog d = new Dog();
        d.makeSound();  // Output: Woof!
    }
}`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">✅ ความแตกต่างระหว่าง Abstraction และ Interface</h2>
      <table className="w-full mt-4 table-auto border text-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="border px-2 py-1">หัวข้อ</th>
            <th className="border px-2 py-1">Abstraction</th>
            <th className="border px-2 py-1">Interface</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-2 py-1">ใช้เพื่อ</td>
            <td className="border px-2 py-1">ซ่อนรายละเอียดการทำงาน</td>
            <td className="border px-2 py-1">บังคับให้ implement เมธอดทั้งหมด</td>
          </tr>
          <tr>
            <td className="border px-2 py-1">สามารถมี code implementation ได้หรือไม่?</td>
            <td className="border px-2 py-1">ได้</td>
            <td className="border px-2 py-1">ไม่ได้ (ในบางภาษา)</td>
          </tr>
          <tr>
            <td className="border px-2 py-1">จำนวนเมธอดที่กำหนด</td>
            <td className="border px-2 py-1">ไม่จำเป็นต้องครบทุกเมธอด</td>
            <td className="border px-2 py-1">ต้องครบตามที่ Interface กำหนด</td>
          </tr>
        </tbody>
      </table>

      <div className="mt-6 p-4 border-l-4 border-blue-400 bg-blue-50 dark:bg-blue-900 dark:text-blue-200">
        ✨ <strong>สรุป:</strong> ทั้ง Abstraction และ Interface ต่างช่วยให้โค้ดมีโครงสร้างที่ชัดเจน
        และรองรับการขยายในอนาคตได้อย่างยืดหยุ่น
      </div>
    </div>
  );
};

export default AbstractionInterfaces;
