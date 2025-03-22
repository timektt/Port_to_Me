import React from "react";

const CompilersInterpreters = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">คอมไพเลอร์ vs อินเทอร์พรีเตอร์</h1>
      <p className="mb-4">
        ในโลกของการเขียนโปรแกรม การแปลภาษาที่มนุษย์เขียนให้กลายเป็นภาษาที่เครื่องเข้าใจได้ เป็นขั้นตอนที่สำคัญมาก ซึ่งเราสามารถแบ่งเครื่องมือสำหรับแปลโค้ดออกเป็น 2 ประเภทหลัก คือ <strong>คอมไพเลอร์ (Compiler)</strong> และ <strong>อินเทอร์พรีเตอร์ (Interpreter)</strong>
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">1. คอมไพเลอร์ (Compiler)</h2>
      <p className="mb-2">
        คอมไพเลอร์จะทำการแปลโค้ดทั้งหมดในคราวเดียว และสร้างเป็นไฟล์ใหม่ที่สามารถนำไปรันได้ทันที เช่น ไฟล์ .exe บน Windows
      </p>
      <p className="mb-2">
        เมื่อมีการรันโปรแกรม จะไม่ต้องแปลใหม่อีก ทำให้ประสิทธิภาพในการรันสูง
      </p>
      <p className="mb-2">ตัวอย่างภาษาโปรแกรมที่ใช้ Compiler ได้แก่:</p>
      <ul className="list-disc ml-6 mb-4">
        <li>C</li>
        <li>C++</li>
        <li>Go</li>
        <li>Rust</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">2. อินเทอร์พรีเตอร์ (Interpreter)</h2>
      <p className="mb-2">
        อินเทอร์พรีเตอร์จะแปลโค้ดแบบบรรทัดต่อบรรทัดทุกครั้งที่โปรแกรมถูกรัน จึงเหมาะสำหรับการพัฒนาโปรแกรมที่ต้องการทดสอบบ่อย ๆ
      </p>
      <p className="mb-2">
        จุดเด่นคือความสะดวกในการดีบัก และสามารถดูผลลัพธ์ได้ทันที
      </p>
      <p className="mb-2">ตัวอย่างภาษาโปรแกรมที่ใช้ Interpreter ได้แก่:</p>
      <ul className="list-disc ml-6 mb-4">
        <li>Python</li>
        <li>JavaScript</li>
        <li>Ruby</li>
        <li>PHP</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">3. ความแตกต่างที่สำคัญ</h2>
      <table className="w-full text-left border-collapse mt-2">
        <thead>
          <tr>
            <th className="border-b p-2">หัวข้อ</th>
            <th className="border-b p-2">Compiler</th>
            <th className="border-b p-2">Interpreter</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border-b p-2">การแปลโค้ด</td>
            <td className="border-b p-2">แปลทั้งหมดก่อนรัน</td>
            <td className="border-b p-2">แปลทีละบรรทัด</td>
          </tr>
          <tr>
            <td className="border-b p-2">ความเร็วในการรัน</td>
            <td className="border-b p-2">เร็วกว่า</td>
            <td className="border-b p-2">ช้ากว่า</td>
          </tr>
          <tr>
            <td className="border-b p-2">ดีบัก</td>
            <td className="border-b p-2">ยากกว่า</td>
            <td className="border-b p-2">ง่ายกว่า</td>
          </tr>
        </tbody>
      </table>

      <p className="mt-4">
        ปัจจุบันมีภาษาโปรแกรมบางภาษา เช่น Java และ Python ที่ใช้เทคนิคแบบผสม (ทั้ง Compilation และ Interpretation) เพื่อให้ได้ประโยชน์ทั้งสองด้าน
      </p>
    </div>
  );
};

export default CompilersInterpreters;
