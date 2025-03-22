import React from "react";

const ComputerExecution = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">101-2: คอมพิวเตอร์ทำงานอย่างไรกับโค้ด</h1>

      <p className="mt-4 text-lg">
        การทำความเข้าใจว่าคอมพิวเตอร์ทำงานกับโค้ดอย่างไร เป็นพื้นฐานสำคัญของการเรียนเขียนโปรแกรม เพราะมันช่วยให้เราเข้าใจเบื้องหลังของสิ่งที่เกิดขึ้นเมื่อเราสั่งให้โปรแกรมทำงาน
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. โครงสร้างพื้นฐานของคอมพิวเตอร์</h2>
      <p className="mt-2">
        คอมพิวเตอร์ประกอบด้วย 4 ส่วนหลัก:
        <ul className="list-disc ml-6 mt-2">
          <li><strong>CPU (Central Processing Unit):</strong> สมองของคอมพิวเตอร์ที่ใช้ประมวลผลคำสั่ง</li>
          <li><strong>Memory (RAM):</strong> ที่เก็บข้อมูลชั่วคราวระหว่างที่โปรแกรมกำลังทำงาน</li>
          <li><strong>Storage:</strong> ที่เก็บข้อมูลถาวร เช่น ฮาร์ดดิสก์ หรือ SSD</li>
          <li><strong>Input/Output:</strong> อุปกรณ์สำหรับรับข้อมูล (เช่น คีย์บอร์ด เมาส์) และแสดงผล (เช่น จอภาพ)</li>
        </ul>
      </p>

      <h2 className="text-2xl font-semibold mt-6">2. จากโค้ดสู่การทำงานจริง</h2>
      <p className="mt-2">
        โค้ดที่เราเขียนจะถูกแปลงเป็นชุดคำสั่งที่คอมพิวเตอร์เข้าใจ โดยมีขั้นตอนหลัก ๆ ดังนี้:
      </p>
      <ul className="list-disc ml-6 mt-2">
        <li>เขียนโค้ด (Source Code)</li>
        <li>แปลงโค้ดเป็นภาษาเครื่อง (ผ่าน Compiler หรือ Interpreter)</li>
        <li>โหลดโปรแกรมเข้าไปใน RAM</li>
        <li>CPU อ่านคำสั่งจาก RAM และทำงานตามลำดับ</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">3. ตัวอย่างง่าย ๆ</h2>
      <p className="mt-2">
        เมื่อเราเขียนโค้ดว่า:
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`print("สวัสดี")`}
      </pre>
      <p className="mt-2">
        ระบบปฏิบัติการจะโหลดโปรแกรม Python ขึ้นมา แล้วโปรแกรมจะอ่านคำสั่ง <code>print()</code> จาก RAM และสั่งให้แสดงข้อความ "สวัสดี" บนหน้าจอ
      </p>

      <h2 className="text-2xl font-semibold mt-6">4. ระบบปฏิบัติการมีบทบาทอย่างไร?</h2>
      <p className="mt-2">
        ระบบปฏิบัติการ (Operating System) เช่น Windows, macOS หรือ Linux จะคอยจัดการการโหลดโปรแกรม การจัดสรรหน่วยความจำ และควบคุมการติดต่อกับอุปกรณ์ต่าง ๆ ของเครื่องคอมพิวเตอร์
      </p>

      <h2 className="text-2xl font-semibold mt-6">5. บทสรุป</h2>
      <p className="mt-2">
        โค้ดที่เราเขียนจะไม่สามารถทำงานได้โดยตรง ต้องผ่านกระบวนการแปลและควบคุมโดยระบบปฏิบัติการและ CPU ซึ่งเข้าใจคำสั่งในระดับล่างที่เรียกว่าภาษาเครื่อง (Machine Code)
      </p>
    </div>
  );
};

export default ComputerExecution;
