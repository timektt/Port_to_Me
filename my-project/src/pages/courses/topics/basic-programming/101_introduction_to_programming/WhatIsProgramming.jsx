import React from "react";

const WhatIsProgramming = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">บทนำสู่การเขียนโปรแกรม (What is Programming?)</h1>
      <p className="text-base sm:text-lg leading-relaxed">
        การเขียนโปรแกรม (Programming) คือกระบวนการเขียนชุดคำสั่งเพื่อให้คอมพิวเตอร์ทำงานตามที่เราต้องการ โดยใช้ภาษาคอมพิวเตอร์ที่มนุษย์สามารถเข้าใจและคอมพิวเตอร์สามารถประมวลผลได้ เช่น Python, JavaScript, C++ เป็นต้น
      </p>

      <h2 className="text-xl font-semibold mt-6">🧠 ทำไมต้องเรียนรู้การเขียนโปรแกรม?</h2>
      <ul className="list-disc ml-5 mt-2 space-y-2">
        <li>เพื่อสร้างแอปพลิเคชันและเว็บไซต์</li>
        <li>ใช้แก้ไขปัญหาหรือทำงานซ้ำ ๆ ให้เป็นอัตโนมัติ</li>
        <li>เข้าใจหลักการทำงานของเทคโนโลยีรอบตัว</li>
        <li>เปิดโอกาสทางอาชีพในสายงานด้านเทคโนโลยี</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">💡 ส่วนประกอบของการเขียนโปรแกรม</h2>
      <ul className="list-disc ml-5 mt-2 space-y-2">
        <li><strong>ตัวแปร (Variables):</strong> ใช้เก็บข้อมูล</li>
        <li><strong>คำสั่งควบคุม (Control Statements):</strong> เช่น if, loop</li>
        <li><strong>ฟังก์ชัน (Functions):</strong> กลุ่มคำสั่งที่สามารถเรียกใช้งานซ้ำได้</li>
        <li><strong>โครงสร้างข้อมูล (Data Structures):</strong> เช่น list, dictionary</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างโค้ดง่าย ๆ</h2>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded-md overflow-x-auto">
        <code>{`name = input("กรุณาป้อนชื่อของคุณ: ")
print("สวัสดี", name)`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🚀 สรุป</h2>
      <p className="text-base sm:text-lg leading-relaxed">
        การเขียนโปรแกรมคือการสื่อสารกับคอมพิวเตอร์เพื่อให้มันทำงานแทนเรา ด้วยเครื่องมือและภาษาที่ถูกออกแบบมาให้มนุษย์เข้าใจได้ง่ายและใช้งานได้อย่างยืดหยุ่น
      </p>
    </div>
  );
};

export default WhatIsProgramming;
