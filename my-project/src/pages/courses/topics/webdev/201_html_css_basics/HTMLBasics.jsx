import React from "react";

const HTMLBasics = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน HTML</h1>
      <p>
        HTML (HyperText Markup Language) เป็นภาษาหลักที่ใช้ในการสร้างโครงสร้างของหน้าเว็บ
        โดยทำหน้าที่กำหนดองค์ประกอบต่างๆ ของหน้าเว็บ เช่น หัวข้อ ย่อหน้า รูปภาพ ลิงก์ และตาราง
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">โครงสร้างของเอกสาร HTML</h2>
      <p>
        เอกสาร HTML ประกอบไปด้วยส่วนสำคัญ ได้แก่
      </p>
      <ul className="list-disc pl-6 mb-4">
        <li><strong>&lt;!DOCTYPE html&gt;</strong> – กำหนดเวอร์ชันของ HTML</li>
        <li><strong>&lt;html&gt;</strong> – ส่วนหลักของเอกสาร HTML</li>
        <li><strong>&lt;head&gt;</strong> – ส่วนของข้อมูลเมตา (Metadata) เช่น Title, Styles</li>
        <li><strong>&lt;body&gt;</strong> – ส่วนของเนื้อหาหลักของเว็บเพจ</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: โครงสร้าง HTML เบื้องต้น</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<!DOCTYPE html>
<html>
<head>
  <title>หน้าเว็บแรกของฉัน</title>
</head>
<body>
  <h1>ยินดีต้อนรับสู่ HTML</h1>
  <p>นี่คือโครงสร้างพื้นฐานของ HTML</p>
</body>
</html>`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">แท็กพื้นฐานของ HTML</h2>
      <p>
        HTML มีแท็กที่ใช้บ่อยในการจัดรูปแบบเนื้อหา เช่น
      </p>
      <ul className="list-disc pl-6 mb-4">
        <li><strong>&lt;h1&gt; - &lt;h6&gt;</strong> – ใช้สำหรับหัวข้อ</li>
        <li><strong>&lt;p&gt;</strong> – ใช้สำหรับย่อหน้า</li>
        <li><strong>&lt;a&gt;</strong> – ใช้สำหรับลิงก์</li>
        <li><strong>&lt;img&gt;</strong> – ใช้สำหรับรูปภาพ</li>
        <li><strong>&lt;ul&gt;, &lt;ol&gt;, &lt;li&gt;</strong> – ใช้สำหรับรายการแบบไม่มีลำดับและมีลำดับ</li>
      </ul>
    </>
  );
};

export default HTMLBasics;
