import React from "react";

const HowTheWebWorks = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การทำงานของเว็บไซต์</h1>
      <p>
        เว็บไซต์ทำงานผ่านเทคโนโลยีต่างๆ ที่ช่วยให้ผู้ใช้สามารถเข้าถึงและโต้ตอบกับเนื้อหาบนอินเทอร์เน็ตได้
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">1. วงจรการร้องขอและตอบกลับ (Request & Response Cycle)</h2>
      <p>
        เมื่อคุณเข้าเว็บไซต์ เบราว์เซอร์ของคุณจะส่ง HTTP Request ไปยังเซิร์ฟเวอร์เว็บ จากนั้นเซิร์ฟเวอร์จะประมวลผลและส่งข้อมูลกลับมาให้แสดงผล
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: คำขอ HTTP พื้นฐาน</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`GET /index.html HTTP/1.1
Host: www.example.com`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">2. บทบาทของ DNS</h2>
      <p>
        ระบบชื่อโดเมน (DNS) แปลงชื่อโดเมนที่มนุษย์อ่านเข้าใจได้ (เช่น google.com) เป็นหมายเลข IP ที่คอมพิวเตอร์ใช้เพื่อค้นหาเซิร์ฟเวอร์
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การแปลงชื่อโดเมน</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`example.com → 192.168.1.1`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">3. เว็บเบราว์เซอร์และการเรนเดอร์</h2>
      <p>
        เว็บเบราว์เซอร์ (Chrome, Firefox, Edge) แปลผล HTML, CSS และ JavaScript เพื่อแสดงหน้าเว็บให้ผู้ใช้เห็น
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: หน้า HTML อย่างง่าย</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<!DOCTYPE html>
<html>
<head>
  <title>Simple Page</title>
</head>
<body>
  <h1>Hello, Web!</h1>
</body>
</html>`}
      </pre>
    </>
  );
};

export default HowTheWebWorks;