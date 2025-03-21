import React from "react";

const ApiIntro = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">🌐 What is an API?</h1>

      <p className="text-lg">
        API (Application Programming Interface) คือชุดของกฎและคำสั่งที่ช่วยให้แอปพลิเคชันต่าง ๆ สามารถสื่อสารกันได้
        โดยไม่จำเป็นต้องรู้โครงสร้างภายในของกันและกัน
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ทำไมต้องใช้ API?</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li>ช่วยให้แอปเชื่อมต่อกับบริการอื่น ๆ ได้ (เช่น Google Maps, Payment Gateway)</li>
        <li>ช่วยให้ระบบภายในแยกส่วนกันได้ชัดเจน (Frontend ↔ Backend)</li>
        <li>สามารถใช้งานซ้ำ (Reusable) ได้ในหลายระบบ</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🧩 ตัวอย่างการใช้งาน API</h2>
      <p className="mt-2">
        สมมติว่าแอปของคุณต้องการแสดงข้อมูลสภาพอากาศ คุณสามารถเรียกใช้ API จาก OpenWeatherMap เพื่อดึงข้อมูลมาแสดงได้ทันที
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-4 overflow-x-auto text-sm">
        <code>{`GET https://api.openweathermap.org/data/2.5/weather?q=Bangkok&appid=YOUR_API_KEY`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📦 ประเภทของ API</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li><strong>REST API</strong> – ใช้ HTTP และ JSON ในการรับส่งข้อมูล</li>
        <li><strong>GraphQL</strong> – API แบบยืดหยุ่นในการ Query ข้อมูล</li>
        <li><strong>SOAP</strong> – โปรโตคอลแบบเก่าที่ใช้ XML</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">✅ สรุป</h2>
      <p className="mt-2">
        API คือหัวใจของระบบสมัยใหม่ ไม่ว่าจะเป็น Web, Mobile หรือ IoT เพราะช่วยให้ระบบต่าง ๆ เชื่อมต่อ แลกเปลี่ยนข้อมูล และทำงานร่วมกันได้อย่างมีประสิทธิภาพ
      </p>
    </div>
  );
};

export default ApiIntro;
