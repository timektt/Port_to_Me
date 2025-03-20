import React from "react";

const DOMManipulation = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การจัดการ DOM ด้วย JavaScript</h1>
      <p>
        Document Object Model (DOM) เป็นโครงสร้างของเอกสาร HTML ที่สามารถเข้าถึงและเปลี่ยนแปลงได้โดยใช้ JavaScript
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การเลือกองค์ประกอบใน DOM</h2>
      <p>
        JavaScript สามารถเลือกองค์ประกอบในหน้าเว็บได้หลายวิธี เช่น <code>getElementById</code>, <code>querySelector</code>, และ <code>getElementsByClassName</code>
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: เลือกและเปลี่ยนข้อความของ Element</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`document.getElementById("title").textContent = "เปลี่ยนข้อความด้วย JavaScript";`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การเปลี่ยนแปลงสไตล์ผ่าน DOM</h2>
      <p>
        JavaScript สามารถใช้ <code>style</code> property เพื่อเปลี่ยนแปลงลักษณะขององค์ประกอบ HTML
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: เปลี่ยนสีพื้นหลังของ Element</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`document.getElementById("box").style.backgroundColor = "blue";`}
      </pre>
    </>
  );
};

export default DOMManipulation;
