import React from "react";

const ApiDocumentation = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">📘 API Documentation & Tools</h1>

      <p className="text-lg">
        การเขียนเอกสาร API (API Documentation) เป็นสิ่งสำคัญอย่างยิ่งในการพัฒนา Web API 
        เพราะช่วยให้ทีมพัฒนาและผู้ใช้งานเข้าใจวิธีการใช้งาน endpoint ต่าง ๆ ได้อย่างถูกต้อง
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 องค์ประกอบที่ควรมีใน API Docs</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li>คำอธิบายแต่ละ Endpoint</li>
        <li>รูปแบบของ URL และ Method ที่ใช้ (GET, POST, PUT, DELETE)</li>
        <li>รูปแบบของ Request (Headers, Body, Parameters)</li>
        <li>Response ที่คาดหวัง (Status Code, JSON, ข้อมูลตัวอย่าง)</li>
        <li>ตัวอย่าง Error Handling และโค้ดที่ตอบกลับเมื่อเกิดข้อผิดพลาด</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🛠️ เครื่องมือยอดนิยมในการเขียนเอกสาร API</h2>
      <ul className="list-disc ml-6 mt-4 space-y-3">
        <li>
          <strong>Swagger (OpenAPI):</strong> Framework ยอดนิยมสำหรับการสร้างและแสดงเอกสาร API
          <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 overflow-x-auto">
            <code>{`paths:
/users:
  get:
    summary: Get user list
    responses:
      200:
        description: OK`}</code>
          </pre>
        </li>
        <li>
          <strong>Postman:</strong> ใช้ในการทดสอบ API แบบ Interactive และแชร์เอกสารร่วมกันในทีม
          <p className="mt-1">สามารถสร้าง Collection และ Export เป็น Docs ได้</p>
        </li>
        <li>
          <strong>Redoc:</strong> ใช้แสดงผล API documentation สวยงามจากไฟล์ OpenAPI YAML/JSON
        </li>
        <li>
          <strong>Insomnia:</strong> คล้าย Postman แต่เน้น Developer Experience และ UI ที่ลื่นไหล
        </li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🚀 ตัวอย่าง Endpoint พร้อมคำอธิบาย</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm overflow-x-auto">
        <code>{`GET /api/products
Description: ดึงรายการสินค้าทั้งหมด

Response: 
[
  {
    "id": 1,
    "name": "Laptop",
    "price": 20000
  },
  ...
]`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6">✅ ประโยชน์ของเอกสาร API ที่ดี</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1">
        <li>ลดเวลาเรียนรู้ของนักพัฒนาคนใหม่</li>
        <li>ช่วย QA ทดสอบ API ได้เร็วขึ้น</li>
        <li>เพิ่มความน่าเชื่อถือให้กับระบบ</li>
        <li>ช่วยให้ระบบมีมาตรฐาน และพร้อมสำหรับการขยายในอนาคต</li>
      </ul>
    </div>
  );
};

export default ApiDocumentation;
