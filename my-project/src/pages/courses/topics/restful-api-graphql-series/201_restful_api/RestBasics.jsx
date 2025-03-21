import React from "react";

const RestBasics = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">พื้นฐานของ RESTful API</h1>
      <p className="mb-4">
        RESTful API เป็นรูปแบบหนึ่งของ API ที่ใช้หลักการของ{" "}
        <strong>REST (Representational State Transfer)</strong>{" "}
        ซึ่งออกแบบให้สามารถทำงานบนโปรโตคอล HTTP ได้อย่างมีประสิทธิภาพ
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 หลักการสำคัญของ REST</h2>
      <ul className="list-disc pl-6 space-y-2">
        <li><strong>Stateless:</strong> ไม่มีการเก็บสถานะระหว่างคำร้องขอ (Request)</li>
        <li><strong>Client-Server:</strong> ฝั่ง Client และ Server แยกออกจากกันชัดเจน</li>
        <li><strong>Cacheable:</strong> รองรับการแคชข้อมูลเพื่อลดโหลดของเซิร์ฟเวอร์</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 HTTP Methods ใน REST</h2>
      <table className="w-full border-collapse border border-gray-300 text-left">
        <thead>
          <tr className="bg-gray-200">
            <th className="border p-2">Method</th>
            <th className="border p-2">การใช้งาน</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border p-2">GET</td>
            <td className="border p-2">ใช้ดึงข้อมูล</td>
          </tr>
          <tr>
            <td className="border p-2">POST</td>
            <td className="border p-2">ใช้สร้างข้อมูลใหม่</td>
          </tr>
          <tr>
            <td className="border p-2">PUT</td>
            <td className="border p-2">ใช้แก้ไขข้อมูล</td>
          </tr>
          <tr>
            <td className="border p-2">DELETE</td>
            <td className="border p-2">ใช้ลบข้อมูล</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default RestBasics;
