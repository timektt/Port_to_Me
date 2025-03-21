import React from "react";

const RestVsGraphQL = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">🔄 REST vs GraphQL</h1>

      <p className="text-lg">
        <strong>REST</strong> และ <strong>GraphQL</strong> เป็นสองแนวทางในการออกแบบ API ที่นิยมในยุคปัจจุบัน
        โดยมีแนวคิดและข้อดีแตกต่างกัน
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 REST API คืออะไร?</h2>
      <p className="mt-2">
        REST (Representational State Transfer) ใช้การออกแบบ API แบบ Resource-based เช่น
        <code className="bg-gray-200 px-1 rounded mx-1">/users</code>, <code className="bg-gray-200 px-1 rounded mx-1">/posts</code>
        โดยใช้ HTTP Methods (GET, POST, PUT, DELETE)
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 GraphQL คืออะไร?</h2>
      <p className="mt-2">
        GraphQL เป็น Query Language ที่พัฒนาโดย Facebook ช่วยให้ Client สามารถดึงข้อมูลได้ตรงตามที่ต้องการ 
        โดยไม่ต้องเรียกหลาย Endpoint
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">📌 REST API</h3>
          <ul className="list-disc ml-5 space-y-1">
            <li>ต้องมีหลาย Endpoint เช่น <code>/users</code>, <code>/posts</code></li>
            <li>ข้อมูลมักถูกส่งมาทั้งชุด แม้ไม่ต้องการทั้งหมด</li>
            <li>เหมาะสำหรับระบบที่มีโครงสร้างตายตัว</li>
          </ul>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">📌 GraphQL</h3>
          <ul className="list-disc ml-5 space-y-1">
            <li>มีเพียง Endpoint เดียว เช่น <code>/graphql</code></li>
            <li>สามารถเลือกเฉพาะข้อมูลที่ต้องการได้</li>
            <li>เหมาะกับ Mobile หรือ Frontend ที่ต้องปรับ UI บ่อย</li>
          </ul>
        </div>
      </div>

      <h2 className="text-xl font-semibold mt-6">📊 เปรียบเทียบ</h2>
      <table className="w-full border border-gray-300 mt-2 text-sm">
        <thead className="bg-gray-200 dark:bg-gray-700">
          <tr>
            <th className="border p-2">หัวข้อ</th>
            <th className="border p-2">REST</th>
            <th className="border p-2">GraphQL</th>
          </tr>
        </thead>
        <tbody className="text-center">
          <tr>
            <td className="border p-2">จำนวน Endpoint</td>
            <td className="border p-2">หลาย Endpoint</td>
            <td className="border p-2">หนึ่ง Endpoint</td>
          </tr>
          <tr>
            <td className="border p-2">ความยืดหยุ่น</td>
            <td className="border p-2">น้อยกว่า</td>
            <td className="border p-2">มากกว่า</td>
          </tr>
          <tr>
            <td className="border p-2">โครงสร้างข้อมูล</td>
            <td className="border p-2">ตายตัว</td>
            <td className="border p-2">กำหนดเองได้</td>
          </tr>
        </tbody>
      </table>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-300 rounded-lg shadow-md">
        💡 <strong>สรุป:</strong> หากคุณต้องการความเรียบง่าย REST ก็เพียงพอ แต่หากต้องการความยืดหยุ่นสูงและลดการโหลดข้อมูลซ้ำซ้อน 
        GraphQL อาจเป็นตัวเลือกที่ดีกว่า
      </div>
    </div>
  );
};

export default RestVsGraphQL;
