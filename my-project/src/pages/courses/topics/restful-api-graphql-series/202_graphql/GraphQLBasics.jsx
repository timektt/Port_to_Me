import React from "react";

const GraphQLBasics = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🔍 GraphQL Basics</h1>
      <p className="mb-4">
        GraphQL เป็นภาษาสำหรับ Query ข้อมูลที่พัฒนาโดย Facebook ซึ่งช่วยให้ Client สามารถระบุข้อมูลที่ต้องการได้แบบละเอียด และลด Over-fetching / Under-fetching ที่มักเกิดใน REST API
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่าง Query</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
        <code>{`{
  user(id: "1") {
    name
    email
    posts {
      title
      createdAt
    }
  }
}`}</code>
      </pre>
      <p className="mt-4">
        ✅ GraphQL ช่วยให้สามารถดึงเฉพาะข้อมูลที่ต้องการ เช่นชื่อ อีเมล และโพสต์ของผู้ใช้ โดยไม่โหลดข้อมูลส่วนเกิน
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">🧱 โครงสร้างพื้นฐานของ GraphQL</h2>
      <ul className="list-disc ml-6 space-y-2">
        <li><strong>Query:</strong> ใช้สำหรับการดึงข้อมูล</li>
        <li><strong>Mutation:</strong> ใช้สำหรับการเพิ่ม แก้ไข หรือลบข้อมูล</li>
        <li><strong>Subscription:</strong> ใช้สำหรับ real-time updates</li>
        <li><strong>Schema:</strong> เป็นที่กำหนดชนิดข้อมูล และโครงสร้าง API</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6 mb-2">🚀 ข้อดีของ GraphQL</h2>
      <ul className="list-disc ml-6 space-y-2">
        <li>Client ควบคุมข้อมูลที่ต้องการได้</li>
        <li>ลดจำนวน request ที่ต้องยิง</li>
        <li>รวมหลาย resource ไว้ใน request เดียว</li>
        <li>เหมาะกับ frontend ที่ซับซ้อน เช่น React, Vue</li>
      </ul>

      <p className="mt-6">
        💡 คุณสามารถทดลองใช้ GraphQL ได้ผ่านเครื่องมือเช่น <code>GraphiQL</code> หรือ <code>Apollo Studio</code>
      </p>
    </div>
  );
};

export default GraphQLBasics;
