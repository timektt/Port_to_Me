import React from "react";

const GraphQLVsRest = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">GraphQL vs REST: ข้อดีและข้อเสีย</h1>
      <p className="mb-4">
        การเปรียบเทียบระหว่าง <strong>GraphQL</strong> และ <strong>REST API</strong> ช่วยให้เข้าใจความเหมาะสมในการใช้งานแต่ละรูปแบบ
      </p>

      <table className="w-full border-collapse border border-gray-500 dark:border-gray-600 text-sm sm:text-base">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white">
          <tr>
            <th className="border border-gray-500 dark:border-gray-600 p-3">Feature</th>
            <th className="border border-gray-500 dark:border-gray-600 p-3">GraphQL</th>
            <th className="border border-gray-500 dark:border-gray-600 p-3">REST</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100">
          <tr>
            <td className="border p-3">Over-fetching</td>
            <td className="border p-3">✅ แก้ปัญหานี้ได้ เพราะเลือก field ได้เอง</td>
            <td className="border p-3">❌ ดึงข้อมูลเกินความจำเป็น</td>
          </tr>
          <tr>
            <td className="border p-3">Under-fetching</td>
            <td className="border p-3">✅ รวมหลาย resource ในคำขอเดียว</td>
            <td className="border p-3">❌ ต้องเรียกหลาย endpoint</td>
          </tr>
          <tr>
            <td className="border p-3">Flexibility</td>
            <td className="border p-3">✅ ยืดหยุ่นสูง เลือกข้อมูลตามต้องการ</td>
            <td className="border p-3">❌ จำกัดตามโครงสร้างที่กำหนด</td>
          </tr>
          <tr>
            <td className="border p-3">Learning Curve</td>
            <td className="border p-3">❗ ซับซ้อนกว่าเล็กน้อย</td>
            <td className="border p-3">✅ เข้าใจง่ายกว่า</td>
          </tr>
          <tr>
            <td className="border p-3">Error Handling</td>
            <td className="border p-3">🟡 จัดการรวมไว้ใน response</td>
            <td className="border p-3">✅ ใช้ HTTP Status Code มาตรฐาน</td>
          </tr>
          <tr>
            <td className="border p-3">Tooling</td>
            <td className="border p-3">✅ มีเครื่องมือเช่น GraphiQL, Apollo</td>
            <td className="border p-3">✅ Postman, Swagger</td>
          </tr>
        </tbody>
      </table>

      <p className="mt-6 text-gray-700 dark:text-gray-300">
        ✨ <strong>สรุป:</strong> ถ้าคุณต้องการความยืดหยุ่นในการดึงข้อมูล ใช้ GraphQL จะเหมาะกว่า แต่ถ้าอยากได้แนวทางที่เรียบง่ายและเข้าใจง่าย REST ก็ยังคงเป็นตัวเลือกที่ดี
      </p>
    </div>
  );
};

export default GraphQLVsRest;
