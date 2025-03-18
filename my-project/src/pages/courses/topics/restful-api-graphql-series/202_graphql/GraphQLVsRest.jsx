import React from "react";

const GraphQLVsRest = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">GraphQL vs REST: ข้อดีและข้อเสีย</h1>
      <p className="mb-4">
        การเปรียบเทียบระหว่าง <strong>GraphQL</strong> และ <strong>REST API</strong>:
      </p>
      <table className="w-full border-collapse border border-gray-600">
        <thead>
          <tr className="bg-gray-700 text-white">
            <th className="border border-gray-600 p-2">Feature</th>
            <th className="border border-gray-600 p-2">GraphQL</th>
            <th className="border border-gray-600 p-2">REST</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border border-gray-600 p-2">Over-fetching</td>
            <td className="border border-gray-600 p-2">✅ แก้ปัญหานี้</td>
            <td className="border border-gray-600 p-2">❌ ดึงข้อมูลเกินความจำเป็น</td>
          </tr>
          <tr>
            <td className="border border-gray-600 p-2">Flexibility</td>
            <td className="border border-gray-600 p-2">✅ ยืดหยุ่นสูง</td>
            <td className="border border-gray-600 p-2">❌ จำกัดตาม API ที่กำหนด</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default GraphQLVsRest;
