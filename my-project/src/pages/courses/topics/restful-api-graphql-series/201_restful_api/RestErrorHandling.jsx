import React from "react";

const RestErrorHandling = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">การจัดการข้อผิดพลาดใน REST API</h1>
      <p className="mb-4">
        ข้อผิดพลาดเป็นสิ่งที่หลีกเลี่ยงไม่ได้ในการพัฒนา API เราจึงต้องออกแบบระบบให้สามารถจัดการกับข้อผิดพลาดได้อย่างถูกต้อง
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 การใช้ HTTP Status Codes</h2>
      <table className="w-full border-collapse border border-gray-300 text-left">
        <thead>
          <tr className="bg-gray-200">
            <th className="border p-2">รหัส</th>
            <th className="border p-2">คำอธิบาย</th>
          </tr>
        </thead>
        <tbody>
          <tr><td className="border p-2">200</td><td className="border p-2">OK (สำเร็จ)</td></tr>
          <tr><td className="border p-2">400</td><td className="border p-2">Bad Request (คำขอไม่ถูกต้อง)</td></tr>
          <tr><td className="border p-2">404</td><td className="border p-2">Not Found (ไม่พบข้อมูล)</td></tr>
          <tr><td className="border p-2">500</td><td className="border p-2">Internal Server Error (ข้อผิดพลาดภายในเซิร์ฟเวอร์)</td></tr>
        </tbody>
      </table>
    </div>
  );
};

export default RestErrorHandling;
