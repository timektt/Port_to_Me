import React from "react";

const ApiSecurityBestPractices = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Best Practices for API Security</h1>
      <p className="mb-4">
        การรักษาความปลอดภัยของ API เป็นเรื่องสำคัญ เราสามารถทำตามแนวทางปฏิบัติเหล่านี้ได้:
      </p>
      <ul className="list-disc ml-6 space-y-2">
        <li>✅ ใช้ HTTPS เพื่อเข้ารหัสข้อมูล</li>
        <li>✅ ใช้ OAuth หรือ JWT แทน Basic Authentication</li>
        <li>✅ กำหนดสิทธิ์การเข้าถึง API อย่างเคร่งครัด</li>
        <li>✅ ใช้ Rate Limiting ป้องกัน DDoS</li>
        <li>✅ ตรวจสอบและบันทึก (Logging) การใช้งาน API</li>
      </ul>
    </div>
  );
};

export default ApiSecurityBestPractices;
