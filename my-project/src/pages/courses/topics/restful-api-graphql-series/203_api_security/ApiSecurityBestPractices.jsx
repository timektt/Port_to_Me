import React from "react";

const ApiSecurityBestPractices = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4 text-gray-900 dark:text-white">
        🔐 Best Practices for API Security
      </h1>

      <p className="mb-4 text-gray-700 dark:text-gray-300">
        การรักษาความปลอดภัยของ API เป็นเรื่องสำคัญ เราสามารถทำตามแนวทางปฏิบัติเหล่านี้เพื่อป้องกันข้อมูลรั่วไหลและการโจมตี:
      </p>

      <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
        <li>✅ ใช้ <strong>HTTPS</strong> เสมอ เพื่อเข้ารหัสข้อมูลที่ส่ง</li>
        <li>✅ ใช้ <strong>OAuth2</strong> หรือ <strong>JWT</strong> แทน Basic Authentication</li>
        <li>✅ กำหนดสิทธิ์การเข้าถึง (Authorization) โดยใช้ Role หรือ Scope</li>
        <li>✅ ใช้ <strong>Rate Limiting</strong> และ <strong>Throttling</strong> เพื่อป้องกัน DDoS</li>
        <li>✅ บันทึก Log การใช้งาน API เพื่อวิเคราะห์ความผิดปกติ</li>
        <li>✅ ตรวจสอบและฟิลเตอร์ข้อมูล Input เพื่อลดความเสี่ยงจาก SQL Injection และ XSS</li>
      </ul>

      <div className="mt-6 p-4 bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 rounded-lg shadow">
        💡 <strong>Tip:</strong> อย่าลืมใช้ API Gateway และระบบตรวจสอบ API เช่น <code>Helmet</code> หรือ <code>Rate-limiter</code> เพื่อเสริมความปลอดภัยอีกขั้น
      </div>
    </div>
  );
};

export default ApiSecurityBestPractices;
