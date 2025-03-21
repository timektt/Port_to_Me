import React from "react";

const ApiDeploymentScaling = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🚀 Deploying & Scaling APIs</h1>
      <p className="mb-4 text-gray-700 dark:text-gray-300">
        การนำ API ขึ้นใช้งานจริง (Deploy) และการปรับขนาดระบบ (Scaling) มีความสำคัญมากเมื่อมีผู้ใช้จำนวนมาก หรือมีการโหลดใช้งานสูง
      </p>

      <h2 className="text-2xl font-semibold mt-6">📦 การ Deploy ด้วย Docker</h2>
      <p className="mb-2">สามารถสร้าง Docker Image เพื่อ Deploy API ได้ง่ายและรวดเร็ว</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`# Dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]`}</code>
      </pre>

      <p className="mt-4">🔧 สร้างและรัน Container:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`docker build -t my-api .
docker run -p 3000:3000 my-api`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🌐 Deploy ไปยังแพลตฟอร์มคลาวด์</h2>
      <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
        <li><strong>Render / Railway / Vercel:</strong> สำหรับ Deploy แบบง่าย</li>
        <li><strong>Heroku:</strong> ใช้งานง่าย และเชื่อมต่อกับ Git ได้</li>
        <li><strong>AWS / GCP / Azure:</strong> สำหรับองค์กรที่ต้องการควบคุมโครงสร้างพื้นฐาน</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📈 การ Scaling API</h2>
      <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
        <li><strong>Horizontal Scaling:</strong> เพิ่มจำนวน Instance (เช่นรันหลาย Container)</li>
        <li><strong>Load Balancer:</strong> กระจายโหลดไปยังหลายเซิร์ฟเวอร์</li>
        <li><strong>Auto-scaling:</strong> เพิ่ม-ลด Resource ตาม Traffic โดยอัตโนมัติ</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">🛡️ เคล็ดลับก่อน Deploy</h2>
      <ul className="list-disc ml-6 space-y-2 text-gray-700 dark:text-gray-300">
        <li>เปิดใช้ CORS และกำหนด Security Header</li>
        <li>เพิ่ม Rate Limiting & Logging</li>
        <li>ใช้ .env แยก Config ที่สำคัญ เช่น DB, API Key</li>
        <li>ติดตั้งระบบ Monitoring (เช่น Prometheus, Grafana, Logtail)</li>
      </ul>
    </div>
  );
};

export default ApiDeploymentScaling;
