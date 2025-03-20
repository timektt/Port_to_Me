import React from "react";

const NodePackageManager = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      {/* ✅ Header */}
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center sm:text-left">
        📦 Node.js Package Manager (NPM & Yarn)
      </h1>

      {/* ✅ คำอธิบาย */}
      <p className="mt-4 text-lg">
        Node.js มีระบบจัดการแพ็กเกจที่ช่วยให้สามารถติดตั้ง ไลบรารี และจัดการ dependencies ได้สะดวกขึ้น 
        โดยตัวที่ใช้กันมากที่สุดคือ <strong>NPM (Node Package Manager)</strong> และ <strong>Yarn</strong>
      </p>

      {/* ✅ แบ่งส่วนหัวข้อ */}
      <div className="mt-6">
        <h2 className="text-xl font-semibold">📌 1. การติดตั้งและใช้งาน NPM</h2>
        <p className="mt-2">NPM มากับ Node.js โดยอัตโนมัติ สามารถตรวจสอบเวอร์ชันได้ด้วยคำสั่ง:</p>
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
          <code>npm -v</code>
        </pre>

        <p className="mt-2">ติดตั้งแพ็กเกจในโปรเจกต์:</p>
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
          <code>npm install express</code>
        </pre>

        <p className="mt-2">ติดตั้งแพ็กเกจแบบ global:</p>
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
          <code>npm install -g nodemon</code>
        </pre>
      </div>

      <div className="mt-6">
        <h2 className="text-xl font-semibold">📌 2. การติดตั้งและใช้งาน Yarn</h2>
        <p className="mt-2">Yarn เป็นอีกหนึ่งตัวจัดการแพ็กเกจที่เร็วและมีประสิทธิภาพสูง สามารถติดตั้ง Yarn ได้ผ่าน NPM:</p>
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
          <code>npm install -g yarn</code>
        </pre>

        <p className="mt-2">ตรวจสอบเวอร์ชันของ Yarn:</p>
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
          <code>yarn -v</code>
        </pre>

        <p className="mt-2">ติดตั้งแพ็กเกจโดยใช้ Yarn:</p>
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
          <code>yarn add express</code>
        </pre>
      </div>

      {/* ✅ เปรียบเทียบ NPM vs Yarn */}
      <div className="mt-6">
        <h2 className="text-xl font-semibold">⚖️ เปรียบเทียบระหว่าง NPM และ Yarn</h2>
        <table className="w-full mt-4 border-collapse border border-gray-700">
          <thead className="bg-gray-700 text-white">
            <tr>
              <th className="p-3 border">คุณสมบัติ</th>
              <th className="p-3 border">NPM</th>
              <th className="p-3 border">Yarn</th>
            </tr>
          </thead>
          <tbody className="text-center">
            <tr className="bg-gray-800 text-white">
              <td className="p-3 border">ความเร็ว</td>
              <td className="p-3 border">ปกติ</td>
              <td className="p-3 border">เร็วกว่า</td>
            </tr>
            <tr>
              <td className="p-3 border">การล็อกเวอร์ชัน</td>
              <td className="p-3 border">package-lock.json</td>
              <td className="p-3 border">yarn.lock</td>
            </tr>
            <tr className="bg-gray-800 text-white">
              <td className="p-3 border">การติดตั้งแพ็กเกจ</td>
              <td className="p-3 border">npm install</td>
              <td className="p-3 border">yarn install</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ✅ สรุป */}
      <div className="mt-6">
        <h2 className="text-xl font-semibold">🔎 สรุป</h2>
        <p className="mt-2">
          <strong>NPM</strong> และ <strong>Yarn</strong> เป็นตัวจัดการแพ็กเกจที่สำคัญในการพัฒนา Node.js สามารถเลือกใช้ตามความสะดวก
          โดย Yarn มักจะเร็วกว่า แต่ NPM มีการพัฒนาให้รองรับฟีเจอร์ใหม่ๆ มากขึ้น
        </p>
      </div>
    </div>
  );
};

export default NodePackageManager;