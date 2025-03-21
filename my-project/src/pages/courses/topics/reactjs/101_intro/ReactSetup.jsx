import React from "react";

const ReactSetup = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">🛠️ การติดตั้ง React.js</h1>

      <p className="text-lg mb-6">
        ในการเริ่มต้นใช้งาน React.js สามารถติดตั้งผ่าน <strong>Vite</strong> หรือ <strong>Create React App (CRA)</strong> ได้ดังนี้:
      </p>

      <h2 className="text-2xl font-semibold mt-8 mb-2">🚀 ติดตั้งด้วย Vite (แนะนำ)</h2>
      <pre className="p-4 rounded-md overflow-x-auto bg-gray-800 text-white mb-2">
{`npx create-vite my-app --template react`}
      </pre>
      <p className="mb-2">จากนั้นเข้าไปที่โฟลเดอร์และติดตั้งแพ็กเกจ:</p>
      <pre className="p-4 rounded-md overflow-x-auto bg-gray-800 text-white">
{`cd my-app
npm install
npm run dev`}
      </pre>

      <h2 className="text-2xl font-semibold mt-10 mb-2">📌 ติดตั้งด้วย Create React App (CRA)</h2>
      <pre className="p-4 rounded-md overflow-x-auto bg-gray-800 text-white">
{`npx create-react-app my-app
cd my-app
npm start`}
      </pre>

      <p className="mt-6">
        ✅ ทั้งสองวิธีจะช่วยสร้างโปรเจกต์ React ที่พร้อมใช้งานทันทีบนเบราว์เซอร์ของคุณ
      </p>
    </div>
  );
};

export default ReactSetup;
