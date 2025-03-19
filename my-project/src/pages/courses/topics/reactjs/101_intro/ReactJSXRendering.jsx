import React from "react";

const ReactSetup = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">การติดตั้ง React.js</h1>
      <p className="mt-4 text-lg">
        ในการเริ่มต้นใช้งาน React.js สามารถติดตั้งผ่าน <strong>Vite</strong> หรือ <strong>Create React App (CRA)</strong> ได้ดังนี้:
      </p>

      <h2 className="text-2xl font-semibold mt-6">🚀 ติดตั้งด้วย Vite (แนะนำ)</h2>
      <pre className="p-4 rounded-md mt-4 overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
        {`npx create-vite my-app --template react`}
      </pre>
      <p className="mt-2">จากนั้นเข้าไปที่โฟลเดอร์และติดตั้งแพ็กเกจ:</p>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
        {`cd my-app
npm install
npm run dev`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 ติดตั้งด้วย Create React App (CRA)</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
        {`npx create-react-app my-app
cd my-app
npm start`}
      </pre>
      <p className="mt-4">ทั้งสองวิธีนี้จะสร้างโปรเจค React ที่พร้อมใช้งาน</p>
    </div>
  );
};

export default ReactSetup;
