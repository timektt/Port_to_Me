import React from "react";

const NodeModules = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">โมดูลใน Node.js</h1>
      <p className="mt-4">
        Node.js ใช้ **Modules** ในการจัดการโค้ดให้สามารถแบ่งเป็นส่วน ๆ ได้ 
        โดยมี **Core Modules** และ **Custom Modules**
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 ตัวอย่างการใช้โมดูล fs (File System)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`const fs = require('fs');

fs.writeFileSync('test.txt', 'Hello from Node.js!');
console.log("ไฟล์ถูกสร้างเรียบร้อย");`}
      </pre>
    </div>
  );
};

export default NodeModules;
