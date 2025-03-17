import React from "react";

const NodeIntro = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">แนะนำ Node.js</h1>
      <p className="mt-4">
        Node.js เป็น JavaScript runtime ที่ช่วยให้สามารถรัน JavaScript นอก Web Browser ได้ 
        โดยใช้ V8 Engine ที่พัฒนาโดย Google
      </p>
      <p className="mt-2">
        จุดเด่นของ Node.js คือ **Non-blocking I/O**, **Event-driven Architecture** 
        และเหมาะสำหรับงาน **Backend Development**
      </p>

      <h2 className="text-xl font-semibold mt-6">🔹 ตัวอย่างการรัน Node.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`console.log("Hello from Node.js");`}
      </pre>
    </div>
  );
};

export default NodeIntro;
