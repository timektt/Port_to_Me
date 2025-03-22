import React from "react";

const ReactIntroWebdev = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">แนะนำ React</h1>
      <p>
        React เป็นไลบรารี JavaScript สำหรับการสร้าง UI ที่มีประสิทธิภาพสูง และใช้แนวทางการพัฒนาแบบ Component-based
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติหลักของ React</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>การทำงานแบบ Declarative UI</li>
        <li>ใช้ Virtual DOM เพื่อเพิ่มประสิทธิภาพ</li>
        <li>สามารถจัดการ State และ Props ได้ง่าย</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: React Component แรก</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`function Welcome() {
  return <h1>สวัสดี React!</h1>;
}
export default Welcome;`}
      </pre>
    </>
  );
};

export default ReactIntroWebdev;
