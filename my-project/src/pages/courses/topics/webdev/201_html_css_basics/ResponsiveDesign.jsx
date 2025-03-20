import React from "react";

const ResponsiveDesign = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การออกแบบเว็บให้รองรับ Responsive</h1>
      <p>
        Responsive Design คือแนวทางในการออกแบบเว็บให้สามารถปรับเปลี่ยนได้ตามขนาดของอุปกรณ์ที่ใช้ เช่น มือถือ แท็บเล็ต และเดสก์ท็อป
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ Media Queries</h2>
      <p>
        CSS Media Queries ใช้สำหรับกำหนดสไตล์ที่แตกต่างกันตามขนาดหน้าจอของอุปกรณ์
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: ใช้ Media Queries</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`@media (max-width: 768px) {
  body {
    background-color: lightgray;
  }
}`}
      </pre>
    </>
  );
};

export default ResponsiveDesign;
