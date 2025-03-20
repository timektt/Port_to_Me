import React from "react";

const CSSGridFlexbox = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">CSS Grid & Flexbox</h1>
      <p>
        CSS Grid และ Flexbox เป็นเทคนิคที่ใช้ในการจัดวางองค์ประกอบของหน้าเว็บให้เป็นระเบียบและใช้งานง่ายขึ้น
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ Flexbox</h2>
      <p>
        Flexbox เป็นเครื่องมือที่ช่วยในการจัดวางองค์ประกอบแบบยืดหยุ่น สามารถกำหนดการเรียงตัวขององค์ประกอบได้ทั้งแนวตั้งและแนวนอน
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Flexbox</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`.container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ CSS Grid</h2>
      <p>
        CSS Grid เป็นระบบจัดวางองค์ประกอบที่ใช้ในการสร้างเลย์เอาต์ที่ซับซ้อนได้ง่ายขึ้น
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ CSS Grid</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`.grid-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}`}
      </pre>
    </>
  );
};

export default CSSGridFlexbox;
