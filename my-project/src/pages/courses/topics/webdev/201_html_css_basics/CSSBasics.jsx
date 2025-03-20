import React from "react";

const CSSBasics = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน CSS</h1>
      <p>
        CSS (Cascading Style Sheets) เป็นภาษาที่ใช้ในการกำหนดรูปแบบของหน้าเว็บ เช่น สี ขนาด และการจัดตำแหน่งขององค์ประกอบ HTML
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้งาน CSS</h2>
      <p>
        CSS สามารถนำไปใช้ได้ 3 วิธีหลัก ได้แก่:
      </p>
      <ul className="list-disc pl-6 mb-4">
        <li><strong>Inline CSS</strong> – กำหนดสไตล์ภายในแท็ก HTML โดยตรง</li>
        <li><strong>Internal CSS</strong> – กำหนดสไตล์ภายในแท็ก &lt;style&gt; ในเอกสาร HTML</li>
        <li><strong>External CSS</strong> – ใช้ไฟล์ CSS แยกจากเอกสาร HTML</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Inline CSS</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<h1 style="color: blue; font-size: 24px;">ยินดีต้อนรับสู่ CSS</h1>`}
      </pre>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Internal CSS</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<style>
  h1 {
    color: blue;
    font-size: 24px;
  }
</style>`}
      </pre>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ External CSS</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`/* ไฟล์ styles.css */
h1 {
  color: blue;
  font-size: 24px;
}`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การเลือกองค์ประกอบใน CSS</h2>
      <p>
        CSS ใช้ตัวเลือก (Selectors) ในการกำหนดสไตล์ให้กับองค์ประกอบ HTML
      </p>
      <ul className="list-disc pl-6 mb-4">
        <li><strong>ตัวเลือกตามแท็ก (Tag Selector)</strong>: กำหนดสไตล์ให้กับแท็กทั้งหมด เช่น h1, p</li>
        <li><strong>ตัวเลือกตามคลาส (Class Selector)</strong>: ใช้จุด (.) นำหน้า เช่น .my-class</li>
        <li><strong>ตัวเลือกตามไอดี (ID Selector)</strong>: ใช้เครื่องหมาย # นำหน้า เช่น #my-id</li>
      </ul>
    </>
  );
};

export default CSSBasics;
