import React from "react";

const SSRvsCSR = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">SSR vs CSR (Server-side Rendering vs Client-side Rendering)</h1>
      <p>
        ในการพัฒนาเว็บแอปพลิเคชัน มีสองแนวทางหลักในการเรนเดอร์หน้าเว็บ: Server-side Rendering (SSR) และ Client-side Rendering (CSR)
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Server-side Rendering (SSR)</h2>
      <p>
        SSR เป็นกระบวนการที่เซิร์ฟเวอร์สร้าง HTML และส่งไปยังเบราว์เซอร์ ทำให้เว็บโหลดเร็วขึ้นและเหมาะสำหรับ SEO
      </p>
      
      <h3 className="text-lg font-medium mt-4">ข้อดีของ SSR</h3>
      <ul className="list-disc pl-6 mb-4">
        <li>เพิ่มประสิทธิภาพ SEO</li>
        <li>โหลดหน้าแรกเร็วขึ้น</li>
        <li>เหมาะสำหรับเว็บที่มีเนื้อหาคงที่</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: SSR ใน Next.js</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`export async function getServerSideProps() {
  const res = await fetch('https://api.example.com/data');
  const data = await res.json();
  return { props: { data } };
}`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Client-side Rendering (CSR)</h2>
      <p>
        CSR เป็นแนวทางที่ใช้ JavaScript ฝั่งไคลเอนต์ในการโหลดและเรนเดอร์หน้าเว็บ โดยใช้เฟรมเวิร์กอย่าง React หรือ Vue.js
      </p>
      
      <h3 className="text-lg font-medium mt-4">ข้อดีของ CSR</h3>
      <ul className="list-disc pl-6 mb-4">
        <li>อินเตอร์เฟซที่ไหลลื่นและโต้ตอบได้ดี</li>
        <li>ลดโหลดของเซิร์ฟเวอร์</li>
        <li>เหมาะสำหรับแอปพลิเคชันที่ใช้ API</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: CSR ใน React</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`import { useEffect, useState } from 'react';

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('https://api.example.com/data')
      .then(response => response.json())
      .then(setData);
  }, []);

  return <div>{data ? JSON.stringify(data) : 'Loading...'}</div>;
}`}
      </pre>
    </>
  );
};

export default SSRvsCSR;
