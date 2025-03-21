import React from "react";

const ReactRouterIntro = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Introduction to React Router</h1>

      <p className="mt-4 text-lg">
        <strong>React Router</strong> คือไลบรารีที่ช่วยให้เราสร้างการนำทางในแอป React แบบ SPA (Single Page Application) 
        โดยไม่ต้องโหลดหน้าเว็บใหม่ทุกครั้งที่เปลี่ยนเส้นทาง
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ติดตั้ง React Router</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`npm install react-router-dom`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 โครงสร้างการใช้งานเบื้องต้น</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </BrowserRouter>
  );
}`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 อธิบายส่วนประกอบ</h2>
      <ul className="list-disc pl-6 space-y-2 mt-2">
        <li><strong>BrowserRouter:</strong> ครอบ component ทั้งหมดเพื่อให้ใช้ routing ได้</li>
        <li><strong>Routes:</strong> รวม route ทั้งหมดในแอป</li>
        <li><strong>Route:</strong> แต่ละเส้นทางที่กำหนด component ที่จะ render</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📌 การนำทางด้วยลิงก์</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav>
      <Link to="/">หน้าแรก</Link>
      <Link to="/about">เกี่ยวกับ</Link>
    </nav>
  );
}`}
      </pre>

      <p className="mt-4">
        การใช้ <code>&lt;Link /&gt;</code> แทน <code>&lt;a /&gt;</code> จะช่วยให้ React Router จัดการการเปลี่ยนหน้าโดยไม่ต้อง reload ทั้งแอป
      </p>
    </div>
  );
};

export default ReactRouterIntro;
