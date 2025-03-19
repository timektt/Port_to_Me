import React from "react";

const ReactRouterIntro = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Introduction to React Router</h1>
      <p className="mt-4 text-lg">
        React Router เป็นไลบรารีที่ช่วยให้เราสร้าง **Single Page Application (SPA)** โดยใช้การนำทางแบบไดนามิก
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ติดตั้ง React Router</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
        {`npm install react-router-dom`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้งานพื้นฐาน</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
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
    </div>
  );
};

export default ReactRouterIntro;
