import React, { Suspense, lazy } from "react";

const LazyComponent = lazy(() => import("../../../../../components/HeavyComponent"));

const LazyLoading = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Lazy Loading with React Router</h1>

      <p className="mt-4 text-lg">
        การใช้ <code>React.lazy</code> และ <code>Suspense</code> ช่วยให้สามารถโหลด Component แบบเลื่อนเวลา (Lazy Load)
        เฉพาะเมื่อผู้ใช้เข้าถึง route นั้น ลดขนาด bundle และเพิ่มประสิทธิภาพของแอป
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Lazy Load ร่วมกับ React Router</h2>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`import { lazy, Suspense } from "react";
const Dashboard = lazy(() => import("./Dashboard"));

<Route path="/dashboard" element={
  <Suspense fallback={<div>Loading...</div>}>
    <Dashboard />
  </Suspense>
} />`}
      </pre>

      <p className="mt-4 text-lg">
        เมื่อมีการเข้าถึง <code>/dashboard</code> เท่านั้น ตัว <code>Dashboard</code> component จะถูกโหลดเข้ามาแบบ dynamic
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการแสดง Component แบบ Lazy ในหน้า</h2>
      <Suspense fallback={<p className="text-gray-600 dark:text-gray-300">Loading Component...</p>}>
        <LazyComponent />
      </Suspense>

      <h2 className="text-2xl font-semibold mt-6">📝 สรุป</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li>ใช้ <code>React.lazy()</code> เพื่อโหลด Component แบบแยกไฟล์</li>
        <li>ใช้ <code>&lt;Suspense&gt;</code> ครอบเพื่อกำหนด fallback UI</li>
        <li>เหมาะสำหรับแอปขนาดใหญ่ที่มีหลายหน้า</li>
      </ul>
    </div>
  );
};

export default LazyLoading;
