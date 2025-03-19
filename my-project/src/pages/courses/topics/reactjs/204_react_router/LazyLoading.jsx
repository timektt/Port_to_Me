import React, { Suspense, lazy } from "react";

const LazyComponent = lazy(() => import("../../../../../components/HeavyComponent")); // 

const LazyLoading = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Lazy Loading with React Router</h1>
      <p className="mt-4 text-lg">
        React Router รองรับ **Lazy Loading** เพื่อโหลดคอมโพเนนต์เฉพาะเมื่อจำเป็น
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้งาน</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
        {`import { lazy, Suspense } from "react";
const Dashboard = lazy(() => import("./Dashboard"));

<Route path="/dashboard" element={
  <Suspense fallback={<div>Loading...</div>}>
    <Dashboard />
  </Suspense>
} />`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 แสดง Component แบบ Lazy</h2>
      <Suspense fallback={<p>Loading Component...</p>}>
        <LazyComponent />
      </Suspense>
    </div>
  );
};

export default LazyLoading;
