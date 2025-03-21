import React from "react";
import { Navigate } from "react-router-dom";

// ✅ Component ที่ใช้สำหรับป้องกันเส้นทางที่ต้องมีการ login
const ProtectedRoutes = ({ isAuthenticated, children }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

const ProtectedRoutesPage = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Protected Routes & Authentication</h1>

      <p className="mt-4 text-lg">
        ในแอปที่มีระบบ Login เราสามารถสร้างหน้า **Protected Routes** 
        เพื่อป้องกันไม่ให้ผู้ใช้ที่ยังไม่ล็อกอินเข้าถึงหน้าเฉพาะได้ โดยใช้ <code>{`<Navigate />`}</code> 
        จาก React Router เพื่อนำผู้ใช้กลับไปที่หน้า Login
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 โค้ดของ ProtectedRoutes</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-sm text-black dark:text-white">
{`const ProtectedRoutes = ({ isAuthenticated, children }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
};`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 การใช้งานใน <code>&lt;Route&gt;</code></h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-sm text-black dark:text-white">
{`<Route path="/dashboard" element={
  <ProtectedRoutes isAuthenticated={userLoggedIn}>
    <Dashboard />
  </ProtectedRoutes>
} />`}
      </pre>

      <p className="mt-4">
        ในตัวอย่างนี้ ถ้า <code>userLoggedIn</code> เป็น <code>false</code> 
        ผู้ใช้จะถูก redirect ไปที่ <code>/login</code> โดยอัตโนมัติ
      </p>
    </div>
  );
};

export { ProtectedRoutes };
export default ProtectedRoutesPage;
