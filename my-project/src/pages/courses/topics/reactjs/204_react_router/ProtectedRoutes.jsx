import React from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoutes = ({ isAuthenticated, children }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return children;
};

const ProtectedRoutesPage = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Protected Routes & Authentication</h1>
      <p className="mt-4 text-lg">
        เราสามารถใช้ `<Navigate />` เพื่อตรวจสอบสิทธิ์ก่อนอนุญาตให้เข้าถึงหน้าใด ๆ
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Protected Route</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
        {`<Route path="/dashboard" element={
  <ProtectedRoutes isAuthenticated={userLoggedIn}>
    <Dashboard />
  </ProtectedRoutes>
} />`}
      </pre>
    </div>
  );
};

export { ProtectedRoutes };
export default ProtectedRoutesPage;
