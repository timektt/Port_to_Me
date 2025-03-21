import React from "react";

const NestedRoutes = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Nested Routes & Dynamic Routes</h1>

      <p className="mt-4 text-lg">
        <strong>Nested Routes</strong> หรือ Route ซ้อนกัน เป็นแนวทางที่ช่วยจัดโครงสร้าง Route ให้มีระเบียบ และสามารถแสดง Layout ร่วมกันได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Nested Routes</h2>
      <p className="mt-2">โครงสร้างนี้จะทำให้ <code>Dashboard</code> เป็น Layout หลักที่แสดงร่วมกับ <code>Profile</code> และ <code>Settings</code></p>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`<Route path="/dashboard" element={<Dashboard />}>
  <Route path="profile" element={<Profile />} />
  <Route path="settings" element={<Settings />} />
</Route>`}
      </pre>

      <p className="mt-4">
        ผู้ใช้สามารถเข้าถึง <code>/dashboard/profile</code> และ <code>/dashboard/settings</code> ได้ภายใต้ Layout เดียวกัน
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Dynamic Routes</h2>
      <p className="mt-2">ใช้สำหรับสร้าง Route ที่มีพารามิเตอร์ เช่น ID ของผู้ใช้</p>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`<Route path="/user/:id" element={<UserProfile />} />`}
      </pre>

      <p className="mt-4">
        ใน Component <code>UserProfile</code> สามารถเข้าถึง <code>id</code> ได้จาก <code>useParams()</code> ของ React Router
      </p>

      <h2 className="text-2xl font-semibold mt-6">🧠 สรุป</h2>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li><strong>Nested Routes</strong>: ใช้สร้างโครงสร้างซ้อนกันเช่นหน้า Dashboard</li>
        <li><strong>Dynamic Routes</strong>: ใช้รับค่าแบบไดนามิกจาก URL เช่น <code>:id</code>, <code>:slug</code></li>
      </ul>
    </div>
  );
};

export default NestedRoutes;
