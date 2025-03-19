import React from "react";

const NestedRoutes = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Nested Routes & Dynamic Routes</h1>
      <p className="mt-4 text-lg">
        Nested Routes ช่วยให้เราสร้างเส้นทางซ้อนกันได้ ทำให้โครงสร้างของแอปมีระเบียบมากขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Nested Routes</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
        {`<Route path="/dashboard" element={<Dashboard />}>
  <Route path="profile" element={<Profile />} />
  <Route path="settings" element={<Settings />} />
</Route>`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Dynamic Routes</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
        {`<Route path="/user/:id" element={<UserProfile />} />`}
      </pre>
    </div>
  );
};

export default NestedRoutes;
