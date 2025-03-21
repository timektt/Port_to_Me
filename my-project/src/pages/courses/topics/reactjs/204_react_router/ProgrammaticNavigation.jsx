import React from "react";
import { useNavigate } from "react-router-dom";

const ProgrammaticNavigation = () => {
  const navigate = useNavigate();

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Programmatic Navigation</h1>

      <p className="mt-4 text-lg">
        ใน React Router เราสามารถเปลี่ยนหน้าแบบโปรแกรมได้ โดยใช้ <code>useNavigate()</code> ซึ่งเหมาะกับการนำทางหลังจากทำบางเงื่อนไขสำเร็จ เช่น login หรือ form submission
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้งานพื้นฐาน</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`import { useNavigate } from "react-router-dom";

const MyComponent = () => {
  const navigate = useNavigate();

  return (
    <button onClick={() => navigate("/dashboard")}>
      Go to Dashboard
    </button>
  );
};`}
      </pre>

      <p className="mt-4">
        ในตัวอย่างนี้ เมื่อคลิกปุ่มจะนำผู้ใช้ไปยังหน้า <code>/dashboard</code> ทันที
      </p>

      <h2 className="text-2xl font-semibold mt-6">🎯 การใช้งานแบบมีเงื่อนไข</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const handleLogin = () => {
  if (loginSuccess) {
    navigate("/dashboard");
  }
};`}
      </pre>

      <p className="mt-4">
        นี่คือการนำทางหลังจากตรวจสอบว่า login สำเร็จ ช่วยให้ UI ตอบสนองต่อเหตุการณ์ได้อย่างมีประสิทธิภาพ
      </p>

      <h2 className="text-2xl font-semibold mt-6">🛠 ปุ่มที่ใช้งานจริง</h2>
      <button 
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-700 transition"
        onClick={() => navigate("/dashboard")}
      >
        Go to Dashboard
      </button>
    </div>
  );
};

export default ProgrammaticNavigation;
