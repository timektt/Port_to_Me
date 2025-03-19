import React from "react";
import { useNavigate } from "react-router-dom";

const ProgrammaticNavigation = () => {
  const navigate = useNavigate();

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Programmatic Navigation</h1>
      <p className="mt-4 text-lg">
        เราสามารถใช้ <code>useNavigate()</code> เพื่อเปลี่ยนหน้าโดยไม่ต้องใช้ <code>&lt;Link&gt;</code>
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้งาน <code>useNavigate()</code></h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
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

      <h2 className="text-2xl font-semibold mt-6">🛠 ตัวอย่างปุ่มที่ทำงานจริง</h2>
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
