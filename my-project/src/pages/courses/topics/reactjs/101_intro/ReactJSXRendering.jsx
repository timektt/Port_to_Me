import React from "react";

const ReactJSXRendering = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">✨ JSX & Rendering ใน React</h1>

      <p className="text-lg mb-4">
        JSX (JavaScript XML) เป็นไวยากรณ์พิเศษที่ใช้เขียน UI ใน React ซึ่งดูคล้าย HTML แต่สามารถใช้ JavaScript แทรกเข้าไปได้
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่าง JSX เบื้องต้น</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`const name = "React";

const element = <h1>Hello, {name}!</h1>;`}</code>
      </pre>

      <p className="mt-4">
        JSX สามารถใช้ JavaScript expression เช่น ตัวแปรหรือฟังก์ชันภายใน <code>{`{}`}</code> ได้โดยตรง
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">🔁 การ Render List</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`const items = ["Apple", "Banana", "Orange"];

const list = (
  <ul>
    {items.map((item, index) => <li key={index}>{item}</li>)}
  </ul>
);`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">⚡ การใช้เงื่อนไขใน JSX</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
        <code>{`const isLoggedIn = true;

const greeting = (
  <div>
    {isLoggedIn ? <p>Welcome back!</p> : <p>Please log in.</p>}
  </div>
);`}</code>
      </pre>

      <p className="mt-4">
        React จะทำการ render ค่าที่ JSX สร้างขึ้นให้อยู่ใน DOM โดยอัตโนมัติผ่าน Virtual DOM
      </p>
    </div>
  );
};

export default ReactJSXRendering;
