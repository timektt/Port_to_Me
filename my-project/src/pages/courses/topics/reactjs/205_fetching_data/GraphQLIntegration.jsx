import React, { useState, useEffect } from "react";

const GraphQLIntegration = () => {
  const [data, setData] = useState(null);
  const query = `
    {
      user(id: "1") {
        name
        email
      }
    }
  `;

  useEffect(() => {
    fetch("https://graphql-placeholder.typicode.com/graphql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    })
      .then((response) => response.json())
      .then((result) => setData(result.data.user))
      .catch(() => setData(null));
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">GraphQL Integration</h1>
      
      <p className="mt-4 text-lg">
        <strong>GraphQL</strong> เป็นภาษาสำหรับ Query ข้อมูลที่ถูกออกแบบโดย Facebook ซึ่งให้ความยืดหยุ่นในการดึงข้อมูลมากกว่า REST API
        โดยผู้ใช้สามารถระบุได้ว่าต้องการข้อมูลอะไรเท่านั้น
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ข้อดีของ GraphQL</h2>
      <ul className="list-disc pl-6 mt-2 space-y-2">
        <li>ลดปัญหา Over-fetching และ Under-fetching</li>
        <li>รวมหลายคำขอไว้ในครั้งเดียวได้</li>
        <li>เหมาะกับแอปที่มี UI ซับซ้อน เช่น Mobile หรือ Dashboard</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง GraphQL Query</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`{
  user(id: "1") {
    name
    email
  }
}`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 การเรียก API ด้วย fetch</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`fetch("https://graphql-placeholder.typicode.com/graphql", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "{ user(id: 1) { name email } }" }),
})
  .then(response => response.json())
  .then(data => console.log(data));`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🧪 ข้อมูลที่ดึงมา</h2>
      {data ? (
        <p className="mt-4 text-lg">
          👤 <strong>{data.name}</strong> - {data.email}
        </p>
      ) : (
        <p className="mt-4">Loading...</p>
      )}
    </div>
  );
};

export default GraphQLIntegration;
