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
        GraphQL เป็น API Query Language ที่ช่วยให้ดึงข้อมูลได้ยืดหยุ่นมากขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้ GraphQL</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
{`fetch("https://graphql-placeholder.typicode.com/graphql", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "{ user(id: 1) { name email } }" }),
})
  .then(response => response.json())
  .then(data => console.log(data));`}
      </pre>

      {data ? <p className="mt-4">{data.name} ({data.email})</p> : <p>Loading...</p>}
    </div>
  );
};

export default GraphQLIntegration;
