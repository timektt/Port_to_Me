import React from "react";

const BuildingGraphQLApi = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🚀 Building a GraphQL API</h1>
      <p className="mb-4">
        คุณสามารถสร้าง GraphQL API ด้วย Node.js และ Express ได้อย่างง่ายดาย โดยใช้ไลบรารี <code>express-graphql</code> และ <code>graphql</code>
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ขั้นตอนติดตั้ง</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mb-4 overflow-x-auto">
        <code>npm install express express-graphql graphql</code>
      </pre>

      <h2 className="text-2xl font-semibold mb-2">📌 โค้ดตัวอย่าง</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
        <code>{`const express = require("express");
const { graphqlHTTP } = require("express-graphql");
const { buildSchema } = require("graphql");

const app = express();

const schema = buildSchema(\`
  type Query {
    hello: String
  }
\`);

const root = {
  hello: () => "Hello, GraphQL!"
};

app.use("/graphql", graphqlHTTP({
  schema: schema,
  rootValue: root,
  graphiql: true
}));

app.listen(4000, () => {
  console.log("GraphQL API running at http://localhost:4000/graphql");
});`}</code>
      </pre>

      <p className="mt-6">
        💡 หลังจากรันเซิร์ฟเวอร์แล้ว ให้เปิดเบราว์เซอร์ที่ <code>http://localhost:4000/graphql</code> เพื่อทดสอบ GraphQL Query ได้เลย
      </p>
    </div>
  );
};

export default BuildingGraphQLApi;
