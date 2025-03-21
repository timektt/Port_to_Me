import React from "react";

const GraphQLSchemaResolvers = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">🔧 GraphQL Schema & Resolvers</h1>

      <p className="mb-4">
        ใน GraphQL <strong>Schema</strong> ใช้กำหนดชนิดของข้อมูลและ Query ที่สามารถเรียกใช้งานได้
        ส่วน <strong>Resolvers</strong> คือฟังก์ชันที่ใช้จัดการการดึงข้อมูลจริงจากฐานข้อมูลหรือแหล่งข้อมูลอื่น
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่าง Schema</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg text-sm overflow-x-auto">
        <code>{`const schema = buildSchema(\`
  type Query {
    user(id: ID!): User
    allUsers: [User]
  }

  type User {
    id: ID!
    name: String
    email: String
  }
\`);`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 ตัวอย่าง Resolvers</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg text-sm overflow-x-auto">
        <code>{`const users = [
  { id: "1", name: "Alice", email: "alice@example.com" },
  { id: "2", name: "Bob", email: "bob@example.com" },
];

const resolvers = {
  user: ({ id }) => users.find((user) => user.id === id),
  allUsers: () => users,
};`}</code>
      </pre>

      <h2 className="text-2xl font-semibold mt-6 mb-2">📌 รวมกับ Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg text-sm overflow-x-auto">
        <code>{`const express = require("express");
const { graphqlHTTP } = require("express-graphql");
const { buildSchema } = require("graphql");

const app = express();
app.use("/graphql", graphqlHTTP({
  schema,
  rootValue: resolvers,
  graphiql: true
}));

app.listen(3000, () => console.log("GraphQL API running on http://localhost:3000/graphql"));`}</code>
      </pre>

      <p className="mt-6">
        ✅ ตอนนี้คุณสามารถทดสอบ API ได้ที่ <code>http://localhost:3000/graphql</code> ผ่าน GraphiQL หรือ Postman
      </p>
    </div>
  );
};

export default GraphQLSchemaResolvers;
