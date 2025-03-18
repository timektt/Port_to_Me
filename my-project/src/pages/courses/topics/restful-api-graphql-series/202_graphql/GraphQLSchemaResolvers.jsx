import React from "react";

const GraphQLSchemaResolvers = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">GraphQL Schema & Resolvers</h1>
      <p className="mb-4">
        GraphQL ใช้ <strong>Schema</strong> เพื่อกำหนดโครงสร้างของข้อมูล และใช้ <strong>Resolvers</strong> เพื่อดึงข้อมูลจริงจาก Database หรือ API.
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const schema = buildSchema(\`
  type Query {
    user(id: ID!): User
  }

  type User {
    id: ID!
    name: String
    email: String
  }
\`);

const resolvers = {
  user: ({ id }) => getUserFromDB(id)
};`}
      </pre>
    </div>
  );
};

export default GraphQLSchemaResolvers;
