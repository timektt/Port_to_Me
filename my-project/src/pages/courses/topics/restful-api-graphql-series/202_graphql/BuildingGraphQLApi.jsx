import React from "react";

const BuildingGraphQLApi = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Building a GraphQL API</h1>
      <p className="mb-4">
        GraphQL API สามารถสร้างได้โดยใช้ Node.js และ Express โดยใช้ไลบรารี <code>express-graphql</code> และ <code>graphql</code>.
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const { graphqlHTTP } = require("express-graphql");
const { buildSchema } = require("graphql");

const schema = buildSchema(\`
  type Query {
    hello: String
  }
\`);

const root = { hello: () => "Hello, GraphQL!" };

app.use("/graphql", graphqlHTTP({
  schema: schema,
  rootValue: root,
  graphiql: true
}));`}
      </pre>
    </div>
  );
};

export default BuildingGraphQLApi;
