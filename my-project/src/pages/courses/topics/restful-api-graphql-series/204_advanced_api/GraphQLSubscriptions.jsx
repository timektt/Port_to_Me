import React from "react";

const GraphQLSubscriptions = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">📡 GraphQL Subscriptions (Real-time API)</h1>
      <p className="mb-4">
        GraphQL Subscriptions ช่วยให้สามารถทำงานแบบ <strong>real-time</strong> ได้โดยใช้ WebSockets ร่วมกับ PubSub
      </p>

      <h2 className="text-xl font-semibold mt-4">📌 ตัวอย่างการใช้งานแบบครบ</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
        <code>{`const { GraphQLServer, PubSub } = require("graphql-yoga");
const pubsub = new PubSub();

const typeDefs = \`
  type Message {
    text: String!
  }

  type Query {
    info: String!
  }

  type Mutation {
    sendMessage(text: String!): Message
  }

  type Subscription {
    newMessage: Message
  }
\`;

const resolvers = {
  Query: {
    info: () => "GraphQL Subscriptions Example",
  },
  Mutation: {
    sendMessage: (_, { text }) => {
      const message = { text };
      pubsub.publish("NEW_MESSAGE", { newMessage: message });
      return message;
    },
  },
  Subscription: {
    newMessage: {
      subscribe: () => pubsub.asyncIterator("NEW_MESSAGE"),
    },
  },
};

const server = new GraphQLServer({
  typeDefs,
  resolvers,
  context: { pubsub },
});

server.start(() => console.log("🚀 Server running at http://localhost:4000"));
`}</code>
      </pre>

      <p className="mt-4 text-gray-700 dark:text-gray-300">
        ✅ เปิดใช้งานที่ <code>http://localhost:4000</code> และทดสอบ Subscriptions ได้ผ่าน GraphQL Playground
      </p>
    </div>
  );
};

export default GraphQLSubscriptions;
