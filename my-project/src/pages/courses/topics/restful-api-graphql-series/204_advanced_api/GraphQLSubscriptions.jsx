import React from "react";

const GraphQLSubscriptions = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">GraphQL Subscriptions (Real-time API)</h1>
      <p className="mb-4">
        GraphQL Subscriptions ช่วยให้สามารถทำงานแบบ real-time ได้โดยใช้ WebSockets.
      </p>
      <h2 className="text-xl font-semibold mt-4">ตัวอย่างการใช้ GraphQL Subscriptions</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const { GraphQLServer, PubSub } = require("graphql-yoga");
const pubsub = new PubSub();

const resolvers = {
  Subscription: {
    newMessage: {
      subscribe: () => pubsub.asyncIterator("NEW_MESSAGE"),
    },
  },
};

const server = new GraphQLServer({ typeDefs, resolvers, context: { pubsub } });
server.start(() => console.log("Server is running..."));`}
      </pre>
    </div>
  );
};

export default GraphQLSubscriptions;
