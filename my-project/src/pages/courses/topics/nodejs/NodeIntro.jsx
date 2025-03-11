import React from "react";

const NodeIntro = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Introduction to Node.js</h1>
      <p>Node.js is a runtime environment for executing JavaScript outside the browser.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`console.log("Hello from Node.js!");`}
      </pre>
    </div>
  );
};

export default NodeIntro;
