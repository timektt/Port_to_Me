import React from "react";

const NodeAsync = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Asynchronous Programming in Node.js</h1>
      <p>Asynchronous code helps improve performance in Node.js.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`setTimeout(() => {
  console.log("Hello after 2 seconds");
}, 2000);`}
      </pre>
    </div>
  );
};

export default NodeAsync;
