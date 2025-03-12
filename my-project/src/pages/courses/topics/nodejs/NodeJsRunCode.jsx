import React from "react";

const NodeJsRunCode = () => {
  return (
    <div className="p-4 md:p-6">
      <h1 className="text-3xl font-bold">Running JavaScript in Node.js</h1>
      <p className="mt-2 text-lg">To run JavaScript using Node.js, follow these steps:</p>
      <ol className="list-decimal ml-6 mt-3 text-lg">
        <li>Create a new file: <code className="bg-gray-700 text-white p-1 rounded">script.js</code></li>
        <li>Add JavaScript code:
          <pre className="bg-gray-800 text-white p-2 rounded mt-2">{`console.log('Hello, Node.js!');`}</pre>
        </li>
        <li>Run in terminal: <code className="bg-gray-700 text-white p-1 rounded">node script.js</code></li>
      </ol>
    </div>
  );
};

export default NodeJsRunCode;
