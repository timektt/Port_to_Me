import React from "react";

const NodeSetup = () => {
  return (
    <div className="p-4 md:p-6">
      <h1 className="text-3xl font-bold">Setting Up Node.js</h1>
      <p className="mt-2 text-lg">
        Node.js is a JavaScript runtime built on Chrome's V8 engine. To install Node.js:
      </p>
      <ol className="list-decimal ml-6 mt-3 text-lg">
        <li>Download from <a href="https://nodejs.org" className="text-blue-400 hover:underline">nodejs.org</a></li>
        <li>Install the downloaded package</li>
        <li>Verify installation: <code className="bg-gray-700 text-white p-1 rounded">node -v</code></li>
      </ol>
    </div>
  );
};

export default NodeSetup;
