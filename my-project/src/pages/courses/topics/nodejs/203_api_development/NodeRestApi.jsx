import React from "react";

const NodeRestApi = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Building a REST API with Node.js</h1>
      <p>REST APIs allow applications to communicate over the web.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`const express = require('express');
const app = express();

app.get('/hello', (req, res) => {
  res.send('Hello, API!');
});

app.listen(3000, () => console.log('Server is running'));`}
      </pre>
    </div>
  );
};

export default NodeRestApi;
