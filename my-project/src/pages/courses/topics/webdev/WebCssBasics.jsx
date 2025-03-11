import React from "react";

const WebCssBasics = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">CSS Basics</h1>
      <p>CSS is used for styling web pages.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`h1 {
  color: blue;
  font-size: 24px;
}`}
      </pre>
    </div>
  );
};

export default WebCssBasics;
