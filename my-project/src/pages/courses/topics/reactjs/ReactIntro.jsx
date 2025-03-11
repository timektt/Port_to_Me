import React from "react";

const ReactIntro = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Introduction to React.js</h1>
      <p>React is a JavaScript library for building UI components.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`const App = () => <h1>Hello, React!</h1>;`}
      </pre>
    </div>
  );
};

export default ReactIntro;
