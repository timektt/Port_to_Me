import React from "react";

const ReactComponents = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">React Components</h1>
      <p>Components are reusable UI elements in React.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`const Hello = () => <h1>Hello, React!</h1>;

export default Hello;`}
      </pre>
    </div>
  );
};

export default ReactComponents;
