import React from "react";

const PythonInputFunction = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Python Input & Functions</h1>
      <p>Functions allow code reuse.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`def greet(name):
    print(f"Hello, {name}")

greet("Alice")`}
      </pre>
    </div>
  );
};

export default PythonInputFunction;
