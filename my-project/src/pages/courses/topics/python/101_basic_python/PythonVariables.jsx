import React from "react";

const PythonVariables = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Python Variables</h1>
      <p>Variables store data in memory.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`x = 5
y = "Hello"
print(x, y)`}
      </pre>
    </div>
  );
};

export default PythonVariables;
