import React from "react";

const PythonControlStructure = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Python Control Structures</h1>
      <p>Control structures manage the flow of a program.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`if x > 0:
    print("Positive number")
else:
    print("Negative number")`}
      </pre>
    </div>
  );
};

export default PythonControlStructure;
