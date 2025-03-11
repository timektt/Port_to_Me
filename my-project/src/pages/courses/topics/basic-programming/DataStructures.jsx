import React from "react";

const DataStructures = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Introduction to Data Structures</h1>
      <p>Data structures help store and organize data efficiently.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`stack = []
stack.append(1)
stack.append(2)
stack.pop()  # Removes 2`}
      </pre>
    </div>
  );
};

export default DataStructures;
