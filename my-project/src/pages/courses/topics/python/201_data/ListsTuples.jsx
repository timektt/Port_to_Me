import React from "react";

const ListsTuples = () => {
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold">Lists & Tuples</h1>
      <p className="mt-4">Lists and Tuples are fundamental data structures in Python.</p>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Lists:</strong> Mutable, ordered collections of elements.</li>
        <li><strong>Tuples:</strong> Immutable, ordered collections of elements.</li>
      </ul>
      <p className="mt-4">Example:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`my_list = [1, 2, 3]\nmy_tuple = (1, 2, 3)`}
      </pre>
    </div>
  );
};

export default ListsTuples;
