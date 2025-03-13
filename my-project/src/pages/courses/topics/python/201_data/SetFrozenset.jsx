import React from "react";

const SetFrozenset = () => {
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold">Set & Frozenset</h1>
      <p className="mt-4">Sets are unordered collections of unique elements.</p>
      <p className="mt-2"><strong>Frozensets</strong> are immutable sets.</p>
      <p className="mt-2">Example:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`my_set = {1, 2, 3}\nmy_frozenset = frozenset([1, 2, 3])`}
      </pre>
    </div>
  );
};

export default SetFrozenset;
