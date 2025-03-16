import React from "react";

const SetFrozenset = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        Set & Frozenset
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Sets are unordered collections of unique elements.
      </p>

      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        <strong>Frozensets</strong> are immutable sets.
      </p>

      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Example:
      </p>

      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`my_set = {1, 2, 3}
my_frozenset = frozenset([1, 2, 3])`}
        </pre>
      </div>
    </div>
  );
};

export default SetFrozenset;
