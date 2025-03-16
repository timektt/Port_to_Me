import React from "react";

const PythonLeetcode = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        Python Leetcode Challenge
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Let's solve a basic coding challenge.
      </p>

      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]`}
        </pre>
      </div>
    </div>
  );
};

export default PythonLeetcode;
