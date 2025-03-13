import React from "react";

const PythonLeetcode = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Python Leetcode Challenge</h1>
      <p>Let's solve a basic coding challenge.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]`}
      </pre>
    </div>
  );
};

export default PythonLeetcode;
