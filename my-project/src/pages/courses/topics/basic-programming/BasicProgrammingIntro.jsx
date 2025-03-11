import React from "react";

const BasicProgrammingIntro = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Introduction to Basic Programming</h1>
      <p>Basic programming concepts include variables, loops, and functions.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`x = 10
while x > 0:
    print(x)
    x -= 1`}
      </pre>
    </div>
  );
};

export default BasicProgrammingIntro;
