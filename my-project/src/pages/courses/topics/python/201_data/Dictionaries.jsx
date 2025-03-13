import React from "react";

const Dictionaries = () => {
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold">Dictionaries</h1>
      <p className="mt-4">Dictionaries are key-value pairs in Python.</p>
      <p className="mt-2">Example:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`my_dict = {"name": "John", "age": 25}\nprint(my_dict["name"])`}
      </pre>
    </div>
  );
};

export default Dictionaries;
