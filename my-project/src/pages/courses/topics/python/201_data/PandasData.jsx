import React from "react";

const PandasData = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        การจัดการข้อมูลด้วย Pandas
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Pandas is a powerful library for data manipulation and analysis.
      </p>

      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        Example:
      </p>

      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`import pandas as pd

data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)`}
        </pre>
      </div>
    </div>
  );
};

export default PandasData;
