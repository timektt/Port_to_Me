import React from "react";

const PandasData = () => {
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold">การจัดการข้อมูลด้วย Pandas</h1>
      <p className="mt-4">Pandas is a powerful library for data manipulation and analysis.</p>
      <p className="mt-2">Example:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`import pandas as pd\n\ndata = {"Name": ["Alice", "Bob"], "Age": [25, 30]}\ndf = pd.DataFrame(data)\nprint(df)`}
      </pre>
    </div>
  );
};

export default PandasData;
