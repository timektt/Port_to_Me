import React from "react";

const SeabornDataVisualization = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Seaborn: Data Visualization</h1>
      <p>Seaborn เป็นไลบรารีสำหรับการวิเคราะห์ข้อมูลที่ซับซ้อน</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import seaborn as sns
import matplotlib.pyplot as plt

# สร้างตัวอย่างข้อมูล
tips = sns.load_dataset("tips")

# สร้างกราฟ
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default SeabornDataVisualization;
