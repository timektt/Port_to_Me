import React from "react";

const ProbabilityDistribution = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Probability & Distribution</h1>
      <p>ความน่าจะเป็นและการแจกแจงข้อมูลมีบทบาทสำคัญในการวิเคราะห์ข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# สร้างข้อมูลจำลองจาก Normal Distribution
data = np.random.normal(loc=50, scale=10, size=1000)

# สร้างกราฟแจกแจงความน่าจะเป็น
sns.histplot(data, kde=True)
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default ProbabilityDistribution;
