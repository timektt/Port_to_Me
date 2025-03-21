import React from "react";

const SeabornDataVisualization = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 การสร้างกราฟข้อมูลด้วย Seaborn</h1>
      <p className="mt-4">
        Seaborn เป็นไลบรารีสำหรับการสร้างกราฟที่สวยงามใน Python
        โดยต่อยอดมาจาก Matplotlib และเหมาะสำหรับงานวิเคราะห์ข้อมูลโดยเฉพาะ
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การติดตั้ง Seaborn</h2>
      <p className="mt-2">ใช้คำสั่งนี้เพื่อติดตั้งไลบรารี:</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`pip install seaborn`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. Scatter Plot — ความสัมพันธ์ระหว่างยอดบิลกับทิป</h2>
      <p className="mt-2">
        กราฟ Scatter Plot ใช้ดูแนวโน้มระหว่างตัวแปร เช่น ยอดบิลกับทิป โดยใช้ <code>hue</code> แยกสีตามวัน
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.title("Total Bill vs Tip by Day")
plt.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. Box Plot — การกระจายยอดบิลตามวันและเพศ</h2>
      <p className="mt-2">
        Box Plot แสดงค่ากลาง ค่ามากสุด-น้อยสุด และ outliers ของยอดบิลแต่ละวัน แยกตามเพศ
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
plt.title("Box Plot of Total Bill by Day and Gender")
plt.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. Heatmap — ความสัมพันธ์ของข้อมูล</h2>
      <p className="mt-2">
        Heatmap แสดง Correlation Matrix เพื่อดูความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลขในชุดข้อมูล
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np

corr = tips.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default SeabornDataVisualization;
