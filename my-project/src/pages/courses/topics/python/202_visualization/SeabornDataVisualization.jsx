import React from "react";

const SeabornDataVisualization = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 การสร้างกราฟข้อมูลด้วย Seaborn</h1>
      <p className="mt-4">
        Seaborn เป็นไลบรารีที่ช่วยให้การสร้างกราฟและวิเคราะห์ข้อมูลทำได้ง่ายขึ้น โดยสามารถใช้ร่วมกับ Matplotlib เพื่อสร้าง Visualization ที่สวยงามและเข้าใจง่าย
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การติดตั้ง Seaborn</h2>
      <p className="mt-2">ก่อนใช้งาน Seaborn ต้องติดตั้งไลบรารีก่อน โดยใช้คำสั่ง:</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`pip install seaborn`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การสร้าง Scatter Plot</h2>
      <p className="mt-2">ใช้ Seaborn เพื่อสร้าง Scatter Plot แสดงความสัมพันธ์ระหว่างค่า Total Bill และค่า Tip</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import seaborn as sns
import matplotlib.pyplot as plt

# โหลดตัวอย่างข้อมูล
tips = sns.load_dataset("tips")

# สร้าง Scatter Plot
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.show()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การสร้าง Box Plot</h2>
      <p className="mt-2">ใช้ Box Plot เพื่อแสดงการกระจายของข้อมูลในแต่ละหมวดหมู่</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
plt.show()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การสร้าง Heatmap</h2>
      <p className="mt-2">Heatmap ใช้แสดงความสัมพันธ์ของข้อมูลในรูปแบบของสี</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np

# คำนวณ Correlation Matrix
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default SeabornDataVisualization;
