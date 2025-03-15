import React from "react";

const RegressionAnalysis = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Regression Analysis</h1>
      <p>การวิเคราะห์ถดถอยใช้ในการสร้างโมเดลความสัมพันธ์ระหว่างตัวแปร</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# สร้างข้อมูลตัวอย่าง
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

# ทำนายค่า
y_pred = model.predict(X)

# แสดงกราฟ
plt.scatter(X, y, color="blue")
plt.plot(X, y_pred, color="red")
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default RegressionAnalysis;
