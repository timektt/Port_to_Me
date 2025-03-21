import React from "react";

const RegressionAnalysis = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📈 การวิเคราะห์ถดถอย (Regression Analysis)</h1>
      <p className="mt-4">
        การวิเคราะห์ถดถอยเป็นเทคนิคในสถิติและ Machine Learning ที่ใช้สร้างความสัมพันธ์ระหว่างตัวแปรอิสระ (X) กับตัวแปรตาม (Y)
      </p>

      {/* ✅ Linear Regression */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การถดถอยเชิงเส้น (Linear Regression)</h2>
      <p className="mt-2">เหมาะสำหรับข้อมูลที่มีความสัมพันธ์เป็นเส้นตรง</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("R² Score:", r2_score(y, y_pred))  # ความแม่นยำของโมเดล

plt.scatter(X, y, color="blue")
plt.plot(X, y_pred, color="red")
plt.show()`}</code>
      </pre>

      {/* ✅ Multiple Regression */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การถดถอยเชิงพหุ (Multiple Linear Regression)</h2>
      <p className="mt-2">ใช้เมื่อมีตัวแปรอิสระหลายตัว</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from sklearn.linear_model import LinearRegression

X_multi = np.array([[1, 2], [2, 3], [3, 5], [4, 7], [5, 11]])
y_multi = np.array([2, 4, 5, 4, 5])

multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)

print("ค่าสัมประสิทธิ์:", multi_model.coef_)
print("Intercept:", multi_model.intercept_)`}</code>
      </pre>

      {/* ✅ Logistic Regression */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การถดถอยโลจิสติก (Logistic Regression)</h2>
      <p className="mt-2">ใช้สำหรับปัญหาการจำแนกประเภท เช่น ใช่/ไม่ใช่</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from sklearn.linear_model import LogisticRegression

X_logistic = np.array([[0], [1], [2], [3], [4]])
y_logistic = np.array([0, 0, 1, 1, 1])

logistic_model = LogisticRegression()
logistic_model.fit(X_logistic, y_logistic)

print("ค่าสัมประสิทธิ์:", logistic_model.coef_)
print("Intercept:", logistic_model.intercept_)
print("ทำนายค่า:", logistic_model.predict([[1.5], [3.5]]))`}</code>
      </pre>
    </div>
  );
};

export default RegressionAnalysis;
