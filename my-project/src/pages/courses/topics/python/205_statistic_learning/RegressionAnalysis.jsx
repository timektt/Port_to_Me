import React from "react";

const RegressionAnalysis = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📈 การวิเคราะห์ถดถอย (Regression Analysis)</h1>
      <p className="mt-4">
        การวิเคราะห์ถดถอยเป็นเทคนิคสำคัญในสถิติและ Machine Learning ที่ใช้ในการสร้างโมเดลความสัมพันธ์ระหว่างตัวแปร
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การถดถอยเชิงเส้น (Linear Regression)</h2>
      <p className="mt-2">ใช้สำหรับสร้างโมเดลเส้นตรงที่สัมพันธ์ระหว่างตัวแปรอิสระ (X) และตัวแปรตาม (Y)</p>
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
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การถดถอยเชิงพหุ (Multiple Linear Regression)</h2>
      <p className="mt-2">ใช้เมื่อต้องการพิจารณาตัวแปรอิสระมากกว่าหนึ่งตัว</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from sklearn.linear_model import LinearRegression

# ข้อมูลตัวอย่าง
X_multi = np.array([[1, 2], [2, 3], [3, 5], [4, 7], [5, 11]])
y_multi = np.array([2, 4, 5, 4, 5])

# สร้างโมเดล
multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)

print("ค่าสัมประสิทธิ์ของตัวแปรอิสระ:", multi_model.coef_)
print("ค่า Intercept:", multi_model.intercept_)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การถดถอยโลจิสติก (Logistic Regression)</h2>
      <p className="mt-2">ใช้สำหรับปัญหาการจำแนกประเภท เช่น ทำนายว่าผู้ใช้จะซื้อสินค้าหรือไม่ (Yes/No)</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from sklearn.linear_model import LogisticRegression

# ตัวอย่างข้อมูล
X_logistic = np.array([[0], [1], [2], [3], [4]])
y_logistic = np.array([0, 0, 1, 1, 1])

# สร้างโมเดล
logistic_model = LogisticRegression()
logistic_model.fit(X_logistic, y_logistic)

print("ค่าสัมประสิทธิ์:", logistic_model.coef_)
print("ค่า Intercept:", logistic_model.intercept_)`}</code>
      </pre>
    </div>
  );
};

export default RegressionAnalysis;
