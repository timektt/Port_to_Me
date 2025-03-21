import React from "react";

const ProbabilityDistribution = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">
        📊 ความน่าจะเป็นและการแจกแจงข้อมูล (Probability & Distribution)
      </h1>

      <p className="mt-4">
        ความน่าจะเป็นและการแจกแจงข้อมูลเป็นหัวใจสำคัญของการวิเคราะห์ทางสถิติ
        และมีบทบาทสำคัญในการเรียนรู้ของ Machine Learning เช่น การสร้างโมเดล, การประเมินผล และการทำ Inference
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ความน่าจะเป็นพื้นฐาน (Basic Probability)</h2>
      <p className="mt-2">
        ความน่าจะเป็น (Probability) คือโอกาสที่เหตุการณ์หนึ่งจะเกิดขึ้น โดยมีค่าอยู่ระหว่าง 0 ถึง 1
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`P(A) = จำนวนเหตุการณ์ที่สนใจ / จำนวนเหตุการณ์ทั้งหมด`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การแจกแจงแบบปกติ (Normal Distribution)</h2>
      <p className="mt-2">
        ใช้สำหรับข้อมูลที่มีการกระจายสมมาตรรอบค่าเฉลี่ย เช่น ความสูง, น้ำหนัก
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.normal(loc=50, scale=10, size=1000)
sns.histplot(data, kde=True, color="skyblue")
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การแจกแจงแบบปัวซอง (Poisson Distribution)</h2>
      <p className="mt-2">
        เหมาะสำหรับเหตุการณ์ที่เกิดขึ้นแบบนับจำนวน เช่น จำนวนการโทรเข้าศูนย์บริการต่อชั่วโมง
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`data_poisson = np.random.poisson(lam=3, size=1000)
sns.histplot(data_poisson, discrete=True, color="orange")
plt.title("Poisson Distribution")
plt.xlabel("Events")
plt.ylabel("Frequency")
plt.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การแจกแจงแบบเบอร์นูลลี (Bernoulli Distribution)</h2>
      <p className="mt-2">
        ใช้สำหรับข้อมูลที่มีเพียงสองทางเลือก เช่น สำเร็จ/ล้มเหลว, หัว/ก้อย
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from scipy.stats import bernoulli

bernoulli_data = bernoulli.rvs(p=0.5, size=1000)
sns.histplot(bernoulli_data, discrete=True, color="green")
plt.title("Bernoulli Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.xticks([0, 1], ["Fail", "Success"])
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default ProbabilityDistribution;
