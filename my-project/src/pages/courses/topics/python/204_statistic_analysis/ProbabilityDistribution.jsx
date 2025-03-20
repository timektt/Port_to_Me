import React from "react";

const ProbabilityDistribution = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 ความน่าจะเป็นและการแจกแจงข้อมูล (Probability & Distribution)</h1>
      <p className="mt-4">
        ความน่าจะเป็นและการแจกแจงข้อมูลมีบทบาทสำคัญในการวิเคราะห์ข้อมูล โดยใช้เพื่ออธิบายแนวโน้มของข้อมูลและพฤติกรรมของตัวแปรทางสถิติ
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ความน่าจะเป็นพื้นฐาน (Basic Probability)</h2>
      <p className="mt-2">สมการพื้นฐานของความน่าจะเป็น:</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`P(A) = จำนวนเหตุการณ์ที่สนใจ / จำนวนเหตุการณ์ทั้งหมด`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การแจกแจงแบบปกติ (Normal Distribution)</h2>
      <p className="mt-2">ใช้สำหรับโมเดลข้อมูลที่มีการกระจายสมมาตร</p>
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
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การแจกแจงแบบปัวซอง (Poisson Distribution)</h2>
      <p className="mt-2">ใช้สำหรับโมเดลเหตุการณ์ที่เกิดขึ้นในช่วงเวลาที่กำหนด</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`data_poisson = np.random.poisson(lam=3, size=1000)
sns.histplot(data_poisson, kde=False)
plt.show()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การแจกแจงแบบเบอร์นูลลี (Bernoulli Distribution)</h2>
      <p className="mt-2">ใช้สำหรับโมเดลที่มีผลลัพธ์เพียงสองแบบ (เช่น หัว-ก้อย)</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from scipy.stats import bernoulli

# สุ่มค่าความน่าจะเป็น
bernoulli_data = bernoulli.rvs(p=0.5, size=1000)
sns.histplot(bernoulli_data, discrete=True)
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default ProbabilityDistribution;
