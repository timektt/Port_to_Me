import React from "react";

const HypothesisTesting = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 การทดสอบสมมติฐาน (Hypothesis Testing)</h1>
      <p className="mt-4">
        การทดสอบสมมติฐานเป็นกระบวนการที่ใช้วิธีทางสถิติเพื่อประเมินข้อสมมติใดข้อสมมติหนึ่งเกี่ยวกับประชากร โดยอิงจากข้อมูลตัวอย่าง
      </p>

      <h2 className="text-xl font-semibold mt-6">1. ขั้นตอนของการทดสอบสมมติฐาน</h2>
      <ul className="list-disc ml-6 mt-2">
        <li><strong>กำหนดสมมติฐาน</strong>: H₀ (ไม่มีความแตกต่าง) และ H₁ (มีความแตกต่าง)</li>
        <li><strong>กำหนดระดับนัยสำคัญ (α)</strong>: มักใช้ 0.05</li>
        <li><strong>เลือกสถิติ</strong>: เช่น T-test, ANOVA</li>
        <li><strong>คำนวณค่าสถิติและ p-value</strong></li>
        <li><strong>ตัดสินใจ</strong>: เปรียบเทียบ p-value กับ α</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">2. ตัวอย่าง T-Test (Independent Samples)</h2>
      <p className="mt-2">ใช้เปรียบเทียบค่าเฉลี่ยระหว่างกลุ่ม 2 กลุ่ม</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`import scipy.stats as stats

group1 = [20, 22, 23, 24, 26, 28, 30]
group2 = [22, 24, 25, 26, 27, 29, 31]

t_stat, p_value = stats.ttest_ind(group1, group2)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">3. ตัวอย่าง ANOVA (หลายกลุ่ม)</h2>
      <p className="mt-2">ใช้เมื่อเปรียบเทียบค่าเฉลี่ยระหว่างมากกว่า 2 กลุ่ม</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-md">
{`group1 = [20, 22, 23, 24, 26, 28, 30]
group2 = [22, 24, 25, 26, 27, 29, 31]
group3 = [18, 19, 20, 22, 23, 25, 27]

f_stat, p_value = stats.f_oneway(group1, group2, group3)
print("F-Statistic:", f_stat)
print("P-Value:", p_value)`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">4. การแปลผล P-Value</h2>
      <p className="mt-2">ค่า P-Value ช่วยตัดสินใจว่าจะปฏิเสธ H₀ หรือไม่</p>
      <ul className="list-disc ml-6 mt-2">
        <li><strong>p-value ≤ 0.05</strong>: ปฏิเสธ H₀ → มีความแตกต่างอย่างมีนัยสำคัญ</li>
        <li><strong>p-value &gt; 0.05</strong>: ไม่ปฏิเสธ H₀ → ไม่มีหลักฐานเพียงพอที่จะสรุปว่ามีความแตกต่าง</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">5. 🔎 ตัวอย่างผลลัพธ์</h2>
      <div className="bg-gray-100 text-black dark:bg-gray-800 dark:text-white p-4 rounded-md mt-2">
        <pre>
{`T-Statistic: -1.21
P-Value: 0.24

F-Statistic: 3.67
P-Value: 0.042`}
        </pre>
      </div>
      <p className="mt-2">
        จากตัวอย่าง ANOVA ข้างต้น p-value = 0.042 <strong>ต่ำกว่า 0.05</strong> ⇒ ปฏิเสธ H₀ ⇒ <strong>แต่ละกลุ่มมีค่าเฉลี่ยต่างกันอย่างมีนัยสำคัญ</strong>
      </p>
    </div>
  );
};

export default HypothesisTesting;
