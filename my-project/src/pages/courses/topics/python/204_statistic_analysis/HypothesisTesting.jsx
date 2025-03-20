import React from "react";

const HypothesisTesting = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 การทดสอบสมมติฐาน (Hypothesis Testing)</h1>
      <p className="mt-4">
        การทดสอบสมมติฐานเป็นวิธีทางสถิติที่ใช้ตรวจสอบข้อสันนิษฐานเกี่ยวกับประชากรโดยใช้ข้อมูลตัวอย่าง
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ขั้นตอนของการทดสอบสมมติฐาน</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>กำหนดสมมติฐาน (Hypothesis)</strong>: ระบุสมมติฐานหลัก (H₀) และสมมติฐานทางเลือก (H₁)</li>
        <li><strong>เลือกระดับนัยสำคัญ (Significance Level, α)</strong>: มักใช้ค่า 0.05</li>
        <li><strong>เลือกสถิติที่ใช้ทดสอบ</strong>: เช่น T-test หรือ ANOVA</li>
        <li><strong>คำนวณค่าสถิติ</strong></li>
        <li><strong>เปรียบเทียบค่า p-value</strong> และสรุปผล</li>
      </ul>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. ตัวอย่างการทดสอบ T-Test</h2>
      <p className="mt-2">T-test ใช้เปรียบเทียบค่าเฉลี่ยของสองกลุ่มตัวอย่าง</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import scipy.stats as stats

# ตัวอย่างข้อมูลจากสองกลุ่ม
group1 = [20, 22, 23, 24, 26, 28, 30]
group2 = [22, 24, 25, 26, 27, 29, 31]

# ทดสอบสมมติฐานโดยใช้ T-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การทดสอบ ANOVA</h2>
      <p className="mt-2">ใช้สำหรับเปรียบเทียบค่าเฉลี่ยของมากกว่าสองกลุ่ม</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`group1 = [20, 22, 23, 24, 26, 28, 30]
group2 = [22, 24, 25, 26, 27, 29, 31]
group3 = [18, 19, 20, 22, 23, 25, 27]

f_stat, p_value = stats.f_oneway(group1, group2, group3)
print("F-Statistic:", f_stat)
print("P-Value:", p_value)`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การแปลผลค่า P-Value</h2>
      <p className="mt-2">ค่า P-Value ใช้เพื่อตัดสินใจว่าเราจะปฏิเสธสมมติฐาน H₀ หรือไม่</p>
      <ul className="list-disc ml-5 mt-2">
        <li>หาก <strong>p-value ≤ 0.05</strong> แสดงว่ามีหลักฐานเพียงพอในการปฏิเสธ H₀</li>
        <li>หาก <strong>p-value  005</strong> แสดงว่าไม่มีหลักฐานเพียงพอในการปฏิเสธ H₀</li>
      </ul>
    </div>
  );
};

export default HypothesisTesting;
