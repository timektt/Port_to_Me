import React from "react";

const HypothesisTesting = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Hypothesis Testing</h1>
      <p>การทดสอบสมมติฐานเป็นวิธีทางสถิติที่ใช้ตรวจสอบข้อสันนิษฐาน</p>
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
    </div>
  );
};

export default HypothesisTesting;
