import React from "react";

const VueIntro = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน Vue.js</h1>
      <p>
        Vue.js เป็นเฟรมเวิร์ก JavaScript ที่มีขนาดเล็ก ใช้งานง่าย และเหมาะสำหรับการพัฒนา UI แบบ Interactive
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติหลักของ Vue.js</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>โครงสร้างที่ยืดหยุ่นและเรียนรู้ได้ง่าย</li>
        <li>รองรับการทำงานแบบ Component-based</li>
        <li>รองรับ Two-way Data Binding</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Vue Component พื้นฐาน</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<template>
  <h1>{{ message }}</h1>
</template>

<script>
export default {
  data() {
    return {
      message: "ยินดีต้อนรับสู่ Vue!"
    };
  }
};
</script>`}
      </pre>
    </>
  );
};

export default VueIntro;
