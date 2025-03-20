import React from "react";

const AngularIntro = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน Angular</h1>
      <p>
        Angular เป็นเฟรมเวิร์ก JavaScript แบบ Full-Featured ที่พัฒนาโดย Google เหมาะสำหรับพัฒนาเว็บแอปพลิเคชันขนาดใหญ่
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติหลักของ Angular</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>ใช้ TypeScript เป็นภาษาหลัก</li>
        <li>รองรับ Component-based Architecture</li>
        <li>มีระบบ Dependency Injection</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Angular Component พื้นฐาน</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  template: '<h1>ยินดีต้อนรับสู่ Angular!</h1>'
})
export class AppComponent {}`}
      </pre>
    </>
  );
};

export default AngularIntro;
