import React from "react";

const ReactVsFrameworks = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">React เทียบกับ Framework อื่น</h1>
      <p className="mt-4 text-lg">
        React เปรียบเทียบกับ Angular และ Vue.js โดยเน้นความเร็ว การใช้งาน และแนวคิดของแต่ละ Framework
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตารางเปรียบเทียบ</h2>

      {/* ✅ เพิ่มตารางเปรียบเทียบ */}
      <div className="overflow-x-auto mt-4">
        <table className="w-full border-collapse border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
          <thead>
            <tr className="bg-gray-300 dark:bg-gray-700">
              <th className="p-3 border">คุณสมบัติ</th>
              <th className="p-3 border">React</th>
              <th className="p-3 border">Angular</th>
              <th className="p-3 border">Vue.js</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-3 border">รูปแบบ</td>
              <td className="p-3 border">Library</td>
              <td className="p-3 border">Framework</td>
              <td className="p-3 border">Framework</td>
            </tr>
            <tr>
              <td className="p-3 border">แนวคิดหลัก</td>
              <td className="p-3 border">Component-based</td>
              <td className="p-3 border">MVVM (Model-View-ViewModel)</td>
              <td className="p-3 border">Component-based</td>
            </tr>
            <tr>
              <td className="p-3 border">การเรียนรู้</td>
              <td className="p-3 border">ง่าย (JSX)</td>
              <td className="p-3 border">ซับซ้อน (TypeScript)</td>
              <td className="p-3 border">ง่าย (Template-based)</td>
            </tr>
            <tr>
              <td className="p-3 border">ขนาดไฟล์</td>
              <td className="p-3 border">เล็ก</td>
              <td className="p-3 border">ใหญ่</td>
              <td className="p-3 border">เล็ก</td>
            </tr>
            <tr>
              <td className="p-3 border">การจัดการสถานะ</td>
              <td className="p-3 border">Redux, Context API</td>
              <td className="p-3 border">RxJS, NgRx</td>
              <td className="p-3 border">Vuex, Pinia</td>
            </tr>
            <tr>
              <td className="p-3 border">การใช้งานร่วมกับ Backend</td>
              <td className="p-3 border">ง่าย</td>
              <td className="p-3 border">ซับซ้อน</td>
              <td className="p-3 border">ง่าย</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-2xl font-semibold mt-6">🎯 สรุป</h2>
      <p className="mt-4">
        - **React:** ใช้งานง่าย ยืดหยุ่น ควบคุมการจัดการสถานะได้ดี<br />
        - **Angular:** มีโครงสร้างแน่นหนา เหมาะสำหรับระบบขนาดใหญ่<br />
        - **Vue.js:** ใช้งานง่ายและเบา เหมาะกับโปรเจคที่ต้องการความเร็ว
      </p>
    </div>
  );
};

export default ReactVsFrameworks;
