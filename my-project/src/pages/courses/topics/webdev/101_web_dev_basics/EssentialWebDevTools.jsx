import React from "react";

const EssentialWebDevTools = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">เครื่องมือสำคัญสำหรับการพัฒนาเว็บ</h1>
      <p>
        นักพัฒนาเว็บไซต์ใช้เครื่องมือหลากหลายเพื่อช่วยในการพัฒนา ทดสอบ และเพิ่มประสิทธิภาพของเว็บแอปพลิเคชัน
        ด้านล่างนี้เป็นเครื่องมือที่สำคัญที่คุณควรรู้จัก
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">1. เครื่องมือเขียนโค้ด (Code Editors)</h2>
      <p>
        โปรแกรมแก้ไขโค้ดยอดนิยม เช่น VS Code, Sublime Text และ Atom มีฟีเจอร์ช่วยในการพัฒนาเว็บ เช่น
        Syntax Highlighting และ Debugging
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">2. เครื่องมือสำหรับนักพัฒนาเว็บ (Browser DevTools)</h2>
      <p>
        เบราว์เซอร์สมัยใหม่ เช่น Chrome และ Firefox มีเครื่องมือช่วยดีบั๊ก JavaScript ตรวจสอบโครงสร้าง DOM
        และวิเคราะห์ประสิทธิภาพของเว็บ
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: เปิด DevTools ใน Chrome</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`คลิกขวาที่หน้าเว็บ → เลือก 'Inspect' → ใช้ 'Console' ในการ Debug`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">3. ระบบควบคุมเวอร์ชัน (Version Control Systems)</h2>
      <p>
        Git เป็นระบบควบคุมเวอร์ชันที่ช่วยติดตามการเปลี่ยนแปลงของโค้ด ทำให้สามารถทำงานร่วมกันกับทีมและจัดการโครงการได้ง่ายขึ้น
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: คำสั่งพื้นฐานของ Git</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`git init
  git add .
  git commit -m "Initial commit"
  git push origin main`}
      </pre>
    </>
  );
};

export default EssentialWebDevTools;