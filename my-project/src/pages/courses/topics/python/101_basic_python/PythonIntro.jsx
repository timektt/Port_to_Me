import React from "react";

const PythonIntro = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">แนะนำ Python</h1>
      <p className="mt-4">
        Python เป็นภาษาโปรแกรมที่เรียนรู้ได้ง่าย มีความนิยมสูง และเหมาะสำหรับการพัฒนาเว็บ, AI, Data Science, Machine Learning, 
        การพัฒนาแอปพลิเคชัน, การวิเคราะห์ข้อมูล และอีกมากมาย
      </p>
      
      <h2 className="text-2xl font-semibold mt-6">คุณสมบัติเด่นของ Python</h2>
      <ul className="list-disc pl-6 mt-4">
        <li>มีไวยากรณ์ที่อ่านง่ายและเข้าใจง่าย</li>
        <li>เป็นภาษาที่มีการใช้งานแบบ Dynamic Typing</li>
        <li>รองรับการทำงานแบบ Object-Oriented Programming (OOP)</li>
        <li>สามารถใช้ได้บนหลายแพลตฟอร์ม (Cross-platform)</li>
        <li>มีไลบรารีและเฟรมเวิร์กมากมายสำหรับงานต่าง ๆ</li>
      </ul>
      
      <h2 className="text-2xl font-semibold mt-6">ตัวอย่างโค้ดแรกของคุณ</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>
          {`print("Hello, Python!")`}
        </code>
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">การติดตั้ง Python</h2>
      <p className="mt-4">
        คุณสามารถดาวน์โหลด Python ได้จากเว็บไซต์ทางการที่ <a href="https://www.python.org" className="text-blue-400">python.org</a> และติดตั้งได้บน Windows, macOS และ Linux
      </p>
      <h3 className="text-xl font-medium mt-4">ตรวจสอบเวอร์ชัน Python ที่ติดตั้ง</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>
          {`python --version`}
        </code>
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">Python ใช้ทำอะไรได้บ้าง?</h2>
      <ul className="list-disc pl-6 mt-4">
        <li>การพัฒนาเว็บด้วย Flask และ Django</li>
        <li>การวิเคราะห์ข้อมูลและ Data Science ด้วย Pandas และ NumPy</li>
        <li>Machine Learning และ AI ด้วย TensorFlow และ Scikit-Learn</li>
        <li>การทำงานกับระบบเครือข่ายและ Automation</li>
        <li>การพัฒนาเกมด้วย Pygame</li>
      </ul>
    </div>
  );
};

export default PythonIntro;