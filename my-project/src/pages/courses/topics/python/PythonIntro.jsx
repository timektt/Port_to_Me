import React from "react";

const PythonIntro = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">แนะนำ Python</h1>
      <p className="mt-4">
        Python เป็นภาษาโปรแกรมที่เรียนรู้ได้ง่าย มีความนิยมสูง และเหมาะสำหรับการพัฒนาเว็บ, AI, Data Science ฯลฯ
      </p>
      <h2 className="text-2xl font-semibold mt-6">ตัวอย่างโค้ดแรกของคุณ</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>
          {`print("Hello, Python!")`}
        </code>
      </pre>
    </div>
  );
};

export default PythonIntro;
