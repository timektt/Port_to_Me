import React from "react";

const CSSPreprocessors = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">CSS Preprocessors (SASS & LESS)</h1>
      <p>
        CSS Preprocessors เช่น SASS และ LESS ช่วยให้การเขียน CSS มีโครงสร้างที่ดีขึ้น โดยสามารถใช้ตัวแปร ฟังก์ชัน และการซ้อนโค้ดได้
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">SASS คืออะไร?</h2>
      <p>
        SASS (Syntactically Awesome Stylesheets) เป็น CSS Preprocessor ที่ช่วยให้สามารถใช้ตัวแปรและมิกซ์อินเพื่อทำให้โค้ด CSS มีประสิทธิภาพมากขึ้น
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ตัวแปรใน SASS</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`$primary-color: blue;

.button {
  background-color: $primary-color;
  padding: 10px;
}`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">LESS คืออะไร?</h2>
      <p>
        LESS (Leaner Style Sheets) เป็น CSS Preprocessor อีกตัวหนึ่งที่มีความคล้ายคลึงกับ SASS แต่มีไวยากรณ์ที่เรียบง่ายกว่า
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ตัวแปรใน LESS</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`@primary-color: blue;

.button {
  background-color: @primary-color;
  padding: 10px;
}`}
      </pre>
    </>
  );
};

export default CSSPreprocessors;
