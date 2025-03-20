import React from "react";

const JavaScriptBasics = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">พื้นฐาน JavaScript</h1>
      <p>
        JavaScript เป็นภาษาที่ใช้ในการพัฒนาเว็บเพื่อทำให้เว็บไซต์มีความโต้ตอบ (Interactive) และสามารถทำงานร่วมกับผู้ใช้ได้
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การประกาศตัวแปร</h2>
      <p>
        JavaScript มี 3 วิธีหลักในการประกาศตัวแปร ได้แก่ <code>var</code>, <code>let</code>, และ <code>const</code>
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การประกาศตัวแปรใน JavaScript</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`let name = "John";
const age = 25;
var city = "Bangkok";

console.log(name, age, city);`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">เงื่อนไขและการควบคุมการทำงาน</h2>
      <p>
        JavaScript ใช้คำสั่งเงื่อนไขเช่น <code>if...else</code> และ <code>switch</code> ในการควบคุมลำดับการทำงานของโปรแกรม
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: เงื่อนไขใน JavaScript</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`let score = 80;
if (score >= 50) {
  console.log("สอบผ่าน");
} else {
  console.log("สอบไม่ผ่าน");
}`}
      </pre>
    </>
  );
};

export default JavaScriptBasics;
