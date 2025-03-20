import React from "react";

const ES6ModernJS = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">ES6+ และ JavaScript สมัยใหม่</h1>
      <p>
        ES6 (ECMAScript 2015) และเวอร์ชันใหม่กว่าได้เพิ่มฟีเจอร์ที่ช่วยให้การเขียน JavaScript มีประสิทธิภาพมากขึ้น อ่านง่ายขึ้น และลดข้อผิดพลาดที่อาจเกิดขึ้น
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ let และ const</h2>
      <p>
        ES6 แนะนำการใช้ <code>let</code> และ <code>const</code> แทน <code>var</code> เพื่อช่วยในการจัดการตัวแปรให้ปลอดภัยมากขึ้น
      </p>
      <ul className="list-disc pl-6 mb-4">
        <li><strong>let</strong> - ใช้สำหรับตัวแปรที่สามารถเปลี่ยนค่าได้</li>
        <li><strong>const</strong> - ใช้สำหรับค่าคงที่ที่ไม่สามารถเปลี่ยนแปลงได้</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ let และ const</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`let name = "Alice";
const age = 30;
console.log(name, age);`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Arrow Functions</h2>
      <p>
        Arrow Functions เป็นรูปแบบใหม่ของฟังก์ชันที่ช่วยให้การเขียนโค้ดมีความกระชับขึ้น โดยใช้เครื่องหมาย <code>=&gt;</code> แทน <code>function</code>
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Arrow Function</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const greet = (name) => \`Hello, \${name}!\`;
console.log(greet("Bob"));`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Template Literals</h2>
      <p>
        Template Literals ช่วยให้สามารถแทรกตัวแปรลงในสตริงได้สะดวกขึ้น โดยใช้เครื่องหมายแบ็คทิค <code>`</code> แทนเครื่องหมายอัญประกาศปกติ
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Template Literals</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const name = "Alice";
const message = \`สวัสดี, \${name}! ยินดีต้อนรับ!\`;
console.log(message);`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Destructuring Assignment</h2>
      <p>
        Destructuring เป็นฟีเจอร์ที่ช่วยให้สามารถดึงค่าจากอาเรย์หรืออ็อบเจ็กต์มาเก็บไว้ในตัวแปรได้ง่ายขึ้น
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Destructuring Array</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const numbers = [1, 2, 3];
const [first, second, third] = numbers;
console.log(first, second, third); // 1, 2, 3`}
      </pre>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Destructuring Object</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const user = { name: "Alice", age: 25 };
const { name, age } = user;
console.log(name, age); // Alice, 25`}
      </pre>
    </>
  );
};

export default ES6ModernJS;