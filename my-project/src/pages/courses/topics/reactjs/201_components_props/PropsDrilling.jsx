import React from "react";

const PropsDrilling = () => {
  return (
    <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-2xl font-bold text-green-600">Props & Prop Drilling</h1>
      <p className="mt-4">
        <strong>Props</strong> (Properties) เป็นค่าที่สามารถส่งจาก Component หนึ่งไปยังอีก Component หนึ่งได้ 
        ช่วยให้สามารถใช้ Components ร่วมกันได้ง่ายขึ้น
      </p>
      
      <h2 className="text-xl font-bold mt-6">✅ การใช้ Props</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`const Greeting = ({ name }) => {
  return <h2>สวัสดี {name}!</h2>;
};

const App = () => {
  return <Greeting name="React" />;
};`}
      </pre>
      
      <h2 className="text-xl font-bold mt-6">🔍 Prop Drilling คืออะไร?</h2>
      <p className="mt-2">
        Prop Drilling คือปัญหาที่เกิดขึ้นเมื่อ Props ถูกส่งต่อผ่านหลาย ๆ Component 
        ทำให้โค้ดซับซ้อนและยากต่อการดูแล
      </p>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`const ComponentA = ({ name }) => <ComponentB name={name} />;
const ComponentB = ({ name }) => <ComponentC name={name} />;
const ComponentC = ({ name }) => <h2>สวัสดี {name}!</h2>;

const App = () => {
  return <ComponentA name="React" />;
};`}
      </pre>
      
      <h2 className="text-xl font-bold mt-6">🚀 แก้ Prop Drilling ด้วย Context API</h2>
      <p className="mt-2">
        React Context API สามารถใช้เพื่อแชร์ข้อมูลระหว่าง Components โดยไม่ต้องส่ง Props ลงไปทีละขั้นตอน
      </p>
    </div>
  );
};

export default PropsDrilling;
