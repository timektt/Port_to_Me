import React from "react";

const FunctionalClassComponents = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 shadow-lg rounded-lg border">
      <h1 className="text-3xl font-bold mb-4">Functional & Class Components ใน React</h1>
      <p className="mb-4">
        ใน React มีสองรูปแบบหลักของ Component ที่ใช้ในการพัฒนา UI ได้แก่:
      </p>
      
      <h2 className="text-2xl font-semibold mb-2">✅ Functional Components</h2>
      <p className="mb-4">
        เป็น Component ที่สร้างขึ้นโดยใช้ฟังก์ชันปกติของ JavaScript และใช้กับ React Hooks ได้โดยตรง
      </p>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`function Greeting() {
  return <h1>สวัสดีจาก Functional Component!</h1>;
}`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6 mb-2">✅ Class Components</h2>
      <p className="mb-4">
        เป็น Component ที่สร้างขึ้นโดยใช้ JavaScript Class และมี lifecycle methods เช่น componentDidMount()
      </p>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`class Greeting extends React.Component {
  render() {
    return <h1>สวัสดีจาก Class Component!</h1>;
  }
}`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">🔥 ควรใช้แบบไหนดี?</h2>
      <p>
        ในปัจจุบัน Functional Components เป็นที่นิยมมากกว่า เนื่องจากใช้งานง่าย และรองรับ React Hooks 
        ซึ่งช่วยให้จัดการ state และ side effects ได้อย่างมีประสิทธิภาพ
      </p>
    </div>
  );
};

export default FunctionalClassComponents;
