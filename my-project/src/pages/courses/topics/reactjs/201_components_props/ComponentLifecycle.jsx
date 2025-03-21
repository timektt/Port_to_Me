import React, { Component } from "react";

const ComponentLifecycle = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold text-purple-600 dark:text-purple-400">
        Component Lifecycle Methods
      </h1>
      <p className="mt-4">
        <strong>Component Lifecycle</strong> คือกระบวนการที่เกิดขึ้นใน React Component ตั้งแต่เริ่มต้นจนถึงการถูกลบออกจาก DOM โดยสามารถแบ่งออกเป็น 3 ช่วงหลัก ๆ
      </p>

      <h2 className="text-xl font-bold mt-6">1. Mounting (เริ่มต้น)</h2>
      <p className="mt-2">เมื่อ Component ถูกสร้างขึ้นมาใน DOM</p>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`class MyComponent extends Component {
  componentDidMount() {
    console.log("Component ถูก mount แล้ว");
  }
  render() {
    return <h2>Mounting Phase</h2>;
  }
}`}
      </pre>

      <h2 className="text-xl font-bold mt-6">2. Updating (อัปเดตข้อมูล)</h2>
      <p className="mt-2">เมื่อ Props หรือ State เปลี่ยนแปลง</p>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`class MyComponent extends Component {
  componentDidUpdate() {
    console.log("Component อัปเดตแล้ว");
  }
  render() {
    return <h2>Updating Phase</h2>;
  }
}`}
      </pre>

      <h2 className="text-xl font-bold mt-6">3. Unmounting (ลบออกจาก DOM)</h2>
      <p className="mt-2">เมื่อ Component ถูกลบออกจาก DOM</p>
      <pre className="p-4 mt-2 rounded-md overflow-x-auto border bg-gray-200 text-black dark:bg-gray-800 dark:text-white">
{`class MyComponent extends Component {
  componentWillUnmount() {
    console.log("Component ถูกลบออกจาก DOM");
  }
  render() {
    return <h2>Unmounting Phase</h2>;
  }
}`}
      </pre>
    </div>
  );
};

export default ComponentLifecycle;
