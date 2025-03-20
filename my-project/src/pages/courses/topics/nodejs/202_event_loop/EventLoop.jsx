import React from "react";

const EventLoop = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-gray-100 dark:bg-gray-900">
      <h1 className="text-2xl md:text-4xl font-bold text-gray-800 dark:text-white text-center">
        🔄 ทำความเข้าใจ Event Loop ใน JavaScript
      </h1>
      <p className="mt-4 text-gray-600 dark:text-gray-300 text-center max-w-2xl">
        Event Loop เป็นกลไกสำคัญของ JavaScript ที่ช่วยให้สามารถทำงานแบบ Asynchronous ได้อย่างมีประสิทธิภาพ
        โดยจัดการการทำงานของ Call Stack, Web APIs, Callback Queue และ Microtasks
      </p>
      
      <div className="mt-6 p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg w-full max-w-2xl">
        <h2 className="text-lg font-semibold text-gray-700 dark:text-white">🔹 Key Concepts:</h2>
        <ul className="list-disc mt-2 pl-5 text-gray-600 dark:text-gray-300">
          <li>📌 Call Stack</li>
          <li>📌 Web APIs</li>
          <li>📌 Callback Queue</li>
          <li>📌 Microtasks & Macrotasks</li>
          <li>📌 Execution Order</li>
        </ul>
      </div>
      
      <h2 className="text-xl font-semibold mt-6 text-gray-700 dark:text-white">📌 ตัวอย่างการทำงานของ Event Loop</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg w-full max-w-2xl">
        <pre>
          <code>{`console.log('Start');

setTimeout(() => {
  console.log('Inside setTimeout');
}, 0);

Promise.resolve().then(() => {
  console.log('Inside Promise');
});

console.log('End');`}</code>
        </pre>
      </div>
      
      <p className="mt-4 text-gray-600 dark:text-gray-300 text-center max-w-2xl">
        ในโค้ดด้านบน Output จะเป็น:
      </p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 shadow-lg w-full max-w-2xl">
        <pre>
          <code>{`Start
End
Inside Promise
Inside setTimeout`}</code>
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6 text-gray-700 dark:text-white">✅ สรุป</h2>
      <p className="mt-2 text-gray-600 dark:text-gray-300 text-center max-w-2xl">
        - <strong>Call Stack</strong>: รันโค้ดหลักแบบ Synchronous ก่อน
        <br/>- <strong>Microtasks</strong> (เช่น Promises) จะถูกรันก่อน Macrotasks
        <br/>- <strong>Macrotasks</strong> (เช่น setTimeout) จะรันหลังจาก Microtasks ทั้งหมดทำงานเสร็จ
      </p>
    </div>
  );
};

export default EventLoop;