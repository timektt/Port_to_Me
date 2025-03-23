import React from "react";

const PromisesAsyncAwait = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto overflow-x-hidden">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold break-words">
        🚀 Promises & Async/Await
      </h1>

      <p className="mt-4 text-lg break-words">
        ใน Node.js <strong>Promise</strong> และ <strong>Async/Await</strong> ช่วยให้การเขียนโค้ดที่เกี่ยวกับงานแบบ Asynchronous อ่านง่ายขึ้น และลดการซ้อนกันของ callback ที่ยุ่งยาก
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Promises</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const getData = () => {
  return new Promise((resolve, reject) => {
    setTimeout(() => resolve("Data Loaded!"), 2000);
  });
};

getData().then((data) => console.log(data));`}</code>
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Async/Await</h2>
      <div className="overflow-x-auto">
        <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2 text-sm sm:text-base">
          <code>{`const fetchData = async () => {
  const data = await getData();
  console.log(data);
};

fetchData();`}</code>
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">🎯 ข้อดีของ Promises & Async/Await</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1 text-base">
        <li>โค้ดดูเป็นลำดับ อ่านเข้าใจง่าย</li>
        <li>หลีกเลี่ยง Callback Hell</li>
        <li>ง่ายต่อการจัดการ error ด้วย try/catch</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">⚠️ สิ่งที่ควรระวัง</h2>
      <ul className="list-disc ml-6 mt-2 space-y-1 text-base">
        <li>ลืมใส่ <code>await</code> อาจทำให้ไม่รอผลลัพธ์</li>
        <li>หาก Promise ไม่ reject เองจะไม่เกิด error</li>
        <li>async functions จะคืนค่าเป็น Promise เสมอ</li>
      </ul>
    </div>
  );
};

export default PromisesAsyncAwait;
