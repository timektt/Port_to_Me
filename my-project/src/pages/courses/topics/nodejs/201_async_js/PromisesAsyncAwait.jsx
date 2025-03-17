import React from "react";

const PromisesAsyncAwait = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🚀 Promises & Async/Await</h1>

      <p className="mt-4 text-lg">
        ใน Node.js **Promise** และ **Async/Await** ช่วยให้การเขียนโค้ดที่เกี่ยวกับงานแบบ Asynchronous อ่านง่ายขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Promises</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const getData = () => {
  return new Promise((resolve, reject) => {
    setTimeout(() => resolve("Data Loaded!"), 2000);
  });
};

getData().then((data) => console.log(data));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการใช้ Async/Await</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const fetchData = async () => {
  const data = await getData();
  console.log(data);
};

fetchData();`}</code>
      </pre>
    </div>
  );
};

export default PromisesAsyncAwait;
