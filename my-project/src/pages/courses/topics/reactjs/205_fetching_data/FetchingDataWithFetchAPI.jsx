import React, { useState, useEffect } from "react";

const FetchingDataWithFetchAPI = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("https://jsonplaceholder.typicode.com/posts/1")
      .then((response) => response.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Fetching Data with Fetch API</h1>

      <p className="mt-4 text-lg">
        <strong>Fetch API</strong> คือฟังก์ชันพื้นฐานที่ใช้สำหรับดึงข้อมูลจาก API ด้วย JavaScript โดยทำงานแบบ Asynchronous
        ซึ่งสามารถใช้งานร่วมกับ <code>.then()</code>, <code>await</code>, และ <code>try/catch</code> ได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่าง Fetch ข้อมูลแบบพื้นฐาน</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`fetch("https://jsonplaceholder.typicode.com/posts/1")
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error(error));`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 การใช้ Fetch API ใน React</h2>
      <p className="mt-2">
        เรามักจะใช้ร่วมกับ <code>useEffect</code> เพื่อเรียก API เมื่อ Component โหลด และจัดการ state ด้วย <code>useState</code>
      </p>

      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm mt-2">
{`const [data, setData] = useState(null);

useEffect(() => {
  fetch("https://api.example.com/data")
    .then(res => res.json())
    .then(setData);
}, []);`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">🧪 แสดงข้อมูลที่ดึงมา</h2>
      {loading ? (
        <p className="mt-4">Loading...</p>
      ) : (
        <p className="mt-4"><strong>Title:</strong> {data?.title}</p>
      )}
    </div>
  );
};

export default FetchingDataWithFetchAPI;
