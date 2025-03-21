import React, { useState, useEffect } from "react";

const HandlingLoadingAndErrors = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("https://jsonplaceholder.typicode.com/posts/9999")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Data not found");
        }
        return response.json();
      })
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Handling Loading & Errors</h1>

      <p className="mt-4 text-lg">
        เวลา Fetch ข้อมูลจาก API จำเป็นต้องจัดการ <strong>สถานะ Loading</strong> และ <strong>ข้อผิดพลาด (Errors)</strong> 
        เพื่อให้ผู้ใช้ได้รับประสบการณ์ที่ดี และแอปไม่แสดงข้อมูลผิดพลาด
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ทำไมต้องจัดการ Loading & Error?</h2>
      <ul className="list-disc pl-6 mt-2 space-y-2">
        <li>เพื่อแสดงข้อความรอโหลดอย่างเหมาะสม</li>
        <li>เพื่อแจ้งผู้ใช้เมื่อเกิดปัญหา เช่น API ไม่ตอบกลับ</li>
        <li>เพื่อป้องกัน UI ค้าง หรือแสดงข้อมูลที่ผิดพลาด</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการจัดการด้วย <code>useEffect</code></h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`useEffect(() => {
  fetch("https://jsonplaceholder.typicode.com/posts/9999")
    .then(response => {
      if (!response.ok) {
        throw new Error("Data not found");
      }
      return response.json();
    })
    .then(data => setData(data))
    .catch(error => setError(error.message));
}, []);`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📦 ผลลัพธ์</h2>
      {loading && <p>⏳ กำลังโหลดข้อมูล...</p>}
      {error && <p className="text-red-600">❌ {error}</p>}
      {data && (
        <div className="mt-4">
          <h3 className="font-semibold text-lg">✅ ข้อมูลที่โหลด:</h3>
          <p>{data?.title}</p>
        </div>
      )}
    </div>
  );
};

export default HandlingLoadingAndErrors;
