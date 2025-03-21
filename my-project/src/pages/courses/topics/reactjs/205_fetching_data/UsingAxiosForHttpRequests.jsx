import React, { useState, useEffect } from "react";
import axios from "axios";

const UsingAxiosForHttpRequests = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios
      .get("https://jsonplaceholder.typicode.com/posts/1")
      .then((response) => {
        setData(response.data);
        setLoading(false);
      })
      .catch((err) => {
        setError("ไม่สามารถโหลดข้อมูลได้");
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Using Axios for HTTP Requests</h1>

      <p className="mt-4 text-lg">
        <strong>Axios</strong> เป็นไลบรารีที่ใช้สำหรับทำ HTTP Requests ได้อย่างสะดวกมากกว่าการใช้ Fetch API
        โดยสามารถใช้งานได้ทั้ง GET, POST, PUT และ DELETE
      </p>

      <h2 className="text-2xl font-semibold mt-6">📦 ข้อดีของ Axios</h2>
      <ul className="list-disc pl-6 mt-2 space-y-2">
        <li>เขียนโค้ดสั้นและอ่านง่าย</li>
        <li>รองรับการตั้งค่า header, baseURL และ interceptors</li>
        <li>สามารถใช้งานกับ async/await ได้ดี</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้ Axios</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`axios.get("https://jsonplaceholder.typicode.com/posts/1")
  .then(response => console.log(response.data))
  .catch(error => console.error(error));`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📥 ผลลัพธ์ที่ได้</h2>
      {loading && <p>⏳ กำลังโหลดข้อมูล...</p>}
      {error && <p className="text-red-600">❌ {error}</p>}
      {data && (
        <div className="mt-4">
          <h3 className="font-semibold">📝 ชื่อโพสต์:</h3>
          <p>{data.title}</p>
        </div>
      )}
    </div>
  );
};

export default UsingAxiosForHttpRequests;
