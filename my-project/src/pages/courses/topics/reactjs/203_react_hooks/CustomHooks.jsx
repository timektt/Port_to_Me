// CustomHooks.jsx
import React, { useState, useEffect } from "react";

// ✅ สร้าง Custom Hook สำหรับ fetch ข้อมูล
const useFetch = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      });
  }, [url]);

  return { data, loading };
};

const CustomHooks = () => {
  const { data, loading } = useFetch("https://jsonplaceholder.typicode.com/posts/1");

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Custom Hooks ใน React</h1>
      <p className="mt-4 text-lg">
        Custom Hooks คือฟังก์ชันที่เริ่มต้นด้วยคำว่า <code>use</code> และช่วยให้เรา
        สามารถนำ logic ที่ใช้ซ้ำกันออกมาแยกใช้งานได้สะดวกขึ้น เช่น การดึงข้อมูลจาก API
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง Custom Hook: useFetch</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-100 dark:bg-gray-800 dark:text-white">
{`const useFetch = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      });
  }, [url]);

  return { data, loading };
};`}
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 การใช้งานใน Component</h2>
      <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-100 dark:bg-gray-800 dark:text-white">
{`const { data, loading } = useFetch("https://jsonplaceholder.typicode.com/posts/1");

return loading ? <p>Loading...</p> : <p>{data.title}</p>;`}
      </pre>

      <div className="mt-6">
        <h2 className="text-xl font-semibold mb-2">🔍 แสดงผลข้อมูลที่ดึงมา</h2>
        {loading ? (
          <p>กำลังโหลดข้อมูล...</p>
        ) : (
          <p className="mt-2"><strong>Title:</strong> {data.title}</p>
        )}
      </div>

      <h2 className="text-xl font-semibold mt-6">📌 ข้อดีของ Custom Hooks</h2>
      <ul className="list-disc pl-6 mt-2 space-y-1">
        <li>เขียน logic reuse ได้สะดวก</li>
        <li>แยก concerns ออกจาก UI</li>
        <li>ทำให้โค้ดอ่านง่ายและจัดการง่ายขึ้น</li>
      </ul>
    </div>
  );
};

export default CustomHooks;
