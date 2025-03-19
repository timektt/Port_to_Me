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
        Fetch API เป็นเครื่องมือสำหรับการดึงข้อมูลจาก API แบบง่าย ๆ ใน JavaScript
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้ Fetch API</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
{`fetch("https://jsonplaceholder.typicode.com/posts/1")
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error(error));`}
      </pre>

      {loading ? <p>Loading...</p> : <p className="mt-4">{data?.title}</p>}
    </div>
  );
};

export default FetchingDataWithFetchAPI;
