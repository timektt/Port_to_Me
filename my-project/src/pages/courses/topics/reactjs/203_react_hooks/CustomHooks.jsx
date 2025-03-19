// CustomHooks.jsx
import React, { useState, useEffect } from "react";

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
      <h1 className="text-3xl font-bold">Custom Hooks</h1>
      <p className="mt-4 text-lg">Custom Hooks ช่วยให้สามารถใช้โค้ดที่ซ้ำกันหลายที่ได้ง่ายขึ้น</p>
      {loading ? <p>กำลังโหลดข้อมูล...</p> : <p className="mt-4">{data.title}</p>}
    </div>
  );
};

export default CustomHooks;
