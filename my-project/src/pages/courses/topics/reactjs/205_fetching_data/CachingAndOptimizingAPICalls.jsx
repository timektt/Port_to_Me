import React from "react";

const CachingAndOptimizingAPICalls = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Caching & Optimizing API Calls</h1>
      <p className="mt-4 text-lg">
        การใช้ Caching และการปรับแต่ง API Calls ช่วยลด Load บน Server และเพิ่มประสิทธิภาพแอปพลิเคชัน
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 วิธีการ Cache API Response</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
{`const cache = new Map();
function fetchData(url) {
  if (cache.has(url)) return Promise.resolve(cache.get(url));

  return fetch(url)
    .then(response => response.json())
    .then(data => {
      cache.set(url, data);
      return data;
    });
}`}
      </pre>
    </div>
  );
};

export default CachingAndOptimizingAPICalls;
