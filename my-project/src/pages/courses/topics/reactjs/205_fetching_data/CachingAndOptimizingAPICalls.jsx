import React from "react";

const CachingAndOptimizingAPICalls = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Caching & Optimizing API Calls</h1>

      <p className="mt-4 text-lg">
        การใช้ Caching และการปรับแต่ง API Calls ช่วยลดการโหลดซ้ำที่ไม่จำเป็น ลดเวลารอ และลดภาระบน Server
        โดยเฉพาะเมื่อมีการเรียกข้อมูลเดิมซ้ำ ๆ หลายครั้ง
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 วิธีการ Cache API Response</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`const cache = new Map();

function fetchData(url) {
  if (cache.has(url)) {
    console.log("📦 ใช้ข้อมูลจาก cache");
    return Promise.resolve(cache.get(url));
  }

  return fetch(url)
    .then(response => response.json())
    .then(data => {
      cache.set(url, data);
      return data;
    });
}`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">📌 เทคนิคอื่นในการ Optimizing</h2>
      <ul className="list-disc pl-6 space-y-2 mt-2">
        <li><strong>Debouncing:</strong> หน่วงการเรียก API เมื่อผู้ใช้ยังพิมพ์ไม่เสร็จ เช่นใน search bar</li>
        <li><strong>Memoization:</strong> ใช้ <code>useMemo</code> หรือ <code>useCallback</code> สำหรับคำนวณข้อมูลที่เปลี่ยนไม่บ่อย</li>
        <li><strong>Pagination / Infinite Scroll:</strong> แบ่งโหลดข้อมูลเป็นหน้าแทนการโหลดทั้งหมดทีเดียว</li>
        <li><strong>Prefetching:</strong> โหลดล่วงหน้าข้อมูลที่ผู้ใช้อาจต้องการในอนาคต</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">📌 ใช้งานร่วมกับ React Hooks</h2>
      <pre className="p-4 rounded-md overflow-x-auto border bg-gray-100 dark:bg-gray-800 text-black dark:text-white text-sm">
{`import { useEffect, useState } from "react";

function useCachedFetch(url) {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchData(url).then(setData);
  }, [url]);

  return data;
}`}
      </pre>

      <p className="mt-4">
        เทคนิคเหล่านี้ช่วยให้เว็บของคุณโหลดเร็วขึ้น และมีประสบการณ์ใช้งานที่ดีขึ้นสำหรับผู้ใช้
      </p>
    </div>
  );
};

export default CachingAndOptimizingAPICalls;
