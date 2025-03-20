import React from "react";

const CachingStrategies = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">กลยุทธ์การแคช (Caching Strategies)</h1>
      <p>
        การแคชเป็นเทคนิคที่ช่วยเพิ่มประสิทธิภาพของระบบโดยลดเวลาการเข้าถึงข้อมูลจากฐานข้อมูลหลัก
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">ประเภทของ Caching</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>Memory Caching - เช่น Redis และ Memcached</li>
        <li>Database Caching - ใช้ Indexing และ Query Optimization</li>
        <li>Browser Caching - ใช้ HTTP Headers เช่น Cache-Control</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Redis สำหรับแคชข้อมูล</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const redis = require('redis');
const client = redis.createClient();

client.set('key', 'value', redis.print);
client.get('key', (err, reply) => {
  console.log(reply); // แสดงค่า "value"
});`}
      </pre>
    </>
  );
};

export default CachingStrategies;
