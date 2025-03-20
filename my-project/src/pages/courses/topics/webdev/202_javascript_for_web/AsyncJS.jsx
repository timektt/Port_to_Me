import React from "react";

const AsyncJS = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การจัดการ Asynchronous JavaScript (Promises & Async/Await)</h1>
      <p>
        JavaScript รองรับการทำงานแบบ Asynchronous ซึ่งช่วยให้โค้ดสามารถดำเนินการแบบไม่ต้องรอคำสั่งก่อนหน้าเสร็จสมบูรณ์
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ Promises</h2>
      <p>
        Promises เป็นวัตถุที่ใช้ในการจัดการงานที่ต้องใช้เวลา เช่น การดึงข้อมูลจาก API
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Promises</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`fetch("https://api.example.com/data")
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error("เกิดข้อผิดพลาด", error));`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ Async/Await</h2>
      <p>
        Async/Await เป็นฟีเจอร์ที่ช่วยให้การเขียนโค้ด Asynchronous อ่านง่ายขึ้น
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Async/Await</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`async function fetchData() {
  try {
    let response = await fetch("https://api.example.com/data");
    let data = await response.json();
    console.log(data);
  } catch (error) {
    console.error("เกิดข้อผิดพลาด", error);
  }
}
fetchData();`}
      </pre>
    </>
  );
};

export default AsyncJS;
