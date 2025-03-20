import React from "react";

const IntroductionToWebDevelopment = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">แนะนำการพัฒนาเว็บไซต์</h1>
      <p className="mb-4">
        การพัฒนาเว็บไซต์ (Web Development) คือกระบวนการสร้างเว็บไซต์และเว็บแอปพลิเคชัน
        ซึ่งรวมถึงการออกแบบเว็บไซต์ การเขียนโปรแกรม และการจัดการฐานข้อมูล
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">Web Development คืออะไร?</h2>
      <p className="mb-4">
        การพัฒนาเว็บไซต์หมายถึงการสร้าง ดูแล และจัดการเว็บไซต์ โดยมีทั้งฝั่ง Frontend (ส่วนติดต่อผู้ใช้)
        และ Backend (การประมวลผลฝั่งเซิร์ฟเวอร์และฐานข้อมูล)
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">ทำไมต้องเรียน Web Development?</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>มีความต้องการสูงในตลาดงานด้านไอที</li>
        <li>สามารถสร้างโปรเจคส่วนตัวหรือทำสตาร์ทอัพได้</li>
        <li>มีโอกาสทำงานทางไกล (Remote Work) เป็นฟรีแลนซ์</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6 mb-2">ตัวอย่างโค้ด: โครงสร้าง HTML พื้นฐาน</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`<!DOCTYPE html>
<html>
<head>
  <title>My First Web Page</title>
</head>
<body>
  <h1>Welcome to Web Development!</h1>
  <p>This is a basic HTML structure.</p>
</body>
</html>`}
      </pre>
    </>
  );
};

export default IntroductionToWebDevelopment;