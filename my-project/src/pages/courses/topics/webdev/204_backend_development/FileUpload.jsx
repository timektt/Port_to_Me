import React from "react";

const FileUpload = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การอัปโหลดไฟล์ & การประมวลผลรูปภาพ</h1>
      <p>
        การอัปโหลดไฟล์เป็นฟังก์ชันสำคัญของเว็บแอปพลิเคชันที่เกี่ยวข้องกับการจัดเก็บไฟล์, การแก้ไขรูปภาพ หรือการประมวลผลไฟล์ประเภทต่าง ๆ
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การจัดการไฟล์ใน Backend</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>Multer - ใช้สำหรับอัปโหลดไฟล์ใน Express.js</li>
        <li>Sharp - ใช้สำหรับการปรับแต่งและแปลงรูปภาพ</li>
        <li>Cloud Storage (AWS S3, Firebase, Cloudinary) - ใช้สำหรับจัดเก็บไฟล์ในระบบ Cloud</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การอัปโหลดไฟล์ด้วย Multer</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const express = require('express');
const multer = require('multer');
const upload = multer({ dest: 'uploads/' });

const app = express();
app.post('/upload', upload.single('file'), (req, res) => {
  res.send('ไฟล์ถูกอัปโหลดแล้ว!');
});

app.listen(3000, () => console.log('Server running on port 3000'));`}
      </pre>
    </>
  );
};

export default FileUpload;
