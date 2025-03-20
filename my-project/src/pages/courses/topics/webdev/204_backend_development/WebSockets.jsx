import React from "react";

const WebSockets = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">WebSockets & Real-time Applications</h1>
      <p>
        WebSockets เป็นเทคโนโลยีที่ช่วยให้สามารถส่งข้อมูลแบบเรียลไทม์ระหว่างไคลเอนต์และเซิร์ฟเวอร์ได้อย่างมีประสิทธิภาพ
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">คุณสมบัติหลักของ WebSockets</h2>
      <ul className="list-disc pl-6 mb-4">
        <li>สามารถสื่อสารแบบสองทาง (Bidirectional Communication)</li>
        <li>ลด Latency เพราะไม่มีการสร้าง HTTP Request ใหม่ทุกครั้ง</li>
        <li>เหมาะสำหรับแอปพลิเคชันเรียลไทม์ เช่น Chat, Stock Market, Gaming</li>
      </ul>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ WebSockets ด้วย Socket.io</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const io = require('socket.io')(3000);
io.on('connection', socket => {
  console.log('Client connected');
  socket.on('message', msg => {
    console.log('Message received:', msg);
    socket.emit('response', 'Message received!');
  });
});`}
      </pre>
    </>
  );
};

export default WebSockets;
