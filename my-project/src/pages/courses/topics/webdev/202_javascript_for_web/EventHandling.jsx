import React from "react";

const EventHandling = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การจัดการ Event ใน JavaScript</h1>
      <p>
        Event Handling เป็นการจัดการเหตุการณ์ที่เกิดขึ้นบนเว็บ เช่น การคลิก การกดแป้นพิมพ์ และการเปลี่ยนค่าในฟอร์ม
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ Event Listener</h2>
      <p>
        JavaScript สามารถใช้ <code>addEventListener</code> เพื่อเพิ่ม Event Handler ให้กับองค์ประกอบ HTML ได้
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ addEventListener</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`document.getElementById("btn").addEventListener("click", function() {
  alert("ปุ่มถูกคลิก!");
});`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">การใช้ Event ใน React</h2>
      <p>
        ใน React สามารถจัดการ Event ได้โดยใช้ Handler เช่น <code>onClick</code> และ <code>onChange</code>
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Event ใน React</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`const handleClick = () => {
  alert("ปุ่มใน React ถูกคลิก!");
};

<button onClick={handleClick}>คลิกที่นี่</button>`}
      </pre>
    </>
  );
};

export default EventHandling;
