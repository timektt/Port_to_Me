import React from "react";

const ApiTypes = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">📚 ประเภทของ API</h1>

      <p className="text-lg">
        API สามารถแบ่งประเภทได้หลากหลายตามลักษณะการใช้งานและการเข้าถึงของผู้ใช้ เพื่อให้เหมาะกับความต้องการของระบบหรือองค์กร
      </p>

      <h2 className="text-2xl font-semibold mt-6">1️⃣ Public APIs (เปิดสาธารณะ)</h2>
      <p className="mt-2">
        เป็น API ที่เปิดให้ทุกคนใช้งานได้ มักใช้เพื่อส่งเสริมการพัฒนาแอปจากบุคคลภายนอก เช่น API ของ Facebook, Twitter, หรือ Google Maps
      </p>

      <h2 className="text-2xl font-semibold mt-6">2️⃣ Private APIs (ใช้ภายในองค์กร)</h2>
      <p className="mt-2">
        เป็น API ที่ใช้ภายในระบบขององค์กรเท่านั้น เช่น การเชื่อมต่อระหว่าง frontend กับ backend ของเว็บไซต์ของบริษัท
        เพื่อควบคุมความปลอดภัยและลดการเข้าถึงจากภายนอก
      </p>

      <h2 className="text-2xl font-semibold mt-6">3️⃣ Partner APIs (สำหรับพันธมิตร)</h2>
      <p className="mt-2">
        ใช้ในการเชื่อมต่อข้อมูลระหว่างองค์กรกับพันธมิตรทางธุรกิจ เช่น API สำหรับระบบชำระเงิน, การจองโรงแรม หรือ API ที่แชร์ระหว่างบริษัทแม่กับบริษัทลูก
      </p>

      <h2 className="text-2xl font-semibold mt-6">4️⃣ Composite APIs (แบบรวมหลาย API)</h2>
      <p className="mt-2">
        เป็น API ที่เรียกใช้งานหลาย API endpoint ได้ภายในครั้งเดียว เหมาะสำหรับลดจำนวนการ request และใช้ในระบบที่มีความซับซ้อน เช่นการดึงข้อมูลผู้ใช้พร้อมข้อมูลคำสั่งซื้อ
      </p>

      <div className="mt-8 p-4 bg-yellow-100 dark:bg-yellow-800 text-yellow-900 dark:text-yellow-200 rounded-lg">
        💡 <strong>Tip:</strong> การเลือกประเภท API ควรพิจารณาจากระดับความปลอดภัย ความยืดหยุ่น และการใช้งานร่วมกับผู้อื่น
      </div>
    </div>
  );
};

export default ApiTypes;
