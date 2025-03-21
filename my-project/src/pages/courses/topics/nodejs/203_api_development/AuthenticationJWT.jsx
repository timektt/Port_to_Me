import React from "react";

const ValidationErrorHandling = () => {
  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Data Validation & Error Handling</h1>
      <p className="mb-4">
        การตรวจสอบความถูกต้องของข้อมูล (Validation) และการจัดการข้อผิดพลาด (Error Handling) เป็นส่วนสำคัญของการพัฒนาแอปพลิเคชัน Node.js ที่ปลอดภัยและเสถียร
      </p>

      <h2 className="text-xl font-semibold mt-6">✅ การตรวจสอบข้อมูลด้วย Joi</h2>
      <p className="mt-2">Joi เป็นไลบรารียอดนิยมสำหรับการตรวจสอบข้อมูลที่ส่งเข้ามาทาง API</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2">
        <code>{`const Joi = require('joi');

const schema = Joi.object({
  name: Joi.string().min(3).required(),
  age: Joi.number().integer().min(18).required(),
});

const result = schema.validate({ name: 'Alice', age: 25 });

if (result.error) {
  console.log('Validation Error:', result.error.details[0].message);
} else {
  console.log('Validation Passed');
}`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">❌ การจัดการ Error ใน Express.js</h2>
      <p className="mt-2">Express.js รองรับ Middleware สำหรับจัดการข้อผิดพลาดได้ง่าย</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2">
        <code>{`app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something went wrong!');
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการรวม Joi กับ Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2">
        <code>{`app.post('/register', (req, res) => {
  const { error } = schema.validate(req.body);
  if (error) return res.status(400).send(error.details[0].message);
  res.send('Register success');
});`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6">🔍 สรุป</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>ควรตรวจสอบข้อมูลที่รับเข้ามาเสมอ เพื่อความปลอดภัย</li>
        <li>การจัดการ Error ที่ดีจะช่วยให้แอปไม่ล่มง่าย และตอบสนองผู้ใช้อย่างเหมาะสม</li>
        <li>แนะนำให้ใช้ Joi หรือ Zod ในการตรวจสอบข้อมูล</li>
        <li>ใช้ Middleware ของ Express จัดการ Error แบบรวมศูนย์</li>
      </ul>
    </div>
  );
};

export default ValidationErrorHandling;
