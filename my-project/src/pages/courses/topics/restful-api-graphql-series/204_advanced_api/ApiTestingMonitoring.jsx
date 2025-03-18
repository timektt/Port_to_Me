import React from "react";

const ApiTestingMonitoring = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Testing & Monitoring APIs</h1>
      <p className="mb-4">
        การทดสอบและติดตาม API มีความสำคัญในการตรวจสอบข้อผิดพลาดและปรับปรุงประสิทธิภาพ.
      </p>
      <h2 className="text-xl font-semibold mt-4">การทดสอบ API ด้วย Jest</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const request = require("supertest");
const app = require("../app");

test("GET /users", async () => {
  const response = await request(app).get("/users");
  expect(response.status).toBe(200);
});`}
      </pre>
    </div>
  );
};

export default ApiTestingMonitoring;
