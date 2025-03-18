import React from "react";

const ApiGatewaysMicroservices = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">API Gateways & Microservices</h1>
      <p className="mb-4">
        API Gateway เป็นตัวกลางที่ช่วยจัดการการร้องขอจากลูกค้าไปยังหลายๆ Microservices.
      </p>
      <h2 className="text-xl font-semibold mt-4">ตัวอย่างการใช้ API Gateway ด้วย Express</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const express = require("express");
const app = express();

app.use("/user", require("./userService"));
app.use("/order", require("./orderService"));

app.listen(3000, () => console.log("API Gateway Running..."));`}
      </pre>
    </div>
  );
};

export default ApiGatewaysMicroservices;
