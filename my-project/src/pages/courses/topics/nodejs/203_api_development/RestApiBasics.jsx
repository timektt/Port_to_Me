import React from "react";

const RestApiBasics = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">🌐 REST API Basics</h1>
      <p className="mt-4">
        RESTful API เป็นรูปแบบมาตรฐานในการพัฒนา Web API โดยใช้ **HTTP Methods** เช่น GET, POST, PUT, DELETE
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่างการสร้าง REST API</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-2">
        <code>{`const express = require("express");
const app = express();

app.get("/api/users", (req, res) => {
  res.json([{ id: 1, name: "John Doe" }]);
});

app.listen(3000, () => console.log("Server running on port 3000"));`}</code>
      </pre>

      <p className="mt-4">🚀 เปิดใช้งานที่: **http://localhost:3000/api/users**</p>
    </div>
  );
};

export default RestApiBasics;
