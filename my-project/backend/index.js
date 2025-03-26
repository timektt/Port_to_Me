require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");

const app = express();

const pool = new Pool({
    user: "postgres",
    password: "Time_44611", // <- ใช้รหัสของคุณตรงนี้
    host: "localhost",
    port: 5432,
    database: "my_project"
  });

app.use(cors());
app.use(express.json());

// ตรวจสอบว่าเชื่อมกับ DB อะไรอยู่
pool.query("SELECT current_database()", (err, result) => {
  if (err) {
    console.error("❌ ไม่สามารถเชื่อมต่อฐานข้อมูล:", err);
  } else {
    console.log("🧠 กำลังเชื่อมต่อกับฐานข้อมูล:", result.rows[0].current_database);
  }
});

app.get("/", (req, res) => res.send("Backend is running!"));

app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  console.log("📥 Email:", email);
  console.log("📥 Password:", password);

  try {
    const query = 'SELECT * FROM "users" WHERE email = $1 AND password = $2';
    const values = [email, password];
    console.log("🔍 Executing:", query, values);

    const result = await pool.query(query, values);

    if (result.rows.length > 0) {
      res.json({ success: true, token: "sample_token" });
    } else {
      res.status(401).json({ success: false, message: "Invalid credentials" });
    }
  } catch (err) {
    console.error("❌ Login error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

app.listen(5000, () => {
  console.log("🚀 Server started on http://localhost:5000");

  pool.query("SELECT NOW()", (err, result) => {
    if (err) {
      console.error("❌ DB Connect Fail:", err);
    } else {
      console.log("✅ DB Connected:", result.rows[0].now);
    }
  });
});
