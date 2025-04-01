require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const bcrypt = require("bcrypt");
const authRoutes = require("./routes/authRoutes");
const app = express();


const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false
  }
});

app.use(cors());
app.use(express.json());
app.use("/api/auth", authRoutes);

// ตรวจสอบว่าเชื่อมกับ DB อะไรอยู่
pool.query("SELECT current_database()", (err, result) => {
  if (err) {
    console.error("❌ ไม่สามารถเชื่อมต่อฐานข้อมูล:", err);
  } else {
    console.log("🧠 กำลังเชื่อมต่อกับฐานข้อมูล:", result.rows[0].current_database);
  }
});

app.get("/", (req, res) => res.send("Backend is running!"));

// ✅ REGISTER
app.post("/api/register", async (req, res) => {
  const { email, password } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    const query = 'INSERT INTO "users" (email, password) VALUES ($1, $2)';
    const values = [email, hashedPassword];

    await pool.query(query, values);
    res.status(201).json({ success: true, message: "User registered" });
  } catch (err) {
    console.error("❌ Register error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// ✅ LOGIN
app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  console.log("📥 Email:", email);
  console.log("📥 Password:", password);

  try {
    const query = 'SELECT * FROM "users" WHERE email = $1';
    const values = [email];
    const result = await pool.query(query, values);

    if (result.rows.length === 0) {
      return res.status(401).json({ success: false, message: "Invalid credentials" });
    }

    const user = result.rows[0];
    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) {
      return res.status(401).json({ success: false, message: "Invalid credentials" });
    }

    res.json({ success: true, message: "Login successful" });
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
app.use((req, res) => {
  res.status(404).json({ message: "🚫 Not Found: " + req.originalUrl });
});