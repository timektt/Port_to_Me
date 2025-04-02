require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const admin = require("./firebaseAdmin"); // ✅ เพิ่ม Firebase Admin
const app = express();
const verifyFirebaseToken = require("./middleware/verifyFirebaseToken");
const userRoutes = require("./routes/userRoutes");
const PORT = process.env.PORT || 5000;


const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false
  }
});

app.use("/api/user", userRoutes);
app.use(cors());
app.use(express.json());

// ✅ Middleware: ตรวจสอบ Firebase Token
const verifyFirebaseToken = async (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ message: "Unauthorized" });
  }

  const token = authHeader.split(" ")[1];

  try {
    const decodedToken = await admin.auth().verifyIdToken(token);
    req.user = decodedToken;
    next();
  } catch (error) {
    console.error("❌ Token verification failed:", error);
    return res.status(401).json({ message: "Invalid token" });
  }
};

// ✅ ตรวจสอบฐานข้อมูล
pool.query("SELECT current_database()", (err, result) => {
  if (err) {
    console.error("❌ ไม่สามารถเชื่อมต่อฐานข้อมูล:", err);
  } else {
    console.log("🧠 กำลังเชื่อมต่อกับฐานข้อมูล:", result.rows[0].current_database);
  }
});

// ✅ Route หลัก
app.get("/", (req, res) => res.send("Backend is running!"));

// ✅ Route ที่ต้อง login ด้วย Firebase Auth
app.get("/api/protected", verifyFirebaseToken, (req, res) => {
  res.json({ message: "✅ Access granted", uid: req.user.uid, email: req.user.email });
});

// ✅ เริ่ม server
app.listen(PORT, () => {
  console.log(`🚀 Server started on http://localhost:${PORT}`);

  pool.query("SELECT NOW()", (err, result) => {
    if (err) {
      console.error("❌ DB Connect Fail:", err);
    } else {
      console.log("✅ DB Connected:", result.rows[0].now);
    }
  });
});

// ✅ Fallback route
app.use((req, res) => {
  res.status(404).json({ message: "🚫 Not Found: " + req.originalUrl });
});
