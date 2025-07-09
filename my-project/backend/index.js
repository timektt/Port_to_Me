require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const verifyFirebaseToken = require("../middlewares/verifyFirebaseToken");
const userRoutes = require("./routes/userRoutes");

const app = express();
const PORT = process.env.PORT || 5000;

// app.listen(PORT, "0.0.0.0", () => {
//   console.log(`🚀 Server running on http://localhost:${PORT}`);
// });
console.log("📦 DATABASE_URL:", process.env.DATABASE_URL);

// ✅ PostgreSQL connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// ✅ Middleware
app.use(cors());
app.use(express.json());

// ✅ Routes
app.use("/api/user", userRoutes);
app.get("/", (req, res) => res.send("Backend is running!"));
app.get("/api/protected", verifyFirebaseToken, (req, res) => {
  res.json({
    message: "✅ Access granted",
    uid: req.user.uid,
    email: req.user.email
  });
});

// ✅ Test DB connection
pool.query("SELECT current_database()", (err, result) => {
  if (err) {
    console.error("❌ DB connect error:", err);
  } else {
    console.log("🧠 Connected to DB:", result.rows[0].current_database);
  }
});

// ✅ Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);

  pool.query("SELECT NOW()", (err, result) => {
    if (err) {
      console.error("❌ DB time fetch error:", err);
    } else {
      console.log("✅ DB Time:", result.rows[0].now);
    }
  });
});

// ✅ 404 fallback
app.use((req, res) => {
  res.status(404).json({ message: `🚫 Not Found: ${req.originalUrl}` });
});
