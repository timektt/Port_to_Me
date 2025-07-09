// userRoutes.js
import express from "express";
import admin from "../firebaseAdmin.js";
import pg from "pg";

const router = express.Router();
const { Pool } = pg;

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
});

// 🔐 รับ token และบันทึก user ลง DB ถ้ายังไม่มี
router.post("/save-user", async (req, res) => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ message: "Unauthorized: Missing token" });
  }

  const token = authHeader.split(" ")[1];

  try {
    const decoded = await admin.auth().verifyIdToken(token);
    const { uid, email, firebase: { sign_in_provider } } = decoded;

    const checkQuery = 'SELECT * FROM "users" WHERE uid = $1';
    const checkResult = await pool.query(checkQuery, [uid]);

    if (checkResult.rows.length === 0) {
      const insertQuery = 'INSERT INTO "users" (uid, email, provider) VALUES ($1, $2, $3)';
      await pool.query(insertQuery, [uid, email, sign_in_provider]);
      console.log("✅ User saved to DB:", email);
    }

    res.status(200).json({ message: "User verified and saved", uid, email });
  } catch (err) {
    console.error("❌ Save user error:", err);
    res.status(401).json({ message: "Unauthorized: Invalid token" });
  }
});

export default router;
