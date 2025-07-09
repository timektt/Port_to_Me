import dotenv from "dotenv";
import pg from "pg";

dotenv.config();

const { Pool } = pg;

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

pool.query("SELECT * FROM users", (err, res) => {
  if (err) {
    console.error("❌ DB ERROR:", err);
  } else {
    console.log("✅ Success:", res.rows);
  }
  pool.end();
});
