// backend/middleware/authMiddleware.js
const jwt = require("jsonwebtoken");

// ✅ Middleware: ตรวจสอบความถูกต้องของ JWT token
const verifyToken = (req, res, next) => {
  try {
    const authHeader = req.headers["authorization"];
    const token = authHeader && authHeader.split(" ")[1]; // "Bearer <token>"

    if (!token) {
      return res.status(401).json({ success: false, message: "Missing token" });
    }

    jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
      if (err) {
        return res.status(403).json({ success: false, message: "Invalid token" });
      }

      req.user = decoded; // ⬅️ เก็บข้อมูล payload (เช่น email, role) ไว้ใน req.user
      next();
    });
  } catch (error) {
    console.error("❌ Token verification error:", error);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

// ✅ Middleware: ตรวจสอบว่าเป็น admin role เท่านั้น
const checkAdminRole = (req, res, next) => {
  if (!req.user || req.user.role !== "admin") {
    return res.status(403).json({ success: false, message: "Admin access only" });
  }
  next();
};

module.exports = {
  verifyToken,
  checkAdminRole,
};
