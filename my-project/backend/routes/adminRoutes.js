// routes/adminRoutes.js
const express = require("express");
const router = express.Router();
const { verifyToken, checkAdminRole } = require("../middleware/authMiddleware");

router.get("/admin-only", verifyToken, checkAdminRole, (req, res) => {
  res.json({ message: "Welcome, admin! 👑" });
});

module.exports = router;
