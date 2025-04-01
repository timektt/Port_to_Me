const admin = require("../firebaseAdmin");

const verifyFirebaseToken = async (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ message: "Unauthorized: Missing token" });
  }

  const token = authHeader.split(" ")[1];

  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded;
    next();
  } catch (err) {
    console.error("❌ Token verify failed:", err);
    res.status(401).json({ message: "Unauthorized: Invalid token" });
  }
};

module.exports = verifyFirebaseToken;
