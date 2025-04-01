const admin = require("firebase-admin");

const serviceAccount = require("./serviceAccountKey.json"); // ⬅ ใส่ไฟล์จาก Firebase Console

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

module.exports = admin;
