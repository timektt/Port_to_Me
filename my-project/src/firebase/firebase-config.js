// src/firebase/firebase-config.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore"; // ⬅️ เพิ่ม Firestore
import { getStorage } from "firebase/storage"; 

const firebaseConfig = {
  apiKey: "AIzaSyAm9rRFycPYzVTDHyk6V22QQ4I7bqtk4ws",
  authDomain: "superbear-792a7.firebaseapp.com",
  projectId: "superbear-792a7",
  storageBucket: "superbear-792a7.appspot.com",
  messagingSenderId: "336251532730",
  appId: "1:336251532730:web:4a27ed9ce66c7becaa5983"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app); // ⬅️ เพิ่มบรรทัดนี้
export default app;
export const storage = getStorage(app); // ✅ เพิ่มบรรทัดนี้
