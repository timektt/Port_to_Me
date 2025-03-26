// src/firebase/firebase-config.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAm9rRFycPYzVTDHyk6V22QQ4I7bqtk4ws",
  authDomain: "superbear-792a7.firebaseapp.com",
  projectId: "superbear-792a7",
  storageBucket: "superbear-792a7.appspot.com", // แก้ไข .app เป็น .app**spot**.com
  messagingSenderId: "336251532730",
  appId: "1:336251532730:web:4a27ed9ce66c7becaa5983"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export default app;
