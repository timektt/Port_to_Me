// src/routes/ProtectedRoute.jsx
import { Navigate } from "react-router-dom";
import { useAuth } from "../components/context/AuthContext";

const ProtectedRoute = ({ children }) => {
  const { user } = useAuth();

  // ✅ ระหว่างยังไม่ได้เช็ค user ให้แสดง loading หรือ null ไปก่อน
  if (user === undefined) return null;

  // ✅ ถ้าไม่ใช่ user -> redirect ไป /login และไม่ให้ย้อนกลับ
  if (!user) return <Navigate to="/login" replace />;

  return children;
};

export default ProtectedRoute;
