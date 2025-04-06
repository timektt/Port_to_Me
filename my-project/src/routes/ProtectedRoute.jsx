// src/routes/ProtectedRoute.jsx
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../components/context/AuthContext";

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  const location = useLocation(); // ใช้เพื่อส่ง path เดิมกลับไป

  if (loading || user === undefined) return null;

  // ✅ ถ้าไม่มี user ให้ redirect ไป login พร้อมส่งตำแหน่งเดิมไปด้วย
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return children;
};

export default ProtectedRoute;
