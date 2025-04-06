import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../components/context/AuthContext";

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  const location = useLocation(); // ✅ เพิ่มบรรทัดนี้

  if (loading || user === undefined) return null;

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />; // ✅ ส่งตำแหน่งเดิมไป
  }

  return children;
};

export default ProtectedRoute;
