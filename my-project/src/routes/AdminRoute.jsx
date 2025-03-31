// src/routes/AdminRoute.jsx
import { Navigate } from "react-router-dom";
import { useAuth } from "../components/context/AuthContext";

const AdminRoute = ({ children }) => {
  const { user, role, loading } = useAuth();

  console.log("✅ AdminRoute - user:", user?.email, "role:", role);

  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  if (role !== "admin") return <Navigate to="/" replace />;

  return children;
};

export default AdminRoute;
