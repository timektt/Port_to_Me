import { FaHome, FaAngleRight } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

const Breadcrumb = ({ courseName, theme }) => {
  const navigate = useNavigate();

  return (
    <div className="flex items-center gap-3 mb-6">
      {/* ปุ่มกลับหน้า Home */}
      <button
        className={`flex items-center gap-2 px-3 py-2 rounded-md font-semibold transition-all ${
          theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-300 text-black hover:bg-gray-400"
        }`}
        onClick={() => navigate("/")}
      >
        <FaHome size={18} />
        <span>Home</span>
      </button>

      <FaAngleRight className="text-gray-400" />

      {/* ชื่อคอร์ส */}
      <span
        className={`px-4 py-2 rounded-md text-lg font-semibold ${
          theme === "dark" ? "bg-green-700 text-white" : "bg-green-300 text-black"
        }`}
      >
        {courseName}
      </span>
    </div>
  );
};

export default Breadcrumb;
