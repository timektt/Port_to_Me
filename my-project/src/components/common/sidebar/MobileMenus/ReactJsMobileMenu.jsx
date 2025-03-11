import { FaTimes, FaChevronRight, FaArrowLeft } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

const ReactJsMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
  return (
    <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}>
      <button className="absolute right-4 top-4 text-2xl" onClick={onClose}><FaTimes /></button>
      <ul className="mt-3 space-y-3">
        <li><button onClick={() => { navigate("/courses/reactjs-series/101"); onClose(); }} className="w-full flex justify-between items-center text-left hover:text-gray-300">101: Introduction <FaChevronRight /></button></li>
      </ul>
    </div>
  );
};

export default ReactJsMobileMenu;
