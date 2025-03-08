import { FaTimes, FaSun, FaMoon } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

const MainMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  return (
    <div className={`fixed top-0 left-0 w-64 h-full ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} p-4 z-50 shadow-lg`}>
      {/* ปุ่มปิดเมนู */}
      <button className="text-white text-2xl absolute right-4 top-4" onClick={onClose}>
        <FaTimes />
      </button>

      {/* ปุ่มเปลี่ยนธีม */}
      <button className="cursor-pointer mt-4" onClick={toggleTheme}>
        {theme === "dark" ? <FaSun className="text-yellow-400 text-xl" /> : <FaMoon className="text-blue-400 text-xl" />}
      </button>

      {/* เมนูหลัก */}
      <ul className="mt-4 space-y-3">
        <li>
          <button onClick={() => { navigate("/courses"); onClose(); }} className="w-full text-left hover:text-gray-300">
            Courses
          </button>
        </li>
        <li>
          <button onClick={() => { navigate("/posts"); onClose(); }} className="w-full text-left hover:text-gray-300">
            Posts
          </button>
        </li>
        <li>
          <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="block hover:text-gray-300">
            Youtube
          </a>
        </li>
        <li>
          <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="block hover:text-gray-300">
            Facebook
          </a>
        </li>
        <li>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="block hover:text-gray-300">
            Github
          </a>
        </li>
      </ul>
    </div>
  );
};

export default MainMobileMenu;
