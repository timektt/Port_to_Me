import { FaTimes, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

const MobileMenu = ({ onClose, theme, setTheme }) => {
    const navigate = useNavigate();

    return (
        <div className="fixed top-0 left-0 w-64 h-full bg-gray-900 text-white p-4 z-50 shadow-lg">
            {/* ✅ ปุ่มปิดเมนู */}
            <button className="text-white text-2xl absolute right-4 top-4" onClick={onClose}>
                <FaTimes />
            </button>

            {/* ✅ โลโก้ + ชื่อ + ปุ่ม Dark/Light (ขยับปุ่มไปทางซ้าย) */}
            <div className="mt-6 flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <img src="/images/logo.png" alt="Logo" className="w-8 h-8 rounded-full" />
                    <span className="text-lg font-bold">Supermhee</span>
                </div>
                {/* ✅ ปรับปุ่ม Dark/Light ไปทางซ้าย & ให้คลิกได้จริง */}
                <button
                    className="cursor-pointer ml-auto mr-2"
                    onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                >
                    {theme === "dark" ? (
                        <FaSun className="text-yellow-400 text-xl" />
                    ) : (
                        <FaMoon className="text-blue-400 text-xl" />
                    )}
                </button>
            </div>

            {/* ✅ ปุ่ม Back to main menu */}
            <button className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3" onClick={onClose}>
                <FaArrowLeft className="mr-2" /> Back to main menu
            </button>

            {/* ✅ หัวข้อหลัก "Python Series" */}
            <div className="px-3 py-2 bg-gray-800 text-lg font-bold text-white rounded">
                Python Series
            </div>

            {/* ✅ รายการตอนเรียน Python */}
            <ul className="mt-3 space-y-3">
                <li>
                    <button onClick={() => { navigate("/courses/python/101"); onClose(); }} 
                        className="flex justify-between items-center w-full text-left hover:text-gray-300">
                        101: Basic Python <FaChevronRight />
                    </button>
                </li>
                <li>
                    <button onClick={() => { navigate("/courses/python/201"); onClose(); }} 
                        className="flex justify-between items-center w-full text-left hover:text-gray-300">
                        201: Data <FaChevronRight />
                    </button>
                </li>
                <li>
                    <button onClick={() => { navigate("/courses/python/202"); onClose(); }} 
                        className="flex justify-between items-center w-full text-left hover:text-gray-300">
                        202: Visualization <FaChevronRight />
                    </button>
                </li>
                <li>
                    <button onClick={() => { navigate("/courses/python/203"); onClose(); }} 
                        className="flex justify-between items-center w-full text-left hover:text-gray-300">
                        203: Data Wrangling & Transform <FaChevronRight />
                    </button>
                </li>
                <li>
                    <button onClick={() => { navigate("/courses/python/204"); onClose(); }} 
                        className="flex justify-between items-center w-full text-left hover:text-gray-300">
                        204: Statistic Analysis <FaChevronRight />
                    </button>
                </li>
                <li>
                    <button onClick={() => { navigate("/courses/python/205"); onClose(); }} 
                        className="flex justify-between items-center w-full text-left hover:text-gray-300">
                        205: Statistic Learning <FaChevronRight />
                    </button>
                </li>
            </ul>
        </div>
    );
};

export default MobileMenu;
