// import { FaTimes, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";
// import { useNavigate } from "react-router-dom";

// const MobileMenu = ({ onClose, theme, setTheme }) => {
//     const navigate = useNavigate();

//     const toggleTheme = () => {
//         const newTheme = theme === "dark" ? "light" : "dark";
//         setTheme(newTheme);
//         localStorage.setItem("theme", newTheme);
//     };

//     return (
//         <div className={`fixed top-0 left-0 w-64 h-full ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} p-4 z-50 shadow-lg`}>
          
//             <button
//                 className={`absolute right-4 top-4 text-2xl ${theme === "dark" ? "text-white" : "text-black"}`}
//                 onClick={onClose}
//             >
//                 <FaTimes />
//             </button>

           
//             <div className="flex items-center mt-6 gap-3 mb-4">
//                 <img src="/spm2.jpg" alt="Logo" className="w-8 h-8 rounded-full" />
//                 <span className="text-lg font-bold">Supermhee</span>
//                 <button className="cursor-pointer ml-auto" onClick={toggleTheme}>
//                     {theme === "dark" ? <FaSun className="text-yellow-400 text-xl" /> : <FaMoon className="text-blue-400 text-xl" />}
//                 </button>
//             </div>

//             {/* ✅ ปุ่ม Back to main menu */}
//             <button className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3" onClick={onClose}>
//                 <FaArrowLeft className="mr-2" /> Back to main menu
//             </button>

//             {/* ✅ หัวข้อหลัก "Python Series" */}
//             <div className={`px-3 py-2 ${theme === "dark" ? "bg-gray-800 text-white" : "bg-gray-300 text-black"} text-lg font-bold rounded`}>
//                 Python Series
//             </div>

//             {/* ✅ รายการตอนเรียน Python */}
//             <ul className="mt-3 space-y-3">
//                 {[
//                     { id: "101", title: "Basic Python", link: "/courses/python/101" },
//                     { id: "201", title: "Data", link: "/courses/python/201" },
//                     { id: "202", title: "Visualization", link: "/courses/python/202" }
//                 ].map((item) => (
//                     <li key={item.id}>
//                         <button
//                             onClick={() => { navigate(item.link); onClose(); }}
//                             className="flex justify-between items-center w-full text-left transition duration-200 ease-in-out hover:text-gray-400"
//                         >
//                             {item.id}: {item.title} <FaChevronRight />
//                         </button>
//                     </li>
//                 ))}
//             </ul>
//         </div>
//     );
// };

// export default MobileMenu;
