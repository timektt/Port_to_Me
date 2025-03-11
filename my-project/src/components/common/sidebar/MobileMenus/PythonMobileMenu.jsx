import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";

const sidebarItems = [
  {
    id: "101",
    title: "101: Basic Python",
    subItems: [
      { id: "101-1", title: "แนะนำ Python", path: "/courses/python-series/intro" },
      { id: "101-2", title: "Variable & Data Type", path: "/courses/python-series/variables" },
      { id: "101-3", title: "Control Structure", path: "/courses/python-series/control-structure" },
      { id: "101-4", title: "การรับ input & function", path: "/courses/python-series/input-function" },
      { id: "101-5", title: "LeetCode Challenge", path: "/courses/python-series/leetcode" },
    ],
  },
  {
    id: "201",
    title: "201: Data",
    subItems: [
      { id: "201-1", title: "Lists & Tuples", path: "/courses/python-series/data" },
      { id: "201-2", title: "Dictionaries", path: "/courses/python-series/dictionaries" },
      { id: "201-3", title: "Set & Frozenset", path: "/courses/python-series/set" },
      { id: "201-4", title: "การจัดการข้อมูลด้วย Pandas", path: "/courses/python-series/pandas" },
    ],
  },
  {
    id: "202",
    title: "202: Visualization",
    subItems: [
      { id: "202-1", title: "Matplotlib Basics", path: "/courses/python-series/matplotlib" },
      { id: "202-2", title: "Seaborn: Data Visualization", path: "/courses/python-series/seaborn" },
      { id: "202-3", title: "Plotly: Interactive Graphs", path: "/courses/python-series/plotly" },
    ],
  },
  {
    id: "203",
    title: "203: Data Wrangling & Transform",
    subItems: [
      { id: "203-1", title: "การล้างข้อมูล", path: "/courses/python-series/data-cleaning" },
      { id: "203-2", title: "การแปลงข้อมูล", path: "/courses/python-series/data-transformation" },
      { id: "203-3", title: "การจัดรูปแบบข้อมูล", path: "/courses/python-series/data-formatting" },
    ],
  },
  {
    id: "204",
    title: "204: Statistic Analysis",
    subItems: [
      { id: "204-1", title: "สถิติพื้นฐาน", path: "/courses/python-series/basic-statistics" },
      { id: "204-2", title: "Probability & Distribution", path: "/courses/python-series/probability" },
      { id: "204-3", title: "Hypothesis Testing", path: "/courses/python-series/hypothesis-testing" },
    ],
  },
  {
    id: "205",
    title: "205: Statistic Learning",
    subItems: [
      { id: "205-1", title: "Regression Analysis", path: "/courses/python-series/regression" },
      { id: "205-2", title: "Clustering Methods", path: "/courses/python-series/clustering" },
      { id: "205-3", title: "Deep Learning Basics", path: "/courses/python-series/deep-learning" },
    ],
  },
];

const PythonMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  return (
    <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 
      ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}>

      {/* ✅ ปุ่มปิดเมนู (X) */}
      <button 
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 
          ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`}
        onClick={onClose}
      >
        <FaTimes />
      </button>

      {/* ✅ โลโก้ + ปุ่ม Dark/Light Mode */}
      <div className="mt-6 flex items-center justify-between mb-3">
        <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
          Supermhee
        </span>
        <button className="cursor-pointer transition-transform transform hover:scale-110" onClick={toggleTheme}>
          {theme === "dark" ? <FaSun className="text-yellow-400 text-xl" /> : <FaMoon className="text-blue-400 text-xl" />}
        </button>
      </div>

      {/* ✅ ปุ่ม Back to Main Menu */}
      <button 
        className="flex items-center text-sm text-gray-400 hover:text-gray-300 mb-3 transition"
        onClick={onClose}
      >
        <FaArrowLeft className="mr-2" /> Back to main menu
      </button>

      {/* ✅ รายการบทเรียน (Dropdown) */}
      <ul className="space-y-2 mt-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700">
            {/* ปุ่มกดขยายหมวดหมู่ */}
            <button
              className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out
                hover:bg-gray-700 hover:shadow-lg text-left"
              onClick={() => toggleSection(item.id)}
            >
              {item.title}
              {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
            </button>

            {/* ✅ แสดงหมวดหมู่ย่อย */}
            {expandedSections[item.id] && (
              <ul className="pl-5 space-y-2 mt-2">
                {item.subItems.map((subItem) => (
                  <li
                    key={subItem.id}
                    className={`p-2 rounded-lg cursor-pointer transition duration-200 ${
                      location.pathname === subItem.path ? "bg-green-500 text-white font-bold" : "hover:bg-gray-600"
                    }`}
                    onClick={() => { navigate(subItem.path); onClose(); }}
                  >
                    {subItem.title}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default PythonMobileMenu;
