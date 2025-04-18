import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  FaTimes,
  FaChevronDown,
  FaChevronRight,
  FaArrowLeft,
  FaMoon,
} from "react-icons/fa";
import { FiSun } from "react-icons/fi"; // ✅ ใช้ FiSun แทน FaSun

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

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <div
      className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"
      }`}
    >
      {/* ปุ่มปิดเมนู */}
      <button
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 ${
          theme === "dark"
            ? "text-white hover:text-gray-400"
            : "text-black hover:text-gray-600"
        }`}
        onClick={onClose}
      >
        <FaTimes />
      </button>

      {/* โลโก้ + Superbear + ปุ่มเปลี่ยนธีม */}
      <div className="mt-6 flex items-center mb-6">
        <img
          src="/spm2.jpg"
          alt="Logo"
          className="w-10 h-10 mr-3 object-cover rounded-full"
        />
        <div className="flex items-center space-x-2">
          <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
            Superbear
          </span>
          <button
            className="cursor-pointer transition-transform transform hover:scale-110"
            onClick={toggleTheme}
          >
            {theme === "dark" ? (
              <FiSun className="text-yellow-400 text-2xl" /> // ✅ ใช้ FiSun
            ) : (
              <FaMoon className="text-blue-400 text-2xl" />
            )}
          </button>
        </div>
      </div>

      {/* ปุ่มกลับไปหน้า Python Series */}
      <button
        className={`w-full text-left text-sm font-medium px-5 py-3 rounded-lg mb-4 transition 
          ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        onClick={() => {
          navigate("/courses/python-series");
          onClose();
        }}
      >
        <FaArrowLeft className="inline-block mr-2" /> Python Series
      </button>

      {/* รายการบทเรียน */}
      <ul className="space-y-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700 pb-2">
            <button
              className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out hover:bg-gray-700 hover:shadow-lg text-left"
              onClick={() => toggleSection(item.id)}
            >
              <span className="font-medium">{item.title}</span>
              {expandedSections[item.id] ? (
                <FaChevronDown />
              ) : (
                <FaChevronRight />
              )}
            </button>

            {expandedSections[item.id] && (
              <ul className="pl-5 space-y-2 mt-2">
                {item.subItems.map((subItem) => (
                  <li
                    key={subItem.id}
                    className={`p-2 rounded-lg cursor-pointer transition duration-200 ${
                      location.pathname === subItem.path
                        ? "bg-green-500 text-white font-bold"
                        : "hover:bg-gray-600"
                    }`}
                    onClick={() => {
                      navigate(subItem.path);
                      onClose();
                    }}
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
