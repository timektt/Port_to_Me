import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  FaTimes,
  FaChevronDown,
  FaChevronRight,
  FaArrowLeft,
  FaMoon,
} from "react-icons/fa";
import { FiSun } from "react-icons/fi";

const sidebarItems = [
  {
    id: "101",
    title: "101: AI Core Concepts",
    subItems: [
      { id: "day1", title: "Day 1: Vectors & Matrices", path: "/courses/ai/intro-to-vectors-matrices" },
      { id: "day2", title: "Day 2: Vector Addition & Scalar Multiplication", path: "/courses/ai/vector-addition&scalarmultiplication" },
      { id: "day3", title: "Day 3: Dot Product & Cosine Similarity", path: "/courses/ai/dot-product&cosinesimilarity" },
      { id: "day4", title: "Day 4: Matrix Multiplication", path: "/courses/ai/matrix-multiplication" },
      { id: "day5", title: "Day 5: Linear Transformation & Feature Extraction", path: "/courses/ai/linear-transformation&feature-extraction" },
      { id: "day6", title: "Day 6: Activation Functions", path: "/courses/ai/activation-functions" },
      { id: "day7", title: "Day 7: Loss Functions & Optimization", path: "/courses/ai/lossfunctions&optimization" },
      { id: "day8", title: "Day 8: Backpropagation & Training Loop", path: "/courses/ai/backpropagation&trainingLoop" },
      { id: "day9", title: "Day 9: Regularization & Generalization", path: "/courses/ai/regularization&generalization" },
      { id: "day10", title: "Day 10: Bias-Variance Tradeoff & Model Capacity", path: "/courses/ai/bias-variancetradeoff&modelcapacity" },
      { id: "day11", title: "Day 11: Cross Validation & Model Evaluation", path: "/courses/ai/cross-validation&modelevaluation" }

    ],
  },
];

const AiMobileMenu = ({ onClose, theme, setTheme }) => {
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
      } overflow-y-auto pb-20`}
    >
      {/* ❌ ปุ่มปิดเมนู */}
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

      {/* ✅ Logo + ชื่อ + ปุ่มสลับธีม */}
      <div className="mt-6 flex items-center mb-3">
        <img
          src="/spm2.jpg"
          alt="Logo"
          className="w-8 h-8 mr-2 object-cover rounded-full"
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
              <FiSun className="text-yellow-400 text-2xl" />
            ) : (
              <FaMoon className="text-blue-400 text-2xl" />
            )}
          </button>
        </div>
      </div>

      {/* ✅ ปุ่มย้อนกลับ */}
      <button
        className={`w-full text-left text-sm font-medium px-5 py-3 rounded-lg mb-4 transition 
          ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        onClick={() => {
          navigate("/courses/ai-series");
          onClose();
        }}
      >
        <FaArrowLeft className="inline-block mr-2" /> AI Series
      </button>

      {/* ✅ เมนูรายวัน (dropdown) */}
      <ul className="space-y-2 mt-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700">
            <button
              className="flex items-center justify-between w-full p-4 rounded-lg transition duration-300 ease-in-out
                hover:bg-gray-700 hover:shadow-lg text-left"
              onClick={() => toggleSection(item.id)}
            >
              {item.title}
              {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
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

export default AiMobileMenu;
