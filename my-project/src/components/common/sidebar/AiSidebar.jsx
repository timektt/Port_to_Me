import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

const sidebarItems = [
    {
      id: "101",
      title: "101Basic: Core Concepts of AI",
      subItems: [
        {
          id: "day1",
          title: "Day 1: Introduction to Vectors & Matrices",
          path: "/courses/ai/intro-to-vectors-matrices",
        },
        {
          id: "day2",
          title: "Day 2: Vector Addition & Scalar Multiplication",
          path: "/courses/ai/vector-addition&scalarmultiplication",
        },
        {
            id: "day3",
            title: "Day 3: Dot Product & Cosine Similarity",
            path: "/courses/ai/dot-product&cosinesimilarity",
          },
        {
          id: "day4",
          title: "Day 4: Matrix Multiplication",
          path: "/courses/ai/matrix-multiplication",
        },
        {
          id: "day5",
          title: "Day 5: Linear Transformation & Feature Extraction",
          path: "/courses/ai/linear-transformation&feature-extraction",
        },
        {
          id: "day6",
          title: "Day 6: Activation Functions",
          path: "/courses/ai/activation-functions",
        },
        {
          id: "day7",
          title: "Day 7: Loss Functions & Optimization",
          path: "/courses/ai/lossfunctions&optimization",
        },
        {
          id: "day8",
          title: "Day 8: Backpropagation & Training Loop",
          path: "/courses/ai/backpropagation&trainingLoop",
        },
        {
          id: "day9",
          title: "Day 9: Regularization & Generalization",
          path: "/courses/ai/regularization&generalization",
        },
        { id: "day10", 
          title: "Day 10: Bias-Variance Tradeoff & Model Capacity", 
          path: "/courses/ai/bias-variancetradeoff&modelcapacity" },
          { id: "day11", title: "Day 11: Cross Validation & Model Evaluation", 
          path: "/courses/ai/cross validation&modelevaluation" }

      ],
    },
  ];
  
const AiSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={`fixed top-16 left-0 w-64 h-[calc(100vh-70px)] overflow-y-auto z-50 p-4 transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
          ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg pb-20`}
      >
        <button
          className="md:hidden absolute top-4 right-4 text-xl"
          onClick={() => setSidebarOpen(false)}
        >
          <FaTimes />
        </button>
        <h2
          className="text-m font-bold mb-4 cursor-pointer transition hover:underline hover:text-yellow-400"
          onClick={() => navigate("/courses/ai-series")}
        >
          <span
  className={`inline-block px-5 py-2 rounded-md text-lg font-semibold ${
    theme === "dark" ? "bg-gray-700" : "bg-gray-200"
  }`}
>
  AI Series
</span>

        </h2>
        <ul className="space-y-2 mt-4 mb-24">
          {sidebarItems.map((item) => (
            <li key={item.id} className="border-b border-gray-700">
              <button
                className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out hover:bg-gray-700 hover:shadow-lg text-left"
                onClick={() => toggleSection(item.id)}
              >
                {item.title}
                {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
              </button>
              {expandedSections[item.id] && (
                <ul className="pl-5 space-y-2 mt-2">
                  {item.subItems.map((sub) => (
                    <li
                      key={sub.id}
                      className={`p-2 rounded-lg cursor-pointer transition duration-200 ${
                        location.pathname === sub.path ?
                          "bg-green-500 text-white font-bold" :
                          "hover:bg-gray-600"
                      }`}
                      onClick={() => {
                        navigate(sub.path);
                        setSidebarOpen(false);
                      }}
                    >
                      {sub.title}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      </aside>
    </>
  );
};

export default AiSidebar;
