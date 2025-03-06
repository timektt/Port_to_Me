import React from "react";
import { useNavigate } from "react-router-dom";
import { FaHome, FaAngleRight } from "react-icons/fa";
import Sidebar from "../../components/Sidebar";
import SupportMeButton from "../../components/SupportMeButton";

const lessons = [
  {
    id: "101",
    title: "Python Introduction",
    image: "/images/python-101.jpg",
    docLink: "#",
  },
  {
    id: "201",
    title: "Basic Data",
    image: "/images/python-201.jpg",
    docLink: "#",
  },
];

const PythonSeries = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div
      className={`min-h-screen flex ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
      }`}
    >
      <Sidebar activeCourse="Python Series" theme={theme} />

      <main className="flex-1 p-6">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-2 mb-4">
            <button
              className={`p-2 rounded-md transition ${
                theme === "dark"
                  ? "bg-gray-700 text-white hover:bg-gray-600"
                  : "bg-gray-300 text-black hover:bg-gray-400"
              }`}
              onClick={() => navigate("/")}
            >
              <FaHome size={18} />
            </button>
            <FaAngleRight className="text-gray-400" />
            <span
              className={`px-3 py-1 rounded-md font-semibold ${
                theme === "dark" ? "bg-gray-700 text-white" : "bg-gray-300 text-black"
              }`}
            >
              Python Series
            </span>
          </div>

          <h1 className="text-4xl font-bold text-left mt-4">Python Series</h1>

          <div
            className={`p-3 rounded-md mt-4 ${
              theme === "dark" ? "bg-yellow-600 text-black" : "bg-yellow-300 text-black"
            }`}
          >
            ⚠ WARNING: เอกสารนี้อาจมีการเปลี่ยนแปลงตามเนื้อหาของหลักสูตร
          </div>

          <div className="mt-6">
            <table
              className={`w-full border text-left rounded-lg shadow-lg overflow-hidden ${
                theme === "dark" ? "border-gray-700" : "border-gray-300"
              }`}
            >
              <thead className={`${theme === "dark" ? "bg-gray-800" : "bg-gray-200"}`}>
                <tr>
                  <th className="p-2">ตอน</th>
                  <th className="p-2">หัวข้อ</th>
                  <th className="p-2">วิดีโอ</th>
                  <th className="p-2">เอกสาร</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map((lesson) => (
                  <tr
                    key={lesson.id}
                    className={`border-t ${
                      theme === "dark" ? "border-gray-700" : "border-gray-300"
                    }`}
                  >
                    <td className="p-2 text-center">{lesson.id}</td>
                    <td className="p-2">{lesson.title}</td>
                    <td className="p-2 text-center">
                      <img
                        src={lesson.image}
                        alt={lesson.title}
                        className="w-24 rounded-lg cursor-pointer hover:opacity-80 transition"
                      />
                    </td>
                    <td className="p-2 text-center">
                      <a
                        href={lesson.docLink}
                        className="text-green-400 hover:underline"
                      >
                        อ่าน
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      <SupportMeButton />
    </div>
  );
};

export default PythonSeries;
