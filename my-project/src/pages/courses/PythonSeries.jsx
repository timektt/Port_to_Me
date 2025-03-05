import React from "react";
import { useNavigate } from "react-router-dom";
import { FaHome, FaAngleRight } from "react-icons/fa";
import Sidebar from "../../components/Sidebar"; // ✅ ใช้ Sidebar เป็น Component
import SupportMeButton from "../../components/SupportMeButton"; // ✅ Footer เอาออก

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

const PythonSeries = () => {
  const navigate = useNavigate();

  return (
    <div className="bg-gray-900 min-h-screen flex">
      {/* ✅ Sidebar (ทำให้ Sidebar ไม่หาย) */}
      <Sidebar activeCourse="Python Series" />

      {/* ✅ Main Content */}
      <main className="flex-1 p-6">
        <div className="max-w-3xl mx-auto">
          {/* ✅ Breadcrumb + Home Button */}
          <div className="flex items-center gap-2 mb-4">
            <button
              className="bg-gray-700 text-white p-2 rounded-md hover:bg-gray-600 transition"
              onClick={() => navigate("/")}
            >
              <FaHome size={18} />
            </button>
            <FaAngleRight className="text-gray-400" />
            <span className="bg-gray-700 px-3 py-1 rounded-md text-white font-semibold">
              Python Series
            </span>
          </div>

          {/* ✅ หัวข้อหลัก */}
          <h1 className="text-4xl font-bold text-left mt-4">Python Series</h1>

          {/* ✅ Warning Box */}
          <div className="bg-yellow-600 text-black p-3 rounded-md mt-4">
            ⚠ WARNING: เอกสารนี้อาจมีการเปลี่ยนแปลงตามเนื้อหาของหลักสูตร
          </div>

          {/* ✅ Table */}
          <div className="mt-6">
            <table className="w-full border border-gray-700 text-left rounded-lg shadow-lg overflow-hidden">
              <thead className="bg-gray-800">
                <tr>
                  <th className="p-2">ตอน</th>
                  <th className="p-2">หัวข้อ</th>
                  <th className="p-2">วิดีโอ</th>
                  <th className="p-2">เอกสาร</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map((lesson) => (
                  <tr key={lesson.id} className="border-t border-gray-700">
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
                      <a href={lesson.docLink} className="text-green-400 hover:underline">
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

      {/* ✅ เอา Footer ออกไป ใช้ที่ `App.jsx` แทน */}

      {/* ✅ ปุ่ม SupportMe */}
      <SupportMeButton />
    </div>
  );
};

export default PythonSeries;
