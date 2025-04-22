import React from "react";
import { useNavigate } from "react-router-dom";
import { FaArrowRight } from "react-icons/fa";
import LatestUpdates from "./LatestUpdates";
import PopularTags from "../pages/courses/PopularTags";
import LatestCourses from "./LatestCourses"; // ✅ ชื่อถูกต้อง

const CourseGrid = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div className={`p-8 max-w-screen-lg mx-auto w-full ${theme === "dark" ? "text-white" : "text-black"}`}>
      <h2 className="text-2xl md:text-3xl font-bold text-left mb-6"></h2>

      {/* ✅ แสดงคอร์สล่าสุดแบบ dynamic */}
      <LatestCourses theme={theme} />

      {/* ✅ ปุ่ม "ดูคอร์สทั้งหมด" ไปที่ AllCourses */}
      <div className="w-full flex justify-end mt-6">
        <button
          className={`flex items-center gap-3 px-6 py-3 rounded-lg shadow-lg text-lg font-semibold transition transform hover:scale-110
            ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-100 text-black hover:bg-gray-200"}`}
          onClick={() => navigate("/courses/all-courses")}
          title="ดูคอร์สทั้งหมด"
        >
          ดูคอร์สทั้งหมด <FaArrowRight className="text-2xl" />
        </button>
      </div>

      {/* ✅ องค์ประกอบอื่น ๆ ที่ยังใช้งาน */}
      <LatestUpdates theme={theme} />
      <PopularTags theme={theme} />
    </div>
  );
};

export default CourseGrid;
