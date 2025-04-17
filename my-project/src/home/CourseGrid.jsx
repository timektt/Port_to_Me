import React from "react";
import { useNavigate } from "react-router-dom";
import { FaArrowRight } from "react-icons/fa";
import LatestUpdates from "./LatestUpdates";
import PopularTags from "../pages/courses/PopularTags";

const courses = [
  { id: "ai-series", image: "/ai_series.png", title: "Ai Series", description: "คอร์สเรียนพื้นฐาน Ai เชิงลึก" },
  { id: "python-series", image: "/python_1.png", title: "Python Series", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
  { id: "nodejs-series", image: "/node_1.png", title: "Node.js Series", description: "เรียนรู้การพัฒนา Backend ด้วย Node.js" }, // ✅ แก้ id ให้ตรงกับเส้นทาง
  { id: "reactjs-series", image: "/react_1.png", title: "React.js Series", description: "คอร์สสอนสร้างโปรเจกต์ด้วย React " },
  { id: "web-development", image: "/webdev_1.png", title: "Web Development 101", description: "คอร์สเรียนพื้นฐานสำหรับเริ่มต้นสร้างเว็บไซต์" },
  { id: "basic-programming", image: "/basicpro_1.png", title: "Basic Programming", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
];

const CourseGrid = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div className={`p-8 max-w-screen-lg mx-auto w-full ${theme === "dark" ? " text-white" : " text-black"}`}>
      <h2 className="text-2xl md:text-3xl font-bold text-left mb-6">🎓 Latest Courses</h2>

      {/* ✅ Grid แสดงคอร์สทั้งหมด */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses.map(course => (
          <div 
            key={course.id} 
            className={`p-4 rounded-lg shadow-lg transition transform hover:scale-105 cursor-pointer
              ${theme === "dark" ? " bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-100 text-black hover:bg-gray-200"}`}
            onClick={() => navigate(`/courses/${course.id}`)}
          >
            <img
              src={course.image}
              alt={course.title}
              className="w-full h-[200px] rounded-lg object-cover transition hover:opacity-90"
            />
            <h3 className="font-semibold text-lg mt-3">{course.title}</h3>
            <p className="text-sm mt-1 text-gray-400">{course.description}</p>
            <a className="text-green-500 hover:underline">อ่าน documents</a>
          </div>
        ))}
      </div>

      {/* ✅ ปุ่ม "ดูคอร์สทั้งหมด" ไปอยู่ฝั่งขวา */}
      <div className="w-full flex justify-end mt-6">
        <button 
          className={`flex items-center gap-3 px-6 py-3 rounded-lg shadow-lg text-lg font-semibold transition transform hover:scale-110
            ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-100 text-black hover:bg-gray-200"}`}
          onClick={() => navigate("/courses")}
          title="ดูคอร์สทั้งหมด"
        >
          ดูคอร์สทั้งหมด <FaArrowRight className="text-2xl" />
        </button>
      </div>

      {/* ✅ องค์ประกอบอื่นๆ บนหน้าแรก */}
      <LatestUpdates theme={theme} />
      <PopularTags theme={theme} />
    </div>
  );
};

export default CourseGrid;
