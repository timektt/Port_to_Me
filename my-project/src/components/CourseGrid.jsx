import React from "react";
import { useNavigate } from "react-router-dom";
import LatestUpdates from "./LatestUpdates";
import PopularTags from "./PopularTags";

const courses = [
  { id: "python-series", image: "/Python.jpg", title: "Python Series", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
  { id: "cpp-dsa", image: "/C++.jpg", title: "C++ Data Structure & Algorithm", description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเรียนพื้นฐานการเขียนโปรแกรมและ Algorithm" },
  { id: "goapi-essential", image: "/Api.jpg", title: "GoAPI Essential", description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเข้าใจ API และ Backend ผ่าน Go" },
  { id: "vue-firebase", image: "/Vue.jpg", title: "Vue Firebase Masterclass", description: "คอร์สสอนสร้างโปรเจกต์ด้วย Vue และ Firebase" },
  { id: "web-development", image: "/Web.jpg", title: "Web Development 101", description: "คอร์สเรียนพื้นฐานสำหรับเริ่มต้นสร้างเว็บไซต์" },
  { id: "basic-programming", image: "/Basic.jpg", title: "Basic Programming", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
];

const CourseGrid = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div className={`p-8 max-w-screen-lg mx-auto w-full ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"}`}>
      <h2 className="text-2xl md:text-3xl font-bold text-left mb-6">🎓 Latest Courses</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses.map(course => (
          <div key={course.id} className={`p-4 rounded-lg shadow-lg ${theme === "dark" ? "bg-gray-800" : "bg-gray-100"}`}>
            <img
              src={course.image}
              alt={course.title}
              className="w-full h-[180px] rounded-lg object-cover cursor-pointer hover:scale-105 transition"
              onClick={() => navigate(`/courses/${course.id}`)}
            />
            <h3 className={`font-semibold text-md mt-3 ${theme === "dark" ? "text-white" : "text-black"}`}>{course.title}</h3>
            <p className={`text-sm mt-1 ${theme === "dark" ? "text-gray-400" : "text-gray-600"}`}>{course.description}</p>
            <a
              onClick={() => navigate(`/courses/${course.id}`)}
              className="text-green-500 cursor-pointer hover:underline"
            >
              อ่าน documents
            </a>
          </div>
        ))}
      </div>

      <LatestUpdates theme={theme} />
      <PopularTags theme={theme} />
    </div>
  );
};

export default CourseGrid;
