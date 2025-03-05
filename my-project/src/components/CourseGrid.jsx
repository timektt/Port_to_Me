import React from "react";
import { useNavigate } from "react-router-dom";
import LatestUpdates from "./LatestUpdates";
import PopularTags from "./PopularTags";

const courses = [
  { 
    id: "python-series",
    image: "/Python.jpg",
    title: "Python Series", 
    description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" 
  },
  { 
    id: "cpp-dsa",
    image: "/C++.jpg",
    title: "C++ Data Structure & Algorithm", 
    description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเรียนพื้นฐานการเขียนโปรแกรมและ Algorithm" 
  },
  { 
    id: "goapi-essential",
    image: "/Api.jpg",
    title: "GoAPI Essential", 
    description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเข้าใจ API และ Backend ผ่าน Go" 
  },
  { 
    id: "vue-firebase",
    image: "/Vue.jpg",
    title: "Vue Firebase Masterclass", 
    description: "คอร์สสอนสร้างโปรเจกต์ด้วย Vue และ Firebase" 
  },
  { 
    id: "web-development",
    image: "/Web.jpg",
    title: "Web Development 101", 
    description: "คอร์สเรียนพื้นฐานสำหรับเริ่มต้นสร้างเว็บไซต์" 
  },
  { 
    id: "basic-programming",
    image: "/Basic.jpg",
    title: "Basic Programming", 
    description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" 
  },
];

const CourseGrid = () => {
  const navigate = useNavigate();

  return (
    <div className="p-8 bg-gray-900 max-w-screen-lg mx-auto w-full">
      <h2 className="text-2xl md:text-3xl font-bold text-white text-left mb-6">
        🎓 Latest Courses
      </h2>
      <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses.map((course) => (
          <div key={course.id} className="bg-gray-800 p-4 rounded-lg shadow-lg">
            {/* ✅ คลิกได้เฉพาะรูป */}
            <img
              src={course.image}
              alt={course.title}
              className="w-full h-[180px] md:h-[220px] rounded-lg object-cover cursor-pointer hover:scale-105 transition"
              onClick={() => navigate(`/courses/${course.id}`)}
            />
            <h3 className="text-white font-semibold text-md md:text-lg mt-3">
              {course.title}
            </h3>
            <p className="text-gray-400 text-sm md:text-base mt-1">
              {course.description}
            </p>
            {/* ✅ คลิกได้เฉพาะปุ่ม "อ่าน documents" */}
            <a 
              className="text-green-400 text-sm md:text-base mt-2 inline-block cursor-pointer hover:underline"
              onClick={(e) => {
                e.preventDefault();
                navigate(`/courses/${course.id}`);
              }}
            >
              อ่าน documents
            </a>
          </div>
        ))}
      </div>

      <LatestUpdates />
      <PopularTags />
    </div>
  );
};

export default CourseGrid;
