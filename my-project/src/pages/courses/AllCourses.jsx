import React from "react";
import { useNavigate } from "react-router-dom";

const courses = [
  { id: "python-series", image: "/Python.jpg", title: "Python Series", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
  { id: "nodejs-series", image: "/nodejs.jpg", title: "Node.js Series", description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเรียนพื้นฐานการใช้ Node.js" }, // ✅ แก้ ID เป็น "nodejs-series"
  { id: "restful-api-graphql-series", image: "/Api.jpg", title: "GoAPI Essential", description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเข้าใจ API และ Backend ผ่าน Go" },
  { id: "reactjs-series", image: "/react.png", title: "React.js", description: "คอร์สสอนสร้างโปรเจกต์ด้วย React" },
  { id: "web-development", image: "/Web.jpg", title: "Web Development 101", description: "คอร์สเรียนพื้นฐานสำหรับเริ่มต้นสร้างเว็บไซต์" },
  { id: "basic-programming", image: "/Basic.jpg", title: "Basic Programming", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
];

const AllCourses = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      
      {/* ✅ Main Content */}
      <main className="flex-1 p-6 pt-16"> {/* ⬅️ `pt-16` ป้องกัน Navbar ทับเนื้อหา */}
        <div className="max-w-5xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold text-center mb-6">📚 Courses</h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {courses.map((course) => (
              <div
                key={course.id}
                className={`p-6 rounded-lg shadow-lg cursor-pointer transition-transform transform hover:scale-105 ${
                  theme === "dark" ? "bg-gray-800" : "bg-gray-200"
                }`}
                onClick={() => navigate(`/courses/${course.id}`)} // ✅ ใช้ navigate เพื่อให้ไปถูกต้อง
              >
                <img
                  src={course.image}
                  alt={course.title}
                  className="w-full h-[200px] rounded-lg object-cover transition hover:opacity-90"
                />
                <h2 className="text-2xl font-bold mt-3">{course.title}</h2>
                <p className="text-sm mt-2">{course.description}</p>
                <a href={`/courses/${course.id}`} className="text-green-500 hover:underline">ดูรายละเอียด</a>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default AllCourses;
