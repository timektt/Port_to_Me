import React from "react";
import { useNavigate } from "react-router-dom";
import { FaBook } from "react-icons/fa";

const courses = [
  { id: "python-series", image: "/python_1.png", title: "Python Series", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
  { id: "nodejs-series", image: "/node_1.png", title: "Node.js Series", description: "เรียนรู้การพัฒนา Backend ด้วย Node.js" },
  { id: "restful-api-graphql-series", image: "/api_1.png", title: "RESTful API & GraphQL", description: "คอร์สนี้เหมาะสำหรับทุกคนที่อยากเข้าใจ API และแนวคิดขึ้นสูง" },
  { id: "reactjs-series", image: "/react_1.png", title: "React.js Series", description: "คอร์สสอนสร้างโปรเจกต์ด้วย React" },
  { id: "web-development", image: "/webdev_1.png", title: "Web Development 101", description: "คอร์สเรียนพื้นฐานสำหรับเริ่มต้นสร้างเว็บไซต์" },
  { id: "basic-programming", image: "/basicpro_1.png", title: "Basic Programming", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
];

const AllCourses = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>

      <main className="flex-1 p-6 pt-20">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold text-center mb-6 flex items-center justify-center gap-2">
            <FaBook className="text-b-800" /> Courses
          </h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {courses.map((course) => (
              <div
                key={course.id}
                className={`p-6 rounded-lg shadow-lg cursor-pointer transition-transform transform hover:scale-105 ${
                  theme === "dark" ? "bg-gray-800" : "bg-gray-200"
                }`}
                onClick={() => navigate(`/courses/${course.id}`)}
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
