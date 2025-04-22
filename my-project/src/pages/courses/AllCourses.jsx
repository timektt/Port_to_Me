import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaBook } from "react-icons/fa";

const AllCourses = ({ theme }) => {
  const [courses, setCourses] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const res = await fetch("/data/courses.json");
        const data = await res.json();
        const sorted = data.sort((a, b) => new Date(b.date) - new Date(a.date));
        setCourses(sorted);
      } catch (err) {
        console.error("❌ Error loading courses.json:", err);
      }
    };

    fetchCourses();
  }, []);

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
                onClick={() => navigate(course.path)}
              >
                <img
                  src={course.image}
                  alt={course.title}
                  className="w-full h-[200px] rounded-lg object-cover transition hover:opacity-90"
                />
                <h2 className="text-2xl font-bold mt-3">{course.title}</h2>
                <p className="text-sm mt-2">{course.description}</p>
                <a href={course.path} className="text-green-500 hover:underline">ดูรายละเอียด</a>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default AllCourses;
