import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const LastCourses = ({ theme }) => {
  const [courses, setCourses] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const res = await fetch("/data/courses.json");
        const data = await res.json();
        const sortedCourses = data.sort((a, b) => new Date(b.date) - new Date(a.date));
        setCourses(sortedCourses.slice(0, 6));
      } catch (err) {
        console.error("❌ Error loading courses.json:", err);
      }
    };

    fetchCourses();
  }, []);

  return (
    <div className="latest-courses mt-8">
      <h2 className=" mb-20">
         
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses.map((course) => (
          <div
            key={course.id}
            className={`p-4 rounded-lg shadow-lg transition transform hover:scale-105 cursor-pointer
              ${theme === "dark" ? " bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-100 text-black hover:bg-gray-200"}`}
            onClick={() => navigate(course.path)}
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

      <div className="w-full flex justify-end mt-6">
 
      </div>
    </div>
  );
};

export default LastCourses;
