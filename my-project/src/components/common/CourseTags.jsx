import React from "react";
import { Link } from "react-router-dom";

const courses = [
  { id: "python-series", title: "Python Series", color: "bg-blue-500", path: "/courses/python-series/intro" },
  { id: "nodejs-series", title: "Node.js Series", color: "bg-green-500", path: "/courses/nodejs-series" },
  { id: "reactjs-series", title: "React.js Series", color: "bg-purple-500", path: "/courses/reactjs-series" },
  { id: "web-development", title: "Web Development", color: "bg-yellow-500", path: "/courses/web-development" },
  { id: "basic-programming", title: "Basic Programming", color: "bg-red-500", path: "/courses/basic-programming" },
  { id: "restful-api-graphql-series", title: "REST API & GraphQL", color: "bg-indigo-500", path: "/courses/restful-api-graphql-series" },
  { id: "ai", title: "AI", color: "bg-pink-500", path: "/courses/ai" },
  { id: "all-courses", title: "All Courses", color: "bg-gray-500", path: "/courses" },

];

const CourseTags = () => {
  return (
    <div className="flex flex-wrap gap-3 p-4 bg-gray-900 rounded-lg shadow-md">
      {courses.map((course) => (
        <Link 
          key={course.id} 
          to={course.path} 
          className={`px-4 py-2 text-white rounded-md ${course.color} hover:opacity-80 transition-all`}
        >
          {course.title}
        </Link>
      ))}
    </div>
  );
};

export default CourseTags;
