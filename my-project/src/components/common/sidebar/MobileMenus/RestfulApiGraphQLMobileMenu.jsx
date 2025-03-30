import React, { useState } from "react";
import { FaTimes, FaChevronDown, FaChevronRight, FaArrowLeft, FaSun, FaMoon } from "react-icons/fa";
import { useNavigate, useLocation } from "react-router-dom"; // ✅ แก้ไข import

const sidebarItems = [
  {
    id: "101",
    title: "101: Introduction to API",
    subItems: [
      { id: "101-1", title: "What is an API?", path: "/courses/restful-api-graphql-series/intro" },
      { id: "101-2", title: "REST vs GraphQL", path: "/courses/restful-api-graphql-series/rest-vs-graphql" },
      { id: "101-3", title: "How APIs Work", path: "/courses/restful-api-graphql-series/how-apis-work" },
      { id: "101-4", title: "Types of APIs (Public, Private, Partner, Composite)", path: "/courses/restful-api-graphql-series/api-types" },
      { id: "101-5", title: "API Documentation & Tools", path: "/courses/restful-api-graphql-series/api-documentation" },
    ],
  },
  {
    id: "201",
    title: "201: RESTful API",
    subItems: [
      { id: "201-1", title: "RESTful API Basics", path: "/courses/restful-api-graphql-series/rest-basics" },
      { id: "201-2", title: "Building RESTful API with Node.js", path: "/courses/restful-api-graphql-series/rest-nodejs" },
      { id: "201-3", title: "CRUD Operations in REST", path: "/courses/restful-api-graphql-series/rest-crud" },
      { id: "201-4", title: "Handling Errors in REST API", path: "/courses/restful-api-graphql-series/rest-error-handling" },
      { id: "201-5", title: "Versioning in REST API", path: "/courses/restful-api-graphql-series/rest-versioning" },
    ],
  },
  {
    id: "202",
    title: "202: GraphQL",
    subItems: [
      { id: "202-1", title: "GraphQL Basics", path: "/courses/restful-api-graphql-series/graphql-basics" },
      { id: "202-2", title: "Building GraphQL API", path: "/courses/restful-api-graphql-series/graphql-api" },
      { id: "202-3", title: "Queries and Mutations", path: "/courses/restful-api-graphql-series/graphql-queries-mutations" },
      { id: "202-4", title: "GraphQL Schema & Resolvers", path: "/courses/restful-api-graphql-series/graphql-schema-resolvers" },
      { id: "202-5", title: "GraphQL vs REST: Pros & Cons", path: "/courses/restful-api-graphql-series/graphql-vs-rest" },
    ],
  },
  {
    id: "203",
    title: "203: API Security",
    subItems: [
      { id: "203-1", title: "Authentication & Authorization", path: "/courses/restful-api-graphql-series/api-security" },
      { id: "203-2", title: "Rate Limiting & CORS", path: "/courses/restful-api-graphql-series/rate-limiting" },
      { id: "203-3", title: "OAuth & API Keys", path: "/courses/restful-api-graphql-series/oauth-api-keys" },
      { id: "203-4", title: "JWT & Session Management", path: "/courses/restful-api-graphql-series/jwt-session" },
      { id: "203-5", title: "Best Practices for API Security", path: "/courses/restful-api-graphql-series/api-security-best-practices" },
    ],
  },
  {
    id: "204",
    title: "204: Advanced API Concepts",
    subItems: [
      { id: "204-1", title: "API Gateways & Microservices", path: "/courses/restful-api-graphql-series/api-gateways-microservices" },
      { id: "204-2", title: "GraphQL Subscriptions (Real-time API)", path: "/courses/restful-api-graphql-series/graphql-subscriptions" },
      { id: "204-3", title: "API Performance Optimization", path: "/courses/restful-api-graphql-series/api-performance" },
      { id: "204-4", title: "Testing & Monitoring APIs", path: "/courses/restful-api-graphql-series/api-testing-monitoring" },
      { id: "204-5", title: "Deploying & Scaling APIs", path: "/courses/restful-api-graphql-series/api-deployment-scaling" },
    ],
  },
];

const ReactJsMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
   const location = useLocation();
   const [expandedSections, setExpandedSections] = useState({});
 
   const toggleTheme = () => {
     const newTheme = theme === "dark" ? "light" : "dark";
     setTheme(newTheme);
     localStorage.setItem("theme", newTheme);
   };
 
   const toggleSection = (id) => {
     setExpandedSections((prev) => ({
       ...prev,
       [id]: !prev[id],
     }));
   };
 
   return (
     <div className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 
       ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} overflow-y-auto pb-20`}>
 
       {/* ✅ ปุ่มปิดเมนู (X) */}
       <button 
         className={`absolute right-4 top-4 text-2xl transition-colors duration-200 
           ${theme === "dark" ? "text-white hover:text-gray-400" : "text-black hover:text-gray-600"}`}
         onClick={onClose}
       >
         <FaTimes />
       </button>
 
       {/* ✅ โลโก้ + Superbear + ปุ่ม Dark/Light Mode */}
       <div className="mt-6 flex items-center mb-3">
         <img src="/spm2.jpg" alt="Logo" className="w-8 h-8 mr-2 object-cover rounded-full" />
         <div className="flex items-center space-x-2">
           <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
             Superbear
           </span>
           <button className="cursor-pointer transition-transform transform hover:scale-110" onClick={toggleTheme}>
             {theme === "dark" ? <FaSun className="text-yellow-400 text-2xl" /> : <FaMoon className="text-blue-400 text-2xl" />}
           </button>
         </div>
       </div>
       <button
        className={`w-full text-left text-sm font-medium px-5 py-3 rounded-lg mb-4 transition 
          ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        onClick={() => {
          navigate("/courses/restful-api-graphql-series");
          onClose();
        }}
      >
         RestfulApiGraphQLSeries
      </button>

 
       {/* ✅ รายการบทเรียน (Dropdown) */}
       <ul className="space-y-2 mt-4">
         {sidebarItems.map((item) => (
           <li key={item.id} className="border-b border-gray-700">
             <button
               className="flex items-center justify-between w-full p-3 rounded-lg transition duration-300 ease-in-out
                 hover:bg-gray-700 hover:shadow-lg text-left"
               onClick={() => toggleSection(item.id)}
             >
               {item.title}
               {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
             </button>
 
             {expandedSections[item.id] && (
               <ul className="pl-5 space-y-2 mt-2">
                 {item.subItems.map((subItem) => (
                   <li
                     key={subItem.id}
                     className={`p-2 rounded-lg cursor-pointer transition duration-200 ${
                       location.pathname === subItem.path ? "bg-green-500 text-white font-bold" : "hover:bg-gray-600"
                     }`}
                     onClick={() => { navigate(subItem.path); onClose(); }}
                   >
                     {subItem.title}
                   </li>
                 ))}
               </ul>
             )}
           </li>
         ))}
       </ul>
     </div>
   );
 };
 
// ✅ แก้ชื่อ component ที่ export
export default ReactJsMobileMenu;
