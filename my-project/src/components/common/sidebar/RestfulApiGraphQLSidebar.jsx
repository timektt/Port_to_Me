import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaTimes, FaChevronDown, FaChevronRight } from "react-icons/fa";

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


const RestfulApiGraphQLSidebar = ({ theme, sidebarOpen, setSidebarOpen }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  
    return (
      <>
        {sidebarOpen && (
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 md:hidden" onClick={() => setSidebarOpen(false)} />
        )}
    
        <aside
          className={`fixed top-16 left-0 w-64 h-[calc(100vh-70px)] overflow-y-auto z-50 p-4 transition-transform duration-300 ease-in-out
            ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
            ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"} shadow-lg pb-20`} // ✅ เพิ่ม pb-20
        >
          <button className="md:hidden absolute top-4 right-4 text-xl" onClick={() => setSidebarOpen(false)}>
            <FaTimes />
          </button>
    
          <h2 className="text-xl font-bold mb-4">
            <span className={`inline-block px-3 py-1 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-200"}`}>
              RESTful API & GraphQL Series
            </span>
          </h2>
    
          <ul className="space-y-2 mt-4 mb-24">  {/* ✅ เพิ่ม mb-24 เพื่อให้ Scroll ได้ */}
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
                        onClick={() => navigate(subItem.path)}
                      >
                        {subItem.title}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </aside>
      </>
    );
    
  };

export default RestfulApiGraphQLSidebar;
