import React from "react";

const sidebarItems = [
  { id: "101", title: "Basic Python" },
  { id: "201", title: "Data" },
  { id: "202", title: "Visualization" },
  { id: "203", title: "Data Wrangling & Transform" },
  { id: "204", title: "Statistic Analysis" },
  { id: "205", title: "Statistic Learning" },
];

const Sidebar = ({ activeCourse, theme }) => {
  return (
    <aside
      className={`w-64 min-h-screen p-6 border-r ${
        theme === "dark"
          ? "bg-gray-900 text-white border-gray-700"
          : "bg-gray-200 text-black border-gray-300"
      }`}
    >
      <h2 className="text-xl font-bold mb-4">
        <span
          className={`inline-block px-3 py-1 rounded-md ${
            theme === "dark" ? "bg-gray-700 text-white" : "bg-gray-300 text-black"
          }`}
        >
          {activeCourse}
        </span>
      </h2>
      <ul className="space-y-2">
        {sidebarItems.map((item) => (
          <li
            key={item.id}
            className={`p-2 rounded-lg cursor-pointer transition ${
              theme === "dark"
                ? "hover:bg-gray-700"
                : "hover:bg-gray-300"
            }`}
          >
            {item.title}
          </li>
        ))}
      </ul>
    </aside>
  );
};

export default Sidebar;
