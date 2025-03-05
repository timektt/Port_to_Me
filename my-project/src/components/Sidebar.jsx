import React from "react";

const sidebarItems = [
  { id: "101", title: "Basic Python" },
  { id: "201", title: "Data" },
  { id: "202", title: "Visualization" },
  { id: "203", title: "Data Wrangling & Transform" },
  { id: "204", title: "Statistic Analysis" },
  { id: "205", title: "Statistic Learning" },
];

const Sidebar = ({ activeCourse }) => {
  return (
    <aside className="w-64 bg-gray-900 text-white min-h-screen p-6 border-r border-gray-700">
      <h2 className="text-xl font-bold mb-4">
        <span className="inline-block bg-gray-700 px-3 py-1 rounded-md">
          {activeCourse}
        </span>
      </h2>
      <ul className="space-y-2">
        {sidebarItems.map((item) => (
          <li
            key={item.id}
            className="p-2 rounded-lg cursor-pointer hover:bg-gray-700 transition"
          >
            {item.title}
          </li>
        ))}
      </ul>
    </aside>
  );
};

export default Sidebar;
