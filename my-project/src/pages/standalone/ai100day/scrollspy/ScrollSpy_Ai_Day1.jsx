// ScrollSpy_Ai_Day1.jsx
import React, { useEffect, useState } from "react";

const headings = [
  { id: "vector", label: "Vector คืออะไร" },
  { id: "matrix", label: "Matrix คืออะไร" },
  { id: "examples", label: "ตัวอย่างใน Python" },
  { id: "why-ai", label: "ทำไม AI ถึงใช้" },
  { id: "exercise", label: "แบบฝึกหัดเสริม" },
  { id: "summary", label: "สรุปตารางรวม" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day1 = () => {
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    const handleScroll = () => {
      let current = null;
      for (const heading of headings) {
        const el = document.getElementById(heading.id);     
        if (el && el.getBoundingClientRect().top < window.innerHeight / 4) {
          current = heading.id;
        }
      }
      setActiveId(current);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="flex flex-col gap-2">
      {headings.map((h) => (
        <button
          key={h.id}
          onClick={() => scrollToSection(h.id)}
          className={`text-sm text-right transition-all hover:underline ${
            activeId === h.id ? "font-bold text-yellow-400" : "text-gray-400"
          }`}
        >
          {h.label}
        </button>
      ))}
    </div>
  );
};

export default ScrollSpy_Ai_Day1;