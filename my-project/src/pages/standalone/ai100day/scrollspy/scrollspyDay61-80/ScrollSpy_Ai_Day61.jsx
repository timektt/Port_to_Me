"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: Reinforcement Learning คืออะไร?" },
  { id: "comparison", label: "2. เปรียบเทียบกับ Learning แบบอื่น" },
  { id: "components", label: "3. ส่วนประกอบพื้นฐานของ RL" },
  { id: "loop", label: "4. Loop การทำงานของ RL" },
  { id: "types", label: "5. Types of Reinforcement Learning" },
  { id: "examples", label: "6. ตัวอย่างปัญหา RL ที่เป็นมาตรฐาน" },
  { id: "advantages", label: "7. จุดแข็งของ RL" },
  { id: "limitations", label: "8. ข้อจำกัดของ RL" },
  { id: "real-world", label: "9. ใช้งานในโลกจริง" },
  { id: "insight-box", label: "10. Insight Box" },
  { id: "references", label: "11. Academic References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day61 = () => {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
    const handleScroll = () => {
      let closestId = "";
      let minOffset = Infinity;

      headings.forEach(({ id }) => {
        const el = document.getElementById(id);
        if (el) {
          const offset = Math.abs(el.getBoundingClientRect().top - 120);
          if (offset < minOffset) {
            minOffset = offset;
            closestId = id;
          }
        }
      });

      setActiveId(closestId);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <nav className="flex flex-col gap-2 text-sm max-w-[240px]">
      {headings.map((h, index) => (
        <button
          key={index}
          onClick={() => scrollToSection(h.id)}
          className={`text-right transition-all hover:underline truncate whitespace-nowrap text-ellipsis ${
            activeId === h.id ? "font-bold text-yellow-400" : "text-gray-400"
          }`}
        >
          {h.label}
        </button>
      ))}
    </nav>
  );
};

export default ScrollSpy_Ai_Day61;
