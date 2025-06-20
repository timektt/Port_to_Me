"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. Text-to-Image Generation คืออะไร?" },
  { id: "evolution", label: "2. Evolution ของ Generative Models สำหรับภาพ" },
  { id: "diffusion-basics", label: "3. แนวคิดพื้นฐานของ Diffusion Models" },
  { id: "stable-diffusion-architecture", label: "4. Stable Diffusion: สถาปัตยกรรม" },
  { id: "prompt-engineering", label: "5. Prompt Engineering" },
  { id: "variants", label: "6. Variants & Improvements" },
  { id: "real-world-applications", label: "7. Applications ในโลกจริง" },
  { id: "ethics", label: "8. Ethical Considerations" },
  { id: "future-trends", label: "9. Future Trends" },
  { id: "insight-box", label: "10. Insight Box" },
  { id: "references", label: "11. Academic References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day60 = () => {
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

export default ScrollSpy_Ai_Day60;
