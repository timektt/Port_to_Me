"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Vision-Language จึงสำคัญ" },
  { id: "captioning-definition", label: "2. Image Captioning คืออะไร?" },
  { id: "early-architectures", label: "3. Early Architectures" },
  { id: "vlp", label: "4. Vision-Language Pretraining (VLP)" },
  { id: "core-architectures", label: "5. Core Architectures" },
  { id: "pretraining-objectives", label: "6. Pretraining Objectives" },
  { id: "use-cases", label: "7. Use Cases & Real-World Applications" },
  { id: "real-systems", label: "8. ตัวอย่างระบบจริง" },
  { id: "challenges", label: "9. ความท้าทาย" },
  { id: "future-trends", label: "10. Future Trends" },
  { id: "insight-box", label: "11. Summary Insight Box" },
  { id: "references", label: "12. References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day56 = () => {
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

export default ScrollSpy_Ai_Day56;
