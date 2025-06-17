"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. Multi-Modal Models คืออะไร?" },
  { id: "concept", label: "2. แนวคิดเบื้องหลัง Multi-Modal Learning" },
  { id: "clip", label: "3. CLIP (Contrastive Language-Image Pretraining)" },
  { id: "flamingo", label: "4. Flamingo (DeepMind, 2022)" },
  { id: "clip-vs-flamingo", label: "5. เปรียบเทียบ CLIP vs Flamingo" },
  { id: "use-cases", label: "6. การใช้งานจริง" },
  { id: "vl-foundations", label: "7. Vision-Language Foundation Models" },
  { id: "challenges-future", label: "8. ความท้าทาย & อนาคต" },
  { id: "insight-box", label: "9. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day57 = () => {
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

export default ScrollSpy_Ai_Day57;
