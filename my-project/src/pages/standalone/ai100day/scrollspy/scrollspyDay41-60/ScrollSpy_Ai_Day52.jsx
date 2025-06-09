"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไมต้องมี Self-Attention & Positional Encoding?" },
  { id: "self-attention", label: "2. Self-Attention คืออะไร?" },
  { id: "multi-head-self-attention", label: "3. Multi-Head Self-Attention" },
  { id: "self-attention-benefits", label: "4. ประโยชน์ของ Self-Attention" },
  { id: "positional-encoding-importance", label: "5. Positional Encoding: ทำไมจำเป็น?" },
  { id: "positional-encoding-types", label: "6. Types of Positional Encoding" },
  { id: "positional-vs-relative", label: "7. Positional Encoding vs Relative Positional Encoding" },
  { id: "contextualized-representation", label: "8. Insight: Self-Attention + Positional Encoding = Contextualized Representation" },
  { id: "research-evolution", label: "9. Research Evolution" },
  { id: "use-cases", label: "10. Use Cases ที่ Self-Attention สำคัญมาก" },
  { id: "insight-box", label: "11. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day52 = () => {
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

export default ScrollSpy_Ai_Day52;
