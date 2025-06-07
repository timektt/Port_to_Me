"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไมต้องมี Transformer?" },
  { id: "origin-paper", label: "2. Paper ต้นกำเนิด" },
  { id: "architecture-overview", label: "3. Architecture Overview ของ Transformer" },
  { id: "self-attention", label: "4. Self-Attention Mechanism" },
  { id: "positional-encoding", label: "5. Positional Encoding" },
  { id: "ffn", label: "6. Feedforward Neural Network (FFN)" },
  { id: "residual-layernorm", label: "7. Residual Connection & Layer Normalization" },
  { id: "benefits-transformer", label: "8. ประโยชน์หลักของ Transformer" },
  { id: "use-cases", label: "9. Use Cases สำคัญ" },
  { id: "research-benchmarks", label: "10. Research Benchmarks & Paper Timeline" },
  { id: "insight-box", label: "11. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day51 = () => {
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

export default ScrollSpy_Ai_Day51;
