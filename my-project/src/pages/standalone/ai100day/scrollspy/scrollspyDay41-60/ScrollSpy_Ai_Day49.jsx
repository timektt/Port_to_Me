"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "basic-concept", label: "1. Encoder-Decoder Overview" },
  { id: "encoder-components", label: "2. Sequence-to-Sequence Models" },
  { id: "decoder-components", label: "3. Attention Mechanism in Encoder-Decoder" },
  { id: "architecture-problems", label: "4. Use Case: Machine Translation" },
  { id: "attention-extension", label: "5. Use Case: Speech Recognition" },
  { id: "use-cases", label: "6. Use Case: Text Summarization" },
  { id: "comparison-transformer", label: "7. Comparison with Transformer Models" },
  { id: "references", label: "8. References" },
  { id: "insight-box", label: "9. Insight Box " },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day49 = () => {
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

export default ScrollSpy_Ai_Day49;
