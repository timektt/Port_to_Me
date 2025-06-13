"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Seq2Seq ถึงสำคัญ" },
  { id: "what-is-seq2seq", label: "2. What is Sequence-to-Sequence?" },
  { id: "transformer-seq2seq", label: "3. Transformer as a Seq2Seq Model" },
  { id: "seq2seq-applications", label: "4. Applications of Seq2Seq Transformers" },
  { id: "key-techniques", label: "5. Key Techniques in Transformer-based Seq2Seq" },
  { id: "advanced-architectures", label: "6. Advanced Architectures" },
  { id: "training-optimization", label: "7. Training & Optimization Tips" },
  { id: "use-cases", label: "8. Use Cases & Industry Impact" },
  { id: "limitations-challenges", label: "9. Limitations & Challenges" },
  { id: "future-directions", label: "10. Future Directions" },
  { id: "insight-box", label: "11. Summary Insight Box" },
  { id: "references", label: "12. References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day55 = () => {
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

export default ScrollSpy_Ai_Day55;
