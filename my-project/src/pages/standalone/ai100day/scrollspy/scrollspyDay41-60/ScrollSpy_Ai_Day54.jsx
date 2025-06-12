"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Fine-tuning จึงสำคัญ" },
  { id: "pretraining-vs-finetuning", label: "2. Pretraining vs Fine-tuning: Workflow เปรียบเทียบ" },
  { id: "finetuning-strategies", label: "3. Types of Fine-tuning Strategies" },
  { id: "finetuning-pipeline", label: "4. Fine-tuning Pipeline" },
  { id: "practical-examples", label: "5. Practical Examples" },
  { id: "advanced-topics", label: "6. Advanced Topics" },
  { id: "cost-considerations", label: "7. Cost Considerations" },
  { id: "engineering-best-practices", label: "8. Engineering Best Practices" },
  { id: "use-cases", label: "9. Use Cases & Industry Impact" },
  { id: "limitations-ethics", label: "10. Limitations & Ethical Considerations" },
  { id: "insight-box", label: "11. Summary Insight Box" },
  { id: "references", label: "12. References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day54 = () => {
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

export default ScrollSpy_Ai_Day54;
