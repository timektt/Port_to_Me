"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม BERT & GPT เปลี่ยนวงการ NLP?" },
  { id: "transformer-recap", label: "2. Transformer Architecture Recap" },
  { id: "bert", label: "3. BERT: Bidirectional Encoder Representations from Transformers" },
  { id: "gpt", label: "4. GPT: Generative Pre-trained Transformer" },
  { id: "bert-vs-gpt", label: "5. BERT vs GPT: เปรียบเทียบ" },
  { id: "key-innovations", label: "6. Key Innovations" },
  { id: "architectural-variants", label: "7. Architectural Variants & Evolutions" },
  { id: "practical-engineering", label: "8. Practical Engineering Considerations" },
  { id: "use-cases", label: "9. Use Cases & Industry Impact" },
  { id: "limitations-ethics", label: "10. Limitations & Ethical Considerations" },
  { id: "insight-box", label: "11. Summary Insight Box" },
  { id: "references", label: "12. References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day53 = () => {
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

export default ScrollSpy_Ai_Day53;
