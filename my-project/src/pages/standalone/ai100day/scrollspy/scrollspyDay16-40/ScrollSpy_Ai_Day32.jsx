import React, { useEffect, useState } from "react";

const headings = [
  { id: "why-positional", label: "1. ทำไมต้องมี Positional Encoding?" },
  { id: "abs-vs-rel", label: "2. Absolute vs Relative Encoding" },
  { id: "sinusoidal", label: "3. Sinusoidal Encoding (Vaswani et al., 2017)" },
  { id: "learned", label: "4. Learned Positional Embeddings" },
  { id: "relative", label: "5. Relative Positional Encoding (Shaw et al., 2018)" },
  { id: "rope", label: "6. Rotary Positional Embedding (RoPE, Su et al., 2021)" },
  { id: "comparison", label: "7. Comparison Table" },
  { id: "visualization", label: "8. Visualization" },
  { id: "research", label: "9. Academic Research" },
  { id: "practical", label: "10. Practical Considerations" },
  { id: "limitations", label: "11. Limitations" },
  { id: "insight-2", label: "12. Insight Box " },
  { id: "summary", label: "13. Summary" },
  { id: "quiz", label: "MiniQuiz: Positional Encoding" },
];

const ScrollSpy_Ai_Day32 = () => {
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

export default ScrollSpy_Ai_Day32;
