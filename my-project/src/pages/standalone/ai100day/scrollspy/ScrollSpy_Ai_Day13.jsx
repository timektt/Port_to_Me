import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "ทำไม Interpretability ถึงสำคัญ?" },
  { id: "blackbox", label: "ปัญหาจากโมเดล Black-box" },
  { id: "global", label: "Global Interpretability" },
  { id: "local", label: "Local Interpretability" },
  { id: "image-text", label: "อธิบายข้อมูลภาพและข้อความ" },
  { id: "tradeoff", label: "Accuracy vs Explainability" },
  { id: "best-practice", label: "Best Practice" },
  { id: "insight", label: "Insight เปรียบเทียบ" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day13 = () => {
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
    <nav className="flex flex-col gap-2 text-sm max-w-[220px]">
      {headings.map((h) => (
        <button
          key={h.id}
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

export default ScrollSpy_Ai_Day13;
