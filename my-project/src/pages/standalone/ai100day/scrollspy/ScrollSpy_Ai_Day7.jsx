import React, { useEffect, useState } from "react";

const headings = [
  { id: "what-is-loss", label: "Loss Function คืออะไร?" },
  { id: "loss-types", label: "ประเภทของ Loss Function" },
  { id: "what-is-optimization", label: "Optimization คืออะไร?" },
  { id: "gradient-descent", label: "Gradient Descent & Learning Rate" },
  { id: "gradient-problems", label: "Vanishing & Exploding Gradients" },
  { id: "insight", label: "Insight รวมทั้งหมด" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day7 = () => {
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visibleSections = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

        if (visibleSections.length > 0) {
          setActiveId(visibleSections[0].target.id);
        }
      },
      {
        rootMargin: "-10% 0px -20% 0px",
        threshold: 0.1,
      }
    );

    const elements = headings.map((h) => document.getElementById(h.id)).filter(Boolean);
    elements.forEach((el) => observer.observe(el));

    return () => {
      elements.forEach((el) => observer.unobserve(el));
    };
  }, []);

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
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

export default ScrollSpy_Ai_Day7;
