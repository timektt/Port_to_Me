import React, { useEffect, useState } from "react";

const headings = [
  { id: "what-is-regularization", label: "Regularization คืออะไร?" },
  { id: "l1-vs-l2", label: "L1 vs L2 Regularization" },
  { id: "dropout", label: "Dropout" },
  { id: "early-stopping", label: "Early Stopping" },
  { id: "data-augmentation", label: "Data Augmentation" },
  { id: "batch-normalization", label: "Batch Normalization" },
  { id: "visualization", label: "Visualization" },
  { id: "combined-code", label: "ตัวอย่างโค้ดรวม" },
  { id: "insight", label: "Insight รวมทั้งหมด" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day9 = () => {
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

export default ScrollSpy_Ai_Day9;
