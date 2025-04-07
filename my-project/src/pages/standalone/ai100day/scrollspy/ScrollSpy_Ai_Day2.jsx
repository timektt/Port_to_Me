import React, { useEffect, useState } from "react";

const headings = [
  { id: "vector-addition", label: "Vector Addition" },
  { id: "vector-visualization", label: "Vector Visualization" },
  { id: "vector-subtraction", label: "Vector Subtraction" },
  { id: "scalar-mult", label: "Scalar Multiplication" },
  { id: "examples", label: "Examples" },
  { id: "concept", label: "Concept" },
  { id: "insight", label: "Insight" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day2 = () => {
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
        rootMargin: "-20% 0px -30% 0px", // ปรับ rootMargin ให้เหมาะสม
        threshold: 0.1, // ลด threshold เพื่อให้ตรวจจับง่ายขึ้น
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
    <div className="flex flex-col gap-2">
      {headings.map((h) => (
        <button
          key={h.id}
          onClick={() => scrollToSection(h.id)}
          className={`text-sm text-right transition-all hover:underline ${
            activeId === h.id ? "font-bold text-yellow-400" : "text-gray-400"
          }`}
        >
          {h.label}
        </button>
      ))}
    </div>
  );
};

export default ScrollSpy_Ai_Day2;
