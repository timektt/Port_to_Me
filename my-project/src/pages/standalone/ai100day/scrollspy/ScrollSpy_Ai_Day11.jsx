import React, { useEffect, useState } from "react";

const headings = [
  { id: "why-cross-validation", label: "ทำไมต้อง Cross Validation?" },
  { id: "types-of-cross-validation", label: "Types of Cross Validation" },
  { id: "kfold-workflow", label: "K-Fold Workflow" },
  { id: "cv-model-selection", label: "Cross Validation กับการเลือกโมเดล" },
  { id: "metrics", label: "Metric ที่ใช้ประเมินโมเดล" },
  { id: "metrics-insight", label: "Insight: Metric เปรียบเทียบ" },
  { id: "best-practice", label: "Best Practice การประเมิน" },
  { id: "insight", label: "Insight ตอนท้าย" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day11 = () => {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
    const handleScroll = () => {
      let closestId = "";
      let minOffset = Infinity;

      headings.forEach(({ id }) => {
        const el = document.getElementById(id);
        if (el) {
          const offset = Math.abs(el.getBoundingClientRect().top - 120); // ปรับ offset ให้ตรงกับ scroll-mt
          if (offset < minOffset) {
            minOffset = offset;
            closestId = id;
          }
        }
      });

      setActiveId(closestId);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // run on mount

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

export default ScrollSpy_Ai_Day11;
