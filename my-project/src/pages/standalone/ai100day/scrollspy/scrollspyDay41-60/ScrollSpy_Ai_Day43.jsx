import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไมต้องมี Pooling Layer?" },
  { id: "types", label: "2. ประเภทของ Pooling" },
  { id: "parameters", label: "3. Pooling Parameters" },
  { id: "vs-convolution", label: "4. Pooling vs Convolution" },
  { id: "spatial-reduction", label: "5. Spatial Reduction Strategy" },
  { id: "overlap", label: "6. Overlapping vs Non-Overlapping Pooling" },
  { id: "translation-invariance", label: "7. Effect of Pooling on Translation Invariance" },
  { id: "alternatives", label: "8. Pooling Alternatives (Modern View)" },
  { id: "visualization", label: "9. Visualization: Before vs After Pooling" },
  { id: "use-cases", label: "10. Use Cases ที่ใช้ Pooling" },
  { id: "insight-box", label: "11. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day43 = () => {
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

export default ScrollSpy_Ai_Day43;
