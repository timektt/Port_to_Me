import React, { useEffect, useState } from "react";

const headings = [
  { id: "filters-overview", label: "1. บทนำ: ภาพรวมของ Filter ใน CNN" },
  { id: "layerwise-filters", label: "2. ฟิลเตอร์ในเลเยอร์ต่าง ๆ" },
  { id: "feature-map", label: "3. Feature Map คืออะไร?" },
  { id: "filter-training", label: "4. การเรียนรู้ Filters ระหว่าง Training" },
  { id: "visualization", label: "5. Visualization ของ Filters และ Feature Maps" },
  { id: "channel-wise", label: "6. การทำงานแบบ Channel-wise" },
  { id: "filter-depth", label: "7. Depth of Filters กับพลังของ CNN" },
  { id: "hierarchy", label: "8. การวิเคราะห์ Feature Hierarchy" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day42 = () => {
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

export default ScrollSpy_Ai_Day42;
