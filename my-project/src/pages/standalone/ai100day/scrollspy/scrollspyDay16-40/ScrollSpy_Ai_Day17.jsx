import React, { useEffect, useState } from "react";

const headings = [
  { id: "overview", label: "1. บทนำ: จากสมองสู่โครงข่ายประสาทเทียม" },
  { id: "perceptron", label: "2. Perceptron คืออะไร?" },
  { id: "mlp", label: "3. Multi-Layer Perceptron (MLP)" },
  { id: "mlplearn", label: "4. การเรียนรู้ใน MLP" },
  { id: "real-world", label: "5. การฝึก MLP บนงานจริง" },
  { id: "advantages", label: "6. ข้อดีและข้อจำกัดของ MLP" },
  { id: "insight", label: "7. Insight Box" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day17 = () => {
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

export default ScrollSpy_Ai_Day17;
