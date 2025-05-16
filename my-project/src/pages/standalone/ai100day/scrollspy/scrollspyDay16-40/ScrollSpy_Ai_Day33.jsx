import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไมต้อง Attention?" },
  { id: "scaled-dot-product", label: "2. Scaled Dot-Product Attention" },
  { id: "self-attention", label: "3. Self-Attention คืออะไร?" },
  { id: "heatmap", label: "4. Visualization: Self-Attention Heatmap" },
  { id: "multihead", label: "5. Multi-Head Attention คืออะไร?" },
  { id: "why-multihead", label: "6. ทำไมต้องมีหลายหัว?" },
  { id: "complexity", label: "7. ความซับซ้อนด้านคำนวณ (Complexity)" },
  { id: "position", label: "8. ตำแหน่งใน Transformer" },
  { id: "research", label: "9. งานวิจัยที่เกี่ยวข้อง" },
  { id: "multi-query", label: "10. Multi-Query Attention" },
  { id: "tip", label: "11. Practical Tip" },
  { id: "insight1", label: "12. Insight Box (MQA)" },
  { id: "insight2", label: "13. Insight Box (อนาคต)" },
  { id: "quiz", label: " Mini Quiz" }
];

const ScrollSpy_Ai_Day33 = () => {
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

export default ScrollSpy_Ai_Day33;
