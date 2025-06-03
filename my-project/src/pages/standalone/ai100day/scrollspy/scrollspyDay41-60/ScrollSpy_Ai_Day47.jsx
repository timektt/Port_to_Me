import React, { useEffect, useState } from "react";

const headings = [
  { id: "vanishing-gradient", label: "1. จุดอ่อนของ RNN ดั้งเดิม: Vanishing Gradient" },
  { id: "lstm", label: "2. LSTM (Long Short-Term Memory) คืออะไร?" },
  { id: "lstm-architecture", label: "3. Visual Flow: LSTM Architecture" },
  { id: "gru", label: "4. GRU (Gated Recurrent Unit)" },
  { id: "lstm-vs-gru", label: "5. เปรียบเทียบ LSTM vs GRU" },
  { id: "use-cases", label: "6. Use Cases จากงานวิจัยจริง" },
  { id: "visualization", label: "7. Visualization: GRU vs LSTM" },
  { id: "references", label: "8. Research & Global References" },
  { id: "insight-box", label: "9. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day47 = () => {
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

export default ScrollSpy_Ai_Day47;
