import React, { useEffect, useState } from "react";

const headings = [
  { id: "motivation", label: "1. Motivation: ทำไม Transformer จึงกลายเป็น Game Changer" },
  { id: "architecture", label: "2. Overall Architecture: Encoder-Decoder แบบ Full Attention" },
  { id: "positional-encoding", label: "3. Positional Encoding: เข้าใจตำแหน่ง โดยไม่ใช้ลำดับ" },
  { id: "self-attention", label: "4. Self-Attention Mechanism" },
  { id: "multi-head", label: "5. Multi-Head Attention" },
  { id: "ffn", label: "6. Feed Forward Network (FFN)" },
  { id: "residual-norm", label: "7. Residual & Layer Normalization" },
  { id: "masked-attention", label: "8. Masked Self-Attention (ใน Decoder)" },
  { id: "visualization", label: "9. Visualization" },
  { id: "references", label: "10. Academic Reference" },
  { id: "practical-tips", label: "11. Practical Tips" },
  { id: "limitations", label: "12. Limitations" },
  { id: "insight", label: "13. Insight Box" },
  { id: "quiz", label: "MiniQuiz: Transformer Architecture" },
];

const ScrollSpy_Ai_Day31 = () => {
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

export default ScrollSpy_Ai_Day31;
