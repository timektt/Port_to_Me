// src/pages/courses/scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day34.jsx

import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไมต้องมี Encoder–Decoder?" },
  { id: "architecture", label: "2. สถาปัตยกรรม Encoder–Decoder เบื้องต้น" },
  { id: "encoder", label: "3. Encoder ทำหน้าที่อะไร?" },
  { id: "decoder", label: "4. Decoder ทำหน้าที่อะไร?" },
  { id: "cross-attention", label: "5. Cross-Attention คืออะไร?" },
  { id: "diagram", label: "6. Visual Diagram: Encoder–Decoder (Transformer Style)" },
  { id: "rnn-vs-transformer", label: "7. ความแตกต่างระหว่าง RNN-Based vs Transformer-Based" },
  { id: "applications", label: "8. Applications จริงของ Encoder–Decoder" },
  { id: "models", label: "9. ตัวอย่างโมเดลที่ใช้ Encoder–Decoder" },
  { id: "challenges", label: "10. ความท้าทายและจุดสำคัญ" },
  { id: "research", label: "11. Research Highlights" },
  { id: "tip", label: "12. Practical Tip" },
  { id: "insight", label: "13. Insight Box" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day34 = () => {
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

export default ScrollSpy_Ai_Day34;
