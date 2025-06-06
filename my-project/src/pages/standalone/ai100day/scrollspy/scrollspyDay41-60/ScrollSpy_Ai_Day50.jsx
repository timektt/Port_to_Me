"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: แนวคิด Attention คืออะไร?" },
  { id: "rnn-problems", label: "2. ปัญหาของ RNN แบบดั้งเดิม" },
  { id: "attention-design", label: "3. การออกแบบ Attention Mechanism" },
  { id: "attention-types", label: "4. ประเภทของ Attention Mechanisms" },
  { id: "transformer-revolution", label: "5. Transformer Architecture → Revolution ด้วย Attention" },
  { id: "attention-use-cases", label: "6. Use Cases ของ Attention ใน Deep Learning" },
  { id: "attention-advantages", label: "7. ข้อดีของ Attention Mechanism" },
  { id: "attention-limitations", label: "8. Limitations และ Challenge" },
  { id: "attention-research", label: "9. Research Benchmarks & State-of-the-art Papers" },
  { id: "insight-box", label: "10. Insight Box" },
  { id: "quiz", label: "Mini Quiz" }, // ถ้ามี MiniQuiz_Day50
];

const ScrollSpy_Ai_Day50 = () => {
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

export default ScrollSpy_Ai_Day50;
