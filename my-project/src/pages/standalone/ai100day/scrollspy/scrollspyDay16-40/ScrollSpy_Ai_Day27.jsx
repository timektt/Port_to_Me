import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. Introduction: ทำไมต้องมี LSTM?" },
  { id: "architecture", label: "2. Architecture Overview: กลไกภายใน LSTM" },
  { id: "flow", label: "3. Flow of Data Inside LSTM" },
  { id: "lstm-equations", label: "4. LSTM Equations (แบบเข้าใจง่าย)" },
  { id: "comparison", label: "5. Differences: LSTM vs Vanilla RNN" },
  { id: "variants", label: "6. Variants of LSTM" },
  { id: "training", label: "7. Training LSTM: ข้อควรระวัง" },
  { id: "use-cases", label: "8. กรณีใช้งานจริงของ LSTM" },
  { id: "visualization", label: "9. Visualization" },
  { id: "code", label: "10. Coding Walkthrough (สรุปโค้ด)" },
  { id: "summary", label: "11. สรุป (Summary)" },
  { id: "quiz", label: "MiniQuiz: LSTM" },
];

const ScrollSpy_Ai_Day27 = () => {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
    const handleScroll = () => {
      let closestId = "";
      let minOffset = Infinity;

      headings.forEach(({ id }) => {
        const el = document.getElementById(id);
        if (el) {
          const offset = Math.abs(el.getBoundingClientRect().top - 160); // ปรับความแม่นยำ
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
      const yOffset = -120; // ระยะเผื่อ scroll-mt
      const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: "smooth" });
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

export default ScrollSpy_Ai_Day27;
