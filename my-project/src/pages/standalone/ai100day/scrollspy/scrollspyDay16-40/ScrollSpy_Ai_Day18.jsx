import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. บทนำ: ทำไมการ Initial Weights ถึงสำคัญ?" },
  { id: "random-initialization", label: "2. ปัญหาที่เกิดจากการ Initial Weights แบบสุ่มธรรมดา" },
  { id: "classic-techniques", label: "3. เทคนิค Weight Initialization แบบคลาสสิก" },
  { id: "xavier-initialization", label: "4. Xavier (Glorot) Initialization" },
  { id: "he-initialization", label: "5. He Initialization" },
  { id: "advanced-techniques", label: "6. Advanced Initialization Techniques" },
  { id: "bias-initialization", label: "7. การ Initial Bias ใน Neural Networks" },
  { id: "choosing-strategy", label: "8. การเลือก Strategy ให้เหมาะกับ Model" },
  { id: "insight", label: "9. Insight Box" },
  { id: "quiz", label: "10. Mini Quiz" }
];

const ScrollSpy_Ai_Day18 = () => {
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

export default ScrollSpy_Ai_Day18;
