import React, { useEffect, useState } from "react";

const headings = [
    { id: "intro", label: "1. บทนำ: ทำไมต้อง Pool และ Stride?" },
    { id: "max-vs-avg", label: "2. Max Pooling vs Average Pooling" },
    { id: "pooling-config", label: "3. วิธีการตั้งค่า Pooling" },
    { id: "pool-placement", label: "4. การวาง Pooling Layer อย่างมีหลักการ" },
    { id: "stride-effects", label: "5. Stride: แนวคิดและผลลัพธ์" },
    { id: "compare-stride-pool", label: "6. Comparative Analysis: Pool vs Strided Convolution" },
    { id: "adaptive-pooling", label: "7. Adaptive Pooling" },
    { id: "pool-alternatives", label: "8. ทางเลือกใหม่แทน Pooling" },
    { id: "insight", label: " Special Insight" },
    { id: "quiz", label: " MiniQuiz: Pool & Stride" },
  ];

const ScrollSpy_Ai_Day23 = () => {
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

export default ScrollSpy_Ai_Day23;
