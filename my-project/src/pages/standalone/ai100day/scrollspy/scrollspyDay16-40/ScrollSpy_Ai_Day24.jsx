import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: CNN กับระบบประมวลผลภาพ" },
  { id: "feature-extraction", label: "2. การสกัดคุณลักษณะและลำดับชั้น" },
  { id: "classification", label: "3. การจำแนกประเภทด้วย CNN" },
  { id: "applications", label: "4. การประยุกต์ใช้ในโลกความจริง" },
  { id: "case-studies", label: "5. กรณีศึกษาและความท้าทาย" },
  { id: "research", label: "6. งานวิจัยเด่นด้าน CNN" },
  { id: "insight-recap", label: "7. บทสรุปเชิงลึก (Insight Recap)" },
  { id: "quiz", label: "MiniQuiz: CNN for Vision" },
];

const ScrollSpy_Ai_Day24 = () => {
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

export default ScrollSpy_Ai_Day24;
