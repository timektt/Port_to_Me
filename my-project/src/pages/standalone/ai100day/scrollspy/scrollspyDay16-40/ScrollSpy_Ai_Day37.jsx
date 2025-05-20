import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Data Augmentation จึงสำคัญ" },
  { id: "principles", label: "2. หลักการพื้นฐานของการ Augmentation" },
  { id: "cv", label: "3. Augmentation สำหรับ Computer Vision" },
  { id: "nlp", label: "4. Augmentation สำหรับ NLP (Text)" },
  { id: "time-series", label: "5. Augmentation สำหรับ Time Series / Audio" },
  { id: "advanced", label: "6. Advanced Augmentation: GAN, SimCLR, Self-Supervised" },
  { id: "smart-strategy", label: "7. การใช้ Augmentation อย่างชาญฉลาด (Smart Strategies)" },
  { id: "evaluation", label: "8. การวัดผลของ Augmentation" },
  { id: "case-study", label: "9. กรณีศึกษาระดับโลก" },
  { id: "code", label: "10. Code Examples (Vision & Text)" },
  { id: "insight", label: "11. Insight Box" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day37 = () => {
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

export default ScrollSpy_Ai_Day37;
