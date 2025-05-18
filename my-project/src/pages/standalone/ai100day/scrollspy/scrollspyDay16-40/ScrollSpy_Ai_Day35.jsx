// src/pages/courses/scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day35.jsx

import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Transfer Learning ถึงเปลี่ยนเกม?" },
  { id: "definition", label: "2. Transfer Learning คืออะไร?" },
  { id: "pretrain-vs-finetune", label: "3. Pretraining vs Finetuning" },
  { id: "transfer-strategies", label: "4. กลยุทธ์การ Transfer" },
  { id: "types", label: "5. ประเภทของ Transfer Learning" },
  { id: "pretraining-research", label: "6. ตัวอย่างงานวิจัย Pretraining สำคัญ" },
  { id: "pretraining-cv", label: "7. Pretraining ในคอมพิวเตอร์วิทัศน์ (CV)" },
  { id: "pretraining-nlp", label: "8. Pretraining ใน NLP และ Multimodal" },
  { id: "scaling-laws", label: "9. Scaling Laws และผลของ Pretraining" },
  { id: "pitfalls", label: "10. ปัญหาและข้อควรระวังของ Transfer Learning" },
  { id: "transfer-techniques", label: "11. เทคนิคช่วยให้ Transfer ดีขึ้น" },
  { id: "realworld-examples", label: "12. ตัวอย่างงานจริงที่ใช้ Transfer Learning" },
  { id: "decision-guide", label: "13. การเลือกว่าจะ Transfer แบบไหน" },
  { id: "research-spotlight", label: "14. Research Spotlight" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day35 = () => {
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

export default ScrollSpy_Ai_Day35;
