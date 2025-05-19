// src/pages/courses/scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day36.jsx

import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: Fine-tuning คือหัวใจของ Transfer Learning" },
  { id: "why", label: "2. ทำไมต้อง Fine-tune?" },
  { id: "types", label: "3. ประเภทของการ Fine-tune" },
  { id: "best-practices", label: "4. Best Practices ก่อน Fine-tune" },
  { id: "architectures", label: "5. สถาปัตยกรรมที่นิยม Fine-tune" },
  { id: "techniques", label: "6. เทคนิค Fine-tune ให้ได้ผลดี" },
  { id: "peft", label: "7. การใช้ Parameter-efficient Fine-tuning (PEFT)" },
  { id: "evaluation", label: "8. การประเมินผลหลัง Fine-tuning" },
  { id: "challenges", label: "9. ความท้าทายและข้อผิดพลาดที่พบบ่อย" },
  { id: "multilingual", label: "10. การ Fine-tune สำหรับ Multilingual หรือ Multimodal" },
  { id: "case-study", label: "11. กรณีศึกษาจริงจากวงการ" },
  { id: "code", label: "12. Code Example (HuggingFace Transformers)" },
  { id: "tools", label: "13. งานวิจัยและเครื่องมือที่แนะนำ" },
  { id: "insight", label: "14. Insight Box" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day36 = () => {
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

export default ScrollSpy_Ai_Day36;
