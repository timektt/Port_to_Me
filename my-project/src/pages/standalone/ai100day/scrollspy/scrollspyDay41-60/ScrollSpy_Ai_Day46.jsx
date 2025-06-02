import React, { useEffect, useState } from "react";

const headings = [
  { id: "why-sequence", label: "1. ทำไมต้องใช้ Sequence Models?" },
  { id: "rnn-basics", label: "2. แนวคิดพื้นฐานของ RNN (Recurrent Neural Network)" },
  { id: "seq2seq-types", label: "3. ประเภทของ Sequence-to-Sequence Tasks" },
  { id: "training-challenges", label: "4. การฝึก RNN และปัญหาที่พบ" },
  { id: "visual-flow", label: "5. Visual Insight: การไหลของข้อมูลใน RNN" },
  { id: "rnn-nlp", label: "6. RNN กับ NLP: การประมวลผลประโยค" },
  { id: "rnn-vs-cnn", label: "7. การเปรียบเทียบ RNN vs CNN" },
  { id: "rnn-limitations", label: "8. ข้อจำกัดของ RNN แบบดั้งเดิม" },
  { id: "research", label: "9. Research & References" },
  { id: "insight-box", label: "10. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day46 = () => {
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

export default ScrollSpy_Ai_Day46;
