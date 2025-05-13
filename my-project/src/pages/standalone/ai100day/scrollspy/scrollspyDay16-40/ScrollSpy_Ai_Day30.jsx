
import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. Introduction: ปัญหาของ RNN/Seq2Seq ที่ Attention มาช่วยแก้" },
  { id: "concept", label: "2. แนวคิดของ Attention: โฟกัสแบบเลือกตำแหน่ง" },
  { id: "seq2seq", label: "3. Attention in Sequence-to-Sequence Models" },
  { id: "types", label: "4. ประเภทของ Attention แบบ Classic" },
  { id: "math", label: "5. คณิตศาสตร์ของ Attention Step-by-Step" },
  { id: "visualization", label: "6. Visualization & Diagram" },
  { id: "comparison", label: "7. เปรียบเทียบ Encoder-Decoder แบบเดิม vs Attention" },
  { id: "applications", label: "8. Real-World Applications" },
  { id: "limitations", label: "9. ข้อจำกัดของ Attention แบบ Classic" },
  { id: "code-example", label: "10. Code Example (PyTorch Pseudo)" },
  { id: "references", label: "11. Academic References" },
  { id: "summary", label: "12. Summary" },
  { id: "quiz", label: "MiniQuiz: Attention Mechanisms" },
];

const ScrollSpy_Ai_Day30 = () => {
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

export default ScrollSpy_Ai_Day30;
