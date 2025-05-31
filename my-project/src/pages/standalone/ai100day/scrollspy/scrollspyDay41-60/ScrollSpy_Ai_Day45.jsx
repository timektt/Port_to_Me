import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไมต้องใช้ Transfer Learning?" },
  { id: "concept", label: "2. แนวคิดหลักของ Transfer Learning" },
  { id: "pipeline", label: "3. โครงสร้างพื้นฐานของ Transfer Learning Pipeline" },
  { id: "model-selection", label: "4. การเลือก Pretrained Model" },
  { id: "resnet-example", label: "5. ตัวอย่าง: ใช้ ResNet50 กับชุดข้อมูลใหม่" },
  { id: "fine-tune", label: "6. เทคนิคการ Fine-Tune อย่างมีประสิทธิภาพ" },
  { id: "use-cases", label: "7. Use Cases ที่ได้ผลลัพธ์สูงจาก Transfer Learning" },
  { id: "caution", label: "8. ข้อควรระวังในการใช้ Transfer Learning" },
  { id: "research", label: "9. Research & References" },
  { id: "insight-box", label: "10. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day45 = () => {
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

export default ScrollSpy_Ai_Day45;

