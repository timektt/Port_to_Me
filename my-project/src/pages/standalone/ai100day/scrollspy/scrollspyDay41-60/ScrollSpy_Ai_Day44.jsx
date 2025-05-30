import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ภาพหนึ่งภาพ เล่าข้อมูลพันคำได้อย่างไร?" },
  { id: "cnn-structure", label: "2. โครงสร้างพื้นฐานของ CNN สำหรับ Image Classification" },
  { id: "cnn-architectures", label: "3. ตัวอย่าง CNN Architectures ที่ใช้จริง" },
  { id: "loss-metrics", label: "4. Loss Function และ Evaluation Metrics" },
  { id: "training-pipeline", label: "5. Training Pipeline สำหรับ Classification" },
  { id: "accuracy-techniques", label: "6. เทคนิคที่ช่วยเพิ่มความแม่นยำ" },
  { id: "visualization", label: "7. Visualization การทำงานของ CNN" },
  { id: "use-cases", label: "8. Real-World Use Cases" },
  { id: "limitations", label: "9. ข้อจำกัดที่ต้องพิจารณา" },
  { id: "insight-box", label: "10. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day44 = () => {
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

export default ScrollSpy_Ai_Day44;
