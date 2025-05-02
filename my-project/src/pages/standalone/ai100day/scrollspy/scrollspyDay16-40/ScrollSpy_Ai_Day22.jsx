import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: โครงสร้าง CNN ไม่ได้มีแค่เลเยอร์ซ้อนกัน" },
  { id: "building-blocks", label: "2. Basic Building Blocks ของ CNN Architecture" },
  { id: "filters", label: "3. การออกแบบ Filter (Kernel)" },
  { id: "hyperparameters", label: "4. Layer Hyperparameters" },
  { id: "layer-arrangement", label: "5. การจัดลำดับ Layers: Depth & Width" },
  { id: "channels", label: "6. การจัด Group & Channel" },
  { id: "transfer-learning", label: "7. Pretrained Filters & Transfer Learning" },
  { id: "real-architectures", label: "8. Insight จาก Architecture จริง" },
  { id: "visualization", label: "9. Visualization: ดูว่า Filter เรียนรู้อะไร" },
  { id: "problems", label: "10. ปัญหาและการแก้ไขในการออกแบบ CNN" },
  { id: "quiz", label: "11. Mini Quiz" }
];

const ScrollSpy_Ai_Day22 = () => {
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

export default ScrollSpy_Ai_Day22;
