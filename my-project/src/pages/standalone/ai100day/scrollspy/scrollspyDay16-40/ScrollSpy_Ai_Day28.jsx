import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. Introduction: GRU คืออะไร และทำไมถึงเกิดขึ้น" },
  { id: "architecture", label: "2. GRU Architecture: ภาพรวมของโครงสร้าง" },
  { id: "equations", label: "3. GRU Equations (เข้าใจง่าย)" },
  { id: "comparison", label: "4. GRU vs LSTM: เปรียบเทียบแบบมืออาชีพ" },
  { id: "visualization", label: "5. Visualization: GRU ในมุมภาพ" },
  { id: "use-cases", label: "6. Use Cases of GRU in Real World" },
  { id: "when-to-use", label: "7. กรณีที่ควรใช้ GRU แทน LSTM" },
  { id: "code", label: "8. ตัวอย่างการเขียนโค้ด GRU ด้วย Keras" },
  { id: "limitations", label: "9. ข้อจำกัดของ GRU" },
  { id: "references", label: "10. Academic References" },
  { id: "summary", label: "11. Summary" },
  { id: "quiz", label: "MiniQuiz: GRU Review" }
];

const ScrollSpy_Ai_Day28 = () => {
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

export default ScrollSpy_Ai_Day28;
