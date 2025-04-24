import React, { useEffect, useState } from "react";

const headings = [
  { id: "overview", label: "1. Overview: ทำไมต้อง Neural Networks" },
  { id: "neuron", label: "2. ส่วนประกอบหลักของ Neural Network" },
  { id: "architecture", label: "3. Architectural Foundation" },
  { id: "feedforward", label: "4. From Input → Output: Feedforward Process" },
  { id: "learning", label: "5. การเรียนรู้: จาก Error สู่การอัปเดต" },
  { id: "depth-width", label: "6. Network Depth vs Width" },
  { id: "activation-choice", label: "7. การเลือก Activation Function" },
  { id: "draw", label: "8. ลองวาด Neural Network ด้วยมือ" },
  { id: "library", label: "9. Neural Network ในไลบรารียอดนิยม" },
  { id: "insight", label: "10. Insight: ปฏิวัติวงการ AI" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day16 = () => {
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
      {headings.map((h) => (
        <button
          key={h.id}
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

export default ScrollSpy_Ai_Day16;
