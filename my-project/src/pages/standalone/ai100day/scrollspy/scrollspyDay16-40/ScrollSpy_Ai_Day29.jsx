import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. Introduction: ทำไมต้องพัฒนา RNN ให้ลึกและสองทาง" },
  { id: "birnn", label: "2. Bidirectional RNN (BiRNN): แนวคิดและโครงสร้าง" },
  { id: "deep-rnn", label: "3. Deep RNNs: แนวตั้งของ RNN หลายชั้น" },
  { id: "architecture-comparison", label: "4. Architecture Comparison" },
  { id: "equations", label: "5. Equations & Diagram" },
  { id: "use-cases", label: "6. Real-World Use Cases" },
  { id: "training-tips", label: "7. Tips for Training" },
  { id: "code-example", label: "8. Code Example (Keras)" },
  { id: "references", label: "9. Academic References" },
  { id: "summary", label: "10. Summary" },
  { id: "quiz", label: "MiniQuiz: Bidirectional & Deep RNNs" },
];

const ScrollSpy_Ai_Day29 = () => {
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

export default ScrollSpy_Ai_Day29;
