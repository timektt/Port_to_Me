import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. Introduction: ทำไม Sequence Modeling สำคัญมาก?" },
  { id: "anatomy", label: "2. Anatomy of Time Series Data" },
  { id: "task-types", label: "3. Sequence Modeling Task Types" },
  { id: "models", label: "4. โมเดลสำหรับ Time Series" },
  { id: "sliding-vs-seq2seq", label: "5. Sliding Window vs Sequence-to-Sequence" },
  { id: "metrics", label: "6. Evaluation Metrics for Sequence Forecasting" },
  { id: "feature-engineering", label: "7. Feature Engineering for Time Series" },
  { id: "challenges", label: "8. Handling Time Series Challenges" },
  { id: "case-study", label: "9. Case Study: Stock Price Forecasting with LSTM" },
  { id: "visualization", label: "10. Visualization Example" },
  { id: "tools-libraries", label: "11. Tools & Libraries" },
  { id: "summary", label: "12. Summary & Real-World Use Cases" },
  { id: "references", label: "References" },
  { id: "quiz", label: "MiniQuiz: Sequence Forecasting" },
  
];

const ScrollSpy_Ai_Day26 = () => {
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

export default ScrollSpy_Ai_Day26;
