import React, { useEffect, useState } from "react";

const headings = [
  { id: "why-rnn", label: "1. RNN กับลำดับข้อมูล" },
  { id: "nlp-use-cases", label: "2. NLP Use Cases" },
  { id: "time-series", label: "3. Time Series Use Cases" },
  { id: "choose-rnn", label: "4. เปรียบเทียบ RNN / LSTM / GRU" },
  { id: "benchmarks", label: "5. Benchmarks" },
  { id: "limitations", label: "6. ข้อจำกัด RNN" },
  { id: "future-trends", label: "7. แนวโน้มในอนาคต" },
  { id: "insight-box", label: "8. Insight" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day48 = () => {
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

export default ScrollSpy_Ai_Day48;
