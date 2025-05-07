import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. บทนำสู่เทคนิค Regularization" },
  { id: "overfitting", label: "2. ปัญหา Overfitting และแนวทางลดความซับซ้อนของโมเดล" },
  { id: "weight-regularization", label: "3. เทคนิค Weight Regularization (L1 / L2)" },
  { id: "dropout", label: "4. เทคนิค Dropout และการป้องกันการพึ่งพา Neuron" },
  { id: "data-augmentation", label: "5. การเพิ่มข้อมูลด้วย Data Augmentation" },
  { id: "batchnorm-earlystop", label: "6. Batch Normalization และ Early Stopping" },
  { id: "summary", label: "7. บทสรุปเชิงลึก (Insight Recap)" },
  { id: "references", label: "8. แหล่งอ้างอิง" },
  { id: "quiz", label: "MiniQuiz: Regularization in CNNs" },
];

const ScrollSpy_Ai_Day25 = () => {
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

export default ScrollSpy_Ai_Day25;
