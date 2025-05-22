import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Best Practices จึงสำคัญใน Supervised Learning?" },
  { id: "problem", label: "2. วางโจทย์ให้ถูกตั้งแต่ต้น" },
  { id: "data-strategy", label: "3. Data Collection & Labeling Strategy" },
  { id: "split", label: "4. Data Splitting ที่ถูกต้อง" },
  { id: "preprocessing", label: "5. Preprocessing & Feature Engineering" },
  { id: "dev-debug", label: "6. Model Development & Debugging" },
  { id: "eval", label: "7. Cross-validation & Evaluation" },
  { id: "ensembling", label: "8. Model Selection & Ensembling" },
  { id: "regularization", label: "9. Regularization & Avoiding Overfitting" },
  { id: "deployment", label: "10. Monitoring & Deployment Readiness" },
  { id: "feedback", label: "11. Feedback Loop & Continuous Learning" },
  { id: "checklist", label: "12. Best Practices Checklist" },
  { id: "case", label: "13. Case Studies" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day39 = () => {
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

export default ScrollSpy_Ai_Day39;
