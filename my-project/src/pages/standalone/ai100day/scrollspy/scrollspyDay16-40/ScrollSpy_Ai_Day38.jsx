import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ทำไม Hyperparameter จึงสำคัญ?" },
  { id: "examples", label: "2. ตัวอย่าง Hyperparameters ที่มักต้องจูน" },
  { id: "manual", label: "3. Manual Tuning (ยุคแรก)" },
  { id: "grid-search", label: "4. Grid Search" },
  { id: "random-search", label: "5. Random Search" },
  { id: "bayesian", label: "6. Bayesian Optimization" },
  { id: "halving", label: "7. Successive Halving & Hyperband" },
  { id: "pbt", label: "8. Population-Based Training (PBT)" },
  { id: "nas", label: "9. Neural Architecture Search (NAS)" },
  { id: "scheduling", label: "10. Smart Scheduling & Early Stopping" },
  { id: "metrics", label: "11. การใช้ Metric ที่เหมาะสม" },
  { id: "case-studies", label: "12. แนวคิดจากโลกจริง (Case Studies)" },
  { id: "quiz", label: "Mini Quiz" }
];

const ScrollSpy_Ai_Day38 = () => {
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

export default ScrollSpy_Ai_Day38;