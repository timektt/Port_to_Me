import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. บทนำ: ทำไม Gradient Descent แบบธรรมดาไม่พอ?" },
  { id: "basic-gradient-descent", label: "2. Review สั้น: Gradient Descent พื้นฐาน" },
  { id: "sgd", label: "3. Stochastic Gradient Descent (SGD)" },
  { id: "mini-batch", label: "4. Mini-Batch Gradient Descent" },
  { id: "momentum", label: "5. Momentum" },
  { id: "nesterov", label: "6. Nesterov Accelerated Gradient (NAG)" },
  { id: "adaptive-methods", label: "7. Adaptive Methods (AdaGrad, RMSProp, Adam)" },
  { id: "optimizer-comparison", label: "8. ตารางเปรียบเทียบ Optimizer ต่าง ๆ" },
  { id: "insight", label: "9. Insight เชิงลึก" },
  { id: "special-box", label: "10. Special Box: Best Practices" },
  { id: "quiz", label: "11. Mini Quiz" }
];

const ScrollSpy_Ai_Day19 = () => {
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

export default ScrollSpy_Ai_Day19;
