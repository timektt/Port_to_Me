import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. บทนำ: ทำไมต้องทำ Normalization ใน Neural Network" },
  { id: "internal-covariate-shift", label: "2. Internal Covariate Shift: ปัญหาที่ซ่อนอยู่" },
  { id: "batch-normalization", label: "3. Batch Normalization (BN)" },
  { id: "batchnorm-practice", label: "4. BatchNorm ใน Practice" },
  { id: "batchnorm-limitations", label: "5. ปัญหาของ Batch Normalization" },
  { id: "layer-normalization", label: "6. Layer Normalization (LN)" },
  { id: "bn-vs-ln", label: "7. เปรียบเทียบ BatchNorm vs LayerNorm" },
  { id: "other-normalizations", label: "8. Other Normalization Techniques" },
  { id: "insight", label: "9. Insight เชิงลึก" },
  { id: "special-box", label: "10. Special Box: Best Practices" },
  { id: "quiz", label: "11. Mini Quiz" }
];

const ScrollSpy_Ai_Day20 = () => {
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

export default ScrollSpy_Ai_Day20;
