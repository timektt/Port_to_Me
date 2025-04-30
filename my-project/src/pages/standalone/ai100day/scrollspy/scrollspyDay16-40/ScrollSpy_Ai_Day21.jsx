import React, { useEffect, useState } from "react";

const headings = [
  { id: "introduction", label: "1. บทนำ: ทำไมต้อง Convolutional Neural Networks" },
  { id: "cnn-structure", label: "2. โครงสร้างพื้นฐานของ CNN" },
  { id: "convolution-layer", label: "3. หลักการทำงานของ Convolution Layer" },
  { id: "activation-functions", label: "4. Activation Functions" },
  { id: "pooling-layers", label: "5. Pooling Layers" },
  { id: "fully-connected", label: "6. การเชื่อมต่อ Fully Connected" },
  { id: "cnn-learns", label: "7. ความเข้าใจเชิงลึก: CNN เรียนรู้อะไร" },
  { id: "cnn-strengths", label: "8. จุดแข็งของ CNN" },
  { id: "cnn-limitations", label: "9. ข้อจำกัดของ CNN" },
  { id: "research-insight", label: "10. Insight จากงานวิจัย" },
  { id: "cnn-vs-transformer", label: "11. เปรียบเทียบ CNN vs Transformer" },
  { id: "special-box", label: "12. Special Box: ข้อควรระวัง" },
  { id: "quiz", label: "13. Mini Quiz" }
];

const ScrollSpy_Ai_Day21 = () => {
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

export default ScrollSpy_Ai_Day21;
