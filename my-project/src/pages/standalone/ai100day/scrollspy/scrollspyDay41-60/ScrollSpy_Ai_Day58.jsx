"use client";
import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. Generative Models คืออะไร?" },
  { id: "principles", label: "2. หลักการของ Generative Modeling" },
  { id: "gans", label: "3. GANs (Generative Adversarial Networks)" },
  { id: "vaes", label: "4. VAEs (Variational Autoencoders)" },
  { id: "gans-vs-vaes", label: "5. เปรียบเทียบ GANs vs VAEs" },
  { id: "research-examples", label: "6. ตัวอย่างงานวิจัยสำคัญ" },
  { id: "industry-use-cases", label: "7. ใช้งานจริงในอุตสาหกรรม" },
  { id: "creative-foundations", label: "8. Generative Models → Foundation of Creative AI" },
  { id: "insight-box", label: "9. Insight Box" },
  { id: "references", label: "10. Academic References" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day58 = () => {
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

export default ScrollSpy_Ai_Day58;
