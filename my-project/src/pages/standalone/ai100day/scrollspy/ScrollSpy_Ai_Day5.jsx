import React, { useEffect, useState } from "react";

const headings = [
  { id: "what-is-linear-transform", label: "Linear Transformation คืออะไร?" },
  { id: "matrix-properties", label: "ข้อจำกัดของ Linear Transformation" },
  { id: "layer-wise-transform", label: "Matrix ในหลายเลเยอร์" },
  { id: "compare-raw-feature", label: "เปรียบเทียบข้อมูลดิบ vs. Feature" },
  { id: "feature-extraction", label: "ดึง Feature จากภาพ/เสียง/ข้อความ" },
  { id: "vector-change", label: "เวกเตอร์เปลี่ยนอย่างไร?" },
  { id: "insight", label: "Insight จาก Matrix" },
  { id: "code-example", label: "ตัวอย่างโค้ด NumPy & PyTorch" },
  { id: "visual-summary", label: "ภาพรวมก่อน-หลังการแปลง" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day5 = () => {
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visibleSections = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

        if (visibleSections.length > 0) {
          setActiveId(visibleSections[0].target.id);
        }
      },
      {
        rootMargin: "-10% 0px -20% 0px",
        threshold: 0.1,
      }
    );

    const elements = headings.map((h) => document.getElementById(h.id)).filter(Boolean);
    elements.forEach((el) => observer.observe(el));

    return () => {
      elements.forEach((el) => observer.unobserve(el));
    };
  }, []);

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <nav className="flex flex-col gap-2 text-sm max-w-[220px]">
      {headings.map((h) => (
        <button
          key={h.id}
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

export default ScrollSpy_Ai_Day5;
