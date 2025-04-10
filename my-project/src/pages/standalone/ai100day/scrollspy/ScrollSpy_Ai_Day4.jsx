import React, { useEffect, useState } from "react";

const headings = [
  { id: "what-is-matrix", label: "Matrix Multiplication คืออะไร?" },
  { id: "matrix-dimension", label: "เข้าใจขนาดเมทริกซ์" },
  { id: "matrix-order", label: "A @ B ≠ B @ A" },
  { id: "matrix-transpose", label: "Matrix Transpose & Attention" },
  { id: "softmax-scaling", label: "Softmax & Scaling" },
  { id: "multi-head-attention", label: "Multi-Head Attention" },
  { id: "linear-transformation-visual", label: "ภาพการ Stretch ด้วยเมทริกซ์" },
  { id: "linear-transform", label: "Linear Transformation คืออะไร?" },
  { id: "broadcasting", label: "Broadcasting คืออะไร?" },
  { id: "batch-matrix", label: "Matrix Multiplication แบบ Batch" },
  { id: "try-interactive", label: "ทดลองคูณเมทริกซ์ " },
  { id: "real-world-examples", label: "ตัวอย่างในโลกจริง" },
  { id: "read-more", label: "แหล่งเรียนรู้ต่อยอด" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day4 = () => {
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
        rootMargin: "-20% 0px -30% 0px",
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

export default ScrollSpy_Ai_Day4;
