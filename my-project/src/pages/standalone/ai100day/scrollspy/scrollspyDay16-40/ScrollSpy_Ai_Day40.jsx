import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "1. บทนำ: ความท้าทายของการ Deploy โมเดล ML" },
  { id: "types", label: "2. ประเภทของการ Deploy โมเดล" },
  { id: "export", label: "3. การ Export โมเดลอย่างปลอดภัย" },
  { id: "api", label: "4. การสร้าง Model API (Serve/Wrap)" },
  { id: "infra", label: "5. การเลือก Infrastructure" },
  { id: "cicd", label: "6. CI/CD สำหรับ Machine Learning" },
  { id: "monitoring", label: "7. การตั้งระบบ Monitoring" },
  { id: "versioning", label: "8. การจัดการเวอร์ชันของโมเดล (Model Registry)" },
  { id: "best", label: "9. Best Practices" },
  { id: "case", label: "10. Case Studies" },
  { id: "insight", label: "11. Insight Box" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day40 = () => {
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

export default ScrollSpy_Ai_Day40;
