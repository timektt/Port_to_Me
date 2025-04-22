import React, { useEffect, useState } from "react";

const headings = [
  { id: "intro", label: "บทนำ: Fairness & Ethics" },
  { id: "definition", label: "ความหมายของ Fairness และ Ethical AI" },
  { id: "real-case", label: "ตัวอย่างกรณีศึกษาในชีวิตจริง" },
  { id: "impact", label: "ผลกระทบจาก AI ที่มี Bias" },
  { id: "bias-types", label: "1. ประเภทของ Bias ที่พบบ่อย" },
  { id: "bias-analysis", label: "2. การวิเคราะห์ Bias ในโมเดล" },
  { id: "bias-solutions", label: "3. วิธีลด Bias อย่างเป็นระบบ" },
  { id: "fairness-testing", label: "4. การทดลองและวัด Fairness" },
  { id: "ethics", label: "5. จริยธรรมของ AI" },
  { id: "human-values", label: "6. ออกแบบ AI ที่เคารพความเป็นมนุษย์" },
  { id: "workflow", label: "7. ฝัง Fairness ใน Workflow" },
  { id: "case-study", label: "8. Case Study: คัดกรองผู้สมัครงาน" },
  { id: "insight", label: "9. Insight: Accuracy vs Fairness" },
  { id: "summary", label: "10. สรุปท้ายบท" },
  { id: "quiz", label: "Mini Quiz" },
];

const ScrollSpy_Ai_Day14 = () => {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
  const NAVBAR_OFFSET = 160; // หรือ 72 ก็ลองได้ตาม header จริง

  const handleScroll = () => {
    let closestId = "";
    let minOffset = Infinity;

    headings.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) {
        const offset = Math.abs(el.getBoundingClientRect().top - NAVBAR_OFFSET);
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

export default ScrollSpy_Ai_Day14;