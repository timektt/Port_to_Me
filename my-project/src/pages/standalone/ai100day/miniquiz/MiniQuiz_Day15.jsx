import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day15 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "AI Governance คืออะไร?",
      options: [
        "การพัฒนาโมเดลด้วยความเร็วสูง",
        "การวางกรอบควบคุมและกำกับดูแลระบบ AI อย่างเป็นระบบ",
        "การฝึกโมเดลด้วยข้อมูลมากที่สุดเท่าที่ทำได้"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "กรอบการประเมินความเสี่ยงใดที่นิยมใช้ใน AI Governance?",
      options: [
        "YOLO Framework",
        "NIST AI RMF",
        "Scikit Risk Grid"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "ประเภทของความเสี่ยงใน AI ที่พบบ่อยคือ?",
      options: [
        "Risk of High Accuracy",
        "Ethical, Legal และ Operational Risks",
        "Gradient Explosion"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "การใช้ Impact-Likelihood Matrix มีไว้เพื่ออะไร?",
      options: [
        "จัดลำดับความซับซ้อนของโค้ด",
        "จัดลำดับความเสี่ยงตามความรุนแรงและโอกาสเกิด",
        "วัดค่า performance"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "ใครควรเป็นผู้รับผิดชอบการตัดสินใจ Deploy โมเดล AI?",
      options: [
        "เฉพาะฝ่าย Data Science",
        "เฉพาะ CTO เท่านั้น",
        "ทีม cross-functional ที่ประกอบด้วย Data, Legal, Product"
      ],
      correct: 2
    },
    {
      id: "q6",
      question: "AI Audit ที่ดีควรมีอะไร?",
      options: [
        "Logging, Monitoring, และ Audit Trails ที่ตรวจสอบย้อนหลังได้",
        "แค่ค่า Accuracy สูง",
        "ข้อมูลขนาดเล็กและไม่หลากหลาย"
      ],
      correct: 0
    },
    {
      id: "q7",
      question: "Model Cards ใช้เพื่ออะไร?",
      options: [
        "ฝึกโมเดล",
        "บันทึก metadata และพฤติกรรมของโมเดลเพื่อความโปร่งใส",
        "ลด latency ใน deployment"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "เครื่องมือใดที่ใช้ตรวจจับ drift หลังจากโมเดลถูก deploy?",
      options: [
        "DriftNet",
        "Post-deployment Drift Detection",
        "Bias Radar"
      ],
      correct: 1
    },
    {
      id: "q9",
      question: "AI Governance Officer มีบทบาทอย่างไร?",
      options: [
        "ดูแลด้านฮาร์ดแวร์ AI",
        "กำกับนโยบาย ตรวจสอบความเสี่ยง และประสานงานทีม",
        "ฝึกโมเดลอย่างเดียว"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "AI Governance ที่ยั่งยืนควรเน้นอะไร?",
      options: [
        "ประสิทธิภาพทางเทคนิคเท่านั้น",
        "ความเร็วในการ deploy",
        "Fairness, Accountability และ Oversight ที่ต่อเนื่อง"
      ],
      correct: 2
    }
  ];

  const handleChange = (qid, index) => {
    setAnswers({ ...answers, [qid]: index });
  };

  const handleSubmit = () => {
    let newScore = 0;
    let wrong = [];

    questions.forEach((q) => {
      const userAnswer = answers[q.id];
      if (userAnswer === q.correct) {
        newScore++;
      } else {
        wrong.push({
          id: q.id,
          question: q.question,
          yourAnswer: q.options[userAnswer],
          correctAnswer: q.options[q.correct]
        });
      }
    });

    setScore(newScore);
    setIncorrect(wrong);
    setSubmitted(true);
  };

  const getFeedback = (qid, index) => {
    if (!submitted) return <FaQuestionCircle className="text-gray-400" />;
    return questions.find((q) => q.id === qid).correct === index ? (
      <FaCheckCircle className="text-green-500" />
    ) : (
      <FaTimesCircle className="text-red-500" />
    );
  };

  return (
    <section id="quiz" className="bg-black/60 text-white rounded-xl p-6 mt-12 border border-yellow-500 shadow-lg">
      <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: AI Governance & Risk Management
      </h2>

      {questions.map((q) => (
        <div key={q.id} className="mb-6">
          <p className="mb-2 font-medium">{q.question}</p>
          <ul className="space-y-2">
            {q.options.map((opt, index) => (
              <li key={index}>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name={q.id}
                    value={index}
                    onChange={() => handleChange(q.id, index)}
                    className="accent-yellow-500"
                    disabled={submitted}
                  />
                  <span>{opt}</span>
                  <span className="ml-2">{getFeedback(q.id, index)}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      ))}

      {!submitted ? (
        <button
          onClick={handleSubmit}
          className="mt-4 px-6 py-2 bg-yellow-500 hover:bg-yellow-600 text-black font-semibold rounded-full flex items-center gap-2"
        >
          <FaCheckCircle /> ตรวจคำตอบ
        </button>
      ) : (
        <>
          <p className="mt-4 text-green-400 font-medium flex items-center gap-2">
            <FaCheckCircle /> ✅ ได้ {score} / {questions.length} คะแนน
          </p>

          {incorrect.length > 0 && (
            <div className="mt-4 text-red-300">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <FaTimesCircle /> ❌ ข้อที่ตอบผิด:
              </p>
              <ul className="list-disc pl-6 space-y-2 text-sm">
                {incorrect.map((item, idx) => (
                  <li key={idx}>
                    <p>🔸 <strong>{item.question}</strong></p>
                    <p>คำตอบที่เลือก: <span className="text-red-400">{item.yourAnswer || "ไม่ได้เลือก"}</span></p>
                    <p>คำตอบที่ถูกต้อง: <span className="text-green-400">{item.correctAnswer}</span></p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
};

export default MiniQuiz_Day15;