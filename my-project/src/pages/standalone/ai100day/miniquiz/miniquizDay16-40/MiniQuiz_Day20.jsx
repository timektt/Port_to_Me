import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day20 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Internal Covariate Shift คืออะไร?",
      options: [
        "การเปลี่ยนแปลงของ distribution ของ input ตลอดการฝึก",
        "การเปลี่ยน shape ของ Model",
        "การเพิ่มขนาด batch แบบสุ่ม"
      ],
      correct: 0
    },
    {
      id: "q2",
      question: "Batch Normalization ทำอะไรกับ Activation?",
      options: [
        "Normalize Activation ภายใน Mini-batch",
        "ลดขนาด Layer อัตโนมัติ",
        "แปลง Activation เป็นค่า Binary"
      ],
      correct: 0
    },
    {
      id: "q3",
      question: "Layer Normalization ต่างจาก Batch Normalization อย่างไร?",
      options: [
        "LN คำนวณ Mean/Variance ข้าม Batch",
        "LN คำนวณ Mean/Variance ต่อหนึ่งตัวอย่าง",
        "LN ใช้ Running Statistics"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "ข้อดีหลักของการใช้ Batch Normalization คืออะไร?",
      options: [
        "ลด Internal Covariate Shift และเร่งการฝึก",
        "เพิ่มความซับซ้อนของโมเดล",
        "ลดขนาดของ Optimizer"
      ],
      correct: 0
    },
    {
      id: "q5",
      question: "เมื่อใดควรใช้ Layer Normalization แทน Batch Normalization?",
      options: [
        "เมื่อ batch size ใหญ่",
        "เมื่อ batch size เล็กมากหรือเป็น RNN, Transformer",
        "เมื่อ training data มี noise สูง"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "Group Normalization แก้ปัญหาอะไร?",
      options: [
        "การลดขนาด Weight",
        "การใช้งานเมื่อ batch size เล็ก",
        "การเพิ่มขนาด Layer"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "Weight Normalization มีแนวคิดหลักอย่างไร?",
      options: [
        "Normalize Activations",
        "Normalize Weights แทน Activations",
        "Normalize Batch Size"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "ค่าพารามิเตอร์ gamma (γ) และ beta (β) ใน Normalization มีไว้เพื่ออะไร?",
      options: [
        "ควบคุม Scale และ Shift หลัง Normalize",
        "กำหนดขนาดของ Batch",
        "ควบคุม Learning Rate"
      ],
      correct: 0
    },
    {
      id: "q9",
      question: "ข้อเสียของ Batch Normalization คืออะไร?",
      options: [
        "ขึ้นอยู่กับลำดับตัวอย่างใน Batch",
        "ไม่สามารถทำงานร่วมกับ CNN ได้",
        "ไม่สามารถใช้ได้กับ GPU"
      ],
      correct: 0
    },
    {
      id: "q10",
      question: "Insight หลักจากการใช้ Normalization คืออะไร?",
      options: [
        "ทำให้ Loss Landscape ลาดชันกว่าเดิม",
        "ทำให้ Loss Landscape เรียบและ Optimization ง่ายขึ้น",
        "ทำให้ Batch Size เล็กลงอัตโนมัติ"
      ],
      correct: 1
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: BatchNorm & LayerNorm
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

export default MiniQuiz_Day20;
