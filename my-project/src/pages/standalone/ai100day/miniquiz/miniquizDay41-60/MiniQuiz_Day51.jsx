"use client";
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day51 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Transformer ถูกพัฒนาขึ้นมาเพื่อแก้ปัญหาหลักข้อใดของ RNN?",
      options: [
        "การทำงานแบบ sequential ที่ขนานไม่ได้",
        "การต้องใช้ hand-crafted features",
        "การไม่สามารถเรียนรู้ pattern แบบ non-linear ได้"
      ],
      correct: 0,
    },
    {
      id: "q2",
      question: "Paper 'Attention Is All You Need' นำเสนอแนวคิดหลักใด?",
      options: [
        "ใช้ self-attention เป็น core component โดยไม่ต้องมี recurrent หรือ convolution layer",
        "เพิ่มจำนวน recurrent layer เพื่อเรียนรู้ข้อมูล sequence ที่ซับซ้อนขึ้น",
        "ใช้ reinforcement learning ร่วมกับ LSTM"
      ],
      correct: 0,
    },
    {
      id: "q3",
      question: "Self-Attention มีข้อดีสำคัญข้อใดเมื่อเทียบกับ RNN?",
      options: [
        "ประมวลผลลำดับได้แบบขนาน (parallelizable)",
        "ต้องใช้ parameter น้อยลงมากกว่าเดิม",
        "ใช้เวลาในการ training มากกว่าเพื่อให้ได้ความแม่นยำสูง"
      ],
      correct: 0,
    },
    {
      id: "q4",
      question: "Positional Encoding ใน Transformer มีบทบาทอย่างไร?",
      options: [
        "ช่วยให้ model เข้าใจลำดับ (order) ของ input tokens",
        "เพิ่มค่า loss ให้อยู่ในระดับที่โมเดลเรียนรู้ได้ง่ายขึ้น",
        "ลดความจำเป็นในการใช้ optimizer ที่ซับซ้อน"
      ],
      correct: 0,
    },
    {
      id: "q5",
      question: "Residual Connection + Layer Normalization มีผลอย่างไรต่อการฝึกโมเดล?",
      options: [
        "ช่วยให้ gradient flow ดีขึ้นและ training มีเสถียรภาพมากขึ้น",
        "ช่วยลดขนาดของโมเดลโดยตรง",
        "ทำให้การเรียนรู้ bias และ variance เป็นอิสระมากขึ้น"
      ],
      correct: 0,
    },
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
          correctAnswer: q.options[q.correct],
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Introduction to Transformers
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

export default MiniQuiz_Day51;
