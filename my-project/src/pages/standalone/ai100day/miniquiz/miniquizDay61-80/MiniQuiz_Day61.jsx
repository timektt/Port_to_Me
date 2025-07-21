"use client";
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day61 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดคือลักษณะเด่นของ Reinforcement Learning เมื่อเทียบกับ Supervised Learning?",
      options: [
        "เรียนรู้จาก label โดยตรงในทุกตัวอย่าง",
        "เรียนรู้จากการให้รางวัล (reward) จากการกระทำ",
        "ไม่ต้องการ feedback ใด ๆ จากสิ่งแวดล้อม"
      ],
      correct: 1,
    },
    {
      id: "q2",
      question: "ส่วนประกอบใดที่ไม่ใช่องค์ประกอบหลักของระบบ RL?",
      options: [
        "Agent",
        "Classifier",
        "Environment"
      ],
      correct: 1,
    },
    {
      id: "q3",
      question: "Q-Learning เป็นเทคนิคในกลุ่มใดของ Reinforcement Learning?",
      options: [
        "Model-based RL",
        "Supervised RL",
        "Model-free RL"
      ],
      correct: 2,
    },
    {
      id: "q4",
      question: "การใช้ reward ที่ล่าช้า (delayed reward) มีผลอย่างไรต่อการเรียนรู้?",
      options: [
        "ทำให้ agent ตัดสินใจได้เร็วขึ้นในทันที",
        "ต้องอาศัยการวางแผนและการเรียนรู้ระยะยาว",
        "ลดความแม่นยำของนโยบาย (policy) ลง"
      ],
      correct: 1,
    },
    {
      id: "q5",
      question: "ข้อใดเป็นตัวอย่างการใช้ RL ในโลกจริง?",
      options: [
        "การจัดหมวดหมู่รูปภาพด้วย CNN",
        "การเล่นเกม Go โดย AlphaGo",
        "การแปลภาษาแบบ Seq2Seq"
      ],
      correct: 1,
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Reinforcement Learning
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

export default MiniQuiz_Day61;
