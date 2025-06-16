"use client";
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day56 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Image Captioning คือการรวมความสามารถใดเข้าด้วยกัน?",
      options: [
        "Computer Vision และ Natural Language Processing",
        "Reinforcement Learning และ Data Mining",
        "Speech Recognition และ Sentiment Analysis"
      ],
      correct: 0,
    },
    {
      id: "q2",
      question: "Model ใดใช้ Contrastive Learning ในการจับคู่ภาพกับข้อความ?",
      options: [
        "CLIP",
        "Show and Tell",
        "GIT (Generative Image-to-Text)"
      ],
      correct: 0,
    },
    {
      id: "q3",
      question: "โมเดล BLIP มีจุดเด่นด้านใด?",
      options: [
        "การใช้ Vision Transformer ผสานกับ Language Decoder",
        "การใช้ rule-based caption generation",
        "การเรียนรู้จากเสียงและภาพพร้อมกัน"
      ],
      correct: 0,
    },
    {
      id: "q4",
      question: "ข้อใดคือความท้าทายหลักของ Image Captioning?",
      options: [
        "Bias ใน dataset และการตีความเชิงเหตุผลที่ซับซ้อน",
        "ไม่สามารถใช้ร่วมกับ pretraining ได้",
        "ไม่สามารถใช้กับระบบ E-commerce ได้"
      ],
      correct: 0,
    },
    {
      id: "q5",
      question: "Flamingo เป็นโมเดล Vision-Language ที่มีคุณสมบัติใด?",
      options: [
        "Few-shot learning โดยใช้ตัวอย่างเพียงเล็กน้อย",
        "ฝึกด้วย supervised loss บน Text-only corpus",
        "ไม่มีการใช้ attention layer ระหว่างภาพกับข้อความ"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Image Captioning & Vision-Language
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

export default MiniQuiz_Day56;
