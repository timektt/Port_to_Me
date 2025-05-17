// src/pages/courses/miniquiz/miniquizDay16-40/MiniQuiz_Day34.jsx

import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day34 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ในสถาปัตยกรรม Encoder–Decoder ส่วนของ Encoder มีหน้าที่หลักใด?",
      options: [
        "สร้าง token ใหม่จากข้อมูลที่มี",
        "แปลงอินพุตเป็น representation ที่มีบริบท",
        "เลือกตำแหน่งของ token ที่จะโฟกัสแบบ random"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "กลไก Cross-Attention ภายใน Decoder ใช้ Key และ Value จากไหน?",
      options: [
        "จาก Decoder เองทุกชั้น",
        "จาก Encoder output",
        "จาก Positional Encoding เท่านั้น"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "ข้อใดคือข้อได้เปรียบของสถาปัตยกรรม Encoder–Decoder เมื่อเทียบกับโมโนลิทิกโมเดล?",
      options: [
        "ใช้พารามิเตอร์น้อยลงเสมอ",
        "ไม่มีการใช้ attention",
        "สามารถแยกการประมวลผล input/output ได้ยืดหยุ่น"
      ],
      correct: 2
    },
    {
      id: "q4",
      question: "ใน Transformer Decoder ใช้ Masked Self-Attention เพื่อวัตถุประสงค์ใด?",
      options: [
        "ให้ Decoder เห็นลำดับถัดไปล่วงหน้า",
        "ป้องกันไม่ให้เกิด Cross-Attention ซ้ำ",
        "จำกัดการมองเห็น token ถัดไปเพื่อสร้างลำดับแบบ autoregressive"
      ],
      correct: 2
    },
    {
      id: "q5",
      question: "ข้อใดกล่าวถึงการใช้งาน Encoder–Decoder ใน Multimodal AI ได้ถูกต้องที่สุด?",
      options: [
        "Encoder ใช้สำหรับเลือกโมเดลที่เหมาะสมใน runtime",
        "Decoder ใช้เฉพาะงาน NLP เท่านั้น",
        "Encoder ประมวลผลข้อมูลภาพ/เสียง ส่วน Decoder แปลงเป็นภาษาธรรมชาติ"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Encoder–Decoder Structure
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

export default MiniQuiz_Day34;
