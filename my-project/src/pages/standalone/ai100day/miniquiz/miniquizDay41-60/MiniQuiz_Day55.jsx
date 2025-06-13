"use client";
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day55 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Seq2Seq Model ถูกออกแบบมาเพื่อแก้ปัญหาประเภทใด?",
      options: [
        "การแปลงข้อมูลจาก sequence หนึ่งเป็นอีก sequence หนึ่ง",
        "การจัดกลุ่มข้อมูลแบบ unsupervised",
        "การทำ object detection บนภาพนิ่ง"
      ],
      correct: 0,
    },
    {
      id: "q2",
      question: "ข้อได้เปรียบสำคัญของ Transformer-based Seq2Seq เมื่อเทียบกับ RNN-based Seq2Seq คืออะไร?",
      options: [
        "สามารถประมวลผลแบบ parallel ได้ดี",
        "ต้องการ training data น้อยกว่า RNN",
        "ไม่ต้องใช้ positional encoding"
      ],
      correct: 0,
    },
    {
      id: "q3",
      question: "Self-Attention ใน Transformer-based Seq2Seq มีบทบาทในด้านใด?",
      options: [
        "ช่วยให้โมเดลจับ context ของ token ได้ทั่วทั้ง sequence",
        "ลด latency ในการ deploy บน edge device",
        "ลดจำนวน parameter ใน encoder layer"
      ],
      correct: 0,
    },
    {
      id: "q4",
      question: "ข้อจำกัดสำคัญที่ต้องพิจารณาเมื่อใช้งาน Seq2Seq Transformer คืออะไร?",
      options: [
        "มีค่า memory และ computational cost สูง",
        "ไม่สามารถแปลภาษาได้แบบ bidirectional",
        "ไม่สามารถนำไปใช้กับ time series data ได้เลย"
      ],
      correct: 0,
    },
    {
      id: "q5",
      question: "เทคนิคใดต่อไปนี้เป็น Best Practice ในการ Train Seq2Seq Transformer?",
      options: [
        "ใช้ Beam Search ในการ Inference",
        "ไม่ต้องทำ data augmentation ใดๆ",
        "Train ด้วย batch size ขนาดเล็กเสมอ"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Sequence-to-Sequence with Transformers
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

export default MiniQuiz_Day55;
