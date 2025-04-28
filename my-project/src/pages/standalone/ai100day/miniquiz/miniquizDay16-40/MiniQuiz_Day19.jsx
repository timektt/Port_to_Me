import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day19 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ปัญหาหลักของ Basic Gradient Descent คืออะไร?",
      options: [
        "ใช้ข้อมูลน้อยเกินไปในการอัปเดต",
        "Convergence ช้าและติด Saddle Points",
        "ต้องการหน่วยความจำต่ำเกินไป"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "SGD มีข้อดีหลักอะไรเมื่อเทียบกับ Full Batch Gradient Descent?",
      options: [
        "เสถียรกว่าและไม่มีความผันผวน",
        "สามารถข้าม Local Minima ได้ด้วย Noise",
        "ใช้เวลา Epoch ต่อรอบมากขึ้น"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "Mini-Batch Gradient Descent เป็นการผสมผสานข้อดีของอะไร?",
      options: [
        "SGD และ RMSProp",
        "SGD และ Full Batch Gradient Descent",
        "Momentum และ NAG"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "Momentum มีบทบาทสำคัญอย่างไรในการฝึกโมเดล?",
      options: [
        "ช่วยให้ Learning Rate เพิ่มขึ้นอัตโนมัติ",
        "เร่งการลู่เข้าและลดการแกว่ง",
        "ลดขนาดของโมเดล"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "หลักการของ Nesterov Accelerated Gradient (NAG) คืออะไร?",
      options: [
        "คำนวณ Gradient ที่ตำแหน่งปัจจุบัน",
        "คำนวณ Gradient ณ ตำแหน่งที่ Lookahead",
        "ไม่ใช้ Momentum ในการอัปเดต"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "AdaGrad มีจุดเด่นอย่างไร?",
      options: [
        "ลด Learning Rate แบบคงที่",
        "ปรับ Learning Rate ตามประวัติ Gradient แต่ละตัว",
        "เพิ่มขนาด Batch Size อัตโนมัติ"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "RMSProp แก้ปัญหาหลักของ AdaGrad อย่างไร?",
      options: [
        "ใช้ Moving Average ของ Squared Gradients",
        "ลด Momentum ทุกรอบ",
        "เพิ่มจำนวนพารามิเตอร์"
      ],
      correct: 0
    },
    {
      id: "q8",
      question: "Adam รวมข้อดีของ Optimizer ใดบ้าง?",
      options: [
        "Momentum และ RMSProp",
        "SGD และ AdaGrad",
        "NAG และ RMSProp"
      ],
      correct: 0
    },
    {
      id: "q9",
      question: "ข้อเสียของการใช้ Adam คืออะไรในบางกรณี?",
      options: [
        "ทำให้โมเดลมีขนาดเล็กเกินไป",
        "อาจนำโมเดลไปติด Local Minima ที่ไม่ดี",
        "ใช้หน่วยความจำน้อยเกินไป"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "แนวทางที่ดีที่สุดในการเลือก Optimizer คืออะไร?",
      options: [
        "ใช้ Adam เสมอในทุกปัญหา",
        "เลือก Optimizer ตามลักษณะข้อมูลและปัญหา",
        "ไม่ต้องจูน Learning Rate"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Gradient Descent Variants
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

export default MiniQuiz_Day19;
