import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day16 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดอธิบายความสามารถหลักของ Neural Network ได้ดีที่สุด?",
      options: [
        "สามารถตีความผลลัพธ์ได้ง่ายเสมอ",
        "เรียนรู้ฟังก์ชันไม่เป็นเชิงเส้นและมีโครงสร้างแบบลึก",
        "เหมาะกับข้อมูลที่มีขนาดเล็กและไม่ซับซ้อน"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "Activation Function มีบทบาทอย่างไรใน Neural Network?",
      options: [
        "ทำให้โมเดลเรียนรู้แบบเชิงเส้นเท่านั้น",
        "สร้างความไม่เชิงเส้นให้กับการเรียนรู้",
        "ใช้เฉพาะใน Output Layer เท่านั้น"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "Hidden Layer มีบทบาทสำคัญอย่างไร?",
      options: [
        "ใช้กำหนดประเภทของข้อมูล",
        "ใช้เรียนรู้ความสัมพันธ์เชิงซ้อนของข้อมูล",
        "มีหน้าที่สร้าง Loss Function"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "Feedforward คืออะไรในบริบทของ Neural Network?",
      options: [
        "การปรับพารามิเตอร์ของโมเดล",
        "การคำนวณค่าจาก Input ไปสู่ Output แบบไม่ย้อนกลับ",
        "การอัปเดต Gradient แบบวนซ้ำ"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "Loss Function ใช้เพื่ออะไร?",
      options: [
        "วัดความผิดพลาดของโมเดล",
        "กำหนดจำนวน Layer",
        "แสดงผลลัพธ์ของโมเดลบนหน้าจอ"
      ],
      correct: 0
    },
    {
      id: "q6",
      question: "ข้อใดเป็นข้อดีของการมี Network ที่ลึกมาก (Deep Network)?",
      options: [
        "ฝึกได้ง่ายเสมอ",
        "สามารถเรียนรู้ Feature ที่ซับซ้อนได้",
        "ไม่เสี่ยง Overfitting"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "ในปัญหา XOR ต้องใช้ Neural Network แบบใดจึงจะทำงานได้ดี?",
      options: [
        "Linear Regression",
        "Shallow Neural Network แบบไม่มี Hidden Layer",
        "Network ที่มี Hidden Layer อย่างน้อย 1 ชั้น"
      ],
      correct: 2
    },
    {
      id: "q8",
      question: "ReLU Activation Function มีสมการใด?",
      options: [
        "f(x) = 1 / (1 + e^-x)",
        "f(x) = tanh(x)",
        "f(x) = max(0, x)"
      ],
      correct: 2
    },
    {
      id: "q9",
      question: "Gradient Descent ทำหน้าที่อะไรในการเรียนรู้ของโมเดล?",
      options: [
        "ใช้คูณกับ Input โดยตรง",
        "ใช้ถ่ายทอดข้อมูลระหว่าง Layer",
        "ใช้ปรับพารามิเตอร์เพื่อลดข้อผิดพลาด"
      ],
      correct: 2
    },
    {
      id: "q10",
      question: "ข้อใดกล่าวถึงประโยชน์ของ Neural Network ได้ถูกต้องที่สุด?",
      options: [
        "สามารถเรียนรู้ข้อมูลแบบไม่เป็นเชิงเส้นได้โดยไม่ต้องพึ่ง Feature Engineering มาก",
        "เหมาะสำหรับข้อมูลเชิงเส้นเท่านั้น",
        "จำเป็นต้องมีจำนวน Feature จำกัดเสมอ"
      ],
      correct: 0
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Introduction to Neural Networks
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

export default MiniQuiz_Day16;