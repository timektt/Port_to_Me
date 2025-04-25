import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day17 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Perceptron ทำงานโดยใช้กลไกอะไรเป็นหลัก?",
      options: [
        "การจับกลุ่มข้อมูลที่ซับซ้อน",
        "การคำนวณ Weighted Sum และ Step Activation",
        "การเรียนรู้แบบย้อนกลับ"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "เหตุใด Perceptron แบบเดี่ยวจึงไม่สามารถแก้ปัญหา XOR ได้?",
      options: [
        "เพราะมี Activation Function ที่ไม่เหมาะสม",
        "เพราะมีจำนวนน้ำหนักไม่เพียงพอ",
        "เพราะเส้นแบ่งข้อมูลไม่เป็นเชิงเส้น"
      ],
      correct: 2
    },
    {
      id: "q3",
      question: "ข้อใดอธิบายสถาปัตยกรรมของ Multi-Layer Perceptron (MLP) ได้ถูกต้อง?",
      options: [
        "Input Layer → Activation → Output",
        "Input → Hidden Layer(s) → Output Layer",
        "Input → Output โดยไม่มี Hidden Layer"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "Activation Function ใดที่มักใช้ใน Hidden Layer ของ MLP?",
      options: ["Softmax", "ReLU", "Linear"],
      correct: 1
    },
    {
      id: "q5",
      question: "ในการฝึก MLP การใช้ Backpropagation มีวัตถุประสงค์ใด?",
      options: [
        "เพิ่มขนาดของข้อมูลเทรน",
        "ปรับน้ำหนักโดยอิงจาก Gradient ของ Loss",
        "สุ่มเลือก Feature ที่สำคัญ"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "การใช้ Hidden Layer จำนวนมากเกินไปอาจนำไปสู่ปัญหาใด?",
      options: [
        "Underfitting",
        "Gradient Exploding และ Overfitting",
        "Learning Rate ต่ำเกินไป"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "ในปัญหา MNIST มักใช้ MLP เพื่ออะไร?",
      options: [
        "การสร้างภาพจากตัวเลข",
        "การจำแนกตัวเลขเขียนด้วยมือ",
        "การลดขนาดของข้อมูลภาพ"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "MLP ต่างจาก Perceptron แบบเดี่ยวอย่างไร?",
      options: [
        "มีค่า Loss ต่ำกว่า",
        "มี Hidden Layer ทำให้เรียนรู้ non-linearity ได้",
        "ไม่มีการใช้ Activation Function"
      ],
      correct: 1
    },
    {
      id: "q9",
      question: "ตัวแปรใดที่ไม่ใช่ Hyperparameter ของ MLP?",
      options: ["จำนวน Layer", "Batch Size", "Output ของ Softmax"],
      correct: 2
    },
    {
      id: "q10",
      question: "ข้อใดคือ Insight ที่ถูกต้องเกี่ยวกับ MLP?",
      options: [
        "เหมาะสำหรับข้อมูลที่เป็นลำดับเท่านั้น",
        "ไม่สามารถใช้กับข้อมูลภาพหรือข้อความ",
        "เป็นรากฐานสำคัญของ Neural Network สมัยใหม่"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Perceptron & Multi-Layer Perceptron
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

export default MiniQuiz_Day17;
