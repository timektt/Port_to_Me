import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day18 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ปัญหาใหญ่ที่มักเกิดจากการ Initial Weights แบบสุ่มธรรมดาคืออะไร?",
      options: [
        "Overfitting อย่างรุนแรง",
        "Dead Neurons และ Symmetry Problem",
        "Training Speed สูงเกินไป"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "Zero Initialization ส่งผลอย่างไรต่อการเรียนรู้ของ Neural Network?",
      options: [
        "ช่วยเร่งความเร็วการฝึก",
        "ทำให้ Neurons เรียนรู้เหมือนกันหมด",
        "ลดโอกาสเกิด Overfitting"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "Xavier Initialization ออกแบบมาเพื่อทำงานได้ดีที่สุดกับฟังก์ชันการกระตุ้นประเภทใด?",
      options: [
        "ReLU และ Variants",
        "Sigmoid และ Tanh",
        "Softmax เท่านั้น"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "He Initialization มีการกำหนด Variance อย่างไร?",
      options: [
        "Variance = 2 / (n_in)",
        "Variance = 1 / (n_in + n_out)",
        "Variance = 1 / (n_out)"
      ],
      correct: 0
    },
    {
      id: "q5",
      question: "Orthogonal Initialization มีประโยชน์อย่างไรใน RNNs?",
      options: [
        "ลดขนาดของเวกเตอร์ข้อมูล",
        "ป้องกัน Gradient หดตัวหรือระเบิด",
        "เพิ่มจำนวน Hidden Layers"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "LeCun Initialization ถูกออกแบบมาให้เหมาะกับฟังก์ชันการกระตุ้นใด?",
      options: [
        "Sigmoid",
        "ReLU",
        "SELU"
      ],
      correct: 2
    },
    {
      id: "q7",
      question: "Data-dependent Initialization เช่น PCA Pretraining มีข้อดีอย่างไร?",
      options: [
        "ช่วยลดขนาดข้อมูลอินพุต",
        "ทำให้โมเดลเริ่มต้นใกล้โครงสร้างข้อมูลจริงมากขึ้น",
        "ลดจำนวนเลเยอร์ของโมเดล"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "ในการใช้ Batch Normalization การเลือก Weight Initialization มีผลน้อยลงเพราะเหตุใด?",
      options: [
        "BatchNorm ช่วยปรับ Mean และ Variance ของ Activation",
        "BatchNorm ช่วยลดขนาดของโมเดล",
        "BatchNorm ลดจำนวนพารามิเตอร์"
      ],
      correct: 0
    },
    {
      id: "q9",
      question: "การตั้งค่า Bias เป็นศูนย์ตั้งแต่แรกมีข้อดีใด?",
      options: [
        "ทำให้สัญญาณเริ่มต้นมีความสมมาตร",
        "ช่วยเร่งให้เกิดการ Saturation ในฟังก์ชัน Sigmoid",
        "ลดจำนวน Epoch ที่ต้องฝึก"
      ],
      correct: 0
    },
    {
      id: "q10",
      question: "ในกรณีใช้ ReLU ควรตั้ง Bias เริ่มต้นอย่างไรเพื่อลดปัญหา Dead Neurons?",
      options: [
        "ตั้งเป็นค่าลบเล็กน้อย",
        "ตั้งเป็นศูนย์เสมอ",
        "ตั้งเป็นค่าบวกเล็กน้อย เช่น 0.01"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Weight Initialization Strategies
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

export default MiniQuiz_Day18;
