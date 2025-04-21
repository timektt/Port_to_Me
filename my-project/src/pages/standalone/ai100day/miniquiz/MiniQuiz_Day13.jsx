import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day13 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Interpretability แตกต่างจาก Accuracy อย่างไร?",
      options: [
        "ใช้วัดว่าโมเดลแม่นแค่ไหน",
        "อธิบายเหตุผลเบื้องหลังการตัดสินใจของโมเดล",
        "วัดค่า Loss ของโมเดล",
      ],
      correct: 1,
    },
    {
      id: "q2",
      question: "Global Interpretability ใช้สำหรับอะไร?",
      options: [
        "ดูพฤติกรรมโมเดลต่อ input รายตัว",
        "อธิบายพฤติกรรมโดยรวมของโมเดล",
        "แสดง Heatmap ของภาพ",
      ],
      correct: 1,
    },
    {
      id: "q3",
      question: "เทคนิค SHAP ช่วยอะไรได้บ้าง?",
      options: [
        "อธิบายว่า feature ใดมีผลต่อผลลัพธ์รายตัว",
        "เพิ่มค่า Accuracy",
        "ใช้ train โมเดล",
      ],
      correct: 0,
    },
    {
      id: "q4",
      question: "ข้อดีของ LIME คืออะไร?",
      options: [
        "สามารถอธิบายผลลัพธ์โดยรวม",
        "ง่ายต่อการเข้าใจ และใช้ได้กับหลายชนิดข้อมูล",
        "ใช้กับภาพเท่านั้น",
      ],
      correct: 1,
    },
    {
      id: "q5",
      question: "Grad-CAM ใช้กับข้อมูลประเภทใด?",
      options: [
        "ข้อมูลตาราง",
        "ข้อความ",
        "ภาพ",
      ],
      correct: 2,
    },
    {
      id: "q6",
      question: "ปัญหาหลักของโมเดล Black-box คืออะไร?",
      options: [
        "แม่นไม่พอ",
        "ตีความหรืออธิบายได้ยาก",
        "ไม่สามารถเทรนได้เร็ว",
      ],
      correct: 1,
    },
    {
      id: "q7",
      question: "โมเดลใดอธิบายง่ายที่สุด?",
      options: [
        "Neural Network",
        "Random Forest",
        "Decision Tree",
      ],
      correct: 2,
    },
    {
      id: "q8",
      question: "เทคนิคใดใช้ดูว่า Token ไหนมีผลมากที่สุดใน NLP?",
      options: [
        "SHAP",
        "Attention Visualization",
        "Grad-CAM",
      ],
      correct: 1,
    },
    {
      id: "q9",
      question: "หากต้องการอธิบายการตัดสินใจรายตัวอย่างควรใช้?",
      options: [
        "Global Feature Importance",
        "Partial Dependence Plot",
        "Local Interpretability เช่น LIME/SHAP",
      ],
      correct: 2,
    },
    {
      id: "q10",
      question: "ข้อใดคือ Best Practice ของ Explainability?",
      options: [
        "แสดงเฉพาะ Accuracy",
        "สร้าง Dashboard และให้ Feedback จากผู้ใช้",
        "เลือกเฉพาะโมเดล Deep Learning เท่านั้น",
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
    <section
      id="quiz"
      className="bg-black/60 text-white rounded-xl p-6 mt-12 border border-yellow-500 shadow-lg"
    >
      <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Explainability & Interpretability
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
                    <p>
                      🔸 <strong>{item.question}</strong>
                    </p>
                    <p>
                      คำตอบที่เลือก: {" "}
                      <span className="text-red-400">
                        {item.yourAnswer || "ไม่ได้เลือก"}
                      </span>
                    </p>
                    <p>
                      คำตอบที่ถูกต้อง: {" "}
                      <span className="text-green-400">
                        {item.correctAnswer}
                      </span>
                    </p>
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

export default MiniQuiz_Day13;