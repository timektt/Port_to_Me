// ✅ MiniQuiz_Day12.jsx
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day12 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Overfitting คืออะไร?",
      options: [
        "โมเดลที่ไม่สามารถเรียนรู้ข้อมูลได้เลย",
        "โมเดลที่เรียนรู้ข้อมูลมากเกินจนไม่ generalize",
        "โมเดลที่ใช้ข้อมูลน้อยเกินไปในการฝึก",
      ],
      correct: 1,
    },
    {
      id: "q2",
      question: "สัญญาณว่าโมเดล Underfit คืออะไร?",
      options: [
        "Train Accuracy สูง แต่ Test Accuracy ต่ำ",
        "Train Accuracy ต่ำ และ Test Accuracy ก็ต่ำ",
        "Test Accuracy สูงมาก",
      ],
      correct: 1,
    },
    {
      id: "q3",
      question: "สิ่งใดช่วยลด Overfitting ได้?",
      options: [
        "เพิ่มจำนวน Epoch",
        "ใช้ Dropout หรือ Regularization",
        "เพิ่ม Layer ของโมเดล",
      ],
      correct: 1,
    },
    {
      id: "q4",
      question: "Bias สูงมักเกี่ยวข้องกับ?",
      options: [
        "โมเดลซับซ้อนเกินไป",
        "โมเดลเรียบง่ายเกินไป",
        "ข้อมูลไม่สมดุล",
      ],
      correct: 1,
    },
    {
      id: "q5",
      question: "Variance สูงบ่งบอกว่าโมเดล?",
      options: [
        "ไม่เข้าใจข้อมูลพื้นฐาน",
        "ตอบสนองต่อความเปลี่ยนแปลงเล็ก ๆ ในข้อมูลมากเกินไป",
        "เหมาะสมกับข้อมูลใหม่",
      ],
      correct: 1,
    },
    {
      id: "q6",
      question: "วิธีใดใช้วิเคราะห์พฤติกรรมโมเดล?",
      options: [
        "Confusion Matrix และ Learning Curve",
        "Precision อย่างเดียว",
        "เพียงแค่ดูค่า Accuracy",
      ],
      correct: 0,
    },
    {
      id: "q7",
      question: "อะไรเป็นข้อดีของการใช้ Early Stopping?",
      options: [
        "ทำให้โมเดลฝึกเร็วขึ้นอย่างเดียว",
        "หลีกเลี่ยงการเรียนรู้ noise จาก training data",
        "ช่วยให้ Test Set ใหญ่ขึ้น",
      ],
      correct: 1,
    },
    {
      id: "q8",
      question: "เมื่อ Validation Loss เพิ่มแต่ Training Loss ลดลงแสดงว่า?",
      options: [
        "เกิด Underfitting",
        "เกิด Overfitting",
        "โมเดลเสถียร",
      ],
      correct: 1,
    },
    {
      id: "q9",
      question: "Ensemble Learning ช่วยในเรื่องใด?",
      options: [
        "ลดความเสี่ยงจาก Underfitting",
        "ลด Variance และเพิ่มความเสถียรของโมเดล",
        "ลดจำนวนข้อมูลที่ใช้",
      ],
      correct: 1,
    },
    {
      id: "q10",
      question: "Checklist ก่อน deploy ควรมีข้อใด?",
      options: [
        "ตรวจเฉพาะ Train Accuracy",
        "ตรวจพฤติกรรม FP/FN และ Validation Stability",
        "ใช้ Test Set ในการเทรนซ้ำ",
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Overfitting & Model Diagnostics
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
                      คำตอบที่เลือก:{" "}
                      <span className="text-red-400">
                        {item.yourAnswer || "ไม่ได้เลือก"}
                      </span>
                    </p>
                    <p>
                      คำตอบที่ถูกต้อง:{" "}
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

export default MiniQuiz_Day12;
