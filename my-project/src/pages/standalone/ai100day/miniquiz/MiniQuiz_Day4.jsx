import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day4 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Matrix Multiplication ใช้ทำอะไรใน Neural Network?",
      options: [
        "แสดงผลลัพธ์ที่เข้าใจง่าย",
        "แปลงข้อมูลและเน้น feature สำคัญ",
        "เพิ่มความเร็วของระบบ"
      ],
      correct: 1,
    },
    {
      id: "q2",
      question: "ถ้าเมทริกซ์ A มีขนาด 3×2 จะสามารถคูณกับเมทริกซ์ B ขนาดใดได้?",
      options: [
        "2×3",
        "2×N",
        "3×2"
      ],
      correct: 1,
    },
    {
      id: "q3",
      question: "Linear Transformation หมายถึงอะไร?",
      options: [
        "การจัดเรียงเวกเตอร์ใหม่ให้สวยงาม",
        "การยืด หมุน หรือสะท้อนเวกเตอร์โดยรักษาเส้นตรง",
        "การสลับตำแหน่งของค่าทั้งหมดในเวกเตอร์"
      ],
      correct: 1,
    },
    {
      id: "q4",
      question: "คำใดสื่อความหมายของ Matrix Multiplication ได้ใกล้เคียงที่สุด?",
      options: ["การหมุนภาพ", "การเปลี่ยนมุมมอง", "การรวมข้อมูล"],
      correct: 1,
    },
    {
      id: "q5",
      question: "ในการทำ Attention เราต้องทำอะไรกับเมทริกซ์ K ก่อนคูณกับ Q?",
      options: ["Transpose", "Normalize", "Invert"],
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
    <section
      id="quiz"
      className="bg-black/60 text-white rounded-xl p-6 mt-12 border border-yellow-500 shadow-lg"
    >
      <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz
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
            <FaCheckCircle /> ✅ คุณได้ {score} / {questions.length} คะแนน
          </p>

          {incorrect.length > 0 && (
            <div className="mt-4 text-red-300">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <FaTimesCircle /> ❌ ข้อที่คุณตอบผิด:
              </p>
              <ul className="list-disc pl-6 space-y-2 text-sm">
                {incorrect.map((item, idx) => (
                  <li key={idx}>
                    <p>
                      🔸 <strong>{item.question}</strong>
                    </p>
                    <p>
                      คุณตอบ: <span className="text-red-400">{item.yourAnswer || "ไม่ได้เลือก"}</span>
                    </p>
                    <p>
                      คำตอบที่ถูกต้อง: <span className="text-green-400">{item.correctAnswer}</span>
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

export default MiniQuiz_Day4;
