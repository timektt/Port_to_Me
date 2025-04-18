// ✅ MiniQuiz_Day10.jsx
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day10 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "โมเดลที่มี Bias สูงมักมีลักษณะอย่างไร?",
      options: [
        "เข้าใจข้อมูลลึกซึ้งและละเอียด",
        "ทำนายได้แม่นยำบนชุดทดสอบ",
        "เรียนรู้น้อย ทำให้เข้าใจข้อมูลไม่พอ",
      ],
      correct: 2,
    },
    {
      id: "q2",
      question: "โมเดลที่มี Variance สูงจะมีปัญหาอย่างไร?",
      options: [
        "จำแนกได้ดีบนข้อมูลใหม่",
        "จำข้อมูลฝึกได้แม่นแต่ทำนายข้อมูลใหม่พลาด",
        "ไม่สามารถฝึกได้เลย",
      ],
      correct: 1,
    },
    {
      id: "q3",
      question: "Bias-Variance Tradeoff หมายถึงอะไร?",
      options: [
        "การเพิ่ม accuracy ด้วยการใช้ข้อมูลเยอะขึ้น",
        "การบาลานซ์ระหว่างโมเดลที่ง่ายและโมเดลที่ซับซ้อน",
        "การตัด feature ที่ไม่จำเป็นออกจากชุดข้อมูล",
      ],
      correct: 1,
    },
    {
      id: "q4",
      question: "วิธีใดช่วยลด Variance โดยไม่ลดศักยภาพของโมเดลมากเกินไป?",
      options: [
        "ใช้โมเดลที่มีพารามิเตอร์น้อยลง",
        "ใช้ Regularization เช่น L1, L2 หรือ Dropout",
        "ฝึกโมเดลด้วย learning rate ที่สูงมาก",
      ],
      correct: 1,
    },
    {
      id: "q5",
      question: "อะไรคือสิ่งที่บ่งบอกว่าโมเดล Overfit?",
      options: [
        "Training loss และ validation loss ลดลงพร้อมกัน",
        "Training loss ลดลง แต่ validation loss เริ่มเพิ่ม",
        "Validation loss คงที่แต่ training accuracy ลดลง",
      ],
      correct: 1,
    },
    {
      id: "q6",
      question: "Model Capacity คืออะไร?",
      options: [
        "ขนาด batch ที่ใช้ฝึกโมเดล",
        "ความสามารถของโมเดลในการเรียนรู้ข้อมูลที่ซับซ้อน",
        "จำนวน data ที่ใช้ในชุดฝึก",
      ],
      correct: 1,
    },
    {
      id: "q7",
      question: "โมเดลขนาดเล็กเกินไปมักเกิดปัญหาอะไร?",
      options: [
        "Overfitting",
        "Underfitting",
        "Generalization สูงเกินไป",
      ],
      correct: 1,
    },
    {
      id: "q8",
      question: "เทคนิค Cross Validation ช่วยในการเลือกอะไร?",
      options: [
        "Optimizer",
        "Model Complexity ที่เหมาะสม",
        "จำนวน epochs ที่ใช้",
      ],
      correct: 1,
    },
    {
      id: "q9",
      question: "โมเดลขนาดใหญ่ควรใช้อะไรเพื่อควบคุม Variance?",
      options: [
        "เพิ่มจำนวน hidden layers",
        "Early Stopping และ Regularization",
        "เพิ่มจำนวนพารามิเตอร์แบบไม่มีข้อจำกัด",
      ],
      correct: 1,
    },
    {
      id: "q10",
      question: "ประโยชน์ของ Learning Curve คืออะไร?",
      options: [
        "แสดงการเปลี่ยนแปลงของ accuracy บน training set เท่านั้น",
        "ช่วยวิเคราะห์ว่าควรเพิ่มหรือลดขนาดโมเดล",
        "ใช้แทน confusion matrix ได้เลย",
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Bias-Variance & Model Capacity
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
                      คำตอบที่เลือก: <span className="text-red-400">{item.yourAnswer || "ไม่ได้เลือก"}</span>
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

export default MiniQuiz_Day10;
