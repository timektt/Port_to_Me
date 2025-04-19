// ✅ MiniQuiz_Day11.jsx
import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day11 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "จุดประสงค์หลักของ Cross Validation คืออะไร?",
      options: [
        "เพื่อฝึกโมเดลหลายรอบใน training set เดียวกัน",
        "เพื่อให้การประเมินโมเดลไม่ขึ้นอยู่กับการแบ่งข้อมูลครั้งเดียว",
        "เพื่อให้ได้ผลลัพธ์จาก test set ที่ดีที่สุด",
      ],
      correct: 1,
    },
    {
      id: "q2",
      question: "ข้อเสียหลักของ K-Fold Cross Validation คืออะไร?",
      options: [
        "ใช้ข้อมูลไม่คุ้มค่า",
        "ไม่สามารถประเมินบนข้อมูลใหม่ได้",
        "ใช้เวลาในการประมวลผลนาน",
      ],
      correct: 2,
    },
    {
      id: "q3",
      question: "Stratified K-Fold เหมาะกับปัญหาแบบใด?",
      options: [
        "Regression",
        "Classification ที่ class imbalance",
        "Time Series Forecasting",
      ],
      correct: 1,
    },
    {
      id: "q4",
      question: "Leave-One-Out Cross Validation เหมาะกับกรณีใด?",
      options: [
        "ข้อมูลขนาดใหญ่",
        "ข้อมูลที่ class สมดุล",
        "ข้อมูลมีจำนวนน้อยมาก",
      ],
      correct: 2,
    },
    {
      id: "q5",
      question: "Metric ใดที่ใช้กับ Regression?",
      options: [
        "F1-Score",
        "Mean Absolute Error (MAE)",
        "Precision",
      ],
      correct: 1,
    },
    {
      id: "q6",
      question: "อะไรคือสิ่งที่ควรหลีกเลี่ยงเมื่อใช้ Test Set?",
      options: [
        "ใช้ในการเลือกโมเดล",
        "ใช้ในการประเมินขั้นสุดท้ายเท่านั้น",
        "ใช้ร่วมกับ Cross Validation",
      ],
      correct: 0,
    },
    {
      id: "q7",
      question: "ROC-AUC ใช้วัดอะไร?",
      options: [
        "ระดับความสมดุลของ precision กับ recall",
        "ความสามารถในการแยก class ได้ดี",
        "ขนาดของ training set ที่เหมาะสม",
      ],
      correct: 1,
    },
    {
      id: "q8",
      question: "การใช้ GridSearchCV ควรใช้ร่วมกับอะไรเพื่อประเมินโมเดล?",
      options: [
        "Test Set",
        "Cross Validation",
        "Confusion Matrix",
      ],
      correct: 1,
    },
    {
      id: "q9",
      question: "เหตุผลที่ไม่ควรพึ่ง Accuracy เพียงอย่างเดียวคืออะไร?",
      options: [
        "ไม่สามารถวัด performance ได้เลย",
        "ใช้ได้เฉพาะกับ Regression",
        "อาจหลอกได้ในกรณีข้อมูลไม่สมดุล",
      ],
      correct: 2,
    },
    {
      id: "q10",
      question: "Best Practice ในการแบ่งข้อมูลคืออะไร?",
      options: [
        "Train/Test 50/50",
        "Train/Validation/Test และไม่ใช้ Test Set ซ้ำ",
        "ใช้ Test Set เพื่อ tuning parameter",
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Cross Validation & Evaluation
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

export default MiniQuiz_Day11;
