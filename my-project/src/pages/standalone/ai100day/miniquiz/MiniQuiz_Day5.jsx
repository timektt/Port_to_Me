import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day5 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Linear Transformation มีคุณสมบัติใด?",
      options: [
        "ทำให้ข้อมูลโค้งงอได้",
        "ยืด/หมุน/สะท้อนเวกเตอร์โดยยังคงความเป็นเส้นตรง",
        "ทำให้เวกเตอร์ขาดความต่อเนื่อง"
      ],
      correct: 1,
    },
    {
      id: "q2",
      question: "เมทริกซ์ใน Neural Network มีบทบาทใด?",
      options: [
        "ทำให้ระบบมีเสียงพูด",
        "สร้างการเปลี่ยนมุมมองของข้อมูลในแต่ละเลเยอร์",
        "ทำให้ระบบรันได้เร็วขึ้นเท่านั้น"
      ],
      correct: 1,
    },
    {
      id: "q3",
      question: "Feature ที่ถูกดึงจากภาพ เช่น ขอบ หรือ texture มาจากอะไร?",
      options: [
        "การคูณเวกเตอร์กับ scalar",
        "การทำ convolution ด้วยเมทริกซ์ filter",
        "การลดทอนความละเอียด"
      ],
      correct: 1,
    },
    {
      id: "q4",
      question: "สิ่งใดคือข้อจำกัดของ Linear Transformation?",
      options: [
        "ไม่สามารถเปลี่ยนทิศทางเวกเตอร์ได้",
        "ใช้ได้เฉพาะกับข้อมูล 2 มิติ",
        "ไม่สามารถเรียนรู้รูปแบบไม่เป็นเส้นตรงได้"
      ],
      correct: 2,
    },
    {
      id: "q5",
      question: "เหตุผลที่ต้องใช้ Activation Function หลังการคูณเมทริกซ์คืออะไร?",
      options: [
        "เพิ่มความซับซ้อนของฟังก์ชันที่เรียนรู้ได้",
        "ลดขนาดของข้อมูล",
        "ทำให้ค่าทุกตัวกลายเป็นศูนย์"
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

export default MiniQuiz_Day5;
