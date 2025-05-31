import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day45 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดคือแนวคิดหลักของ Transfer Learning?",
      options: [
        "ฝึกโมเดลใหม่จากข้อมูลจำนวนมาก",
        "ใช้โมเดลที่ฝึกไว้แล้วมาประยุกต์กับงานใหม่",
        "สร้างโมเดลหลายตัวเพื่อทดสอบงานเดียวกัน"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "การ Freeze Layer ใน Transfer Learning มีจุดประสงค์ใด?",
      options: [
        "เพื่อฝึก layer ทั้งหมดพร้อมกัน",
        "เพื่อป้องกันไม่ให้ layer ที่เรียนรู้มาก่อนเปลี่ยนค่าระหว่างการฝึก",
        "เพื่อเพิ่มจำนวนพารามิเตอร์ให้โมเดลเรียนรู้ได้มากขึ้น"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "ข้อใดกล่าวถึงประโยชน์ของการใช้ Pretrained Model ได้ถูกต้อง?",
      options: [
        "ช่วยลดเวลาการฝึกและเพิ่มความแม่นยำ",
        "ต้องใช้ dataset เดิมเสมอในการฝึกซ้ำ",
        "เหมาะเฉพาะกับข้อมูลชนิดเดียวกันเท่านั้น"
      ],
      correct: 0
    },
    {
      id: "q4",
      question: "การปรับ Fine-tuning ใน Transfer Learning ควรทำอย่างไรในขั้นตอนหลัง?",
      options: [
        "ฝึกเฉพาะ Fully Connected Layer เท่านั้น",
        "ฝึกเฉพาะ Convolution Layer เท่านั้น",
        "ค่อย ๆ ปลดล็อก layer ลึก ๆ เพื่อปรับจูนให้เหมาะกับข้อมูลใหม่"
      ],
      correct: 2
    },
    {
      id: "q5",
      question: "ข้อใดไม่ใช่ข้อจำกัดของ Transfer Learning?",
      options: [
        "ข้อมูลใหม่ต้องมีความใกล้เคียงกับข้อมูลต้นทาง",
        "สามารถใช้ได้กับข้อมูลทุกชนิดโดยไม่ต้องปรับอะไร",
        "อาจต้องใช้เวลาทดลองเพื่อหา configuration ที่เหมาะสม"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Transfer Learning
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

export default MiniQuiz_Day45;
