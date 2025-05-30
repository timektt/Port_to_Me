import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day44 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดกล่าวถึงผลกระทบของ Pooling ต่อ Translation Invariance ได้ถูกต้องที่สุด?",
      options: [
        "ลดความสามารถในการตรวจจับการเลื่อนตำแหน่งของวัตถุ",
        "เพิ่มความไวต่อความเปลี่ยนแปลงเล็กน้อยในภาพ",
        "ช่วยให้โมเดลทนทานต่อการเลื่อนตำแหน่งของวัตถุ"
      ],
      correct: 2
    },
    {
      id: "q2",
      question: "การใช้ Overlapping Pooling มีข้อดีเด่นใดเมื่อเทียบกับ Non-overlapping?",
      options: [
        "ลดเวลาในการคำนวณ",
        "ให้ข้อมูลที่หลากหลายมากขึ้นจากการทับซ้อน",
        "ลดความซ้ำซ้อนของข้อมูล"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "Global Average Pooling แตกต่างจาก Max Pooling อย่างไร?",
      options: [
        "Global Average ใช้ค่ามากที่สุดของแต่ละฟีเจอร์",
        "Global Average คำนวณเฉพาะขอบภาพ",
        "Global Average รวมค่าทุกพิกเซลในแต่ละฟีเจอร์แล้วหารเฉลี่ย"
      ],
      correct: 2
    },
    {
      id: "q4",
      question: "ข้อใดกล่าวถึงข้อจำกัดของการใช้ Pooling Layer ใน CNN ที่นักวิจัยยุคใหม่เริ่มวิจารณ์?",
      options: [
        "ทำให้เกิดการ overfitting ได้ง่าย",
        "การลดทอนข้อมูลที่สำคัญเกินไป",
        "เพิ่มความซับซ้อนของโครงสร้างโมเดล"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "การเปรียบเทียบระหว่าง Convolution กับ Pooling ในแง่การเรียนรู้คือข้อใด?",
      options: [
        "ทั้งคู่เป็นการลดมิติที่มีพารามิเตอร์เรียนรู้ได้",
        "Pooling ไม่เรียนรู้พารามิเตอร์ แต่ Convolution เรียนรู้ได้",
        "Pooling มีความสามารถสูงกว่าในการเรียนรู้ feature"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Pooling Layers & Spatial Reduction
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

export default MiniQuiz_Day44;
