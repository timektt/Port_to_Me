import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day14 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Fairness ใน AI หมายถึงอะไร?",
      options: [
        "การทำให้โมเดลเรียนรู้เร็วขึ้น",
        "การทำให้โมเดลมีความแม่นยำสูง",
        "การตัดสินใจของโมเดลไม่ลำเอียงต่อกลุ่มใดกลุ่มหนึ่ง"
      ],
      correct: 2
    },
    {
      id: "q2",
      question: "Sampling Bias เกิดจากอะไร?",
      options: [
        "การเลือกข้อมูลที่ไม่หลากหลาย",
        "การเก็บข้อมูลจากหลายแหล่ง",
        "การเทรนโมเดลด้วยข้อมูลจำลอง"
      ],
      correct: 0
    },
    {
      id: "q3",
      question: "Metric ใดที่ใช้วัด fairness ระหว่างกลุ่ม?",
      options: [
        "RMSE",
        "Equal Opportunity",
        "Precision"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "Label Bias คืออะไร?",
      options: [
        "โมเดลไม่สามารถระบุ label ได้",
        "การให้ label โดยอิงจากความเห็นเฉพาะกลุ่ม",
        "label มีค่า missing ใน dataset"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "เทคนิคในขั้นตอน preprocessing เพื่อลด bias คือ?",
      options: [
        "Gradient Clipping",
        "Reweighing",
        "Early Stopping"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "สิ่งใดเป็นหลักการของ Ethical AI?",
      options: [
        "Accuracy, Latency, Overfitting",
        "Transparency, Fairness, Accountability",
        "Gradient, Loss, Bias"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "AI Alignment หมายถึงอะไร?",
      options: [
        "การปรับโมเดลให้แม่นยำขึ้น",
        "การปรับให้เป้าหมายของ AI สอดคล้องกับมนุษย์",
        "การลดเวลาในการเทรนโมเดล"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "Confusion Matrix แบบรายกลุ่มใช้เพื่ออะไร?",
      options: [
        "ดูค่าความแม่นยำรวมทั้งหมด",
        "ดู performance แยกตามกลุ่ม",
        "ดูว่าโมเดลเข้าใจ feature หรือไม่"
      ],
      correct: 1
    },
    {
      id: "q9",
      question: "Framework ใดเกี่ยวข้องกับจริยธรรม AI?",
      options: [
        "FastAI",
        "EU AI Act",
        "YOLO"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "Disparate Impact หมายถึงอะไร?",
      options: [
        "ความแตกต่างของผลลัพธ์ระหว่างกลุ่ม",
        "ค่าความสูญเสียของโมเดล",
        "การเปลี่ยนแปลงของ gradient"
      ],
      correct: 0
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
    <section
      id="quiz"
      className="bg-black/60 text-white rounded-xl p-6 mt-12 border border-yellow-500 shadow-lg"
    >
      <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Fairness & Ethical AI
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

export default MiniQuiz_Day14;