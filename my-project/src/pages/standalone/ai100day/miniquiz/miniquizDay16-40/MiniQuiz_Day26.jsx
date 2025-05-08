import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day26 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดคือคุณสมบัติสำคัญของข้อมูล Time Series?",
      options: [
        "ข้อมูลทุกแถวมีความเป็นอิสระ",
        "ลำดับของข้อมูลมีความสำคัญ",
        "สามารถสลับลำดับข้อมูลได้โดยไม่กระทบผลลัพธ์"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "Univariate Time Series หมายถึงอะไร?",
      options: [
        "ใช้หลายตัวแปรพร้อมกัน",
        "ใช้ตัวแปรเดียวที่เปลี่ยนตามเวลา",
        "ใช้ข้อมูลที่ไม่มีลำดับเวลา"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "ข้อใดคือลักษณะของข้อมูล Non-Stationary?",
      options: [
        "ค่าเฉลี่ยและความแปรปรวนคงที่",
        "ค่าเฉลี่ยและความแปรปรวนเปลี่ยนแปลงตามเวลา",
        "ไม่มีการเปลี่ยนแปลงในข้อมูล"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "Sequence → Value task เหมาะกับงานแบบใด?",
      options: [
        "การทำนายราคาหุ้นวันถัดไป",
        "การแปลประโยคภาษาอังกฤษ",
        "การจัดประเภทข้อความ"
      ],
      correct: 0
    },
    {
      id: "q5",
      question: "โมเดลใดเหมาะกับการเรียนรู้ลำดับระยะยาว?",
      options: [
        "Linear Regression",
        "LSTM",
        "Decision Tree"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "ข้อใดคือข้อดีของการใช้ Transformer กับ Time Series?",
      options: [
        "ประมวลผลแบบลำดับได้เท่านั้น",
        "สามารถเรียนรู้ context ที่อยู่ไกลได้ดี",
        "ใช้พารามิเตอร์น้อยมาก"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "Sliding Window Approach ใช้ประโยชน์อย่างไร?",
      options: [
        "วิเคราะห์ข้อมูลทั้งหมดพร้อมกัน",
        "ใช้ค่าล่าสุดเพื่อพยากรณ์ค่าถัดไป",
        "สุ่มข้อมูลมาใช้ในการเรียนรู้"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "MAE เหมาะกับสถานการณ์ใด?",
      options: [
        "ต้องการ penalize error ขนาดใหญ่",
        "เน้นความผิดพลาดสัมบูรณ์ในทุกช่วง",
        "ข้อมูลมีหน่วยต่างกัน"
      ],
      correct: 1
    },
    {
      id: "q9",
      question: "Feature Engineering ใดเหมาะกับข้อมูลที่มีฤดูกาล?",
      options: [
        "Lag Features",
        "Fourier Transform",
        "One-hot Encoding"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "Darts, Prophet, GluonTS เป็นไลบรารีที่ใช้กับอะไร?",
      options: [
        "Computer Vision",
        "Reinforcement Learning",
        "Time Series Forecasting"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Sequence Modeling & Time Series
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

export default MiniQuiz_Day26;
