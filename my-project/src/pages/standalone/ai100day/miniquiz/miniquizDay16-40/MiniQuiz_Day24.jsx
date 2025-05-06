import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day24 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดคือหน้าที่หลักของ CNN ในงานประมวลผลภาพ?",
      options: [
        "จำแนกวัตถุตามเสียงที่ได้ยิน",
        "ดึงคุณลักษณะเชิงพื้นที่จากภาพ",
        "แทนที่การประมวลผลด้วย RNN"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "เหตุใดการใช้ Convolution จึงเหมาะกับภาพ?",
      options: [
        "เพราะไม่ต้องใช้การเรียนรู้พารามิเตอร์",
        "เพราะช่วยลดขนาดภาพได้ทันที",
        "เพราะสามารถตรวจจับลักษณะเฉพาะของพื้นที่ในภาพได้"
      ],
      correct: 2
    },
    {
      id: "q3",
      question: "Feature Hierarchy ใน CNN หมายถึงอะไร?",
      options: [
        "การเรียนรู้ภาพจากระดับ pixel ไปยังระดับแนวคิด",
        "การเรียงลำดับ label ในภาพ",
        "การลดขนาดของ output map"
      ],
      correct: 0
    },
    {
      id: "q4",
      question: "เลเยอร์สุดท้ายใน CNN สำหรับการจำแนกมักเป็นอะไร?",
      options: [
        "Convolution Layer",
        "Fully Connected Layer",
        "Pooling Layer"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "Global Average Pooling ช่วยลดความเสี่ยงอะไรในโมเดล?",
      options: [
        "Overfitting จาก Fully Connected Layer",
        "Underfitting จาก Convolution",
        "Gradient Vanishing"
      ],
      correct: 0
    },
    {
      id: "q6",
      question: "โมเดลใดต่อไปนี้ใช้ CNN เป็นแกนหลัก?",
      options: [
        "Transformer",
        "RNN",
        "VGGNet"
      ],
      correct: 2
    },
    {
      id: "q7",
      question: "เหตุใด CNN จึงเหมาะสำหรับการจำแนกรูปภาพมากกว่า MLP?",
      options: [
        "CNN ใช้หน่วยความจำมากกว่า",
        "CNN ไม่ต้องการพารามิเตอร์จำนวนมาก",
        "CNN จับ spatial structure ได้ดีกว่า"
      ],
      correct: 2
    },
    {
      id: "q8",
      question: "ข้อใดคือการประยุกต์ของ CNN ที่ไม่ใช่งานประมวลผลภาพ?",
      options: [
        "การวิเคราะห์ข้อความ",
        "การเล่นเกม",
        "การทำนายเวลาในอนาคต"
      ],
      correct: 0
    },
    {
      id: "q9",
      question: "ข้อใดกล่าวถูกต้องเกี่ยวกับการใช้ Transfer Learning กับ CNN?",
      options: [
        "เริ่มจากการฝึกใหม่ทั้งหมดทุกครั้ง",
        "ใช้โมเดลที่ฝึกกับ dataset ใหญ่มาปรับกับงานใหม่",
        "เหมาะกับการเรียนรู้แบบ reinforcement เท่านั้น"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "Activation Function ที่นิยมใน CNN คืออะไร?",
      options: [
        "Sigmoid",
        "Tanh",
        "ReLU"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: CNN for Vision
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

export default MiniQuiz_Day24;
