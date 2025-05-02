import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day22 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "อะไรคือเหตุผลที่ AlexNet ใช้ ReLU แทน Sigmoid?",
      options: [
        "ReLU ช่วยลดขนาดของโมเดล",
        "ReLU มีความสามารถในการเรียนรู้ non-linearity ที่ลึกกว่า",
        "ReLU ช่วยให้ฝึกโมเดลได้เร็วขึ้นและลดปัญหา gradient หาย"
      ],
      correct: 2
    },
    {
      id: "q2",
      question: "ข้อใดไม่ใช่ layer พื้นฐานใน CNN Architecture?",
      options: [
        "Convolutional Layer",
        "Decision Tree Layer",
        "Pooling Layer"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "การใช้ 1x1 Convolution มีประโยชน์อย่างไร?",
      options: [
        "เพิ่ม spatial resolution",
        "ลดจำนวน channel และผสมข้อมูลข้าม channel",
        "เพิ่มจำนวนพารามิเตอร์ของโมเดล"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "ข้อใดคือข้อดีของการใช้ Residual Block ใน ResNet?",
      options: [
        "เพิ่มจำนวน filter ในแต่ละ layer",
        "ช่วยให้โมเดลสามารถฝึกได้ลึกโดยไม่เกิดปัญหา vanishing gradient",
        "ทำให้การเรียนรู้เร็วขึ้นด้วยการลดจำนวนเลเยอร์"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "การใช้ Group Convolution มีวัตถุประสงค์ใด?",
      options: [
        "เพิ่มจำนวน layer โดยไม่ลดประสิทธิภาพ",
        "แยก channel ออกเป็นกลุ่มย่อยเพื่อลดพารามิเตอร์",
        "เพิ่ม feature map ให้ละเอียดขึ้น"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "ข้อใดคือเทคนิคยอดนิยมในการปรับโมเดลให้เหมาะกับข้อมูลใหม่โดยไม่ฝึกทั้งหมด?",
      options: [
        "Feature Slicing",
        "Transfer Learning",
        "Data Pruning"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "Grad-CAM ถูกใช้เพื่ออะไรใน CNN?",
      options: [
        "การเพิ่มจำนวน filter ในแต่ละเลเยอร์",
        "การตรวจสอบว่าโมเดลสนใจส่วนใดของภาพในการตัดสินใจ",
        "การลดขนาด input image"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "ข้อใดไม่ใช่แนวทางการแก้ปัญหา Overfitting ใน CNN?",
      options: [
        "Dropout",
        "Data Augmentation",
        "เพิ่มจำนวนชั้น Convolution โดยไม่มี regularization"
      ],
      correct: 2
    },
    {
      id: "q9",
      question: "ใน EfficientNet การสเกลโมเดลทำผ่านแนวคิดใด?",
      options: [
        "Random Scaling",
        "Compound Scaling",
        "Feature Scaling"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "ข้อใดคือ insight จากการใช้ Visualization กับ CNN?",
      options: [
        "ช่วยเลือก activation function ที่เหมาะสม",
        "ช่วยสร้าง model ให้แม่นยำขึ้นแบบอัตโนมัติ",
        "ช่วยให้เข้าใจว่า filter แต่ละตัวเรียนรู้อะไรจากภาพ"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: CNN Architecture & Filters
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

export default MiniQuiz_Day22;
