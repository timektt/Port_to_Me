import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day25 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดคือจุดประสงค์ของการใช้ Regularization?",
      options: [
        "เพิ่มความซับซ้อนของโมเดล",
        "ลด Overfitting",
        "เพิ่มจำนวนพารามิเตอร์"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "L1 Regularization ช่วยอะไรในการเรียนรู้ของโมเดล?",
      options: [
        "เพิ่มขนาดของพารามิเตอร์",
        "ทำให้บางพารามิเตอร์กลายเป็นศูนย์",
        "ทำให้ค่าพารามิเตอร์ทั้งหมดเท่ากัน"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "Dropout คืออะไรในระบบ CNN?",
      options: [
        "การลบข้อมูลจากชุดฝึก",
        "การลดจำนวนเลเยอร์",
        "การสุ่มปิดบาง Neuron ชั่วคราวขณะฝึก"
      ],
      correct: 2
    },
    {
      id: "q4",
      question: "การใช้ Dropout มีผลอย่างไรต่อการฝึกโมเดล?",
      options: [
        "ทำให้การเรียนรู้เร็วขึ้น",
        "ลดการพึ่งพา Neuron เฉพาะตัว",
        "เพิ่มความแม่นยำของข้อมูลเทรน"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "Data Augmentation ช่วยป้องกัน Overfitting ได้อย่างไร?",
      options: [
        "ทำให้โมเดลเล็กลง",
        "เพิ่มตัวอย่างข้อมูลเทียมให้โมเดลเรียนรู้ได้หลากหลายขึ้น",
        "ลบข้อมูลที่ซ้ำออกไป"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "Early Stopping จะหยุดการฝึกเมื่อใด?",
      options: [
        "เมื่อ Loss ต่ำที่สุดใน Train Set",
        "เมื่อ Validation Loss ไม่ลดลงต่อเนื่องหลายรอบ",
        "เมื่อความแม่นยำถึง 100%"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "Batch Normalization ช่วยให้การฝึกโมเดลเป็นอย่างไร?",
      options: [
        "ไม่ต้องใช้ Activation Function",
        "ลดความจำเป็นในการใช้ Learning Rate",
        "การกระจายของข้อมูลในแต่ละเลเยอร์มีเสถียรภาพมากขึ้น"
      ],
      correct: 2
    },
    {
      id: "q8",
      question: "Regularization มีผลต่อ Bias-Variance อย่างไร?",
      options: [
        "เพิ่ม Bias ลด Variance",
        "ลด Bias เพิ่ม Variance",
        "เพิ่มทั้ง Bias และ Variance"
      ],
      correct: 0
    },
    {
      id: "q9",
      question: "เทคนิคใดใช้ในการปรับค่า weight โดยเฉพาะ?",
      options: [
        "Dropout",
        "L2 Regularization",
        "Data Augmentation"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "ข้อใดต่อไปนี้เป็นผลของการ Overfitting?",
      options: [
        "โมเดลให้ผลดีทั้ง Train และ Validation",
        "โมเดลจำข้อมูลฝึกได้แม่นยำแต่ล้มเหลวกับข้อมูลใหม่",
        "โมเดลไม่สามารถเรียนรู้ข้อมูลใดๆ ได้เลย"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Regularization in CNNs
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

export default MiniQuiz_Day25;
