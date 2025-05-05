import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day23 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดกล่าวถึง Max Pooling ได้ถูกต้องที่สุด?",
      options: [
        "เลือกค่าเฉลี่ยจากทุกตำแหน่งใน filter",
        "เลือกค่าต่ำสุดจากทุกตำแหน่งใน filter",
        "เลือกค่าสูงสุดจากทุกตำแหน่งใน filter"
      ],
      correct: 2
    },
    {
      id: "q2",
      question: "Stride ที่มีค่ามากขึ้นมีผลอย่างไรต่อ output?",
      options: [
        "เพิ่มขนาดของ output feature map",
        "ลดขนาดของ output feature map",
        "ไม่ส่งผลต่อขนาดของ output"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "Average Pooling ใช้เพื่อวัตถุประสงค์ใด?",
      options: [
        "เก็บเฉพาะ feature ที่โดดเด่นที่สุด",
        "ลด spatial dimension โดยรักษาความเฉลี่ยของข้อมูล",
        "ทำให้โมเดลเรียนรู้ความสัมพันธ์ที่ไม่เชิงเส้น"
      ],
      correct: 1
    },
    {
      id: "q4",
      question: "Global Average Pooling มักใช้ในขั้นตอนใดของ CNN?",
      options: [
        "ก่อน convolution layer แรก",
        "หลัง fully connected layer",
        "ก่อน fully connected layer เพื่อแทน flatten"
      ],
      correct: 2
    },
    {
      id: "q5",
      question: "การใช้ stride มากเกินไปอาจนำไปสู่ผลกระทบใด?",
      options: [
        "ทำให้โมเดลเรียนรู้ได้เร็วขึ้นโดยไม่มีข้อเสีย",
        "ทำให้สูญเสียข้อมูลสำคัญจาก spatial structure",
        "เพิ่มจำนวนพารามิเตอร์ให้กับโมเดล"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "การใช้ Pooling Layer ช่วยลดสิ่งใดในโมเดล?",
      options: [
        "จำนวนพารามิเตอร์และการคำนวณ",
        "จำนวน class ในการจำแนก",
        "ขนาดของ batch ที่ใช้ในการฝึก"
      ],
      correct: 0
    },
    {
      id: "q7",
      question: "ข้อใดคือข้อดีของการใช้ Stride แทน Pooling?",
      options: [
        "ไม่ลดขนาดของ output map",
        "ลดขนาด spatial โดยไม่มี layer เพิ่มเติม",
        "ช่วยเพิ่มจำนวน feature maps"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "ข้อใดไม่ใช่ประเภทของ Pooling ที่ใช้จริง?",
      options: [
        "Max Pooling",
        "Min Pooling",
        "Random Pooling"
      ],
      correct: 2
    },
    {
      id: "q9",
      question: "ทำไมบางโมเดลจึงหลีกเลี่ยงการใช้ Pooling?",
      options: [
        "เพื่อหลีกเลี่ยงการสูญเสียข้อมูลสำคัญ",
        "เพราะ Pooling เพิ่มความซับซ้อนของโมเดล",
        "เพราะ Pooling ใช้หน่วยความจำมากเกินไป"
      ],
      correct: 0
    },
    {
      id: "q10",
      question: "การใช้ stride ใน convolution layer แทน pooling มักพบในสถาปัตยกรรมใด?",
      options: [
        "VGGNet",
        "ResNet",
        "All Convolutional Net"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Pooling & Stride Techniques
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

export default MiniQuiz_Day23;
