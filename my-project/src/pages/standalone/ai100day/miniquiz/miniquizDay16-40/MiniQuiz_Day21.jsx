import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day21 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "ข้อใดคือปัญหาของการใช้ MLP กับข้อมูลภาพ?",
      options: [
        "MLP ใช้ Filter ขนาดเล็ก",
        "MLP สูญเสียโครงสร้างเชิงพื้นที่ของภาพเมื่อ flatten",
        "MLP เรียนรู้ Feature เชิงลึกได้ดีกว่า CNN"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "หน้าที่หลักของ Convolution Layer คืออะไร?",
      options: [
        "แปลงภาพให้เป็น grayscale",
        "ลดขนาดภาพลงแบบอัตโนมัติ",
        "ดึง Feature ท้องถิ่นจากภาพผ่านการเลื่อน Filter"
      ],
      correct: 2
    },
    {
      id: "q3",
      question: "ReLU activation function ทำหน้าที่อะไรใน CNN?",
      options: [
        "ตัดค่าที่ต่ำกว่า 0 ออกเพื่อเพิ่ม non-linearity",
        "คูณค่าทุก pixel ด้วย -1",
        "เพิ่มขนาด input ด้วย zero padding"
      ],
      correct: 0
    },
    {
      id: "q4",
      question: "ข้อใดคือข้อดีของ Max Pooling?",
      options: [
        "เพิ่ม resolution ของ Feature Map",
        "ลดจำนวนพารามิเตอร์ลงและเน้น Feature เด่น",
        "ทำให้ภาพชัดขึ้นในระหว่างการเรียนรู้"
      ],
      correct: 1
    },
    {
      id: "q5",
      question: "CNN ใช้การ Weight Sharing เพื่ออะไร?",
      options: [
        "ลดพารามิเตอร์และเพิ่มความสามารถในการเรียนรู้",
        "ขยายภาพโดยใช้ kernel",
        "เพิ่มจำนวน class label"
      ],
      correct: 0
    },
    {
      id: "q6",
      question: "ในเลเยอร์สูงของ CNN โมเดลมักเรียนรู้อะไร?",
      options: [
        "Edge และ texture",
        "ลักษณะเชิง semantic เช่น ใบหน้า หรือ object",
        "Noise และ gradient map"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "Global Average Pooling มักถูกใช้แทนอะไรใน CNN รุ่นใหม่?",
      options: [
        "Input Layer",
        "Fully Connected Layer",
        "Convolution Layer"
      ],
      correct: 1
    },
    {
      id: "q8",
      question: "ข้อใดคือข้อจำกัดของ CNN ที่ถูกพูดถึงมากที่สุด?",
      options: [
        "ไม่สามารถทำงานร่วมกับข้อมูลภาพได้",
        "ไม่สามารถเรียนรู้ global context ได้โดยตรง",
        "เรียนรู้ได้ช้าเมื่อใช้ GPU"
      ],
      correct: 1
    },
    {
      id: "q9",
      question: "วิธีใดที่นิยมใช้เพื่อแก้ปัญหา spatial transformation ใน CNN?",
      options: [
        "การใช้ kernel ขนาดใหญ่",
        "การใช้ Spatial Transformer Networks",
        "การใช้ batch normalization ซ้ำหลายชั้น"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "CoAtNet และ ConvNeXt มีจุดเด่นอย่างไร?",
      options: [
        "ผสมข้อดีของ CNN และ Transformer",
        "ลดความแม่นยำเพื่อเพิ่มความเร็ว",
        "ไม่ใช้ activation function ใด ๆ เลย"
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
    <section id="quiz" className="bg-black/60 text-white rounded-xl p-6 mt-12 border border-yellow-500 shadow-lg">
      <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Convolutional Neural Networks
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

export default MiniQuiz_Day21;
