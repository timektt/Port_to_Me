import React, { useState } from "react";
import { FaCheckCircle, FaTimesCircle, FaQuestionCircle } from "react-icons/fa";

const MiniQuiz_Day31 = ({ theme }) => {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [incorrect, setIncorrect] = useState([]);

  const questions = [
    {
      id: "q1",
      question: "Self-Attention มีข้อได้เปรียบหลักอะไรเมื่อเทียบกับ RNN?",
      options: [
        "ประมวลผลแบบลำดับได้เร็วขึ้น",
        "สามารถคำนวณแบบขนานและเข้าใจความสัมพันธ์ระยะไกลได้ดีกว่า",
        "ต้องใช้หน่วยความจำน้อยกว่า"
      ],
      correct: 1
    },
    {
      id: "q2",
      question: "เหตุผลที่ต้องมี Positional Encoding คืออะไร?",
      options: [
        "เพื่อเพิ่มความเร็วในการฝึก",
        "เพื่อให้โมเดลสามารถเข้าใจลำดับของข้อมูล",
        "เพื่อให้ Encoder กับ Decoder สามารถ share weights ได้"
      ],
      correct: 1
    },
    {
      id: "q3",
      question: "อะไรคือหน้าที่หลักของ Multi-Head Attention?",
      options: [
        "ลดค่า loss",
        "เพิ่มจำนวน parameters",
        "จับความสัมพันธ์จากหลายมุมมองพร้อมกัน"
      ],
      correct: 2
    },
    {
      id: "q4",
      question: "Feed Forward Network (FFN) ใน Transformer ทำหน้าที่ใด?",
      options: [
        "แปลงค่าจาก Positional Encoding",
        "ประมวลผลลำดับแบบลูป",
        "เพิ่ม non-linearity และแปลงความหมายของ token"
      ],
      correct: 2
    },
    {
      id: "q5",
      question: "Layer Normalization มีบทบาทอย่างไรใน Transformer?",
      options: [
        "ลดจำนวน layer",
        "เพิ่ม gradient stability และความเสถียรในการฝึก",
        "ใช้แทน Residual Connection"
      ],
      correct: 1
    },
    {
      id: "q6",
      question: "เหตุใด Decoder จึงต้องใช้ Masked Self-Attention?",
      options: [
        "เพื่อเพิ่ม batch size",
        "เพื่อไม่ให้เห็นคำในอนาคต",
        "เพื่อเชื่อมต่อกับ Encoder"
      ],
      correct: 1
    },
    {
      id: "q7",
      question: "Self-Attention มี time complexity เป็นเท่าไร?",
      options: [
        "O(n)",
        "O(log n)",
        "O(n²)"
      ],
      correct: 2
    },
    {
      id: "q8",
      question: "วิธีไหนที่ใช้แก้ข้อจำกัดของ Transformer ในลำดับยาว?",
      options: [
        "ใช้ FFN ขนาดเล็กลง",
        "ใช้ Efficient Transformer เช่น Longformer, Performer",
        "ใช้ Dropout แทน Attention"
      ],
      correct: 1
    },
    {
      id: "q9",
      question: "ความแตกต่างระหว่าง Encoder กับ Decoder คืออะไร?",
      options: [
        "Decoder ไม่มี Positional Encoding",
        "Decoder มี Masked Self-Attention",
        "Encoder ไม่มี Multi-Head Attention"
      ],
      correct: 1
    },
    {
      id: "q10",
      question: "Insight หลักของ Transformer คืออะไร?",
      options: [
        "การคำนวณความสัมพันธ์แบบลำดับ",
        "การเรียนรู้ความสัมพันธ์แบบทั้งหมดพร้อมกัน",
        "การประมวลผลด้วย LSTM แบบใหม่"
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
        <FaQuestionCircle className="text-yellow-500" /> Mini Quiz: Transformer Architecture
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

export default MiniQuiz_Day31;
