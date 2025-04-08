import React, { useState } from "react";

const TryDotProduct = () => {
  const [v1, setV1] = useState([2, 3]);
  const [v2, setV2] = useState([4, 1]);

  const parse = (input) => {
    return input
      .split(",")
      .map((n) => parseFloat(n.trim()))
      .filter((n) => !isNaN(n));
  };

  const handleChangeV1 = (e) => setV1(parse(e.target.value));
  const handleChangeV2 = (e) => setV2(parse(e.target.value));

  const dotProduct = v1.length === v2.length
    ? v1.reduce((sum, val, i) => sum + val * v2[i], 0)
    : "❌ มิติต้องเท่ากัน";

  const cosineSimilarity =
    v1.length === v2.length
      ? (() => {
          const dot = v1.reduce((sum, val, i) => sum + val * v2[i], 0);
          const norm1 = Math.sqrt(v1.reduce((sum, val) => sum + val * val, 0));
          const norm2 = Math.sqrt(v2.reduce((sum, val) => sum + val * val, 0));
          return (dot / (norm1 * norm2)).toFixed(4);
        })()
      : "❌";

  return (
    <section id="try-it-live" className="mt-10 mb-16">
      <h2 className="text-xl font-semibold mb-3"> ลองเล่นดูได้ครับ </h2>
      <p className="mb-2">ใส่เวกเตอร์แล้วดูค่าผลลัพธ์ dot product และ cosine similarity ทันที</p>
      <div className="grid gap-3 mb-4">
        <input
          type="text"
          defaultValue="2, 3"
          onChange={handleChangeV1}
          placeholder="เวกเตอร์ 1 (เช่น 2, 3)"
          className="p-2 border rounded-md bg-white text-black"
        />
        <input
          type="text"
          defaultValue="4, 1"
          onChange={handleChangeV2}
          placeholder="เวกเตอร์ 2 (เช่น 4, 1)"
          className="p-2 border rounded-md bg-white text-black"
        />
      </div>
      <div className="bg-gray-800 text-white p-4 rounded-xl text-sm shadow mb-4">
        <div>🧮 Dot Product: <strong>{dotProduct}</strong></div>
        <div>📐 Cosine Similarity: <strong>{cosineSimilarity}</strong></div>
      </div>
      <p className="text-green-500 font-semibold">✅ Tip: ลองเปลี่ยนมุม เช่น [1, 0] กับ [0, 1] → ควรได้ dot = 0, cosine = 0</p>
    </section>
  );
};

export default TryDotProduct;
