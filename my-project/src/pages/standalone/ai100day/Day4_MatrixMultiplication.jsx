import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day4 from "./scrollspy/ScrollSpy_Ai_Day4";
import MiniQuiz_Day4 from "./miniquiz/MiniQuiz_Day4";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";
const Day4_MatrixMultiplication = ({theme}) => {
     const navigate = useNavigate();
     return (
      <div
        className={`relative min-h-screen ${
          theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
        }`}
      >
        {/* ✅ AiSidebar (เฉพาะ Desktop) */}
        <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
          <AiSidebar theme={theme} />
        </div>
      <main className="max-w-3xl mx-auto p-6 pt-20">
      <h1 className="text-3xl font-bold mb-6">
        Day 4: Matrix Multiplication — พลังของ "การเปลี่ยนมุมมอง"
      </h1>

      <p className="mb-6 text-lg">
        ถ้าเวกเตอร์คือ <strong>"ข้อมูล"</strong> — เมทริกซ์คือ <strong>"เลนส์"</strong> ที่ AI ใช้มองข้อมูลนั้นๆ
      </p>

      <section id="what-is-matrix" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Matrix Multiplication คืออะไร?</h2>

  <img
    src="/MatrixMultiplication.png"
    alt="Matrix Multiplication"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    Matrix Multiplication คือการนำ <strong>เมทริกซ์</strong> มาคูณกับ <strong>เวกเตอร์</strong> หรือเมทริกซ์อีกตัวหนึ่ง เพื่อเปลี่ยนหรือแปลงข้อมูลในรูปแบบที่เข้าใจง่ายขึ้น
    เป็นกระบวนการพื้นฐานที่สุดที่อยู่เบื้องหลังโมเดล AI เกือบทุกประเภทเลย
  </p>

  <p className="mb-2">
    เมทริกซ์สามารถมองเป็น “ชุดของเลเยอร์” ที่ใช้แปลงข้อมูล เช่น ข้อมูลภาพ, เสียง, หรือข้อความ
    เมื่อเวกเตอร์ input ผ่านเข้าไป → เมทริกซ์จะเปลี่ยนข้อมูลนั้นให้กลายเป็น “feature ใหม่” หรือข้อมูลที่เน้นบางมิติที่สำคัญมากขึ้น
  </p>

  <p className="mb-2">
    การคูณเมทริกซ์จึงไม่ใช่แค่การคำนวณเลข แต่เป็นการ “เปลี่ยนมุมมองของข้อมูล”
    เหมือนใส่เลนส์ใหม่ให้กับ input เพื่อให้ระบบเข้าใจสิ่งนั้นได้ลึกยิ่งขึ้น
  </p>

  <p className="mb-2">
    ใน Neural Network → น้ำหนักของแต่ละเลเยอร์ คือเมทริกซ์
    input ที่เข้ามาคือเวกเตอร์ข้อมูล เช่น รูปภาพ, คำ, หรือเสียง
    พอคูณกับน้ำหนัก → ได้เวกเตอร์ใหม่ที่มีข้อมูลแฝง (feature) ที่สำคัญมากขึ้น
  </p>

  <p className="mb-2">
    ยกตัวอย่างเช่น ถ้า x คือ input และ W คือเมทริกซ์ของเลเยอร์
    ผลลัพธ์ที่ได้คือ: <code>output = W @ x</code>
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np

W = np.array([[1, 0], [0, 1]])
x = np.array([5, 10])

result = np.dot(W, x)
print(result)  # Output: [5, 10]`}</pre>

  <p className="mb-2">
    ในตัวอย่างข้างบน → W เป็นเมทริกซ์เอกลักษณ์ (identity matrix) ซึ่งหมายความว่าไม่ได้เปลี่ยนข้อมูลเลย
    ถ้าเปลี่ยนค่าในเมทริกซ์ → ข้อมูล output จะเปลี่ยนตามทันที
  </p>

  <p className="mb-2">
    Matrix Multiplication ถูกใช้ในการประมวลผลรูปภาพ โดย input คือภาพที่ถูกแปลงเป็นเวกเตอร์
    แล้วคูณกับเมทริกซ์ที่เรียนรู้มา → ผลลัพธ์คือ “feature map” ที่ทำให้โมเดลเข้าใจว่าส่วนไหนคือขอบ, พื้นผิว, หรือวัตถุ
  </p>

  <p className="mb-2">
    ใน NLP → vector ของคำ (เช่น “love”) จะถูกคูณกับเมทริกซ์ weight
    เพื่อให้ได้ embedding ใหม่ที่สะท้อนความหมายเชิงบริบท
  </p>

  <p className="mb-2">
    ทุกครั้งที่โมเดลเรียนรู้ → จริง ๆ แล้วคือการ “ปรับค่าในเมทริกซ์”
    ทำให้เมื่อ input เข้ามาในครั้งต่อไป → ข้อมูลที่ถูกแปลงจะมีคุณภาพดีขึ้น
  </p>

  <p className="mb-2">
    เมทริกซ์แต่ละอันใน Neural Network คือเลเยอร์หนึ่ง ๆ ที่รับผิดชอบการเปลี่ยนแปลงเฉพาะทาง
    เช่น เปลี่ยนเสียงพูด → เป็น feature frequency
    หรือแปลงภาพ → เป็น pattern ของวัตถุ
  </p>

  <p className="mb-2">
    จุดเด่นของ Matrix Multiplication คือรองรับข้อมูลแบบหลายมิติได้ดี
    เช่น แถวของภาพที่เป็น RGB หรือ embedding vector ของคำที่มี 768 มิติ
  </p>

  <p className="mb-2">
    เมื่อ matrix A คูณกับ matrix B → ต้องแน่ใจว่าขนาดตรงกันในมิติภายใน
    เช่น A เป็น 2x3 ต้องคูณกับ B ที่เป็น 3xN เท่านั้น → จะได้ผลลัพธ์เป็น 2xN
  </p>

  <p className="mb-2">
    การคูณเมทริกซ์ใน AI มักใช้ร่วมกับฟังก์ชัน activation
    เช่น ReLU หรือ Sigmoid → เพื่อเพิ่ม non-linearity ให้กับการแปลงข้อมูล
  </p>

  <p className="mb-2">
    โดยรวมแล้ว Matrix Multiplication คือ “หัวใจ” ของกระบวนการเรียนรู้
    เพราะมันเป็นกลไกที่ทำให้โมเดลสามารถเปลี่ยน input → ให้เป็น feature ที่มีความหมาย
  </p>

  <p className="mb-2">
    ทุกครั้งที่ค่าความแม่นยำของโมเดลดีขึ้น
    หมายถึงค่าภายในเมทริกซ์เหล่านี้กำลังเข้าใกล้การแปลงที่ถูกต้อง
  </p>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong>  
    เมทริกซ์ใน Neural Network คือเลเยอร์สมองของ AI ที่รับข้อมูลดิบ แล้วคูณเปลี่ยนเป็นข้อมูลที่เข้าใจง่ายขึ้น
    ทุกการเรียนรู้ ก็คือการปรับเมทริกซ์ให้มองเห็นความจริงได้ดีขึ้นเรื่อย ๆ
  </div>
</section>



<section id="matrix-dimension" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">
    เข้าใจขนาดเมทริกซ์ (Matrix Shape & Dimension)
  </h2>

  <img
    src="/MatrixShape&Dimension.png"
    alt="MatrixShape & Dimension"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    การคูณเมทริกซ์ไม่สามารถทำได้ทุกคู่แบบอิสระ ต้องมีเงื่อนไขที่สำคัญที่สุดคือ:{" "}
    <strong className="text-red-500">
      จำนวนคอลัมน์ของเมทริกซ์แรก ต้องเท่ากับจำนวนแถวของเมทริกซ์ที่สอง
    </strong>
  </p>

  <p className="mb-4">
    ถ้า A มีขนาด{" "}
    <code className="bg-gray-100 dark:bg-gray-800 px-2 rounded">3 × 4</code>{" "}
    หมายถึง 3 แถว 4 คอลัมน์ และ B มีขนาด{" "}
    <code className="bg-gray-100 dark:bg-gray-800 px-2 rounded">4 × 2</code> → สามารถคูณได้
    เพราะ <strong>4 (คอลัมน์ของ A)</strong> = <strong>4 (แถวของ B)</strong>
  </p>

  <div className="overflow-x-auto mb-6">
    <table className="min-w-full text-sm text-left border border-gray-300 dark:border-gray-700">
      <thead>
        <tr className="bg-gray-100 dark:bg-gray-800">
          <th className="border px-4 py-2">เมทริกซ์</th>
          <th className="border px-4 py-2">ขนาด</th>
          <th className="border px-4 py-2">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody className="bg-white dark:bg-gray-900">
        <tr>
          <td className="border px-4 py-2">A</td>
          <td className="border px-4 py-2">3 × 4</td>
          <td className="border px-4 py-2">3 แถว 4 คอลัมน์</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">B</td>
          <td className="border px-4 py-2">4 × 2</td>
          <td className="border px-4 py-2">4 แถว 2 คอลัมน์</td>
        </tr>
        <tr className="bg-yellow-50 dark:bg-yellow-900 font-semibold text-black dark:text-yellow-200">
          <td className="border px-4 py-2" colSpan="2">
            ผลลัพธ์
          </td>
          <td className="border px-4 py-2">ขนาด = 3 × 2</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p className="mb-4">
    ถ้าไม่เข้าใจเรื่องขนาด อาจทำให้เกิดข้อผิดพลาดบ่อย เช่น บางครั้งใช้ matrix คูณกับเวกเตอร์แล้วเกิด error
    เพราะขนาดไม่ตรงกัน ต้องตรวจสอบเสมอว่าจำนวน "มิติภายใน" (inner dimensions) ตรงกันหรือไม่
  </p>

  <p className="mb-4">
    ลองดูอีกตัวอย่างหนึ่ง:
    <code className="block bg-gray-800 text-white p-3 text-sm rounded mt-4 mb-4 overflow-x-auto">{`# A: 2 × 3
# B: 3 × 1
# คูณได้: 2 × 1

import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])   # 2×3
B = np.array([[1], [0], [1]])         # 3×1

result = np.dot(A, B)
print(result)
# Output:
# [[4]
#  [10]]`}</code>
  </p>

  <p className="mb-4">
    ขนาดผลลัพธ์บอกได้ทันทีว่า output จะมีลักษณะกว้างเท่าไหร่ เช่น คูณกับ vector ที่มี 1 คอลัมน์ → จะได้ผลลัพธ์เป็น vector เช่นกัน  
    ส่วนการคูณกับเมทริกซ์หลายคอลัมน์จะให้ output เป็นเมทริกซ์ใหม่ที่มีหลาย “มุมมอง”
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-200 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong>
    <br />
    การเข้าใจรูปแบบ (shape) ของเมทริกซ์อย่างถ่องแท้ คือพื้นฐานของการเขียนโปรแกรม AI ที่แม่นยำ  
    เพราะทุกขั้นตอนของ Neural Network ล้วนผ่านการคูณเมทริกซ์ และข้อผิดพลาดมักเกิดจากขนาดไม่ตรงกัน
  </div>
</section>

<section id="matrix-order" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">A @ B ≠ B @ A เสมอไป</h2>

  <img
    src="/A&B.png"
    alt="A&B"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />


  <p className="mb-4">
    การคูณเมทริกซ์ <strong>ไม่สามารถสลับลำดับได้</strong> เหมือนการคูณเลขทั่วไป  
    หรือพูดอีกแบบว่า <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">A @ B ≠ B @ A</code> ในเกือบทุกกรณี
  </p>

  <p className="mb-4">
    ไม่ใช่แค่ผลลัพธ์ที่ต่างกัน แต่บางครั้ง <strong>B @ A อาจคูณไม่ได้เลย</strong> ถ้าขนาดไม่รองรับ  
    นี่คือสิ่งสำคัญที่ต้องระวังในการออกแบบ Neural Network
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ตัวอย่างที่ 1: ขนาดเท่ากัน แต่ค่าต่างกัน</h3>

  <pre className="bg-gray-800 text-white p-3 rounded text-sm overflow-x-auto mb-4">{`import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A @ B =\\n", A @ B)
print("B @ A =\\n", B @ A)`}</pre>

  <p className="mb-2 font-medium">ผลลัพธ์:</p>

  <pre className="bg-gray-900 text-white p-3 rounded text-sm overflow-x-auto mb-4 border border-yellow-500 shadow-lg">
{`A @ B =
[[19 22]
 [43 50]]

B @ A =
[[23 34]
 [31 46]]`}
</pre>

  <p className="mb-4">
    ถึงแม้ขนาดเมทริกซ์จะเท่ากัน แต่ลำดับมีผลกับค่าภายในโดยตรง เพราะรูปแบบการคูณเปลี่ยน
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ตัวอย่างที่ 2: คูณได้ทั้งสอง แต่ขนาดต่างกัน</h3>

  <pre className="bg-gray-800 text-white p-3 rounded text-sm overflow-x-auto mb-4">{`A = np.array([[1, 2, 3]])
B = np.array([[4], [5], [6]])

# A: 1×3, B: 3×1 → A @ B = scalar (1x1)
# B @ A = 3×3 matrix

print("A @ B =", A @ B)
print("B @ A =\\n", B @ A)`}</pre>

  <p className="mb-4">
    คำสั่งทั้งสองคูณได้ แต่ผลลัพธ์ต่างกันมาก  
    <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">A @ B</code> → สเกลาร์,  
    <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">B @ A</code> → เมทริกซ์ 3×3
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ผลกระทบต่อ AI และโมเดล</h3>

  <p className="mb-4">
    ลำดับเมทริกซ์มีผลต่อการเรียนรู้ เช่น  
    หากต้องการเปลี่ยนเวกเตอร์จาก 300 มิติ → 128 มิติ  
    ต้องใช้เมทริกซ์ขนาด <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">128 × 300</code>  
    หากสลับ → คูณไม่ได้หรือแปลงผิด
  </p>

  <p className="mb-4">
    การฝึกโมเดล (training) เช่นในขั้นตอน Backpropagation  
    ก็ต้องพึ่งพาลำดับที่ถูกต้อง → หากสลับแม้เล็กน้อย การเรียนรู้จะล้มเหลว
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-200 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การคูณเมทริกซ์ต้อง “ระวังลำดับ” เสมอ  
    ลำดับที่ผิดอาจให้ผลลัพธ์ผิด → ส่งผลต่อการเข้าใจข้อมูลในโมเดลทั้งระบบ
  </div>
</section>

<section id="matrix-transpose" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Matrix Transpose และการใช้ใน Attention</h2>

  <img
    src="/MatrixTransposeanditsuseinAttention.png"
    alt="Matrix Transposeand its use in Attention"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    การ <strong>Transpose</strong> เมทริกซ์ คือการสลับแถวกับคอลัมน์  
    หรือพูดง่าย ๆ คือ เปลี่ยนจาก A<sub>ij</sub> → A<sub>ji</sub>
  </p>

  <p className="mb-4">
    ใน Python / NumPy → ใช้ <code>.T</code> เพื่อ transpose ได้โดยตรง เช่น: <code>A.T</code>
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
print("A Transpose:\\n", A.T)          # 2x3 matrix`}</pre>

  <p className="mb-4">
    Transpose ใช้ในหลายสถานการณ์ใน AI เช่น การหมุนมิติของข้อมูล, การ reshape ระหว่าง layer  
    แต่ที่สำคัญที่สุดคือใช้ใน <strong>Attention Mechanism</strong> โดยเฉพาะในโมเดล Transformer
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">Self-Attention และ Transpose</h3>

  <p className="mb-2">
    ในกลไก Self-Attention → มี 3 เวกเตอร์หลัก: <strong>Query (Q)</strong>, <strong>Key (K)</strong>, <strong>Value (V)</strong>
  </p>

  <p className="mb-2">
    เพื่อหาว่า “แต่ละคำควรสนใจคำอื่นแค่ไหน” → จะคูณ Q กับ K<sup>T</sup> (transpose ของ Key)  
    เพื่อให้ Q ของทุก token เปรียบเทียบกับ K ของทุก token ได้ในครั้งเดียว
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`# Q: (seq_len, dim)
# K: (seq_len, dim)
# K.T: (dim, seq_len)
# Q @ K.T → (seq_len, seq_len)

import numpy as np

Q = np.random.rand(4, 3)  # 4 คำ, dim = 3
K = np.random.rand(4, 3)

scores = Q @ K.T
print(scores.shape)  # (4, 4)`}</pre>

  <p className="mb-2">
    ผลลัพธ์ที่ได้คือเมทริกซ์ขนาด <code>seq_len × seq_len</code> ที่เก็บ “ความคล้าย” ของแต่ละคำกับคำอื่นทั้งหมด  
    ซึ่งจะนำไป softmax แล้วใช้คูณกับ V → กลายเป็น output ของ attention
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">สรุปการใช้งาน Transpose ใน Attention</h3>
  <ul className="list-disc pl-6 space-y-2 text-sm sm:text-base">
    <li>เปลี่ยน Key จาก (N, D) → (D, N) เพื่อให้คูณกับ Q ได้</li>
    <li>ช่วยให้คำนวณ dot product แบบคู่ทุกคำกับทุกคำในลำดับ</li>
    <li>เพิ่มประสิทธิภาพ: คูณทีเดียวได้เมทริกซ์ score ทั้งหมด</li>
    <li>ใช้ในโมเดล BERT, GPT, ViT, T5 และ Transformer ทุกประเภท</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Transpose ไม่ใช่แค่การสลับมิติ แต่เป็นกลไกสำคัญที่ทำให้ Attention เข้าใจ “ความสัมพันธ์” แบบทั่วถึง  
    เป็นจุดเชื่อมต่อระหว่างเวกเตอร์หลายตัวเพื่อให้เกิดการสนใจแบบอัจฉริยะในโมเดลภาษา
  </div>
</section>

<section id="softmax-scaling" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Softmax & Scaling ใน Attention Score</h2>

  <img
    src="/Softmax&Scaling.png"
    alt="Softmax&Scaling"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    หลังจากคำนวณ <strong>Q @ K<sup>T</sup></strong> ในกลไก Self-Attention → จะได้เมทริกซ์ที่เก็บค่า “ความคล้าย” ระหว่างคำกับคำทั้งหมดในลำดับ  
    ค่านี้เรียกว่า <strong>Score</strong> หรือ <strong>Attention Logits</strong>
  </p>

  <p className="mb-4">
    แต่ค่าเหล่านี้ยังไม่เหมาะกับการใช้คูณกับเวกเตอร์ต่อทันที เพราะบางค่ามีค่าสูงมาก อาจทำให้เกิดปัญหา gradient ระเบิดได้  
    จึงต้องใช้กระบวนการ **Scaling + Softmax**
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">1. Scaling: หารด้วย √(dim)</h3>
  <p className="mb-2">
    หากเวกเตอร์มีมิติเยอะ เช่น 64 หรือ 128 → การ dot product จะให้ค่าใหญ่  
    เพื่อให้ค่าพอดี → ต้อง <strong>หารด้วย √(dim)</strong> เช่น:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`# Assume dim = 64
scaled_scores = (Q @ K.T) / np.sqrt(64)`}</pre>

  <p className="mb-2">
    Scaling นี้ช่วยลดความรุนแรงของค่าคะแนน ช่วยให้ softmax ทำงานได้ดียิ่งขึ้น
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">2. Softmax: เปลี่ยนคะแนนเป็นความน่าจะเป็น</h3>
  <p className="mb-2">
    เมื่อได้เมทริกซ์ Score ที่คูณและ Scaling แล้ว → จะใช้ <strong>Softmax</strong> เปลี่ยนค่าคะแนนในแต่ละแถว  
    ให้กลายเป็นค่าระหว่าง 0 ถึง 1 ที่รวมกันเท่ากับ 1
  </p>

  <p className="mb-2">
    นี่คือขั้นตอนที่เปลี่ยนจาก “ความคล้ายแบบดิบ” → เป็น “น้ำหนักการให้ความสนใจ” เช่น:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

scores = np.array([1.0, 2.0, 3.0])
probs = softmax(scores)
print(probs)  # Output: [0.09, 0.24, 0.66]`}</pre>

  <p className="mb-2">
    ค่าคะแนน 3 ตัวข้างต้น → หลัง softmax จะบอกว่า “ให้ความสนใจกับตำแหน่งสุดท้ายมากที่สุด” (ค่ามากสุด)
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ภาพรวมกลไก Attention</h3>
  <p className="mb-2">
    1. คำนวณ <code>Q @ K<sup>T</sup></code> → ได้ Score  
    2. หารด้วย √(dim) → ได้ Scaled Score  
    3. Softmax → ได้ Attention Weights  
    4. คูณกับ V → ได้ Output ของ Attention
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`attention = softmax((Q @ K.T) / sqrt(d_k)) @ V`}</pre>

  <p className="mb-2">
    การ Scaling และ Softmax คือหัวใจที่ทำให้โมเดลเข้าใจว่า “จะสนใจจุดไหนมากที่สุด” โดยไม่ต้องเข้าใจภาษามนุษย์เลย  
    มันคือเครื่องมือที่เปลี่ยนค่าความคล้าย → ให้กลายเป็น “ระดับความสำคัญ”
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Softmax คือประตูสุดท้ายของ Attention ที่แปลง Score → ให้กลายเป็นความน่าจะเป็น  
    Scaling ช่วยให้การเรียนรู้มีเสถียรภาพมากขึ้นในเวกเตอร์ขนาดใหญ่  
    ทั้งสองอย่างนี้ทำให้ Attention ทรงพลังพอจะเปลี่ยนโลก NLP และ Vision ได้จริง
  </div>
</section>

<section id="multi-head-attention" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Multi-Head Attention คืออะไร?</h2>

  <img
    src="/Multi-Head Attention.png"
    alt="Multi-Head Attention"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    ใน Attention ธรรมดา จะมีการสร้างเวกเตอร์ Q, K, V เพียงชุดเดียว → คำนวณ score → softmax → output  
    แต่การทำแบบนี้จะมองความสัมพันธ์ในมิติเดียวเท่านั้น เช่น เน้นแค่คำใกล้ ๆ หรือคำที่เด่นสุดเพียงคำเดียว
  </p>

  <p className="mb-4">
    <strong>Multi-Head Attention</strong> คือการ “คูณ” ชุดของ Q, K, V ออกมาหลายชุด (เรียกว่า “หัว” หรือ heads)  
    เพื่อให้แต่ละหัว “มองบริบทคนละแบบ” เช่น:
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-1">
    <li>หัวที่ 1 → มองความสัมพันธ์ใกล้ ๆ (local attention)</li>
    <li>หัวที่ 2 → มองคำที่สัมพันธ์เชิงไวยากรณ์</li>
    <li>หัวที่ 3 → เน้นคำที่มีอารมณ์ หรือโฟกัสซ้ำคำสำคัญ</li>
    <li>หัวอื่น ๆ → มองจากมุม semantic, topic, หรือลำดับ</li>
  </ul>

  <p className="mb-4">
    จากนั้นแต่ละหัวจะได้ output attention คนละชุด → นำมารวม (concatenate) แล้วแปลงอีกครั้ง  
    โมเดลจะได้มุมมองรวมที่หลากหลาย แต่รวมกันกลมกลืนเป็นหนึ่งเดียว
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ภาพรวมกระบวนการ:</h3>
  <ol className="list-decimal pl-6 mb-4 space-y-1">
    <li>รับ input vector → คูณกับเมทริกซ์ W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> สำหรับทุกหัว</li>
    <li>คำนวณ Attention แยกแต่ละหัว → ใช้ Q@K<sup>T</sup>/√d → softmax → @V</li>
    <li>Concat ทุกหัวเข้าด้วยกัน → คูณกับ W<sub>output</sub> เพื่อรวมความหมายทั้งหมด</li>
  </ol>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`MultiHead(Q, K, V) = Concat(head1, head2, ..., headN) @ W_output

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`}</pre>

  <p className="mb-4">
    จุดเด่นคือ แต่ละหัวใช้ weight matrix ต่างกัน  
    ทำให้สามารถ “เรียนรู้” มุมมองหลากหลายที่จำเป็นสำหรับการเข้าใจภาษาหรือภาพ
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Multi-Head Attention ทำให้โมเดลไม่จำเป็นต้องเลือกมุมมองเดียว → แต่มองหลายมุมพร้อมกัน แล้วสรุปเป็นภาพรวม  
    นี่คือสิ่งที่ทำให้โมเดลอย่าง GPT หรือ BERT เข้าใจประโยคได้ลึก และสามารถให้คำตอบที่ชาญฉลาดได้จริง
  </div>
</section>
<section id="linear-transformation-visual" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">ภาพแสดงการ Stretch ด้วยเมทริกซ์</h2>
  <img
    src="/Stretch.png"
    alt="Stretch"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />
  <p className="mb-4">
    การคูณเวกเตอร์กับเมทริกซ์สามารถมองเป็นการ "ดึง", "หมุน", หรือ "เปลี่ยนมุมมอง" ของเวกเตอร์ในพื้นที่ 2D ได้  
    ภาพด้านล่างแสดงผลของการ stretch บนแกน X → เวกเตอร์แดงและน้ำเงินถูกแปลงเป็นเวกเตอร์ใหม่ (ส้มและฟ้า)
  </p>

</section>
<section id="linear-transform" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">
    Linear Transformation คืออะไร? ทำไม Matrix คือรากฐานของมัน
  </h2>

  <p className="mb-4">
    การคูณเมทริกซ์ไม่ใช่แค่การเปลี่ยนตัวเลข แต่มันคือการ <strong>"เปลี่ยนมุมมอง"</strong> ของเวกเตอร์  
    หรือที่ทางคณิตศาสตร์เรียกว่า <strong>Linear Transformation</strong>
  </p>

  <p className="mb-4">
    Linear Transformation คือกระบวนการที่เปลี่ยนตำแหน่งของเวกเตอร์ในพื้นที่ โดยยังคงความสัมพันธ์เชิงเส้นเอาไว้  
    เช่น การหมุน, การยืดในทิศทางใดทิศทางหนึ่ง, หรือการสะท้อน
  </p>

  <p className="mb-4">
    ถ้าเปรียบเวกเตอร์เป็น "ลูกศร" ในพิกัด 2D หรือ 3D → Matrix จะทำหน้าที่เหมือน "ฟิลเตอร์" ที่เปลี่ยนทิศ, ขนาด หรือทิศทางของลูกศรเหล่านั้น  
    แต่ยังรักษาแก่นความสัมพันธ์ของมันไว้ เช่น ถ้าสองเวกเตอร์อยู่ในเส้นตรงเดียวกัน → หลังคูณเมทริกซ์ ก็ยังอยู่ในเส้นเดียวกัน
  </p>

  <p className="mb-4">
    ใน AI → การใช้เมทริกซ์แปลงข้อมูลจึงเทียบเท่ากับการ "ฉาย" หรือ "ปรับมุมมอง" ข้อมูลให้เข้าสู่พื้นที่ใหม่ที่เหมาะกับการวิเคราะห์  
    เช่น จากภาพ → เป็น feature, จากเวกเตอร์ของคำ → เป็น embedding
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ตัวอย่างเชิงภาพ: ยืดและหมุน</h3>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np
import matplotlib.pyplot as plt

# เวกเตอร์ตั้งต้น
vectors = np.array([[1, 0], [0, 1]])
origin = np.zeros((2, 2))

# Matrix ที่ใช้ "ยืด" ในแนวแกน X
stretch_x = np.array([[2, 0], [0, 1]])
transformed = vectors @ stretch_x.T

plt.quiver(*origin, vectors[:,0], vectors[:,1], color=["r", "b"], scale=1, scale_units="xy")
plt.quiver(*origin, transformed[:,0], transformed[:,1], color=["orange", "cyan"], scale=1, scale_units="xy")
plt.xlim(-1, 3)
plt.ylim(-1, 2)
plt.gca().set_aspect("equal")
plt.title("Linear Transform: Stretch on X-axis")
plt.grid()
plt.show()`}</pre>

  <p className="mb-4">
    ในกราฟนี้:  
    🔴🔵 คือเวกเตอร์ก่อนเปลี่ยน, 🟠🔷 คือเวกเตอร์หลังการแปลง → จะเห็นว่าถูกยืดออกด้านขวา  
    ถ้าเปลี่ยนเมทริกซ์ให้มีค่าลบ หรือมีค่าหมุน → ก็จะได้การสะท้อน หรือหมุนเวกเตอร์แทน
  </p>

  <p className="mb-4">
    Linear Transformation คือหลักการเบื้องหลังของ Convolution, Embedding Layer, Self-Attention และแม้กระทั่ง GAN  
    เพราะมันคือการเปลี่ยน "ข้อมูลดิบ" → ให้กลายเป็นข้อมูลที่มีรูปแบบง่ายขึ้นต่อการเข้าใจ
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Matrix ไม่ใช่แค่ตารางตัวเลข แต่คือ "กลไกการแปลง" ที่ทรงพลังในโลกของ AI  
    ทุกครั้งที่เมทริกซ์คูณกับข้อมูล → กำลังเกิด Linear Transformation ที่ช่วยให้โมเดลเข้าใจข้อมูลลึกขึ้นแบบที่มนุษย์มองไม่เห็น
  </div>
</section>

<section id="broadcasting" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">
    Broadcasting คืออะไร? ทำไม NumPy ใช้คูณเวกเตอร์ได้เลย
  </h2>

  <p className="mb-4">
    ใน Python/NumPy เราสามารถคูณเมทริกซ์กับเวกเตอร์ได้โดยไม่ต้อง reshape ทุกครั้ง  
    เพราะมีกลไกที่เรียกว่า <strong>Broadcasting</strong>
  </p>

  <p className="mb-4">
    Broadcasting คือการ "ปรับขนาดโดยอัตโนมัติ" เพื่อให้สามารถคูณหรือบวกเวกเตอร์กับเมทริกซ์ได้  
    เช่น คูณ <code>(3×3)</code> กับ <code>(3,)</code> → NumPy จะมองว่าเวกเตอร์นั้นเป็น <code>(3×1)</code> หรือ <code>(1×3)</code> ตามบริบท
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ตัวอย่าง: คูณเมทริกซ์กับเวกเตอร์</h3>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np

M = np.array([[1, 2], [3, 4], [5, 6]])   # 3×2
v = np.array([1, 2])                    # 2, จะ broadcast เป็น (2×1) โดยอัตโนมัติ

result = M @ v
print(result)  # Output: [ 5 11 17]`}</pre>

  <p className="mb-4">
    แม้ขนาดจะไม่ตรงพอดี แต่ NumPy จะ broadcast ให้พอดีตามกฎ ซึ่งสะดวกมากในการใช้งาน Deep Learning จริง
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Broadcasting คือเคล็ดลับที่ทำให้เราไม่ต้อง reshape ข้อมูลเองทุกครั้ง  
    ช่วยให้โค้ดการประมวลผลเวกเตอร์หรือเมทริกซ์สั้น กระชับ และปลอดภัยขึ้น
  </div>
</section>

<section id="batch-matrix" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Matrix Multiplication แบบ Batch</h2>

  <p className="mb-4">
    ในการประมวลผลข้อมูลหลายชิ้นพร้อมกัน (เช่น ข้อความหลายประโยค, รูปภาพหลายภาพ)  
    เราจะใช้เมทริกซ์ที่มีขนาด 3 มิติ เช่น <code>(Batch, N, D)</code>
  </p>

  <p className="mb-4">
    การคูณแบบ Batch คือการคูณเมทริกซ์แบบหลายชุดพร้อมกัน เช่น  
    <code>(Batch, N, D) @ (Batch, D, D2)</code> → ผลลัพธ์คือ <code>(Batch, N, D2)</code>
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ตัวอย่างใน PyTorch:</h3>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import torch

# ข้อมูล 2 ชุด (batch=2), แต่ละอันมี 4 token (seq_len) และ dim=3
Q = torch.randn(2, 4, 3)
K = torch.randn(2, 3, 3)

# คูณแบบ batch
score = torch.bmm(Q, K)  # ผลลัพธ์: (2, 4, 3)
print(score.shape)`}</pre>

  <p className="mb-4">
    ฟังก์ชัน <code>bmm</code> ย่อมาจาก <strong>Batch Matrix Multiply</strong>  
    ใช้กับข้อมูล 3 มิติแบบ batch ได้โดยตรง ไม่ต้อง loop
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    โมเดลภาษาและภาพทุกตัว เช่น BERT, GPT, CNN ต่างทำงานกับข้อมูลเป็น “batch” เสมอ  
    การเข้าใจการคูณแบบ batch ช่วยให้เข้าใจ dimension ของ tensor ในโลกจริงได้แม่นยำ
  </div>
</section>


<section id="try-interactive" className="mb-16 scroll-mt-32">
        <h2 className="text-2xl font-semibold mb-3">ทดลองคูณเมทริกซ์ </h2>
        <p className="mb-4">
          ลองใส่เมทริกซ์และเวกเตอร์ด้านล่างเพื่อดูผลลัพธ์ที่ได้ทันที
        </p>
        <p className="mb-4">
          เมทริกซ์เปรียบเสมือนเลนส์ที่ปรับการมองข้อมูลในระบบประสาทของ AI เมื่อนำมาคูณกับเวกเตอร์ที่แทนข้อมูลดิบ เช่น ภาพหรือคำ เมทริกซ์จะปรับค่าหรือเน้นจุดสำคัญ ทำให้โมเดลสามารถแยกแยะสิ่งต่าง ๆ ได้ชัดเจนขึ้น
        </p>
        <p className="mb-4">
          การทดลองนี้ช่วยให้เห็นการเปลี่ยนแปลงของเวกเตอร์หลังถูกคูณด้วยเมทริกซ์แบบเรียลไทม์ โดยสามารถเปลี่ยนค่าต่าง ๆ และสังเกตผลลัพธ์ได้ทันที เช่น เมทริกซ์แบบ identity จะให้ผลเหมือนเดิม เมทริกซ์ที่มีค่าสูงในบางแถวจะเน้นเฉพาะข้อมูลส่วนนั้นออกมา
        </p>
        <p className="mb-4">
          ตัวอย่างเช่น หากเวกเตอร์แทนลักษณะของภาพหนึ่งภาพ แล้วเมทริกซ์ถูกฝึกให้รู้ว่าลักษณะไหนคือ "แมว" การคูณเมทริกซ์นั้นกับเวกเตอร์ของภาพ จะได้ผลลัพธ์ที่แสดงความเป็นแมวของภาพนั้นในเชิงตัวเลข
        </p>
        <p className="mb-4">
          ในกรณีของ NLP เมทริกซ์คูณเวกเตอร์ของคำ ทำให้สามารถดึงความสัมพันธ์ระหว่างคำและประโยค เช่น คำว่า "bank" ในประโยค "go to the river bank" กับ "go to the bank to deposit money" จะให้บริบทต่างกัน หลังจากคูณกับเมทริกซ์ attention แล้ว
        </p>
        <p className="mb-4">
          ระบบแนะนำ (Recommendation) ใช้เวกเตอร์ที่แทนผู้ใช้งานหรือสินค้า มาคูณกับเมทริกซ์ที่ฝึกมาเพื่อค้นหาความสอดคล้อง ผลลัพธ์ที่ได้คือค่าที่บ่งบอกระดับความน่าสนใจ เช่น ผู้ใช้นี้ชอบหนังแนวไซไฟ เมทริกซ์จะชูคุณลักษณะของหนังแนวนั้นขึ้นมา
        </p>
        <p className="mb-4">
          เปรียบเทียบกับชีวิตจริง เมทริกซ์เหมือนเลนส์ของแว่นสายตา การมองภาพเดียวกันผ่านเลนส์ต่างกัน จะทำให้เห็นรายละเอียดที่ต่างออกไป โมเดล AI ก็ใช้เมทริกซ์ในลักษณะเดียวกัน เพื่อให้มองข้อมูลได้ชัดเจนยิ่งขึ้นในมุมที่ต้องการเรียนรู้
        </p>
        <p className="mb-4">
          ฟีเจอร์นี้มีไว้เพื่อให้ลองปรับค่าและวิเคราะห์ผลลัพธ์โดยไม่ต้องเขียนโค้ดเอง เพิ่มความเข้าใจภาพรวมของ matrix multiplication อย่างเป็นธรรมชาติ
        </p>
      </section>

      <section id="real-world-examples" className="mb-16 scroll-mt-32">
        <h2 className="text-2xl font-semibold mb-3">ตัวอย่างในโลกจริง</h2>
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>Image Classification:</strong> รูปภาพ × เมทริกซ์ = label เช่น "แมว" หรือ "หมา"</li>
          <li><strong>NLP:</strong> คำ × attention matrix → ความเข้าใจความหมายในประโยค</li>
          <li><strong>Recommendation:</strong> พฤติกรรมผู้ใช้ × latent matrix = ความน่าจะชอบของหนัง</li>
          <li><strong>Object Detection:</strong> กล่อง bounding box × transformation matrix = พิกัดใหม่ของวัตถุ</li>
          <li><strong>Style Transfer:</strong> ภาพ content × style matrix = ภาพศิลปะที่มีเนื้อหาต้นฉบับ</li>
          <li><strong>Face Recognition:</strong> รูปใบหน้า × เมทริกซ์ feature = embedding vector ที่ใช้เปรียบเทียบ</li>
          <li><strong>Graph Neural Networks:</strong> node vector × adjacency matrix = การแพร่กระจายข้อมูลในกราฟ</li>
          <li><strong>Voice Classification:</strong> เวกเตอร์เสียง × เมทริกซ์น้ำหนัก = การวิเคราะห์โทนเสียงหรือคำพูด</li>
        </ul>
        <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
          <strong>Insight:</strong> ในทุกระบบ AI ที่มีการตัดสินใจ เมทริกซ์คือกลไกหลักที่เปลี่ยนข้อมูลให้กลายเป็นมุมมองใหม่ เพื่อสร้างความเข้าใจระดับลึก แม้ในกรณีที่ไม่มีการเขียนโปรแกรมโดยตรง ก็ยังมีเมทริกซ์อยู่เบื้องหลังเสมอครับ
        </div>
      </section>


      <section id="read-more" className="mb-20">
        <h3 className="text-lg font-semibold mb-3"> แหล่งเรียนรู้ต่อยอด</h3>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li><a className="text-blue-500 hover:underline" href="https://jalammar.github.io/illustrated-word2vec/" target="_blank">Word2Vec Visualization</a></li>
          <li><a className="text-blue-500 hover:underline" href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need (paper)</a></li>
          <li><a className="text-blue-500 hover:underline" href="https://huggingface.co/" target="_blank">HuggingFace Models</a></li>
        </ul>
      </section>
      
      <section id="quiz" className="mb-16 scroll-mt-32 ">
          <MiniQuiz_Day4 theme={theme} />
        </section>

        <div className="flex justify-between items-center max-w-5xl mx-auto px-4 mt-4">
          <div className="flex items-center">
            <span className="text-lg font-bold">Tags:</span>
            <button
              onClick={() => navigate("/tags/ai")}
              className="ml-2 px-3 py-1 border border-gray-500 rounded-lg text-green-700 cursor-pointer hover:bg-gray-700 transition"
            >
              ai
            </button>
          </div>
        </div>
            
        <Comments theme={theme} />
        </main>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day4 />
      </div>

      <SupportMeButton />
    </div>
    
  );
};

export default Day4_MatrixMultiplication;
