import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import MiniQuiz_Day3 from "./miniquiz/MiniQuiz_Day3";
import ScrollSpy_Ai_Day3 from "./scrollspy/ScrollSpy_Ai_Day3";
import TryDotProduct from "../../courses/topics/interactive/TryDotProduct";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";




const Day3_DotProduct = ({ theme }) => {
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
          Day 3: Dot Product & ความหมายเชิงมุม (Cosine Similarity)
        </h1>

        <p className="mb-6 text-lg">
           “การคูณเวกเตอร์” หรือ <strong>Dot Product</strong> ทำงานอย่างไร ? และทำไมมันถึงเป็นหัวใจของ AI แทบทุกระบบ ทั้ง NLP, Search, และ Recommendation
        </p>

<section id="dot-product" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Dot Product คืออะไร?</h2>
  <img
    src="/DotProduct.png"
    alt="DotProduct"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />


  <p className="mb-4">
    Dot Product หรือผลิตภัณฑ์จุด คือกระบวนการคูณเวกเตอร์สองตัวเข้าด้วยกัน  
    โดยผลลัพธ์จะออกมาเป็น “ค่าจำนวนเดียว” (scalar) ไม่ใช่เวกเตอร์ใหม่  
    และค่านี้มีความหมายมากในการบอกว่า เวกเตอร์สองตัวนั้น “สัมพันธ์กันแค่ไหน”
  </p>

  <p className="mb-2">
    การคำนวณ dot product นั้นง่ายมาก — แค่คูณค่าของแต่ละมิติเข้าด้วยกัน แล้วรวมผลทั้งหมด เช่น:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`v1 = [2, 3]
v2 = [4, 1]

Dot = 2*4 + 3*1 = 8 + 3 = 11`}</pre>

  <p className="mb-2">
    จากตัวอย่างข้างต้น <code>v1</code> และ <code>v2</code> มี dot product เท่ากับ 11  
    ซึ่งหมายความว่าเวกเตอร์ทั้งสอง “ชี้ไปในทิศทางที่ใกล้เคียงกัน” พอสมควร
  </p>

  <p className="mb-2">
    จุดเด่นของ dot product คือ มันสามารถใช้วัด “ความสัมพันธ์เชิงมุม” ได้  
    - ถ้ามุมระหว่างเวกเตอร์น้อย (มุมแหลม) → ค่าจะมาก  
    - ถ้ามุม 90 องศา → dot product = 0  
    - ถ้ามุมมากกว่า 90 องศา หรือชี้ตรงข้าม → ค่าจะติดลบ
  </p>

  <p className="mb-2">
    ดังนั้น เราไม่ได้แค่คูณเลขเพื่อหาค่าเล่น ๆ แต่เรากำลังใช้ dot product เพื่อ “วัดทิศทางของความหมาย”
  </p>

  <p className="mb-2">
    ตัวอย่างในชีวิตจริง:
    - ถ้าเรามีเวกเตอร์แทน "รสนิยมของผู้ใช้ A" กับ "เพลง X" → การใช้ dot product จะบอกว่า เพลงนี้ตรงกับความชอบของผู้ใช้แค่ไหนครับ
    - ใน NLP → dot product ของเวกเตอร์คำ เช่น “happy” กับ “joyful” จะให้ค่าสูง เพราะมีความหมายคล้ายกันมาก
    - ถ้าเอา “happy” กับ “sad” → dot product จะติดลบ เพราะเป็นความหมายตรงข้ามกันเลย
  </p>

  <p className="mb-2">
    ในทางคณิตศาสตร์ ถ้า v1 และ v2 เป็นเวกเตอร์ 2 ตัว เราคำนวณ dot product ได้ด้วยสูตร:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`v1 • v2 = |v1| * |v2| * cos(θ)

(θ คือมุมระหว่างเวกเตอร์ทั้งสอง)`}</pre>

  <p className="mb-2">
    สูตรนี้แสดงให้เห็นว่า dot product ผูกอยู่กับ <strong>cosine ของมุม</strong>  
    นั่นแปลว่า เราสามารถใช้ dot product เพื่อนำไปคำนวณ cosine similarity ได้ด้วยครับ
  </p>

  <p className="mb-2">
    Dot product ยังสามารถตีความเป็น “พลังงานที่ทิศทางตรงกัน” ได้ เช่น  
    - เวกเตอร์ A แทนแรง  
    - เวกเตอร์ B แทนการเคลื่อนที่  
    → ถ้า A กับ B ไปทางเดียวกัน → dot product จะสูง แปลว่า “มีงานเกิดขึ้น”
  </p>

  <p className="mb-2">
    ในด้าน AI → dot product ถูกใช้อย่างแพร่หลายมาก เช่น:
    - ใช้ใน Self-Attention ของ Transformer
    - ใช้เปรียบเทียบ embedding vector ของคำใน Word2Vec
    - ใช้ใน Recommendation System เพื่อตรวจสอบความคล้ายกันของ user vs item
  </p>

  <p className="mb-2">
    และที่น่าสนใจคือ dot product ทำงานได้ดีมากใน space ที่มีมิติมาก ๆ เช่น 300D, 768D หรือแม้แต่ 1024D  
    โมเดล AI ใช้ dot product เพื่อสรุปว่า embedding หนึ่ง ๆ “คล้ายกับอีก embedding แค่ไหน” แบบแม่นยำ
  </p>

  <p className="mb-2">
    นอกจากนี้ dot product ยังสามารถใช้เป็น “เครื่องมือเลือกความสำคัญ” เช่นใน Attention Mechanism  
    โมเดลจะใช้ dot product เปรียบเทียบระหว่าง Query และ Key เพื่อดูว่า "จะให้ความสนใจกับข้อมูลไหน"
  </p>

  <p className="mb-2">
    ตัวอย่างการใช้งานใน Python:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np

v1 = np.array([2, 3])
v2 = np.array([4, 1])

dot = np.dot(v1, v2)
print("Dot Product:", dot)  # Output: 11`}</pre>

  <p className="mb-2">
    หรือจะใช้ v1 @ v2 ก็ได้เหมือนกัน เพราะใน NumPy เครื่องหมาย <code>@</code> ใช้แทน dot product โดยตรง
  </p>

  <p className="mb-2">
    สิ่งที่ควรระวังคือ dot product จะมีความหมายก็ต่อเมื่อทั้งสองเวกเตอร์มีขนาดเท่ากัน  
    ถ้าขนาดไม่เท่ากันจะคำนวณไม่ได้
  </p>

  <p className="mb-2">
    อีกจุดที่สำคัญคือ dot product จะรับรู้ทั้ง “ทิศทาง” และ “ความยาว”  
    ดังนั้นถ้าเราอยากวัดเฉพาะ “ทิศ” เราจะต้อง normalize เวกเตอร์ก่อน แล้วค่อยหาค่า cosine similarity
  </p>

  <p className="mb-2">
    ในทางกราฟิก → dot product ยังใช้ตรวจสอบว่า วัตถุชี้เข้าหากล้องไหม เช่นใน 3D Game  
    ถ้าผลลัพธ์ &gt; 0 = หันเข้าหากล้อง, ถ้า &lt; 0 = หันออก
  </p>

  <p className="mb-2">
    ในฟิสิกส์ → dot product ใช้หางาน (work) = แรง • การกระจัด  
    ถ้าทิศเดียวกันจะได้ค่าบวก ถ้าทิศตรงข้ามจะได้ค่าลบ (งานลบ)
  </p>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    ถ้า dot product สูง = เวกเตอร์สองตัว "ไปในทางเดียวกัน" มาก → โมเดล AI ใช้ตรงนี้เพื่อตัดสินว่า "ข้อมูลนี้เหมือนหรือไม่เหมือน"  
    ยิ่งคล้าย → ค่ายิ่งสูง  
    → สิ่งนี้คือแก่นของ Search, ChatGPT, และ Recommendation System
  </div>
</section>



<section id="cosine-similarity" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Cosine Similarity คืออะไร?</h2>

  <img
    src="/CosineSimilarity.png"
    alt="Cosine Similarity"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    หลังจากเราเข้าใจ Dot Product แล้ว — เราจะต่อยอดไปสู่ <strong>Cosine Similarity</strong>  
    ซึ่งเป็นเครื่องมือที่ AI ใช้บ่อยมากในการวัดว่า “เวกเตอร์สองตัวนี้คล้ายกันแค่ไหน” โดย <u>ไม่สนขนาด</u> แต่ดูเฉพาะ “มุม” เท่านั้น
  </p>

  <p className="mb-2">
    โดยทั่วไป เวกเตอร์แต่ละตัวอาจมีความยาวต่างกัน เช่น คำว่า “king” กับ “queen” อาจมีเวกเตอร์ยาวไม่เท่ากัน  
    ถ้าเราใช้ dot product อย่างเดียว อาจทำให้ผลลัพธ์เบี่ยงเบนได้
  </p>

  <p className="mb-2">
    ดังนั้น Cosine Similarity จะ <strong>normalize</strong> ทั้งสองเวกเตอร์ก่อน แล้วค่อยคำนวณ dot product  
    ผลลัพธ์จะอยู่ในช่วง <code>-1</code> ถึง <code>1</code> ซึ่งสะท้อน “มุม” ระหว่างเวกเตอร์
  </p>

  <p className="mb-2">
    สูตรคำนวณ Cosine Similarity คือ:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`cos(θ) = (v1 • v2) / (||v1|| * ||v2||)`}</pre>

  <p className="mb-2">
    โดยที่:
    <br />• <code>v1 • v2</code> คือ dot product  
    <br />• <code>||v1||</code> และ <code>||v2||</code> คือความยาว (norm) ของเวกเตอร์แต่ละตัว
  </p>

  <p className="mb-2">
    ความหมายของผลลัพธ์:
    <ul className="list-disc pl-6 mt-2 space-y-1">
      <li><code>1</code> → เวกเตอร์ชี้ไปทางเดียวกัน (คล้ายกันมาก)</li>
      <li><code>0</code> → มุมฉากกัน (ไม่มีความเกี่ยวข้อง)</li>
      <li><code>-1</code> → ชี้ตรงข้ามกัน (ความหมายตรงข้าม)</li>
    </ul>
  </p>

  <p className="mb-2">
    Cosine Similarity มีประโยชน์ตรงที่ว่า เราไม่ต้องสนใจว่าเวกเตอร์จะยาวแค่ไหน  
    แต่ดูแค่ว่า “มุมของมันต่างกันแค่ไหน” ซึ่งทำให้เหมาะมากกับการเปรียบเทียบความหมาย
  </p>

  <p className="mb-2">
    ตัวอย่างเช่น:
    - คำว่า <code>run</code> กับ <code>jog</code> อาจมีเวกเตอร์ยาวไม่เท่ากัน  
    - แต่ชี้ไปในทิศเดียวกัน → cosine similarity จะสูง (≈ 0.9)
  </p>

  <p className="mb-2">
    ในทางกลับกัน <code>run</code> กับ <code>sleep</code> → อาจชี้คนละทิศ → cosine ≈ 0 หรือติดลบ
  </p>

  <p className="mb-2">
    ลองดูตัวอย่างการคำนวณด้วย Python:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`import numpy as np

v1 = np.array([2, 3])
v2 = np.array([4, 1])

dot = np.dot(v1, v2)
cos_sim = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
print("Cosine Similarity:", cos_sim)`}
  </pre>

  <p className="mb-2">
    ในโค้ดข้างต้น เราคูณ dot product แล้วหารด้วยผลคูณของ norm ทั้งสอง  
    จะได้ค่า cosine similarity ประมาณ <strong>0.82</strong> → แสดงว่า v1 กับ v2 ชี้ไปทางเดียวกันพอสมควร
  </p>

  <p className="mb-2">
    Cosine Similarity ถูกใช้เป็นพื้นฐานของ:
    <ul className="list-disc pl-6 mt-2 space-y-1">
      <li>การหาความคล้ายของคำ (word similarity)</li>
      <li>การแนะนำสินค้า (recommendation system)</li>
      <li>การหาข้อความที่คล้ายกันในเอกสาร (semantic search)</li>
      <li>การเปรียบเทียบ embedding vectors ของภาพ/เสียง</li>
    </ul>
  </p>

  <p className="mb-2">
    ถ้าเรากำลังสร้างโมเดลที่ต้อง “จับคู่ข้อมูล” เช่น user vs content หรือ query vs document  
    Cosine Similarity คือหนึ่งในเครื่องมือที่เร็วและแม่นยำที่สุด
  </p>

  <p className="mb-2">
    ตัวอย่างง่าย ๆ ที่เข้าใจง่าย:
    <br />– ถ้าผู้ใช้ A มีเวกเตอร์ <code>[1, 1]</code>  
    <br />– เพลง X มีเวกเตอร์ <code>[2, 2]</code>  
    → แม้จะยาวกว่ากัน แต่ชี้ไปทางเดียวกัน → cosine similarity = 1
  </p>

  <p className="mb-2">
    แต่ถ้าเพลง B มีเวกเตอร์ <code>[-1, -1]</code> → ชี้ตรงข้าม → cosine = -1  
    → แสดงว่า user นี้ “ไม่น่าชอบ” เพลง B
  </p>

  <p className="mb-2">
    ดังนั้น การใช้ cosine similarity ช่วยตัด noise จาก “ความแรงของเวกเตอร์”  
    แล้วเน้นที่ “ทิศทางของความหมาย” แทน
  </p>

  <p className="mb-2">
    AI รุ่นใหม่ ๆ เช่น BERT, GPT, CLIP ก็ล้วนแต่ใช้ cosine similarity ในการตัดสินว่า embedding ตัวไหน “คล้ายกัน”
  </p>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    <span className="block mt-1">
      ถ้า dot product วัดพลังของความสัมพันธ์ → cosine similarity วัด “ทิศทางของความหมาย”  
      เป็นตัวบอกว่า “สิ่งนี้ใกล้กับอีกสิ่งหนึ่งแค่ไหน?” โดยไม่สนใจปริมาณ
    </span>
    <span className="block mt-2 text-green-600">
      ✅ เหมาะกับงานที่ต้องการความแม่นยำในการวัดความคล้าย เช่น ChatGPT, Search, Translation, และ Music Recommendation
    </span>
  </div>
</section>

<h3 className="text-lg font-semibold mt-8 mb-3">เปรียบเทียบ Dot Product กับ Cosine Similarity</h3>

<div className="overflow-x-auto mb-6 rounded-lg border border-gray-300 dark:border-gray-700">
  <table className="min-w-full table-auto text-sm text-left">
    <thead>
      <tr className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-100">
        <th className="border px-4 py-2 dark:border-gray-700">หัวข้อ</th>
        <th className="border px-4 py-2 dark:border-gray-700">Dot Product</th>
        <th className="border px-4 py-2 dark:border-gray-700">Cosine Similarity</th>
      </tr>
    </thead>
    <tbody className="bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-200">
      <tr>
        <td className="border px-4 py-2 dark:border-gray-700">ผลลัพธ์</td>
        <td className="border px-4 py-2 dark:border-gray-700">จำนวนจริง (อิงขนาด)</td>
        <td className="border px-4 py-2 dark:border-gray-700">ค่าระหว่าง -1 ถึง 1</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800/40">
        <td className="border px-4 py-2 dark:border-gray-700">สนใจทิศหรือขนาด?</td>
        <td className="border px-4 py-2 dark:border-gray-700">ทิศ + ขนาด</td>
        <td className="border px-4 py-2 dark:border-gray-700">เฉพาะทิศทาง (normalized)</td>
      </tr>
      <tr>
        <td className="border px-4 py-2 dark:border-gray-700">อ่อนไหวต่อ scale?</td>
        <td className="border px-4 py-2 dark:border-gray-700">ใช่</td>
        <td className="border px-4 py-2 dark:border-gray-700">ไม่</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800/40">
        <td className="border px-4 py-2 dark:border-gray-700">ใช้งานเหมาะกับ</td>
        <td className="border px-4 py-2 dark:border-gray-700">การวัด “พลัง” หรือผลรวม</td>
        <td className="border px-4 py-2 dark:border-gray-700">การวัด “ความคล้ายเชิงความหมาย”</td>
      </tr>
      <tr>
        <td className="border px-4 py-2 dark:border-gray-700">ตัวอย่างที่ใช้บ่อย</td>
        <td className="border px-4 py-2 dark:border-gray-700">Self-Attention, Recommendation</td>
        <td className="border px-4 py-2 dark:border-gray-700">Word Similarity, Semantic Search</td>
      </tr>
    </tbody>
  </table>
</div>



<section id="dot-limitations" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">ข้อจำกัดของ Dot Product</h2>

  <p className="mb-4">
    แม้ว่า Dot Product จะทรงพลังและใช้แพร่หลายใน AI แต่ก็มีข้อจำกัดบางอย่างที่เราควรเข้าใจ  
    เพราะมันอาจทำให้ผลลัพธ์เบี่ยงเบน หากเราใช้โดยไม่รู้บริบท
  </p>

  <h3 className="text-lg font-semibold mb-2">1. ไม่สามารถวัด "มุม" ได้โดยตรง</h3>
  <p className="mb-2">
    Dot Product ให้ค่าจำนวนเดียว (scalar) ที่แสดง “ระดับความสัมพันธ์” ระหว่างเวกเตอร์  
    แต่ <strong>มันไม่ได้ให้ค่ามุม (angle)</strong> ระหว่างเวกเตอร์โดยตรง  
    ถ้าเราต้องการรู้มุมจริง ๆ → เราต้องนำค่าที่ได้ไป normalize ก่อนแล้วค่อยใช้ <code>arccos()</code>
  </p>

  <p className="mb-2">
    สูตรที่แท้จริงคือ:  
    <code>cos(θ) = (v1 • v2) / (||v1|| * ||v2||)</code>  
    จากนั้นใช้ <code>arccos</code> เพื่อได้มุม <code>θ</code>
  </p>

  <p className="mb-2">
    ถ้าเราคำนวณ dot product อย่างเดียวโดยไม่ normalize → เราจะได้ค่าใหญ่หรือเล็กตามความยาวของเวกเตอร์  
    ไม่ใช่เพราะมุมเปลี่ยน แต่เพราะ “ขนาด” เปลี่ยน
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">2. Dot Product แพ้ "ขนาดของเวกเตอร์"</h3>
  <p className="mb-2">
    อีกข้อจำกัดใหญ่คือ dot product <strong>ไวต่อขนาดของเวกเตอร์</strong>  
    ถ้าเวกเตอร์อันหนึ่งยาวกว่ามาก → ค่าผลลัพธ์จะเพิ่มขึ้น แม้ว่าทิศทางจะเหมือนเดิม
  </p>

  <p className="mb-2">
    ตัวอย่างเช่น:
    <br />– <code>v1 = [1, 1]</code>, <code>v2 = [2, 2]</code> → dot = 4  
    <br />– <code>v1 = [1, 1]</code>, <code>v2 = [100, 100]</code> → dot = 200  
    แม้ทิศทางจะเหมือนกัน → แต่ค่าที่ได้ต่างกันมาก เพราะเวกเตอร์ยาวต่างกัน
  </p>

  <p className="mb-2">
    ดังนั้น ถ้าเราจะใช้ dot product เพื่อวัด "ความคล้าย"  
    <strong>เราต้อง normalize เวกเตอร์ก่อน</strong> เพื่อไม่ให้ความยาวมีผลกับการตัดสิน
  </p>

  <p className="mb-2">
    ถ้าไม่ normalize → เวกเตอร์ที่ยาวจะ dominate การตัดสินของโมเดล  
    ทำให้บาง embedding มีน้ำหนักมากเกินไปโดยไม่เกี่ยวกับความหมาย
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">3. Sensitive ต่อสเกลข้อมูล (Scale-sensitive)</h3>
  <p className="mb-2">
    สมมุติว่าเรามี embedding 2 ชุดที่คล้ายกัน แต่ถูกขยายต่างกัน (เช่น *2 หรือ *10)  
    Dot product จะให้ผลลัพธ์ต่างกันอย่างมาก  
    ซึ่งในบางกรณีอาจนำไปสู่ความเข้าใจผิดของโมเดล
  </p>

  <p className="mb-2">
    ตัวอย่างในงาน NLP หรือ Search Engine:
    - embedding ของคำ “chat” กับ “talk” อาจใกล้เคียงกัน  
    - แต่ถ้า embedding ถูก scale ต่างกัน → dot product จะเพี้ยน  
    → Cosine similarity แก้ปัญหานี้ได้
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">4. ใช้กับข้อมูลที่มี Noise ไม่ดีนัก</h3>
  <p className="mb-2">
    ถ้าข้อมูลของเรามี noise หรือ outlier เยอะ ๆ → dot product จะไวต่อค่าที่ผิดปกติมาก  
    เช่น เวกเตอร์ที่มีค่า spike 1 ช่อง อาจลากค่า dot product ให้สูงผิดปกติ
  </p>

  <p className="mb-2">
    ตัวอย่าง:
    - <code>v1 = [1, 1, 1, 100]</code> กับ <code>v2 = [1, 1, 1, 0]</code>  
    → dot = 3 (ปกติ) กับ <code>v1</code> ใหม่จะได้ dot = 103 → ค่าถูกเบี่ยงเบนจาก outlier
  </p>

  <p className="mb-2">
    แนวทางคือใช้ cosine similarity แทน หรือทำ data preprocessing ก่อนคำนวณ
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">5. ไม่สามารถใช้กับเวกเตอร์ต่างมิติได้</h3>
  <p className="mb-2">
    Dot Product ต้องใช้กับเวกเตอร์ที่มีมิติเหมือนกันเท่านั้น  
    ถ้า <code>v1</code> มี 3 มิติ แต่ <code>v2</code> มี 4 มิติ → คำนวณไม่ได้เลย
  </p>

  <p className="mb-2">
    ดังนั้นในงานจริง เราต้องแน่ใจว่า embedding, feature vector หรือ data structure มี shape ที่ถูกต้อง  
    ซึ่งเป็นอีกข้อจำกัดสำคัญ
  </p>

  <p className="mb-2">
    แม้ dot product จะเร็วและคำนวณง่ายมาก แต่ถ้าไม่ระวังเรื่อง dimension หรือ scale → อาจทำให้ผลลัพธ์ผิดได้
  </p>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    แม้ Dot Product จะทรงพลัง แต่มันมีจุดอ่อนหลายจุด  
    โดยเฉพาะเมื่อใช้กับข้อมูลที่มีมิติมาก, scale ต่างกัน, หรือมี noise  
    → ถ้าเราอยากวัด “ความคล้าย” อย่างแท้จริงใน AI → อย่าลืม normalize ก่อนเสมอ หรือใช้ cosine similarity แทนครับ
  </div>
</section>


<section id="ai-application" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">Dot Product ใช้ใน AI อย่างไร?</h2>

  <img
    src="/DotinAI.png"
    alt="Dot Product ใช้ใน AI"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    ในโลกของ AI “ความคล้ายกัน” คือเรื่องสำคัญมาก ไม่ว่าจะเป็นความคล้ายของคำ, รูปภาพ, หรือความชอบของผู้ใช้  
    และเครื่องมือเบื้องหลังที่ใช้วัดความคล้ายเหล่านี้บ่อยที่สุดก็คือ <strong>Dot Product</strong>
  </p>

  <ul className="list-disc pl-6 space-y-3 text-base">
    <li>
      <strong>Word2Vec:</strong>  
      เปลี่ยนคำเป็นเวกเตอร์ แล้วใช้ dot product เปรียบเทียบความหมายของคำ  
      เช่น dot("king", "queen") จะได้ค่าสูง เพราะทั้งคู่มีความหมายใกล้กันใน embedding space
    </li>

    <li>
      <strong>BERT / GPT:</strong>  
      ในโมเดลภาษาสมัยใหม่อย่าง BERT หรือ GPT → ใช้ dot product ใน <strong>Self-Attention</strong>  
      เพื่อตรวจว่า "คำนี้ควรสนใจคำไหนมากที่สุด?" เช่น ในประโยค “I saw a cat and it was sleeping”  
      คำว่า <code>it</code> จะคำนวณ dot product กับคำว่า <code>cat</code> และ <code>sleeping</code> เพื่อจับ context
    </li>

    <li>
      <strong>Vision Transformers (ViT):</strong>  
      ภาพถูกแปลงเป็น patch แล้วแต่ละ patch จะถูกแปลงเป็นเวกเตอร์  
      จากนั้นใช้ dot product วัดว่า patch ไหน “คล้ายกัน” → เพื่อเข้าใจความสัมพันธ์ของพื้นที่ในภาพ
    </li>

    <li>
      <strong>Recommendation System:</strong>  
      แต่ละ user และ content จะมีเวกเตอร์แทนตัวตนของมัน  
      ถ้า dot(user, item) ให้ค่าสูง → แปลว่า content นี้ตรงกับความชอบของ user นั้น  
      เช่น Spotify หรือ Netflix ก็ใช้แนวคิดนี้ในการแนะนำเพลงหรือหนัง
    </li>

    <li>
      <strong>Contrastive Learning:</strong>  
      เทคนิคการเรียนรู้ที่ให้โมเดลแยกแยะว่าอะไร "เหมือน" หรือ "ต่าง" กัน  
      เช่นในโมเดล CLIP ของ OpenAI → ใช้ dot product วัดว่า “คำบรรยาย” ตรงกับ “ภาพ” แค่ไหน
    </li>
  </ul>

  <p className="mt-6">
    นอกจากนี้ยังใช้ในหลายส่วนของ AI เช่นการจัดอันดับผลลัพธ์ของ Search Engine, การจับคู่คำใน Machine Translation,  
    และการตัดสินว่า node ไหนควร connect กันใน Graph Neural Network ก็อิงจาก dot product ด้วยเช่นกัน
  </p>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong> ถ้า AI ต้องตัดสินใจว่า “สิ่งไหนคล้ายสิ่งไหน” → เบื้องหลังมักคือการคูณเวกเตอร์แล้วดูค่าที่ได้  
    เพราะ dot product ให้ค่าความคล้ายที่ทั้ง “เร็ว” และ “แม่นยำ” โดยไม่ต้องเข้าใจเนื้อหาแบบมนุษย์เลย!
  </div>
</section>


<section id="examples" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">ตัวอย่างใน Python</h2>

  <p className="mb-4">
    มาลองใช้โค้ด Python ง่าย ๆ เพื่อเข้าใจการคำนวณ <strong>Dot Product</strong> และ <strong>Cosine Similarity</strong> แบบจับต้องได้  
    เราจะใช้ไลบรารียอดนิยมอย่าง <code>NumPy</code> ที่ถูกใช้ในเกือบทุกโปรเจกต์ AI
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`import numpy as np

v1 = np.array([2, 3])
v2 = np.array([4, 1])

# dot product
dot = np.dot(v1, v2)
print("Dot Product:", dot)

# cosine similarity
cos_sim = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
print("Cosine Similarity:", cos_sim)`}
  </pre>

  <p className="mb-2">
    ในตัวอย่างนี้:
    <br />- <code>v1</code> คือเวกเตอร์แรก <code>[2, 3]</code>  
    <br />- <code>v2</code> คือเวกเตอร์ที่สอง <code>[4, 1]</code>
  </p>

  <p className="mb-2">
    จากนั้นเราใช้ <code>np.dot()</code> เพื่อคำนวณ <strong>Dot Product</strong> ได้ค่า <code>11</code>  
    เพราะ: <code>2×4 + 3×1 = 8 + 3 = 11</code>
  </p>

  <p className="mb-2">
    ต่อมาใช้สูตร <code>cosine similarity</code> เพื่อหาความคล้ายของเวกเตอร์ โดยตัดเรื่องขนาดออก  
    โดยใช้ <code>np.linalg.norm()</code> เพื่อคำนวณความยาวของแต่ละเวกเตอร์ แล้วหารออก
  </p>

  <p className="mb-2">
    ถ้าเวกเตอร์ทั้งสอง “ชี้ไปในทิศใกล้กัน” → cosine similarity จะเข้าใกล้ 1  
    ถ้าตรงข้าม = ใกล้ -1 และถ้าไม่เกี่ยวกัน = ใกล้ 0
  </p>

  <p className="mb-2">
    เราสามารถทดลองเปลี่ยนค่าเวกเตอร์ <code>v1</code> และ <code>v2</code> แล้วดูว่าค่า dot กับ cosine similarity เปลี่ยนยังไงบ้าง  
    มันช่วยให้เข้าใจลึกว่า “เวกเตอร์สัมพันธ์กันแบบไหนบ้าง”
  </p>

  <p className="mb-2">
    ตัวอย่างเช่น:  
    <code>v1 = [1, 0]</code> กับ <code>v2 = [0, 1]</code> → dot = 0, cosine = 0 → แสดงว่า “ตั้งฉากกัน”  
    <code>v1 = [1, 0]</code> กับ <code>v2 = [1, 0]</code> → dot สูงสุด, cosine = 1 → “ทิศเดียวกัน”  
    <code>v1 = [1, 0]</code> กับ <code>v2 = [-1, 0]</code> → dot = -1, cosine = -1 → “ตรงข้ามกัน”
  </p>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong> dot product และ cosine similarity เป็นสูตรที่ใช้จริงทุกวันในงาน AI —  
    ยิ่งเข้าใจภาพรวมทั้งเชิงคณิตศาสตร์และเชิงความหมายเร็วเท่าไหร่ เราก็ยิ่งมี “เครื่องมือวัดความคล้าย” ที่ทรงพลังในมือแล้ว  
    โดยเฉพาะงาน NLP, Recommendation, หรือ Search ที่ต้องหาความสัมพันธ์ของข้อมูลแทบตลอดเวลา
  </div>
</section>

{/* Interactive Try-it-live */}
<section id="interactive-try" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">ลองเองแบบ Interactive 🔄</h2>
  <p className="mb-4">
    ใส่เวกเตอร์ลงไป แล้วดูผลลัพธ์ของ Dot Product และ Cosine Similarity ได้ทันทีครับ  
    เครื่องมือนี้ช่วยให้เข้าใจภาพรวมแบบทดลองจริง
  </p>
  <TryDotProduct />
</section>


<section id="real-world" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">ตัวอย่างในโลกจริง</h2>

  <p className="mb-4">
    แม้ Dot Product จะดูเหมือนสูตรคณิตธรรมดา ๆ แต่จริง ๆ แล้วมันเป็นหัวใจสำคัญของหลายระบบ AI ที่เราใช้ทุกวันโดยไม่รู้ตัว  
    ลองดูตัวอย่างการใช้งานจริง ๆ เหล่านี้ แล้วเราจะเห็นว่า “เบื้องหลังความฉลาดของระบบต่างๆว่ามันฉลาดแค่ไหน” คือแค่การคูณเวกเตอร์ธรรมดา
  </p>

  <ul className="list-disc pl-6 space-y-3 text-sm sm:text-base">
    <li>
      <strong>Search Engine (เช่น Google, YouTube, TikTok):</strong>  
      เวลาเราพิมพ์คำค้นหา ระบบจะเปลี่ยนข้อความของเราให้เป็นเวกเตอร์  
      แล้วใช้ dot product เปรียบเทียบกับเวกเตอร์ของเนื้อหาทั้งหมด → อันดับที่ได้มาคือ “สิ่งที่ใกล้กับความหมายของเราที่สุดครับ”
    </li>

    <li>
      <strong>ChatGPT & LLMs:</strong>  
      ใน Self-Attention กลไกหลักของ GPT หรือ BERT จะใช้ dot product เพื่อดูว่า  
      “คำไหนควรให้ความสนใจมากที่สุด” เวลาประมวลผลประโยค เช่น  
      ในประโยค “The cat sat on the mat” → โมเดลจะใช้ dot product ระหว่างคำว่า “sat” กับคำอื่น  
      เพื่อวิเคราะห์ว่าคำไหนมีความหมายที่เชื่อมโยงกันมากที่สุด
    </li>

    <li>
      <strong>AI แปลภาษา (Translation):</strong>  
      ใช้การวัดความคล้ายของเวกเตอร์ของคำต้นฉบับ (อังกฤษ) กับคำแปล (ไทย)  
      เช่น embedding ของ “love” → ใกล้กับ “รัก” มาก → เลือกเป็นคำแปล  
      ทั้งหมดทำผ่าน dot product ใน vector space ที่ฝึกมาจากภาษาต่าง ๆ
    </li>

    <li>
      <strong>ระบบแนะนำ (Recommendation Systems):</strong>  
      เช่น ทุกครั้งที่เราดูหนังบน Netflix หรือฟังเพลงใน Spotify ระบบจะมีเวกเตอร์แทนตัวเรา
      และอีกเวกเตอร์แทนรายการนั้น ๆ → dot product ระหว่างทั้งสอง = “ความชอบ”  
      ยิ่งสูง = ยิ่งแนะนำ เพราะเวกเตอร์ user กับ item “ชี้ไปในทิศทางเดียวกัน”
    </li>

    <li>
      <strong>AI ตรวจจับวัตถุในภาพ:</strong>  
      Vision Transformers จะใช้ dot product เพื่อตรวจสอบว่า “จุดเล็ก ๆ บนภาพ” คล้ายกันหรือไม่  
      เพื่อจับว่า “อันนี้คือแมว” หรือ “นี่คือรถ” โดยดูจาก patch vector ในภาพ
    </li>
  </ul>

  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong>  
    การคำนวณที่ดูเรียบง่ายอย่าง dot product กลายเป็นเครื่องมือหลักของ AI ทั่วโลก  
    ไม่ว่าจะใช้ค้นหา, แปลภาษา, สนทนา, วิเคราะห์ภาพ, หรือแนะนำสิ่งที่เราชอบ  
    → ถ้าเราเข้าใจมันดีพอ เราก็เข้าใจแก่นแท้ของ “ความเข้าใจ” ที่ AI สร้างขึ้นได้เลย!
  </div>
</section>

<section id="summary-flow" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-bold mb-4">สรุปภาพรวมความเข้าใจ </h2>

  <p className="mb-4">
    ตลอดบทเรียนนี้ เราได้เรียนรู้เกี่ยวกับ <strong>Dot Product</strong>  
    → พัฒนาไปสู่ <strong>Cosine Similarity</strong>  
    → และเข้าใจว่าแนวคิดนี้ถูกนำไปใช้ใน <strong>ระบบ AI</strong> อย่างลึกซึ้งแค่ไหน  
  </p>

  <p className="mb-4">
    ด้านล่างนี้คือแผนภาพที่ช่วยให้เราเห็นเส้นทางของความเข้าใจทั้งหมด:
  </p>

  {/* Flow Diagram (สามารถแทนด้วยภาพได้ในอนาคต) */}
  <div className=" dark:bg-gray-800 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow-sm">
    <div className="text-center text-sm sm:text-base">
      <p> <strong>Vector A</strong> &nbsp; ⨯ &nbsp; <strong>Vector B</strong></p>
      <p>↓</p>
      <p><strong>Dot Product</strong></p>
      <p>↓</p>
      <p><strong>Normalize → Cosine Similarity</strong></p>
      <p>↓</p>
      <p><strong>วัดความคล้ายของข้อมูล</strong></p>
      <p>↓</p>
      <p>
        นำไปใช้ใน → <strong>Word Embedding</strong>, <strong>Search Engine</strong>,{" "}
        <strong>Self-Attention</strong>, <strong>Recommendation</strong>
      </p>
    </div>
  </div>

  {/* Final Insight */}
  <div className="bg-yellow-200 text-black p-6 rounded-xl mt-6 border-l-4 border-yellow-500 shadow-md text-sm sm:text-base leading-relaxed">
    <strong> Insight สุดท้าย:</strong><br />
    จากจุดเริ่มต้นของเวกเตอร์ธรรมดา ๆ → เราสามารถคำนวณ dot product →  
    นำไปสู่ cosine similarity → แล้วใช้วัด "ความคล้ายของความหมาย" ได้อย่างทรงพลัง  
    นี่คือแก่นของ AI ที่เข้าใจภาษา, รูปภาพ, ความชอบ, และบริบทของมนุษย์ได้อย่างน่าทึ่ง  
    <br className="my-2" />
     ถ้าเราเข้าใจภาพรวมนี้ เราได้ก้าวผ่านแค่ “สูตรคณิต” → เข้าสู่ “ความเข้าใจในหัวใจของ AI” แล้วจริงๆครับ
  </div>
</section>


<section id="further-reading" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">แหล่งเรียนรู้ต่อยอด</h2>

  <p className="mb-4">
    ถ้าอยากเข้าใจเชิงลึกยิ่งขึ้นเกี่ยวกับการใช้ Dot Product และ Cosine Similarity ใน AI จริง ๆ  
    ต่อไปนี้คือแหล่งข้อมูลแนะนำที่ทั้งอ่านง่าย และมีประโยชน์มากสำหรับการเรียนรู้ต่อยอด
  </p>

  <ul className="list-disc pl-6 space-y-3 text-sm sm:text-base">
    <li>
      <a
        href="https://en.wikipedia.org/wiki/Word2vec"
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 underline hover:text-blue-800"
      >
        🔗 Word2Vec - Wikipedia
      </a><br />
      อธิบายหลักการแปลงคำให้เป็นเวกเตอร์ และการใช้ dot product วัดความคล้ายกันของคำ
    </li>

    <li>
      <a
        href="https://arxiv.org/abs/1706.03762"
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 underline hover:text-blue-800"
      >
        🔗 Attention is All You Need (Transformer Paper)
      </a><br />
      เปเปอร์ต้นกำเนิดของ Self-Attention ซึ่งใช้ dot product เป็นหัวใจหลัก
    </li>

    <li>
      <a
        href="https://huggingface.co/docs"
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 underline hover:text-blue-800"
      >
        🔗 HuggingFace Documentation
      </a><br />
      คู่มือเครื่องมือ AI/ML ยอดนิยมที่ใช้โมเดลเช่น BERT, GPT พร้อมตัวอย่างการใช้งาน
    </li>

    <li>
      <a
        href="https://jalammar.github.io/illustrated-transformer/"
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 underline hover:text-blue-800"
      >
        🔗 Illustrated Transformer (Jalammar)
      </a><br />
      บทความภาพสวยเข้าใจง่าย อธิบายการทำงานของ Attention และ Dot Product อย่างละเอียด
    </li>

    <li>
      <a
        href="https://machinelearningmastery.com/cosine-similarity/"
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 underline hover:text-blue-800"
      >
        🔗 Cosine Similarity - Machine Learning Mastery
      </a><br />
      อธิบายวิธีคำนวณ cosine similarity พร้อมโค้ด Python และการใช้งานจริง
    </li>
  </ul>
</section>



        <section id="quiz" className="mb-10 mt-10">
          <MiniQuiz_Day3 theme={theme} />
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
        <ScrollSpy_Ai_Day3 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day3_DotProduct;
