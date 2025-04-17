import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import MiniQuiz_Day2 from "./miniquiz/MiniQuiz_Day2";
import ScrollSpy_Ai_Day2 from "./scrollspy/ScrollSpy_Ai_Day2";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day2_VectorOperations = ({ theme }) => {
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
          Day 2: Vector Addition & Scalar Multiplication
        </h1>

        <p className="mb-6 text-lg">
  วันนี้เราจะมาทำความเข้าใจว่า เวกเตอร์สามารถบวก ลบ หรือขยายได้ยังไง  
  และสิ่งเหล่านี้ไม่ใช่แค่เรื่องของตัวเลข แต่คือ “ภาษาที่ AI ใช้ในการคิดและเรียนรู้” ยังไง?
</p>

<section id="vector-addition" className="mb-10 scroll-mt-10">
<h2 className="text-2xl font-semibold mb-3">Vector Addition คืออะไร?</h2>
{/* ✅ ภาพประกอบเวกเตอร์ */}
<img
    src="/VectorAddition.png"
    alt="Vector Addition"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4 text-base">
    เวกเตอร์ไม่ใช่แค่กลุ่มตัวเลข — แต่มันแทนสิ่งที่มีทั้ง “ขนาด” และ “ทิศทาง”  
    แล้วถ้าเรามีเวกเตอร์หลายตัวล่ะ? เราจะ “รวม” ผลของพวกมันได้ไหม? 
    <br />
    คำตอบคือได้ครับ และเราทำแบบนั้นด้วยการ **บวกเวกเตอร์** หรือเรียกทางคณิตว่า <strong>Vector Addition</strong>
  </p>

  <p className="mb-4 text-base">
    การบวกเวกเตอร์ทำได้ง่ายมาก แค่บวกค่าทุกมิติเข้าด้วยกัน เช่น แกน X บวกกับ X, แกน Y บวกกับ Y  
    ผลลัพธ์ที่ได้คือเวกเตอร์ใหม่ ที่มีทิศทางและขนาดแสดงถึง “ผลรวมของการเคลื่อนไหว” หรือ “ผลรวมของข้อมูล” จากทุกเวกเตอร์ก่อนหน้า
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`v1 = [2, 3]
v2 = [1, -1]

v1 + v2 = [3, 2]`}
  </pre>

  <p className="mb-2">
    จากตัวอย่างนี้:  
    - <code>v1</code> คือการเคลื่อนที่ 2 หน่วยไปขวา และ 3 หน่วยขึ้น  
    - <code>v2</code> คือการเคลื่อนที่ 1 หน่วยไปขวา และ 1 หน่วยลง  
    - รวมกันจะได้ <code>[3, 2]</code> = เดิน 3 ขวา และ 2 ขึ้น
  </p>

<p className="mb-2">
  ถ้าให้นึกภาพในชีวิตจริง เช่น หุ่นยนต์เคลื่อนที่บนพื้น 2D — เวกเตอร์จะใช้แทนทิศทางและความแรงของคำสั่งเคลื่อนที่  
  การรวมหลายเวกเตอร์ก็เหมือนกับการ “สั่งหุ่นยนต์ให้ขยับหลายครั้ง แล้วรวมผลลัพธ์สุดท้าย”
</p>

<p className="mb-2">
  เช่น ถ้าหุ่นยนต์ได้รับคำสั่งจากระบบหนึ่งให้เดินไปขวา และอีกระบบให้เดินไปข้างหน้า → สุดท้ายตำแหน่งของมันคือการบวกเวกเตอร์ 2 ตัวนั้นเข้าด้วยกัน
</p>

<p className="mb-2">
  เวกเตอร์ยังใช้ในการรวม “ข้อมูล” แทน “การเคลื่อนไหว” ได้ด้วย เช่น เสียงที่พูด + สีภาพ + คำอธิบาย → รวมเป็น 1 เวกเตอร์ที่โมเดลเข้าใจว่า “นี่คือคลิปวิดีโอที่อธิบายอะไรบางอย่าง”
</p>

<p className="mb-2">
  ในระบบ NLP เช่น ChatGPT เอง ก็ใช้การบวกเวกเตอร์เพื่อรวมความหมายของคำ เช่น “not” + “happy” = ความรู้สึก “เศร้า”  
  เพราะเวกเตอร์ของคำว่า “not” จะลบคุณสมบัติบางอย่างออกจากเวกเตอร์ของ “happy”
</p>

<p className="mb-2">
  ถ้าดูในโมเดลแบบ Transformer เช่น BERT หรือ GPT → ตัว hidden state ที่โมเดลใช้ส่งต่อระหว่างคำ  
  จะถูก “บวก” กันแบบเวกเตอร์ เพื่อให้ระบบเข้าใจความสัมพันธ์ของประโยคโดยรวม
</p>

<p className="mb-2">
  นอกจากนี้ เวกเตอร์ในระบบ recommendation เช่น Netflix ก็จะถูกนำมาบวกกันเพื่อหาความสนใจรวมของผู้ใช้  
  เช่น: ชอบหนังผจญภัย + ชอบเนื้อหาวิทยาศาสตร์ → รวมเป็นความชอบในหนัง Sci-Fi Adventure
</p>

<p className="mb-2">
  การบวกเวกเตอร์ยังแสดงถึงการ “รวมความรู้” จากหลายแหล่ง เช่น ใน multi-modal AI ที่รับ input จากทั้งภาพ + เสียง + คำ  
  การบวกเวกเตอร์จะรวม 3 อย่างเข้าด้วยกันเป็น “ความเข้าใจแบบองค์รวม”
</p>

<p className="mb-2">
  ในภาพรวม การบวกเวกเตอร์ = การรวมความเข้าใจจากหลายมุมมองเข้าด้วยกัน  
  ไม่ว่าจะเป็นการเคลื่อนที่ในกายภาพ หรือความหมายในข้อมูลดิจิทัล
</p>

<p className="mb-2">
  ยกตัวอย่างเช่น เรามีข้อมูลจาก sensor 3 ตัวในหุ่นยนต์ → แต่ละตัวให้เวกเตอร์ที่แทนมุมมองของมัน  
  พอรวมกันแล้วจะได้ข้อมูลที่ "แม่นยำกว่า" และลด noise
</p>

<p className="mb-2">
  และถ้าคุณเข้าใจวิธีบวกเวกเตอร์ในทางคณิตศาสตร์ เช่น <code>[x1, y1] + [x2, y2]</code> = <code>[x1+x2, y1+y2]</code>  
  คุณก็สามารถขยายความเข้าใจนี้ไปในเวกเตอร์ที่มี 100 หรือ 1,000 มิติได้ทันที เพราะหลักการเหมือนกัน 100%
</p>

<p className="mb-2">
  ใน deep learning เวกเตอร์แต่ละตัวอาจแทนความหมายซับซ้อนมาก เช่น concept “ความสุข”, “ความเศร้า”, “สุนัข”, “อารมณ์”  
  การบวกเวกเตอร์เหล่านี้คือการ “ปั้น” ความรู้ใหม่จากสิ่งเดิม
</p>

<p className="mb-2">
  นอกจากนี้การบวกเวกเตอร์ยังสามารถใช้สร้างสิ่งใหม่ เช่น การใช้เวกเตอร์ของ “king” ลบ “man” แล้วบวก “woman”  
  ผลลัพธ์จะใกล้เคียง “queen” → นี่คือตัวอย่างที่โด่งดังของ Word2Vec
</p>

<p className="mb-2">
  อย่าลืมว่าเวกเตอร์ไม่ได้จำกัดแค่ 2 มิติอย่างที่เรานึกภาพง่าย ๆ  
  เวกเตอร์ใน AI มีหลายร้อยหรือพันมิติ และการบวกก็คือการบวกแต่ละมิติไปทีละตัวเหมือนกัน
</p>

<p className="mb-2">
  โดยทั่วไปโครงข่ายประสาทเทียมจะรับ input เป็นเวกเตอร์ เช่น รูปภาพขนาด 28x28 พิกเซลจะถูกแปลงเป็นเวกเตอร์ขนาด 784  
  และบวกกับเวกเตอร์ของ bias หรือน้ำหนักระหว่างเลเยอร์เพื่อให้ได้ output ที่ใหม่
</p>

<p className="mb-2">
  แม้แต่ใน GAN (Generative Adversarial Networks) ที่ใช้สร้างภาพ → noise input ที่ใช้สุ่มภาพขึ้นมาก็เป็นเวกเตอร์  
  และการบวก noise กับ latent vector อื่น ๆ ก็จะเปลี่ยนผลลัพธ์ของภาพ
</p>

<p className="mb-2">
  ใน Reinforcement Learning เช่น AlphaGo → เวกเตอร์แทน state ของกระดานปัจจุบัน  
  และการ “เปลี่ยนสถานะ” ก็คือการบวกเวกเตอร์ของ action เข้าไปใน state เดิม
</p>

<p className="mb-2">
  ลองจินตนาการว่าเวกเตอร์คือ “พลังงานบางอย่าง” และการบวกเวกเตอร์ก็คือการรวมพลังเพื่อขยับโลก  
  → สิ่งนี้คือแก่นแท้ของการทำงานเบื้องหลัง AI
</p>

<p className="mb-2">
  ที่น่าสนใจคือ การบวกเวกเตอร์สามารถใช้ “ลบ” ได้เช่นกัน เพราะการลบคือการบวกเวกเตอร์ตรงข้าม  
  เช่น <code>[2, 3] - [1, 1]</code> = <code>[2, 3] + [-1, -1]</code> = <code>[1, 2]</code>
</p>

<p className="mb-2">
  หลายครั้งใน AI จะใช้เทคนิค residual connection → ซึ่งคือการ “บวกค่าเดิมกลับเข้าไป” เพื่อไม่ให้โมเดลลืมข้อมูลเก่า  
  → ตรงนี้ก็ใช้หลักการเวกเตอร์บวกธรรมดา
</p>

<p className="mb-2">
  สุดท้าย เวกเตอร์ addition คือภาษาสากลของทุกระบบปัญญาประดิษฐ์  
  มันคือเครื่องมือที่เรียบง่าย แต่ทรงพลังที่สุดในโลกของการเรียนรู้ของเครื่อง
</p>

<p className="mb-2">
  ถ้าคุณเข้าใจการบวกเวกเตอร์ → คุณก็เริ่มเข้าใจวิธีที่ AI “เรียนรู้โลก”  
  เพราะโลกของ AI ไม่มีภาพ เสียง หรือคำ แต่มีแค่เวกเตอร์
</p>

<p className="mb-2 font-semibold text-green-400">
  ✅ Tip: ลองวาดเวกเตอร์ด้วยมือลงกระดาษ แล้วบวกมันดู จะช่วยให้คุณเข้าใจมากขึ้น  
  และสิ่งที่คุณเรียนรู้นี้ จะใช้ได้ไปตลอดในทุกสาย AI!
</p>

</section>

<section id="vector-visualization" className="mb-10 scroll-mt-14">
  <h2 className="text-2xl font-semibold mb-4"> Vector Visualization</h2>
  <img
    src="/VectorVisualization.png"
    alt="Vector Visualization"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4 text-base">
    ถึงแม้ว่าเวกเตอร์จะเป็นแนวคิดทางคณิตศาสตร์ แต่การวาดภาพจะช่วยให้เข้าใจได้ลึกขึ้นมาก
    เพราะเราจะได้เห็นทิศทาง การเคลื่อนที่ และผลรวมของเวกเตอร์ต่าง ๆ อย่างชัดเจนบนระนาบกราฟ 2D
  </p>

  <h3 className="text-xl font-semibold mb-2 mt-6"> การวาดเวกเตอร์ด้วย Matplotlib</h3>
  <p className="mb-2">ลองใช้ Python + Matplotlib เพื่อวาดเวกเตอร์อย่างง่าย เช่น:</p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`import matplotlib.pyplot as plt
import numpy as np

# สร้างเวกเตอร์
v1 = np.array([2, 3])
v2 = np.array([1, -1])

# วาดเวกเตอร์จากจุด (0,0)
origin = np.array([[0, 0], [0, 0]])
vectors = np.array([v1, v2])

plt.quiver(*origin, vectors[:,0], vectors[:,1], angles='xy', scale_units='xy', scale=1, color=['r','b'])
plt.xlim(-1, 5)
plt.ylim(-2, 5)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Vector Visualization')
plt.show()`}
  </pre>

  <p className="mb-2">
    จากภาพนี้คุณจะเห็นเวกเตอร์ 2 ตัวที่มีทิศและขนาดแตกต่างกันชัดเจน
    สีแดง = เวกเตอร์ <code>v1</code>, สีฟ้า = <code>v2</code>
  </p>

  <h3 className="text-xl font-semibold mb-2 mt-6"> การวาด Vector Addition</h3>
  <p className="mb-2">
    เมื่อต้องการดูผลรวมของเวกเตอร์ เราสามารถวาดเวกเตอร์ที่สองโดยให้ปลายอยู่ที่ปลายของเวกเตอร์แรก แล้วลากเวกเตอร์ผลรวม (resultant vector)
  </p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`# ผลรวมของเวกเตอร์
v3 = v1 + v2

# วาดเวกเตอร์ใหม่รวมถึงผลรวม
vectors = np.array([v1, v2, v3])
colors = ['r', 'b', 'g']

# เริ่มที่ origin สำหรับ v1
# v2 เริ่มที่ปลาย v1 = (2,3)
# v3 แสดงจาก origin ถึงผลรวม (3,2)
plt.quiver(*origin, vectors[:,0], vectors[:,1], angles='xy', scale_units='xy', scale=1, color=colors)
plt.quiver(2, 3, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b')  # วาด v2 จากปลาย v1
plt.xlim(-1, 6)
plt.ylim(-2, 6)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Vector Addition Visualization')
plt.show()`}
  </pre>

  <h3 className="text-xl font-semibold mt-6 mb-2"> การขยายเวกเตอร์ด้วย Scalar</h3>
  <p className="mb-2">
    การคูณ scalar จะทำให้เวกเตอร์ยาวขึ้นหรือลดลง ลองดูตัวอย่างนี้:
  </p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`# คูณเวกเตอร์ด้วย scalar
v_scaled = 3 * v1

# แสดงเวกเตอร์เดิมและเวกเตอร์ที่ถูกขยาย
vectors = np.array([v1, v_scaled])
colors = ['gray', 'orange']

plt.quiver(*origin, vectors[:,0], vectors[:,1], angles='xy', scale_units='xy', scale=1, color=colors)
plt.xlim(-1, 10)
plt.ylim(-1, 10)
plt.title('Scalar Multiplication Visualization')
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()`}
  </pre>

  <h3 className="text-xl font-semibold mt-6 mb-2"> เปรียบเทียบภาพใน AI จริง</h3>
  <ul className="list-disc pl-6 space-y-2">
    <li>ภาพพิกเซล RGB จะถูกวาดเป็นเวกเตอร์ 3D หรือมากกว่า</li>
    <li>เสียงถูกวาดเป็นเวกเตอร์ของความสูง amplitude ในแต่ละช่วงเวลา</li>
    <li>เวกเตอร์ของ embedding คำต่าง ๆ จะถูกลดมิติด้วย PCA หรือ t-SNE เพื่อนำมาวาดใน 2D</li>
  </ul>

  <p className="mt-4">
    การเห็นภาพเวกเตอร์ไม่ใช่แค่ทำให้เข้าใจง่ายขึ้น แต่ยังช่วยในการ debug, การวิเคราะห์ข้อมูล, และสร้างความมั่นใจในสิ่งที่โมเดล AI กำลังเรียนรู้
  </p>

  <div className="mt-6 bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
    <strong> Bonus Tip:</strong> ลองใช้ไลบรารี <code>plotly</code> เพื่อให้สามารถหมุนกราฟได้ในแบบ interactive โดยเฉพาะถ้ามีเวกเตอร์ 3D เช่น [x, y, z]
  </div>
</section>

<section id="vector-subtraction" className="mb-26 scroll-mt-20">
  <span  className="block h-1"></span>
  <h2 className="text-2xl font-semibold mb-3">Vector Subtraction</h2>
  <img
    src="/VectorSubtraction.png"
    alt="Vector Subtraction"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-4">
    การลบเวกเตอร์ (Vector Subtraction) เป็นกระบวนการพื้นฐานถัดจากการบวกเวกเตอร์  
    ซึ่งใช้เพื่อหาความแตกต่างของทิศทางและขนาดระหว่างเวกเตอร์สองตัว  
    หลักการก็ง่าย ๆ คือ <strong>ลบค่าของแต่ละมิติออกจากกัน</strong> เช่น:
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`v1 = [5, 7]
v2 = [2, 3]

v1 - v2 = [3, 4]`}
  </pre>

  <p className="mb-2">
    ในตัวอย่างด้านบน เราเอา <code>v2</code> ลบออกจาก <code>v1</code>  
    → ผลลัพธ์คือเวกเตอร์ใหม่ที่แสดงถึง “การเคลื่อนที่จาก v2 ไปยัง v1”
  </p>

  <p className="mb-2">
    ถ้าเปรียบกับการเดินทางในชีวิตจริง เช่น:
  </p>
  <ul className="list-disc pl-6 mb-4">
    <li><code>v1 = [5, 7]</code> → จุดหมายที่อยู่ขวา 5 และบน 7 หน่วย</li>
    <li><code>v2 = [2, 3]</code> → จุดเริ่มต้น</li>
    <li>ผลลัพธ์ <code>[3, 4]</code> หมายถึงการเคลื่อนที่จาก <code>v2</code> ไปยัง <code>v1</code></li>
  </ul>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ใช้ใน AI อย่างไร?</h3>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>วัดความแตกต่างของเวกเตอร์คำ เช่น “happy” - “sad”</li>
    <li>แยกอารมณ์หรือคุณลักษณะบางอย่างออกจากข้อมูล เช่น “king” - “man” = “royalty”</li>
    <li>ใช้ใน Recommendation System เพื่อหาความต่างของ preferences</li>
    <li>ใช้ใน Vector Space Models เพื่อตรวจจับทิศทางของความหมาย</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2"> ตัวอย่างใน Python</h3>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`import numpy as np

v1 = np.array([5, 7])
v2 = np.array([2, 3])

sub = v1 - v2
print("ผลลบเวกเตอร์:", sub)  # Output: [3 4]`}
  </pre>

  <p className="mb-4">
    การลบเวกเตอร์ใน Python ก็เหมือนกับการบวก เพียงแค่ใช้เครื่องหมายลบ (-) แทนเท่านั้น  
    NumPy รองรับการลบอัตโนมัติโดยไม่ต้องใช้ loop ใด ๆ
  </p>

  <h3 className="text-lg font-semibold mb-2">🧠 เปรียบเทียบกับการบวก</h3>
  <ul className="list-disc pl-6 space-y-2">
    <li>การบวกเวกเตอร์: รวมผลกระทบ → ได้ผลลัพธ์ใหม่ที่รวมทิศทางและขนาด</li>
    <li>การลบเวกเตอร์: หาความต่าง → เหมือนย้อนกลับเส้นทางจากจุดหนึ่งไปอีกจุดหนึ่ง</li>
    <li>ถ้า <code>a + b = c</code> → <code>c - b = a</code> (เหมือนในคณิตศาสตร์พื้นฐาน)</li>
  </ul>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ตัวอย่างในงานจริง</h3>
  <ul className="list-disc pl-6 space-y-2 text-sm sm:text-base">
    <li>ใน NLP: ลบคำคุณศัพท์ออกจากคำหลัก เช่น “not” - “positive” = “neutral”</li>
    <li>ใน AI Vision: หาความแตกต่างระหว่างภาพ 2 รูปแบบ เช่น “ภาพกลางวัน” - “แสง” = “กลางคืน”</li>
    <li>ในการสร้าง latent space ของ GANs: ปรับคุณลักษณะของภาพ เช่น ลบความยิ้มออกจากใบหน้า</li>
  </ul>
   <section id= "insight-box" className="scroll-mt-20 mb-16">
  <h3 className="text-lg font-semibold mt-6 mb-2"> Insight Box</h3>
  <div className="bg-yellow-100 text-black p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <p>
      ถ้า “การบวกเวกเตอร์” คือการรวมสิ่งต่าง ๆ เข้าด้วยกัน  
      <br />
      “การลบเวกเตอร์” ก็คือการหาว่าอะไร "ขาดหาย" หรือ "แตกต่าง"
    </p>
    <p className="mt-2">
      ในโลกของ AI → เวกเตอร์ subtraction ช่วยให้โมเดลเข้าใจความเปลี่ยนแปลงของข้อมูล  
      เช่น จากอารมณ์ A → B, จากภาพ A → B หรือจากความหมาย A → B
    </p>
  </div>

  <p className="mt-6 font-semibold text-green-400">
    ✅ Tip: ลองวาดเวกเตอร์สองเส้นบนกระดาษ แล้วหาความแตกต่างของทิศทาง  
    จะเห็นว่าการลบเวกเตอร์คือการสร้างเวกเตอร์ใหม่ที่ “พาเรา” จากจุด A → จุด B
  </p>
</section>
</section>



<section id="scalar-mult" className="mb-16 scroll-mt-20">
  <span  className="block h-1"></span>
  <h2 className="text-2xl font-semibold mb-3">Scalar Multiplication คืออะไร?</h2>

  <img
    src="/ScalarMultiplication.png"
    alt="Scalar Multiplication"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`v = [2, 3]
scalar = 4

v * 4 = [8, 12]`}
  </pre>

  <p className="mb-2">
    Scalar Multiplication คือการนำ “ตัวเลขตัวเดียว” (เรียกว่า scalar) ไปคูณกับเวกเตอร์ทั้งหมด  
    โดยจะคูณเข้าไปในทุกมิติของเวกเตอร์นั้น เช่น <code>[2, 3]</code> × 4 = <code>[8, 12]</code>
  </p>

  <p className="mb-2">
    ผลที่เกิดขึ้นคือ “ขยาย” ขนาดของเวกเตอร์ออกไปในทิศทางเดิม แต่ยาวขึ้น → เหมือนเราขยายลูกศรที่ชี้ไปทิศหนึ่งให้ยาวขึ้น 4 เท่า
  </p>

  <p className="mb-2">
    ถ้า scalar เป็นค่าลบ เช่น × -1 → เวกเตอร์จะ “กลับทิศทาง” ทันที โดยยังคงขนาดเดิม → ใช้ในกรณีต้องการหาทิศทางตรงข้าม
  </p>

  <p className="mb-2">
    การคูณแบบนี้เกิดขึ้นตลอดเวลาในโลก AI เช่น การคูณเวกเตอร์ข้อมูลเข้ากับเวกเตอร์น้ำหนักของโมเดล  
    หรือการขยายผลของ feature ที่สำคัญให้เด่นชัดขึ้น
  </p>

  <p className="mb-2">
    ในด้านภาพ (image processing) → ถ้าภาพถูกแปลงเป็นเวกเตอร์ของพิกเซล เช่น <code>[100, 150, 200]</code>  
    แล้วคูณด้วย scalar 0.5 → เราจะได้ภาพที่ “มืดลง” เพราะค่าพิกเซลลดลงครึ่งหนึ่ง
  </p>

  <p className="mb-2">
    ในด้านเสียง → สัญญาณเสียงจะถูกแทนด้วยเวกเตอร์ของค่า amplitude ในแต่ละช่วงเวลา  
    การคูณด้วย scalar เช่น 2.0 จะเพิ่มความดังของเสียงเท่าตัว (amplify) และถ้าใช้ 0.1 ก็ลดเสียงลง
  </p>

  <p className="mb-2">
    ในการ normalize ข้อมูล (การปรับข้อมูลให้อยู่ในสเกลที่เหมาะสม) → เรามักหารเวกเตอร์ด้วย norm หรือคูณด้วย scalar  
    เช่น คูณด้วย <code>1 / ||v||</code> เพื่อให้เวกเตอร์มีขนาดเท่ากัน
  </p>

  <p className="mb-2">
    การ normalize นี้สำคัญใน AI เพราะจะช่วยให้โมเดลเรียนรู้ได้ง่ายขึ้น และไม่ถูกค่าที่สูงมาก “เบี่ยงเบนการเรียนรู้”
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ประโยชน์ใน AI:</h3>
  <ul className="list-disc pl-6 space-y-2">
    <li>ใช้ควบคุมความแรงของข้อมูล (เช่น scaling feature ให้มีค่า 0-1 หรือ -1 ถึง 1)</li>
    <li>ใช้ในการ “ขยายผล” หรือ “ลดทอน” ค่าที่โมเดลประมวลผลระหว่างชั้นใน neural network</li>
    <li>คูณกับ weights เพื่อ “ปรับน้ำหนัก” ของผลลัพธ์ตามความสำคัญ</li>
    <li>ใช้เพื่อแปลงความรู้สึก, ความหมาย, หรือทิศทางใน embedding space เช่น emotion intensity</li>
  </ul>

  <p className="mt-4 mb-2">
    ถ้าคุณเข้าใจว่า scalar multiplication คือการควบคุม “น้ำหนัก” หรือ “พลัง” ของเวกเตอร์  
    คุณจะเริ่มเห็นภาพว่า AI ใช้แนวคิดนี้กับทุกอย่าง เช่น การเน้นคำสำคัญ, การขยายพลังของภาพ, หรือการ suppress เสียง noise
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2"> ตัวอย่างในโลกจริง:</h3>
  <ul className="list-disc pl-6 space-y-2 text-sm sm:text-base">
    <li>Spotify ปรับน้ำหนัก feature “จังหวะ” หรือ “อารมณ์เพลง” โดยคูณเวกเตอร์ของเพลงนั้นด้วยค่า scalar ที่เหมาะสม</li>
    <li>รถยนต์ไร้คนขับ ปรับความสำคัญของการตรวจจับวัตถุหน้ารถโดยใช้ scalar multiplication เพื่อเน้นระยะทาง/ความเร่ง</li>
    <li>ระบบ recommendation ใช้ scalar ในการ balance น้ำหนักของเวกเตอร์ user vs. เวกเตอร์ของ content</li>
    <li>การตั้งค่า learning rate ใน neural network → เป็นการคูณ scalar เข้ากับ gradient เวกเตอร์ เพื่อปรับ step ในการเรียนรู้</li>
  </ul>

  <h3 className="text-lg font-semibold mt-6 mb-2"> Insight Box:</h3>
  <div className="bg-yellow-100 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
    <strong>ถ้าเวกเตอร์คือทิศทางของความหมาย — scalar คือความแรงของความหมายนั้น</strong><br />
    เช่น "ฉันรักคุณ" = [1, 0.8, 0.5] × 1.0  
    แต่ "ฉันรักคุณมากกกกก" = [1, 0.8, 0.5] × 1.8  
    → แค่นี้ AI ก็เข้าใจ "น้ำหนักทางอารมณ์" ที่เพิ่มขึ้นได้!
  </div>

  <p className="mt-6 font-semibold text-green-400">
     ลองสร้างเวกเตอร์ใน Python แล้วคูณด้วย scalar หลายค่า แล้ว plot กราฟดู จะเห็นว่าเวกเตอร์เปลี่ยน "ความยาว" แต่ไม่เปลี่ยนทิศ
  </p>
</section>

<section id="examples" className="mb-10 scroll-mt-20">
  <h2 className="text-2xl font-semibold mb-3"> ตัวอย่างใน Python</h2>
  <p className="mb-4">
    เพื่อให้เข้าใจจริง ลองมาดูการใช้งาน <strong>Vector Addition</strong> และ <strong>Scalar Multiplication</strong> แบบง่าย ๆ ด้วย Python และ NumPy ซึ่งเป็นไลบรารีหลักที่นิยมใช้ในงาน AI
  </p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`import numpy as np

v1 = np.array([2, 3])
v2 = np.array([1, -1])

# บวกเวกเตอร์
add = v1 + v2
print("บวกเวกเตอร์:", add)

# คูณเวกเตอร์ด้วย scalar
scaled = 4 * v1
print("ขยายเวกเตอร์:", scaled)`}
  </pre>
  <p className="mb-2">
    - บรรทัดแรกเราสร้างเวกเตอร์ <code>v1</code> และ <code>v2</code> โดยใช้ <code>np.array</code>  
    - จากนั้นบวกทั้งสองเวกเตอร์ได้ผลลัพธ์ <code>[3, 2]</code> ซึ่งก็คือการรวมผลกระทบจากทิศทางทั้งสอง
  </p>
  <p className="mb-2">
    - ต่อมาคือการขยาย <code>v1</code> ด้วย scalar <code>4</code> ซึ่งหมายถึงการเพิ่มขนาดของเวกเตอร์ให้มากขึ้น 4 เท่า  
    → ผลลัพธ์ที่ได้คือ <code>[8, 12]</code>
  </p>
  <p className="mb-2">
    ใน AI จริง เราจะใช้แนวคิดนี้ตลอดเวลา เช่น  
    - การขยายค่า embedding  
    - การเพิ่ม/ลดผลกระทบของ feature บางตัว  
    - หรือแม้แต่การควบคุมน้ำหนักการเรียนรู้ของโมเดล
  </p>
</section>



<section id="concept" className="mb-10 scroll-mt-20">
  <h2 className="text-2xl font-semibold mb-3"> เชิงความเข้าใจ</h2>
  <p className="mb-4">
    นี่คือหัวใจของสิ่งที่คุณเรียนในวันนี้ — ถ้าคุณเข้าใจแนวคิดเบื้องหลังสองกระบวนการนี้  
    คุณจะเข้าใจว่าการประมวลผลข้อมูลของ AI แท้จริงก็คือ "การจัดการเวกเตอร์" นั่นเอง
  </p>
  <ul className="list-disc pl-6 space-y-3 text-base">
    <li>
      <strong>การบวกเวกเตอร์ = การรวมพลังหลายทิศทาง</strong><br />
      เช่น การรวมข้อมูลจากหลายเซนเซอร์, หลาย modal เช่น ภาพ+เสียง+ข้อความ  
      หรือแม้แต่ hidden state หลายชั้นใน RNN/Transformer
    </li>
    <li>
      <strong>การคูณเวกเตอร์ = การเพิ่มน้ำหนักให้สิ่งที่สำคัญ</strong><br />
      ใช้เมื่อต้องการเน้นบาง feature, ลดความสำคัญของ noise  
      หรือปรับสมดุลระหว่าง input ต่าง ๆ
    </li>
    <li>
      <strong>สองอย่างนี้คือรากฐานของ:</strong><br />
      - การทำ Attention<br />
      - การ normalize ค่าใน training<br />
      - การอัปเดตน้ำหนัก (weight updates) ใน Neural Network<br />
      - การคิดเชิงภาพ (เช่น Image Filtering) ก็ใช้การคูณเวกเตอร์กับเมทริกซ์ Kernel
    </li>
  </ul>
  <p className="mt-6">
    ถ้าคุณเห็นว่า “เวกเตอร์คือนามธรรมของทุกอย่าง” → คุณกำลังเข้าใจภาษาหลักของ AI แล้ว  
    ไม่ว่าจะเป็น NLP, Computer Vision, หรือ Reinforcement Learning
  </p>
</section>

<section id="insight" className="mb-12 mt-10">
  <h2 className="text-2xl font-semibold mb-3">Insight Box </h2>
  <blockquote className="border-l-4 border-yellow-500 pl-4 italic space-y-4">

    <p>
      เวกเตอร์ไม่ใช่แค่ตัวเลขในคณิตศาสตร์ — มันคือภาษาของ <strong>"การเคลื่อนที่"</strong> และ <strong>"การเปลี่ยนแปลง"</strong>  
      ทุกสิ่งที่มีทิศทาง มีความแรง หรือส่งผลต่อสิ่งอื่น ล้วนสามารถถูกแทนด้วยเวกเตอร์ได้
    </p>

    <p>
      ลองนึกถึงโลกจริง:  
      - ถ้าคุณกำลังเดินไปทางทิศตะวันออก = นั่นคือเวกเตอร์หนึ่ง  
      - ถ้าลมพัดคุณไปข้างหลังเล็กน้อย = อีกเวกเตอร์  
      <br />
      <strong>เมื่อรวมเวกเตอร์ทั้งสอง</strong> → คุณได้ตำแหน่งใหม่ = <u>ผลกระทบรวม</u> จากหลายแรงที่กระทำ
    </p>

    <p>
      ในโลกของ AI ก็เช่นกัน:  
      - ข้อมูลภาพ = เวกเตอร์ของ pixel  
      - ข้อความ = เวกเตอร์ของความหมาย  
      - เสียง = เวกเตอร์ของคลื่น  
      <br />
      <strong>เมื่อเราบวกหรือคูณเวกเตอร์เหล่านี้</strong> → เรากำลังรวมผลกระทบของสิ่งต่าง ๆ เพื่อเข้าใจสิ่งที่ซับซ้อนขึ้นไปอีก
    </p>

    <p>
      การบวกเวกเตอร์ = การรวมหลายมุมมอง  
      การคูณเวกเตอร์ = การขยายสิ่งที่สำคัญ  
      → ทั้งสองคือหัวใจของ <strong>“การเรียนรู้ของเครื่อง”</strong> เช่น  
      - การรวม context ใน ChatGPT  
      - การเพิ่มน้ำหนัก feature สำคัญในโมเดลภาพ  
      - การปรับค่าผ่าน backpropagation ใน Neural Network
    </p>

    <p>
      ถ้าเข้าใจสิ่งนี้ดีพอ —  
      <span className="text-yellow-400 font-bold">คุณเข้าใจกลไกเบื้องหลังของ AI แล้วอย่างน้อยก็ครึ่งหนึ่งครับผม</span>
    </p>

  </blockquote>
</section>


        <section id="quiz" className="mb-10 mt-10">
        <MiniQuiz_Day2 theme={theme} />
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
        <ScrollSpy_Ai_Day2 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day2_VectorOperations;
