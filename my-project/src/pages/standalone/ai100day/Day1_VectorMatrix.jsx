// src/pages/standalone/ai100day/Day1_VectorMatrix.jsx
import React from "react";
import { useNavigate } from "react-router-dom"; // ✅ ใช้ Outlet และ useParams
import { FaPlay } from "react-icons/fa";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import MiniQuiz_Day1 from "./miniquiz/MiniQuiz_Day1";
import ScrollSpy_Ai_Day1 from "./scrollspy/ScrollSpy_Ai_Day1";

const Day1_VectorMatrix = ({ theme }) => {
  const navigate = useNavigate();
  return (
    <div
      className={`relative min-h-screen ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
      }`}
    >
      {/* ✅ เนื้อหาหลัก */}
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">
          Day 1: Introduction to Vectors & Matrices
        </h1>

        <p className="mb-6 text-lg">
          ในบทเรียนนี้ คุณจะได้เข้าใจว่าทำไมเวกเตอร์ (Vector) และเมทริกซ์ (Matrix)
          จึงเป็นหัวใจของงานด้านปัญญาประดิษฐ์และการเรียนรู้ของเครื่อง (AI/ML)
        </p>

        <section id="vector" className="mb-10">
  <h2 className="text-2xl font-semibold mb-3">Vector คืออะไร</h2>

  {/* ✅ ภาพประกอบเวกเตอร์ */}
  <img
    src="/vector.png"
    alt="Vector Illustration"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />

  <p className="mb-2">
    เวกเตอร์ คือ ตัวแทนของข้อมูลที่มีทั้ง “ขนาด” และ “ทิศทาง” เช่น ถ้าคุณเดิน 5 ก้าวไปทางขวา การกระทำนี้มีทั้งขนาด (5) และทิศทาง (ขวา) — นี่คือ เวกเตอร์ในชีวิตจริง
  </p>

  <p className="mb-2">
    ในคณิตศาสตร์ เวกเตอร์มักเขียนเป็นกลุ่มของตัวเลข เช่น <code>[3, 4]</code> ซึ่งหมายถึง จุดหนึ่งในพิกัด XY โดยมีความยาว (หรือ norm) เท่ากับ 5 (จากสูตรพีทาโกรัส √(3²+4²))
  </p>

  <p className="mb-2">
    ถ้าเป็นระบบที่มีหลายมิติ เช่น 3D เราอาจใช้เวกเตอร์ <code>[x, y, z]</code> เช่น <code>[1, 2, 3]</code> เพื่อแสดงตำแหน่งในอวกาศ เช่นในเกม 3 มิติ หรือระบบหุ่นยนต์
  </p>

  <p className="mb-2">
    ข้อมูลประเภทอื่น ๆ เช่น สีของภาพ RGB ก็สามารถเป็นเวกเตอร์ เช่น <code>[255, 120, 60]</code> หมายถึงสีแดงเข้ม หรือแม้แต่ข้อมูลข้อความก็สามารถถูกแปลงเป็นเวกเตอร์ของค่า embedding ได้ เช่นเวกเตอร์ที่แทนคำว่า "apple"
  </p>

  <p className="mb-2">
    ในงาน AI/ML เวกเตอร์จะถูกใช้แทนข้อมูล 1 ชิ้น เช่น:
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-1">
    <li>ข้อมูลภาพ 28x28 พิกเซล จะถูก flatten เป็นเวกเตอร์ขนาด 784</li>
    <li>เสียง จะถูกแปลงเป็นเวกเตอร์ของค่าความถี่ในช่วงเวลาหนึ่ง</li>
    <li>ข้อความ จะถูกแปลงเป็นเวกเตอร์ผ่านการ embedding เช่น Word2Vec หรือ BERT</li>
  </ul>

  <p className="mb-2">
    เวกเตอร์ยังสามารถใช้วัด “ความคล้าย” ระหว่างสิ่งต่าง ๆ ได้ เช่น การใช้ cosine similarity เพื่อดูว่าคำสองคำมีความหมายใกล้เคียงกันแค่ไหน
  </p>

  <p className="mb-2">
    ในเชิงคณิตศาสตร์ เราสามารถบวก ลบ คูณเวกเตอร์ได้ เช่น การบวกเวกเตอร์สองตัวจะหมายถึงการ “รวม” ทิศทางและขนาดของข้อมูลเข้าด้วยกัน
  </p>

  <p>
    สรุปคือ เวกเตอร์เป็นสิ่งสำคัญมากในโลกของ AI เพราะสามารถใช้แทนทุกสิ่งที่เราต้องการสื่อ เช่น ภาพ เสียง ข้อความ หรือแม้แต่ตำแหน่งของวัตถุในโลกจริง
  </p>
</section>

<section id="matrix" className="mb-10">
  <h2 className="text-2xl font-semibold mb-3">Matrix คืออะไร</h2>

  <img
    src="/matrix.png"
    alt="Matrix illustration"
    className="rounded-xl shadow-lg mb-6 w-full max-w-md mx-auto shadow-md border border-yellow-400"
  />

  <p className="mb-2">
    <strong>Matrix</strong> (เมทริกซ์) คือโครงสร้างข้อมูลที่เก็บตัวเลขหลายตัวในรูปแบบ “ตาราง” ซึ่งประกอบด้วยแถว (rows) และคอลัมน์ (columns)
  </p>

  <p className="mb-2">
    ถ้าคุณเคยเรียนตารางคูณ หรือเคยดูตาราง Excel — นั่นคือแนวคิดเดียวกับ Matrix
  </p>

  <p className="mb-2">
    ตัวอย่างเช่น Matrix ขนาด 2x3 (2 แถว 3 คอลัมน์):
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-3">
{`[
  [1, 2, 3],
  [4, 5, 6]
]`}
  </pre>

  <p className="mb-2">
    - แถวที่ 1 คือเวกเตอร์ [1, 2, 3]  
    - แถวที่ 2 คือเวกเตอร์ [4, 5, 6]
  </p>

  <p className="mb-4">
    เพราะฉะนั้น Matrix ก็คือ “กลุ่มของเวกเตอร์ที่มีขนาดเท่ากันและเรียงซ้อนกัน”  
    ถ้าเรามีข้อมูล 100 รูปภาพ (vector) แต่ละรูปแทนด้วย 784 พิกเซล → เราจะได้ Matrix ขนาด 100x784
  </p>

  <h3 className="text-xl font-semibold mb-2">เมทริกซ์ในมุมมอง AI</h3>
  <ul className="list-disc list-inside mb-4 space-y-1">
    <li>รูปภาพ = เมทริกซ์ของพิกเซล (เช่น รูปขาวดำ 28x28 พิกเซล = Matrix 28x28)</li>
    <li>ข้อความ = ถูกแปลงเป็นเวกเตอร์ทีละคำ แล้วรวมเป็น Matrix</li>
    <li>เสียง = แปลงสัญญาณเวลา → Matrix ของค่า amplitude</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">Matrix ใช้ทำอะไรในโมเดล AI</h3>
  <p className="mb-2">
    โมเดล AI เช่น Neural Networks จะรับข้อมูลทั้งหมดในรูปของ Matrix
    แล้วนำไป “คูณ” กับ Matrix ของน้ำหนัก (weights) เพื่อให้ได้ค่าที่เหมาะสม
  </p>
  <p className="mb-4">
    จากนั้นจะผ่านฟังก์ชันเช่น <code>ReLU</code> หรือ <code>Sigmoid</code> เพื่อให้ได้ผลลัพธ์ที่โมเดลเข้าใจ  
    พูดง่าย ๆ ก็คือ การเรียนรู้ของ AI = การคำนวณกับ Matrix นั่นเอง
  </p>

  <h3 className="text-xl font-semibold mb-2">Matrix ใน Python ด้วย NumPy</h3>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-3">
{`import numpy as np

# สร้าง Matrix 2x2
m1 = np.array([
  [1, 2],
  [3, 4]
])

# บวกเมทริกซ์
m2 = np.array([
  [5, 6],
  [7, 8]
])

result = m1 + m2
print(result)`}
  </pre>

  <p className="mb-2">
    การใช้ <code>np.array</code> จะช่วยให้เราสร้างและจัดการกับเมทริกซ์ได้ง่ายมาก ทั้งการบวก ลบ คูณ และอื่น ๆ
  </p>

  <h3 className="text-xl font-semibold mb-2">การแสดงภาพ Matrix</h3>
  <p className="mb-2">
    หากคุณใช้ไลบรารีอย่าง Matplotlib หรือ Seaborn คุณสามารถแปลง Matrix เป็นภาพความร้อนได้ เช่น
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-3">
{`import matplotlib.pyplot as plt
import seaborn as sns

data = np.array([
  [1, 2],
  [3, 4]
])

sns.heatmap(data, annot=True)
plt.show()`}
  </pre>

  <p className="mb-2">
    สิ่งนี้มีประโยชน์มากเมื่อต้องตรวจสอบพฤติกรรมของ Matrix ในการเรียนรู้ หรือดูการกระจายของข้อมูล
  </p>

  <h3 className="text-xl font-semibold mb-2">ข้อควรรู้เกี่ยวกับ Matrix</h3>
  <ul className="list-disc list-inside mb-2 space-y-1">
    <li>Matrix ที่มีขนาดไม่เท่ากัน ไม่สามารถบวก/ลบ กันได้</li>
    <li>Matrix ต้องมีจำนวน column ของ matrix ซ้าย = จำนวน row ของ matrix ขวา หากจะคูณกัน</li>
    <li>Matrix Transpose (<code>.T</code>) คือการสลับแถว-คอลัมน์ เช่น จาก 2x3 → 3x2</li>
  </ul>

  <p className="mt-6">
    เมื่อเข้าใจ Matrix แล้ว คุณจะเริ่มเห็นว่าทำไม AI ทุกระบบในปัจจุบัน  
    ตั้งแต่ ChatGPT ไปจนถึงระบบตรวจจับใบหน้าหรือรถยนต์ไร้คนขับ ล้วนต้องพึ่งพา Matrix ในการประมวลผล
  </p>
</section>



        <section  id="examples" className="mb-10">
          <h2 className="text-2xl font-semibold mb-3">
            ตัวอย่างการใช้งาน Vector และ Matrix ใน Python ด้วย NumPy
          </h2>
          <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
{`import numpy as np

# เวกเตอร์ 1 มิติ
v1 = np.array([2, 3])
v2 = np.array([-1, 4])

# บวกเวกเตอร์
result = v1 + v2
print("Vector Addition:", result)

# เมทริกซ์ 2 มิติ
m1 = np.array([[1, 2], [3, 4]])
print("Matrix:", m1)

# เมทริกซ์ * เมทริกซ์
m2 = np.array([[5, 6], [7, 8]])
product = m1 @ m2
print("Matrix Product:", product)`}
          </pre>
          <p className="mt-4">
            ไลบรารี NumPy ถูกพัฒนามาเพื่อให้เราจัดการข้อมูลแบบเวกเตอร์และเมทริกซ์
            ได้ง่ายและรวดเร็ว และยังรองรับการคำนวณขนาดใหญ่ที่จำเป็นในงาน AI
          </p>
        </section>

        <section id = "why-ai" className="mb-10">
  <h2 className="text-2xl font-semibold mb-3">ทำไม AI ถึงใช้ Vector และ Matrix</h2>

   {/* ✅ ภาพประกอบเวกเตอร์ */}
   <img
    src="/whyai_uses.png"
    alt="Whyai uses"
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />


  <p className="mb-2">
    ในระบบ AI ทุกวันนี้ ไม่ว่าจะเป็น ChatGPT, ระบบแปลภาษา, หรือระบบจดจำภาพ — ล้วนต้องแปลงข้อมูลให้อยู่ในรูปแบบที่คอมพิวเตอร์เข้าใจได้ ซึ่งก็คือ “ตัวเลข”
  </p>

  <p className="mb-2">
    การแปลงข้อมูลให้อยู่ในรูปของเวกเตอร์ (Vector) หรือเมทริกซ์ (Matrix)  
    ช่วยให้สามารถนำไปประมวลผลด้วยโมเดลทางคณิตศาสตร์ได้สะดวกและรวดเร็ว
  </p>

  <ul className="list-disc pl-6 space-y-3 mt-4">
    <li>
      <strong>ทุกอย่างสามารถแปลงเป็นเวกเตอร์ได้</strong>  
      <br />
      เช่น:
      <ul className="list-disc pl-6 mt-1 text-sm space-y-1">
        <li>ข้อความ → เปลี่ยนเป็นเวกเตอร์ของ token ผ่าน Embedding</li>
        <li>รูปภาพ → แปลง pixel เป็นเวกเตอร์</li>
        <li>เสียง → แปลงคลื่นเสียงเป็นเวกเตอร์เวลา/แอมปลิจูด</li>
      </ul>
    </li>

    <li>
      <strong>โมเดล AI จะรับข้อมูลในรูป Matrix</strong>  
      แล้วนำไป “คูณกับน้ำหนัก” (Weights Matrix) ที่เรียนรู้มาเพื่อให้ได้ผลลัพธ์ เช่น
      <br />
      <code className="text-sm text-yellow-300">Output = Input Matrix × Weights + Bias</code>
    </li>

    <li>
      <strong>ภาพตัวอย่าง:</strong>  
      ภาพสีขนาด 28x28 = 784 pixel  
      → แปลงเป็นเวกเตอร์ยาว 784 (1 มิติ)  
      → โมเดลจะรับเป็น Matrix ขนาด 1x784 หรือ batch 100 รูป = Matrix 100x784
    </li>

    <li>
      <strong>ข้อดี:</strong>  
      การใช้ Matrix ทำให้ AI คำนวณแบบขนานได้เร็วมาก  
      GPU/TPU ก็ออกแบบมาเพื่อคูณ Matrix โดยเฉพาะ
    </li>
  </ul>

  <p className="mt-6">
    ดังนั้น ไม่ว่าจะเป็น AI แบบใด การเข้าใจโครงสร้างของ Vector และ Matrix ถือเป็นพื้นฐานสำคัญที่จะทำให้เข้าใจการทำงานเบื้องหลังได้ดีขึ้น
  </p>
</section>

<section id= "exercise" className="mb-10">
  <h2 className="text-2xl font-semibold mb-3">แบบฝึกหัดเสริม</h2>
  <p className="mb-2">
    เพื่อเสริมความเข้าใจจากบทเรียน ลองฝึกเขียนโค้ดจริงใน Python ดังนี้:
  </p>

  <ul className="list-decimal pl-6 space-y-4 text-sm sm:text-base">
    <li>
      <strong>สร้างเวกเตอร์ 3 มิติ และหาผลลัพธ์ของการบวก/ลบ</strong>
      <pre className="bg-gray-800 text-white p-3 rounded-md overflow-x-auto mt-2">
{`import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, -1, 0])

add = v1 + v2
sub = v1 - v2

print("บวก:", add)
print("ลบ:", sub)`}
      </pre>
    </li>

    <li>
      <strong>สร้างเมทริกซ์ 3x3 แล้วคูณกับเมทริกซ์ 3x1</strong>
      <pre className="bg-gray-800 text-white p-3 rounded-md overflow-x-auto mt-2">
{`m1 = np.array([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
])

m2 = np.array([
  [1],
  [0],
  [1]
])

result = np.dot(m1, m2)
print("ผลคูณ:", result)`}
      </pre>
    </li>

    <li>
      <strong>ลองใช้ <code>np.linalg.norm</code> เพื่อหาความยาว (magnitude) ของเวกเตอร์</strong>
      <pre className="bg-gray-800 text-white p-3 rounded-md overflow-x-auto mt-2">
{`v = np.array([3, 4])
norm = np.linalg.norm(v)

print("ขนาดเวกเตอร์:", norm)  # ควรได้ 5.0 (ตามทฤษฎีปีทาโกรัส)`}
      </pre>
    </li>
  </ul>

  <p className="mt-6">
    การฝึกแบบนี้จะช่วยให้คุณคุ้นเคยกับการคิดเชิงเวกเตอร์และเมทริกซ์  
    ซึ่งเป็นพื้นฐานของทุกขั้นตอนในระบบ AI สมัยใหม่
  </p>
</section>


<section  id = "summary" className="mb-12">
  <h2 className="text-2xl font-semibold mb-3">ตารางสรุปภาพรวม</h2>
  <table className="table-auto w-full text-sm text-left border border-gray-600 text-white">
    <thead className="bg-gray-700 text-white">
      <tr>
        <th className="px-4 py-2 border">หัวข้อ</th>
        <th className="px-4 py-2 border">สิ่งที่ควรรู้</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-gray-800">
        <td className="px-4 py-2 border">Vector</td>
        <td className="px-4 py-2 border">
          ข้อมูลเชิงปริมาณที่มี “ขนาด” และ “ทิศทาง” เช่น [1, 2, 3] ใช้แทนข้อมูล 1 ชุด เช่น feature ของรูปภาพ
        </td>
      </tr>
      <tr className="bg-gray-900">
        <td className="px-4 py-2 border">Matrix</td>
        <td className="px-4 py-2 border">
          กลุ่มของเวกเตอร์หลายชุดรวมกัน เช่น [[1, 2], [3, 4]] ใช้แทนข้อมูลหลายรายการ เช่น batch ของภาพ
        </td>
      </tr>
      <tr className="bg-gray-800">
        <td className="px-4 py-2 border">Tensor</td>
        <td className="px-4 py-2 border">
          ข้อมูลหลายมิติ (เกิน 2 มิติ) เช่น 3D, 4D ใช้บ่อยใน Deep Learning เช่น [batch, channel, width, height]
        </td>
      </tr>
      <tr className="bg-gray-900">
        <td className="px-4 py-2 border">NumPy</td>
        <td className="px-4 py-2 border">
          ไลบรารีใน Python ที่ใช้สร้างและคำนวณเวกเตอร์/เมทริกซ์ เช่น บวก ลบ คูณ หาร หา norm
        </td>
      </tr>
      <tr className="bg-gray-800">
        <td className="px-4 py-2 border">Dot Product</td>
        <td className="px-4 py-2 border">
          การคูณเวกเตอร์ 2 ตัวเพื่อหา scalar ที่แสดงถึงความคล้ายคลึง ใช้ในการวัดมุมหรือ similarity
        </td>
      </tr>
      <tr className="bg-gray-900">
        <td className="px-4 py-2 border">AI Model</td>
        <td className="px-4 py-2 border">
          โมเดล AI จะรับ input เป็น matrix แล้วคูณกับ weights (matrix) และ bias เพื่อคำนวณผลลัพธ์
        </td>
      </tr>
      <tr className="bg-gray-800">
        <td className="px-4 py-2 border">GPU Acceleration</td>
        <td className="px-4 py-2 border">
          การใช้ GPU ช่วยคำนวณ matrix ได้เร็วขึ้นหลายเท่า เช่น Matrix Multiplication ขนาดใหญ่ใน Neural Network
        </td>
      </tr>
      <tr className="bg-gray-900">
        <td className="px-4 py-2 border">Batch Processing</td>
        <td className="px-4 py-2 border">
          การประมวลผลข้อมูลหลายชุดพร้อมกัน โดยจัดข้อมูลเป็น Matrix ขนาดใหญ่ (batch_size x features)
        </td>
      </tr>
      <tr className="bg-gray-800">
        <td className="px-4 py-2 border">Embedding</td>
        <td className="px-4 py-2 border">
          การแปลงข้อความ (เช่น คำหรือประโยค) ให้อยู่ในรูปเวกเตอร์ที่มีความหมายในเชิงคณิตศาสตร์
        </td>
      </tr>
    </tbody>
  </table>
</section>
      <section id="quiz" className="mb-10">
        <MiniQuiz_Day1 theme={theme} />
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


      {/* ✅ ScrollSpy แบบลอยขวา ไม่ดัน content */}
      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day1 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day1_VectorMatrix;