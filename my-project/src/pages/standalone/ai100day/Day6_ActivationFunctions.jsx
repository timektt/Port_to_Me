import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day6 from "./scrollspy/ScrollSpy_Ai_Day6";
import MiniQuiz_Day6 from "./miniquiz/MiniQuiz_Day6";

const Day6_ActivationFunctions = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div
      className={`relative min-h-screen ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
      }`}
    >
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 6: Activation Functions</h1>

        <section id="why-not-linear" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">ทำไม Linear อย่างเดียวไม่พอ?</h2>

  <img
    src="/ActivationFunctions1.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4">
    ถ้าโมเดลใช้แต่เลเยอร์แบบ Linear ต่อกัน แม้จะหลายชั้น แต่สุดท้ายจะยังเทียบเท่ากับการคูณเมทริกซ์เพียงครั้งเดียว (Linear of Linear = Linear) → นั่นคือข้อจำกัดที่ทำให้โมเดลไม่สามารถเข้าใจรูปแบบที่ซับซ้อนหรือไม่เป็นเส้นตรงได้เลย
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-200 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mb-6">
    <strong>Insight:</strong> ถ้าไม่มี Activation Function → Neural Network ก็เป็นเพียงการคูณเมทริกซ์แบบยาว ๆ ที่ไม่มีพลังในการ “เข้าใจ” ข้อมูลในรูปแบบที่ซับซ้อน
  </div>

  <p className="mb-4">
    ลองพิจารณาปัญหาคลาสสิกอย่าง <strong>XOR</strong> ซึ่งไม่สามารถแยกแยะได้ด้วยเส้นตรง (linear boundary):
  </p>

  <pre className="bg-gray-800 text-white p-4 text-sm rounded-md overflow-x-auto mb-4"># XOR Problem
# Input: [0,0] → 0
#        [0,1] → 1
#        [1,0] → 1
#        [1,1] → 0
# ไม่มีเส้นตรงใดที่สามารถแยก output = 1 ออกจาก output = 0 ได้
  </pre>

  <p className="mb-4">
    ปัญหานี้ต้องการโมเดลที่สามารถ “งอเส้น” หรือ “โค้งเส้นเขตแดน” ได้ ซึ่งเป็นไปไม่ได้เลยถ้าใช้แต่ Linear Layer ต่อกันโดยไม่มีความไม่เป็นเชิงเส้น (non-linearity) เข้ามาแทรกแซง
  </p>

  <div className="grid sm:grid-cols-2 gap-6 my-8">
    <div>
      <h3 className="text-lg font-semibold mb-2"> Linear-only Model</h3>
      <ul className="list-disc pl-6 text-sm space-y-1">
        <li>ไม่มีการงอเส้น decision boundary</li>
        <li>เรียนรู้ pattern ง่าย ๆ เท่านั้น</li>
        <li>ไม่สามารถแก้ปัญหาอย่าง XOR, ภาพซับซ้อน, ภาษาธรรมชาติได้</li>
      </ul>
    </div>
    <div>
      <h3 className="text-lg font-semibold mb-2"> Linear + Activation</h3>
      <ul className="list-disc pl-6 text-sm space-y-1">
        <li>สามารถงอ decision boundary ได้</li>
        <li>เข้าใจ pattern ที่ซับซ้อนได้มากขึ้น</li>
        <li>ทำให้โมเดลมี “ชีวิต” และสามารถตัดสินใจได้แบบฉลาด</li>
      </ul>
    </div>
  </div>

  <p className="mb-4">
    การใส่ Activation Function ระหว่าง Linear Layers เป็นการทำให้ Neural Network กลายเป็น <strong>Universal Function Approximator</strong> กล่าวคือสามารถประมาณค่าได้แทบทุกฟังก์ชันในธรรมชาติ ทั้งโค้งเว้า, คลื่น, หรือสัญญาณที่ซับซ้อน
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2 text-center">เปรียบเทียบ: ก่อนและหลังมี Activation</h3>

  <img
    src="/ActivationFunctions2.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />
  <div className="grid sm:grid-cols-2 gap-6 items-center">
    <div>
      <p className="text-sm mb-2"> ไม่มี Activation (Linear Only):</p>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>แปลงเวกเตอร์ แต่ไม่เปลี่ยนพฤติกรรมของฟังก์ชัน</li>
        <li>Decision Boundary ยังเป็นเส้นตรง</li>
        <li>Performance ต่ำกับปัญหาที่ซับซ้อน</li>
      </ul>
    </div>
    <div>
      <p className="text-sm mb-2"> มี Activation (Linear + ReLU):</p>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>สร้าง decision boundary ที่โค้ง</li>
        <li>เรียนรู้ Feature ลึกหลายระดับ</li>
        <li>ใช้งานได้จริงในโมเดล AI ยุคใหม่ เช่น CNN, RNN, Transformer</li>
      </ul>
    </div>
  </div>


  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การแปลงเวกเตอร์แบบ linear เปรียบเหมือนเลนส์ธรรมดา แต่ Activation เปรียบเสมือนเลนส์ที่ปรับโฟกัสได้เองตามสิ่งที่เห็น — นี่คือเหตุผลที่ Neural Network ต้องมีมัน
  </div>
</section>


<section id="what-is-activation" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Activation Function คืออะไร?</h2>

  <img
    src="/ActivationFunctions3.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4 text-base leading-relaxed">
    Activation Function คือฟังก์ชันที่ถูกแทรกไว้หลังการทำ Linear Transformation ในแต่ละเลเยอร์ของ Neural Network เพื่อสร้างความไม่เป็นเส้นตรง (Non-linearity) ให้กับระบบ ซึ่งถือเป็นหัวใจสำคัญของการทำให้โมเดลสามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนในข้อมูลได้
  </p>

  <p className="mb-4 text-base leading-relaxed">
    หากไม่มี Activation Function โมเดลที่มีหลายเลเยอร์แบบ Linear จะยังคงสามารถถูกรวมเป็นเพียงการคูณเมทริกซ์เดียว (Linear of Linear = Linear) นั่นหมายถึงไม่ว่าจะซับซ้อนแค่ไหน สุดท้ายโมเดลจะไม่สามารถจำแนกข้อมูลที่มีความซับซ้อนทางโครงสร้าง เช่น ข้อมูลที่ไม่สามารถแยกได้ด้วยเส้นตรง (non-linearly separable) อย่าง XOR หรือกลุ่มข้อมูลที่มีความโค้งงอหรือหลายมิติได้เลย
  </p>

  <h3 className="text-xl font-semibold mb-3">ตัวอย่างเปรียบเทียบ (XOR Problem)</h3>


  <p className="mb-4">
    ปัญหา XOR เป็นตัวอย่างคลาสสิกที่ Linear Model ไม่สามารถเรียนรู้ได้ เพราะจุดข้อมูลในลักษณะนี้ไม่สามารถแบ่งได้ด้วยเส้นตรงเพียงเส้นเดียว ต้องมีความโค้งงอหรือการตัดกันที่จำเป็นต้องใช้ความไม่เป็นเส้นตรงเข้ามาช่วย ซึ่ง Activation Function ทำหน้าที่ตรงนี้
  </p>

  <img
    src="/ActivationFunctions4.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />


  <h3 className="text-xl font-semibold mb-3">หน้าที่ของ Activation Function</h3>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>เพิ่มความไม่เป็นเชิงเส้น:</strong> ทำให้โมเดลเรียนรู้ฟังก์ชันที่ซับซ้อนได้</li>
    <li><strong>เลือกข้อมูล:</strong> กรองข้อมูลบางส่วน เช่น ทำให้ค่าติดลบกลายเป็นศูนย์ (ReLU)</li>
    <li><strong>จำกัดขอบเขต:</strong> ทำให้ค่าที่ได้ไม่เกินช่วงที่กำหนด เช่น -1 ถึง 1 (Tanh)</li>
    <li><strong>สร้างความต่อเนื่อง:</strong> ช่วยให้การเรียนรู้ด้วย Gradient Descent ดำเนินไปได้อย่างราบรื่น</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">เปรียบเทียบ Activation Function ที่ใช้บ่อย</h3>
  <div className="overflow-x-auto mb-6">
    <table className="w-full table-auto text-sm border border-yellow-500">
      <thead>
        <tr className="bg-yellow-100 dark:bg-yellow-900 text-left">
          <th className="p-2 border border-yellow-400">ชื่อ</th>
          <th className="p-2 border border-yellow-400">นิยาม</th>
          <th className="p-2 border border-yellow-400">ช่วงค่า</th>
          <th className="p-2 border border-yellow-400">จุดเด่น</th>
          <th className="p-2 border border-yellow-400">ข้อเสีย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">ReLU</td>
          <td className="p-2 border border-yellow-400">f(x) = max(0, x)</td>
          <td className="p-2 border border-yellow-400">[0, ∞)</td>
          <td className="p-2 border border-yellow-400">ง่าย, คำนวณเร็ว, ช่วยลด vanishing gradient</td>
          <td className="p-2 border border-yellow-400">เกิด dead neuron ได้เมื่อค่าเข้า &lt; 0</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">Sigmoid</td>
          <td className="p-2 border border-yellow-400">f(x) = 1 / (1 + e^(-x))</td>
          <td className="p-2 border border-yellow-400">(0, 1)</td>
          <td className="p-2 border border-yellow-400">เหมาะกับ binary output, เข้าใจง่าย</td>
          <td className="p-2 border border-yellow-400">gradient หายเมื่อค่าเข้าสูง/ต่ำเกินไป</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">Tanh</td>
          <td className="p-2 border border-yellow-400">f(x) = (e^x - e^(-x)) / (e^x + e^(-x))</td>
          <td className="p-2 border border-yellow-400">[-1, 1]</td>
          <td className="p-2 border border-yellow-400">มีศูนย์กลางที่ 0, gradient ดีกว่า sigmoid</td>
          <td className="p-2 border border-yellow-400">ยังเจอปัญหา vanishing gradient ได้</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-3">Insight สำคัญ</h3>
  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
    Activation Function เปรียบเสมือน "จิตวิญญาณ" ที่ทำให้โมเดลมีความสามารถในการแยกแยะและเข้าใจข้อมูลที่ซับซ้อนได้ หากไม่มีมัน โมเดลจะเป็นเพียงสมการเชิงเส้นธรรมดาที่ไม่สามารถเรียนรู้สิ่งใหม่จากข้อมูลโลกจริงได้เลย
  </div>

</section>


<section id="compare-activations" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">เปรียบเทียบ ReLU, Sigmoid, Tanh</h2>

  <p className="mb-6 text-base">
    Activation Function มีหลายรูปแบบ แต่ละแบบมีจุดแข็ง จุดอ่อน และเหมาะกับบริบทที่แตกต่างกัน
    ด้านล่างนี้คือการเปรียบเทียบอย่างละเอียดของ ReLU, Sigmoid และ Tanh
    เพื่อให้เข้าใจความแตกต่างทั้งในเชิงคณิตศาสตร์และการใช้งานจริง
  </p>

  <div className="grid md:grid-cols-3 gap-6">
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow border border-yellow-400">
      <h3 className="text-lg font-semibold mb-2">🔹 ReLU (Rectified Linear Unit)</h3>
      <p className="text-sm mb-2">f(x) = max(0, x)</p>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>คำนวณเร็วมากและไม่ซับซ้อน</li>
        <li>เหมาะสำหรับโมเดลลึก (Deep Neural Networks)</li>
        <li>ลดปัญหา vanishing gradient ได้ดี</li>
        <li>มีปัญหา Dead Neuron — ค่าติดลบทั้งหมดกลายเป็น 0</li>
        <li>ไม่มีขีดจำกัดด้านบน ทำให้สามารถขยายค่าได้ดีในทิศทางบวก</li>
      </ul>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow border border-blue-400">
      <h3 className="text-lg font-semibold mb-2">🔹 Sigmoid</h3>
      <p className="text-sm mb-2">f(x) = 1 / (1 + e^-x)</p>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>แปลงค่าให้อยู่ในช่วง 0 ถึง 1</li>
        <li>เหมาะกับงาน classification ที่มี 2 classes (binary)</li>
        <li>มี interpretability สูง (เข้าใจง่าย)</li>
        <li>เกิดปัญหา vanishing gradient ได้ง่ายเมื่อค่าเข้าอยู่ไกลจาก 0</li>
        <li>การเรียนรู้ช้าในเลเยอร์ลึก</li>
      </ul>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow border border-pink-400">
      <h3 className="text-lg font-semibold mb-2">🔹 Tanh (Hyperbolic Tangent)</h3>
      <p className="text-sm mb-2">f(x) = (e^x - e^-x) / (e^x + e^-x)</p>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>แปลงค่าให้อยู่ในช่วง -1 ถึง 1</li>
        <li>สมดุลดีกว่า sigmoid เพราะ centered ที่ 0</li>
        <li>เหมาะกับ input ที่ normalize แล้ว</li>
        <li>ยังมีปัญหา vanishing gradient เหมือน sigmoid</li>
        <li>ใช้งานได้ดีในบางโมเดล recurrent เช่น RNN</li>
      </ul>
    </div>
  </div>

  <div className="mt-10">
  <h3 className="text-xl font-semibold mb-3">ตารางเปรียบเทียบแบบสรุป</h3>

  <div className="overflow-x-auto">
    <table className="min-w-[600px] text-sm border-collapse border">
      <thead>
        <tr className="bg-yellow-200 text-black">
          <th className="border px-3 py-2 text-left">Activation</th>
          <th className="border px-3 py-2 text-left">Range</th>
          <th className="border px-3 py-2 text-left">ใช้กับ</th>
          <th className="border px-3 py-2 text-left">ข้อดี</th>
          <th className="border px-3 py-2 text-left">ข้อเสีย</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-700">
          <td className="border px-3 py-2">ReLU</td>
          <td className="border px-3 py-2">[0, ∞)</td>
          <td className="border px-3 py-2">CNN, DNN</td>
          <td className="border px-3 py-2">เร็ว, ลด gradient หาย</td>
          <td className="border px-3 py-2">Dead Neuron</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-800">
          <td className="border px-3 py-2">Sigmoid</td>
          <td className="border px-3 py-2">[0, 1]</td>
          <td className="border px-3 py-2">Binary Output</td>
          <td className="border px-3 py-2">เข้าใจง่าย</td>
          <td className="border px-3 py-2">Vanishing Gradient</td>
        </tr>
        <tr className="bg-white dark:bg-gray-700">
          <td className="border px-3 py-2">Tanh</td>
          <td className="border px-3 py-2">[-1, 1]</td>
          <td className="border px-3 py-2">RNN, NLP</td>
          <td className="border px-3 py-2">Centered ที่ 0</td>
          <td className="border px-3 py-2">ยังมี gradient หาย</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>


  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 mt-8 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    ไม่มี Activation ตัวไหนที่ดีที่สุดในทุกสถานการณ์ — การเลือกใช้ควรพิจารณาจากประเภทข้อมูล, ความลึกของโมเดล, และปัญหาเฉพาะทาง
  </div>
</section>

<section id="modern-activations" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">Activation Function สมัยใหม่: Leaky ReLU, GELU, Swish</h2>

  <p className="mb-4">
    แม้ว่า ReLU จะเป็น Activation Function ยอดนิยมในหลายปีที่ผ่านมา แต่ก็มีข้อจำกัด เช่น ปัญหา <strong>Dead Neuron</strong> ที่เกิดจากค่าลบถูกตัดเป็นศูนย์เสมอ ทำให้เกิดการเรียนรู้ที่ไม่สมบูรณ์ในบางกรณี จึงได้มีการพัฒนา Activation Function แบบใหม่ ๆ เพื่อแก้ปัญหาเหล่านี้ และเพิ่มความสามารถในการเรียนรู้ที่ลึกและละเอียดมากขึ้นในโมเดลสมัยใหม่ เช่น BERT และ GPT
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-800 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow mb-6">
    <strong>Insight:</strong><br />
    Activation Function ใหม่ไม่เพียงเพิ่มความยืดหยุ่นให้โมเดล แต่ยังช่วยให้ Gradient ไหลได้ดีขึ้น ลดการหายของ Gradient ในโมเดลลึก ๆ ได้อย่างชัดเจน
  </div>

  <h3 className="text-xl font-semibold mb-3"> Leaky ReLU</h3>
  <p className="mb-2">
    Leaky ReLU คือเวอร์ชันที่พัฒนาเพิ่มเติมจาก ReLU โดยไม่ตัดค่าลบออกหมด แต่ให้ค่าลบมี slope เล็กน้อย เช่น 0.01
  </p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`f(x) = x  if x > 0
     = αx if x <= 0  (โดยทั่วไป α = 0.01)`}
  </pre>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ลดโอกาสเกิด Dead Neuron</li>
    <li>ยังคงความเรียบง่ายของ ReLU</li>
    <li>นิยมใช้ใน GANs หรือโมเดลภาพ</li>
  </ul>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-6">
{`# PyTorch
import torch.nn as nn

layer = nn.Sequential(
  nn.Linear(64, 32),
  nn.LeakyReLU(negative_slope=0.01)
)`}
  </pre>

  <h3 className="text-xl font-semibold mb-3"> GELU (Gaussian Error Linear Unit)</h3>
  <p className="mb-2">
    GELU ถูกใช้ใน BERT และ Transformer-based models หลายตัว โดยมีลักษณะเป็นการคูณ x ด้วย sigmoid-like function ซึ่งให้ความ <strong>smooth</strong> และต่อเนื่องมากกว่า ReLU หรือ Tanh
  </p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`f(x) = 0.5 * x * (1 + tanh(\u221a(2/\u03c0)*(x + 0.044715x^3)))`}
  </pre>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ให้ gradient ที่ลื่นไหล (smooth)</li>
    <li>ไม่มีปัญหา dead zone เหมือน ReLU</li>
    <li>เหมาะกับโมเดลภาษาและ sequence model</li>
  </ul>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-6">
{`# PyTorch
import torch.nn as nn
import torch.nn.functional as F

class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        return F.gelu(self.linear(x))`}
  </pre>

  <h3 className="text-xl font-semibold mb-3">Swish</h3>
  <p className="mb-2">
    Swish ถูกเสนอโดย Google เป็น Activation Function ที่มีลักษณะเหมือน x คูณกับ sigmoid(x) ให้ผลลัพธ์ที่ไม่จำกัดด้านลบและมี gradient ลื่นไหล
  </p>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">
{`f(x) = x * sigmoid(x)`}
  </pre>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ให้ gradient ที่ไหลดีในทุกช่วง</li>
    <li>สามารถให้ค่าติดลบเล็กน้อยได้ (แตกต่างจาก ReLU)</li>
    <li>ใช้ใน EfficientNet และโมเดลที่ต้องการ balance ดีระหว่าง speed และ accuracy</li>
  </ul>
  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-6">
{`# TensorFlow
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.activations import swish

x = Dense(64)(input_tensor)
x = Activation(swish)(x)`}
  </pre>

  <h3 className="text-xl font-semibold mb-4"> เปรียบเทียบระหว่าง ReLU vs LeakyReLU vs GELU vs Swish</h3>
  <div className="overflow-x-auto">
    <table className="min-w-full text-sm text-left border border-gray-300 dark:border-gray-600">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="p-2 border">Function</th>
          <th className="p-2 border">ลักษณะเด่น</th>
          <th className="p-2 border">ข้อดี</th>
          <th className="p-2 border">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border">ReLU</td>
          <td className="p-2 border">ตัดค่าติดลบเป็น 0</td>
          <td className="p-2 border">เร็ว, คำนวณง่าย</td>
          <td className="p-2 border">Dead Neuron</td>
        </tr>
        <tr>
          <td className="p-2 border">Leaky ReLU</td>
          <td className="p-2 border">ปล่อยค่าติดลบเล็กน้อย</td>
          <td className="p-2 border">ลด Dead Neuron</td>
          <td className="p-2 border">มีพารามิเตอร์ที่ต้องกำหนด (α)</td>
        </tr>
        <tr>
          <td className="p-2 border">GELU</td>
          <td className="p-2 border">smooth non-linearity</td>
          <td className="p-2 border">ให้ผลการเรียนรู้ดีในโมเดล NLP</td>
          <td className="p-2 border">คำนวณซับซ้อน</td>
        </tr>
        <tr>
          <td className="p-2 border">Swish</td>
          <td className="p-2 border">x * sigmoid(x)</td>
          <td className="p-2 border">เหมาะกับโมเดลลึก & EfficientNet</td>
          <td className="p-2 border">ช้าเล็กน้อยกว่า ReLU</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<section id="activation-gradient-flow" className="mb-20 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    ผลของ Activation Function ต่อ Gradient Flow
  </h2>
  <img
    src="/ActivationFunctions5.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4 text-base">
    ในการฝึกโมเดล Neural Network โดยเฉพาะโมเดลที่ลึกมาก (Deep Neural Network) การส่งผ่านค่า Gradient ย้อนกลับในแต่ละเลเยอร์คือหัวใจของการเรียนรู้ แต่ในหลายกรณี ค่า Gradient อาจ <strong>หายไป (Vanishing Gradient)</strong> หรือ <strong>ระเบิด (Exploding Gradient)</strong> ขึ้นอยู่กับลักษณะของ Activation Function ที่เลือกใช้
  </p>
  <h3 className="text-xl font-semibold mt-6 mb-2"> ปัญหา Vanishing Gradient คืออะไร?</h3>
  <p className="mb-4">
    เมื่อค่า Gradient มีขนาดเล็กมากใกล้ศูนย์ โดยเฉพาะถ้า Activation Function มีช่วงที่ slope หรืออนุพันธ์เข้าใกล้ 0 → จะทำให้เลเยอร์ต้น ๆ ของโมเดล <strong>ไม่สามารถอัปเดตน้ำหนักได้</strong> อย่างมีประสิทธิภาพ ส่งผลให้โมเดลเรียนรู้ได้ช้ามากหรือไม่เรียนรู้เลย
  </p>

  <h3 className="text-xl font-semibold mb-2"> ปัญหา Exploding Gradient คืออะไร?</h3>
  <p className="mb-4">
    ในขณะเดียวกัน หากค่า Gradient ใหญ่มากขึ้นเรื่อย ๆ เมื่อย้อนกลับผ่านเลเยอร์หลายชั้น → อาจทำให้ weight เปลี่ยนแปลงรุนแรง จนโมเดลไม่เสถียร หรือ loss กลายเป็น NaN ได้
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3"> ผลของ Activation Function ต่อ Gradient</h3>
  <table className="table-auto w-full text-sm text-left border mb-6">
    <thead>
      <tr className="bg-yellow-200 dark:bg-yellow-900">
        <th className="border px-3 py-2">Activation</th>
        <th className="border px-3 py-2">ลักษณะ Gradient</th>
        <th className="border px-3 py-2">ปัญหา</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-3 py-2">Sigmoid</td>
        <td className="border px-3 py-2">Gradient ใกล้ 0 เมื่อ x → ±∞</td>
        <td className="border px-3 py-2">Vanishing Gradient</td>
      </tr>
      <tr>
        <td className="border px-3 py-2">Tanh</td>
        <td className="border px-3 py-2">เหมือน Sigmoid แต่มีช่วงกว้างขึ้น</td>
        <td className="border px-3 py-2">Vanishing (แต่ดีกว่า Sigmoid)</td>
      </tr>
      <tr>
        <td className="border px-3 py-2">ReLU</td>
        <td className="border px-3 py-2">Gradient = 1 เมื่อ x &gt; 0</td>
        <td className="border px-3 py-2">Dead Neuron (x &lt; 0)</td>
      </tr>
      <tr>
        <td className="border px-3 py-2">Leaky ReLU</td>
        <td className="border px-3 py-2">Gradient มีค่าต่ำสุด = 0.01</td>
        <td className="border px-3 py-2">ลด Dead Neuron</td>
      </tr>
      <tr>
        <td className="border px-3 py-2">GELU</td>
        <td className="border px-3 py-2">Smooth และมี slope ที่ดี</td>
        <td className="border px-3 py-2">รักษา Gradient ได้ดี</td>
      </tr>
    </tbody>
  </table>


  <h3 className="text-xl font-semibold mt-6 mb-3"> ทางเลือกเพื่อแก้ปัญหา Vanishing/Exploding</h3>
  <ul className="list-disc pl-6 space-y-2">
    <li>เลือก Activation ที่มี slope ชัดเจน เช่น ReLU, Leaky ReLU, GELU</li>
    <li>ใช้ Weight Initialization ที่เหมาะสม (เช่น Xavier หรือ He init)</li>
    <li>ใช้ Batch Normalization เพื่อรักษาช่วงของค่าไว้ไม่ให้เบี่ยงเบน</li>
    <li>เลือก Learning Rate อย่างพอเหมาะ</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 mt-8 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    การไหลของ Gradient เปรียบเหมือนกระแสเลือดของโมเดล หาก Activation Function ไม่เหมาะสม จะทำให้เลือดไหลไม่ทั่วร่างกาย ส่งผลให้โมเดลเรียนรู้ช้า หรือหยุดเรียนรู้ในที่สุด
  </div>
</section>



<section id="activation-in-code" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">ตัวอย่างโค้ด: การใช้ Activation ในโมเดลจริง</h2>

  <p className="mb-4 text-base">
    ในโมเดล AI การวาง Activation Function อย่างถูกต้องในแต่ละเลเยอร์มีผลอย่างมากต่อการเรียนรู้
    โดยโค้ดด้านล่างจะเปรียบเทียบวิธีการเขียนใน PyTorch และ TensorFlow (Keras)
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-2"> PyTorch</h3>
  <p className="mb-2">
    ใน PyTorch จะใช้ <code>nn.Sequential</code> เพื่อจัดวางลำดับเลเยอร์ โดยใส่ Activation แบบแยกเลเยอร์ชัดเจน เช่น ReLU, Sigmoid หรือ Tanh
  </p>

  <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto mb-4">
{`import torch
import torch.nn as nn

model = nn.Sequential(
  nn.Linear(4, 8),   # Linear layer 1
  nn.ReLU(),         # Activation: ReLU
  nn.Linear(8, 4),   # Linear layer 2
  nn.Tanh(),         # Activation: Tanh
  nn.Linear(4, 1),   # Linear layer 3
  nn.Sigmoid()       # Activation: Sigmoid (สำหรับ output ระดับ 0-1)
)

x = torch.randn(1, 4)
output = model(x)
print(output)`}
  </pre>

  <p className="mb-4">
    จุดเด่นคือความยืดหยุ่น สามารถเปลี่ยน activation ได้ในแต่ละเลเยอร์
    หรือแม้แต่สร้าง activation ของตัวเองได้ เช่น LeakyReLU หรือ Custom Function
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-2"> TensorFlow (Keras)</h3>
  <p className="mb-2">
    ใน Keras สามารถใช้แบบแยกชั้น <code>Activation()</code> หรือแบบใส่ชื่อ activation ตรงใน <code>Dense(..., activation="relu")</code> ก็ได้
  </p>

  <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto mb-4">
{`from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation

# แบบแยกเลเยอร์ Activation ออกมา
model = Sequential([
  Dense(8, input_shape=(4,)),
  Activation('relu'),
  Dense(4),
  Activation('tanh'),
  Dense(1),
  Activation('sigmoid')
])`}
  </pre>

  <p className="mb-4">หรือจะเขียนแบบรวมในบรรทัดเดียว:</p>

  <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto mb-4">
{`# แบบรวม activation ใน Dense
model = Sequential([
  Dense(8, activation='relu', input_shape=(4,)),
  Dense(4, activation='tanh'),
  Dense(1, activation='sigmoid')
])`}
  </pre>

  <p className="mb-4">
    Keras ใช้งานง่าย เหมาะกับการสร้างโมเดลแบบรวดเร็ว และมี activation function ให้เลือกหลากหลาย เช่น Softmax, Swish, GELU
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การใส่ Activation Function อย่างเหมาะสมระหว่างเลเยอร์คือหัวใจของความฉลาดในโมเดล
    หากไม่มี activation → ไม่ว่าจะกี่เลเยอร์ ก็ยังเป็นแค่การแปลงเชิงเส้นธรรมดา
  </div>

  <p className="mt-6">
     ต่อไปในหัวข้อถัดไป เราจะดู <strong>Insight เปรียบเทียบ</strong> ว่า Activation Function เปลี่ยน “ชีวิตของโมเดล” ได้อย่างไรในเชิงแนวคิด
  </p>
</section>




{/* วาง Section นี้ตรงนี้ก่อน Mini Quiz */}
<section id="activation-selection" className="mb-20 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-6 text-center">การเลือก Activation Function ตามประเภทของงาน</h2>
  <img
    src="/ActivationFunctions6.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />


  <p className="mb-4 text-base leading-relaxed">
    การเลือกใช้ Activation Function ไม่มีสูตรตายตัว แต่ขึ้นอยู่กับลักษณะของปัญหา (task) และข้อมูลที่ใช้งาน  
    หากเลือกผิด อาจทำให้โมเดลเรียนรู้ได้ช้า หรือติดอยู่กับค่าที่ไม่สามารถพัฒนาได้ (เช่น Dead Neuron หรือ Vanishing Gradient)
  </p>

  <p className="mb-4">
    ตารางด้านล่างสรุปการเลือกใช้ Activation Function ตามประเภทของงานที่พบบ่อย:
  </p>

  <div className="overflow-x-auto mb-6">
    <table className="w-full text-left border border-gray-400 text-sm rounded overflow-hidden">
      <thead className="bg-gray-200 dark:bg-gray-700 text-black dark:text-white">
        <tr>
          <th className="p-3 border border-gray-400">ประเภทงาน</th>
          <th className="p-3 border border-gray-400">Activation ที่แนะนำ</th>
          <th className="p-3 border border-gray-400">เหตุผล</th>
        </tr>
      </thead>
      <tbody className="bg-white dark:bg-gray-800 text-black dark:text-white">
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">Classification (Binary)</td>
          <td className="p-3 border border-gray-400">Sigmoid</td>
          <td className="p-3 border border-gray-400">แปลง output ให้อยู่ระหว่าง 0–1 สำหรับตัดสินคลาส</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">Classification (Multiclass)</td>
          <td className="p-3 border border-gray-400">Softmax</td>
          <td className="p-3 border border-gray-400">แปลงเป็นความน่าจะเป็นรวม = 1 ใช้สำหรับเลือกคลาสที่น่าจะเป็นไปได้ที่สุด</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">Regression</td>
          <td className="p-3 border border-gray-400">Linear / None</td>
          <td className="p-3 border border-gray-400">ไม่ควรบีบค่า output เพราะค่าอาจอยู่นอกขอบเขตจำกัด</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">NLP (Transformer, BERT, GPT)</td>
          <td className="p-3 border border-gray-400">GELU / Swish</td>
          <td className="p-3 border border-gray-400">ให้ gradient ที่ smooth และเสถียร เหมาะกับโมเดลลึก</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">Vision (CNN, ImageNet)</td>
          <td className="p-3 border border-gray-400">ReLU / Leaky ReLU</td>
          <td className="p-3 border border-gray-400">รวดเร็ว ใช้งานง่าย แก้ปัญหา Dead Neuron ได้ในบางกรณี</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">GAN Generator</td>
          <td className="p-3 border border-gray-400">ReLU / Tanh</td>
          <td className="p-3 border border-gray-400">ReLU ทำให้ข้อมูล sparse ส่วน Tanh เหมาะกับการ normalize output</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">GAN Discriminator</td>
          <td className="p-3 border border-gray-400">Leaky ReLU</td>
          <td className="p-3 border border-gray-400">ช่วยให้ gradient ไม่หายขณะ training ด้วยข้อมูลที่หลากหลาย</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 font-semibold">Autoencoder</td>
          <td className="p-3 border border-gray-400">Tanh / ReLU</td>
          <td className="p-3 border border-gray-400">Tanh เหมาะกับค่าที่ normalize แล้ว ส่วน ReLU เหมาะกับภาพหรือเสียง</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-lg font-semibold mt-8 mb-2">คำแนะนำเพิ่มเติม</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4 text-base">
    <li>ไม่จำเป็นต้องใช้ Activation เดียวทุกเลเยอร์ — บางโมเดลใช้ ReLU ชั้นแรก และใช้ Swish ในชั้นลึก</li>
    <li>บางงานทดลองใช้ GELU แทน ReLU แล้วได้ผลดีกว่า แม้จะช้ากว่าเล็กน้อย</li>
    <li>ลองทดสอบหลาย Activation เพื่อเลือกที่เหมาะสมกับ task จริง ไม่ควรยึดติดแบบใดแบบหนึ่ง</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Activation Function ไม่ใช่เพียง “ฟังก์ชันทางคณิตศาสตร์” แต่คือปัจจัยกำหนด "ภาษาที่โมเดลจะเข้าใจข้อมูล"  
    เลือกให้เหมาะ = เปิดศักยภาพโมเดลได้เต็มที่
  </div>
</section>

<section id="insight-activation" className="mb-20 scroll-mt-32">
  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow-xl">
    <h2 className="text-xl font-semibold mb-4">Insight: Activation = "ความมีชีวิต" ของโมเดล</h2>

    <p className="mb-4">
      หากไม่มี Activation Function โมเดลจะเป็นเพียงเครื่องคูณเมทริกซ์ซ้ำ ๆ หลายชั้น โดยผลลัพธ์ยังอยู่ในโลกเชิงเส้น ซึ่งไม่สามารถแก้ปัญหาที่มีลักษณะไม่เป็นเส้นตรง เช่น การแยกกลุ่มของข้อมูลที่ซับซ้อน การรู้จำลายมือ หรือการเข้าใจภาษามนุษย์ได้เลย
    </p>

    <p className="mb-4">
      Activation Function ทำหน้าที่คล้ายกับ "ประตูทางเลือก" ในสมอง ที่ช่วยให้โมเดลเปิดหรือปิดการตอบสนองต่อข้อมูลบางส่วน ทำให้ระบบสามารถเรียนรู้ลักษณะที่ซ่อนอยู่ และปรับการตอบสนองในแต่ละบริบทได้
    </p>

    <p className="mb-4">
      เปรียบเทียบง่าย ๆ หากไม่มี Activation → โมเดลจะเหมือนหุ่นยนต์ที่คิดตามสูตรอย่างเดียว แต่เมื่อใส่ Activation → โมเดลจะมีความยืดหยุ่น คิดเป็น เรียนรู้ลึกขึ้น และสามารถตัดสินใจได้อย่างชาญฉลาดมากขึ้น
    </p>

    <p className="mb-4">
      ยกตัวอย่างเช่น ถ้ามีเพียง Linear Layer 10 ชั้นเรียงกัน จะสามารถรวมเป็น Matrix เดียวได้ ซึ่งก็คือ Linear อีกชั้นหนึ่ง นั่นหมายความว่าความลึกที่แท้จริงไม่มีผลเลยหากขาด Non-linearity ซึ่ง Activation เป็นตัวทำให้โมเดลลึกมีพลัง
    </p>

    <p className="mb-4">
      Activation ยังช่วยให้เกิดการ "บีบหรือขยาย" ช่วงของข้อมูล เช่น ReLU จะตัดค่าติดลบออกเพื่อเน้นเฉพาะสัญญาณที่สำคัญ หรือ Sigmoid จะบีบค่าทุกอย่างให้อยู่ในช่วง 0 ถึง 1 ทำให้เหมาะสำหรับงาน Classification
    </p>

    <p className="mb-4">
      อย่างไรก็ตาม ไม่มี Activation Function แบบเดียวที่ดีที่สุดในทุกกรณี → การเลือกใช้ควรขึ้นอยู่กับลักษณะของปัญหา เช่น ถ้าเป็นภาพ ใช้ ReLU มักให้ผลดีเพราะเร็วและไม่มีค่า Saturation, ถ้าเป็นปัญหาต้องการค่า probability → Sigmoid จะเหมาะสมกว่า
    </p>

    <p className="mb-4">
      โมเดลที่ไม่มี Activation เปรียบได้กับเส้นตรงที่ไม่มีทางเลี้ยว ไม่มีความสามารถในการแยกแยะข้อมูลที่ซับซ้อน ในขณะที่โมเดลที่มี Activation สามารถวาดเส้นโค้ง แยกกลุ่ม สร้างเส้นแบ่งใหม่ที่ซับซ้อนได้
    </p>

    <p className="mb-4">
      นอกจากนี้ Activation ยังเกี่ยวข้องโดยตรงกับการเกิด Gradient → ฟังก์ชันที่มี Slope แปรผันต่อ input จะช่วยให้โมเดลสามารถไหลย้อนกลับ (Backpropagation) ได้ดี ส่งผลต่อการเรียนรู้โดยรวมของโมเดล
    </p>

    <p className="mb-4">
      ในงานวิจัยและโลกของ Deep Learning มีการพัฒนา Activation แบบใหม่ ๆ ขึ้นเรื่อย ๆ เช่น Leaky ReLU, GELU, Swish ซึ่งพยายามแก้ปัญหาเดิม เช่น Dead Neuron หรือ Gradient Vanishing เพื่อให้โมเดลเรียนรู้ได้เร็วและลึกขึ้น
    </p>

    <p className="mb-4 italic text-gray-700 dark:text-gray-400">
      "Activation คือชีพจรของสมองกล — หากไม่มีมัน AI จะเป็นเพียงเครื่องคำนวณที่ไม่มีชีวิตจิตใจ"
    </p>
  </div>
</section>


<section id="read-more" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">แหล่งเรียนรู้เพิ่มเติม</h2>

  <p className="mb-4">
    หากคุณต้องการศึกษาเรื่อง Activation Functions อย่างเจาะลึกมากขึ้น
    หรือดูการใช้งานจริงในโมเดลสมัยใหม่ เช่น BERT, GPT หรือ CNN ต่อไปนี้คือแหล่งอ้างอิงที่แนะนำ:
  </p>

  <ul className="list-disc pl-6 space-y-3 text-base">
    <li>
       <a href="https://cs231n.github.io/neural-networks-1/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">CS231n - Activation Functions</a><br />
      แหล่งอ้างอิงคลาสสิกจาก Stanford University อธิบายเรื่อง activation ได้เข้าใจง่าย พร้อมภาพและตัวอย่างการใช้งานจริง
    </li>

    <li>
       <a href="https://arxiv.org/abs/1606.08415" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">Swish: A Self-Gated Activation Function</a><br />
      งานวิจัยจาก Google ที่เสนอ Swish ซึ่งให้ผลลัพธ์ดีกว่า ReLU ในหลายงาน
    </li>

    <li>
       <a href="https://huggingface.co/transformers/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">HuggingFace Transformers</a><br />
      ดูตัวอย่างโค้ดจริงของโมเดล Transformer และ Activation Function ที่ใช้ใน BERT, GPT เป็นต้น
    </li>

    <li>
       <a href="https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">PyTorch Activation Docs</a><br />
      เอกสารอย่างเป็นทางการของ PyTorch สำหรับ Activation Layers เช่น ReLU, GELU, Tanh
    </li>

    <li>
       <a href="https://keras.io/api/layers/activations/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">Keras Activation Functions</a><br />
      เอกสารทางการของ TensorFlow/Keras ที่รวม activation functions ทั้งหมดที่ใช้ได้
    </li>
  </ul>

</section>


        <section id="quiz" className="mb-16 scroll-mt-32">
          <MiniQuiz_Day6 theme={theme} />
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
        <ScrollSpy_Ai_Day6 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day6_ActivationFunctions;
