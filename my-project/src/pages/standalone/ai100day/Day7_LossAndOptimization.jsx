import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day7 from "./scrollspy/ScrollSpy_Ai_Day7";
import MiniQuiz_Day7 from "./miniquiz/MiniQuiz_Day7";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";


const Day7_LossOptimization = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } }); // ใส่ชื่อ cloud ของคุณ
  const img1 = cld
  .image('LossFunction1')
  .format('auto')
  .quality('auto')
  .resize(scale().width(700));

  const img2 = cld
  .image('LossFunction2')
  .format('auto')
  .quality('auto')
  .resize(scale().width(700));

  const img3 = cld
  .image('LossFunction3')
  .format('auto')
  .quality('auto')
  .resize(scale().width(700));

  const img4 = cld
  .image('LossFunction4')
  .format('auto')
  .quality('auto')
  .resize(scale().width(700));

  return (
    <div
      className={`relative min-h-screen ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
      }`}
    >
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 7: Loss Functions & Optimization</h1>

        {/* Section: What is Loss Function */}
        <section id="what-is-loss" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">Loss Function คืออะไร?</h2>

  <div className="my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <p className="mb-4 text-base leading-relaxed">
    Loss Function คือเครื่องมือที่ใช้วัดว่าโมเดลของเราทำนายได้ใกล้เคียงกับค่าจริงมากน้อยแค่ไหน — ถ้าทำนายได้ถูกต้อง Loss จะต่ำ แต่ถ้าทำนายผิดมาก Loss จะสูง ซึ่งช่วยให้เรารู้ว่าโมเดลต้องปรับปรุงหรือไม่ และมากแค่ไหน
  </p>

  <p className="mb-4 text-base leading-relaxed">
    เปรียบเทียบง่าย ๆ: <strong>Loss คือคะแนนสอบ</strong> ของโมเดลในแต่ละรอบของการฝึก (Epoch) ถ้าคะแนนต่ำ → แสดงว่าทำข้อสอบถูกเยอะ ถ้าคะแนนสูง → ยังเข้าใจไม่พอ ต้องเรียนรู้ต่อ
  </p>

  <p className="mb-4 text-base leading-relaxed">
    การใช้ Loss ไม่ได้เป็นเพียงแค่การวัดผลเท่านั้น แต่ยังนำไปใช้ในกระบวนการ <strong>Backpropagation</strong> เพื่อคำนวณ Gradient และปรับค่าพารามิเตอร์ในโมเดล ทำให้โมเดลฉลาดขึ้นในแต่ละรอบการเรียนรู้
  </p>

  <div className="grid md:grid-cols-2 gap-6 my-8">
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow border border-yellow-400">
      <h3 className="text-lg font-semibold mb-2">ตัวอย่าง Loss สำหรับ Classification</h3>
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>Binary Cross Entropy: ใช้กับปัญหาที่มี 2 คลาส เช่น แมว vs สุนัข</li>
        <li>Categorical Cross Entropy: ใช้กับปัญหาที่มีหลายคลาส เช่น 10 ตัวเลข</li>
        <li>Loss จะสูงมากเมื่อโมเดลมั่นใจผิด</li>
      </ul>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow border border-blue-400">
      <h3 className="text-lg font-semibold mb-2">ตัวอย่าง Loss สำหรับ Regression</h3>
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>Mean Squared Error (MSE): วัดค่าความต่างกำลังสองของค่าจริงกับค่าทำนาย</li>
        <li>Mean Absolute Error (MAE): วัดค่าความต่างแบบสัมบูรณ์</li>
        <li>MSE เน้นความผิดพลาดใหญ่, MAE ทนต่อ Outlier ได้ดีกว่า</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3">เปรียบเทียบ Loss แบบต่าง ๆ</h3>
  <div className="overflow-x-auto">
    <table className="w-full text-sm border border-yellow-500">
      <thead className="bg-yellow-100 dark:bg-yellow-800">
        <tr>
          <th className="p-3 border border-yellow-400">ประเภท</th>
          <th className="p-3 border border-yellow-400">Loss Function</th>
          <th className="p-3 border border-yellow-400">ใช้กับ</th>
          <th className="p-3 border border-yellow-400">ข้อดี</th>
          <th className="p-3 border border-yellow-400">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-3 border border-yellow-400">Classification</td>
          <td className="p-3 border border-yellow-400">Binary Cross Entropy</td>
          <td className="p-3 border border-yellow-400">2 คลาส (0 หรือ 1)</td>
          <td className="p-3 border border-yellow-400">เข้าใจง่าย, คำนวณรวดเร็ว</td>
          <td className="p-3 border border-yellow-400">ไวต่อค่าผิดแบบมั่นใจผิด</td>
        </tr>
        <tr>
          <td className="p-3 border border-yellow-400">Classification</td>
          <td className="p-3 border border-yellow-400">Categorical Cross Entropy</td>
          <td className="p-3 border border-yellow-400">หลายคลาส</td>
          <td className="p-3 border border-yellow-400">ใช้ได้กับ Softmax output</td>
          <td className="p-3 border border-yellow-400">ต้องจัดการ One-hot encoding</td>
        </tr>
        <tr>
          <td className="p-3 border border-yellow-400">Regression</td>
          <td className="p-3 border border-yellow-400">Mean Squared Error (MSE)</td>
          <td className="p-3 border border-yellow-400">ค่าต่อเนื่อง เช่น ราคา, น้ำหนัก</td>
          <td className="p-3 border border-yellow-400">ลงโทษความผิดพลาดรุนแรง</td>
          <td className="p-3 border border-yellow-400">ไวต่อ Outlier</td>
        </tr>
        <tr>
          <td className="p-3 border border-yellow-400">Regression</td>
          <td className="p-3 border border-yellow-400">Mean Absolute Error (MAE)</td>
          <td className="p-3 border border-yellow-400">ค่าต่อเนื่อง</td>
          <td className="p-3 border border-yellow-400">ทนต่อ Outlier ได้ดีกว่า MSE</td>
          <td className="p-3 border border-yellow-400">Gradient อาจไม่ smooth</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">บทสรุปแนวคิด</h3>
  <div className="my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <p className="mb-4 text-base">
    Loss Function คือเข็มทิศของ Neural Network ที่บอกว่าเรากำลังไปถูกทางหรือไม่ ถ้า loss ยังสูง → แสดงว่าเรายัง “ห่างจากคำตอบที่ถูกต้อง” อยู่มาก → ต้องให้ Optimizer ช่วยปรับเส้นทาง
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    ถ้าไม่มี Loss Function → โมเดลจะไม่รู้เลยว่าคำตอบของตัวเองดีหรือไม่ดีแค่ไหน
    Loss เปรียบเสมือน "คะแนนสอบ" และเป็นตัวกำหนดทิศทางของการเรียนรู้ทั้งหมดในระบบ AI
  </div>
</section>


<section id="loss-types" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">ประเภทของ Loss Function</h2>

  <p className="mb-4 text-base">
    ในการเทรนโมเดล Deep Learning เราจำเป็นต้องเลือก Loss Function ให้เหมาะกับประเภทของงาน เพราะ Loss แต่ละชนิดจะมีคุณสมบัติ และวิธีคำนวณที่แตกต่างกัน ซึ่งส่งผลโดยตรงต่อความสามารถในการเรียนรู้ของโมเดล
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mb-6">
    <strong>Insight:</strong><br />
    การเลือก Loss ที่เหมาะสมเหมือนกับการวัดผลด้วยไม้บรรทัดที่ถูกชนิด — ถ้าวัดผิด ก็ปรับโมเดลผิดไปตลอด
  </div>

  <h3 className="text-xl font-semibold mb-2 text-center">Loss สำหรับ Classification</h3>
  <div className="my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>Binary Cross Entropy:</strong> ใช้กับปัญหา Binary Classification เช่น ทายว่าอีเมลเป็น spam หรือไม่</li>
    <li><strong>Categorical Cross Entropy:</strong> ใช้กับปัญหาที่มีหลายคลาส เช่น จำแนกภาพหมา-แมว-นก โดยมี label แบบ one-hot</li>
    <li><strong>Sparse Categorical Cross Entropy:</strong> คล้ายกับ categorical แต่ label เป็นเลขแทน one-hot (เหมาะกับข้อมูลใหญ่)</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2 text-center"> Loss สำหรับ Regression</h3>
  <div className="my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>Mean Squared Error (MSE):</strong> คำนวณค่าความคลาดเคลื่อนโดยยกกำลังสอง — ชอบลงโทษค่าที่ผิดเยอะ</li>
    <li><strong>Mean Absolute Error (MAE):</strong> คำนวณค่าเฉลี่ยของความต่างแบบสัมบูรณ์ — มีความทนต่อ outlier</li>
    <li><strong>Huber Loss:</strong> ผสมระหว่าง MSE กับ MAE เพื่อให้ได้ผลลัพธ์ที่บาลานซ์</li>
  </ul>

  <h3 className="text-xl font-semibold mb-4"> เปรียบเทียบ Loss Function</h3>
  <div className="overflow-x-auto">
    <table className="w-full table-auto text-sm border border-yellow-500">
      <thead className="bg-yellow-100 dark:bg-yellow-800">
        <tr>
          <th className="p-2 border border-yellow-400 text-left">Loss Function</th>
          <th className="p-2 border border-yellow-400 text-left">เหมาะกับงาน</th>
          <th className="p-2 border border-yellow-400 text-left">ลักษณะเด่น</th>
          <th className="p-2 border border-yellow-400 text-left">ข้อเสีย</th>
        </tr>
      </thead>
      <tbody className="bg-white dark:bg-gray-900 text-black dark:text-white">
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">Binary Cross Entropy</td>
          <td className="p-2 border border-yellow-400">Binary Classification</td>
          <td className="p-2 border border-yellow-400">ให้ probabilistic output</td>
          <td className="p-2 border border-yellow-400">ไม่เหมาะกับ multiclass</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">Categorical Cross Entropy</td>
          <td className="p-2 border border-yellow-400">Multiclass Classification</td>
          <td className="p-2 border border-yellow-400">ใช้ร่วมกับ softmax ได้ดี</td>
          <td className="p-2 border border-yellow-400">ต้องแปลง label เป็น one-hot</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">MSE</td>
          <td className="p-2 border border-yellow-400">Regression</td>
          <td className="p-2 border border-yellow-400">ไวต่อความผิดพลาดขนาดใหญ่</td>
          <td className="p-2 border border-yellow-400">ไม่ทนต่อ outliers</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">MAE</td>
          <td className="p-2 border border-yellow-400">Regression</td>
          <td className="p-2 border border-yellow-400">ทนต่อ outlier ได้ดี</td>
          <td className="p-2 border border-yellow-400">Gradient คงที่ ทำให้ช้ากว่า MSE</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400 font-semibold">Huber</td>
          <td className="p-2 border border-yellow-400">Regression (robust)</td>
          <td className="p-2 border border-yellow-400">สมดุลระหว่าง MSE และ MAE</td>
          <td className="p-2 border border-yellow-400">ต้องเลือกพารามิเตอร์ δ ให้ดี</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-2"> หมายเหตุการเลือกใช้งาน</h3>
  <ul className="list-disc pl-6 space-y-2 text-base">
    <li>งานที่มี label เป็น category → ใช้ Cross Entropy</li>
    <li>งานที่ค่าตอบเป็นตัวเลข → ใช้ MSE หรือ MAE</li>
    <li>ถ้าเจอข้อมูลเบี้ยว (outlier เยอะ) → ให้ลอง Huber</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 mt-6 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    Loss คือกระจกสะท้อนว่าโมเดลเข้าใจข้อมูลดีแค่ไหน — ยิ่ง Loss ที่เลือกตรงกับงาน ยิ่งสะท้อน “ความผิด” ได้ชัดเจน และช่วยให้ Optimizer ทำงานแม่นยำขึ้น
  </div>
</section>


<section id="what-is-optimization" className="mb-16 scroll-mt-32">
          <h2 className="text-2xl font-semibold mb-4">Optimization คืออะไร?</h2>

          <p className="mb-4 text-base leading-relaxed">
            Optimization คือกระบวนการสำคัญในกระบวนการเรียนรู้ของโมเดล ที่ใช้ปรับค่าน้ำหนัก (weights) ภายในโครงข่ายประสาทเทียม เพื่อให้โมเดลสามารถทำนายได้แม่นยำมากขึ้น โดยอาศัยแนวคิดพื้นฐานคือการ <strong>ลดค่า Loss</strong> ให้ต่ำที่สุดผ่านการเรียนรู้หลายรอบ (epoch)
          </p>

          <p className="mb-4 text-base leading-relaxed">
            หลักการทำงานของ Optimization คล้ายกับการ "ไต่ลงเขา" ในภูมิประเทศที่ซับซ้อน (Loss Landscape) จุดมุ่งหมายคือการหาจุดต่ำสุดที่แท้จริง (global minimum) หรืออย่างน้อยก็ใกล้เคียงที่สุด (local minimum ที่ดี) ซึ่งหมายถึงโมเดลมีความผิดพลาดต่ำที่สุด
          </p>

          <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mb-6">
            <strong>Insight:</strong><br />
            Loss คือระดับความผิดพลาด ส่วน Optimizer คือวิธีการขยับโมเดลไปในทิศทางที่ดีขึ้น — หากไม่มี Optimizer โมเดลจะไม่สามารถเรียนรู้หรือพัฒนาได้เลย
          </div>

          <h3 className="text-xl font-semibold mb-3"> การทำงานของ Optimizer</h3>
          <ul className="list-disc pl-6 space-y-2 mb-6 text-base">
            <li>เริ่มจากการคำนวณ Loss ว่าโมเดลผิดพลาดแค่ไหน</li>
            <li>ใช้เทคนิค Backpropagation เพื่อส่ง Gradient ย้อนกลับไปยังเลเยอร์ก่อนหน้า</li>
            <li>Optimizer จะใช้ Gradient ที่ได้มา ปรับค่าน้ำหนักให้เหมาะสมขึ้น</li>
            <li>ทำซ้ำกระบวนการนี้ทุก epoch → โมเดลจะค่อย ๆ ดีขึ้น</li>
          </ul>

          <h3 className="text-xl font-semibold mb-3"> ภาพจำลองการไต่ลงเขา</h3>
          <img src="/loss_landscape.png" alt="Loss Landscape" className="w-full max-w-xl mx-auto rounded shadow border border-yellow-400 mb-4" />
          <p className="text-center text-sm text-gray-600 dark:text-gray-400">เส้นทางจากจุดเริ่มต้นสู่จุดต่ำสุดของ Loss</p>

          <h3 className="text-xl font-semibold mt-6 mb-3"> ทำไม Optimization ถึงสำคัญ?</h3>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>โมเดลจะไม่สามารถเรียนรู้จากข้อมูลได้เลยหากไม่มีการอัปเดตน้ำหนัก</li>
            <li>Optimizer ที่ดีจะช่วยให้โมเดล converge เร็วขึ้น และได้ผลลัพธ์ที่แม่นยำกว่า</li>
            <li>สามารถหลีกเลี่ยงจุดต่ำสุดปลอม ๆ (local minima) หรือปัญหา gradient หายได้</li>
          </ul>

          <h3 className="text-xl font-semibold mt-6 mb-3"> ประเภทของ Optimization ที่พบบ่อย</h3>
          <table className="w-full text-sm border border-yellow-500 mb-6">
            <thead className="bg-yellow-100 dark:bg-yellow-800">
              <tr>
                <th className="p-3 border">ชื่อ</th>
                <th className="p-3 border">หลักการ</th>
                <th className="p-3 border">ข้อดี</th>
                <th className="p-3 border">ข้อจำกัด</th>
              </tr>
            </thead>
            <tbody>
              <tr className="bg-white dark:bg-gray-700">
                <td className="p-3 border">Vanilla Gradient Descent</td>
                <td className="p-3 border">ปรับทุก batch พร้อมกัน</td>
                <td className="p-3 border">เข้าใจง่าย</td>
                <td className="p-3 border">ช้า, ไม่เหมาะกับ dataset ใหญ่</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800">
                <td className="p-3 border">SGD</td>
                <td className="p-3 border">ปรับครั้งละตัวอย่าง (stochastic)</td>
                <td className="p-3 border">เร็ว, หลีกเลี่ยง local minima</td>
                <td className="p-3 border">ผลลัพธ์ไม่เสถียร</td>
              </tr>
              <tr className="bg-white dark:bg-gray-700">
                <td className="p-3 border">Momentum</td>
                <td className="p-3 border">ใช้ทิศทางสะสมจากรอบก่อน</td>
                <td className="p-3 border">ลดการแกว่ง, เร็วขึ้น</td>
                <td className="p-3 border">ตั้งค่าพารามิเตอร์ยาก</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800">
                <td className="p-3 border">RMSProp</td>
                <td className="p-3 border">ลด learning rate ของพารามิเตอร์ที่มี gradient สูง</td>
                <td className="p-3 border">เสถียร, ดีใน RNN</td>
                <td className="p-3 border">อ่อนไหวกับค่าพารามิเตอร์</td>
              </tr>
              <tr className="bg-white dark:bg-gray-700">
                <td className="p-3 border">Adam</td>
                <td className="p-3 border">รวม Momentum + RMSProp</td>
                <td className="p-3 border">ดีที่สุดในหลายกรณี</td>
                <td className="p-3 border">อาจ overfit, ใช้ resource มาก</td>
              </tr>
            </tbody>
          </table>

          <h3 className="text-xl font-semibold mt-6 mb-3"> คำแนะนำในการเลือก Optimizer</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>เริ่มจาก Adam → ดีที่สุดสำหรับโมเดลทั่วไป</li>
            <li>ใช้ SGD ถ้าต้องการควบคุมรายละเอียดมากขึ้น</li>
            <li>ลอง RMSProp ถ้าใช้กับ RNN หรือ sequence</li>
          </ul>

          <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 mt-6 rounded-xl text-sm border-l-4 border-yellow-500 shadow">
            <strong>Insight:</strong><br />
            การเลือก Optimizer ที่เหมาะสมเปรียบเสมือนการเลือก "วิธีเดินเขา" ที่เร็วและมั่นคง — ยิ่งเลือกดี โมเดลยิ่งไปถึงเป้าหมายไวและไม่หลงทาง
          </div>
        </section>


        <section id="gradient-descent" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">Gradient Descent & Learning Rate</h2>

  <p className="mb-4">
    Gradient Descent คือกระบวนการพื้นฐานที่สุดในการอัปเดตค่าน้ำหนักของโมเดล โดยใช้ค่าความชัน (gradient) ของฟังก์ชัน loss เพื่อบอกทิศทางในการปรับค่าพารามิเตอร์ให้ loss ลดลง
  </p>

  <p className="mb-4">
    ถ้านึกภาพ loss เป็น "ภูเขา" Gradient Descent ก็เหมือนกับการไต่ลงเขาไปหาจุดต่ำสุด โดยการก้าวลงแต่ละครั้งคือการปรับค่า weight ของโมเดลไปในทิศทางที่ลดค่า loss
  </p>

  <img src="/gradient_descent_landscape.png" alt="Gradient Descent Visual" className="rounded-xl shadow border border-yellow-500 mx-auto my-6" />

  <h3 className="text-xl font-semibold mt-6 mb-2">Learning Rate คืออะไร?</h3>
  <p className="mb-4">
    Learning Rate (LR) คือความแรงในการปรับน้ำหนักในแต่ละครั้ง หากค่า LR สูงเกินไปจะทำให้กระโดดข้ามจุดต่ำสุด แต่ถ้าต่ำเกินไปจะทำให้เรียนรู้ช้าและอาจติดอยู่ที่ local minimum
  </p>

  <div className="grid md:grid-cols-3 gap-6 mb-6">
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow">
      <h4 className="font-semibold mb-2">🔹 Learning Rate ต่ำ</h4>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>Loss ลดช้า</li>
        <li>อาจติดอยู่ที่ local minimum</li>
        <li>ใช้เวลา train นาน</li>
      </ul>
    </div>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow">
      <h4 className="font-semibold mb-2"> Learning Rate เหมาะสม</h4>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>Loss ลดได้เร็วและต่อเนื่อง</li>
        <li>โมเดล converge ได้ดี</li>
      </ul>
    </div>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl shadow">
      <h4 className="font-semibold mb-2"> Learning Rate สูง</h4>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>Loss แกว่งไปมา</li>
        <li>โมเดลไม่เสถียร</li>
        <li>อาจเกิด NaN</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-4">เปรียบเทียบ Algorithm Optimization</h3>
  <div className="overflow-x-auto">
    <table className="w-full table-auto text-sm border border-yellow-500">
      <thead className="bg-yellow-100 dark:bg-yellow-900">
        <tr>
          <th className="p-2 border border-yellow-400">Optimizer</th>
          <th className="p-2 border border-yellow-400">หลักการ</th>
          <th className="p-2 border border-yellow-400">ข้อดี</th>
          <th className="p-2 border border-yellow-400">ข้อเสีย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border border-yellow-400">Vanilla GD</td>
          <td className="p-2 border border-yellow-400">ใช้ข้อมูลทั้งหมด</td>
          <td className="p-2 border border-yellow-400">แม่นยำ, เสถียร</td>
          <td className="p-2 border border-yellow-400">ช้า, ไม่เหมาะกับ dataset ใหญ่</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400">SGD</td>
          <td className="p-2 border border-yellow-400">สุ่มบางส่วน</td>
          <td className="p-2 border border-yellow-400">เร็ว, ช่วยหลุดจาก local min</td>
          <td className="p-2 border border-yellow-400">แกว่งแรง, ไม่เสถียร</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400">Momentum</td>
          <td className="p-2 border border-yellow-400">จดจำทิศทางเก่า</td>
          <td className="p-2 border border-yellow-400">เรียนรู้เร็วขึ้น</td>
          <td className="p-2 border border-yellow-400">จูนยากกว่า</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400">RMSProp</td>
          <td className="p-2 border border-yellow-400">ลด LR สำหรับค่าที่ gradient สูง</td>
          <td className="p-2 border border-yellow-400">เหมาะกับ RNN</td>
          <td className="p-2 border border-yellow-400">ต้อง normalize ดี</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400">Adam</td>
          <td className="p-2 border border-yellow-400">Momentum + RMSProp</td>
          <td className="p-2 border border-yellow-400">ใช้งานง่าย, เสถียร, เร็ว</td>
          <td className="p-2 border border-yellow-400">บางกรณี converge ช้า</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div className="bg-yellow-100 dark:bg-yellow-800 text-black dark:text-yellow-100 p-4 mt-6 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    Gradient Descent คือการรู้ "ทิศทาง" ส่วน Learning Rate คือ "จังหวะ" — ทั้งสองอย่างร่วมกันจะกำหนดว่าเราจะถึงจุดหมายได้เร็วหรือหลงทางไปเรื่อย ๆ
  </div>
</section>

<section id="gradient-problems" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">Vanishing & Exploding Gradients คืออะไร?</h2>

  <p className="mb-4 text-base leading-relaxed">
    ปัญหานี้มักเกิดขึ้นในโมเดล Neural Network ที่มีความลึกมาก (Deep Neural Network)
    โดยเฉพาะเมื่อใช้ Activation Function แบบ sigmoid หรือ tanh ซึ่งค่าความชัน (Gradient) ที่คำนวณ
    กลับมาจากเลเยอร์หลัง ๆ อาจค่อย ๆ <strong>ลดลงจนใกล้ 0</strong> (เรียกว่า Vanishing) หรือ <strong>เพิ่มขึ้นจนใหญ่มาก</strong> (เรียกว่า Exploding)
    ส่งผลให้การเรียนรู้ไม่มีประสิทธิภาพ
  </p>

  <p className="mb-4 text-base">
    ในการฝึกโมเดล เราจะใช้ Backpropagation เพื่อส่งค่าความผิดพลาด (Error) จากเลเยอร์ปลายทางกลับไปยังต้นทาง
    ซึ่งค่าที่ส่งกลับนี้จะถูกคูณต่อกันหลายครั้งตามจำนวนเลเยอร์ → ทำให้ค่า Gradient
    <strong>ยิ่งน้อยลงเรื่อย ๆ หรือพุ่งสูงขึ้นแบบควบคุมไม่ได้</strong>
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow mb-6">
    <strong>ตัวอย่างง่าย ๆ:</strong> ถ้า Gradient ที่เลเยอร์นึง = 0.9 และมี 100 เลเยอร์ → Gradient สุดท้ายจะกลายเป็น 0.9^100 ≈ 0.00003 (เกือบ 0)
  </div>

  <h3 className="text-xl font-semibold mb-2">Vanishing Gradient</h3>
  <p className="mb-4 text-base">
    คือปรากฏการณ์ที่ Gradient มีค่าน้อยมากเมื่อย้อนกลับไปยังเลเยอร์ต้น ๆ ทำให้โมเดลไม่สามารถอัปเดตพารามิเตอร์ในเลเยอร์นั้นได้เลย
    เกิดบ่อยใน Activation Function ที่มี output อยู่ในช่วงแคบ เช่น sigmoid หรือ tanh ซึ่งทำให้ Gradient ติดค่าน้อยและหายไป
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>โมเดลไม่สามารถเรียนรู้จากข้อมูลได้ลึก ๆ</li>
    <li>Training Loss อาจไม่ลดเลย</li>
    <li>เลเยอร์ต้น ๆ จะกลายเป็นเหมือนเลเยอร์ "ตาย"</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">Exploding Gradient</h3>
  <p className="mb-4 text-base">
    คือปรากฏการณ์ตรงข้ามกับ Vanishing — เมื่อค่า Gradient ถูกคูณกันหลายครั้งแล้วพุ่งสูงขึ้นมากจน Overflow
    โมเดลจะไม่สามารถเรียนรู้ได้เลย หรือ weight จะมีค่าใหญ่เกินไปจนอาจเกิด NaN หรือ Inf
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>ทำให้ Loss กลายเป็น NaN</li>
    <li>โมเดลเทรนแล้วผลลัพธ์ไม่นิ่ง (unstable)</li>
    <li>โมเดลอาจดูเหมือนเรียนรู้เร็ว แต่สุดท้ายผลลัพธ์แย่</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">ผลกระทบต่อการเทรนโมเดล</h3>
  <p className="mb-4 text-base">
    ทั้งสองปัญหานี้ทำให้โมเดลเทรนไม่ได้ผล หรือเรียนรู้ได้ช้ามาก
    ในบางกรณีโมเดลอาจ overfit หรือเทรนได้ผลไม่เสถียร เพราะค่าที่ไหลย้อนกลับมากผิดธรรมชาติ
  </p>

  <h3 className="text-xl font-semibold mb-2">แนวทางการป้องกัน</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ใช้ Activation Function ที่ไม่ทำให้ Gradient หาย เช่น ReLU แทน sigmoid</li>
    <li>ใช้ Weight Initialization ให้เหมาะสม เช่น Xavier หรือ He Initialization</li>
    <li>ใช้เทคนิค Batch Normalization เพื่อทำให้ค่ากระจายของ Gradient เสถียร</li>
    <li>ใช้ Residual Connection (เช่นใน ResNet) เพื่อให้ Gradient ไหลย้อนกลับได้ตรงขึ้น</li>
    <li>ใช้ Gradient Clipping เพื่อจำกัดไม่ให้ Gradient ใหญ่เกินไป</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-800 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    ปัญหา Vanishing & Exploding Gradients คือความท้าทายที่ผลักดันให้เกิดการพัฒนาโครงสร้างโมเดลที่ดีขึ้น เช่น LSTM, GRU, ResNet
    และแนวคิดเหล่านี้คือรากฐานของ Deep Learning ยุคใหม่ที่ช่วยให้โมเดลลึกสามารถเรียนรู้ได้จริง
  </div>
</section>




<section id="insight" className="mb-20 scroll-mt-32">
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow-xl">
    <h2 className="text-xl font-semibold mb-4">
      Insight: <span className="italic">Loss = เข็มทิศ, Optimizer = การเดินทาง</span>
    </h2>

    <p className="mb-4">
      การฝึกโมเดล AI ไม่ต่างจากการพานักเรียนคนหนึ่งไปให้ถึงเป้าหมายการเรียนรู้ การมีแค่ข้อมูล (input/output) อย่างเดียวไม่พอ — เราต้องมีระบบที่บอกด้วยว่าเด็กคนนี้ “ตอบผิดตรงไหน” และ “ควรปรับตัวอย่างไร”
    </p>

    <div className="bg-white/10 border border-yellow-300 p-4 rounded-xl mb-6">
      <p className="font-semibold mb-2 text-yellow-300"> การวนรอบของโมเดล (Training Loop)</p>
      <ul className="list-disc pl-6 space-y-1 text-sm">
        <li>ทำนาย → เปรียบเทียบกับความจริง → คำนวณ Loss</li>
        <li>ใช้ Optimizer ในการอัปเดตพารามิเตอร์ให้ดีขึ้น</li>
        <li>ทำซ้ำวนไปเรื่อย ๆ จน loss ต่ำที่สุดเท่าที่ทำได้</li>
      </ul>
    </div>

    <h3 className="text-lg font-semibold mb-3"> อุปมา: เดินป่า & เข็มทิศ</h3>
    <p className="mb-4">
      ลองจินตนาการว่าโมเดลของเราคือ “นักเดินป่า” ที่กำลังพยายามเดินทางไปยังจุดหมายซึ่งอยู่ในหุบเขา (จุดที่ Loss ต่ำที่สุด)
    </p>
    <ul className="list-disc pl-6 space-y-1 mb-6 text-sm">
      <li><strong>Loss Function</strong> = บอกระยะทางและทิศทางว่าอยู่ห่างจากเป้าหมายแค่ไหน</li>
      <li><strong>Gradient</strong> = บอกว่าควรเคลื่อนที่ไปทางไหน</li>
      <li><strong>Optimizer</strong> = วิธีการเดิน เช่น วิ่งลงเนิน เดินทีละก้าว กระโดดลง หรือเดินวนหาเส้นทางที่สั้นที่สุด</li>
    </ul>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl text-sm border-l-4 border-yellow-500 mb-6">
      <p className="font-semibold mb-2"> Insight เปรียบเทียบ:</p>
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <h4 className="text-base font-medium mb-2"> Loss</h4>
          <ul className="list-disc pl-6 space-y-1">
            <li>ใช้ประเมินคุณภาพของการทำนาย</li>
            <li>เป็นค่าตัวเลขที่ใช้สำหรับวัดว่า “โมเดลพลาดแค่ไหน”</li>
            <li>ช่วยบอกให้โมเดล “รู้ตัวว่าผิด” เพื่อแก้ไข</li>
          </ul>
        </div>
        <div>
          <h4 className="text-base font-medium mb-2"> Optimizer</h4>
          <ul className="list-disc pl-6 space-y-1">
            <li>ใช้กำหนดวิธีในการปรับพารามิเตอร์</li>
            <li>กำหนดความเร็ว ทิศทาง และแนวโน้มของการเรียนรู้</li>
            <li>สามารถเลือกวิธีการเดินที่เหมาะกับภูมิประเทศ (loss landscape)</li>
          </ul>
        </div>
      </div>
    </div>

    <h3 className="text-lg font-semibold mb-3"> ตัวอย่างจากโลกจริง</h3>
    <p className="mb-2">พิจารณาโมเดลที่ใช้ทำนายราคาอสังหาริมทรัพย์:</p>
    <ul className="list-disc pl-6 space-y-1 text-sm mb-6">
      <li>หากโมเดลทำนายราคาบ้านผิด 3 แสนบาท — <strong>Loss = 300,000</strong></li>
      <li>Gradient จะคำนวณว่าส่วนไหนของพารามิเตอร์ทำให้โมเดลพลาด</li>
      <li>Optimizer จะใช้ข้อมูลนี้เพื่อปรับ weight ให้ทำนายแม่นขึ้นในรอบถัดไป</li>
    </ul>

    <h3 className="text-lg font-semibold mb-3"> หากไม่มี Loss หรือ Optimizer จะเกิดอะไรขึ้น?</h3>
    <ul className="list-disc pl-6 space-y-2 mb-6 text-sm">
      <li>ไม่มี Loss → โมเดลไม่รู้ว่าทำผิด</li>
      <li>ไม่มี Optimizer → โมเดลไม่รู้ว่าจะปรับตัวอย่างไร</li>
      <li>ไม่มีทั้งสองอย่าง → โมเดลเรียนรู้ไม่ได้เลย</li>
    </ul>

    <h3 className="text-lg font-semibold mb-3"> สรุปแนวคิด</h3>
    <p className="mb-4">
      การฝึกโมเดลให้ฉลาดไม่ใช่แค่การมีข้อมูลเยอะหรือโมเดลลึกเท่านั้น แต่ต้องมี “กลไกการเรียนรู้” ที่ดี ซึ่ง Loss และ Optimizer คือหัวใจสำคัญของการเรียนรู้นั้น
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-800 p-4 rounded-xl border-l-4 border-yellow-500">
      <p className="italic text-sm text-gray-700 dark:text-yellow-100">
        "Loss คือเสียงสะท้อนว่าเราพลาดตรงไหน — ส่วน Optimizer คือการตัดสินใจว่าควรเปลี่ยนแปลงอะไรบ้างเพื่อไม่ให้พลาดซ้ำอีก"
      </p>
    </div>
  </div>
</section>


        <section id="quiz" className="mb-16 scroll-mt-32">
          <MiniQuiz_Day7 theme={theme} />
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
        <ScrollSpy_Ai_Day7 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day7_LossOptimization;
