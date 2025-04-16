import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day8 from "./scrollspy/ScrollSpy_Ai_Day8";
import MiniQuiz_Day8 from "./miniquiz/MiniQuiz_Day8";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";

const Day8_Backpropagation = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  const img1 = cld.image('Backpropagation1').format('auto').quality('auto').resize(scale().width(700));
  const img2 = cld.image('Backpropagation2').format('auto').quality('auto').resize(scale().width(500));
  const img3 = cld.image('Backpropagation3').format('auto').quality('auto').resize(scale().width(500));
  const img4 = cld.image('Backpropagation4').format('auto').quality('auto').resize(scale().width(700));
  const img5 = cld.image('Backpropagation5').format('auto').quality('auto').resize(scale().width(500));
  const img6 = cld.image('Backpropagation6').format('auto').quality('auto').resize(scale().width(700));
  const img7 = cld.image('Backpropagation7').format('auto').quality('auto').resize(scale().width(700));
  const img8 = cld.image('Backpropagation8').format('auto').quality('auto').resize(scale().width(500));
  const img9 = cld.image('Backpropagation9').format('auto').quality('auto').resize(scale().width(500));
  const img10 = cld.image('Backpropagation10').format('auto').quality('auto').resize(scale().width(500));
  const img11 = cld.image('Backpropagation11').format('auto').quality('auto').resize(scale().width(600));
  const img12 = cld.image('Backpropagation12').format('auto').quality('auto').resize(scale().width(500));
  const img13= cld.image('Backpropagation13').format('auto').quality('auto').resize(scale().width(500));
  const img14= cld.image('Backpropagation14').format('auto').quality('auto').resize(scale().width(500));
  const img15= cld.image('Backpropagation15').format('auto').quality('auto').resize(scale().width(500));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 8: Backpropagation & Training Loop</h1>

        {/* Section: What is Backpropagation */}
        <section id="what-is-backprop" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Backpropagation คืออะไร?</h2>
  <div className="my-6"><AdvancedImage cldImg={img1} /></div>

  <p className="mb-4 leading-relaxed">
    Backpropagation หรือชื่อเต็ม ๆ ว่า "Backward Propagation of Errors" เป็นกลไกเบื้องหลังการเรียนรู้ของ Neural Network
    ซึ่งช่วยให้โมเดลสามารถเรียนรู้จากข้อผิดพลาดและปรับปรุงตัวเองให้ดีขึ้นในแต่ละรอบการฝึก
  </p>

  <p className="mb-4 leading-relaxed">
    แนวคิดหลักของ Backpropagation คือการคำนวณ Gradient ของ Loss Function โดยไล่ย้อนกลับจาก Output Layer → Hidden Layer → Input Layer
    เพื่อดูว่าความผิดพลาดนั้นมาจากจุดใด และควรแก้ไขอย่างไร
  </p>

  <p className="mb-4 leading-relaxed">
    เราใช้ Gradient เพื่อบอกว่า "ควรปรับน้ำหนักมากแค่ไหน" โดยอิงจากหลักของ <strong>Chain Rule</strong> ในแคลคูลัส
    ซึ่งช่วยให้โมเดลสามารถเรียนรู้แบบเป็นระบบได้
  </p>

  <p className="mb-4 leading-relaxed">
    ถ้าไม่มี Backpropagation โมเดลจะไม่สามารถอัปเดตน้ำหนักได้อย่างมีประสิทธิภาพ เพราะจะไม่รู้ว่าแต่ละน้ำหนักมีผลกับผลลัพธ์สุดท้ายอย่างไร
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-2 text-center">ภาพรวมการทำงาน</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img2} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>เริ่มจากการทำ Forward Pass เพื่อคำนวณผลลัพธ์จาก Input</li>
    <li>คำนวณ Loss จากผลลัพธ์ที่ได้เทียบกับค่าจริง</li>
    <li>ใช้ Backpropagation เพื่อคำนวณ Gradient ย้อนกลับ</li>
    <li>ใช้ Optimizer เพื่อปรับน้ำหนักตาม Gradient ที่ได้</li>
  </ul>

  <h3 className="text-xl font-semibold mt-6 mb-2 text-center">ประโยชน์ของ Backpropagation</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img3} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ทำให้การฝึก Neural Network เป็นไปอย่างมีระบบ</li>
    <li>ช่วยลดค่า Loss ได้อย่างมีประสิทธิภาพ</li>
    <li>สามารถใช้กับโมเดลที่มีความซับซ้อนได้</li>
    <li>สามารถรวมเข้ากับ Optimizer แบบต่าง ๆ เช่น SGD, Adam</li>
  </ul>

  <h3 className="text-xl font-semibold mt-6 mb-2">ตัวอย่างการเปรียบเทียบแบบง่าย</h3>
  <p className="mb-4">
    ลองนึกถึงการเรียนหนังสือ: ถ้าเราไม่รู้ว่าผิดตรงไหน เราจะปรับตัวเองให้เก่งขึ้นได้อย่างไร? Backpropagation คือการ "ตรวจข้อสอบ"
    ที่ช่วยให้โมเดลรู้ว่าผิดตรงไหนและควรแก้ยังไง
  </p>

  <p className="mb-4">
    ในชีวิตจริง เช่น แอปแปลภาษา → หากระบบแปลผิด Backpropagation จะช่วยให้โมเดลรู้ว่า Output ไหนผิด และจะเรียนรู้ใหม่ให้แปลดีขึ้นในรอบถัดไป
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-2">การเชื่อมโยงกับ Chain Rule</h3>
  <p className="mb-4">
    Backpropagation ใช้ Chain Rule ในการไล่คำนวณ Gradient ระหว่างเลเยอร์ เช่น:
    ถ้าเรามีฟังก์ชัน f(g(x)) → เราต้องใช้ df/dg × dg/dx ซึ่งตรงกับแนวคิดของการคูณ Gradient ไปทีละขั้นตอนจาก Output ไป Input
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-800 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow mb-6">
    <strong>Insight:</strong><br />
    Backpropagation คือเครื่องมือเบื้องหลังความฉลาดของ AI → เป็นวิธีที่ช่วยให้โมเดลรู้ว่าทำผิดที่ไหน และควรปรับปรุงอย่างไรอย่างเป็นระบบ
  </div>
</section>

        {/* Section: Chain Rule */}
        <section id="math-backprop" className="mb-16 scroll-mt-32">
          <h2 className="text-2xl font-semibold mb-4 text-center">คณิตศาสตร์ของ Backpropagation</h2>
          <div className="my-6"><AdvancedImage cldImg={img4} /></div>
    
          <p className="mb-4">
            Backpropagation ใช้หลักการของ <strong>Chain Rule</strong> ในแคลคูลัส เพื่อหาว่าแต่ละพารามิเตอร์ในโมเดลส่งผลต่อ Loss อย่างไร
            ซึ่งเราจะคำนวณ Gradient ของ Loss เทียบกับพารามิเตอร์ทั้งหมด โดยเริ่มจากเลเยอร์สุดท้ายแล้วไล่ย้อนกลับมาทีละชั้น
          </p>

          <p className="mb-4">
            ตัวอย่าง Chain Rule แบบง่าย:
            <code className="bg-gray-800 text-yellow-300 px-2 py-1 rounded ml-2">dL/dx = dL/dz * dz/dx</code>
            หมายถึง หาก z ขึ้นกับ x และ Loss (L) ขึ้นกับ z → เราสามารถใช้อนุพันธ์ของแต่ละส่วนมา chain ต่อกันเพื่อหาผลรวมรวมของ dL/dx
          </p>

          <p className="mb-4">
            ใน Neural Network จริง ๆ เราจะมีหลาย Layer ที่เชื่อมต่อกัน เช่น:
            <br />
            <code className="bg-gray-800 text-yellow-300 px-2 py-1 rounded">a = f(Wx + b)</code> → คือ Output ของ layer หนึ่ง
          </p>

          <p className="mb-4">
            สมมุติเรามี Loss Function L ที่พึ่งพา output y ของโมเดล:
            <br />
            เราต้องหาว่า <code className="bg-gray-800 text-yellow-300 px-1 py-0.5 rounded">∂L/∂W</code>
            เท่ากับเท่าไร ซึ่งต้องใช้ chain rule หลายชั้น:
          </p>

          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li><strong>Forward step:</strong> คำนวณ y = f(Wx + b)</li>
            <li><strong>Compute Loss:</strong> L = loss(y, target)</li>
            <li><strong>Backward step:</strong> หา <code>dL/dy, dy/da, da/dz, dz/dW</code> แล้ว chain ต่อกัน</li>
          </ul>

          <p className="mb-4">
            การใช้ Matrix ช่วยให้เราเขียนคำนวณได้ง่ายขึ้น เช่น:
            <code className="bg-gray-800 text-yellow-300 px-2 py-1 rounded ml-2">
                {'∂L/∂W = δᵀ x'}
                </code>
            ซึ่ง x คือ input, \(\delta\) คือ gradient ที่ไหลย้อนกลับจากเลเยอร์ถัดไป
          </p>

          <h3 className="text-xl font-semibold mb-2">โครงสร้าง Chain Rule ในโมเดลลึก</h3>
          <p className="mb-4">
            ในโมเดลลึก เช่น มี 5 ชั้น → Gradient จาก Loss จะต้องไหลย้อนผ่านทุกชั้นจนถึง Input
            ถ้าชั้นใดใช้ activation function ที่ทำให้ gradient หาย (vanishing) จะมีผลให้โมเดลฝึกไม่ได้เลย
          </p>

          <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl mb-6 text-sm">
            <strong>ตัวอย่างการไล่ gradient แบบ Chain:</strong>
            <pre className="mt-2 whitespace-pre-wrap">// 1. z = Wx + b
// 2. a = ReLU(z)
// 3. y_hat = softmax(a)
// 4. Loss = CrossEntropy(y_hat, y_true)
// ต้องการ dL/dW:
dL/dW = dL/dy_hat * dy_hat/da * da/dz * dz/dW</pre>
          </div>

          <h3 className="text-xl font-semibold mb-2">การใช้ Autograd ในไลบรารี</h3>
          <p className="mb-4">
            ใน PyTorch และ TensorFlow เราไม่ต้องคำนวณด้วยมือ เพราะระบบจะใช้ Autograd ช่วยตาม Chain Rule โดยอัตโนมัติ
            แต่การเข้าใจว่าเบื้องหลังมีการไล่ gradient อย่างไรจะช่วยให้เราปรับโมเดลได้ถูกต้องเวลาเกิดปัญหา
          </p>

          <h3 className="text-xl font-semibold mb-2 text-center">Gradient กับการเรียนรู้</h3>
          <div className="flex justify-center my-6"><AdvancedImage cldImg={img5} /></div>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li>Gradient จะมีค่าบวกหรือลบ ขึ้นกับทิศทางของความผิด</li>
            <li>หาก Gradient = 0 → หมายถึงโมเดลไม่รู้จะปรับยังไง (learning หยุด)</li>
            <li>Gradient flow ที่ดีจะช่วยให้ Optimizer ปรับพารามิเตอร์ได้แม่นยำขึ้น</li>
          </ul>

          <h3 className="text-xl font-semibold mb-2">Insight:</h3>
          <div className="bg-yellow-100 dark:bg-yellow-800 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500">
            <p className="text-sm">
              Chain Rule คือหัวใจของการเรียนรู้แบบย้อนกลับ — ถ้าไม่มีหลักการนี้ Neural Network จะไม่สามารถเรียนรู้ได้
              การเข้าใจ Chain Rule ไม่ใช่แค่เรื่องคณิตศาสตร์ แต่คือการเข้าใจกลไกการปรับตัวของโมเดล
            </p>
          </div>
        </section>

        <section id="training-loop" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Training Loop ของ Neural Network</h2>
  <div className="my-6"><AdvancedImage cldImg={img6} /></div>

  <p className="mb-4 leading-relaxed">
    Training Loop คือกระบวนการหลักของการเรียนรู้ใน Neural Network ซึ่งวนซ้ำหลายรอบเพื่อให้โมเดลสามารถเรียนรู้จากข้อมูล
    โดยมีโครงสร้างที่ชัดเจน ตั้งแต่การป้อนข้อมูล → คำนวณผลลัพธ์ → วัดความผิดพลาด → และปรับปรุงโมเดล
  </p>

  <h3 className="text-xl font-semibold mb-2">1. ขั้นตอนหลัก</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Forward Pass:</strong> ป้อน input เข้าโมเดลเพื่อคำนวณ output</li>
    <li><strong>Loss Evaluation:</strong> คำนวณค่าความผิดพลาดจาก output เทียบกับค่าจริง</li>
    <li><strong>Backward Pass:</strong> คำนวณ gradient ผ่าน Backpropagation (ดูรายละเอียดใน Section ก่อนหน้า)</li>
    <li><strong>Parameter Update:</strong> ส่งต่อ gradient ไปยัง Optimizer เพื่อปรับ weight/bias</li>
    <li><strong>Repeat:</strong> ทำซ้ำรอบนี้หลาย epoch เพื่อให้โมเดลเรียนรู้ได้ลึกขึ้น</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">2. Epoch vs Iteration</h3>
  <p className="mb-4">
    - <strong>Epoch:</strong> การใช้ข้อมูลทั้งหมด 1 รอบในการฝึกโมเดล  
    - <strong>Iteration:</strong> จำนวนรอบย่อยภายใน 1 epoch ซึ่งขึ้นอยู่กับขนาดของ mini-batch
  </p>
  <p className="mb-4">
    ยกตัวอย่าง: หากมี 10,000 ตัวอย่างและ batch size = 100 → จะมี 100 iterations ต่อ 1 epoch
  </p>

  <h3 className="text-xl font-semibold mb-2">3. รูปแบบของ Batch</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Batch:</strong> ใช้ข้อมูลทั้งหมดใน 1 รอบการอัปเดต (ใช้ resource สูง)</li>
    <li><strong>Mini-Batch:</strong> แบ่งข้อมูลออกเป็นชุดย่อย (เช่น 32, 64) เป็นที่นิยมที่สุด</li>
    <li><strong>Stochastic (SGD):</strong> ใช้ข้อมูลเพียง 1 ตัวอย่างในการอัปเดตแต่ละครั้ง</li>
  </ul>
  <p className="mb-4">
    Mini-Batch เป็นรูปแบบที่นิยมที่สุดเพราะสามารถปรับสมดุลระหว่างความเร็วและเสถียรภาพของการเรียนรู้ได้ดี
  </p>

  <h3 className="text-xl font-semibold mb-2">4. ทำไมต้องเข้าใจ Training Loop?</h3>
  <p className="mb-4">
    การเข้าใจโครงสร้างของ Training Loop ช่วยให้คุณสามารถ:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ดีบักปัญหาที่เกิดขึ้นระหว่างการฝึก เช่น loss ไม่ลด</li>
    <li>ตัดสินใจได้ว่าเมื่อใดควรใช้ early stopping</li>
    <li>ปรับ batch size หรือจำนวน epoch ให้เหมาะสมกับ dataset</li>
    <li>เข้าใจจุดเชื่อมโยงกับ Backpropagation และ Optimizer</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    Training Loop คือ “กรอบกระบวนการเรียนรู้” ของ AI — เปรียบเหมือนตารางฝึกซ้อมของนักเรียนที่ต้องทำซ้ำอย่างมีแบบแผนเพื่อเก่งขึ้นทีละนิดในทุก ๆ รอบ
  </div>
</section>


        {/* Section: Batch Types */}
        <section id="batch-types" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Batch vs Mini-Batch vs Stochastic</h2>
  <div className="my-6"><AdvancedImage cldImg={img7} /></div>

  <p className="mb-4 leading-relaxed">
    หนึ่งในปัจจัยสำคัญที่มีผลต่อประสิทธิภาพของการฝึก Neural Network คือการเลือกวิธีการแบ่งข้อมูลเพื่อใช้ในการอัปเดต
    ซึ่งสามารถแบ่งได้เป็น 3 รูปแบบหลัก ๆ ได้แก่ Batch, Mini-Batch และ Stochastic โดยแต่ละแบบมีข้อดีข้อจำกัดที่ต่างกันไป
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">1. Batch Gradient Descent</h3>
  <div className="flex justify-center my-6"><AdvancedImage cldImg={img8} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>หลักการ:</strong> ใช้ข้อมูลทั้งหมดทั้ง training set ในการคำนวณ Gradient และอัปเดตน้ำหนักใน 1 รอบ</li>
    <li><strong>ข้อดี:</strong> ผลลัพธ์เสถียร, Loss ลดลงอย่างต่อเนื่อง</li>
    <li><strong>ข้อเสีย:</strong> ใช้ทรัพยากรสูง, ช้า, ไม่เหมาะกับข้อมูลขนาดใหญ่</li>
  </ul>
  <p className="mb-4">
    ตัวอย่าง: Dataset มี 50,000 ตัวอย่าง → ในแต่ละ epoch จะใช้ทั้ง 50,000 ตัวอย่างในการอัปเดต 1 ครั้ง
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">2. Stochastic Gradient Descent (SGD)</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img9} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>หลักการ:</strong> อัปเดตน้ำหนักทุก 1 ตัวอย่างทันทีหลังจากคำนวณ Loss</li>
    <li><strong>ข้อดี:</strong> เร็ว, ไม่ต้องรอทั้ง batch, ช่วยหลุดจาก local minima ได้ง่าย</li>
    <li><strong>ข้อเสีย:</strong> Loss แกว่ง, ไม่เสถียร, ต้องใช้เทคนิคปรับ Learning Rate</li>
  </ul>
  <p className="mb-4">
    ตัวอย่าง: สำหรับ dataset 50,000 ตัวอย่าง → จะมี 50,000 updates ต่อ epoch
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">3. Mini-Batch Gradient Descent</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img10} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>หลักการ:</strong> แบ่งข้อมูลออกเป็นกลุ่มย่อย เช่น 32 หรือ 64 ตัวอย่างต่อรอบ</li>
    <li><strong>ข้อดี:</strong> สมดุลระหว่างความเร็วและความเสถียร, สามารถใช้กับ hardware ได้ง่าย</li>
    <li><strong>ข้อเสีย:</strong> ต้องเลือกขนาด batch ให้เหมาะสม, มี trade-off ระหว่าง memory และ accuracy</li>
  </ul>
  <p className="mb-4">
    ตัวอย่าง: ถ้าใช้ batch size = 64 กับ dataset 50,000 → จะมีประมาณ 781 mini-batches ต่อ epoch
  </p>

  <h3 className="text-xl font-semibold mb-2">4. เปรียบเทียบทั้งสามแบบ</h3>
  <div className="overflow-x-auto mb-4">
    <table className="table-auto w-full text-sm border border-yellow-500">
      <thead className="bg-yellow-100 dark:bg-yellow-800">
        <tr>
          <th className="p-2 border border-yellow-400">วิธี</th>
          <th className="p-2 border border-yellow-400">การอัปเดต</th>
          <th className="p-2 border border-yellow-400">ความเร็ว</th>
          <th className="p-2 border border-yellow-400">ความเสถียร</th>
          <th className="p-2 border border-yellow-400">เหมาะกับ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border border-yellow-400">Batch</td>
          <td className="p-2 border border-yellow-400">ทั้งชุดข้อมูล</td>
          <td className="p-2 border border-yellow-400">ช้า</td>
          <td className="p-2 border border-yellow-400">สูง</td>
          <td className="p-2 border border-yellow-400">Dataset ขนาดเล็ก</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400">Stochastic</td>
          <td className="p-2 border border-yellow-400">1 ตัวอย่าง/รอบ</td>
          <td className="p-2 border border-yellow-400">เร็ว</td>
          <td className="p-2 border border-yellow-400">ต่ำ</td>
          <td className="p-2 border border-yellow-400">โมเดลทดลอง, หลุดจาก local min</td>
        </tr>
        <tr>
          <td className="p-2 border border-yellow-400">Mini-Batch</td>
          <td className="p-2 border border-yellow-400">32–256 ตัวอย่าง/รอบ</td>
          <td className="p-2 border border-yellow-400">เร็วปานกลาง</td>
          <td className="p-2 border border-yellow-400">สมดุล</td>
          <td className="p-2 border border-yellow-400">งานจริงส่วนใหญ่</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-2">5. การเลือกใช้งานในชีวิตจริง</h3>
  <p className="mb-4">
    ในทางปฏิบัติ Mini-Batch มักเป็นทางเลือกที่ดีที่สุด เพราะสามารถใช้ร่วมกับ hardware เช่น GPU/TPU ได้ดี
    โดย batch size ยอดนิยมคือ 32, 64, 128 ทั้งนี้ขึ้นอยู่กับขนาดข้อมูลและหน่วยความจำที่มี
  </p>
  <p className="mb-4">
    นักวิจัยมักเริ่มจาก batch size เล็ก และค่อย ๆ เพิ่มเมื่อเห็นว่าโมเดลยังไม่ converge หรือ memory ยังเพียงพอ
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    การเลือกขนาด batch และรูปแบบการอัปเดตที่เหมาะสม คือหนึ่งในกุญแจสำคัญที่ทำให้โมเดลเรียนรู้ได้เร็ว เสถียร และแม่นยำ
  </div>
</section>

    {/* Section: Parameter Updates */}
<section id="update-params" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">การอัปเดตพารามิเตอร์</h2>

  <p className="mb-4 leading-relaxed">
    หลังจากที่คำนวณค่า Gradient ผ่าน Backpropagation เสร็จเรียบร้อย ขั้นตอนต่อมาคือการนำ Gradient เหล่านั้น
    มาใช้ในการอัปเดตค่าพารามิเตอร์ของโมเดล เช่น Weight และ Bias ให้โมเดลเรียนรู้และปรับตัวดีขึ้นในรอบถัดไป
  </p>

  <p className="mb-4">
    เราสามารถอธิบายกระบวนการนี้ด้วยสูตรพื้นฐานของการอัปเดตแบบ Gradient Descent:
    <br/>
    <code className="bg-gray-800 text-yellow-300 px-2 py-1 rounded">w = w - lr * dw</code>
    <br/>
    โดยที่:
    <ul className="list-disc pl-6 space-y-2 my-2">
      <li><code>w</code> คือพารามิเตอร์ (เช่น weight)</li>
      <li><code>lr</code> คือ Learning Rate</li>
      <li><code>dw</code> คือ Gradient ของ Loss ต่อพารามิเตอร์นั้น</li>
    </ul>
  </p>

  <h3 className="text-xl font-semibold mb-2">1. Learning Rate คืออะไร?</h3>
  <p className="mb-4">
    Learning Rate คือความเร็วในการปรับพารามิเตอร์ หากค่ามากเกินไป → อาจกระโดดข้ามจุดที่ดีที่สุด
    หากน้อยเกินไป → จะช้ามากในการเรียนรู้ หรืออาจติดอยู่ใน local minimum
  </p>

  <div className="grid md:grid-cols-3 gap-4 mb-6">
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
      <h4 className="font-semibold mb-2"> Learning Rate ต่ำ</h4>
      <ul className="list-disc pl-6 text-sm space-y-1">
        <li>Loss ลดช้า</li>
        <li>ใช้เวลาฝึกนาน</li>
        <li>อาจติดอยู่ที่ local minima</li>
      </ul>
    </div>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
      <h4 className="font-semibold mb-2"> Learning Rate เหมาะสม</h4>
      <ul className="list-disc pl-6 text-sm space-y-1">
        <li>Loss ลดอย่างต่อเนื่อง</li>
        <li>ฝึกได้เร็วและมีเสถียรภาพ</li>
      </ul>
    </div>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
      <h4 className="font-semibold mb-2"> Learning Rate สูง</h4>
      <ul className="list-disc pl-6 text-sm space-y-1">
        <li>Loss แกว่ง ไม่ลดลง</li>
        <li>อาจทำให้เกิด NaN</li>
        <li>โมเดลไม่ converge</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-2">2. Optimizer ช่วยจัดการอย่างไร?</h3>
  <p className="mb-4">
    Optimizer คืออัลกอริธึมที่นำ Gradient ไปใช้เพื่อปรับค่าพารามิเตอร์ให้เหมาะสมที่สุด โดยมีหลายชนิด เช่น:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>SGD:</strong> ปรับค่าตาม Gradient แบบตรง ๆ</li>
    <li><strong>Momentum:</strong> เพิ่มแรงส่งเพื่อไม่ติดจุดต่ำปลอม</li>
    <li><strong>RMSProp:</strong> ปรับ learning rate ตามพฤติกรรม Gradient</li>
    <li><strong>Adam:</strong> รวม Momentum + RMSProp เข้าด้วยกัน → ใช้บ่อยที่สุด</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">3. ตัวอย่างการอัปเดตใน PyTorch</h3>
<div className="bg-gray-800 text-yellow-100 text-sm rounded-xl font-mono mb-4 overflow-x-auto">
  <pre className="p-4 whitespace-pre">
{`# Assume loss has been calculated
loss.backward()           # คำนวณ gradient
optimizer.step()          # อัปเดตพารามิเตอร์
optimizer.zero_grad()     # เคลียร์ gradient เดิม`}
  </pre>
</div>

<h3 className="text-xl font-semibold mb-2">4. การอัปเดตใน TensorFlow/Keras</h3>
<div className="bg-gray-800 text-yellow-100 text-sm rounded-xl font-mono overflow-x-auto">
  <pre className="p-4 whitespace-pre">
{`model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100)`}
  </pre>
</div>


  <h3 className="text-xl font-semibold mt-6 mb-3">5. ปัญหาที่พบบ่อย</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>อัปเดตเร็วเกินไปจนเกิด overfitting</li>
    <li>ไม่ปรับ learning rate เมื่อเข้าสู่ช่วงใกล้ convergence</li>
    <li>Gradient หาย (vanishing) → weight ไม่ถูกอัปเดต</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    การอัปเดตพารามิเตอร์ไม่ใช่แค่เรื่องของสูตรคำนวณ แต่คือ "กระบวนการเรียนรู้ของ AI" — ยิ่งเข้าใจลึกเท่าไหร่ ยิ่งสามารถควบคุมและพัฒนาโมเดลได้อย่างแม่นยำ
  </div>
</section>

{/* Section: Monitoring */}
<section id="monitoring" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">การ Monitor การฝึก</h2>
  <div className="flex justify-center my-6"><AdvancedImage cldImg={img11} /></div>

  <p className="mb-4 leading-relaxed">
    การฝึก Neural Network ไม่ใช่แค่ใส่ข้อมูลแล้วปล่อยให้โมเดลเรียนรู้เองโดยไม่ดูผลลัพธ์ — การ "Monitor" หรือเฝ้าติดตามกระบวนการฝึก
    เป็นสิ่งสำคัญที่ช่วยให้เรามั่นใจว่าโมเดลกำลังเรียนรู้ในทิศทางที่ดี ไม่ Overfitting หรือ Underfitting และสามารถตัดสินใจได้ว่าเมื่อไหร่ควรหยุดหรือปรับแต่งพารามิเตอร์ใหม่
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">1. Metrics ที่นิยมใช้</h3>
  <div className="flex justify-center my-6"><AdvancedImage cldImg={img12} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Loss:</strong> ค่าความผิดพลาดของโมเดลในแต่ละรอบ — ควรลดลงเรื่อย ๆ</li>
    <li><strong>Accuracy:</strong> อัตราการทำนายถูกต้อง (ใช้กับ classification)</li>
    <li><strong>Precision, Recall, F1-Score:</strong> ใช้เมื่อต้องการวัดประสิทธิภาพที่ละเอียดขึ้น</li>
    <li><strong>Validation Loss/Accuracy:</strong> วัด performance บนข้อมูลที่โมเดลไม่เคยเห็น</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 text-center">2. Visualization ด้วย Plot</h3>
  <div className="flex justify-center my-6"><AdvancedImage cldImg={img13} /></div>
  <p className="mb-4">
    การ plot กราฟช่วยให้เราเห็นภาพรวมการเรียนรู้อย่างชัดเจน เช่น:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>Loss curve ที่ค่อย ๆ ลดลงในแต่ละ epoch</li>
    <li>Accuracy curve ที่เพิ่มขึ้นใน training และ validation</li>
    <li>Gap ระหว่าง train กับ validation → ใช้วิเคราะห์ overfitting</li>
  </ul>

  <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono overflow-auto mb-6">
    <pre>{`# ตัวอย่างการ plot ด้วย matplotlib
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-3 text-center">3. Early Stopping</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img13} /></div>
  <p className="mb-4">
    เทคนิค Early Stopping คือการหยุดการฝึกเมื่อโมเดลไม่พัฒนาอีกแล้ว
    เช่น เมื่อ Validation Loss ไม่ลดลงเป็นจำนวนรอบที่กำหนด (patience)
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6 ">
    <li>ช่วยป้องกัน Overfitting โดยไม่ฝึกเกินความจำเป็น</li>
    <li>ประหยัดเวลาในการฝึก และใช้ resource ได้คุ้มค่า</li>
    <li>เหมาะกับงาน production ที่ต้องการประสิทธิภาพสูงสุดในเวลาน้อย</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 text-center">4. Overfitting & Underfitting</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img14} /></div>
  <p className="mb-4">
    - <strong>Overfitting:</strong> โมเดลเรียนรู้เฉพาะข้อมูลฝึกจนขาดความสามารถในการ generalize → validation loss สูง
    <br/>
    - <strong>Underfitting:</strong> โมเดลยังเรียนรู้ไม่พอ → ทั้ง train และ val performance ต่ำ
  </p>
  <p className="mb-4">
    การสังเกตพฤติกรรมเหล่านี้จากกราฟช่วยให้เราตัดสินใจปรับโมเดล เช่น เพิ่มข้อมูล, ลดความลึก, ใช้ regularization
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">5. Callback ที่มีประโยชน์</h3>
  <div className="flex justify-center my-6 "><AdvancedImage cldImg={img15} /></div>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>ModelCheckpoint:</strong> เก็บ weights ที่ดีที่สุดระหว่างการฝึก</li>
    <li><strong>ReduceLROnPlateau:</strong> ลด learning rate เมื่อ performance ไม่ดีขึ้น</li>
    <li><strong>TensorBoard:</strong> เครื่องมือ visualize training แบบ real-time</li>
  </ul>

  <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono overflow-auto mb-6">
    <pre>{`# ตัวอย่าง EarlyStopping ด้วย Keras
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X, y, validation_split=0.2, callbacks=[es])`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-3">6. Insight สำคัญ</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <p>
      การ Monitor ไม่ใช่แค่การดูว่าค่า loss เป็นเท่าไหร่ แต่เป็นการมองภาพรวมของโมเดลทั้งการเรียนรู้ ความเสถียร ความสามารถในการ generalize
      และการตัดสินใจหยุดฝึกอย่างชาญฉลาด → นำไปสู่โมเดลที่มีคุณภาพสูงสุด
    </p>
  </div>
</section>
{/* Section: Insight */}
  {/* Section: Example Code */}
<section id="code-example" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">ตัวอย่างโค้ด Training Loop</h2>
  <p className="mb-4">
    ในการฝึก Neural Network จริง เรามักใช้เฟรมเวิร์กอย่าง PyTorch หรือ TensorFlow/Keras ซึ่งมีรูปแบบที่ต่างกันบ้างแต่แนวคิดพื้นฐานเหมือนกัน คือ:
  </p>

  <ol className="list-decimal pl-6 space-y-2 mb-6">
    <li>เตรียมข้อมูล (dataset & dataloader)</li>
    <li>กำหนดโครงสร้างโมเดล (Model Definition)</li>
    <li>กำหนด Loss Function และ Optimizer</li>
    <li>สร้าง Training Loop → forward → loss → backward → update</li>
  </ol>

  <h3 className="text-xl font-semibold mb-2"> ตัวอย่าง PyTorch แบบเต็ม</h3>
  <p className="mb-4">
    ตัวอย่างโค้ดนี้แสดงการฝึกโมเดลแบบ manual training loop ด้วย PyTorch:
  </p>

  <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono overflow-auto mb-4">
    <pre>{`# PyTorch Full Example
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")`}</pre>
  </div>

  <p className="mb-4">
    จุดสำคัญคือ `zero_grad()` ต้องถูกเรียกก่อน `backward()` ทุกครั้งเพื่อเคลียร์ Gradient เดิม และ `loss.item()` ใช้แปลงค่า tensor เป็นตัวเลขธรรมดาสำหรับแสดงผล
  </p>

  <h3 className="text-xl font-semibold mb-2"> การใช้ GPU กับ PyTorch</h3>
  <p className="mb-4">คุณสามารถเพิ่ม `.to(device)` เพื่อใช้ GPU ได้แบบนี้:</p>
  <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono overflow-auto mb-4">
    <pre>{`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-2"> ตัวอย่าง Keras แบบสั้นและง่าย</h3>
  <p className="mb-4">
    Keras ใช้ `.fit()` ซึ่งทำงานแบบ auto-loop ได้เลยโดยไม่ต้องเขียน training loop เอง:
  </p>
  <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono overflow-auto mb-4">
    <pre>{`# Keras แบบกระชับ
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-2"> การใช้ Callback กับ Keras</h3>
  <p className="mb-4">
    คุณสามารถควบคุมการฝึกด้วย callback เช่น EarlyStopping หรือ ModelCheckpoint ได้ เช่น:
  </p>
  <div className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono overflow-auto mb-4">
    <pre>{`from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[callback])`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-2"> เปรียบเทียบ PyTorch vs Keras</h3>
  <table className="w-full table-auto text-sm border border-yellow-500 mb-6">
    <thead className="bg-yellow-100 dark:bg-yellow-800">
      <tr>
        <th className="p-2 border border-yellow-400">หัวข้อ</th>
        <th className="p-2 border border-yellow-400">PyTorch</th>
        <th className="p-2 border border-yellow-400">Keras</th>
      </tr>
    </thead>
    <tbody className="bg-white dark:bg-gray-800">
      <tr>
        <td className="p-2 border">โครงสร้าง Training</td>
        <td className="p-2 border">ยืดหยุ่น, ต้องเขียน loop เอง</td>
        <td className="p-2 border">สะดวก ใช้ fit()</td>
      </tr>
      <tr>
        <td className="p-2 border">ระดับควบคุม</td>
        <td className="p-2 border">สูงมาก เหมาะกับงานวิจัย</td>
        <td className="p-2 border">ดีสำหรับงานทั่วไป</td>
      </tr>
      <tr>
        <td className="p-2 border">การใช้ GPU</td>
        <td className="p-2 border">ต้องใส่ .to(device) เอง</td>
        <td className="p-2 border">Keras จัดการให้โดยอัตโนมัติ</td>
      </tr>
    </tbody>
  </table>

  <div className="bg-yellow-50 dark:bg-yellow-800 text-black dark:text-yellow-100 p-4 mt-6 rounded-xl border-l-4 border-yellow-500">
    <strong>Insight:</strong><br />
    ไม่ว่าคุณจะใช้ PyTorch หรือ Keras สิ่งสำคัญคือการเข้าใจ flow การฝึกอย่างลึกซึ้ง: forward → loss → backward → update
    เพราะนั่นคือหัวใจของการสร้าง AI ที่เรียนรู้ได้จริง
  </div>
</section>

<section id="insight" className="mb-16 scroll-mt-32">
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow-xl">
    <h2 className="text-xl font-semibold mb-4">Insight: Backpropagation = สมองของการเรียนรู้</h2>

    <p className="mb-4">
      Backpropagation ไม่ได้เป็นเพียงแค่กลไกเชิงคณิตศาสตร์ แต่เป็นกระบวนการที่เปรียบเสมือน "สมองของระบบการเรียนรู้"
      เพราะมันคือกลไกที่ช่วยให้โมเดล "รู้ว่าผิดตรงไหน" และสามารถนำข้อผิดพลาดนั้นมาเป็นข้อมูลย้อนกลับในการปรับปรุงตนเองได้
    </p>

    <p className="mb-4">
      ลองจินตนาการว่าโมเดลคือ "นักเรียน" ที่กำลังฝึกทำข้อสอบ — ในแต่ละครั้งเขาจะได้รับผลลัพธ์ (คำตอบ) จากการคิดของตัวเอง (forward pass)
      หลังจากนั้นจะได้รับการเฉลยจากครู (loss function) และเรียนรู้ว่าผิดตรงไหน
      สิ่งสำคัญที่สุดคือการ "ย้อนกลับไปดูว่าทำไมถึงตอบผิด" และปรับความเข้าใจ (backward pass)
    </p>

    <h3 className="text-lg font-semibold mt-6 mb-2"> เหตุผลที่ Backpropagation คือสมองของ AI</h3>
    <ul className="list-disc pl-6 space-y-2 text-base">
      <li>ช่วยให้โมเดลรู้ว่าน้ำหนัก (weight) ตัวไหนต้องปรับ</li>
      <li>กระจายความผิดพลาดแบบมีระบบ (ผ่าน Chain Rule)</li>
      <li>สามารถขยายไปยังโมเดลลึกหลายร้อยเลเยอร์ได้</li>
      <li>ทำงานร่วมกับ Optimizer เพื่อให้เกิดการเรียนรู้อย่างเป็นขั้นตอน</li>
    </ul>

    <h3 className="text-lg font-semibold mt-6 mb-2"> เปรียบเทียบกับสมองมนุษย์</h3>
    <p className="mb-4">
      การเรียนรู้ของมนุษย์เกิดจากการ "ลองผิด – รู้ว่าผิด – ปรับตัว" ซึ่งคือสิ่งเดียวกันกับที่เกิดขึ้นใน Neural Network
      โมเดลไม่สามารถปรับปรุงตนเองได้เลยหากไม่มีข้อมูลว่าตัวเอง "ผิดตรงไหน" และนั่นคือสิ่งที่ Backpropagation ทำหน้าที่แทนสมอง
    </p>

    <div className="bg-white/10 border border-yellow-300 p-4 rounded-xl mb-6">
      <p className="font-semibold mb-2 text-yellow-300">อุปมาเปรียบเทียบ:</p>
      <ul className="list-disc pl-6 text-sm space-y-1">
        <li><strong>Forward pass</strong> = นักเรียนลองตอบคำถาม</li>
        <li><strong>Loss</strong> = เฉลยบอกว่าถูกหรือผิดมากแค่ไหน</li>
        <li><strong>Backward pass</strong> = วิเคราะห์ว่าอะไรทำให้ตอบผิด</li>
        <li><strong>Optimizer</strong> = การฝึกฝนให้ดีขึ้นรอบถัดไป</li>
      </ul>
    </div>

    <h3 className="text-lg font-semibold mt-6 mb-2"> ทำไม Backprop ถึงสำคัญในยุค Deep Learning?</h3>
    <p className="mb-4">
      ในยุคที่โมเดลมีจำนวนเลเยอร์มากขึ้น เช่น GPT หรือ ResNet ที่มีหลายร้อยเลเยอร์ — การคำนวณ Gradient
      ต้องแม่นยำและไหลกลับได้อย่างเสถียร ซึ่ง Backpropagation คือรากฐานของกระบวนการนี้
    </p>
    <p className="mb-4">
      ทุกโมเดล AI ที่คุณใช้อยู่ในปัจจุบัน ไม่ว่าจะเป็นแปลภาษา การรู้จำภาพ หรือการตอบคำถาม ล้วนแล้วแต่เรียนรู้จากข้อมูลผ่าน Backpropagation ทั้งสิ้น
    </p>

    <h3 className="text-lg font-semibold mt-6 mb-2"> การวนลูปของการเรียนรู้</h3>
    <p className="mb-4">
      โมเดลจะไม่เก่งในทันที แต่จะค่อย ๆ พัฒนาได้ผ่านลูปของการทำนาย → ตรวจสอบ → ปรับ → ทำซ้ำ
      คล้ายกับกระบวนการของมนุษย์ที่ต้องล้มแล้วลุกใหม่ พร้อมการเข้าใจว่า "ทำไมถึงล้ม" และปรับแนวทางให้ดีขึ้น
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl text-sm border-l-4 border-yellow-500 mb-6">
      <p className="font-semibold mb-2"> บทสรุป Insight:</p>
      <ul className="list-disc pl-6">
        <li>ไม่มี Backpropagation = ไม่มีการเรียนรู้ → AI ไม่สามารถฉลาดขึ้นได้</li>
        <li>Backprop ไม่ใช่เพียงการคำนวณ แต่เป็นระบบคิด-วิเคราะห์ของ AI</li>
        <li>เป็นหัวใจสำคัญของ Deep Learning ทุกแขนง ตั้งแต่ภาพ เสียง จนถึงภาษา</li>
      </ul>
    </div>

    <p className="italic text-sm text-yellow-900 dark:text-yellow-100">
      "Backpropagation คือเครื่องย้อนเวลาให้โมเดลกลับไปดูอดีตของตัวเองแล้วใช้มันปรับอนาคตให้ดีขึ้น"
    </p>
  </div>
</section>



        <section id="quiz" className="mb-16 scroll-mt-32">
          <MiniQuiz_Day8 theme={theme} />
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
        <ScrollSpy_Ai_Day8 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day8_Backpropagation;
