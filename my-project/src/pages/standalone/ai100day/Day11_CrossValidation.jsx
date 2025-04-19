import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day11 from "./scrollspy/ScrollSpy_Ai_Day11";
import MiniQuiz_Day11 from "./miniquiz/MiniQuiz_Day11";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day11_CrossValidation = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  const img1 = cld.image('CrossVal1').format('auto').quality('auto').resize(scale().width(500));
  const img2 = cld.image('CrossVal2').format('auto').quality('auto').resize(scale().width(500));
  const img3 = cld.image('CrossVal3').format('auto').quality('auto').resize(scale().width(500));
  const img4 = cld.image('CrossVal4').format('auto').quality('auto').resize(scale().width(400));
  const img5 = cld.image('CrossVal5').format('auto').quality('auto').resize(scale().width(400));
  const img6 = cld.image('CrossVal6').format('auto').quality('auto').resize(scale().width(400));
  const img7 = cld.image('CrossVal7').format('auto').quality('auto').resize(scale().width(400));
  const img8 = cld.image('CrossVal8').format('auto').quality('auto').resize(scale().width(400));
  const img9 = cld.image('CrossVal9').format('auto').quality('auto').resize(scale().width(500));
  const img10 = cld.image('CrossVal10').format('auto').quality('auto').resize(scale().width(400));
  const img11 = cld.image('CrossVal11').format('auto').quality('auto').resize(scale().width(400));
  const img12 = cld.image('CrossVal12').format('auto').quality('auto').resize(scale().width(500));
  const img13 = cld.image('CrossVal13').format('auto').quality('auto').resize(scale().width(400));
  const img14 = cld.image('CrossVal14').format('auto').quality('auto').resize(scale().width(500));
  const img15 = cld.image('CrossVal15').format('auto').quality('auto').resize(scale().width(400));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 11: Cross Validation & Model Evaluation</h1>

        {/* Section: ทำไมต้อง Cross Validation */}
        <section id="why-cross-validation" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">ทำไมต้อง Cross Validation?</h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในโลกของ Machine Learning การประเมินความสามารถของโมเดลถือเป็นขั้นตอนที่สำคัญที่สุดเพื่อให้แน่ใจว่าโมเดลไม่ได้เรียนรู้เพียงข้อมูลฝึก
    (training data) เท่านั้น แต่ยังสามารถ “generalize” ไปยังข้อมูลใหม่ได้ด้วย ซึ่งกระบวนการนี้เรียกว่า “Model Evaluation”
    หนึ่งในวิธีที่ได้รับความนิยมและถือว่ามีประสิทธิภาพสูงในการประเมินโมเดลคือ <strong>Cross Validation</strong>
  </p>

  <h3 className="text-xl font-semibold mb-3">1. ปัญหาของการแบ่ง Train/Test แบบธรรมดา</h3>
  <p className="mb-4 leading-relaxed">
    โดยทั่วไปแล้ว การประเมินโมเดลอาจเริ่มต้นด้วยการแบ่งข้อมูลออกเป็น 2 ส่วน ได้แก่ <em>Training Set</em> สำหรับฝึกโมเดล และ
    <em>Test Set</em> สำหรับทดสอบผลลัพธ์ โมเดลจะเรียนรู้จาก training และเราจะดูประสิทธิภาพจาก test อย่างไรก็ตาม การแบ่งเพียงครั้งเดียวนี้มีข้อเสียคือ:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ผลลัพธ์ขึ้นอยู่กับวิธีแบ่งข้อมูล — อาจโชคดีหรือโชคร้ายได้ข้อมูลที่ bias</li>
    <li>ใช้ข้อมูลได้ไม่เต็มที่ เพราะบางส่วนถูกกันไว้เป็น test set</li>
    <li>ความแม่นยำอาจไม่เสถียรหากข้อมูลน้อยหรือไม่สมดุล</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">2. ทางออกด้วย Cross Validation</h3>
  <p className="mb-4 leading-relaxed">
    Cross Validation เข้ามาเพื่อแก้ปัญหานี้ โดยการแบ่งข้อมูลออกเป็นหลายส่วน (เช่น 5 หรือ 10 ส่วน) แล้วใช้แต่ละส่วนเป็น test set
    สลับกันไปในแต่ละรอบ ฝึกโมเดลบนส่วนที่เหลือ และประเมินบนส่วนที่เหลืออีกหนึ่ง ทำซ้ำจนทุกส่วนได้เป็น test ครบทุกครั้ง
  </p>

  <h3 className="text-xl font-semibold mb-3">3. ทำไมวิธีนี้ถึงแม่นยำกว่า?</h3>
  <p className="mb-4 leading-relaxed">
    Cross Validation ช่วยให้ได้ผลเฉลี่ยจากหลายรอบการประเมิน จึงลดอิทธิพลของการสุ่มข้อมูลลง และสะท้อน performance จริงของโมเดลมากกว่า
    เช่น ถ้าเรามีชุดข้อมูลเล็ก การแบ่งแบบ 80/20 อาจทำให้ test set ไม่หลากหลาย แต่ถ้าใช้ K-Fold Cross Validation แล้วเฉลี่ยคะแนน
    โมเดลจะถูกประเมินบนชุดข้อมูลที่ต่างกันทั้งหมด
  </p>

  <h3 className="text-xl font-semibold mb-3">4. ใช้ Cross Validation ตอนไหน?</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ตอนเปรียบเทียบระหว่างโมเดลหลายแบบ (Model Selection)</li>
    <li>ตอนปรับค่าพารามิเตอร์ เช่น Grid Search หรือ Hyperparameter Tuning</li>
    <li>ตอนต้องการรู้ว่าโมเดลมีแนวโน้ม overfit หรือ underfit หรือไม่</li>
    <li>ตอนข้อมูลไม่มากและต้องใช้ให้เกิดประโยชน์สูงสุด</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 text-center">5. ประโยชน์ที่ได้จาก Cross Validation</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>ใช้ข้อมูลได้เต็มที่:</strong> ไม่มีส่วนไหนของข้อมูลที่ถูกทิ้ง</li>
    <li><strong>ประเมินได้หลากหลาย:</strong> ทุกจุดในข้อมูลได้มีโอกาสเป็น test</li>
    <li><strong>ลดความลำเอียง:</strong> เพราะไม่พึ่งพาการสุ่มครั้งเดียว</li>
    <li><strong>ให้ภาพรวมที่แม่นยำกว่า:</strong> ค่าเฉลี่ยของหลายรอบช่วยลด noise</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">6. แล้ว Cross Validation มีข้อเสียไหม?</h3>
  <p className="mb-4 leading-relaxed">
    แม้จะมีข้อดีมาก แต่ก็มีข้อเสียหลักที่ควรพิจารณา เช่น ใช้เวลานานในการประมวลผล เพราะต้องฝึกและทดสอบหลายรอบ
    โดยเฉพาะเมื่อโมเดลซับซ้อน หรือมีข้อมูลจำนวนมาก อาจทำให้ต้องใช้ computing power สูงขึ้น
  </p>

  <h3 className="text-xl font-semibold mb-3">7. สรุปภาพรวม</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2"><strong>Train/Test Split ธรรมดา</strong> — เร็วแต่ไม่เสถียร, เหมาะกับการทดสอบเบื้องต้น</p>
    <p className="mb-2"><strong>Cross Validation</strong> — ช้าแต่เสถียรกว่า, เหมาะกับการเลือกโมเดลหรือปรับพารามิเตอร์</p>
    <p className="mb-2">ถ้าข้อมูลน้อย <strong>Cross Validation</strong> คือเพื่อนแท้ของคุณ</p>
  </div>

  <h3 className="text-xl font-semibold mb-3 mt-8 text-center"> Insight:</h3>
  <p className="italic text-center text-gray-700 dark:text-gray-300">
    “การประเมินโมเดลด้วยวิธีที่หลากหลาย คือการเตรียมโมเดลให้พร้อมรับทุกสนาม ไม่ใช่แค่สอบผ่านบนกระดาษ แต่ใช้ได้จริงในโลกจริง”  
  </p>
</section>

   {/* Section: ประเภทของ Cross Validation */}
<section id="types-of-cross-validation" className="mb-16 scroll-mt-32 min-h-[300px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Types of Cross Validation</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <p className="mb-4 leading-relaxed">
    Cross Validation คือกระบวนการแบ่งข้อมูลออกเป็นหลายส่วนเพื่อใช้ในการประเมินความสามารถของโมเดล
    โดยช่วยให้การประเมินไม่ขึ้นกับการแบ่งข้อมูลครั้งเดียว ซึ่งอาจนำไปสู่ความลำเอียง
    การเลือกประเภทของ Cross Validation ที่เหมาะสมจะช่วยให้เราเข้าใจศักยภาพของโมเดลในสถานการณ์ต่าง ๆ ได้ชัดเจนขึ้น
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">1. Holdout Validation</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <p className="mb-4 leading-relaxed">
    การแบ่งข้อมูลออกเป็นสองส่วน ได้แก่ Training Set และ Validation Set เช่น 80/20 หรือ 70/30 โดยฝึกโมเดลด้วย training set
    แล้วประเมินผลบน validation set วิธีนี้เรียบง่ายและใช้เวลาน้อย แต่ผลลัพธ์อาจขึ้นกับการแบ่งข้อมูลครั้งเดียว ซึ่งไม่เหมาะกับข้อมูลขนาดเล็ก
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">2. K-Fold Cross Validation</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <p className="mb-4 leading-relaxed">
    แบ่งข้อมูลออกเป็น K ส่วนที่มีขนาดใกล้เคียงกัน เรียกว่า “fold” จากนั้นทำการฝึกโมเดล K ครั้ง โดยในแต่ละครั้งจะใช้ 1 fold เป็น validation set
    และอีก K-1 fold ที่เหลือเป็น training set สุดท้ายนำผลลัพธ์จาก K ครั้งมาหาค่าเฉลี่ย
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ลดการพึ่งพาการแบ่งข้อมูลครั้งเดียว</li>
    <li>ช่วยให้ใช้ข้อมูลได้เต็มประสิทธิภาพ</li>
    <li>เหมาะกับชุดข้อมูลขนาดเล็กถึงกลาง</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 text-center">3. Stratified K-Fold (เหมาะสำหรับ Classification)</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <p className="mb-4 leading-relaxed">
    ใช้หลักการเหมือน K-Fold แต่จะรักษาสัดส่วนของแต่ละ class ให้เหมือนกันในทุก fold เพื่อป้องกันไม่ให้เกิด class imbalance
    เหมาะสำหรับงาน classification โดยเฉพาะในกรณีที่ข้อมูลมีการกระจายไม่สมดุล
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">4. Leave-One-Out Cross Validation (LOOCV)</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <p className="mb-4 leading-relaxed">
    เป็นกรณีพิเศษของ K-Fold โดยที่ K เท่ากับจำนวนตัวอย่างทั้งหมด ทุกรอบจะใช้ 1 ตัวอย่างเป็น validation และที่เหลือทั้งหมดเป็น training
    แม้จะให้ค่าประเมินที่ละเอียด แต่ใช้เวลาประมวลผลมาก และเหมาะเฉพาะกับชุดข้อมูลที่เล็กมากเท่านั้น
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">5. Repeated K-Fold</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <p className="mb-4 leading-relaxed">
    ทำ K-Fold Cross Validation หลายรอบ โดยแต่ละรอบจะสลับการแบ่ง fold ใหม่ เพิ่มความน่าเชื่อถือของผลลัพธ์
    เหมาะสำหรับการประเมินที่ต้องการความมั่นใจสูงและลดอิทธิพลจาก random seed
  </p>

  <h3 className="text-xl font-semibold mb-3">6. Time Series Split</h3>
  <p className="mb-4 leading-relaxed">
    ใช้กับข้อมูลลำดับเวลา (เช่น ราคาหุ้น, ข้อมูล IoT) ซึ่งไม่สามารถสลับลำดับข้อมูลได้เหมือน K-Fold
    ในแต่ละรอบของการ validation จะใช้ข้อมูลจากอดีตในการ train และข้อมูลจากอนาคตในการ test เพื่อเลียนแบบสถานการณ์จริง
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow my-6">
    <strong>Insight:</strong> ไม่ใช่ Cross Validation ทุกแบบจะเหมาะกับทุกปัญหา <br />
    เลือกให้สอดคล้องกับลักษณะของข้อมูล เช่นใช้ Stratified กับ Classification และใช้ Time Series Split กับข้อมูลที่มีลำดับเวลา
  </div>

  <h3 className="text-xl font-semibold mb-3">เปรียบเทียบภาพรวมแต่ละประเภท</h3>
  <div className="overflow-x-auto mb-4">
    <table className="table-auto w-full border-collapse border border-gray-400 text-sm text-left">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">ประเภท</th>
          <th className="border px-4 py-2">ข้อดี</th>
          <th className="border px-4 py-2">ข้อจำกัด</th>
          <th className="border px-4 py-2">เหมาะกับ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Holdout</td>
          <td className="border px-4 py-2">เร็ว, ใช้งานง่าย</td>
          <td className="border px-4 py-2">อาจไม่แม่นยำ, ขึ้นกับการแบ่ง</td>
          <td className="border px-4 py-2">ข้อมูลใหญ่</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">K-Fold</td>
          <td className="border px-4 py-2">ใช้ข้อมูลเต็ม, แม่นยำ</td>
          <td className="border px-4 py-2">ใช้เวลานานขึ้น</td>
          <td className="border px-4 py-2">ทั่วไป</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stratified</td>
          <td className="border px-4 py-2">คงสัดส่วน Class</td>
          <td className="border px-4 py-2">ใช้ได้กับ classification เท่านั้น</td>
          <td className="border px-4 py-2">Class imbalance</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">LOOCV</td>
          <td className="border px-4 py-2">แม่นยำมาก, ใช้ข้อมูลเต็ม</td>
          <td className="border px-4 py-2">ช้ามาก</td>
          <td className="border px-4 py-2">ข้อมูลน้อยมาก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Repeated K-Fold</td>
          <td className="border px-4 py-2">ลดอคติจากการแบ่ง fold</td>
          <td className="border px-4 py-2">ใช้เวลาและ compute เพิ่ม</td>
          <td className="border px-4 py-2">ต้องการผลเสถียร</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Time Series Split</td>
          <td className="border px-4 py-2">เหมาะกับข้อมูลลำดับเวลา</td>
          <td className="border px-4 py-2">ใช้ข้อมูลฝึกน้อยในรอบแรก</td>
          <td className="border px-4 py-2">Time series / Forecasting</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p className="mb-4 leading-relaxed">
    การเลือกประเภทของ Cross Validation ไม่ได้มีคำตอบที่ตายตัว ต้องพิจารณาจากหลายปัจจัย เช่น
    ประเภทของงาน, ปริมาณข้อมูล, ความสมดุลของ class, ความสัมพันธ์ตามลำดับเวลา และทรัพยากรที่มีอยู่ในการประมวลผล
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-800 p-4 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
    <strong>Best Practice:</strong> เริ่มจาก K-Fold หากไม่มีปัญหาเฉพาะ เช่น class imbalance หรือข้อมูลลำดับเวลา แล้วปรับใช้ให้เหมาะสมกับโจทย์
  </div>
</section>


     {/* Section: K-Fold Workflow */}
<section id="kfold-workflow" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">K-Fold Cross Validation: การทำงานแบบเป็นขั้นตอน</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <p className="mb-4 leading-relaxed">
    K-Fold Cross Validation เป็นเทคนิคที่ได้รับความนิยมมากที่สุดในการประเมินโมเดล เนื่องจากสามารถใช้ข้อมูลได้อย่างมีประสิทธิภาพโดยไม่ต้องเสียสละข้อมูลสำหรับ validation แยกออกมามากเกินไป 
    วิธีนี้ช่วยลดความเอนเอียงของการประเมิน และให้ผลที่เสถียรกว่า train/test split ธรรมดา
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">ภาพรวมของ K-Fold Cross Validation</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <p className="mb-4 leading-relaxed">
    แนวคิดหลักคือ การแบ่งข้อมูลทั้งหมดออกเป็น “K ส่วน (folds)” เท่า ๆ กัน จากนั้นโมเดลจะถูกฝึกและประเมิน K ครั้ง โดยในแต่ละครั้งจะใช้ 1 fold เป็น validation set และอีก K-1 fold เป็น training set
    เมื่อทำครบ K รอบแล้ว จะนำผลลัพธ์จากแต่ละรอบมาหาค่าเฉลี่ย เพื่อให้ได้คะแนนประเมินที่น่าเชื่อถือและไม่เอนเอียง
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">ขั้นตอนการทำงานอย่างละเอียด</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <ol className="list-decimal pl-6 space-y-2 mb-6">
    <li>กำหนดจำนวน K ที่ต้องการ เช่น 5 หรือ 10</li>
    <li>สุ่มและแบ่งชุดข้อมูลทั้งหมดออกเป็น K fold ที่ไม่ซ้ำกัน</li>
    <li>ทำการวนลูป K ครั้ง:</li>
    <ul className="list-disc pl-6">
      <li>ครั้งที่ i: ใช้ fold ที่ i เป็น validation set</li>
      <li>ใช้ fold อื่น ๆ (K-1 fold) เป็น training set</li>
      <li>ฝึกโมเดลด้วย training set</li>
      <li>วัดผลกับ validation set และบันทึกค่า metric เช่น Accuracy, F1, MSE ฯลฯ</li>
    </ul>
    <li>หลังจบ K รอบ ให้นำค่าที่ได้จากแต่ละรอบมาหาค่าเฉลี่ย (mean) และส่วนเบี่ยงเบน (std)</li>
  </ol>

  <h3 className="text-xl font-semibold mb-3">ข้อดีของ K-Fold</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ใช้ข้อมูลอย่างคุ้มค่า เพราะทุกข้อมูลได้ถูกใช้ทั้งในการฝึกและประเมิน</li>
    <li>ลดความเอนเอียงจากการสุ่ม train/test split</li>
    <li>ช่วยให้เข้าใจว่าประสิทธิภาพของโมเดลเสถียรหรือไม่ (ผ่านค่า std)</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">ข้อเสียของ K-Fold</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ต้องฝึกโมเดลหลายรอบ ทำให้ใช้เวลานานโดยเฉพาะกับโมเดลขนาดใหญ่</li>
    <li>อาจเกิดข้อมูลซ้ำในการแบ่ง fold หากไม่สุ่มดีพอ</li>
    <li>ไม่เหมาะกับ time series โดยตรง (ควรใช้ TimeSeriesSplit)</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">การเลือกจำนวน K ที่เหมาะสม</h3>
  <p className="mb-4 leading-relaxed">
    ค่าของ K ที่ใช้บ่อยที่สุดคือ 5 หรือ 10 เพราะให้ความสมดุลระหว่างความแม่นยำและเวลาในการฝึก หากใช้ K มากขึ้น จะได้ผลประเมินที่เสถียรกว่า แต่ต้องใช้เวลามากขึ้น
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>K = 5:</strong> ใช้เวลาน้อยกว่า แต่ความแม่นยำอาจผันผวนได้มากกว่า</li>
    <li><strong>K = 10:</strong> แม่นยำกว่า และลด variance ของผลลัพธ์</li>
    <li><strong>K = N (Leave-One-Out):</strong> ใช้ข้อมูลได้สูงสุด แต่เสียเวลามาก</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">K-Fold vs Holdout</h3>
  <div className="overflow-x-auto mb-4">
    <table className="min-w-full table-auto border border-gray-400 text-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-800">
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Holdout</th>
          <th className="border px-4 py-2">K-Fold</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การใช้ข้อมูล</td>
          <td className="border px-4 py-2">ใช้แค่บางส่วน</td>
          <td className="border px-4 py-2">ใช้ทั้งหมด</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความเอนเอียง</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">ต่ำ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">เวลาในการฝึก</td>
          <td className="border px-4 py-2">เร็ว</td>
          <td className="border px-4 py-2">ช้า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">เหมาะกับข้อมูล</td>
          <td className="border px-4 py-2">ขนาดใหญ่</td>
          <td className="border px-4 py-2">ขนาดเล็ก-กลาง</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-3">การแสดงผลลัพธ์</h3>
  <p className="mb-4 leading-relaxed">
    นอกจากการพิมพ์ค่าเฉลี่ยของ Accuracy หรือ F1 แล้ว การแสดง standard deviation ช่วยให้เข้าใจความเสถียรของโมเดลได้ เช่น
  </p>
  <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre-wrap mb-4">
  <code>
    from sklearn.model_selection import cross_val_score<br />
    scores = cross_val_score(model, X, y, cv=5)<br />
    print(f"Accuracy: {'{'}scores.mean():.2f{'}'} ± {'{'}scores.std():.2f{'}'}")
  </code>
</pre>


  <h3 className="text-xl font-semibold mb-3">ข้อแนะนำการใช้งาน</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ใช้ K-Fold คู่กับ <code>GridSearchCV</code> เพื่อปรับ hyperparameters</li>
    <li>ใช้ <code>StratifiedKFold</code> หากเป็นงาน classification ที่ class ไม่สมดุล</li>
    <li>สำหรับ time-series ให้ใช้ <code>TimeSeriesSplit</code> แทน</li>
    <li>ควร set random_state เพื่อให้ reproducible</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <strong>Insight:</strong><br />
    การฝึกโมเดลให้ดีเหมือนการสอบให้ผ่านหลายวิชา K-Fold คือการสอบซ้ำหลายรอบ เพื่อให้มั่นใจว่าโมเดลไม่ได้เก่งแค่บางรอบ แต่เข้าใจเนื้อหาจริงในทุกบริบท
  </div>
</section>


    {/* Section: Cross Validation ในการเลือกโมเดล */}
<section id="cv-model-selection" className="mb-16 scroll-mt-32 min-h-[500px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Cross Validation ในการเลือกโมเดล</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <p className="mb-4 leading-relaxed">
    การเลือกโมเดลที่เหมาะสมไม่ใช่เพียงแค่การทดลองแบบสุ่มแล้วดูผลลัพธ์ที่ดีที่สุด แต่เป็นกระบวนการที่ต้องอาศัยการประเมินแบบรอบคอบ
    โดยเฉพาะในปัจจุบันที่โมเดลมีความซับซ้อนและหลากหลาย การใช้ <strong>Cross Validation (CV)</strong> จึงเป็นหัวใจสำคัญ
    ในการประเมินว่าโมเดลหนึ่ง ๆ มีประสิทธิภาพที่แท้จริงหรือไม่ โดยไม่พึ่งพาแค่ Train/Test Split ที่อาจมีความลำเอียง
  </p>

  <h3 className="text-xl font-semibold mb-3">1. ทำไม Cross Validation ถึงสำคัญในการเลือกโมเดล?</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ลดความลำเอียงจากการแบ่งชุดข้อมูลแบบสุ่มเพียงครั้งเดียว</li>
    <li>ช่วยให้เห็นประสิทธิภาพของโมเดลจากมุมมองหลายชุดข้อมูล</li>
    <li>สามารถเปรียบเทียบโมเดลหลายแบบได้อย่างเป็นธรรม</li>
    <li>ให้ผลลัพธ์ที่เสถียรกว่าการประเมินด้วย Test Set เพียงรอบเดียว</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">2. ใช้ Cross Validation เพื่อเปรียบเทียบโมเดล</h3>
  <p className="mb-4 leading-relaxed">
    สมมุติเรามีโมเดล 3 แบบ ได้แก่ Logistic Regression, Random Forest และ XGBoost หากประเมินแค่ด้วย Test Set เดียว
    อาจได้ผลลัพธ์ที่ชี้นำผิดจากความจริง เช่น Test Set นั้นอาจเหมาะกับโมเดลหนึ่งแต่ไม่เหมาะกับอีกโมเดล
    Cross Validation ช่วยให้เราแบ่งข้อมูลออกเป็นหลายชุด และวนเทรน+วัดผลหลายรอบเพื่อหาค่าเฉลี่ย performance
    ทำให้การเปรียบเทียบมีความยุติธรรมมากขึ้น
  </p>

  <div className="overflow-x-auto my-6">
    <table className="min-w-full table-auto border border-gray-300 dark:border-gray-600 text-sm text-center">
      <thead className="bg-gray-100 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">Model</th>
          <th className="border px-4 py-2">Accuracy (CV)</th>
          <th className="border px-4 py-2">Std Dev</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Logistic Regression</td>
          <td className="border px-4 py-2">0.82</td>
          <td className="border px-4 py-2">±0.03</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Random Forest</td>
          <td className="border px-4 py-2">0.86</td>
          <td className="border px-4 py-2">±0.02</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">XGBoost</td>
          <td className="border px-4 py-2">0.87</td>
          <td className="border px-4 py-2">±0.01</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-3">3. ใช้ร่วมกับ Grid Search / Hyperparameter Tuning</h3>
  <p className="mb-4 leading-relaxed">
    ในการหา hyperparameter ที่เหมาะสมที่สุด เช่น ค่า learning rate, depth ของ tree หรือจำนวน neurons ใน neural network
    เราสามารถใช้ Cross Validation ภายใน Grid Search เพื่อประเมินว่าแต่ละชุดค่าทำงานได้ดีแค่ไหน
    โดยค่าที่ได้จากคะแนนเฉลี่ยของ Cross Validation จะนำมาใช้เลือกพารามิเตอร์ที่ดีที่สุด
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre-wrap">
{`from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-3">4. ช่วยเลือก Model Complexity ที่เหมาะสม</h3>
  <p className="mb-4 leading-relaxed">
    ความซับซ้อนของโมเดลมีผลต่อความสามารถในการ generalize หากโมเดลซับซ้อนเกินไป อาจ overfit กับข้อมูลฝึก
    แต่หากเรียบง่ายเกินไป อาจ underfit ข้อมูลจริง Cross Validation จะช่วยให้เรารู้ว่าโมเดลแบบไหนให้ผลลัพธ์ดีที่สุดโดยไม่ต้องหวังจาก Test Set
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>Train/Validation Accuracy ต่างกันมาก → โมเดลอาจ overfit</li>
    <li>Train/Validation Accuracy ต่างกันน้อยแต่ต่ำทั้งคู่ → underfit</li>
    <li>Train/Validation Accuracy ใกล้เคียงและสูง → โมเดลสมดุล</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">5. คำแนะนำในการใช้ Cross Validation อย่างถูกต้อง</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>เลือก K ที่เหมาะสม เช่น K=5 หรือ K=10 เป็นมาตรฐาน</li>
    <li>ใช้ Stratified K-Fold หากทำ classification โดย class ไม่สมดุล</li>
    <li>ไม่ใช้ Test Set ซ้ำในการปรับโมเดล</li>
    <li>ใช้ค่าเฉลี่ย CV เพื่อสรุป ไม่ใช้ค่าจากรอบใดรอบหนึ่ง</li>
    <li>ใช้ Standard Deviation ประกอบ เพื่อดูเสถียรภาพ</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">6. Insight: Cross Validation คือเครื่องมือวัด “ความน่าเชื่อถือ”</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">
      โมเดลที่แม่นใน Training Set ไม่ได้แปลว่าแม่นในชีวิตจริง — แต่ Cross Validation จะเป็นเหมือนการสอบหลายรอบในหลายห้องสอบ
      ถ้าโมเดลสอบผ่านทุกสนามสอบได้ดี → นั่นแหละคือโมเดลที่น่าเชื่อถือ
    </p>
    <ul className="list-disc pl-6 space-y-2 text-sm mt-2">
      <li><strong>เทรนดี → ไม่พอ</strong> ต้องสอบได้ด้วย</li>
      <li><strong>สอบผ่านหลายรอบ → น่าเชื่อถือ</strong></li>
      <li><strong>ไม่ใช้ Test ซ้ำ</strong> เพราะเหมือนดูข้อสอบล่วงหน้า</li>
    </ul>
  </div>
</section>


{/* Section: ตัวอย่างโค้ด */}
<section id="code-example" className="mb-16 scroll-mt-32 min-h-[500px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">ตัวอย่างโค้ดการใช้งาน Cross Validation</h2>

  <p className="mb-4 leading-relaxed">
    ในส่วนนี้เราจะมาเรียนรู้การใช้งาน Cross Validation ผ่านไลบรารี <strong>Scikit-learn</strong> ซึ่งเป็นหนึ่งในไลบรารียอดนิยมของ Python สำหรับ Machine Learning โดยจะยกตัวอย่างทั้งการใช้ <code>cross_val_score</code>, <code>KFold</code>, <code>StratifiedKFold</code> รวมไปถึงการใช้งานร่วมกับ <code>Pipeline</code> และ <code>GridSearchCV</code>
  </p>

  <h3 className="text-xl font-semibold mb-2">1. ตัวอย่างพื้นฐาน: cross_val_score</h3>
  <p className="mb-4 leading-relaxed">
    ฟังก์ชัน <code>cross_val_score</code> เป็นวิธีง่ายที่สุดในการทำ Cross Validation โดยไม่ต้องสร้างลูปเอง ใช้สำหรับการวัดค่า score แบบอัตโนมัติในแต่ละรอบของ K-Fold
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
{`from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=5)

print("Accuracy per fold:", scores)
print("Mean Accuracy:", scores.mean())`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">2. การใช้งาน KFold แบบกำหนดเอง</h3>
  <p className="mb-4 leading-relaxed">
    หากต้องการควบคุมลำดับหรือการแบ่ง fold ด้วยตัวเอง สามารถใช้คลาส <code>KFold</code> ร่วมกับ <code>cross_val_score</code> ได้ โดยกำหนดจำนวน fold, การสับลำดับ และ seed
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
{`from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

print("Fold Scores:", scores)
print("Mean:", scores.mean())`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">3. ใช้ StratifiedKFold สำหรับ Classification</h3>
  <p className="mb-4 leading-relaxed">
    สำหรับปัญหา Classification ที่คลาสไม่สมดุล ควรใช้ <code>StratifiedKFold</code> ซึ่งจะรักษาสัดส่วนของแต่ละคลาสไว้ในทุก fold เพื่อป้องกันความลำเอียง
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
{`from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

print("Stratified Fold Scores:", scores)
print("Mean Accuracy:", scores.mean())`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">4. ใช้งานร่วมกับ Pipeline</h3>
  <p className="mb-4 leading-relaxed">
    การใช้ <code>Pipeline</code> ทำให้สามารถรวมหลายขั้นตอนเข้าด้วยกัน เช่น การปรับขนาดข้อมูล (Scaler) และการเทรนโมเดลในขั้นตอนเดียว พร้อมใช้ Cross Validation ได้เลย
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
{`from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
scores = cross_val_score(pipeline, X, y, cv=5)

print("Pipeline Accuracy:", scores.mean())`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">5. การค้นหา Hyperparameter ที่ดีที่สุดด้วย GridSearchCV</h3>
  <p className="mb-4 leading-relaxed">
    <code>GridSearchCV</code> จะทำการ Cross Validation โดยอัตโนมัติในทุกค่า hyperparameter ที่กำหนดไว้ใน grid และเลือกค่าที่ดีที่สุดให้เรา
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
{`from sklearn.model_selection import GridSearchCV

param_grid = { 'C': [0.1, 1, 10] }
grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid.fit(X, y)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">6. การใช้คะแนนเฉลี่ยและการประเมินผล</h3>
  <p className="mb-4 leading-relaxed">
    หลังจากได้ผลลัพธ์แต่ละ fold แล้ว การดูค่าเฉลี่ย (mean) และส่วนเบี่ยงเบนมาตรฐาน (std) จะช่วยให้เรารู้ว่าโมเดลของเรามี performance ที่สม่ำเสมอแค่ไหนในแต่ละชุดข้อมูลที่แตกต่างกัน
  </p>

  <div className="overflow-x-auto mb-4">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
{`import numpy as np

mean_score = scores.mean()
std_score = scores.std()

print(f"Mean Accuracy: {mean_score:.4f}")
print(f"Standard Deviation: {std_score:.4f}")`}
    </pre>
  </div>
</section>



            {/* Section: Evaluation Metrics */}
            <section id="metrics" className="mb-16 scroll-mt-32 min-h-[500px]">
          <h2 className="text-2xl font-semibold mb-4 text-center">Metric ที่ใช้ในการประเมินโมเดล</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img13} />
          </div>

          <p className="mb-4 leading-relaxed">
            การประเมินประสิทธิภาพของโมเดลใน Machine Learning ถือเป็นหัวใจสำคัญที่ช่วยให้เราตัดสินใจได้ว่าโมเดลควรถูกใช้งานต่อหรือปรับปรุงเพิ่มเติม การเลือก Metric ที่เหมาะสมขึ้นอยู่กับประเภทของปัญหา เช่น การจำแนกประเภท (classification) หรือการทำนายค่าต่อเนื่อง (regression) และขึ้นอยู่กับลักษณะของข้อมูล เช่น ความไม่สมดุลของ class หรือความสำคัญของ false positives/negatives
          </p>

          <h3 className="text-xl font-semibold mb-3">1. Accuracy</h3>
          <p className="mb-4 leading-relaxed">
            เป็น metric พื้นฐานที่ใช้บ่อยใน classification โดยคำนวณจากสัดส่วนของจำนวนตัวอย่างที่ทำนายถูกต้องทั้งหมดต่อจำนวนตัวอย่างทั้งหมด อย่างไรก็ตาม accuracy ไม่เหมาะกับข้อมูลที่ class ไม่สมดุล เช่น กรณีที่ class ใด class หนึ่งมีจำนวนมากกว่าอีก class มาก ๆ
          </p>

          <h3 className="text-xl font-semibold mb-3">2. Precision, Recall และ F1-Score</h3>
          <p className="mb-4 leading-relaxed">
            - <strong>Precision:</strong> สัดส่วนของการทำนายว่าเป็น positive แล้วถูกต้องจริง ๆ
            <br />- <strong>Recall:</strong> สัดส่วนของตัวอย่าง positive ทั้งหมดที่ถูกจับได้โดยโมเดล
            <br />- <strong>F1-Score:</strong> ค่าเฉลี่ยแบบ harmonic mean ของ precision และ recall ซึ่งให้ค่าที่สมดุลระหว่างทั้งสอง
          </p>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li>Precision สูง → โมเดลไม่พลาดทำนาย false positive</li>
            <li>Recall สูง → โมเดลไม่พลาดจับ false negative</li>
            <li>F1 เหมาะสำหรับกรณีที่ต้องการ balance ทั้ง precision และ recall</li>
          </ul>

          <h3 className="text-xl font-semibold mb-3">3. Confusion Matrix</h3>
          <p className="mb-4 leading-relaxed">
            ใช้ดูผลการทำนายแยกตาม class อย่างละเอียด แสดงจำนวนของ True Positive (TP), False Positive (FP), True Negative (TN), และ False Negative (FN) ทำให้สามารถวิเคราะห์ข้อผิดพลาดที่โมเดลทำได้อย่างชัดเจน เช่น โมเดลทำนายว่าผู้ป่วยไม่เป็นโรค ทั้งที่จริงแล้วเป็น (FN) เป็นต้น
          </p>

          <h3 className="text-xl font-semibold mb-3">4. ROC Curve และ AUC (Area Under Curve)</h3>
          <p className="mb-4 leading-relaxed">
            ROC Curve แสดงความสัมพันธ์ระหว่าง True Positive Rate (Recall) กับ False Positive Rate ที่ค่า Threshold ต่าง ๆ ยิ่งกราฟอยู่ใกล้มุมบนซ้ายมากเท่าไร ยิ่งแสดงว่าโมเดลดี AUC คือพื้นที่ใต้กราฟ ROC ซึ่งบ่งบอกว่าโมเดลสามารถแยก class ได้ดีเพียงใด (1 คือดีที่สุด)
          </p>

          <h3 className="text-xl font-semibold mb-3">5. Log Loss</h3>
          <p className="mb-4 leading-relaxed">
            ใช้ในกรณี classification ที่มีการทำนายแบบ probabilistic โดยจะ penalize ความมั่นใจที่ผิดมากกว่า เช่น หากทำนายว่า class A มีความน่าจะเป็น 0.99 แต่คำตอบจริงคือ class B จะโดนค่าความผิดพลาดสูง
          </p>

          <h3 className="text-xl font-semibold mb-3">6. Mean Squared Error (MSE) และ Mean Absolute Error (MAE)</h3>
          <p className="mb-4 leading-relaxed">
            ในปัญหา regression MSE และ MAE เป็น metric ที่ใช้วัดความคลาดเคลื่อนของค่าทำนาย:
            <br />- <strong>MSE:</strong> เน้นค่าความผิดพลาดที่มากผิดปกติ เพราะยกกำลังสอง (sensitive to outliers)
            <br />- <strong>MAE:</strong> วัดความผิดพลาดเฉลี่ยที่แท้จริง เหมาะกับข้อมูลที่มี outliers
          </p>

          <h3 className="text-xl font-semibold mb-3">7. R² Score (Coefficient of Determination)</h3>
          <p className="mb-4 leading-relaxed">
            ใช้วัดความสามารถของโมเดลในการอธิบายความแปรปรวนของข้อมูลจริง ค่า R² ใกล้ 1 แสดงว่าโมเดลอธิบายข้อมูลได้ดี ส่วนค่าที่ติดลบหมายถึงโมเดลทำงานแย่กว่าการทำนายค่าคงที่
          </p>

          <h3 className="text-xl font-semibold mb-3">8. Custom Metrics</h3>
          <p className="mb-4 leading-relaxed">
            บางกรณี metric มาตรฐานอาจไม่เหมาะสม จึงมีการออกแบบ custom metrics เช่น Profit per prediction, Time to detect anomaly, หรือ Cost-sensitive metrics ที่พิจารณาผลกระทบทางธุรกิจจริง
          </p>

          <h3 className="text-xl font-semibold mb-3">9. การเลือก Metric ให้เหมาะสม</h3>
          <p className="mb-4 leading-relaxed">
            ควรพิจารณาจาก:
          </p>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li>ลักษณะปัญหา (Classification vs Regression)</li>
            <li>ความสำคัญของ false positive/negative</li>
            <li>ความไม่สมดุลของ class</li>
            <li>ลักษณะของธุรกิจและการใช้งานจริง</li>
          </ul>

          <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow mt-6">
            <strong>Insight:</strong><br/>
            การเลือก metric ที่ถูกต้องสำคัญพอ ๆ กับการออกแบบโมเดลที่ดี เพราะมันคือกระจกสะท้อนว่าความฉลาดของโมเดลนั้น ตรงกับสิ่งที่เราต้องการหรือไม่
          </div>
        </section>


        <section id="metrics-insight" className="mb-16 scroll-mt-32 min-h-[500px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">Insight เปรียบเทียบ Metric</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img14} />
          </div>

  <p className="mb-4 leading-relaxed">
    การเลือก Metric ที่เหมาะสมในการประเมินโมเดล Machine Learning ไม่ใช่เรื่องง่าย เพราะแต่ละ Metric มีจุดแข็ง จุดอ่อน และเหมาะกับบริบทที่แตกต่างกัน
    Insight ที่ลึกซึ้งจะช่วยให้เราเลือก Metric ได้ตรงกับลักษณะของข้อมูล และเป้าหมายของการเรียนรู้
  </p>

  <h3 className="text-xl font-semibold mb-2">1. ทำไม Accuracy ถึงหลอกเราได้?</h3>
  <p className="mb-4 leading-relaxed">
    สมมุติว่าคุณกำลังสร้างโมเดลตรวจจับโรคร้ายแรงที่พบในประชากรเพียง 1% หากโมเดลทำนายว่า “ไม่มีใครเป็นโรค” เลยทุกคน 
    จะได้ Accuracy = 99% ทันที ทั้งที่จริงแล้วโมเดลล้มเหลวในการตรวจจับคนที่มีโรคจริง!
  </p>
  <div className="bg-yellow-50 dark:bg-yellow-800 p-4 rounded-xl border-l-4 border-yellow-500 mb-6">
    <strong>Insight:</strong> เมื่อ Class ไม่สมดุล (เช่น 99% vs 1%) การใช้ Accuracy เพียงอย่างเดียวไม่สามารถสะท้อนประสิทธิภาพของโมเดลได้
  </div>

  <h3 className="text-xl font-semibold mb-2">2. Precision vs Recall: ควรใช้เมื่อไร?</h3>
  <p className="mb-4 leading-relaxed">
    - <strong>Precision</strong> คือสัดส่วนของ Positive ที่ทำนายถูกจริงจากทั้งหมดที่ทำนายว่าเป็น Positive<br />
    - <strong>Recall</strong> คือสัดส่วนของ Positive ที่ทำนายถูกจริงจากทั้งหมดที่ควรจะเป็น Positive
  </p>

  <div className="grid md:grid-cols-2 gap-4 mb-6">
    <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold text-lg mb-2">ใช้ Precision เมื่อต้องการลด False Positive</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ตรวจจับ Spam: ไม่อยากบล็อกอีเมลที่ไม่ใช่ Spam</li>
        <li>โมเดลเลือกคนสัมภาษณ์: ไม่อยากพลาดคนเก่งที่ควรถูกเรียก</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold text-lg mb-2">ใช้ Recall เมื่อต้องการลด False Negative</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>การตรวจโรคร้ายแรง: ไม่อยากปล่อยให้คนที่เป็นโรคหลุดไป</li>
        <li>ตรวจจับผู้ก่อการร้าย: ยอมให้เตือนผิด ยังดีกว่าปล่อยให้รอด</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-2">3. F1-Score: เมตริกกลางที่สมดุล</h3>
  <p className="mb-4 leading-relaxed">
    F1 คือค่าเฉลี่ยฮาร์มอนิกของ Precision และ Recall ช่วยวัดสมดุลของโมเดลโดยเฉพาะในกรณีที่ Class ไม่สมดุล และเราให้ความสำคัญทั้งการจับให้ถูก (Precision)
    และไม่พลาด (Recall) ไปพร้อมกัน
  </p>
  <p className="mb-4 leading-relaxed">
    ค่า F1 ที่ดีอยู่ระหว่าง 0 ถึง 1 โดยที่ 1 หมายถึงสมบูรณ์แบบทั้ง Precision และ Recall
  </p>

  <h3 className="text-xl font-semibold mb-2">4. ROC Curve และ Threshold</h3>
  <p className="mb-4 leading-relaxed">
    ROC (Receiver Operating Characteristic) Curve คือกราฟแสดงความสัมพันธ์ระหว่าง True Positive Rate (Recall) กับ False Positive Rate เมื่อเราปรับค่า Threshold
    การเลือก Threshold มีผลต่อความแม่นยำของโมเดลโดยตรง
  </p>

  <p className="mb-4 leading-relaxed">
    พื้นที่ใต้กราฟ ROC (AUC - Area Under Curve) เป็นตัวบ่งชี้คุณภาพโดยรวมของโมเดล ยิ่ง AUC ใกล้ 1 ยิ่งดี
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-800 p-4 rounded-xl border-l-4 border-yellow-500 mb-6">
    <strong>Insight:</strong> ในการ deploy โมเดลจริง อย่าลืมปรับ threshold ตามจุดสมดุลที่เหมาะสมต่อบริบท ไม่ใช่ใช้ค่า default ที่ 0.5 เสมอไป
  </div>

  <h3 className="text-xl font-semibold mb-2">5. Metric สำหรับ Regression</h3>
  <p className="mb-4 leading-relaxed">
    การประเมินโมเดล Regression ต้องใช้ Metric ที่ต่างจาก Classification เช่น:
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>MSE (Mean Squared Error):</strong> ลงโทษความผิดพลาดที่ห่างไกลมากเป็นพิเศษ</li>
    <li><strong>MAE (Mean Absolute Error):</strong> วัดค่าเฉลี่ยของความผิดพลาดแบบไม่เน้น outlier</li>
    <li><strong>R² Score:</strong> วัดว่าสัดส่วนของความแปรปรวนในข้อมูลสามารถอธิบายได้โดยโมเดล</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">6. ควรใช้ Metric หลายตัวพร้อมกัน</h3>
  <p className="mb-4 leading-relaxed">
    ไม่ควรพึ่งพาเพียงค่าเดียว เช่น Accuracy หรือ F1-Score เพราะมันอาจไม่ครอบคลุมทุกมิติของปัญหา เช่น precision ดีแต่ recall แย่ หรือ error ต่ำแต่โมเดล bias มาก
  </p>
  <p className="mb-4 leading-relaxed">
    การดู metric หลายตัวพร้อมกัน เช่น Accuracy + F1 + AUC ช่วยให้เห็นภาพรวมและข้อจำกัดของโมเดลได้ชัดเจนขึ้น
  </p>

  <h3 className="text-xl font-semibold mb-2">7. Insight สุดท้าย</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow-xl">
    <p className="mb-4 text-lg font-semibold">
      “ไม่มี Metric ใดที่ดีที่สุด มีแต่ Metric ที่เหมาะสมกับบริบทที่คุณกำลังเผชิญ”
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>อย่าเลือกโมเดลจาก Accuracy สูงสุดเพียงอย่างเดียว</li>
      <li>ให้พิจารณาภาพรวม เช่น Class balance, ค่า threshold, ความเสี่ยงของ false positive/negative</li>
      <li>ใช้ Metric ที่สะท้อนเป้าหมายจริงของโปรเจกต์ เช่น ปลอดภัย, ประหยัด, หรือแม่นยำ</li>
    </ul>
  </div>
</section>

<section id="best-practice" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Best Practice ในการประเมินโมเดล</h2>
       <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img15} />
          </div>

  <p className="mb-4 leading-relaxed">
    การประเมินโมเดลอย่างเป็นระบบไม่ใช่เพียงการดูค่า accuracy หรือ score ตัวใดตัวหนึ่งเท่านั้น แต่ต้องพิจารณาปัจจัยหลายด้านอย่างรอบคอบ เพื่อให้มั่นใจว่าโมเดลที่เลือกมีความเหมาะสมกับปัญหาและสามารถใช้งานได้ในสถานการณ์จริง โดยเฉพาะอย่างยิ่งในงานที่ข้อมูลไม่สมดุล หรือมีผลกระทบสูงจากความผิดพลาด.
  </p>

  <h3 className="text-xl font-semibold mb-3">1. แยกข้อมูลอย่างเป็นระบบ: Train, Validation, Test</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ควรแบ่งข้อมูลออกเป็น 3 ส่วน: <strong>Train</strong> สำหรับฝึกโมเดล, <strong>Validation</strong> สำหรับเลือกโมเดลและปรับค่า, และ <strong>Test</strong> สำหรับวัดผลสุดท้าย.</li>
    <li>อย่าใช้ Test set ซ้ำในการปรับโมเดล เพราะจะทำให้ผลประเมินลำเอียง.</li>
    <li>ในกรณีข้อมูลน้อย ควรใช้ K-Fold Cross Validation แทนการแบ่งข้อมูลแบบ Holdout.</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">2. ประเมินด้วยหลาย Metric พร้อมกัน</h3>
  <p className="mb-4 leading-relaxed">
    การดูเพียงค่า accuracy อาจหลอกได้ โดยเฉพาะเมื่อ class ไม่สมดุล เช่น ในกรณีที่ 95% ของข้อมูลเป็น class เดียว โมเดลที่ทำนาย class เดียวตลอดก็ยังได้ accuracy สูงถึง 95% แต่ไม่มีประโยชน์.
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ใช้ <strong>Precision, Recall, F1-Score</strong> ร่วมกันเสมอสำหรับ classification.</li>
    <li>ใช้ <strong>Confusion Matrix</strong> เพื่อดูรายละเอียดว่าโมเดลพลาดที่ไหน.</li>
    <li>ใน regression ใช้ <strong>MSE, RMSE, MAE</strong> เพื่อดูทั้งค่าเฉลี่ยและค่าความเบี่ยงเบน.</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">3. อย่าใช้ Test Set ซ้ำๆ</h3>
  <p className="mb-4 leading-relaxed">
    ข้อมูล Test มีไว้ใช้เพียงครั้งเดียวในตอนสุดท้าย เพื่อให้สะท้อนผลลัพธ์ในโลกจริง. หากใช้ซ้ำ ๆ โมเดลอาจ "เรียนรู้" test set ไปโดยไม่รู้ตัว ซึ่งทำให้ผลประเมินบิดเบือน.
  </p>

  <h3 className="text-xl font-semibold mb-3">4. ใช้ Visualization ช่วยในการวิเคราะห์</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>แสดง <strong>Confusion Matrix</strong> แบบ heatmap เพื่อให้เข้าใจได้ง่าย.</li>
    <li>ใช้ <strong>ROC Curve</strong> และ <strong>Precision-Recall Curve</strong> เพื่อวิเคราะห์ trade-off.</li>
    <li>สำหรับ regression ให้ plot เส้นค่าทำนายเทียบกับค่าจริง (y_pred vs y_true).</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">5. สังเกตพฤติกรรมของโมเดลในกลุ่มย่อยของข้อมูล</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลอาจมี performance ดีโดยรวมแต่แย่มากในกลุ่มย่อยบางกลุ่ม เช่น gender, age หรือ segment ลูกค้า. 
    ควรวิเคราะห์แยกเป็นกลุ่มย่อยเพื่อตรวจสอบ fairness และ bias.
  </p>

  <h3 className="text-xl font-semibold mb-3">6. ใช้ Unseen Data เสมอก่อน Deploy</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ควรมีชุดข้อมูลที่ไม่เกี่ยวข้องกับกระบวนการ train/validate เลย เพื่อทดสอบ performance จริงใน field.</li>
    <li>หรือใช้ live A/B Testing ถ้าอยู่ในระบบ production เพื่อเปรียบเทียบ performance.</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">7. อย่าลืมดู Variance ของผลลัพธ์</h3>
  <p className="mb-4 leading-relaxed">
    การดูแค่ค่าเฉลี่ยของ performance ไม่พอ ต้องดูการกระจายของ performance ด้วย เช่น standard deviation ของ score แต่ละ fold.
    หาก variance สูง แสดงว่าโมเดลไม่เสถียร.
  </p>

  <h3 className="text-xl font-semibold mb-3">8. ทำ Experiment Tracking อย่างมีระบบ</h3>
  <p className="mb-4 leading-relaxed">
    บันทึกว่าแต่ละรอบที่ train ใช้พารามิเตอร์อะไร, random seed อะไร, dataset ไหน เพื่อสามารถทำซ้ำหรือย้อนกลับมาเปรียบเทียบภายหลังได้.
    ควรใช้เครื่องมืออย่าง MLflow, Weights & Biases หรือแม้แต่ Excel ก็ยังดีกว่าไม่เก็บเลย.
  </p>

  <div className="mt-6 bg-yellow-100 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
    <p className="text-lg font-semibold mb-3">Insight:</p>
    <p>
      "การประเมินโมเดลที่ดีไม่ใช่แค่หาโมเดลที่ให้คะแนนดีที่สุด แต่คือการเข้าใจพฤติกรรมของโมเดล ภายใต้สถานการณ์ที่เปลี่ยนไป — เพื่อให้เรามั่นใจว่าโมเดลจะไม่เพียงแค่เรียนรู้จากอดีต แต่จะยังตอบได้ถูกในอนาคตด้วย."
    </p>
  </div>
</section>
<section id="insight" className="mb-16 scroll-mt-32 min-h-[500px]">
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow-xl">
    <h2 className="text-2xl font-bold mb-6 text-center"> Insight ตอนท้าย: ประเมินให้ลึก เข้าใจให้จริง</h2>

    <p className="mb-4 leading-relaxed">
      ในยุคของข้อมูลขนาดใหญ่และโมเดลที่ทรงพลังมากขึ้นทุกวัน การประเมินโมเดลไม่ใช่แค่การดูค่า Accuracy หรือ Loss เพียงตัวเดียว
      แต่มันคือกระบวนการที่ต้องพิจารณาหลายมิติ ทั้งด้าน performance, ความเข้าใจข้อมูล, ความยืดหยุ่นในการใช้งานจริง และความโปร่งใสของการตัดสินใจ
    </p>

    <p className="mb-4 leading-relaxed">
      Cross Validation คือเสาหลักของการประเมินโมเดลในยุคใหม่ มันช่วยให้เราเห็นพฤติกรรมของโมเดลจากมุมมองต่าง ๆ
      ไม่ใช่แค่การฝึกบนชุดข้อมูลชุดเดียวแล้วหวังว่าจะโชคดี
      แต่คือการ “ทดลองสอบหลายรอบในสนามจริง” แล้ววัดผลเฉลี่ยว่านักเรียนของเราทำได้ดีแค่ไหน
    </p>

    <p className="mb-4 leading-relaxed">
      หลายครั้งที่โมเดลดูดีมากใน Training Set และแม้แต่ใน Validation Set แบบเดิม ๆ
      แต่พอเปลี่ยนบริบทนิดเดียว เช่น ข้อมูลมาจากช่วงเวลาที่ต่างออกไป หรือกลุ่มผู้ใช้งานใหม่ ผลลัพธ์กลับเละไม่เป็นท่า
      นั่นเพราะการประเมินไม่ได้รัดกุมพอ หรือเลือก Metric ที่ไม่เหมาะสมกับปัญหา
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-3"> การเลือก Metric สำคัญพอ ๆ กับการเลือกโมเดล</h3>
    <ul className="list-disc pl-6 space-y-2 mb-4">
      <li>Accuracy ใช้ได้ดีในข้อมูลที่ class distribution สมดุล แต่จะหลอกเราในกรณี imbalance</li>
      <li>F1-Score เหมาะกับกรณีที่ต้องการบาลานซ์ Precision และ Recall โดยเฉพาะในงาน classification</li>
      <li>ROC-AUC ช่วยวัดศักยภาพของโมเดลโดยไม่อิง threshold เดียว</li>
      <li>สำหรับ regression, MAE และ MSE สะท้อนความผิดพลาดในมุมต่างกัน: MAE แสดงความคลาดเคลื่อนจริง, MSE ขยายความผิดพลาดใหญ่ ๆ</li>
    </ul>

    <p className="mb-4 leading-relaxed">
      โมเดลที่ดีจึงไม่ใช่แค่แม่นยำในอดีต แต่ต้อง “สอบผ่านในบริบทใหม่” อย่างสม่ำเสมอ
      เพราะในโลกจริง ข้อมูลเปลี่ยนแปลงตลอดเวลา ความต้องการของผู้ใช้งานก็เปลี่ยนแปลงเช่นกัน
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-3">ความโปร่งใส: ความสามารถที่หลายคนมองข้าม</h3>
    <p className="mb-4 leading-relaxed">
      เราอาจจะสร้างโมเดลที่มี Accuracy สูง แต่ถ้ามันไม่สามารถอธิบายได้ว่า “ทำไมถึงตัดสินใจแบบนั้น” 
      โมเดลก็จะถูกตั้งคำถามในเชิงจริยธรรม ความน่าเชื่อถือ และการนำไปใช้ในสภาพแวดล้อมที่มีความเสี่ยงสูง เช่น การแพทย์, การเงิน, กฎหมาย
    </p>

    <p className="mb-4 leading-relaxed">
      การประเมินที่ดีควรมีส่วนของ Explainability เช่น LIME, SHAP หรือวิธี Visualize feature importance
      เพื่อให้เรามองเห็นว่าเบื้องหลังการทำนายคืออะไร และมีเหตุผลเพียงพอหรือไม่
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-3"> การเรียนรู้ไม่ควรหยุดที่ Validation Score</h3>
    <p className="mb-4 leading-relaxed">
      ความเข้าใจลึกลงไปถึงความผิดพลาด (Error Analysis) ช่วยให้เราเห็นว่าโมเดลพลาดตรงไหน และพลาดด้วยเหตุผลอะไร
      การวิเคราะห์ Confusion Matrix หรือดูตัวอย่างที่โมเดลทำนายผิด คือวิธีที่เรียบง่ายแต่ทรงพลังที่สุดในการพัฒนาโมเดลให้ดีขึ้น
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-3"> เปรียบเทียบให้เห็นภาพ</h3>
    <ul className="list-disc pl-6 space-y-2 mb-4">
      <li><strong>โมเดลที่ดีแต่ไม่ประเมิน:</strong> เหมือนนักกีฬาที่ฝึกอย่างหนัก แต่ไม่เคยลองลงแข่งสนามจริง</li>
      <li><strong>โมเดลที่แม่นแต่ไม่โปร่งใส:</strong> เหมือนหมอดูที่ทำนายถูก แต่ไม่มีเหตุผลรองรับ → ใช้งานไม่ได้ในองค์กรจริง</li>
      <li><strong>โมเดลที่มี Metric สูงแต่ Bias แฝง:</strong> เหมือนครูที่ให้คะแนนเฉพาะนักเรียนบางกลุ่ม → ไม่ยุติธรรม</li>
    </ul>

    <div className="mt-6 bg-white dark:bg-gray-800 p-5 rounded-xl shadow border border-yellow-400">
      <p className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-300 text-center">
        “การประเมินโมเดลที่ดี คือการไม่หลงเชื่อเพียงตัวเลข แต่เข้าใจความหมายที่ซ่อนอยู่เบื้องหลังมัน” 
      </p>
    </div>

    <p className="mt-6 text-center italic text-base text-gray-800 dark:text-gray-100">
      จงเลือก Metric ให้เหมาะกับเป้าหมาย, เลือกชุดข้อมูลให้เป็นธรรม, และเลือก Insight ให้มีพลังมากพอที่จะพาโมเดลไปถึงโลกแห่งความจริง
    </p>
  </div>
</section>


        {/* Section: Mini Quiz */}
        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day11 theme={theme} />
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
        <ScrollSpy_Ai_Day11 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day11_CrossValidation;