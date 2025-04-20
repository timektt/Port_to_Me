
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day12 from "./scrollspy/ScrollSpy_Ai_Day12";
import MiniQuiz_Day12 from "./miniquiz/MiniQuiz_Day12";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day12_Overfitting = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  // Placeholder รูปภาพ สามารถเปลี่ยนชื่อภายหลังได้
  const img1 = cld.image('Overfit1').format('auto').quality('auto').resize(scale().width(500));
  const img2 = cld.image('Overfit2').format('auto').quality('auto').resize(scale().width(500));
  const img3 = cld.image('Overfit3').format('auto').quality('auto').resize(scale().width(500));
  const img4 = cld.image('Overfit4').format('auto').quality('auto').resize(scale().width(500));
  const img5 = cld.image('Overfit5').format('auto').quality('auto').resize(scale().width(500));
  const img6 = cld.image('Overfit6').format('auto').quality('auto').resize(scale().width(500));
  const img7 = cld.image('Overfit7').format('auto').quality('auto').resize(scale().width(500));
  const img8 = cld.image('Overfit8').format('auto').quality('auto').resize(scale().width(250));
  const img9 = cld.image('Overfit9').format('auto').quality('auto').resize(scale().width(250));
  const img10 = cld.image('Overfit10').format('auto').quality('auto').resize(scale().width(500));
  const img11 = cld.image('Overfit11').format('auto').quality('auto').resize(scale().width(500));
  const img12 = cld.image('Overfit12').format('auto').quality('auto').resize(scale().width(500));
  const img13 = cld.image('Overfit13').format('auto').quality('auto').resize(scale().width(500));
  const img14 = cld.image('Overfit14').format('auto').quality('auto').resize(scale().width(500));
  const img15 = cld.image('Overfit15').format('auto').quality('auto').resize(scale().width(500));
  const img16 = cld.image('Overfit16').format('auto').quality('auto').resize(scale().width(500));
  const img17 = cld.image('Overfit17').format('auto').quality('auto').resize(scale().width(500));
  const img18 = cld.image('Overfit18').format('auto').quality('auto').resize(scale().width(500));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 12: Overfitting, Underfitting & Model Diagnostics</h1>

        {/* Section Template: เพิ่มเนื้อหาแต่ละหัวข้อภายหลังได้ */}
        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Overfitting คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  {/* นิยามและลักษณะของ Overfitting */}
  <h3 className="text-xl font-semibold mb-3">Overfitting คือ?</h3>
  <p className="mb-4 leading-relaxed">
    Overfitting คือภาวะที่โมเดลเรียนรู้รายละเอียดหรือ noise จากชุดฝึกมากเกินไป ส่งผลให้ไม่สามารถ generalize กับข้อมูลใหม่ได้ดี แม้จะได้ accuracy สูงใน training set แต่กลับพลาดบ่อยใน validation/test set
  </p>

  <h3 className="text-xl font-semibold mb-3">สัญญาณเตือนที่พบบ่อย</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>Train Accuracy สูงมาก (ใกล้ 100%)</li>
    <li>Validation Accuracy ต่ำผิดปกติ</li>
    <li>Validation Loss เพิ่มขึ้นหลังจาก train ต่อ</li>
  </ul>

  {/* สาเหตุของ Overfitting */}
  <h3 className="text-xl font-semibold mb-3 text-center">สาเหตุหลักของ Overfitting</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>โมเดลมีพารามิเตอร์มากเกินไป เช่น neural network ขนาดใหญ่</li>
    <li>จำนวนข้อมูลน้อยเมื่อเทียบกับความซับซ้อนของโมเดล</li>
    <li>มีฟีเจอร์มากเกินไป โดยไม่มีการ regularization</li>
    <li>ข้อมูลมี noise หรือผิดพลาดจำนวนมาก</li>
    <li>ฝึกโมเดลนานเกินไปโดยไม่หยุด (overtraining)</li>
  </ul>

  {/* เปรียบเทียบกับนักเรียน */}
  <h3 className="text-xl font-semibold mb-3">เปรียบเทียบง่าย ๆ</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ overfit ก็เหมือนนักเรียนที่ “ท่องจำข้อสอบเก่า” ได้หมด แต่พอเปลี่ยนโจทย์นิดเดียวก็ตอบไม่ได้ เพราะไม่เข้าใจหลักการจริง
  </p>

  {/* วิธีตรวจสอบ Overfitting */}
  <h3 className="text-xl font-semibold mb-3 text-center">วิธีสังเกต Overfitting</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>กราฟ Training Loss ลดลงเรื่อย ๆ แต่ Validation Loss เพิ่มขึ้น</li>
    <li>ช่องว่างระหว่าง Training กับ Validation Accuracy เริ่มกว้างขึ้น</li>
    <li>ผลลัพธ์เปลี่ยนมากเมื่อใช้ข้อมูลใหม่หรือมี noise</li>
  </ul>

  {/* วิธีลด Overfitting */}
  <h3 className="text-xl font-semibold mb-3 text-center">วิธีป้องกันหรือแก้ไข Overfitting</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ใช้ Regularization เช่น L1/L2 เพื่อลดพารามิเตอร์ที่ไม่จำเป็น</li>
    <li>ใช้ Dropout ใน Neural Networks เพื่อลดการพึ่งพา neuron บางตัว</li>
    <li>ทำ Data Augmentation เช่น หมุน ตัด ปรับแสงภาพ เพื่อเพิ่มความหลากหลายของข้อมูล</li>
    <li>ใช้ Early Stopping หยุดการฝึกเมื่อ validation ไม่ดีขึ้น</li>
    <li>ลดขนาดโมเดลให้พอดีกับข้อมูล เช่น ลดจำนวน Layer หรือ Neuron</li>
    <li>ใช้ K-Fold Cross Validation เพื่อประเมินความเสถียรของโมเดล</li>
  </ul>

  {/* ตัวอย่างจริง */}
  <h3 className="text-xl font-semibold mb-3">ตัวอย่าง: Polynomial Regression</h3>
  <p className="mb-4 leading-relaxed">
    สมมุติเรามีข้อมูลจุดกระจาย แต่ใช้ polynomial degree สูงมากเพื่อให้เส้นผ่านทุกจุด จะได้โมเดลที่แม่นมากใน training set แต่ทำนายไม่ได้ในจุดใหม่ = overfit
  </p>

  {/* ความเข้าใจผิดที่ควรระวัง */}
  <h3 className="text-xl font-semibold mb-3">ความเข้าใจผิดที่พบบ่อย</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>คิดว่า accuracy สูงใน training คือโมเดลดี → อาจ overfit</li>
    <li>ใช้ preprocessing มากเกิน → โมเดลอาจจำ pattern จาก preprocessing</li>
  </ul>

  {/* Insight สรุป */}
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      Overfitting ไม่ได้แปลว่าโมเดล “ไม่เก่ง” แต่แปลว่าโมเดล “ยังไม่เข้าใจ” ว่าอะไรคือสิ่งสำคัญจริง ๆ ในข้อมูล — เป้าหมายคือความเข้าใจ ไม่ใช่ความจำ
    </p>
  </div>
</section>



<section id="underfitting" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Underfitting คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <p className="mb-4 leading-relaxed">
    Underfitting เป็นภาวะที่โมเดลไม่สามารถเรียนรู้หรือเข้าใจโครงสร้างของข้อมูลได้ดีพอ ทำให้แม้แต่ในชุดข้อมูลฝึกโมเดลก็ยังทำผลงานได้ไม่ดี ซึ่งต่างจาก Overfitting ที่แม้จะแม่นใน Training แต่พลาดใน Test อย่างน้อยยังเข้าใจข้อมูลเดิมได้บางส่วน
  </p>

  <p className="mb-4 leading-relaxed">
    อาการของ Underfitting สะท้อนว่าโมเดลยังไม่สามารถจับ pattern หลักในข้อมูลได้เลย ไม่ว่าจะเป็นเพราะโมเดลเรียบง่ายเกินไป ฟีเจอร์ไม่ครบ การประมวลผลล่วงหน้าไม่เหมาะสม หรือฝึกไม่พอ การพัฒนาโมเดลที่สามารถหลีกเลี่ยง underfitting ได้นั้นถือเป็นหนึ่งในเป้าหมายหลักของการสร้างระบบที่ generalize ได้จริง
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">อาการที่พบบ่อย</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ความแม่นยำ (Accuracy) ต่ำทั้งใน Training และ Validation</li>
    <li>ค่า Loss สูงต่อเนื่อง ไม่มีแนวโน้มลดลงแม้ฝึกหลายรอบ</li>
    <li>การพยากรณ์ค่าต่าง ๆ อยู่ในช่วงแคบ ไม่สามารถจับการเปลี่ยนแปลงได้ดี</li>
    <li>โมเดลไม่ตอบสนองต่อความซับซ้อนของข้อมูล เช่น เส้น decision boundary เรียบเกินไป</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 text-center">สาเหตุหลักของ Underfitting</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>เลือกโมเดลที่เรียบง่ายเกินไป เช่น Linear Regression กับข้อมูลไม่เชิงเส้น</li>
    <li>ใช้ Regularization ที่แรงเกิน ทำให้โมเดลถูกจำกัดไม่ให้เรียนรู้เต็มที่</li>
    <li>จำนวนรอบการฝึก (Epoch) ไม่เพียงพอ โดยเฉพาะกับข้อมูลปริมาณมาก</li>
    <li>ข้อมูลผ่าน preprocessing ไม่เหมาะสม เช่น scaling ผิดหรือ feature ขาด</li>
    <li>จำนวนฟีเจอร์ไม่พอ หรือฟีเจอร์ไม่สื่อสารกับ target ได้ดี</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">ตัวอย่างภาพการเรียนรู้ของโมเดลที่ Underfit</h3>
  <div className="grid md:grid-cols-2 gap-6 my-6">
    <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
      <p className="mb-2 text-sm text-center font-medium">การจำแนกลักษณะ Underfitting</p>
      <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
      
    </div>
    <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
      <p className="mb-2 text-sm text-center font-medium">โมเดลที่เรียนรู้ได้ดี (Good Fit)</p>
      <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3">วิธีแก้ปัญหา Underfitting</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>เพิ่มความซับซ้อนของโมเดล เช่น เปลี่ยนไปใช้ Neural Network หรือ Tree-Based Model</li>
    <li>เพิ่มจำนวน Epoch หรือเปลี่ยน optimizer ให้เรียนรู้ได้ลึกขึ้น</li>
    <li>ลด Regularization เช่น ลดค่า L1/L2 penalty</li>
    <li>เพิ่มหรือปรับ Feature ให้ครอบคลุม pattern สำคัญ</li>
    <li>ใช้เทคนิค Feature Engineering เช่น interaction term, polynomial features</li>
    <li>ใช้ Preprocessing ที่เหมาะสม เช่น Scaling, One-Hot Encoding</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">Insight เปรียบเทียบกับ Overfitting</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">Overfitting จำ noise → underfit ไม่จำอะไรเลย</p>
    <p className="mb-2">การตรวจพบ early stage ของ underfitting ทำให้แก้ไขได้ก่อนโมเดลจะถูก deploy ไปใช้งานจริง</p>
    <p className="mb-2">Underfitting บอกว่า “โมเดลยังไม่ได้เข้าใจ” ไม่ใช่ “เข้าใจผิด” แบบ overfit</p>
  </div>

  <h3 className="text-xl font-semibold mt-6 mb-3">สรุปสั้นท้ายบท</h3>
  <p className="leading-relaxed">
    Underfitting ไม่ใช่ปัญหาที่ร้ายแรงที่สุด แต่ถ้าไม่ได้รับการแก้ไขตั้งแต่ต้น จะไม่มีวันได้โมเดลที่ใช้งานได้ดี
    จุดสำคัญคือการตรวจพบ early signal เช่น accuracy ต่ำ, loss ไม่ลด และการเลือกโมเดลหรือ config ที่เหมาะสม
  </p>

  <p className="mt-6 text-center italic text-gray-700 dark:text-gray-300">
    การวางรากฐานให้โมเดลเข้าใจข้อมูลดีก่อน คือก้าวแรกของระบบที่ดีในระยะยาว
  </p>
</section>

<section id="bias-variance" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Bias-Variance Tradeoff</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <p className="mb-4 leading-relaxed">
    การเข้าใจความสัมพันธ์ระหว่าง Bias และ Variance เป็นหัวใจของการพัฒนาโมเดลที่มีความสามารถในการทำนายที่ดีในโลกจริง โดยไม่ขึ้นอยู่กับแค่การแม่นยำในชุดข้อมูลฝึกเท่านั้น แต่เน้นไปที่ความสามารถในการ generalize ไปยังข้อมูลที่ไม่เคยเห็นมาก่อน.
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">Bias คืออะไร?</h3>
  <p className="mb-4 leading-relaxed">
    Bias คือข้อผิดพลาดจากการสันนิษฐานที่เรียบง่ายเกินไปของโมเดล ตัวอย่างเช่น การใช้เส้นตรงทำนายข้อมูลที่มีความโค้งงอสูง จะทำให้โมเดลไม่สามารถจับ pattern ที่แท้จริงได้ และเกิดข้อผิดพลาดสูงทั้งในชุดฝึกและชุดทดสอบ.
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">Variance คืออะไร?</h3>
  <p className="mb-4 leading-relaxed">
    Variance คือความไวต่อความเปลี่ยนแปลงของข้อมูลฝึก หากโมเดลมี variance สูง มันจะตอบสนองต่อความเปลี่ยนแปลงเล็กน้อยในข้อมูลมากเกินไป จนทำให้เกิดการ overfit และไม่สามารถ generalize ได้ดี.
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">ตัวอย่างเปรียบเทียบ</h3>
  <div className="grid md:grid-cols-3 gap-6 my-6">
    <div className="p-4 border rounded-lg bg-white dark:bg-gray-800 shadow">
      <h4 className="font-semibold mb-2 text-center">High Bias</h4>
      <p className="text-sm text-gray-700 dark:text-gray-300">
        โมเดลไม่สามารถเรียนรู้จากข้อมูลได้เพียงพอ เช่น ใช้ linear regression กับข้อมูลโค้ง
      </p>
    </div>
    <div className="p-4 border rounded-lg bg-white dark:bg-gray-800 shadow">
      <h4 className="font-semibold mb-2 text-center">High Variance</h4>
      <p className="text-sm text-gray-700 dark:text-gray-300">
        โมเดลเรียนรู้ noise ในข้อมูลมากเกินไป เช่น โมเดลลึกที่ไม่มี regularization
      </p>
    </div>
    <div className="p-4 border rounded-lg bg-white dark:bg-gray-800 shadow">
      <h4 className="font-semibold mb-2 text-center">Balanced</h4>
      <p className="text-sm text-gray-700 dark:text-gray-300">
        โมเดลเข้าใจ pattern หลักของข้อมูลและสามารถปรับตัวกับข้อมูลใหม่ได้ดี
      </p>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">วิธีวิเคราะห์ Bias-Variance</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>ถ้า Training Error สูง และ Validation Error สูง → โมเดลมี Bias สูง</li>
    <li>ถ้า Training Error ต่ำ แต่ Validation Error สูง → โมเดลมี Variance สูง</li>
    <li>เป้าหมายคือการหาจุดสมดุลระหว่างทั้งสองค่า</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">เทคนิคปรับสมดุล Bias-Variance</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>เพิ่มข้อมูลฝึกเพื่อช่วยลด variance</li>
    <li>เพิ่ม regularization เพื่อลดความซับซ้อนของโมเดล</li>
    <li>ใช้โมเดลที่เหมาะสมกับความซับซ้อนของปัญหา</li>
    <li>ใช้ cross-validation เพื่อเลือกโมเดลที่ generalize ได้ดีที่สุด</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">ภาพรวมกราฟ</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <p className="mb-4 leading-relaxed">
    จุดตัดกันของกราฟ bias และ variance คือจุดที่โมเดลมี generalization ที่ดีที่สุด หากเลือกโมเดลที่ซับซ้อนเกินไปจะเสี่ยงต่อ overfit แต่ถ้าเรียบง่ายเกินไปจะ underfit ทันที
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การเรียนรู้เชิงเครื่องที่แท้จริงไม่ได้มุ่งให้โมเดลแม่นที่สุดกับข้อมูลใดข้อมูลหนึ่ง แต่ต้องสร้างความเข้าใจที่ยืดหยุ่นพอสำหรับโลกที่เปลี่ยนตลอดเวลา
  </div>
</section>



<section id="diagnostics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">วิธีสังเกตอาการ Overfit / Underfit</h2>


  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <p className="mb-4 leading-relaxed">
    การวิเคราะห์พฤติกรรมของโมเดลเป็นส่วนสำคัญในการพัฒนา Machine Learning ที่ใช้งานได้จริง การรู้ว่าโมเดลกำลัง Overfit หรือ Underfit ช่วยให้ตัดสินใจปรับโมเดลได้ตรงจุด
  </p>

  <h3 className="text-xl font-semibold mb-3">1. สังเกตจาก Learning Curve</h3>
  <p className="mb-4 leading-relaxed">
    Learning Curve แสดงความสัมพันธ์ระหว่างประสิทธิภาพของโมเดลกับจำนวนข้อมูลหรือรอบการเทรน หาก Training Accuracy สูง แต่ Validation Accuracy ต่ำ แสดงว่า Overfit ในทางกลับกัน ถ้าทั้งคู่ต่ำ แสดงว่า Underfit
  </p>

  <div className="grid md:grid-cols-2 gap-6 mb-6">
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold text-lg mb-2">อาการ Overfitting</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>Accuracy บน Training สูง แต่ Validation ต่ำ</li>
        <li>Loss บน Validation เพิ่มขึ้นเมื่อเทรนต่อ</li>
        <li>โมเดลตอบได้ดีเฉพาะข้อมูลฝึก แต่ล้มเหลวบนข้อมูลใหม่</li>
        <li>ผลลัพธ์ไม่เสถียรเมื่อเปลี่ยนชุดข้อมูลทดสอบ</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold text-lg mb-2">อาการ Underfitting</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>Accuracy ต่ำทั้ง Training และ Validation</li>
        <li>โมเดลจับ Pattern ไม่ได้แม้เทรนนาน</li>
        <li>Loss สูงตลอด ไม่ลดลงเมื่อเทรน</li>
        <li>โมเดลตอบแบบเดา ไม่เรียนรู้จากข้อมูล</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3">2. ช่องว่างระหว่าง Train กับ Validation (Validation Gap)</h3>
  <p className="mb-4 leading-relaxed">
    การตรวจสอบความแตกต่างระหว่าง Train Score กับ Validation Score เป็นสัญญาณสำคัญ หากค่า Train สูงมาก แต่ Validation ต่ำ แสดงว่าโมเดลเรียนรู้เฉพาะจุดข้อมูล ไม่สามารถสรุปภาพรวมได้
  </p>

  <h3 className="text-xl font-semibold mb-3">3. พฤติกรรม Loss เมื่อเทรนต่อเนื่อง</h3>
  <p className="mb-4 leading-relaxed">
    กราฟ Training Loss ที่ลดลงเรื่อย ๆ แต่ Validation Loss กลับเริ่มเพิ่มขึ้น เป็นสัญญาณของ Overfitting ในทางกลับกัน หากกราฟทั้งคู่ลดช้า หรือไม่ลดเลย อาจหมายถึง Underfitting
  </p>

  <h3 className="text-xl font-semibold mb-3">4. พฤติกรรมโมเดลเมื่อใช้ข้อมูลนอกชุดฝึก</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ Overfit มักให้ผลลัพธ์แย่เมื่อเจอกับข้อมูลใหม่หรือ noisy data ขณะที่ Underfit จะแสดง performance ต่ำไม่ว่าใช้ข้อมูลชุดใด
  </p>

  <h3 className="text-xl font-semibold mb-3">5. วิเคราะห์ด้วย Confusion Matrix</h3>
  <p className="mb-4 leading-relaxed">
    ใน Classification การดูความสมดุลของ TP, TN, FP, FN จาก Confusion Matrix ช่วยให้เห็นว่าโมเดลพลาดด้านใด ถ้าโมเดลจำเพาะเจาะจงกับข้อมูลบางกลุ่ม อาจ Overfit
  </p>

  <h3 className="text-xl font-semibold mb-3">6. Cross Validation Score</h3>
  <p className="mb-4 leading-relaxed">
    หากค่า Mean Score สูงแต่ค่า Standard Deviation สูงเช่นกัน แสดงถึงความไม่เสถียร ซึ่งอาจเกิดจาก Overfitting การตรวจสอบความสม่ำเสมอของผลลัพธ์ทุก fold เป็นแนวทางวิเคราะห์ที่แม่นยำ
  </p>

  <h3 className="text-xl font-semibold mb-3">7. การดู Feature Importance</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ Overfit มักให้ความสำคัญกับ Feature ที่ไม่มีสาระสำคัญ หรือ Noise การวิเคราะห์ Feature Importance สามารถช่วยตัดสิ่งรบกวนได้อย่างมีประสิทธิภาพ
  </p>

  <h3 className="text-xl font-semibold mb-3">8. การดูพฤติกรรมบนข้อมูลที่มี Noise</h3>
  <p className="mb-4 leading-relaxed">
    หากโมเดลตอบสนองไวเกินไปกับข้อมูลที่ถูกปรับแต่งนิดเดียว เช่น เพิ่ม Noise หรือเปลี่ยนคำบางคำ อาจแปลว่ามี Overfit ต่อรายละเอียดเล็ก ๆ ในข้อมูล
  </p>

  <h3 className="text-xl font-semibold mb-3">9. กรณีโมเดลซับซ้อนเกินไป</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่มี parameter มากเกินจำเป็น เช่น neural network ที่มีหลาย layer โดยไม่ปรับ regularization อาจ Overfit ง่าย ควรเริ่มจากโมเดลที่ง่ายที่สุดก่อน
  </p>

  <h3 className="text-xl font-semibold mb-3">10. การใช้ Ensemble ช่วยวินิจฉัย</h3>
  <p className="mb-4 leading-relaxed">
    หากการใช้ Ensemble (เช่น Bagging) ช่วยลด variance อย่างมีนัยสำคัญ แสดงว่าโมเดลต้นทางมีแนวโน้ม Overfit ขณะที่การใช้ Ensemble แล้วผลไม่ต่าง แสดงว่าโมเดลเดิมมีความเสถียร
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <strong>Insight:</strong><br />
    อาการ Overfitting และ Underfitting เปรียบได้กับการเดินทางที่หลุดเส้นทาง การวิเคราะห์ Learning Curve, Validation Gap และพฤติกรรมโมเดลจากหลายมุมมอง คือเข็มทิศสำคัญในการปรับทิศทางให้แม่นยำยิ่งขึ้น
  </div>
</section>


<section id="techniques" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">เทคนิคลด Overfitting</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img15} />
  </div>

  <p className="mb-4 leading-relaxed">
    Overfitting คือการที่โมเดลเรียนรู้รายละเอียดหรือ noise มากเกินไป ทำให้ขาดความสามารถในการ generalize ไปยังข้อมูลใหม่
    เทคนิคหลายอย่างสามารถช่วยลดความเสี่ยงนี้ได้ ไม่ว่าจะในขั้นตอนการออกแบบโมเดลหรือการฝึกโมเดล
  </p>

  <h3 className="text-xl font-semibold mb-3">1. Regularization</h3>
  <p className="mb-4 leading-relaxed">
    การเพิ่ม penalty ต่อพารามิเตอร์ที่มีค่าสูง เพื่อควบคุมไม่ให้โมเดลฟิตข้อมูลเกินไป L1 และ L2 เป็นสองวิธีที่นิยมใช้
  </p>

  <h3 className="text-xl font-semibold mb-3">2. Dropout</h3>
  <p className="mb-4 leading-relaxed">
    สำหรับ Neural Networks Dropout จะสุ่มปิด neuron บางส่วนระหว่างการฝึกเพื่อป้องกันไม่ให้โมเดลจำเจเกินไป
  </p>

  <h3 className="text-xl font-semibold mb-3">3. Data Augmentation</h3>
  <p className="mb-4 leading-relaxed">
    การสร้างข้อมูลใหม่จากข้อมูลเดิม เช่น หมุน, ยืด, ตัดภาพ หรือปรับแสง ช่วยเพิ่มความหลากหลายของข้อมูล
  </p>

  <h3 className="text-xl font-semibold mb-3">4. Early Stopping</h3>
  <p className="mb-4 leading-relaxed">
    หยุดการฝึกเมื่อ validation loss ไม่ลดลงอีก เพื่อหลีกเลี่ยงการเรียนรู้ noise ที่เกิดขึ้นใน epoch หลัง ๆ
  </p>

  <h3 className="text-xl font-semibold mb-3">5. ลดขนาดของโมเดล</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่มีขนาดใหญ่เกินไปอาจมีพารามิเตอร์มากจนเรียนรู้รายละเอียดที่ไม่จำเป็น การลด layer หรือ neuron ช่วยควบคุม capacity
  </p>

  <h3 className="text-xl font-semibold mb-3">6. ลดจำนวน Feature</h3>
  <p className="mb-4 leading-relaxed">
    การใช้ฟีเจอร์ที่จำเป็นเท่านั้น โดยตัดฟีเจอร์ที่มี correlation ต่ำหรือ noise สูงออก เช่น ใช้ PCA หรือ Feature Importance จากโมเดล
  </p>

  <h3 className="text-xl font-semibold mb-3">7. Cross Validation</h3>
  <p className="mb-4 leading-relaxed">
    การใช้ Cross Validation ช่วยให้มั่นใจว่าโมเดลไม่ได้ฟิตกับ subset ใด subset หนึ่งเท่านั้น และเห็นภาพรวมของ performance
  </p>

  <h3 className="text-xl font-semibold mb-3">8. Batch Normalization</h3>
  <p className="mb-4 leading-relaxed">
    ทำให้การกระจายของค่าในแต่ละชั้นสม่ำเสมอ ช่วยให้การฝึกมีเสถียรภาพและลดโอกาส overfitting โดยไม่จำเป็นต้องเพิ่ม regularization อื่น ๆ
  </p>

  <h3 className="text-xl font-semibold mb-3">9. Noise Injection</h3>
  <p className="mb-4 leading-relaxed">
    การเติม noise แบบตั้งใจลงไปใน input หรือ layer ซ่อน ช่วยให้โมเดลเรียนรู้แบบยืดหยุ่นมากขึ้นและไม่ sensitive ต่อจุดเล็ก ๆ ของข้อมูล
  </p>

  <h3 className="text-xl font-semibold mb-3">10. Ensemble Learning</h3>
  <p className="mb-4 leading-relaxed">
    ใช้โมเดลหลายตัวและรวมผลลัพธ์เข้าด้วยกัน เช่น Random Forest หรือ Voting Classifier เพื่อกระจายความผิดพลาดจากโมเดลเดี่ยว
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-6">
    <p className="mb-2 font-semibold">Insight:</p>
    <p className="mb-2">
      การลด Overfitting คือการออกแบบให้โมเดลเรียนรู้เฉพาะสิ่งที่จำเป็นและสามารถปรับตัวกับข้อมูลใหม่ได้อย่างยืดหยุ่น
    </p>
    <ul className="list-disc pl-6 text-sm space-y-1">
      <li>Regularization และ Dropout เป็นการควบคุมภายในโมเดล</li>
      <li>Data Augmentation และ Noise Injection เพิ่มความหลากหลายจากข้อมูล</li>
      <li>Early Stopping ช่วยตัดจังหวะการเรียนรู้ที่เกินพอดี</li>
      <li>Ensemble สร้างการตัดสินใจแบบกลุ่มเพื่อลด bias และ variance</li>
    </ul>
  </div>
</section>


<section id="error-analysis" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">การวิเคราะห์ข้อผิดพลาดของโมเดล (Error Analysis)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img16} />
  </div>


  <p className="mb-4 leading-relaxed">
    หลังจากฝึกโมเดลเสร็จ การวิเคราะห์ว่ามัน "พลาดที่ไหน" ถือเป็นกระบวนการสำคัญที่ช่วยยกระดับคุณภาพการพัฒนาโมเดลแบบเจาะลึก โดยไม่พึ่งพาแค่ค่า Accuracy หรือ F1 เพียงตัวเดียว แต่ไปถึงระดับตัวอย่างที่โมเดลพลาด ชนิดของความผิด และกลยุทธ์การปรับปรุง.
  </p>

  <h3 className="text-xl font-semibold mb-3">1. ตรวจสอบตัวอย่างที่โมเดลพลาด</h3>
  <p className="mb-4 leading-relaxed">
    เริ่มจากการสุ่มเลือกตัวอย่างที่โมเดลทำนายผิด เช่น ทำนายว่าเป็นแมว ทั้งที่จริงเป็นสุนัข การดูรายละเอียดของ input เช่น ความชัดเจน ความยาว ความซับซ้อน หรือความกำกวมของข้อมูล สามารถชี้ถึงจุดที่โมเดลยังไม่เข้าใจ และให้ไอเดียในการปรับปรุง.
  </p>

  <h3 className="text-xl font-semibold mb-3">2. ใช้ Confusion Matrix วิเคราะห์ข้อผิดพลาดเชิงระบบ</h3>
  <p className="mb-4 leading-relaxed">
    การสร้าง confusion matrix จะช่วยให้เห็นว่าคลาสใดที่โมเดลพลาดบ่อย และคลาสใดที่มั่นใจผิด เช่น ทำนายผิดระหว่างเลข 1 กับเลข 7 ซ้ำ ๆ แสดงว่า feature extraction ยังไม่สามารถแยกแยะระหว่างกลุ่มได้ดี.
  </p>

  <h3 className="text-xl font-semibold mb-3">3. วิเคราะห์ False Positive vs False Negative</h3>
  <p className="mb-4 leading-relaxed">
    การพลาดแบบ False Positive (ทำนายว่าเป็นแต่จริง ๆ ไม่ใช่) กับ False Negative (ไม่ทำนายแต่จริง ๆ ใช่) มีผลกระทบต่างกัน เช่น ในงานแพทย์ FN อาจร้ายแรงกว่า FP ดังนั้นการเน้นลด FN จึงสำคัญในบริบทบางอย่าง
  </p>

  <h3 className="text-xl font-semibold mb-3">4. การจัดกลุ่มข้อผิดพลาด</h3>
  <p className="mb-4 leading-relaxed">
    แยกข้อผิดพลาดเป็นกลุ่มย่อย เช่น "พลาดเพราะข้อมูลเบลอ", "พลาดเพราะคำซ้อน", "พลาดเพราะภาษาแสลง" ช่วยให้สามารถจัดกลุ่มแนวทางการปรับปรุง เช่น สร้าง preprocessor สำหรับแต่ละกลุ่ม
  </p>

  <h3 className="text-xl font-semibold mb-3">5. Visualize จุดที่โมเดลสนใจ</h3>
  <p className="mb-4 leading-relaxed">
    เทคนิคอย่าง Attention Map, Grad-CAM, LIME หรือ SHAP ช่วยให้เข้าใจว่าโมเดลให้ความสำคัญกับส่วนใดของ input เช่น ข้อความช่วงใด ภาพตรงไหน หากจุดที่สนใจไม่สัมพันธ์กับผลลัพธ์ อาจบ่งชี้ว่าโมเดลกำลัง overfit กับ feature ที่ไม่สำคัญ
  </p>

  <h3 className="text-xl font-semibold mb-3">6. วิเคราะห์ข้อผิดพลาดตามกลุ่มข้อมูล</h3>
  <p className="mb-4 leading-relaxed">
    เปรียบเทียบ performance ของโมเดลในแต่ละ segment เช่น กลุ่มเพศ, อายุ, ภาษา, หรือภูมิภาค ช่วยให้รู้ว่าโมเดลมี bias หรือไม่ เช่น โมเดลทำงานได้ดีเฉพาะกับภาษาอังกฤษ แต่พลาดหนักในภาษาอื่น
  </p>

  <h3 className="text-xl font-semibold mb-3">7. ตรวจสอบข้อผิดพลาดเชิงเวลา (Temporal Drift)</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลอาจมี performance ลดลงกับข้อมูลในช่วงเวลาที่เปลี่ยนไป เช่น ข่าวหรือเทรนด์ใหม่ ๆ การวิเคราะห์ error แยกตามช่วงเวลาช่วยให้รู้ว่าโมเดลยังปรับตัวได้หรือไม่
  </p>

  <h3 className="text-xl font-semibold mb-3">8. ตรวจสอบสมดุลของข้อมูลที่พลาด</h3>
  <p className="mb-4 leading-relaxed">
    หากพบว่าโมเดลพลาดเฉพาะคลาสที่มีตัวอย่างน้อย อาจเป็นปัญหา class imbalance การใช้เทคนิค resampling, SMOTE หรือการเพิ่มข้อมูลในกลุ่มนั้นจะช่วยได้
  </p>

  <h3 className="text-xl font-semibold mb-3">9. สร้าง Error Dataset สำหรับทดสอบภายหลัง</h3>
  <p className="mb-4 leading-relaxed">
    นำข้อผิดพลาดที่พบบ่อยมาสร้างเป็น Error Benchmark Set เพื่อทดสอบในรอบถัดไปว่าปรับปรุงแล้วดีขึ้นหรือไม่ เป็นวิธีทำ regression testing ในงาน ML
  </p>

  <h3 className="text-xl font-semibold mb-3">10. เชื่อม Error กับ Feedback จากผู้ใช้</h3>
  <p className="mb-4 leading-relaxed">
    หากมีระบบ production การใช้ feedback หรือการกด report จากผู้ใช้จริง ช่วยให้เข้าใจว่าข้อผิดพลาดใดที่มีผลกระทบสูง และควรแก้ก่อน
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p className="mb-2">การวิเคราะห์ข้อผิดพลาดเปรียบเสมือนการถามโมเดลว่า "ยังไม่เข้าใจอะไร" มากกว่าจะถามว่า "ผิดอะไร" เพราะข้อผิดพลาดแต่ละจุดคือจุดเริ่มต้นของการเรียนรู้ใหม่</p>
    <p className="text-sm mt-2 italic">การมองข้อผิดพลาดอย่างมีระบบ ช่วยให้ไม่เพียงพัฒนาโมเดลให้แม่นยำขึ้น แต่ยังลดความเสี่ยงของโมเดลในโลกจริงอย่างยั่งยืน</p>
  </div>
</section>

<section id="case-study" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Case Study: ตัวอย่างโมเดลที่ Overfit และแนวทางแก้</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img17} />
  </div>
  <p className="mb-4 leading-relaxed">
    ตัวอย่างจริงของการ Overfitting เกิดขึ้นในงาน OCR ที่ใช้โมเดล CNN แบบลึกกับชุดข้อมูลขนาดเล็ก พบว่า accuracy ใน train สูงกว่า 98% แต่ test accuracy ต่ำกว่า 70%.
  </p>
  <p className="mb-4 leading-relaxed">
    หลังวิเคราะห์พบว่าโมเดลเรียนรู้ “ขอบภาพ” และ “noise pattern” มากเกินไป ทีมงานจึงปรับโดยใช้ Dropout 0.4, EarlyStopping และ Data Augmentation เช่น การสลับมุม และปรับความคมของภาพ.
  </p>
  <p className="mb-4 leading-relaxed">
    หลังปรับปรุงแล้ว test accuracy เพิ่มขึ้นเป็น 86% และกราฟ validation loss มีความนิ่งขึ้นกว่าเดิมอย่างชัดเจน.
  </p>
  <p className="mb-4 leading-relaxed">
    กรณีนี้ตอกย้ำว่า Overfitting สามารถแก้ได้หากรู้จักวิเคราะห์พฤติกรรมของโมเดลและไม่มองเพียงแค่ตัวเลข accuracy
  </p>
</section>


<section id="diagnostic-checklist" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Checklist: วินิจฉัย Overfitting & Underfitting</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img18} />
  </div>
  <p className="mb-4 leading-relaxed">
    ก่อนนำโมเดลไปใช้งานจริงในโลกภายนอก ควรผ่านกระบวนการตรวจสอบอย่างเป็นระบบ โดยเช็กลิสต์ด้านล่างออกแบบมาเพื่อครอบคลุมมุมมองเชิงเทคนิค การทดสอบเชิงพฤติกรรม และการประเมินผลที่สอดคล้องกับสถานการณ์จริงในโปรเจกต์หลากหลายรูปแบบ
  </p>

  <div className="grid md:grid-cols-2 gap-6 mb-6">
    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
      <h3 className="text-xl font-semibold mb-3">Learning Behavior</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li> ตรวจสอบ Learning Curve: กราฟ loss และ accuracy ควรมีทิศทางที่สมเหตุผล ไม่แยกทางกันมากเกินไป</li>
        <li> ดูจังหวะ Early Stopping: ควรหยุดเมื่อ validation loss เริ่มนิ่งหรือลดน้อยลง</li>
        <li> ตรวจสอบว่ารอบการเทรน (Epoch) มากเกินไปหรือไม่</li>
        <li> ตรวจพฤติกรรมแบบ underfit หรือ overfit ในรอบเทรนที่ผ่านมา</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
      <h3 className="text-xl font-semibold mb-3">Performance Stability</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li> ใช้ Cross Validation (เช่น K-Fold): ค่าความแม่นยำไม่ควรสวิงมากระหว่างแต่ละ fold</li>
        <li> ตรวจค่า Mean กับ Standard Deviation ของ Accuracy หรือ F1</li>
        <li> ทดสอบความเสถียรของโมเดลเมื่อใช้ random seed ที่ต่างกัน</li>
        <li> ทดสอบข้อมูลชุดใหม่ที่ไม่เคยเห็นมาก่อน และดูพฤติกรรมโมเดล</li>
      </ul>
    </div>
  </div>

  <div className="grid md:grid-cols-2 gap-6 mb-6">
    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
      <h3 className="text-xl font-semibold mb-3">Error Pattern</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li> วิเคราะห์ FP / FN: พลาดแบบไหนบ่อย? มี bias หรือไม่?</li>
        <li> ตรวจความแม่นยำใน class ที่สำคัญ เช่น minority class</li>
        <li> ดู Confusion Matrix: มีคลาสใดที่โมเดลพลาดซ้ำซากหรือไม่</li>
        <li> ใช้ metric อื่นนอกจาก Accuracy เช่น Precision, Recall, AUC</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
      <h3 className="text-xl font-semibold mb-3">Model Complexity & Features</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li> โมเดลซับซ้อนเกินไปหรือไม่เมื่อเทียบกับข้อมูล</li>
        <li> ดู Feature Importance: โมเดลใช้ feature ใดบ้างในการตัดสินใจ</li>
        <li> มีฟีเจอร์ที่ noisy หรือไม่จำเป็น? ควรทำ Feature Selection</li>
        <li> พารามิเตอร์หรือ layer มากเกินไปหรือเปล่าใน deep learning</li>
      </ul>
    </div>
  </div>

  <div className="grid md:grid-cols-1 gap-6 mb-6">
    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
      <h3 className="text-xl font-semibold mb-3">Robustness & External Validation</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li> ลองใส่ noise เข้าไปใน input แบบจำลอง เพื่อดูว่าโมเดลเปลี่ยนพฤติกรรมไหม</li>
        <li> ทดสอบ performance กับกลุ่มย่อยของข้อมูล เช่น ภาษา เพศ วัย</li>
        <li> สร้างชุดทดสอบพิเศษเพื่อจับ blindspot ของโมเดล</li>
        <li> ทดสอบระบบ E2E แบบเหมือนใช้งานจริงก่อน deploy</li>
      </ul>
    </div>
  </div>

  <div className="bg-green-100 dark:bg-green-900 text-black dark:text-green-100 p-5 rounded-xl border-l-4 border-green-500 shadow mt-6">
    <strong>Tip:</strong><br />
    การทำ checklist อย่างรัดก่อน deploy ไม่เพียงช่วยเพิ่มความแม่นยำ แต่ยังลดต้นทุนการแก้ไขระบบภายหลัง และเพิ่มความมั่นใจใน production
  </div>
</section>



        {/* Quiz Section */}
        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day12 theme={theme} />
        </section>

        {/* Tags & Navigation */}
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
        <ScrollSpy_Ai_Day12 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day12_Overfitting;
