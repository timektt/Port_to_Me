import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day10 from "./scrollspy/ScrollSpy_Ai_Day10";
import MiniQuiz_Day10 from "./miniquiz/MiniQuiz_Day10";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day10_BiasVariance = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  const img1 = cld.image('BiasVariance1').format('auto').quality('auto').resize(scale().width(600));
  const img2 = cld.image('BiasVariance2').format('auto').quality('auto').resize(scale().width(500));
  const img3 = cld.image('BiasVariance3').format('auto').quality('auto').resize(scale().width(500));
  const img4 = cld.image('BiasVariance4').format('auto').quality('auto').resize(scale().width(600));
  const img5 = cld.image('BiasVariance5').format('auto').quality('auto').resize(scale().width(500));
  const img6 = cld.image('BiasVariance6').format('auto').quality('auto').resize(scale().width(500));
  const img7 = cld.image('BiasVariance7').format('auto').quality('auto').resize(scale().width(500));
  const img8 = cld.image('BiasVariance8').format('auto').quality('auto').resize(scale().width(500));
  const img9 = cld.image('BiasVariance9').format('auto').quality('auto').resize(scale().width(600));
  const img10 = cld.image('BiasVariance10').format('auto').quality('auto').resize(scale().width(500));
  const img11 = cld.image('BiasVariance11').format('auto').quality('auto').resize(scale().width(600));
  const img12 = cld.image('BiasVariance12').format('auto').quality('auto').resize(scale().width(500));
  const img13 = cld.image('BiasVariance13').format('auto').quality('auto').resize(scale().width(600));
  const img14 = cld.image('BiasVariance14').format('auto').quality('auto').resize(scale().width(600));
  const img15 = cld.image('BiasVariance15').format('auto').quality('auto').resize(scale().width(600));
  const img16 = cld.image('BiasVariance16').format('auto').quality('auto').resize(scale().width(500));
  const img17 = cld.image('BiasVariance17').format('auto').quality('auto').resize(scale().width(500));
  const img18 = cld.image('BiasVariance18').format('auto').quality('auto').resize(scale().width(600));
  const img19 = cld.image('BiasVariance19').format('auto').quality('auto').resize(scale().width(500));
  const img20 = cld.image('BiasVariance20').format('auto').quality('auto').resize(scale().width(600));
  const img21 = cld.image('BiasVariance21').format('auto').quality('auto').resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 10: Bias-Variance Tradeoff & Model Capacity</h1>

        <section id="bias-vs-variance" className="mb-16 scroll-mt-20">
  <h2 className="text-2xl font-semibold mb-4 text-center">Bias vs Variance คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในโลกของ Machine Learning ความสำเร็จของโมเดลไม่ได้ขึ้นอยู่แค่กับการเรียนรู้ข้อมูลฝึก (training data)
    แต่ยังขึ้นอยู่กับความสามารถในการ “generalize” หรือการนำสิ่งที่เรียนรู้ไปใช้กับข้อมูลใหม่ที่ไม่เคยเจอมาก่อนด้วย
    สองแนวคิดสำคัญที่ส่งผลต่อการ generalize ของโมเดลคือ “Bias” และ “Variance”
  </p>

  <h3 className="text-xl font-semibold mb-2">1. Bias คืออะไร?</h3>
  <p className="mb-4 leading-relaxed">
    Bias (อคติของโมเดล) หมายถึงข้อผิดพลาดที่เกิดจากการที่โมเดล “มองโลกแคบเกินไป”
    หรือไม่ได้เรียนรู้รูปแบบที่แท้จริงของข้อมูล เช่น การใช้โมเดลที่เรียบง่ายเกินไปในการแก้ปัญหาที่ซับซ้อน
    โมเดลจะมีแนวโน้มทำนายผิดซ้ำๆ แม้แต่กับข้อมูลที่มี pattern ชัดเจน
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>โมเดลที่มี bias สูงมักเป็นโมเดลที่ underfit</li>
    <li>ไม่สามารถจับความสัมพันธ์ที่แท้จริงของข้อมูลได้</li>
    <li>เหมือนนักเรียนที่อ่านแค่หัวข้อ แต่ไม่เข้าใจแนวคิดเบื้องหลัง</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">2. Variance คืออะไร?</h3>
  <p className="mb-4 leading-relaxed">
    Variance (ความแปรปรวนของโมเดล) คือความไวของโมเดลต่อการเปลี่ยนแปลงเล็กๆ น้อยๆ ในข้อมูลฝึก
    โมเดลที่มี variance สูงมักจะเรียนรู้ noise หรือรายละเอียดเล็กๆ มากเกินไป ทำให้เมื่อเจอข้อมูลใหม่
    ผลลัพธ์กลับไม่แม่นยำเพราะไปจำข้อมูลฝึกมากเกินไป
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>โมเดลที่มี variance สูงมักเกิด overfit</li>
    <li>ทำนายได้แม่นในข้อมูลฝึก แต่พลาดในข้อมูลใหม่</li>
    <li>เหมือนนักเรียนที่ท่องจำข้อสอบเก่าได้หมด แต่ทำข้อสอบใหม่ไม่ได้</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">3. เปรียบเทียบ Bias กับ Variance</h3>
  <div className="overflow-x-auto mb-4">
    <table className="min-w-full table-auto border-collapse border border-gray-500 text-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border border-gray-500 px-4 py-2">หัวข้อ</th>
          <th className="border border-gray-500 px-4 py-2">Bias สูง</th>
          <th className="border border-gray-500 px-4 py-2">Variance สูง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border border-gray-500 px-4 py-2">ความแม่นยำกับข้อมูลฝึก</td>
          <td className="border border-gray-500 px-4 py-2">ต่ำ</td>
          <td className="border border-gray-500 px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border border-gray-500 px-4 py-2">ความแม่นยำกับข้อมูลใหม่</td>
          <td className="border border-gray-500 px-4 py-2">ต่ำ</td>
          <td className="border border-gray-500 px-4 py-2">ต่ำ</td>
        </tr>
        <tr>
          <td className="border border-gray-500 px-4 py-2">โมเดลลักษณะ</td>
          <td className="border border-gray-500 px-4 py-2">เรียบง่ายเกินไป</td>
          <td className="border border-gray-500 px-4 py-2">ซับซ้อนเกินไป</td>
        </tr>
        <tr>
          <td className="border border-gray-500 px-4 py-2">ตัวอย่าง</td>
          <td className="border border-gray-500 px-4 py-2">Linear regression กับข้อมูลไม่เป็นเส้นตรง</td>
          <td className="border border-gray-500 px-4 py-2">Polynomial degree 20 กับข้อมูลเล็ก</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-2 text-center">4. ภาพจำลอง: ยิงเป้าเปรียบเทียบ</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <p className="mb-4 leading-relaxed">
    ลองจินตนาการถึงการยิงลูกดอกไปที่เป้า:
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>Bias สูง + Variance ต่ำ:</strong> ยิงพลาดซ้ำจุดเดิม ไม่โดนเป้าเลย → Underfitting</li>
    <li><strong>Bias ต่ำ + Variance สูง:</strong> ยิงกระจายไปทั่ว มีโดนเป้าแต่ไม่แม่นยำ → Overfitting</li>
    <li><strong>Bias ต่ำ + Variance ต่ำ:</strong> ยิงโดนเป้าซ้ำๆ แบบแม่นยำ → Generalization ดี</li>
    <li><strong>Bias สูง + Variance สูง:</strong> ยิงไม่โดนเป้าและกระจายมั่ว → โมเดลแย่ทั้งคู่</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">5. ความเกี่ยวข้องกับ Training และ Validation</h3>
  <p className="mb-4 leading-relaxed">
    โดยทั่วไปเราสามารถวิเคราะห์ bias และ variance ได้จากพฤติกรรมของ loss บน training และ validation set:
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>Train loss สูง + Val loss สูง:</strong> Bias สูง → โมเดลไม่เข้าใจ pattern</li>
    <li><strong>Train loss ต่ำ + Val loss สูง:</strong> Variance สูง → โมเดลจำมากเกินไป</li>
    <li><strong>Train loss ต่ำ + Val loss ต่ำ:</strong> โมเดลดี → เข้าใจ pattern และ generalize ได้</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2 text-center">6. ความสำคัญในการออกแบบโมเดล</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <p className="mb-4 leading-relaxed">
    การเข้าใจ bias และ variance จะช่วยให้เราสามารถเลือกความซับซ้อนของโมเดลที่เหมาะสม
    เช่นเลือกจำนวน layer, ขนาด hidden units, หรือแม้แต่ feature ที่ใช้เรียนรู้ เพื่อไม่ให้โมเดลเรียนรู้น้อยไปหรือจำมากเกินไป
  </p>

  <h3 className="text-xl font-semibold mb-2">7. Insight</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    Bias คือการ “เข้าใจผิดตั้งแต่แรก”<br/>
    Variance คือการ “เข้าใจมากเกินไปในสิ่งที่ไม่ควรเข้าใจ”<br/>
    เป้าหมายของเราใน Machine Learning คือการสร้างโมเดลที่ “เข้าใจได้พอดี”
  </div>
</section>


            {/* Section: Training vs Validation Loss */}
            <section id="training-vs-validation-loss" className="scroll-mt-20 mb30 min-h-[500px]">

                
          <h2 className="text-2xl font-semibold mb-4 text-center">Training Loss vs Validation Loss</h2>
          <div className="flex justify-center my-6">
                    <AdvancedImage cldImg={img4} />
                </div>
          

          <p className="mb-4 leading-relaxed">
            เมื่อฝึกโมเดล Machine Learning หนึ่งในเครื่องมือสำคัญที่ใช้ตรวจสอบพฤติกรรมของโมเดล คือการเปรียบเทียบระหว่าง Training Loss และ Validation Loss 
            ทั้งสองค่ามีความสำคัญในการบอกว่าโมเดลกำลังเรียนรู้อย่างมีประสิทธิภาพหรือไม่
          </p>

          <p className="mb-4 leading-relaxed">
            <strong>Training Loss</strong> คือค่าความผิดพลาดเฉลี่ยที่โมเดลคำนวณจากชุดข้อมูลฝึก (training set) ซึ่งเป็นข้อมูลที่โมเดลเห็นและใช้เรียนรู้โดยตรง 
            ส่วน <strong>Validation Loss</strong> คือค่าความผิดพลาดจากชุดข้อมูลตรวจสอบ (validation set) ที่ไม่ได้ใช้ในการฝึก แต่ใช้เพื่อดูว่าโมเดลสามารถ generalize ได้ดีหรือไม่
          </p>

          <h3 className="text-xl font-semibold mb-2 text-center">ลักษณะของ Loss ที่ดี</h3>
          <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img5} />
            </div>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li>Training Loss ควรลดลงต่อเนื่องเมื่อ Epoch เพิ่มขึ้น</li>
            <li>Validation Loss ควรลดลงตามในช่วงต้น และมีแนวโน้มคงที่เมื่อโมเดลเริ่มเข้าใจข้อมูล</li>
            <li>หากทั้งสองเส้นใกล้เคียงกัน แสดงถึงโมเดลที่ generalize ได้ดี</li>
          </ul>

          <h3 className="text-xl font-semibold mb-2 text-center">อาการของ Overfitting</h3>
          <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img6} />
            </div>
          <p className="mb-4 leading-relaxed">
            เมื่อโมเดลเริ่มเรียนรู้รายละเอียดเล็ก ๆ ในชุดข้อมูลฝึกมากเกินไป โดยไม่สามารถนำไปประยุกต์ใช้กับข้อมูลใหม่ได้ 
            มักจะเกิดอาการที่ Training Loss ลดลงเรื่อย ๆ แต่ Validation Loss กลับพุ่งสูงขึ้นในภายหลัง
          </p>
          <div className="bg-yellow-50 dark:bg-yellow-800 p-4 rounded-xl border-l-4 border-yellow-500 mb-4">
            <strong>Insight:</strong> Overfitting เปรียบเสมือนการท่องจำข้อสอบเดิมทุกคำ แต่พอเปลี่ยนข้อสอบกลับไม่เข้าใจเนื้อหาเลย
          </div>

          <h3 className="text-xl font-semibold mb-2 text-center">อาการของ Underfitting</h3>
          <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img7} />
            </div>
          <p className="mb-4 leading-relaxed">
            เมื่อโมเดลยังไม่สามารถเข้าใจ pattern ของข้อมูลได้เลยแม้แต่น้อย 
            ทั้ง Training และ Validation Loss จะสูงและแทบไม่ลดลง 
            อาการนี้มักเกิดจากโมเดลเล็กเกินไป หรือยังฝึกไม่เพียงพอ
          </p>

          <h3 className="text-xl font-semibold mb-2 text-center">วิธีวิเคราะห์จากกราฟ</h3>
          <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img8} />
            </div>
          <ul className="list-decimal pl-6 space-y-2 mb-4">
            <li>ดูจังหวะที่ Validation Loss เริ่มแยกจาก Training Loss</li>
            <li>ตรวจสอบว่ามี plateau หรือจุดหยุดของ improvement หรือไม่</li>
            <li>ใช้กราฟเพื่อเลือก Epoch ที่ดีที่สุดในการทำ Early Stopping</li>
          </ul>

          <h3 className="text-xl font-semibold mb-2 text-center">ตัวอย่างกราฟทั่วไป</h3>
          <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img9} />
            </div>
          <p className="mb-4 leading-relaxed">
            - ในช่วงแรก Training และ Validation Loss จะลดลงพร้อมกัน<br />
            - จากนั้น Validation Loss จะคงที่หรือเริ่มเพิ่ม หากโมเดลเริ่ม overfit<br />
            - จุดที่ Validation Loss ต่ำที่สุดมักใช้เป็นจุดหยุดการฝึกที่เหมาะสม
          </p>

          <h3 className="text-xl font-semibold mb-2">การปรับตามกราฟ</h3>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li>หาก Overfit: ใช้ Regularization เช่น L2, Dropout หรือเพิ่มข้อมูล</li>
            <li>หาก Underfit: เพิ่มขนาดโมเดล, เพิ่ม Epoch หรือปรับ learning rate</li>
            <li>ใช้ EarlyStopping เพื่อตัดจังหวะก่อน validation loss แย่ลง</li>
          </ul>

          <h3 className="text-xl font-semibold mb-2">สูตร Loss ที่ควรรู้</h3>
          <div className="overflow-x-auto mb-4">
            <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre-wrap">
{`# ตัวอย่าง Loss Function ที่ใช้ทั่วไป
# สำหรับ Regression
Loss = MeanSquaredError(y_true, y_pred)

# สำหรับ Classification
Loss = CrossEntropyLoss(y_true, y_pred)

# พร้อม Regularization
Loss = Loss + λ * RegularizationTerm`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold mb-2">สัญญาณที่ควรหยุดการฝึก</h3>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li>Validation Loss เริ่มสูงขึ้นเรื่อย ๆ หลังผ่านไปหลาย Epoch</li>
            <li>Validation Accuracy ไม่พัฒนาหรือเริ่มแย่ลง</li>
            <li>Training Loss ยังลดแต่ Validation ไม่เปลี่ยน</li>
          </ul>

          <h3 className="text-xl font-semibold mb-2">Insight สุดท้าย</h3>
          <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
            “Training Loss vs Validation Loss เป็นเหมือนเสียงเตือนจากสมองว่าเมื่อไหร่ควรเรียนรู้เพิ่ม และเมื่อไหร่ควรหยุดเพื่อไม่ให้หลงทาง”
          </div>
        </section>


       {/* Section: Bias-Variance Tradeoff */}
       <section id="bias-variance-tradeoff" className="mb-16 scroll-mt-20">
  <h2 className="text-2xl font-semibold mb-4 text-center">Bias-Variance Tradeoff</h2>
  <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img10} />
            </div>

  <p className="mb-4 leading-relaxed">
    ปัญหา Bias-Variance เป็นหัวใจสำคัญของการสร้างโมเดล Machine Learning ที่มีประสิทธิภาพและแม่นยำ 
    โดยพื้นฐานแล้วมันคือการหาสมดุลระหว่าง “การเรียนรู้ที่มากพอ” กับ “การไม่จำรายละเอียดเกินไป”
  </p>

  <h3 className="text-xl font-semibold mb-2">Bias คืออะไร?</h3>
  <p className="mb-4 leading-relaxed">
    Bias คือความเบี่ยงเบนของการทำนายจากความจริง เช่น โมเดลทำนายค่าเฉลี่ยตลอดเวลาโดยไม่พยายามเข้าใจข้อมูล
    โมเดลแบบนี้มักจะง่ายมาก เช่น Linear Regression แบบเส้นตรงกับข้อมูลที่โค้ง → เกิด Bias สูง เพราะเข้าไม่ถึง pattern แท้จริง
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>Bias สูง → โมเดลเข้าใจผิดพื้นฐาน ทำให้ทำนายผิดแม้บนข้อมูลฝึก</li>
    <li>มักพบในโมเดลง่ายเกินไป เช่น Linear บนข้อมูลที่มีลักษณะโค้งหรือซับซ้อน</li>
    <li>เกิดจากสมมุติฐานของโมเดลไม่เพียงพอ (Underfitting)</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2 text-center">Variance คืออะไร?</h3>
  <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img11} />
            </div>
  <p className="mb-4 leading-relaxed">
    Variance คือความผันผวนของโมเดลกับข้อมูลชุดฝึกเล็ก ๆ หากโมเดลมีความละเอียดมากเกินไป 
    เช่น Neural Network ที่มีพารามิเตอร์มาก มันอาจ “จำ” noise ในชุดฝึกได้ → ทำให้พอเจอข้อมูลใหม่ก็ทำนายพลาด
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>Variance สูง → โมเดลเปลี่ยนค่าทำนายมากเมื่อเจอข้อมูลฝึกต่างชุด</li>
    <li>มักพบในโมเดลซับซ้อนเกินไป เช่น Deep Network หรือ Polynomial สูงมาก</li>
    <li>เกิดจากการจำข้อมูลมากเกินไป (Overfitting)</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">เปรียบเทียบ Bias vs Variance</h3>
  <div className="overflow-x-auto mb-4">
    <table className="table-auto w-full text-left border border-gray-300 dark:border-gray-700">
      <thead>
        <tr className="bg-gray-100 dark:bg-gray-800">
          <th className="px-4 py-2">ลักษณะ</th>
          <th className="px-4 py-2">Bias สูง</th>
          <th className="px-4 py-2">Variance สูง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การทำนาย</td>
          <td className="border px-4 py-2">ค่าคาดเคลื่อนจากความจริง</td>
          <td className="border px-4 py-2">เปลี่ยนไปตามข้อมูลฝึก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ลักษณะโมเดล</td>
          <td className="border px-4 py-2">เรียบ ง่าย เกินไป</td>
          <td className="border px-4 py-2">ซับซ้อน จำ noise ได้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ผลที่ตามมา</td>
          <td className="border px-4 py-2">Underfitting</td>
          <td className="border px-4 py-2">Overfitting</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-2 text-center">จุดสมดุล: Bias-Variance Tradeoff</h3>
  <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img12} />
            </div>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ดีไม่ควร Bias สูงเกินไปหรือ Variance สูงเกินไป → ต้องอยู่ตรงกลาง 
    นี่แหละที่เรียกว่า Tradeoff: ลด bias → variance เพิ่ม, ลด variance → bias เพิ่ม 
    ดังนั้นต้อง “หาจุดที่ดีที่สุด” ที่ทั้งสองไม่สูงเกินไป
  </p>

  <p className="mb-4 leading-relaxed">
    วิธีหนึ่งที่ใช้ได้คือดูจาก Learning Curve: หาก training loss และ validation loss ต่างกันมาก → variance สูง  
    หาก training loss สูง → bias สูง
  </p>

  <h3 className="text-xl font-semibold mb-2">ผลจากความซับซ้อนของโมเดล</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>โมเดลง่ายเกินไป → ไม่เข้าใจข้อมูล (Bias สูง)</li>
    <li>โมเดลซับซ้อนเกินไป → จำ noise (Variance สูง)</li>
    <li>โมเดลกลางๆ ที่กำลังพอดี → เข้าใจ pattern โดยไม่จำมากเกินไป</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">ความเกี่ยวข้องกับ Regularization</h3>
  <p className="mb-4 leading-relaxed">
    Regularization คือวิธีการควบคุมความซับซ้อนของโมเดล → ช่วยลด variance โดยไม่ทำให้ bias เพิ่มขึ้นมากนัก  
    เช่น L1, L2, Dropout, Early Stopping เป็นเครื่องมือสำคัญในการบาลานซ์ Bias-Variance
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">ตัวอย่างจาก Polynomial Regression</h3>
  <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img13} />
            </div>
  <p className="mb-4 leading-relaxed">
    สมมุติมีข้อมูลที่โค้งน้อย → ใช้ Polynomial degree 1 → เส้นตรง → Bias สูง  
    หากใช้ degree 20 → เส้นจะพับเยอะมากจนจำทุกจุดในชุดฝึกได้ → Variance สูง  
    Degree ที่เหมาะสมอาจอยู่แค่ 3–5 ซึ่งเรียนรู้รูปทรงได้ดีโดยไม่จำ noise
  </p>

  <h3 className="text-xl font-semibold mb-2">Insight</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    Bias คือการมองโลกแคบ ไม่เข้าใจสิ่งที่ซับซ้อน<br/>
    Variance คือการมองโลกกว้างเกินไป จนจำทุกเรื่องแม้ไม่สำคัญ<br/>
    ศิลปะของโมเดลดี คือ “เรียนรู้พอดี” ไม่มากเกินไป ไม่น้อยเกินไป
  </div>

  <p className="mb-4 leading-relaxed">
    ปรับสมดุลนี้ให้ดี → โมเดลจะ generalize ได้ดี → ใช้งานจริงแม่นยำขึ้น  
    เพราะไม่ได้ทำนายดีแค่ในห้องเรียน แต่ลงสนามจริงแล้วยังตอบได้ถูก
  </p>
</section>

      {/* Section: Model Capacity */}
      <section id="model-capacity" className="mb-16 scroll-mt-20">
  <h2 className="text-2xl font-semibold mb-4 text-center">Model Capacity คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <p className="mb-4 leading-relaxed">
    Model Capacity คือความสามารถในการเรียนรู้ฟังก์ชันหรือ pattern ที่ซับซ้อนได้ของโมเดล ยิ่งมี capacity สูง โมเดลยิ่งสามารถเรียนรู้ฟังก์ชันที่มีลักษณะโค้งงอ ซับซ้อน หรือไม่เป็นเชิงเส้นได้ดี ในทางกลับกัน ถ้า capacity ต่ำ โมเดลจะสามารถเรียนรู้ได้แค่ฟังก์ชันง่าย ๆ เท่านั้น
  </p>

  <p className="mb-4 leading-relaxed">
    ความสามารถของโมเดลขึ้นอยู่กับจำนวนพารามิเตอร์ ขนาดของเลเยอร์ ความลึกของโครงสร้าง และประเภทของโมเดล เช่น Linear Regression, Decision Trees, Neural Networks โดยโมเดลที่ซับซ้อนกว่าจะมี capacity สูงกว่า แต่ก็มาพร้อมความเสี่ยงที่จะ overfit ได้ง่ายหากไม่มีการควบคุม
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">ความสัมพันธ์ระหว่าง Capacity และการเรียนรู้</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img15} />
  </div>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>โมเดลขนาดเล็ก (low capacity): เรียนรู้ได้ช้า ไม่สามารถจับ pattern ที่ซับซ้อนได้ ทำให้ underfit</li>
    <li>โมเดลขนาดใหญ่ (high capacity): เรียนรู้ได้ลึกและซับซ้อน แต่อาจจำข้อมูลเกินไปจนเกิด overfit</li>
    <li>โมเดลที่มี capacity พอเหมาะ: สามารถเรียนรู้ pattern จริงในข้อมูลได้โดยไม่จดจำ noise</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">ตัวอย่างเปรียบเทียบ</h3>
  <p className="mb-4 leading-relaxed">
    สมมติให้ชุดข้อมูลมีลักษณะโค้งแบบพาราโบลา ถ้าใช้โมเดลเส้นตรง (linear model) จะได้เส้นที่พยายามตัดผ่านตรงกลาง ซึ่งไม่สอดคล้องกับข้อมูล → underfit แต่ถ้าใช้ polynomial degree สูงเกินไป เช่น degree=10 อาจได้เส้นที่แกว่งมากไปมาจับข้อมูลทุกจุด → overfit
  </p>

  <h3 className="text-xl font-semibold mb-2">Model Capacity ขึ้นอยู่กับอะไรบ้าง?</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>จำนวนพารามิเตอร์:</strong> โมเดลที่มีพารามิเตอร์มากสามารถเรียนรู้ฟังก์ชันซับซ้อนได้มากกว่า</li>
    <li><strong>โครงสร้างโมเดล:</strong> ความลึก ความกว้าง และรูปแบบการเชื่อมต่อมีผลต่อ capacity</li>
    <li><strong>ประเภทโมเดล:</strong> โมเดลบางประเภทมี capacity ที่ถูกจำกัด เช่น linear models</li>
    <li><strong>Feature ที่ป้อนเข้า:</strong> แม้โมเดลจะใหญ่ แต่ถ้า features มีข้อมูลไม่ครบ ก็อาจจำกัดการเรียนรู้</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">กรณีศึกษาของ Model Capacity</h3>
  <p className="mb-4 leading-relaxed">
    ในการทำ Polynomial Regression หากเลือก degree=1 จะได้เส้นตรง → underfit, degree=15 → overfit จุดที่เหมาะสมคือ degree ที่สามารถจับแนวโน้มหลักได้โดยไม่แกว่งไปมาเกินความจำเป็น ซึ่งต้องดูจาก validation loss และการประเมิน model บนข้อมูลที่ไม่เคยเห็น
  </p>

  <h3 className="text-xl font-semibold mb-2">เทคนิคการควบคุม Capacity</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>เลือกขนาดของโมเดลให้พอดีกับปริมาณข้อมูล</li>
    <li>ใช้เทคนิคอย่าง regularization (L1, L2) เพื่อจำกัดความยืดหยุ่นของโมเดล</li>
    <li>ใช้ Early Stopping เพื่อไม่ให้โมเดลเรียนรู้ noise จากข้อมูลมากเกินไป</li>
    <li>ทำ Feature Selection เพื่อลด dimension ที่ไม่จำเป็น</li>
    <li>ใช้ Cross Validation เพื่อเลือก hyperparameter ที่เหมาะสมที่สุด</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2 text-center">Model Complexity ≠ Accuracy เสมอไป</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img16} />
  </div>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ใหญ่ขึ้นไม่ได้แปลว่าจะแม่นยำขึ้นเสมอไป เพราะอาจเรียนรู้ noise และเกิด overfit ได้ การเลือกโมเดลต้องดู context เช่น ขนาด dataset, ความซับซ้อนของปัญหา, และความต้องการด้านเวลาและทรัพยากรในการเทรน
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">Visualization การเปรียบเทียบ</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img17} />
  </div>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>เส้น decision boundary ของโมเดลที่ capacity ต่ำ จะเรียบง่าย ไม่โค้งงอมาก</li>
    <li>โมเดลที่มี capacity สูงมาก จะมี decision boundary ที่พยายามจับข้อมูลทุกจุด แม้แต่จุดผิดพลาด</li>
    <li>จุดที่เหมาะสมคือ boundary ที่โค้งพอดี ไม่เรียบเกินหรือซับซ้อนเกิน</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">คำแนะนำในการเลือก Model Capacity</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>เริ่มจากโมเดลเรียบง่าย แล้วเพิ่มความซับซ้อนทีละขั้นพร้อมประเมิน validation performance</li>
    <li>ใช้ learning curve เพื่อตรวจว่าโมเดลเรียนรู้ได้ดีแค่ไหนกับข้อมูลเพิ่มขึ้น</li>
    <li>หาก training error และ validation error ห่างกันมาก แปลว่าโมเดลอาจซับซ้อนเกิน</li>
  </ul>

  <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
    <strong>Insight:</strong><br />
    Model Capacity คือศิลปะของการเลือกเครื่องมือให้เหมาะกับปัญหา ถ้าใช้เลื่อยตัดไม้ใหญ่ก็จะเปลืองแรงเกินไป แต่ถ้าใช้ใบมีดเล็กไปก็ตัดไม่ขาด
  </div>
</section>


     {/* Section: Polynomial Regression Case Study */}
     <section id="case-study-polynomial" className="mb-16 scroll-mt-20">
  <h2 className="text-2xl font-semibold mb-4 text-center">Case Study: Polynomial Regression</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img18} />
  </div>

  <p className="mb-4 leading-relaxed">
    Polynomial Regression คือเทคนิคหนึ่งที่ใช้ในการจำลองความสัมพันธ์แบบโค้งระหว่างตัวแปรต้นและตัวแปรตาม
    แทนที่จะใช้เส้นตรงเหมือน Linear Regression ทั่วไป เทคนิคนี้สามารถจับ pattern ที่ซับซ้อนได้มากกว่า
    โดยเพิ่มพจน์กำลังสอง กำลังสาม หรือสูงกว่านั้นเข้าไปในสมการ
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">ความแตกต่างระหว่าง Linear และ Polynomial Regression</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img19} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>Linear Regression พยายามหาความสัมพันธ์เชิงเส้น ซึ่งไม่เหมาะกับข้อมูลที่มีรูปแบบโค้ง</li>
    <li>Polynomial Regression สามารถสร้างเส้นโค้งที่ปรับเข้ากับข้อมูลได้ดีขึ้น โดยไม่ต้องเปลี่ยนโมเดลทั้งหมด</li>
    <li>ยิ่งเพิ่ม degree ของพหุนามมากเท่าไร เส้นก็จะสามารถโค้งไปตามข้อมูลได้มากขึ้น</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">ทดลองใช้ Polynomial Regression</h3>
<p className="mb-4 leading-relaxed">
  ในการทดลองนี้จะเริ่มจากสร้างข้อมูลเทียมที่มีความโค้งเล็กน้อย จากนั้นฝึกโมเดลทั้ง Linear และ Polynomial เปรียบเทียบผลลัพธ์
  การวัดผลจะใช้ Mean Squared Error (MSE) เพื่อดูว่าโมเดลเข้าใจข้อมูลมากน้อยแค่ไหน
</p>

<div className="overflow-x-auto mb-4">
  <div className="inline-block bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre">
    <pre>
{`from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลจำลอง
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Polynomial Regression (degree=3)
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# Visualization
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, y_pred_linear, color='blue', label='Linear')
plt.plot(X, y_pred_poly, color='red', label='Polynomial (deg=3)')
plt.legend()
plt.show()`}
    </pre>
  </div>
</div>



  <h3 className="text-xl font-semibold mb-2 text-center">ผลลัพธ์จากการทดลอง</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img20} />
  </div>
  <p className="mb-4 leading-relaxed">
    เมื่อเปรียบเทียบระหว่างเส้นของ Linear Regression กับ Polynomial Regression พบว่า:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>Linear Regression ให้เส้นตรงที่พยายามลากผ่านจุดข้อมูลโดยเฉลี่ย ซึ่งไม่สามารถจับโครงสร้างของข้อมูลได้ดี</li>
    <li>Polynomial Regression สามารถจับลักษณะโค้งของข้อมูลได้ชัดเจน แม้ว่าจะมี noise ปะปนอยู่บ้าง</li>
    <li>กรณีที่ใช้ degree สูงเกินไป เช่น 15 ขึ้นไป เส้นจะพยายามผ่านจุดทุกจุด → เกิด Overfitting</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">ผลกระทบจากระดับของ Degree</h3>
  <p className="mb-4 leading-relaxed">
    ระดับของ degree เปรียบเสมือนขนาดของสมองของโมเดล ถ้าเล็กเกินไปจะคิดไม่พอ (Underfit)
    ถ้าใหญ่เกินไปจะคิดมากเกินไปจนเก็บรายละเอียดเล็กน้อยที่ไม่สำคัญ (Overfit)
  </p>
  <p className="mb-4 leading-relaxed">
    การเลือก degree ที่เหมาะสมต้องดูจาก validation loss หรือใช้เทคนิค cross-validation เพื่อป้องกันการ bias
    และให้โมเดลสามารถ generalize ไปยังข้อมูลใหม่ได้ดีขึ้น
  </p>

  <h3 className="text-xl font-semibold mb-2">Visualization: Degree ต่างกันแล้วเป็นยังไง?</h3>
  <p className="mb-4 leading-relaxed">
    จากภาพตัวอย่าง เมื่อใช้ degree ต่ำ (1–2) โมเดลจะพลาด pattern หลัก
    เมื่อใช้ degree สูง (10–15) โมเดลจะเก็บ noise ทุกจุด
    จุดที่เหมาะสมจะอยู่กลาง ๆ (เช่น 3–5) ขึ้นกับจำนวนและลักษณะข้อมูล
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">เปรียบเทียบในมุมของ Bias/Variance</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img21} />
  </div>
  
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Degree ต่ำ:</strong> Bias สูง, Variance ต่ำ → โมเดลเข้าใจไม่พอ</li>
    <li><strong>Degree สูง:</strong> Bias ต่ำ, Variance สูง → โมเดลเข้าใจเยอะเกิน จนเก็บ noise</li>
    <li><strong>Degree กลาง ๆ:</strong> Bias-Variance Tradeoff ที่สมดุล → โมเดลพอดี</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">การปรับ degree ให้เหมาะสม</h3>
  <p className="mb-4 leading-relaxed">
    ใช้เทคนิค Grid Search ร่วมกับ Cross Validation เพื่อทดลองหลาย ๆ degree และดูผลลัพธ์จาก validation set
    ควรเลือก degree ที่ให้ validation error ต่ำสุด ไม่ใช่ training error ต่ำสุด
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    Polynomial Regression เปรียบเหมือนการเลือกเลนส์ในการมองโลก<br />
    ถ้าเลนส์เบลอเกิน จะมองไม่เห็นสิ่งสำคัญ<br />
    ถ้าเลนส์ซูมมากเกินไป จะเห็นแค่รายละเอียดเล็ก ๆ จนพลาดภาพรวม<br />
    ศิลปะอยู่ที่การเลือกเลนส์ที่ “พอดี” สำหรับโจทย์แต่ละชุด
  </div>
</section>


           {/* Section: Model Selection Techniques */}
           <section id="model-complexity-selection" className="mb-16 scroll-mt-20">
          <h2 className="text-2xl font-semibold mb-4 text-center">การเลือก Model Complexity อย่างมีหลักการ</h2>
          <p className="mb-4 leading-relaxed">
            การเลือกความซับซ้อนของโมเดล (Model Complexity) ถือเป็นจุดเริ่มต้นที่สำคัญของการสร้างโมเดลที่ทั้งแม่นยำและ generalize ได้ดี
            เพราะโมเดลที่ซับซ้อนเกินไปมีแนวโน้มจะ overfit ข้อมูล ในขณะที่โมเดลที่เรียบง่ายเกินไปอาจ underfit จับ pattern ที่สำคัญไม่ได้
            ดังนั้นการตัดสินใจเลือกขนาดของโมเดลจึงควรมีแนวทางและหลักการชัดเจน
          </p>

          <h3 className="text-xl font-semibold mb-3">1. เริ่มจากโมเดลง่ายเสมอ</h3>
          <p className="mb-4 leading-relaxed">
            การเริ่มต้นด้วยโมเดลที่ง่าย เช่น linear regression, logistic regression, หรือ shallow tree model ช่วยให้เข้าใจภาพรวมของปัญหา
            และสามารถใช้ baseline เหล่านี้เปรียบเทียบกับโมเดลที่ซับซ้อนในภายหลังได้
          </p>

          <h3 className="text-xl font-semibold mb-3">2. ใช้ Learning Curve ประเมินความสามารถ</h3>
          <p className="mb-4 leading-relaxed">
            การวาดกราฟ Learning Curve ของ training loss และ validation loss ตามขนาดของข้อมูลหรือจำนวน epoch เป็นเครื่องมือสำคัญในการวิเคราะห์
            ว่าโมเดลกำลัง overfit หรือ underfit อย่างไร
          </p>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li>เส้น loss ทั้งสองห่างกันมาก → overfitting</li>
            <li>เส้น loss ทั้งสองสูงและใกล้กัน → underfitting</li>
            <li>เส้น converged ใกล้กันในระดับต่ำ → generalization ดี</li>
          </ul>

          <h3 className="text-xl font-semibold mb-3">3. ค่อยๆ เพิ่ม Complexity พร้อม Monitor</h3>
          <p className="mb-4 leading-relaxed">
            แทนที่จะเริ่มด้วยโมเดลขนาดใหญ่ ควรเพิ่มขนาดทีละน้อย เช่น เพิ่มจำนวน neuron ต่อเลเยอร์, จำนวนเลเยอร์ หรือเพิ่ม degree ของ polynomial regression
            พร้อม monitor performance ผ่าน validation set และ cross-validation score
          </p>

          <h3 className="text-xl font-semibold mb-3">4. ใช้ Cross Validation</h3>
          <p className="mb-4 leading-relaxed">
            K-Fold Cross Validation เป็นเครื่องมือมาตรฐานในการประเมินว่าโมเดลนั้น generalize ได้ดีแค่ไหนกับข้อมูลที่ไม่เคยเห็น
            ยิ่ง variance ของ score แต่ละ fold ต่ำ แสดงว่าโมเดลมีความเสถียรดี
          </p>

          <h3 className="text-xl font-semibold mb-3">5. ลองหลายค่า Hyperparameters</h3>
          <p className="mb-4 leading-relaxed">
            การใช้ Grid Search หรือ Random Search เพื่อหาค่า hyperparameters ที่เหมาะสม เช่น learning rate, number of layers, dropout rate, regularization strength
            ช่วยให้เข้าใจว่าโมเดลตอบสนองต่อความซับซ้อนอย่างไร
          </p>

          <h3 className="text-xl font-semibold mb-3">6. ดูค่า Validation Accuracy ร่วมกับ Loss</h3>
          <p className="mb-4 leading-relaxed">
            บางครั้ง loss อาจลดลงอย่างต่อเนื่องแต่ accuracy ไม่เพิ่มขึ้นหรือลดลง แสดงว่าโมเดลอาจจะ overfit หรือตัดสินใจไม่แม่นยำพอ
            การดูสองค่าไปพร้อมกันช่วยให้เลือกโมเดลได้แม่นยำยิ่งขึ้น
          </p>

          <h3 className="text-xl font-semibold mb-3">7. ใช้ Early Stopping ป้องกัน Overfitting</h3>
          <p className="mb-4 leading-relaxed">
            แม้โมเดลจะซับซ้อน แต่ถ้าใช้ early stopping อย่างเหมาะสมก็สามารถป้องกันไม่ให้โมเดลเรียนรู้ noise เกินไป
            ซึ่งช่วยให้รักษาความสามารถในการ generalize ได้
          </p>

          <h3 className="text-xl font-semibold mb-3">8. ลอง Visualize Decision Boundary</h3>
          <p className="mb-4 leading-relaxed">
            ในปัญหา classification แบบ 2D การ plot decision boundary ช่วยให้เข้าใจว่าโมเดลกำลังจับ pattern อย่างไร
            เส้นตัดที่ซับซ้อนเกินไปมักจะเป็นสัญญาณของ overfitting
          </p>

          <h3 className="text-xl font-semibold mb-3">9. ใช้ Validation Curve เพื่อดูผลของ Hyperparameters</h3>
          <p className="mb-4 leading-relaxed">
            การวาดกราฟที่แกน x เป็นค่าของ hyperparameter และแกน y เป็น performance score
            จะช่วยบอกว่าโมเดลกำลัง overfit หรือ underfit ในแต่ละค่า
          </p>

          <h3 className="text-xl font-semibold mb-3">10. Insight: Complexity ≠ Better</h3>
          <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow-xl">
            <p>
              การเลือกโมเดลไม่ควรอิงแค่ความล้ำของสถาปัตยกรรมหรือขนาด parameter
              แต่ควรเลือกจากความเหมาะสมกับปัญหา, ความง่ายในการ deploy และ generalization
              โมเดลที่ดีคือโมเดลที่เข้าใจ pattern สำคัญในข้อมูลได้พอดี ไม่มากไปหรือน้อยไป
            </p>
          </div>
        </section>

      {/* Section: Insight */}
      <section id="insight" className="mb-16 scroll-mt-20 min-h-[500px]">
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow-xl">
    <h2 className="text-2xl font-bold mb-4"> Insight: ศิลปะแห่งสมดุลทางปัญญา</h2>
    <p className="mb-4 leading-relaxed">
      โมเดลใน Machine Learning เปรียบได้กับสมองที่ต้องเรียนรู้จากประสบการณ์ ถ้าจำได้น้อยเกินไป
      ก็จะกลายเป็นคนหัวแข็งที่ดึงแต่ความเชื่อเดิม ๆ (bias สูง) และถ้าจำทุกสิ่งแบบเป๊ะเกินไป
      ก็เหมือนคนขี้ระแวง ที่กลัวทุกการเปลี่ยนแปลง (variance สูง)
    </p>
    <p className="mb-4 leading-relaxed">
      ศาสตร์ของ Bias-Variance จึงไม่ใช่แค่เรื่องตัวเลขหรือสูตรคณิตศาสตร์
      แต่มันคือการเข้าใจธรรมชาติของการเรียนรู้ ว่าไม่ใช่ทุกอย่างที่ควรถูกจดจำ และไม่ใช่ทุกสิ่งควรถูกละเลย
    </p>
    <p className="mb-4 leading-relaxed">
      การเลือกความซับซ้อนของโมเดล จึงเป็นการตัดสินใจทางจิตวิทยาแบบหนึ่ง
      ระหว่าง “อยากรู้ให้ลึก” กับ “อยากเข้าใจให้จริง” เพราะความรู้ที่มากไปก็กลายเป็นกับดัก
      และความเข้าใจที่ผิวเผินก็อาจทำให้เดินทางผิด
    </p>
    <p className="mb-4 leading-relaxed">
      Model Capacity เป็นตัวชี้วัดขนาดของสมองในการรับรู้ บางโมเดลเหมือนเด็กน้อยที่มองโลกง่าย ๆ
      บางโมเดลก็เหมือนนักปรัชญาที่เห็นทุกความซับซ้อน
      แต่สิ่งที่สำคัญกว่าคือการรู้ว่าเมื่อไหร่ควรใช้โมเดลแบบไหน
    </p>
    <p className="mb-4 leading-relaxed">
      Regularization จึงเข้ามาเหมือนเครื่องมือที่ช่วยพาโมเดลกลับสู่สมดุล
      ไม่ให้เอียงข้างไปกับข้อมูลฝึกมากเกินไป เช่นเดียวกับคนที่ควรมีสติระหว่างการรับข้อมูล
      รู้จักตัด noise ทิ้ง และเก็บสาระที่แท้จริงเอาไว้
    </p>
    <p className="mb-4 leading-relaxed">
      Dropout, Early Stopping, L1/L2 Regularization, BatchNorm — ล้วนคือวิธีเบรกไม่ให้โมเดลบ้าพลัง
      เพราะปัญญาที่แท้จริงไม่ได้เกิดจากการ “จำมากที่สุด” แต่จากการ “จำเป็นและใช้เป็น”
    </p>
    <p className="mb-4 leading-relaxed">
      Visualization อย่าง Learning Curve ก็เปรียบเหมือนกระจกที่สะท้อนภาพพฤติกรรมของโมเดล
      ว่าเริ่มจำเกินไปหรือยัง หรือยังเข้าใจไม่พอ
    </p>
    <p className="mb-4 leading-relaxed">
      เมื่อเทรนโมเดลครั้งต่อไป ลองไม่รีบมองแค่ accuracy แต่กลับไปดูว่ามันเข้าใจ “สาระ” ของข้อมูลแล้วหรือยัง
      เพราะโมเดลที่ดี ไม่ใช่โมเดลที่จำทุกอย่างได้หมด แต่คือโมเดลที่รู้จัก “ตัดสิ่งไม่จำเป็นออก”
    </p>
    <p className="mb-4 leading-relaxed">
      เช่นเดียวกับคนที่ประสบความสำเร็จ มักไม่ได้รู้เยอะที่สุด แต่มักรู้ว่าสิ่งไหนไม่ควรรู้
      และใช้สิ่งที่รู้ให้ตรงจุดเสมอ
    </p>

    <div className="mt-6 bg-white dark:bg-gray-800 p-5 rounded-xl shadow border border-yellow-400">
      <p className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-300">
        “Bias-Variance คือศิลปะแห่งการเลือกขนาดสมองให้พอดี — ไม่เล็กเกินจนมองอะไรไม่เห็น และไม่ใหญ่เกินจนคิดมากไปทุกเรื่องที่ไม่สำคัญ”
      </p>
      <ul className="list-disc pl-6 space-y-2 text-base">
        <li><strong>Bias สูง:</strong> มองโลกง่ายเกินไป → ทำนายไม่แม่นเพราะเข้าใจไม่ลึก</li>
        <li><strong>Variance สูง:</strong> มองทุกอย่างละเอียดเกินไป → สับสนเมื่อเจอสิ่งใหม่</li>
        <li><strong>จุดสมดุล:</strong> คือการเข้าใจที่พอดี พอเข้าใจโครงสร้างโดยไม่หลงในรายละเอียด</li>
      </ul>
    </div>

    <div className="mt-6 text-center text-base text-gray-800 dark:text-gray-100 italic">
      ในยุคที่ข้อมูลมหาศาล ศิลปะของการเรียนรู้คือการ “รู้ให้พอดี”  
      ไม่จำหมด แต่จำให้ลึก และไม่มองข้ามสิ่งสำคัญ
    </div>
  </div>
</section>


        {/* Section: Mini Quiz */}
        <section id="quiz" className="mb-16 scroll-mt-20 min-h-[500px]">
          <MiniQuiz_Day10 theme={theme} />
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
        <ScrollSpy_Ai_Day10 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day10_BiasVariance;
