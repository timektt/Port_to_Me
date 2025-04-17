import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day9 from "./scrollspy/ScrollSpy_Ai_Day9";
import MiniQuiz_Day9 from "./miniquiz/MiniQuiz_Day9";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";

const Day9_Regularization = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  // Placeholder images
  const img1 = cld.image('Regularization1').format('auto').quality('auto').resize(scale().width(600));
  const img2 = cld.image('Regularization2').format('auto').quality('auto').resize(scale().width(600));
  const img3 = cld.image('Regularization3').format('auto').quality('auto').resize(scale().width(600));
  const img4 = cld.image('Regularization4').format('auto').quality('auto').resize(scale().width(600));
  const img5 = cld.image('Regularization5').format('auto').quality('auto').resize(scale().width(600));
  const img6 = cld.image('Regularization6').format('auto').quality('auto').resize(scale().width(600));
  const img7 = cld.image('Regularization7').format('auto').quality('auto').resize(scale().width(500));
  const img8 = cld.image('Regularization8').format('auto').quality('auto').resize(scale().width(500));
  const img9 = cld.image('Regularization9').format('auto').quality('auto').resize(scale().width(500));
  const img10 = cld.image('Regularization10').format('auto').quality('auto').resize(scale().width(500));
  const img11 = cld.image('Regularization11').format('auto').quality('auto').resize(scale().width(500));
  const img12 = cld.image('Regularization12').format('auto').quality('auto').resize(scale().width(500));
  const img13 = cld.image('Regularization13').format('auto').quality('auto').resize(scale().width(600));
  const img14 = cld.image('Regularization14').format('auto').quality('auto').resize(scale().width(500));
  const img15 = cld.image('Regularization15').format('auto').quality('auto').resize(scale().width(600));
  const img16 = cld.image('Regularization16').format('auto').quality('auto').resize(scale().width(500));
  const img17 = cld.image('Regularization17').format('auto').quality('auto').resize(scale().width(600));
  const img18 = cld.image('Regularization18').format('auto').quality('auto').resize(scale().width(500));

  

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 9: Regularization & Generalization</h1>

      {/* 1. Regularization */}
<section id="what-is-regularization" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Regularization คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>
  {/* เริ่มต้นอธิบาย */}
  <p className="mb-4 leading-relaxed">
    เมื่อโมเดลเรียนรู้จากข้อมูล บางครั้งจะจดจำ “เสียงรบกวน” หรือรายละเอียดเล็กๆ ที่ไม่สำคัญไว้ด้วย  
    พอเจอข้อมูลใหม่ที่ต่างจากเดิม โมเดลก็จะทำผิดพลาดเพราะไปยึดติดกับสิ่งที่ไม่ใช่สาระสำคัญ  
    Regularization ทำหน้าที่เหมือน “เบรก” ที่คอยชะลอไม่ให้โมเดลเก็บทุกรายละเอียดจนเกินไป  
    ทำให้โฟกัสที่รูปแบบหลักของข้อมูลจริง แทนที่จะไปจับ noise เล็กๆ น้อยๆ
  </p>
  <p className="mb-4 leading-relaxed">
    ลองนึกภาพคนเรียนหนังสือ ถ้าเซฟแค่เฉพาะข้อสอบเก่าแบบเป๊ะๆ แต่ไม่ได้จับแนวคิดหลัก  
    พอเจอคำถามใหม่ๆ ก็จะสับสน เพราะไปจำผิดๆ ถูกๆ มาเต็มหัว  
    Regularization ก็เปรียบเหมือนการสอนให้จับเฉพาะ concept หลัก ไม่ให้จดจำตัวเลข  
    หรือสีสันของกระดาษข้อสอบเก่า จึงพร้อมรับคำถามรูปแบบใหม่ได้ดีขึ้น
  </p>
  <h3 className="text-xl font-semibold mb-2">ทำไม Overfit ถึงไม่ดี?</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>โมเดลพริ้นท์ถูกทุกชุดข้อมูลฝึก แต่พอเจอชุดใหม่ทำนายผิดพลาดมาก</li>
    <li>เหมือนคนอ่านข้อสอบเก่าวนไปวนมา จับ pattern เดิม แต่ไม่ได้เข้าใจตรรกะ</li>
    <li>ใช้ resource ในการปรับน้ำหนักมากเกินไปโดยไม่จำเป็น</li>
  </ul>
  <h3 className="text-xl font-semibold mb-2 text-center">Generalize คืออะไร?</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
    </div>
  <p className="mb-4 leading-relaxed">
    เมื่อโมเดล generalize ได้ดี หมายถึงจับ pattern ที่สำคัญแล้วปรับใช้กับข้อมูลที่ไม่เคยเห็นมาก่อนได้  
    เปรียบกับการเข้าใจคอนเซปต์จริงๆ มากกว่าจำข้อสอบเก่า ถ้าเจอโจทย์ใหม่ก็ยังรู้แนวทางตีโจทย์ได้
  </p>
  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-yellow-500 mb-4">
    <strong>Insight:</strong><br />
    Regularization ไม่ใช่ลดความสามารถ แต่คือการโฟกัสที่ “สิ่งสำคัญ”  
    เหมือนปรับเลนส์ให้เบลอ background เพื่อดึง subject ให้เด่นขึ้น
  </div>
  <h3 className="text-xl font-semibold mb-2">เมื่อไรต้องใส่ Regularization?</h3>
  <p className="mb-4 leading-relaxed">
    ถ้า loss บนชุดฝึกลดลงเร็วมาก แต่ validation loss ไม่ลดตาม หรือเริ่มเพิ่ม นั่นคือสัญญาณของ overfit  
    ช่วงนั้นจึงเหมาะจะเปิดเบรกด้วย regularization เพื่อให้โมเดลหยุดจับ noise
  </p>
  <h3 className="text-xl font-semibold mb-2 text-center">สัญญาณบ่งบอก Overfitting</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <ol className="list-decimal pl-6 mb-4 space-y-2">
    <li>Training loss ต่ำ แต่ validation loss สูงหรือไม่ลดลง</li>
    <li>Training accuracy สูง แต่ validation accuracy ตก</li>
    <li>โมเดลทำนายรูปแบบเดิมแม่น แต่รูปแบบใหม่หลุดหมด</li>
  </ol>
  <p className="mb-4 leading-relaxed">
    เมื่อเห็นอาการเหล่านี้ ให้เริ่มใช้ regularization เช่น L1/L2, Dropout, หรือเพิ่มข้อมูล  
    เพื่อเบรกโมเดลไม่ให้เรียนรู้ส่วนที่ไม่สำคัญจนเกินไป
  </p>
  {/* จบ section */}
</section>


     {/* 2. L1 vs L2 */}
<section id="l1-vs-l2" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">L1 vs L2 Regularization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในการฝึกโมเดล เราต้องระวังไม่ให้มัน “จำ” ข้อมูลแบบเป๊ะจนแทบจะจำ noise ใน dataset ได้หมด
    L1 กับ L2 Regularization คือเครื่องมือที่คอยเบรกไม่ให้ weight ในโมเดลพุ่งสูงเกินไป
    หรือบางตัวกลายเป็นศูนย์อย่างรวดเร็วจนโมเดลต้องพยายามเรียนรู้แต่ส่วนที่สำคัญเท่านั้น
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">หลักการทำงาน</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>
      <strong>L1 Regularization (Lasso):</strong><br/>
      ใน Loss Function จะมีค่า λ × ∑|w| ต่อท้ายเข้ามา
      ซึ่งเป็นการบวกค่าความผิดพลาดของ weight ทุกตัวแบบ absolute
      ทำให้บาง weight ถูกบีบจนกลายเป็น 0 อย่างเด็ดขาด
    </li>
    <li>
      <strong>L2 Regularization (Ridge):</strong><br/>
      ใน Loss Function จะมีค่า λ × ∑w² ต่อท้าย
      เมื่อเพิ่ม term นี้ โมเดลจะต้องแลกระหว่างลด Loss กับลดค่ากำลังสองของ weight
      ส่งผลให้ weight ต่ำลงโดยไม่หายไปหมด
    </li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">เปรียบเทียบง่าย ๆ</h3>
  <p className="mb-4">
    ลองนึกถึง weight แต่ละตัวเสมือนนักวิ่งในการแข่งขัน<br/>
    • L1 คือการตัดคะแนนนักวิ่งที่วิ่งช้าจนกลายเป็นศูนย์ ไม่ให้ได้อันดับเลย<br/>
    • L2 คือการเอาคะแนนมาเป็นกำลังสอง ลดคะแนนทุกคนลง แต่ไม่มีใครถูกตัดออกจากการแข่งขัน
  </p>

  <h3 className="text-xl font-semibold mb-2">สูตรรวมใน Loss</h3>
  <div className="overflow-x-auto mb-4">
    <code className="bg-gray-800 text-yellow-300 px-2 py-1 rounded">
      Loss = OriginalLoss + λ₁ * Σ|w|   {/* L1 */}
    </code><br/>
    <code className="bg-gray-800 text-yellow-300 px-2 py-1 rounded">
      Loss = OriginalLoss + λ₂ * Σw²  {/* L2 */}
    </code>
  </div>

  <h3 className="text-xl font-semibold mb-2">เมื่อใช้บนจริง</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>โมเดลที่ใช้ L1 จะได้โครงสร้างเลเยอร์ที่กระชับ มี weight ที่ไม่จำเป็นถูกกำจัด</li>
    <li>โมเดลที่ใช้ L2 จะเรียนรู้ทุก feature แต่จะแจก gradient ให้ weight ทุกตัวลดลงอย่างสม่ำเสมอ</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">โค้ดตัวอย่าง PyTorch</h3>
<div className="overflow-x-auto mb-4">
  <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre-wrap w-full max-w-full">
{`# สร้าง optimizer พร้อม L2 regularization (weight decay)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.01,
                            weight_decay=0.001)  # λ₂ = 0.001`}
  </pre>
</div>

<h3 className="text-xl font-semibold mb-2">โค้ดตัวอย่าง Keras</h3>
<div className="overflow-x-auto mb-4">
  <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono whitespace-pre-wrap w-full max-w-full">
{`# L1
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'],
              kernel_regularizer=regularizers.l1(0.001))

# L2
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'],
              kernel_regularizer=regularizers.l2(0.001))`}
  </pre>
</div>


  <h3 className="text-xl font-semibold mb-2 text-center">ข้อควรระวัง</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ไม่ควรตั้ง λ สูงเกิน เพราะโมเดลจะ underfit ไม่เรียนรู้อะไรเลย</li>
    <li>ค่าที่เหมาะสมขึ้นกับขนาด dataset และ complexity ของโมเดล</li>
    <li>ทดลองปรับ λ ทีละเล็กน้อย แล้วดูผลการเปลี่ยน loss/accuracy curve</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">Insight</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    ระหว่างการเรียนรู้ บางครั้งโมเดลต้องการแค่ “เบรก” เล็กน้อย
    ไม่ให้วิ่งเร็วเกินไปจนหลงทางใน noise ของข้อมูล
    L1 กับ L2 คือคุณสมบัติที่ช่วยตัดหรือชะลอ
    เพื่อให้โมเดลโฟกัสกับ pattern สำคัญจริง ๆ
  </div>
</section>


<section id="dropout" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Dropout</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>


  <p className="mb-4 leading-relaxed">
    ในช่วงการฝึก โมเดลมักจะติดอยู่กับ pattern เดิม ๆ จนเรียนรู้ noise มากเกินไป ทำให้ประสิทธิภาพลดลงเมื่อนำไปใช้กับข้อมูลใหม่
    Dropout ถูกออกแบบมาเหมือนการสุ่มเลือกให้เซลล์ประสาทบางส่วน “หลับ” ในแต่ละรอบการฝึก เพื่อบังคับให้โมเดลไม่พึ่งพาเส้นทางเดียว
    แต่เรียนรู้ representation ที่กระจายตัวมากขึ้น
  </p>

  <p className="mb-4 leading-relaxed">
    เปรียบเสมือนการเรียนกลุ่ม เมื่อมีผู้เรียนบางคนถูกพักการเข้าเรียน จะทำให้แต่ละคนต้องแบ่งบทบาทกันมากขึ้น
    กลุ่มที่แข็งแรงจะยังสอบผ่านแม้ขาดคนไปบ้าง ต่างจากกลุ่มที่พึ่งคนเก่งคนเดียว
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">1. หลักการทำงาน</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ในระหว่าง <strong>training</strong> แต่ละรอบ จุดเชื่อมต่อ (neuron) จะมีโอกาสถูกปิดการใช้งานตามสัดส่วน p</li>
    <li>เมื่อปิด neuron แล้ว input จะไม่ส่งไปยัง neuron นั้น ทำให้ gradient ไม่ย้อนกลับ</li>
    <li>เมื่อไปถึงช่วง <strong>evaluation</strong> ทุก neuron จะเปิดใช้งาน และน้ำหนักจะถูกปรับอัตราส่วนตามค่า p</li>
    <li>วิธีนี้ช่วยลดการ co-adaptation ของ neuron แต่ละตัว ให้เรียนรู้ feature ที่เป็นตัวแทนจริง ๆ</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2 text-center">2. การเลือกค่า p</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <p className="mb-4 leading-relaxed">
    ค่า p คือสัดส่วนการปิดการใช้งาน neuron ยิ่งค่า p สูงเท่าไร การสุ่มก็ยิ่งโหดขึ้น
    แต่ก็อาจทำให้โมเดลเรียนรู้ช้ากว่าเดิม ถ้าตั้ง p ต่ำเกินไป การป้องกัน overfit ก็ไม่ชัดเจนพอ
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>p = 0.1–0.3:</strong> เหมาะกับชั้นแรกหรือเลเยอร์กว้าง</li>
    <li><strong>p = 0.4–0.6:</strong> มักใช้กับชั้นซ่อนลึกหรือ dense layers</li>
    <li><strong>p มากกว่า 0.7:</strong> อาจทำให้โมเดลฝึกไม่ขึ้น เพราะถูก shutdown เยอะเกินไป</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">3. ตัวอย่างโค้ด PyTorch</h3>
  <div className="bg-gray-800 text-yellow-100 text-sm rounded-xl font-mono overflow-x-auto mb-4">
    <pre className="p-4 whitespace-pre">
{`import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">4. ตัวอย่างโค้ด Keras</h3>
  <div className="bg-gray-800 text-yellow-100 text-sm rounded-xl font-mono overflow-x-auto mb-4">
    <pre className="p-4 whitespace-pre">
{`from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2 text-center">5. ผลกระทบต่อการเรียนรู้</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <p className="mb-4 leading-relaxed">
    การเปิดใช้ Dropout จะทำให้ loss บนชุดฝึกขึ้นสูงขึ้นเล็กน้อยในช่วงแรก เนื่องจากโมเดลถูกบังคับให้ไม่พึ่ง neuron เดิม
    แต่พอฝึกไปสักพัก validation loss จะลดลงต่อเนื่องแสดงถึงความ generalization ที่ดีขึ้น
  </p>

  <h3 className="text-xl font-semibold mb-2 text-center">6. เปรียบเทียบ Before / After</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Before:</strong> loss ฝึกต่ำมาก แต่ validation loss สูง → บ่งชี้ overfit ชัดเจน</li>
    <li><strong>After:</strong> loss ฝึกอาจสูงกว่าเดิมเล็กน้อย แต่ validation loss ลดตามกันลง → generalize ดีขึ้น</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2 text-center">7. การใช้งานร่วมกับเทคนิคอื่น</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <p className="mb-4 leading-relaxed">
    Dropout มักใช้ควบคู่กับ L2 regularization หรือ batch normalization เพื่อเสริมประสิทธิภาพกันและกัน
    แต่ต้องระวังไม่ให้เกิดการเบรกหนักเกินไปจนหมดแรงเรียนรู้
  </p>

  <h3 className="text-xl font-semibold mb-2">8. ข้อควรระวัง</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ไม่ใช้ Dropout ใน layer output โดยตรง ถ้าเป็น classification จะทำให้ผลลัพธ์ไม่เสถียร</li>
    <li>ระวังการตั้ง p สูงเกินไป สูญเสีย information จนโมเดลเรียนไม่ได้</li>
    <li>เมื่อนำโมเดลไป predict ให้สลับเป็น .eval() เพื่อปิด Dropout</li>
  </ul>

  <h3 className="text-xl font-semibold mb-2">9. Insight</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    Dropout คือกลไกสำคัญที่ช่วยให้โมเดลไม่จำ pattern ตายตัว แต่เรียนรู้ feature ที่หลากหลาย
    เปรียบเหมือนการสอนกลุ่มให้ทุกคนพร้อมสนับสนุนกัน เมื่อคนหลักหลับไป กลุ่มยังสามารถทำงานได้ดี
  </div>
</section>

<section id="early-stopping" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Early Stopping</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <p className="mb-4 text-base leading-relaxed">
    การหยุดฝึกอัตโนมัติเกิดขึ้นเมื่อโมเดลเรียนรู้ถึงจุดหนึ่งแล้ว performance บน validation set ไม่ดีขึ้นต่อเนื่อง
    เทคนิคนี้เปรียบได้กับการหยุดการฝึกของนักกีฬาเมื่อเวลาในการวิ่งแข่งขันไม่ลดลงเกินกว่าค่า threshold
    เพื่อหลีกเลี่ยงการ over-exertion และรักษาสมรรถภาพเอาไว้
  </p>

  <h3 className="text-lg font-semibold mb-2 text-center">หลักการทำงาน</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>
      เริ่มต้นตั้งเงื่อนไขเช็ค <strong>monitor</strong> (เช่น validation loss หรือ validation accuracy)
    </li>
    <li>
      หากค่าตรงนี้ไม่ดีขึ้นเป็นจำนวนรอบที่กำหนด (<strong>patience</strong>) ระบบจะหยุดการฝึกทันที
    </li>
    <li>
      ป้องกันการฝึกต่อเนื่องที่ไม่ก่อให้เกิดประโยชน์ ลดความเสี่ยง overfitting
    </li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">ตัวอย่างการใช้งานใน Keras</h3>
<div className="overflow-x-auto mb-4">
  <pre className="bg-gray-800 text-yellow-100 text-sm rounded-xl p-4 font-mono whitespace-pre-wrap w-full max-w-full">
{`from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',    # ตัวชี้วัดที่ติดตาม
    min_delta=0.001,        # การปรับปรุงขั้นต่ำที่ยอมรับได้
    patience=5,             # จำนวน epoch ที่จะรอ
    restore_best_weights=True # โหลด weights ที่ดีที่สุดกลับมา
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[es]
)`}
  </pre>
</div>

<h3 className="text-lg font-semibold mb-2">ตัวอย่างการใช้งานใน PyTorch</h3>
<div className="overflow-x-auto mb-4">
  <pre className="bg-gray-800 text-yellow-100 text-sm rounded-xl p-4 font-mono whitespace-pre-wrap w-full max-w-full">
{`# Pseudo-code สำหรับ Early Stopping ใน PyTorch
best_loss = float('inf')
counter = 0
patience = 5

for epoch in range(num_epochs):
    train()  # forward, backward, update
    val_loss = validate()

    if val_loss < best_loss - 0.001:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print('หยุดการฝึกเนื่องจากไม่มีการปรับปรุง')
            break`}
  </pre>
</div>


  <h3 className="text-lg font-semibold mb-2">เมื่อไหร่ควรใช้ Early Stopping</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>
      เมื่อต้องการรักษา model ที่ performance ดีที่สุดบน validation ก่อนที่จะเกิด overfitting
    </li>
    <li>
      ในงานที่ใช้เวลาฝึกนาน และไม่ต้องการฝึกเกินจำเป็น เพื่อประหยัดทรัพยากร
    </li>
    <li>
      เมื่อต้องการควบคุมกระบวนการ training โดยอัตโนมัติ ไม่ต้องเฝ้าดู manual
    </li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">ข้อควรระวัง</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>
      การตั้ง patience น้อยเกินไปอาจหยุดฝึกก่อนโมเดลเรียนรู้เต็มที่
    </li>
    <li>
      min_delta ที่ตั้งให้ใหญ่เกินไปอาจมองข้ามการปรับปรุงเล็กน้อยที่สำคัญ
    </li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">ผลลัพธ์ที่คาดหวัง</h3>
  <p className="mb-4 text-base leading-relaxed">
    หลังจากฝึกเสร็จ โมเดลจะหยุดที่จุดที่ให้ validation loss ต่ำสุดและ weights ไม่ถูก overfit
    ส่งผลให้สามารถนำโมเดลนี้ไปใช้งานจริงด้วย performance ที่น่าเชื่อถือ
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    Early Stopping เปรียบเสมือนสัญญาณไฟจราจรที่เตือนให้หยุดรถก่อนทางโค้งอันตราย ช่วยหยุดโมเดลก่อนที่จะลื่นไถลเข้าสู่ overfitting
  </div>
</section>


<section id="data-augmentation" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Data Augmentation</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img15} />
  </div>
 
  <p className="mb-4 text-base leading-relaxed">
    การเพิ่มข้อมูล (Data Augmentation) คือการสร้างตัวอย่างใหม่จากข้อมูลต้นฉบับ โดยใช้เทคนิคต่างๆ
    เพื่อหลบเลี่ยงปัญหา overfitting และทำให้โมเดลเรียนรู้ pattern ได้หลากหลายยิ่งขึ้น เหมือนการฝึกฝน
    นักกีฬาในสนามรบแบบต่างๆ ไม่ใช่ซ้อมในสนามเดิมซ้ำๆ เท่านั้น
  </p>

  <h3 className="text-lg font-semibold mb-2">1. ข้อดีของ Data Augmentation</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>สร้างความหลากหลายในชุดข้อมูล โดยไม่ต้องหาข้อมูลเพิ่มเติม</li>
    <li>ช่วยลด overfitting เพราะไม่ให้โมเดลจำข้อมูลดิบแท้ๆ เพียงชุดเล็กๆ</li>
    <li>เพิ่ม robustness ให้โมเดลรับมือกับข้อมูลจริงที่มี noise หรือความผิดเพี้ยน</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">2. เทคนิคสำหรับภาพ</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Flip (พลิกภาพ):</strong> พลิกแนวนอนหรือแนวตั้ง เปรียบเหมือนสลับซ้ายกับขวา ทำให้เห็นมุมมองใหม่</li>
    <li><strong>Rotation (หมุน):</strong> หมุนภาพไม่กี่องศา เพิ่มมุมมอง slight tilt เหมือนหมุนกล้อง</li>
    <li><strong>Crop (ครอป):</strong> ตัดเศษของภาพออก แสดงให้ชัดบางส่วน ยกตัวอย่างการโฟกัสกับรายละเอียด</li>
    <li><strong>Color Jitter (ปรับสี):</strong> เปลี่ยนค่าความสว่าง, คอนทราสต์, ความอิ่มตัว เหมือนถ่ายภาพในสภาพแสงต่างกัน</li>
    <li><strong>Gaussian Noise (ใส่ noise):</strong> เติมเม็ด noise เล็กน้อย เหมือนเวลาถ่ายภาพกล้องเก่า</li>
    <li><strong>Random Erasing (ลบพื้นที่):</strong> ลบส่วนนึงของภาพ เหมือนมีสิ่งกีดขวางบังบางส่วน ให้โมเดลเรียนรู้จาก incomplete view</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">3. เทคนิคสำหรับเสียง</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Time Shift (เลื่อนเวลา):</strong> เลื่อนตำแหน่งเสียงไปข้างหน้า/หลัง เหมือนสัญญาณ delay เล็กน้อย</li>
    <li><strong>Pitch Shift (ปรับความถี่):</strong> เปลี่ยนโทนเสียงสูง/ต่ำ เหมือนพูดในโทนต่างกัน</li>
    <li><strong>Add Noise (ใส่เสียงรบกวน):</strong> ใส่ noise พื้นหลัง เช่น เสียงลม รถ ให้โมเดลเรียนรู้เสียงจริง</li>
    <li><strong>Time Stretch (เปลี่ยนความเร็ว):</strong> ยืดหรือเร่งความเร็ว เหมือนพูดช้าหรือเร็วขึ้นเล็กน้อย</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">4. เทคนิคสำหรับข้อความ</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Synonym Replacement (แทนด้วยคำเหมือน):</strong> แทนคำบางคำด้วยคำที่มีความหมายใกล้เคียง เหมือน paraphrase</li>
    <li><strong>Random Insertion (เพิ่มคำ):</strong> แทรกคำที่เกี่ยวข้องลงไป ลดการจำ pattern ตายตัว</li>
    <li><strong>Random Deletion (ลบคำ):</strong> ลบคำบางคำ เหมือนข้อความขาดหาย ให้โมเดลเรียนรู้ความหมายจากส่วนน้อยลง</li>
    <li><strong>Back Translation (แปลกลับ):</strong> แปลประโยคไป-กลับ ระหว่างภาษาสองภาษาขึ้นไป เหมือนจับข้อความวนซ้ำหลายครั้ง</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">5. การปรับใช้กับโมเดล</h3>
  <p className="mb-4 text-base leading-relaxed">
    บน PyTorch ใช้ torchvision transforms หรือ torchaudio transforms<br/>
    บน Keras ใช้ ImageDataGenerator, TextVectorization, tf.data API<br/>
    ส่วน library อื่นๆ เช่น albumentations, nlpaug ก็ช่วยให้ปรับแต่งง่ายขึ้น
  </p>

  <h3 className="text-lg font-semibold mb-2">6. เมื่อไหร่ควรใช้ Data Augmentation</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>มีข้อมูลดิบจำกัด ต้องการสร้างความหลากหลายเพิ่มเติม</li>
    <li>ต้องการให้โมเดลทนต่อ noise และเงื่อนไขจริงในโลกภายนอก</li>
    <li>ต้องการเพิ่ม performance โดยไม่ต้องเก็บข้อมูลใหม่มากขึ้น</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">7. ข้อควรระวัง</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>Augmentation มากเกินไปอาจทำให้ข้อมูลเบี่ยงเบนจากความเป็นจริง</li>
    <li>ไม่ควรใช้ transformation ที่เปลี่ยนความหมายจนผิดเพี้ยน</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2 text-center">ผลลัพธ์ที่คาดหวัง</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img16} />
  </div>
  <p className="mb-4 text-base leading-relaxed">
    หลังใช้ augmentation แล้ว โมเดลจะ generalize ดีขึ้น รับมือข้อมูลที่ไม่เคยเห็นได้หลากหลายกว่า
    loss จะลดลงต่อเนื่องบน validation set และ performance บน test set ดีขึ้น
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br/>
    Data Augmentation คือการฝึกซ้อมในสนามจำลองหลายแบบ จึงพร้อมสำหรับสนามจริงที่ไม่เคยเจอมาก่อน
  </div>
</section>
<section id="batch-normalization" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4 text-center">Batch Normalization (BN)</h2>
  <div className="flex justify-center items-center my-6">
    <AdvancedImage cldImg={img17} />
  </div>

  <p className="mb-4 text-base leading-relaxed">
    Batch Normalization คือเทคนิคที่ใช้ในการปรับ distribution ของ activation ในแต่ละชั้น
    ให้มี mean ใกล้ 0 และ variance ใกล้ 1 ก่อนจะส่งต่อไปยังเลเยอร์ถัดไป
    ทำให้ gradient flow มีความเสถียรขึ้น และช่วยให้กระบวนการฝึกเรียนรู้รวดเร็วขึ้น
  </p>

  <h3 className="text-lg font-semibold mb-2">1. ที่มาของ Internal Covariate Shift</h3>
  <p className="mb-4 text-base leading-relaxed">
    ก่อนมี Batch Normalization โมเดลมักเจอปัญหา Internal Covariate Shift
    คือ distribution ของ input ในแต่ละชั้นเปลี่ยนแปลงตลอดระหว่างการฝึก
    เปรียบเหมือนการวิ่งในสนามที่พื้นผิวเปลี่ยนไปทุกจังหวะ ทำให้ต้องปรับตัวใหม่เสมอ
  </p>

  <h3 className="text-lg font-semibold mb-2">2. หลักการทำงาน</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>คำนวณ mean และ variance ของ activation ใน batch ปัจจุบัน</li>
    <li>ทำ normalization: หัก mean และหารด้วย standard deviation</li>
    <li>ใช้ parameter gamma และ beta เพื่อ scale และ shift output ตามต้องการ</li>
    <li>gamma, beta จะเป็นพารามิเตอร์ที่เรียนรู้ได้ เช่นเดียวกับ weight ของเลเยอร์</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">3. ประโยชน์ที่ได้</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ช่วยลด Internal Covariate Shift ทำให้โมเดลเรียนรู้ได้เสถียรขึ้น</li>
    <li>เร่งกระบวนการ convergence ของการฝึก ลดจำนวน epoch ที่ต้องใช้</li>
    <li>ทำหน้าที่เป็น regularizer เบื้องต้น ลด overfitting ได้บ้าง</li>
    <li>ช่วยให้ใช้ learning rate สูงขึ้นได้ โดยไม่เกิน limit</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">4. การใช้งานใน PyTorch</h3>
<div className="overflow-x-auto mb-4">
  <pre className="bg-gray-800 text-yellow-100 text-sm rounded-xl p-4 font-mono whitespace-pre-wrap w-full max-w-full">
{`# PyTorch: ใช้ nn.BatchNorm2d กับข้อมูลภาพ
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)

# สำหรับข้อมูลแบบ vector ใช้ nn.BatchNorm1d
model_vec = nn.Sequential(
    nn.Linear(100, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
)
`}
  </pre>
</div>

<h3 className="text-lg font-semibold mb-2">5. การใช้งานใน Keras</h3>
<div className="overflow-x-auto mb-4">
  <pre className="bg-gray-800 text-yellow-100 text-sm rounded-xl p-4 font-mono whitespace-pre-wrap w-full max-w-full">
{`# Keras: ใช้ tf.keras.layers.BatchNormalization
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(64, 3, padding='same', input_shape=(224,224,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
])
`}
  </pre>
</div>


  <h3 className="text-lg font-semibold mb-2">6. การทำงานระหว่าง Training & Inference</h3>
  <p className="mb-4 text-base leading-relaxed">
    - ในช่วง training จะคำนวณ mean/variance ของ batch จริง  
    - ในช่วง inference จะใช้ running mean/variance ที่คำนวณสะสมระหว่างฝึก  
    ทำให้ output มีความเสถียร ไม่ขึ้นกับ batch input ใหม่
  </p>

  <h3 className="text-lg font-semibold mb-2">7. ข้อควรระวัง</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>batch size เล็กเกินไป ทำให้ estimation mean/variance ผิดเพี้ยน</li>
    <li>ควรใช้ batch size พอเหมาะ เช่น 32 ขึ้นไป เพื่อค่าที่แม่นยำ</li>
    <li>บางกรณีอาจใช้ Layer Normalization แทนสำหรับ NLP หรือข้อมูล sequence</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">8. เมื่อไหร่ควรใช้ BatchNorm</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>เมื่อต้องการเร่ง convergence ของโมเดลลึกหลายชั้น</li>
    <li>เมื่อต้องการลดจำนวน epoch เพื่อประหยัดเวลา</li>
    <li>เมื่อต้องการลด overfitting เบื้องต้นในโมเดล convolutional</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2 text-center"> ผลลัพธ์ที่คาดหวัง</h3>
  <div className="flex justify-center items-center my-6">
    <AdvancedImage cldImg={img18} />
  </div>
  <p className="mb-4 text-base leading-relaxed">
    หลังเพิ่ม BatchNorm โมเดลจะเรียนรู้เร็วขึ้น loss curve จะลดลงชัดเจน
    และ accuracy บน validation set จะสูงขึ้นในช่วงแรกของการฝึก
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br/>
    BatchNorm คือเปลี่ยนสนามฝึกให้เรียบเหมือนสนามแข่งมาตรฐาน ช่วยให้โมเดลวิ่งได้เร็วไม่สะดุด
  </div>
</section>


<section id="visualization" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">Visualization</h2>

  <div className="flex justify-center mb-6">
  </div>

  <p className="mb-4 leading-relaxed">
    สังเกตภาพแรก จะเห็นเส้นกราฟของโมเดลที่ Overfitting ลดลงบนข้อมูลฝึกซ้ำซาก
    แต่เมื่อใช้ข้อมูลใหม่ เส้นกราฟของ validation loss จะพุ่งขึ้นทันที แสดงถึงการจับ noise เกินไป
  </p>
  <p className="mb-4 leading-relaxed">
    เมื่อโมเดล Underfitting เส้นกราฟทั้ง train และ validation loss จะลดลงช้า
    หรือแทบไม่ลดเลย สะท้อนว่าโมเดลยังเรียนรู้ pattern ไม่พอ
  </p>

  <div className="flex justify-center mb-6">
   
  </div>

  <p className="mb-4 leading-relaxed">
    กราฟ Learning Curves จะชัดเจนว่า training loss ลดลงจนเกือบติดดิน
    ส่วน validation loss ลดตามแล้วถึงจุด equilibrium ก่อนที่จะเริ่มไต่สูงขึ้น
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ช่วงแรก train และ validation loss ลดพร้อมกันอย่างสม่ำเสมอ</li>
    <li>เมื่อทั้งสองเส้นแยกกัน validation loss เริ่มวิ่งขึ้น แสดงถึง Overfitting</li>
    <li>train loss อาจลดต่อได้ แต่ validation loss จะไม่ลดตาม</li>
    <li>ควรหยุดฝึกก่อน validation loss พุ่งสูง เพื่อรักษาความ generalization</li>
  </ul>

  <h3 className="text-lg font-semibold mb-3">การปรับแต่ง Visualization</h3>
  <p className="mb-4 leading-relaxed">
    ใช้สีและ marker แยกระหว่าง train กับ validation ให้ชัด
    เพิ่มเส้นแนวตั้ง (vertical line) เพื่อบอกจุด early stopping
  </p>

  <div className="overflow-x-auto mb-6">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm whitespace-pre-wrap">
{`import matplotlib.pyplot as plt

epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.axvline(x=best_epoch, color='green', linestyle='--', label='Stop Point')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
`}
    </pre>
  </div>

  <p className="mb-4 leading-relaxed">
    เส้น Stop Point จะช่วยให้เห็นชัดว่าควรยุติการฝึกเมื่อใด
    เพื่อให้ loss บน validation ไม่สูงเกินไป
  </p>

  <h3 className="text-lg font-semibold mb-3">Interactive Dashboard</h3>
  <p className="mb-4 leading-relaxed">
    ในระบบที่ใช้ TensorBoard หรือ WandB สามารถดูกราฟแบบ Real-time
    วิเคราะห์ batch แรกหรือ batch สุดท้ายได้ทันที
  </p>

  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ดู trend ของ loss และ metric อื่นๆ เช่น accuracy</li>
    <li>ตรวจ batch-level performance โดยไม่ต้องรอ epoch จบ</li>
    <li>ใช้ zoom-in เพื่อโฟกัสช่วง epoch ที่สนใจ</li>
  </ul>

  <p className="mb-4 leading-relaxed">
    การดูกราฟแบบเรียลไทม์ลดเวลา debug ข้อผิดพลาดของ training loop
    เพราะสามารถเห็น pattern ผิดปกติได้ทันที
  </p>

  <h3 className="text-lg font-semibold mb-3">ประโยชน์หลักของ Visualization</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ตรวจสอบ Overfitting และ Underfitting ได้ชัดเจน</li>
    <li>ช่วยตัดสินใจปรับ hyperparameters ทันที</li>
    <li>เปิดมุมมองใหม่ในการวางแผน training strategy</li>
  </ul>

  <p className="mb-4 leading-relaxed">
    การเห็นภาพมากกว่าตัวเลข จะทำให้เข้าใจพฤติกรรมของโมเดลได้ดีกว่า
    และปรับปรุงได้ตรงจุดยิ่งขึ้น
  </p>

  <h3 className="text-lg font-semibold mb-3">การฝังกราฟใน React</h3>
  <p className="mb-4 leading-relaxed">
    สามารถใช้ Chart.js ร่วมกับ React เพื่อแสดงกราฟ responsive
    โดยไม่ต้องเขียน CSS เยอะ
  </p>

  <div className="overflow-x-auto mb-6">
    <pre className="bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm whitespace-pre-wrap">
{`import { Line } from 'react-chartjs-2';

const data = {
  labels: epochs,
  datasets: [
    { label: 'Train Loss', data: train_loss },
    { label: 'Val Loss', data: val_loss },
  ],
};

<Line data={data} options={{ responsive: true }} />
`}
    </pre>
  </div>

  <p className="mb-4 leading-relaxed">
    ด้วย options ที่เหมาะสม กราฟจะปรับขนาดตาม container
    ทำให้แสดงผลดีทั้งบนมือถือและเดสก์ท็อป
  </p>

  <h3 className="text-lg font-semibold mb-3">สรุป Visualization</h3>
  <p className="mb-4 leading-relaxed">
    การแสดงข้อมูลเป็นภาพช่วยให้เข้าใจลึกขึ้น
    และเป็นเครื่องมือสำคัญในการปรับปรุงโมเดลอย่างมีประสิทธิภาพ
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br />
    Visualization คือสะพานเชื่อมระหว่างข้อมูลและการตัดสินใจ
    ช่วยให้เกิดโมเดลที่เรียนรู้ได้เร็ว แม่นยำ และใช้งานได้จริง
  </div>
</section>



        {/* 9. Combined Code */}
        <section id="combined-code" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-4">ตัวอย่างโค้ดรวม</h2>
  <p className="mb-4 leading-relaxed">
    โค้ดตัวอย่างนี้ประกอบด้วยการใช้ Dropout ร่วมกับ L2 regularization (weight_decay)
    และ EarlyStopping ในขั้นตอนการฝึกโมเดลเดียวกัน เพื่อป้องกัน Overfitting
  </p>
  <p className="mb-4 leading-relaxed">
    เริ่มจากการกำหนดโมเดลที่มี Dropout layer ตามด้วยตัวเลือก L2 ใน Optimizer
    แล้วติดตั้ง EarlyStopping callback เพื่อหยุดการฝึกอัตโนมัติเมื่อไม่มีการปรับปรุง
  </p>

  <h3 className="text-xl font-semibold mb-2">1. การกำหนดโมเดลและ Regularization</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>สร้างคลาสโมเดลที่มีเลเยอร์ Linear ตามด้วย Dropout เพื่อสุ่มปิด neuron บางส่วน</li>
    <li>โครงสร้างเน้นลดการพึ่งพา feature เดียวและกระจายการเรียนรู้ไปทั่วในแต่ละรอบ</li>
    <li>การกำหนด weight_decay จะทำใน Optimizer ไม่ได้อยู่ในโมเดลตรงๆ</li>
  </ul>
  <div className="overflow-x-auto bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono mb-4">
    <pre>{`import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedModel(nn.Module):
    def __init__(self):
        super(RegularizedModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x`}
    </pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">2. การกำหนด Optimizer พร้อม L2 Regularization</h3>
  <p className="mb-4 leading-relaxed">
    เมื่อโมเดลพร้อมแล้ว จะใช้ Optimizer แบบ Adam พร้อมตั้ง weight_decay 
    เพื่อบังคับให้พารามิเตอร์ไม่โตเกินไป ช่วยลด overfitting ในระยะยาว
  </p>
  <div className="overflow-x-auto bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono mb-4">
    <pre>{`model = RegularizedModel().to(device)
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 regularization term
)`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">3. การตั้ง EarlyStopping</h3>
  <p className="mb-4 leading-relaxed">
    ใช้ PyTorch Lightning หรือเขียน callback เองเพื่อหยุดการฝึกเมื่อ validation loss
    ไม่มีการปรับปรุงเกิน patience ที่กำหนด
  </p>
  <div className="overflow-x-auto bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono mb-4">
    <pre>{`from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">4. การรัน Training Loop พร้อม Callback</h3>
  <p className="mb-4 leading-relaxed">
    ผสมผสานทุกส่วนเข้ากับ Trainer ของ PyTorch Lightning โดยส่ง callback เข้าไปในตัว Trainer
    เพื่อให้หยุดฝึกอัตโนมัติเมื่อถึงเงื่อนไขที่เหมาะสม
  </p>
  <div className="overflow-x-auto bg-gray-800 text-yellow-100 p-4 rounded-xl text-sm font-mono mb-4">
    <pre>{`import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[early_stopping],
    gpus=1
)

trainer.fit(model, train_dataloader, val_dataloader)`}</pre>
  </div>

  <h3 className="text-xl font-semibold mb-2">5. ประยุกต์ใช้ร่วมกัน</h3>
  <p className="mb-4 leading-relaxed">
    เมื่อมี Dropout, L2 regularization และ EarlyStopping รวมกันในโมเดลเดียว
    ผลลัพธ์ที่ได้คือโมเดลเรียนรู้ได้ลึกขึ้นโดยไม่ overfit และหยุดฝึกเมื่อถึงจุดเหมาะสม
  </p>
  <p className="mb-4 leading-relaxed">
    L2 ช่วยจำกัดขนาดน้ำหนัก, Dropout กระจายการเรียนรู้, EarlyStopping ป้องกันฝึกเกินจำเป็น
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <strong>Insight:</strong><br/>
    การรวมเทคนิคเหล่านี้เหมือนการติดตั้งเบรกและโช้กอัพให้รถวิ่งได้เร็วอย่างมั่นคง
  </div>
</section>
<section id="insight" className="mb-16 scroll-mt-32">
  <div className="flex items-center mb-4">
    <svg
      className="w-6 h-6 text-yellow-400 mr-2 flex-shrink-0"
      fill="currentColor"
      viewBox="0 0 20 20"
    >
      <path d="M11 3a1 1 0 10-2 0 5 5 0 00-3.546 8.546A4 4 0 017 15v1a1 1 0 102 0v-1a2 2 0 01.293-1.004A5 5 0 0011 3zM9 17a1 1 0 112 0h-2z" />
    </svg>
    <h2 className="text-2xl font-semibold">Insight</h2>
  </div>

  <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-xl border-l-8 border-yellow-400">
    <p className="mb-4 text-base leading-relaxed">
      โลกของ Machine Learning ต้องบาลานซ์ระหว่างการเรียนรู้กับการรักษาความแม่นยำ  
      Regularization ทำหน้าที่เป็นแรงชะลอ ไม่ให้โมเดลจดจำ noise จนไปโฟกัสที่รายละเอียดเล็กน้อย  
      แต่ช่วยให้จับ pattern หลักในข้อมูลจริงได้แน่นยิ่งขึ้น
    </p>

    <ul className="list-disc pl-6 mb-4 space-y-2 text-base">
      <li>
        เสมือนการติดโช้กอัพให้รถ ให้วิ่งเร็วได้แต่ไม่กระเด้งจนหลุดการควบคุม
      </li>
      <li>
        ลดการเรียนรู้ noise ทำให้โมเดลโฟกัสกับ pattern แท้จริงที่ซ่อนอยู่ในข้อมูล
      </li>
      <li>
        Generalization คือมารยาทสำคัญของโมเดล ที่ต้องทายผลข้อมูลใหม่ได้อย่างแม่นยำ
      </li>
    </ul>

    <p className="mb-4 text-base leading-relaxed">
      แต่ละเทคนิคมีข้อดีข้อจำกัดต่างกัน เช่น L1 จะตัดบางพารามิเตอร์ให้เป็นศูนย์ เหมาะกับการลด feature  
      L2 จะลดทุกพารามิเตอร์ลงนิดหน่อย ไม่ให้ใครตกกระป๋องง่ายเกินไป  
      Dropout สุ่มปิด neuron บางส่วน เพื่อกระจายการเรียนรู้  
      Early Stopping คอยบอกเมื่อถึงจุดหยุดเหมาะสม  
      BatchNorm ช่วยเรียบสนามฝึกให้วิ่งได้ลื่น  
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-lg border border-yellow-300 mt-4">
      <p className="font-semibold">
        “Regularization คือเบรกชั้นดี ให้โมเดลเรียนรู้ pattern สำคัญ  
        และ Generalization คือรางวัลเมื่อโมเดลเข้าใจข้อมูลจริง ไม่ใช่จำฝึกเพียงอย่างเดียว.”
      </p>
    </div>
  </div>
</section>


        {/* Quiz */}
        <section id="quiz" className="mb-16 scroll-mt-32">
          <MiniQuiz_Day9 theme={theme} />
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
        <ScrollSpy_Ai_Day9 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day9_Regularization;
