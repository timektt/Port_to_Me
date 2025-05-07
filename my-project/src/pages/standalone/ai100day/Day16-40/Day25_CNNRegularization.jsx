import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day25 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day25";
import MiniQuiz_Day25 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day25";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day25_CNNRegularization = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("CNNRegularization1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("CNNRegularization2").format("auto").quality("auto").resize(scale().width(590));
  const img3 = cld.image("CNNRegularization3").format("auto").quality("auto").resize(scale().width(590));
  const img4 = cld.image("CNNRegularization4").format("auto").quality("auto").resize(scale().width(590));
  const img5 = cld.image("CNNRegularization5").format("auto").quality("auto").resize(scale().width(580));
  const img6 = cld.image("CNNRegularization6").format("auto").quality("auto").resize(scale().width(580));
  const img7 = cld.image("CNNRegularization7").format("auto").quality("auto").resize(scale().width(450));
  const img8 = cld.image("CNNRegularization8").format("auto").quality("auto").resize(scale().width(580));
  const img9 = cld.image("CNNRegularization9").format("auto").quality("auto").resize(scale().width(590));
  const img10 = cld.image("CNNRegularization10").format("auto").quality("auto").resize(scale().width(590));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 25: Regularization Techniques in CNNs</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          <section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำสู่เทคนิค Regularization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การฝึกโมเดล Deep Learning ที่มีจำนวนพารามิเตอร์มากมักนำไปสู่ปัญหา <strong>Overfitting</strong> ซึ่งหมายถึงสถานการณ์ที่โมเดลเรียนรู้รายละเอียดเฉพาะของชุดข้อมูลฝึกจนสูญเสียความสามารถในการทำนายกับข้อมูลใหม่
    </p>

    <p>
      Regularization คือกระบวนการควบคุมไม่ให้โมเดลซับซ้อนเกินไป โดยยังคงประสิทธิภาพในการเรียนรู้จากข้อมูลไว้ให้มากที่สุด เป็นองค์ประกอบสำคัญที่ช่วยให้โมเดลสามารถ <strong>generalize</strong> ไปยังข้อมูลที่ไม่เคยเห็นมาก่อนได้อย่างมีประสิทธิภาพ
    </p>

    <p>
      Stanford CS231n อธิบายว่า Regularization คือการแทรกเงื่อนไขพิเศษเข้าไปในกระบวนการเรียนรู้ของโมเดล เช่น การจำกัดค่าน้ำหนักหรือเพิ่ม noise เพื่อไม่ให้โมเดลจำข้อมูลฝึกอย่างเจาะจงเกินไป
    </p>

    <h3 className="text-xl font-semibold">เป้าหมายหลักของ Regularization</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ลด Overfitting โดยไม่ลดความสามารถของโมเดลในชุดข้อมูลฝึก</li>
      <li>เพิ่มความสามารถในการ generalize ไปยังข้อมูลใหม่</li>
      <li>ส่งเสริมความมั่นคงในการฝึกแม้มี noise หรือข้อมูลขาดหาย</li>
      <li>ป้องกันการเรียนรู้ pattern ที่ไม่เกี่ยวข้องกับการทำนายจริง</li>
    </ul>

    <h3 className="text-xl font-semibold">ความเข้าใจเชิงทฤษฎีจาก MIT 6.S191</h3>
    <p>
      ในหลักสูตร MIT Deep Learning (6.S191) อธิบายว่า Overfitting มักเกิดขึ้นเมื่อพารามิเตอร์ในโมเดลมากกว่าขนาดข้อมูลหรือเมื่อใช้ Epoch มากเกินไป
      Regularization ช่วยควบคุม <em>capacity</em> ของโมเดลให้อยู่ในระดับที่เหมาะสม โดยเฉพาะอย่างยิ่งเมื่อข้อมูลมี noise หรือ class imbalance
    </p>

    <h3 className="text-xl font-semibold">รูปแบบของ Regularization ที่พบมากที่สุด</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>L1/L2 Weight Penalty:</strong> เพิ่มค่าบทลงโทษ (penalty) ใน loss function เพื่อจำกัดขนาดของพารามิเตอร์
      </li>
      <li>
        <strong>Dropout:</strong> ปิดบาง neuron แบบสุ่มระหว่างการฝึก ช่วยลดการพึ่งพา neuron ใด neuron หนึ่งมากเกินไป
      </li>
      <li>
        <strong>Data Augmentation:</strong> สร้างข้อมูลเพิ่มโดยการแปลงภาพ เช่น การหมุน, ขยาย, ตัดขอบ, เพิ่ม noise
      </li>
      <li>
        <strong>Early Stopping:</strong> หยุดการฝึกเมื่อ performance บน validation ไม่ดีขึ้นอีก แม้ training จะดีขึ้นต่อไป
      </li>
    </ul>

    <h3 className="text-xl font-semibold text-center">กราฟแสดงผลของ Overfitting</h3>
    <div className="flex justify-center my-6">
      <AdvancedImage cldImg={img3} />
    </div>
    <p>
      จากภาพด้านบน: โมเดลที่ overfit จะแสดงค่า loss บน training set ลดลงต่อเนื่อง แต่ validation loss จะเริ่มเพิ่มขึ้นเมื่อฝึกต่อไปเรื่อย ๆ จุดนี้แสดงถึงช่วงที่โมเดลเริ่มจำข้อมูลฝึกมากเกินไป
    </p>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Regularization คือกุญแจสำคัญในการสร้างโมเดลที่แข็งแกร่งในโลกจริง</li>
        <li>โมเดลที่ไม่มี Regularization มักจดจำ noise และรายละเอียดที่ไม่จำเป็น</li>
        <li>การผสมผสาน Regularization หลายประเภทให้ผลดีที่สุด เช่น L2 + Dropout</li>
        <li>DeepMind และ OpenAI ใช้ Regularization เพื่อควบคุมโมเดลขนาดใหญ่ เช่น GPT และ AlphaFold</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-6 text-sm space-y-2">
      <li>Ng, Andrew. "CS229: Machine Learning." Stanford University. (L2 Regularization)</li>
      <li>Goodfellow, Bengio, Courville. "Deep Learning." MIT Press. (Chapter 7: Regularization)</li>
      <li>MIT 6.S191: "Deep Learning for Self-Driving Cars." (Lecture on Generalization)</li>
      <li>Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR, 2014</li>
      <li>Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning." J Big Data, 2019</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      Regularization เป็นกลไกที่จำเป็นในการออกแบบโมเดล Deep Learning ที่นำไปใช้งานจริงได้
      การทำความเข้าใจกลยุทธ์ที่เหมาะสมและการเลือกเทคนิคให้สอดคล้องกับลักษณะของข้อมูล จะเป็นรากฐานที่นำไปสู่การสร้างโมเดลที่มีเสถียรภาพ มีความยืดหยุ่น และสามารถ generalize ได้อย่างแท้จริง
    </p>
  </div>
</section>


<section id="overfitting" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ปัญหา Overfitting และแนวทางลดความซับซ้อนของโมเดล</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      ปัญหา Overfitting เกิดขึ้นเมื่อโมเดลสามารถเรียนรู้ข้อมูลฝึกได้ดีเกินไป จนจำรายละเอียดเฉพาะหรือ noise ของข้อมูลนั้นอย่างผิดธรรมชาติ ส่งผลให้ความสามารถในการ generalize ไปยังข้อมูลใหม่ลดลงอย่างมีนัยสำคัญ
    </p>

    <p>
      งานวิจัยของ Goodfellow, Bengio และ Courville อธิบายว่า Overfitting มีความสัมพันธ์โดยตรงกับความซับซ้อนของโมเดล โดยเฉพาะจำนวนพารามิเตอร์, โครงสร้างที่ลึก หรือจำนวน epoch ที่มากเกินไป
    </p>

    <h3 className="text-xl font-semibold">สัญญาณที่บ่งชี้ว่าโมเดล Overfit</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Training loss ลดลงอย่างต่อเนื่อง ขณะที่ validation loss เพิ่มขึ้น</li>
      <li>Accuracy บน training สูงมาก แต่ต่ำบน validation หรือ test set</li>
      <li>โมเดลตอบสนองแม้กับความผิดปกติเล็กน้อยในข้อมูล</li>
      <li>โมเดลให้ผลลัพธ์ไม่เสถียรเมื่อตรวจสอบด้วย Cross Validation</li>
    </ul>

    <h3 className="text-xl font-semibold text-center">ตัวอย่าง Visualization จาก Stanford CS231n</h3>
    <div className="flex justify-center my-6">
      <AdvancedImage cldImg={img5} />
    </div>
    <p>
      กราฟด้านบนแสดงให้เห็นว่า เมื่อจำนวน epoch มากขึ้น Training loss ลดลงเรื่อย ๆ แต่ Validation loss เริ่มเพิ่มขึ้นหลังจากจุดหนึ่งซึ่งเป็นจุดเริ่มของ Overfitting
    </p>

    <h3 className="text-xl font-semibold">สาเหตุหลักที่นำไปสู่ Overfitting</h3>
    <ul className="list-decimal list-inside ml-6 space-y-2">
      <li>ขนาดข้อมูลฝึกมีจำนวนน้อย หรือกระจายไม่สมดุล</li>
      <li>โมเดลมีพารามิเตอร์จำนวนมากเกินความจำเป็น</li>
      <li>การฝึกโมเดลเป็นเวลานานโดยไม่มี Early Stopping</li>
      <li>ไม่มีการ regularize หรือใช้ technique ลด variance</li>
    </ul>

    <h3 className="text-xl font-semibold">แนวทางลดความซับซ้อนเพื่อป้องกัน Overfitting</h3>
    <p>
      การควบคุม capacity ของโมเดลมีความสำคัญสูงในการป้องกัน Overfitting นักวิจัยจาก MIT 6.S191 แนะนำกลยุทธ์การลดความซับซ้อนที่สามารถใช้งานได้จริง เช่น:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ลดจำนวน layer หรือจำนวน neuron ในแต่ละ layer</li>
      <li>ปรับ dropout ให้เหมาะสม (เช่น 0.2–0.5)</li>
      <li>ใช้ regularization เช่น L2 เพื่อลดขนาด weight</li>
      <li>ใช้ Batch Normalization เพื่อควบคุม distribution</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Overfitting เปรียบเสมือนนักเรียนที่ท่องจำคำตอบโดยไม่เข้าใจโจทย์</li>
        <li>โมเดลที่ generalize ได้ดีควรเรียนรู้ pattern โดยไม่จดจำเฉพาะข้อมูลฝึก</li>
        <li>ความซับซ้อนที่ไม่เหมาะสมสร้างภาระ computation และเสี่ยงต่อ overfit</li>
        <li>โมเดลที่เรียบง่ายแต่วิเคราะห์ข้อมูลได้แม่นยำมีค่ามากกว่าความลึกเพียงอย่างเดียว</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">กรณีศึกษา: การใช้ Dropout เพื่อลด Overfitting</h3>
    <pre><code className="language-python">
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)
    </code></pre>

    <p>
      จากตัวอย่างจะเห็นว่า การใช้ Dropout ที่ความน่าจะเป็น 30% ช่วยลดการ over-reliance ต่อ neuron เฉพาะ ทำให้โมเดลเรียนรู้ pattern ที่ general มากขึ้น
    </p>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      การเข้าใจและจัดการกับปัญหา Overfitting คือทักษะสำคัญของการพัฒนาโมเดล Machine Learning ที่นำไปใช้ได้จริง การปรับความซับซ้อนของโมเดลให้เหมาะสมควบคู่กับการ regularize อย่างถูกต้อง จะเป็นกุญแจสำคัญในการเพิ่มประสิทธิภาพการทำนายของโมเดลบนข้อมูลใหม่
    </p>
  </div>
</section>

<section id="weight-regularization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. เทคนิค Weight Regularization (L1 / L2)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Weight Regularization เป็นเทคนิคที่มีประสิทธิภาพในการควบคุมความซับซ้อนของโมเดล โดยการเพิ่มบทลงโทษ (penalty) ต่อค่าน้ำหนักที่มีค่ามากเกินไป ซึ่งช่วยลดปัญหา Overfitting และปรับสมดุลระหว่างความแม่นยำบน Training Set และ Generalization บน Test Set
    </p>

    <h3 className="text-xl font-semibold">พื้นฐานแนวคิดของ Weight Regularization</h3>
    <p>
      แนวคิดเบื้องหลัง Weight Regularization คือการเพิ่มค่า penalty term เข้ากับ loss function ของโมเดล เพื่อส่งเสริมให้โมเดลเรียนรู้ค่า weight ที่มีขนาดเล็กและสม่ำเสมอ แทนที่จะปล่อยให้ weight เติบโตจนขาดการควบคุม
    </p>

    <pre><code className="language-python">
# รูปแบบทั่วไปของ Loss พร้อม Regularization
Loss_total = Loss_data + λ * Regularization_term
    </code></pre>

    <p>
      โดยที่:
      <ul className="list-disc list-inside ml-6 space-y-2">
        <li><code>Loss_data</code>: ค่า loss จาก prediction เช่น CrossEntropy หรือ MSE</li>
        <li><code>λ</code> (lambda): ค่าคงที่ที่ควบคุมความรุนแรงของ regularization</li>
        <li><code>Regularization_term</code>: Term ที่ใช้สำหรับลงโทษ weight</li>
      </ul>
    </p>

    <h3 className="text-xl font-semibold">L2 Regularization (Ridge Regression)</h3>
    <p>
      L2 Regularization ลงโทษค่าของ weight โดยการเพิ่มผลรวมของค่ากำลังสองของน้ำหนักทั้งหมดเข้าไปใน Loss function สูตรคือ:
    </p>

    <pre><code className="language-latex">
L2(w) = λ * Σ wᵢ²
    </code></pre>

    <p>
      ผลที่ได้คือ weight ที่มีค่ามากจะถูกลงโทษรุนแรง ทำให้ weight ส่วนใหญ่มีขนาดเล็กลง ส่งผลให้โมเดลมีเสถียรภาพมากขึ้น
    </p>

    <h4 className="text-lg font-medium">จุดเด่นของ L2 Regularization</h4>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ควบคุมความซับซ้อนของโมเดลได้อย่างต่อเนื่อง</li>
      <li>ช่วยกระจายค่าพารามิเตอร์ให้ไม่กระจุกอยู่ที่ feature เดียว</li>
      <li>ใช้กันอย่างแพร่หลายในโมเดล Deep Learning</li>
    </ul>

    <h3 className="text-xl font-semibold">L1 Regularization (Lasso)</h3>
    <p>
      L1 Regularization ใช้ค่าผลรวมสัมบูรณ์ของ weight มาเป็น penalty term:
    </p>

    <pre><code className="language-latex">
L1(w) = λ * Σ |wᵢ|
    </code></pre>

    <p>
      ความพิเศษของ L1 คือสามารถทำให้ weight ของบาง feature กลายเป็น 0 ได้ ทำให้โมเดลมีความสามารถในการเลือก feature อัตโนมัติ (feature selection)
    </p>

    <h4 className="text-lg font-medium">จุดเด่นของ L1 Regularization</h4>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ช่วยลดจำนวน feature ที่โมเดลต้องพิจารณา</li>
      <li>สามารถใช้ได้ดีกับข้อมูลที่มี sparsity สูง</li>
      <li>เหมาะสำหรับโมเดลที่เน้นการตีความ (interpretable)</li>
    </ul>

    <h3 className="text-xl font-semibold">การเปรียบเทียบ L1 กับ L2</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700 text-left">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="py-2 px-4 border-r">คุณสมบัติ</th>
          <th className="py-2 px-4 border-r">L1 Regularization</th>
          <th className="py-2 px-4">L2 Regularization</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">ประเภท Penalty</td>
          <td className="py-2 px-4 border-r">|w|</td>
          <td className="py-2 px-4">w²</td>
        </tr>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">ผลต่อ Weight</td>
          <td className="py-2 px-4 border-r">บาง weight เป็น 0</td>
          <td className="py-2 px-4">ทุก weight มีค่าเล็กลง</td>
        </tr>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">การใช้งาน</td>
          <td className="py-2 px-4 border-r">Feature Selection</td>
          <td className="py-2 px-4">Prevent Overfitting</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ตัวอย่างใน PyTorch</h3>
    <pre><code className="language-python">
# ตัวอย่างการใช้ L2 Regularization ใน PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

# หมายเหตุ: weight_decay คือตัวควบคุม λ ของ L2
    </code></pre>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>L2 เหมาะกับงานที่ต้องการโมเดลที่เสถียร ไม่ตัด feature ออก</li>
        <li>L1 เหมาะกับงานที่ข้อมูลมี sparsity หรือเน้นการตีความง่าย</li>
        <li>สามารถใช้ Elastic Net ซึ่งรวม L1 และ L2 เพื่อดึงข้อดีทั้งสองแบบ</li>
      </ul>
    </div>
  </div>
</section>


<section id="dropout" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. เทคนิค Dropout และการป้องกันการพึ่งพา Neuron</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Dropout เป็นเทคนิค Regularization ที่มีประสิทธิภาพในการลด Overfitting โดยถูกพัฒนาขึ้นโดย Geoffrey Hinton และทีมงานจาก University of Toronto ซึ่งได้รับความนิยมอย่างแพร่หลายในการฝึก Neural Networks โดยเฉพาะ Deep Networks ที่มีจำนวนพารามิเตอร์จำนวนมาก
    </p>

    <h3 className="text-xl font-semibold">หลักการของ Dropout</h3>
    <p>
      ในระหว่างการฝึก (training phase) Dropout จะสุ่ม “ปิด” การทำงานของ neuron บางตัวภายในแต่ละชั้น ด้วยความน่าจะเป็นที่กำหนดไว้ (เช่น 0.5 หรือ 50%) ส่งผลให้โมเดลไม่สามารถพึ่งพา neuron ใด neuron หนึ่งมากเกินไป และช่วยให้เกิดการเรียนรู้แบบกระจาย (distributed learning)
    </p>

    <pre><code className="language-python">
# ตัวอย่าง Dropout Layer ใน PyTorch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # ปิด neuron 50%
    nn.Linear(256, 10)
)
    </code></pre>

    <p>
      ใน phase การทดสอบ (inference/test) Dropout จะถูกปิดและ neuron ทั้งหมดจะเปิดใช้งาน เพื่อใช้ค่าถัวเฉลี่ยจากการฝึก โดย scaling ค่า weight อัตโนมัติ
    </p>

    <h3 className="text-xl font-semibold">ข้อดีของการใช้ Dropout</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ลด Overfitting โดยไม่ต้องพึ่งการปรับลดขนาดโมเดล</li>
      <li>เพิ่มความสามารถในการ Generalize ไปยังข้อมูลใหม่</li>
      <li>เพิ่ม robust ของโมเดลต่อข้อมูล noisy หรือ incomplete</li>
      <li>ทำหน้าที่เหมือนการรวมโมเดลหลายตัว (ensemble)</li>
    </ul>

    <h3 className="text-xl font-semibold">ทำไม Dropout จึงทำงานได้ดี?</h3>
    <p>
      การปิด neuron แบบสุ่มในแต่ละ epoch ทำให้โมเดลเรียนรู้หลาย subnetworks ที่แตกต่างกัน ส่งผลให้การเรียนรู้รวมผลแบบ ensemble โดยไม่ต้องฝึกหลายโมเดล ซึ่งช่วยลดการฟิตเฉพาะ training set และเพิ่มความสามารถในการ generalization
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
      <p className="font-medium">คำอธิบายจากงานวิจัยต้นฉบับ:</p>
      <p>
        “A good way to reduce overfitting is to combine the predictions of many different models. Dropout does this cheaply by training many different sub-models that share parameters.” — Hinton et al., 2014
      </p>
    </div>

    <h3 className="text-xl font-semibold">ค่าที่นิยมใช้ใน Dropout</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>0.5</strong>: สำหรับ fully connected layers</li>
      <li><strong>0.2–0.3</strong>: สำหรับ convolutional layers</li>
      <li><strong>0.1 หรือน้อยกว่า</strong>: สำหรับ layer ที่สำคัญหรือ sensitive</li>
    </ul>

    <h3 className="text-xl font-semibold">ผลลัพธ์เมื่อใช้ Dropout</h3>
    <p>
      จากการทดลองใน ImageNet, MNIST และ CIFAR-10 พบว่า Dropout สามารถลด error rate ลงได้อย่างมีนัยสำคัญ และช่วยให้โมเดล deep neural networks มีประสิทธิภาพเทียบเท่าการฝึก ensemble หลายโมเดล
    </p>

    <h3 className="text-xl font-semibold">Dropout ใน TensorFlow / Keras</h3>
    <pre><code className="language-python">
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
    </code></pre>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ Dropout</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ไม่ควรใช้กับทุก layer โดยเฉพาะ output layer</li>
      <li>อาจต้องเพิ่มจำนวน epoch และ learning rate เล็กน้อย</li>
      <li>ไม่แนะนำให้ใช้กับ BatchNorm พร้อมกันใน layer เดียวกัน</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Dropout ทำหน้าที่คล้ายการฝึกโมเดล ensemble โดยใช้ parameter ร่วมกัน</li>
        <li>เหมาะอย่างยิ่งกับ fully connected layers ที่มีจำนวน neuron มาก</li>
        <li>ต้องปรับ learning rate และจำนวน epoch ให้เหมาะสมเมื่อนำ Dropout มาใช้</li>
      </ul>
    </div>
  </div>
</section>


<section id="data-augmentation" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. การเพิ่มข้อมูลด้วย Data Augmentation</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Data Augmentation คือเทคนิคที่ใช้ปรับปรุงชุดข้อมูลฝึกโดยการสร้างตัวอย่างใหม่จากข้อมูลที่มีอยู่ผ่านการแปลงต่างๆ โดยไม่เปลี่ยนแปลง class label ของข้อมูล เป็นแนวทางที่ได้รับความนิยมสูงสุดในงาน Computer Vision และกำลังขยายไปสู่ NLP และ Audio ด้วย
    </p>

    <h3 className="text-xl font-semibold">เหตุผลที่ต้องใช้ Data Augmentation</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เพิ่มจำนวนข้อมูลฝึกโดยไม่ต้องเก็บข้อมูลใหม่</li>
      <li>ลด Overfitting โดยเพิ่มความหลากหลายของข้อมูล</li>
      <li>เพิ่มความสามารถของโมเดลในการ generalize สู่ข้อมูลใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold">ประเภทของ Data Augmentation</h3>
    <p className="mt-4">
      การเลือกใช้เทคนิคต่างๆ ขึ้นอยู่กับ domain และประเภทของข้อมูล ตัวอย่างการใช้ในงานภาพ:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Flip</strong>: สะท้อนภาพแนวนอน (Horizontal Flip)</li>
      <li><strong>Rotation</strong>: หมุนภาพเล็กน้อย เช่น -15 ถึง +15 องศา</li>
      <li><strong>Scaling</strong>: ขยายหรือย่อขนาดภาพ</li>
      <li><strong>Translation</strong>: เลื่อนภาพตามแกน x/y</li>
      <li><strong>Color Jitter</strong>: เปลี่ยนแปลงความสว่าง, ความอิ่มตัว, คอนทราสต์</li>
      <li><strong>Gaussian Noise</strong>: ใส่สัญญาณรบกวน</li>
      <li><strong>Cutout</strong>: ซ่อนบางส่วนของภาพด้วยสี่เหลี่ยมดำ</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างใน PyTorch</h3>
    <pre><code className="language-python">
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
    </code></pre>

    <h3 className="text-xl font-semibold">AutoAugment & RandAugment</h3>
    <p>
      เทคนิคที่พัฒนาขึ้นโดย Google Research ซึ่งใช้ reinforcement learning เพื่อเลือกชุดของการแปลงที่ให้ผลลัพธ์ดีที่สุดต่อ accuracy
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
      <p className="font-medium">Insight:</p>
      <p>
        “AutoAugment improves generalization by learning augmentation policies from data itself, outperforming human-designed heuristics.” — Google Brain, 2019
      </p>
    </div>

    <h3 className="text-xl font-semibold">Data Augmentation สำหรับงาน NLP และ Audio</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>NLP</strong>: Back translation, Synonym replacement, Random deletion</li>
      <li><strong>Audio</strong>: Time shifting, Pitch shifting, Adding background noise</li>
    </ul>

    <h3 className="text-xl font-semibold">ผลกระทบของ Data Augmentation ต่อโมเดล</h3>
    <p>
      งานวิจัยหลายฉบับ เช่นใน ImageNet และ CIFAR-100 แสดงให้เห็นว่าเมื่อใช้ Data Augmentation อย่างเหมาะสม สามารถเพิ่ม Top-1 Accuracy ได้ถึง 2–5% โดยไม่ต้องเปลี่ยนโมเดล
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างใน Keras</h3>
    <pre><code className="language-python">
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
    </code></pre>

    <h3 className="text-xl font-semibold">การประยุกต์ใช้ใน Production</h3>
    <p>
      ควรใช้ Data Augmentation เฉพาะในขั้นตอน training เท่านั้น และปิด augmentation เมื่อทำ validation หรือ testing เพื่อให้ได้การประเมินผลที่แม่นยำ
    </p>

    <h3 className="text-xl font-semibold">ข้อควรระวัง</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ไม่ควรใช้การแปลงที่เปลี่ยน class label เช่น flip บนข้อมูลที่มีทิศทางเฉพาะ</li>
      <li>อาจเพิ่มเวลาในการฝึก เนื่องจากข้อมูลมีความซับซ้อนสูงขึ้น</li>
      <li>ควรใช้ร่วมกับเทคนิค regularization อื่นๆ เพื่อประสิทธิภาพสูงสุด</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h4 className="font-semibold mb-2">Insight Box:</h4>
      <ul className="list-disc list-inside space-y-1">
        <li>Data Augmentation เปรียบเสมือนการเพิ่มมิติความหลากหลายให้กับข้อมูล</li>
        <li>ช่วยสร้าง robustness โดยไม่เพิ่มต้นทุนด้านการจัดเก็บ</li>
        <li>ควบคุมได้ง่ายผ่าน library ที่รองรับ เช่น torchvision, albumentations</li>
      </ul>
    </div>
  </div>
</section>

<section id="batchnorm-earlystop" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Batch Normalization และ Early Stopping</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <h3 className="text-xl font-semibold">Batch Normalization คืออะไร?</h3>
    <p>
      Batch Normalization (BN) เป็นเทคนิคที่เสนอโดย Ioffe และ Szegedy ในปี 2015 เพื่อแก้ปัญหา internal covariate shift ซึ่งหมายถึงการเปลี่ยนแปลงของ distribution ภายในระหว่าง training โดยการปรับค่า activation ให้มี mean ≈ 0 และ variance ≈ 1 ทำให้ training มีความเสถียรขึ้น
    </p>

    <h3 className="text-xl font-semibold">หลักการทำงานของ Batch Normalization</h3>
    <ul className="list-decimal list-inside ml-6 space-y-2">
      <li>คำนวณค่า mean และ variance ของ activation ในแต่ละ mini-batch</li>
      <li>ทำ normalization โดยการลบ mean และหารด้วย standard deviation</li>
      <li>ปรับสเกลและเลื่อนค่าด้วย parameter trainable: gamma และ beta</li>
    </ul>

    <pre><code className="language-python">
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU()
)
    </code></pre>

    <h3 className="text-xl font-semibold">ข้อดีของ Batch Normalization</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เร่งความเร็วในการฝึกโมเดล</li>
      <li>ลดความต้องการใช้ค่า learning rate ต่ำ</li>
      <li>ช่วยลดโอกาสเกิด overfitting</li>
      <li>ลดผลกระทบจากการเลือก weight initialization ไม่ดี</li>
    </ul>

    <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
      <p className="font-medium">Insight:</p>
      <p>
        ใน ResNet และ Inception Network ทุก layer มักแทรก BN ระหว่าง Linear และ Activation function ซึ่งช่วยให้ network สามารถลึกได้โดยไม่เกิด vanishing gradient
      </p>
    </div>

    <h3 className="text-xl font-semibold">Early Stopping คืออะไร?</h3>
    <p>
      Early Stopping เป็นเทคนิค regularization ที่หยุดการฝึกโมเดลเมื่อ performance บน validation set เริ่มแย่ลง ทั้งที่ training loss ยังลดลง โดยมีจุดประสงค์หลักเพื่อป้องกัน overfitting
    </p>

    <h3 className="text-xl font-semibold">ขั้นตอนของ Early Stopping</h3>
    <ol className="list-decimal list-inside ml-6 space-y-2">
      <li>แบ่งข้อมูลเป็น training และ validation</li>
      <li>ฝึกโมเดลและติดตาม validation loss ในแต่ละ epoch</li>
      <li>หยุดฝึกทันทีที่ validation loss ไม่ดีขึ้นหลังจาก N epoch (patience)</li>
    </ol>

    <pre><code className="language-python">
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
model.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stop])
    </code></pre>

    <h3 className="text-xl font-semibold">ข้อดีของ Early Stopping</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ลดเวลาในการฝึกโมเดล</li>
      <li>ป้องกันการเรียนรู้ noise เกินจำเป็น</li>
      <li>ไม่ต้องกำหนดจำนวน epoch ล่วงหน้าอย่างแม่นยำ</li>
    </ul>

    <h3 className="text-xl font-semibold">การใช้ร่วมกัน</h3>
    <p>
      Batch Normalization และ Early Stopping เป็นเทคนิคที่เสริมกันได้ดี โดย BN ทำให้ training เสถียรขึ้น และ Early Stopping ช่วยหยุดโมเดลเมื่อเริ่ม overfit ส่งผลให้โมเดลที่ได้ generalize ได้ดี
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-800 p-4 rounded-lg">
      <h4 className="font-semibold mb-2">Insight Box:</h4>
      <ul className="list-disc list-inside space-y-1">
        <li>BatchNorm ทำให้สามารถใช้ learning rate สูงขึ้นได้อย่างปลอดภัย</li>
        <li>Early Stopping เป็นแนวทาง "soft regularization" ที่เน้นการ monitor validation</li>
        <li>เมื่อใช้ร่วมกับ Data Augmentation และ Dropout มักให้ performance ที่ดีที่สุด</li>
      </ul>
    </div>

  </div>
</section>


<section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. บทสรุปเชิงลึก (Insight Recap)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <h3 className="text-xl font-semibold">ภาพรวมของ Regularization</h3>
    <p>
      Regularization คือชุดของเทคนิคที่ใช้ควบคุมความซับซ้อนของโมเดล โดยมีเป้าหมายเพื่อเพิ่มความสามารถในการ generalize กับข้อมูลใหม่ โดยไม่จดจำ noise ในชุดฝึกหัด ซึ่งถือเป็นปัจจัยสำคัญต่อการป้องกัน overfitting ในงานประมวลผลภาพ ภาษา และข้อมูลขนาดใหญ่
    </p>

    <h3 className="text-xl font-semibold">Insight สำคัญจากแต่ละเทคนิค</h3>
    <ul className="list-disc list-inside space-y-3">
      <li>
        <strong>Weight Regularization (L1 / L2):</strong> L1 สนับสนุนการเกิด sparsity ของพารามิเตอร์ ในขณะที่ L2 ส่งเสริมการจำกัดค่าพารามิเตอร์ให้อยู่ใกล้ศูนย์ ซึ่งเหมาะกับโมเดลที่มีพารามิเตอร์จำนวนมาก
      </li>
      <li>
        <strong>Dropout:</strong> ทำหน้าที่ป้องกันการพึ่งพา neuron เฉพาะจุดมากเกินไป ทำให้ network ต้องเรียนรู้ representation ที่กระจายตัวและ robust ต่อการลบ node แบบสุ่ม
      </li>
      <li>
        <strong>Data Augmentation:</strong> เพิ่ม diversity ให้กับ dataset โดยสร้างข้อมูลใหม่จากข้อมูลเดิมผ่านการแปลง เช่น การหมุน ขยาย ลดแสง ซึ่งมีบทบาทสำคัญอย่างยิ่งในงานภาพและเสียง
      </li>
      <li>
        <strong>Batch Normalization:</strong> ปรับค่า activation ให้มี distribution ที่เหมาะสมในแต่ละ mini-batch ส่งผลให้การฝึกโมเดลมีเสถียรภาพ ลดความไวต่อการตั้งค่า hyperparameter
      </li>
      <li>
        <strong>Early Stopping:</strong> หยุดการฝึกทันทีเมื่อ validation loss หยุดพัฒนา แม้ training loss จะยังคงลดลง เพื่อป้องกันการ overfit แบบเงียบ (silent overfit)
      </li>
    </ul>

    <h3 className="text-xl font-semibold">กลยุทธ์การผสมผสานเพื่อผลลัพธ์สูงสุด</h3>
    <p>
      งานวิจัยจาก Google Research และ DeepMind พบว่า การใช้ Regularization หลายประเภทร่วมกันในสัดส่วนที่เหมาะสม เช่น Dropout + L2 + Data Augmentation สามารถเพิ่มความแม่นยำของโมเดลได้อย่างมีนัยสำคัญ
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ในงาน Computer Vision: ใช้ Data Augmentation ควบคู่กับ BatchNorm และ L2</li>
      <li>ใน NLP: ใช้ Dropout และ EarlyStopping ร่วมกับ embedding regularization</li>
      <li>ในงาน Time-Series: ใช้ EarlyStopping + Gaussian Noise Injection</li>
    </ul>

    <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-xl space-y-2">
      <h4 className="text-lg font-semibold">Insight Box: แนวโน้มของ Regularization ในยุค Foundation Model</h4>
      <p>
        การพัฒนาโมเดลขนาดใหญ่ (เช่น GPT, BERT, CLIP) มีการ shift จาก regularization แบบดั้งเดิมไปสู่เทคนิคใหม่ เช่น:
      </p>
      <ul className="list-disc list-inside ml-6 space-y-1">
        <li>Stochastic Depth และ Mixout ในงาน Vision Transformer</li>
        <li>Label Smoothing และ Sharpness-Aware Minimization ในงาน Classification</li>
        <li>Adaptive Dropout ที่เปลี่ยน rate ตาม gradient dynamics</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">คำแนะนำการประยุกต์ใช้งานจริง</h3>
    <p>
      ควรเริ่มจากการใช้ L2 Regularization และ Early Stopping เป็น baseline แล้วเพิ่ม Dropout หรือ Data Augmentation ตามลักษณะของข้อมูล หากพบว่าโมเดลยัง overfit ให้ปรับเพิ่มความเข้มข้นของ regularization ทีละน้อย โดยไม่ควรใช้ทุกเทคนิคพร้อมกันในครั้งแรก
    </p>

    <h3 className="text-xl font-semibold">ตารางสรุปการเลือกใช้ Regularization</h3>
    <div className="overflow-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-100 dark:bg-gray-800">
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">เหมาะกับงาน</th>
            <th className="border px-4 py-2">จุดเด่น</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">L1 / L2</td>
            <td className="border px-4 py-2">ทุกประเภท</td>
            <td className="border px-4 py-2">คุมขนาด weight</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Dropout</td>
            <td className="border px-4 py-2">NLP, Vision</td>
            <td className="border px-4 py-2">ลด co-adaptation</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Data Augmentation</td>
            <td className="border px-4 py-2">Vision, Speech</td>
            <td className="border px-4 py-2">เพิ่ม diversity</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Batch Norm</td>
            <td className="border px-4 py-2">Deep Network</td>
            <td className="border px-4 py-2">เสถียรภาพในการฝึก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Early Stopping</td>
            <td className="border px-4 py-2">ทุกประเภท</td>
            <td className="border px-4 py-2">หยุดก่อน overfit</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</section>


<section id="references" className="mb-20 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-6 text-center">แหล่งอ้างอิง</h2>
  <ul className="list-disc list-inside space-y-2 px-4 md:px-12 lg:px-20 text-base">
    <li>
      Stanford CS231n: Convolutional Neural Networks for Visual Recognition — <a href="http://cs231n.stanford.edu/" target="_blank" className="text-blue-600 underline">http://cs231n.stanford.edu/</a>
    </li>
    <li>
      MIT 6.S191: Introduction to Deep Learning — <a href="https://introtodeeplearning.mit.edu/" target="_blank" className="text-blue-600 underline">https://introtodeeplearning.mit.edu/</a>
    </li>
    <li>
      Goodfellow et al., Deep Learning Book — Chapter 7: Regularization
    </li>
    <li>
      Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Srivastava et al. (2014)
    </li>
  </ul>
</section>



          {/* Quiz Section */}
          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day25 theme={theme} />
          </section>

          {/* Tags Section */}
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
        </div>
      </div>

      {/* ScrollSpy Sidebar */}
      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day25 />
      </div>

      {/* Support Me Button */}
      <SupportMeButton />
    </div>
  );
};

export default Day25_CNNRegularization;
