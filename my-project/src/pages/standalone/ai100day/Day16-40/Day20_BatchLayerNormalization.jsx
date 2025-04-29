import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day20 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day20";
import MiniQuiz_Day20 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day20";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day20_BatchLayerNormalization = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("batchnorm1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("batchnorm2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("batchnorm3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("batchnorm4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("batchnorm5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("batchnorm6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("batchnorm7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("batchnorm8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("batchnorm9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("batchnorm10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("batchnorm11").format("auto").quality("auto").resize(scale().width(599));
  const img12 = cld.image("batchnorm12").format("auto").quality("auto").resize(scale().width(599));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 20: Batch Normalization & Layer Normalization</h1>

        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img1} />
        </div>
        <div className="w-full flex justify-center my-12">
      <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
    </div>
        {/* Sections */}
        <section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้องทำ Normalization ใน Neural Network</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img2} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Neural Network ที่ลึกสร้างมาใน Deep Learning เป็นระบบบ่นหลัก มีโครงสร้างหลายและปัญหาแปรปรนแปลงให้การฝึ้นได้ยาก แต่ Internal Covariate Shift ปัญหาสอนในการเรียนแพร Distribution ของ Activations เมื่อที่การอัปเดต Weights หรือ Bias
    </p>

    <p>
      การเปลี่ยน Distribution ของ Hidden Layer แบบมีความเปลี่ยนบ่อยบ่อยไปเรื่อย่า ระหว่า Activations เปลี่ยน ส่งผลให้ Training ยากขึ้น ต้องเปลี่ยน Learning Rate ใหม่สม่ำเสมอ และทำให้การ Convergence เป็นไปได้ยาก
    </p>


    <h3 className="text-xl font-semibold">Internal Covariate Shift คืออะไร?</h3>
    <p>
      ปัญหาสาคัญเพาะสำคัญโดย Sergey Ioffe และ Christian Szegedy (Google Research, 2015) ได้นิยามไว้ว่า ในระหว่าการฝึ้งโปรแลยโปรแบบลหรือชั้นบน Hidden Layer ต่างๆ ทำให้ Distribution ของ Inputs แปลงเปลี่ยนตลอดบ่อยไป จากการอัปเดต Weights ส่งผลให้ Hidden Layers ต่างๆ ส่งผลเปลี่ยนตลอดส่อบ่อยไป
    </p>

    <h3 className="text-xl font-semibold">ผลกระทบจาก Internal Covariate Shift</h3>
    <ul className="list-disc pl-6">
      <li>ทำให้ต้องลด Learning Rate ลงเพื่อให้การฝึกเสถียร</li>
      <li>ต้องการการตั้งค่า Hyperparameter อย่างละเอียดมากขึ้น</li>
      <li>Training Loss ผันผวนสูง ทำให้ Convergence ช้ามาก</li>
      <li>เพิ่มโอกาส Overfitting ในกรณีโมเดลลึกและซับซ้อน</li>
    </ul>

    <h3 className="text-xl font-semibold">ทางแก้ปัญหา: Normalization Techniques</h3>
    <p>
      แนวคิดการ Normalization ได้ถูกเสนอเพื่อแก้ไขปัญหา Internal Covariate Shift ตัวอย่างเช่น Batch Normalization, Layer Normalization, Group Normalization และ Instance Normalization ซึ่งแต่ละวิธีมีหลักการและการใช้งานที่เหมาะกับสถานการณ์ต่างกันไป
    </p>

    <div className="flex flex-col md:flex-row gap-6">
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">Batch Normalization</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Normalize activations โดยใช้ mean และ variance ของ mini-batch</li>
          <li>เหมาะกับ CNN, MLP</li>
          <li>เร่งการลู่เข้าอย่างมากใน Training</li>
        </ul>
      </div>

      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">Layer Normalization</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Normalize ต่อ 1 ตัวอย่าง โดยใช้ feature dimension</li>
          <li>เหมาะกับ RNNs, Transformers</li>
          <li>ทำงานได้แม้ batch มีขนาดเล็กมาก</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Insight จากงานวิจัยสากล</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm">
        <li>Batch Normalization ลดความจำเป็นในการจูน Learning Rate อย่างมาก (Ioffe & Szegedy, 2015)</li>
        <li>Layer Normalization ส่งผลดีอย่างชัดเจนต่อความเสถียรของ RNN และ Transformer (Ba et al., 2016)</li>
        <li>Normalization ช่วยให้ Landscape ของ Loss Smooth ขึ้น ทำให้การ Optimization ง่ายขึ้น (Santurkar et al., 2018)</li>
        <li>Normalization techniques ยังมีผลเป็น Regularization ทางอ้อม ลด Overfitting ได้</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">สรุปบทนำ</h3>
    <p>
      การทำ Normalization กลายเป็นหนึ่งในเทคนิคที่ขาดไม่ได้ในการฝึก Neural Networks สมัยใหม่ เนื่องจากช่วยแก้ปัญหา Internal Covariate Shift เพิ่มความเสถียรและความเร็วในการฝึกโมเดลลึก และเป็นรากฐานสำคัญที่ช่วยให้ Deep Learning พัฒนาขึ้นมาอย่างก้าวกระโดดในช่วงทศวรรษที่ผ่านมา
    </p>
  </div>
</section>


<section id="internal-covariate-shift" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Internal Covariate Shift: ปัญหาที่ซ่อนอยู่</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img3} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <p>
      Internal Covariate Shift คือหนึ่งในอุปสรรคสำคัญที่ทำให้การฝึก Neural Networks โดยเฉพาะ Deep Networks มีความซับซ้อนและใช้เวลานานขึ้นอย่างมาก คำว่า Covariate Shift โดยทั่วไปหมายถึงการที่ distribution ของ input เปลี่ยนไปจากที่โมเดลได้เรียนรู้ไว้ ในขณะที่ Internal Covariate Shift เกิดขึ้นภายในเครือข่ายเอง โดยเป็นการเปลี่ยนแปลง distribution ของ activation ของ hidden layers ระหว่างขั้นตอนการฝึก
    </p>

    <h3 className="text-xl font-semibold">นิยามอย่างเป็นทางการ</h3>
    <p>
      จากงานวิจัยของ Sergey Ioffe และ Christian Szegedy (Google Research, 2015) ที่นำเสนอแนวคิด Batch Normalization ได้ให้คำจำกัดความ Internal Covariate Shift ไว้ว่าเป็น "การเปลี่ยนแปลงของ distribution ของ input ของแต่ละ layer ขณะที่ parameters ของ layers ก่อนหน้าเปลี่ยนแปลงระหว่างการฝึก"
    </p>

    <p>
      กล่าวอีกนัยหนึ่งคือ ในทุก ๆ การอัปเดตพารามิเตอร์ของเครือข่าย ข้อมูลที่ส่งต่อไปยังเลเยอร์ถัดไปก็มี distribution ที่เปลี่ยนแปลงไปด้วย ทำให้เลเยอร์ถัดไปต้องพยายามเรียนรู้จากข้อมูลที่มี distribution ไม่คงที่อยู่ตลอดเวลา
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="text-lg font-semibold mb-2">Insight: ทำไมปัญหานี้ร้ายแรง?</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>Training ยากขึ้นเพราะต้องตามการเปลี่ยนแปลง distribution ตลอดเวลา</li>
        <li>ต้องใช้ Learning Rate เล็กลงเพื่อให้เสถียร ทำให้การฝึกช้าลงมาก</li>
        <li>เสี่ยงต่อการติด local minima ที่ไม่ดี เนื่องจากพื้นผิว loss landscape เปลี่ยนรูปร่างตลอดเวลา</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold text-center">ตัวอย่างเชิงภาพ: Covariate Shift ภายใน</h3>
    <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img4} />
        </div>
    <div className="flex flex-col md:flex-row gap-6">
      <div className="flex-1">
        
        <p className="text-center text-sm mt-2">ก่อนฝึก: Distribution คงที่</p>
      </div>
      <div className="flex-1">
    
        <p className="text-center text-sm mt-2">ระหว่างฝึก: Distribution เปลี่ยนตลอดเวลา</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10">ผลกระทบของ Internal Covariate Shift</h3>
    <ul className="list-disc pl-6 space-y-3">
      <li><strong>จำเป็นต้องจูน Learning Rate อย่างละเอียด:</strong> Distribution ที่เปลี่ยนไปทำให้การใช้ค่า Learning Rate เดียวกันตลอดการฝึกไม่เหมาะสม</li>
      <li><strong>การฝึกต้องการเวลาเพิ่มขึ้น:</strong> เนื่องจากโมเดลต้องใช้ iteration เพิ่มขึ้นเพื่อตามการเปลี่ยนแปลงของ distribution</li>
      <li><strong>Training Loss มีการแกว่งตัว:</strong> ไม่สามารถลดได้อย่างราบรื่น</li>
      <li><strong>Validation Loss มีความไม่เสถียร:</strong> กระทบต่อการวัด performance และการเลือกโมเดลที่ดีที่สุด</li>
      <li><strong>ต้องใช้เทคนิคเสริม:</strong> เช่น Batch Normalization, Weight Normalization หรือ Layer Normalization เพื่อบรรเทาปัญหานี้</li>
    </ul>

    <h3 className="text-xl font-semibold">ทำไม Deep Network ยิ่งลึก → ยิ่งเสี่ยง Internal Covariate Shift รุนแรง</h3>
    <p>
      ในเครือข่ายที่มีหลายชั้นมากขึ้น (Deep Neural Networks) ปัญหา Internal Covariate Shift จะสะสมในแต่ละเลเยอร์ ยิ่งเลเยอร์มากเท่าไหร่ การเปลี่ยนแปลง distribution ของข้อมูลที่ส่งผ่านก็ยิ่งซับซ้อนและรุนแรงมากขึ้นไปด้วย ทำให้การฝึกเครือข่ายลึกเป็นไปได้ยากและต้องใช้กลยุทธ์พิเศษเพื่อแก้ไข
    </p>

    <div className="grid md:grid-cols-2 gap-6 my-8">
      <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-lg">
        <h4 className="font-semibold mb-2">Neural Network ตื้น (Shallow)</h4>
        <ul className="list-disc pl-6 text-sm">
          <li>Distribution เปลี่ยนไม่มาก</li>
          <li>Training เสถียรโดยไม่ต้องทำ Normalization</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-lg">
        <h4 className="font-semibold mb-2">Neural Network ลึก (Deep)</h4>
        <ul className="list-disc pl-6 text-sm">
          <li>Distribution เปลี่ยนอย่างรุนแรงในแต่ละเลเยอร์</li>
          <li>Training ช้ามากหรือไม่เสถียรหากไม่ทำ Normalization</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างปัญหา Internal Covariate Shift จริง</h3>
    <p>
      ในงานวิจัย "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" โดย Ioffe และ Szegedy พบว่า ResNet-50 ที่ไม่ใช้ BatchNorm ไม่สามารถฝึกให้ convergence ได้เลยในทางปฏิบัติ และมี validation error สูงถึง 45% ขณะที่เมื่อใช้ BatchNorm validation error ลดลงเหลือเพียง 23% เท่านั้น
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <h4 className="text-lg font-semibold mb-2">Key Points จากผลการทดลอง</h4>
      <ul className="list-disc pl-6 text-sm">
        <li>Internal Covariate Shift เป็น bottleneck หลักใน Deep Learning</li>
        <li>Batch Normalization มีผลลดปัญหานี้อย่างเห็นได้ชัด</li>
        <li>เมื่อจัดการ Internal Covariate Shift ได้ การฝึกโมเดลลึกเป็นไปได้จริงและได้ผลลัพธ์ระดับโลก</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">วิธีการแก้ไข Internal Covariate Shift</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Batch Normalization:</strong> ปรับ Mean และ Variance ของ activation แต่ละเลเยอร์ให้คงที่</li>
      <li><strong>Layer Normalization:</strong> เหมาะสำหรับงานที่มี batch size เล็ก หรือ Sequential Models</li>
      <li><strong>Weight Normalization:</strong> ปรับน้ำหนักแทนการปรับ activation โดยตรง</li>
      <li><strong>Group Normalization:</strong> ใช้เมื่อ batch size มีความไม่แน่นอน เช่นในงาน Object Detection หรือ Video Processing</li>
    </ul>

    <div className="flex flex-col md:flex-row gap-6 mt-8">
      <div className="bg-green-100 dark:bg-green-900 p-5 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">BatchNorm</h4>
        <p className="text-sm">เหมาะกับ CNN และงานที่มี batch size ใหญ่</p>
      </div>
      <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">LayerNorm</h4>
        <p className="text-sm">เหมาะกับ RNN และ Transformer ที่ batch size เปลี่ยนไป</p>
      </div>
      <div className="bg-red-100 dark:bg-red-900 p-5 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">GroupNorm</h4>
        <p className="text-sm">เหมาะกับงานที่ใช้ batch size เล็กหรือไม่สม่ำเสมอ</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10">สรุป</h3>
    <p>
      การเข้าใจปัญหา Internal Covariate Shift เป็นกุญแจสำคัญในการออกแบบเครือข่ายลึกที่สามารถฝึกได้จริงในทางปฏิบัติ การใช้เทคนิคต่าง ๆ เช่น Batch Normalization, Layer Normalization และ Group Normalization ไม่เพียงแต่ช่วยให้ training เสถียรขึ้น แต่ยังเร่งความเร็วการเรียนรู้ ลดความจำเป็นในการจูน hyperparameters อย่างละเอียด และปรับปรุง generalization ของโมเดลในงานจริงอย่างมีนัยสำคัญ
    </p>

  </div>
</section>


<section id="batch-normalization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Batch Normalization (BN)</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img5} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Batch Normalization (BN) เป็นหนึ่งในเทคนิคที่ปฏิวัติวงการ Deep Learning โดยมีเป้าหมายเพื่อลดปัญหา Internal Covariate Shift ที่เกิดขึ้นระหว่างการฝึกโมเดลลึก ซึ่งส่งผลต่อความเร็วและเสถียรภาพในการฝึกอย่างมีนัยสำคัญ งานวิจัยต้นฉบับจาก Sergey Ioffe และ Christian Szegedy (2015) เป็นรากฐานของเทคนิคนี้ และยังคงเป็นองค์ประกอบสำคัญในสถาปัตยกรรม Neural Networks ชั้นนำ เช่น ResNet, Transformer และ EfficientNet.
    </p>

    <h3 className="text-xl font-semibold">หลักการทำงานของ Batch Normalization</h3>
    <p>
      Batch Normalization ทำการ Normalize ค่า Activation ของแต่ละ Layer ผ่านการคำนวณค่า Mean และ Variance ของ Activation ภายใน Mini-Batch หนึ่ง จากนั้นปรับสเกลและเลื่อนศูนย์ด้วยพารามิเตอร์ trainable γ (gamma) และ β (beta).
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
    <pre className="text-sm overflow-x-auto">
{`1. คำนวณ Mean:
   μ_B = (1/m) Σ_{i=1}^m x_i

2. คำนวณ Variance:
   σ_B² = (1/m) Σ_{i=1}^m (x_i - μ_B)²

3. Normalize:
   ẋ_i = (x_i - μ_B) / √(σ_B² + ε)

4. Scale and Shift:
   y_i = γ ẋ_i + β`}
</pre>

    </div>

    <p>
      ค่า \( \epsilon \) เป็นค่าคงที่ขนาดเล็กที่ใช้เพื่อป้องกันการหารด้วยศูนย์ โดยปกติตั้งไว้ที่ \( 10^{-5} \) หรือ \( 10^{-6} \).
    </p>

    <h3 className="text-xl font-semibold">ประโยชน์หลักของ Batch Normalization</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ลด Internal Covariate Shift ซึ่งช่วยเร่งกระบวนการเรียนรู้</li>
      <li>ทำให้โมเดลสามารถใช้ Learning Rate ที่สูงขึ้นได้โดยไม่ทำให้ Diverge</li>
      <li>เพิ่มเสถียรภาพในการฝึกโมเดลที่ลึกมาก</li>
      <li>มีลักษณะของ Regularization โดยช่วยลดโอกาส Overfitting</li>
    </ul>

    <h3 className="text-xl font-semibold">ตำแหน่งการวาง Batch Normalization</h3>
    <p>
      มีการถกเถียงในงานวิจัยว่า BN ควรวางไว้ก่อนหรือหลัง Activation Function เช่น ReLU โดยผลสรุปจาก ResNet และ VGG ชี้ว่าการวาง BN ก่อน Activation (Pre-activation) ให้ผลที่ดีกว่าในหลายกรณี.
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-lg shadow border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-4">Pre-Activation BN</h4>
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>BatchNorm ➔ Activation ➔ Weight Layer</li>
          <li>ช่วยให้ Gradient Flow ผ่าน Network ได้ดีขึ้น</li>
          <li>นิยมใช้ใน ResNet v2 และสถาปัตยกรรมสมัยใหม่</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-900 p-6 rounded-lg shadow border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-4">Post-Activation BN</h4>
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>Weight Layer ➔ Activation ➔ BatchNorm</li>
          <li>นิยมใช้ในสถาปัตยกรรมคลาสสิกเช่น VGG</li>
          <li>ง่ายต่อการทำความเข้าใจแต่ประสิทธิภาพต่ำกว่าเล็กน้อย</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">ผลกระทบต่อการฝึกและ Performance</h3>
    <p>
      จากการศึกษาของ Google Brain และ DeepMind พบว่าการใช้ BN สามารถลดจำนวน Epochs ที่จำเป็นในการฝึกโมเดลได้ถึง 2-3 เท่า และเพิ่มโอกาสในการฝึกโมเดลขนาดใหญ่มากที่มีชั้นหลายร้อยถึงพันชั้นโดยไม่เกิดปัญหา Vanishing Gradient.
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้ Batch Normalization ใน PyTorch</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
      <pre className="text-sm overflow-x-auto">
{`import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ Batch Normalization</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Batch Size ที่เล็กเกินไป (เช่น &lt; 32) อาจทำให้สถิติ Mean และ Variance ไม่น่าเชื่อถือ</li>
      <li>ไม่เหมาะกับงานที่ Batch Size เปลี่ยนตลอดเวลา เช่นบางประเภทของ RNNs หรือ Reinforcement Learning</li>
      <li>ต้องระมัดระวังระหว่าง Training และ Inference เพราะสถิติที่ใช้แตกต่างกัน</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight จากงานวิจัยล่าสุด</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>BatchNorm ช่วยปรับ Landscape ของ Loss ให้ Smooth ขึ้น ส่งผลให้ Optimization ง่ายขึ้น</li>
        <li>การ Fine-tune γ และ β ช่วยให้โมเดลปรับตัวกับงานเฉพาะทางได้ดีขึ้น เช่น Transfer Learning</li>
        <li>การใช้ SyncBatchNorm ช่วยให้ BN ใช้งานได้ดีใน Distributed Training ขนาดใหญ่</li>
        <li>มีงานวิจัยที่พัฒนา Virtual BN และ Ghost BN เพื่อแก้ปัญหา Batch Size เล็ก</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      Batch Normalization เป็นเครื่องมือทรงพลังที่ช่วยให้ Deep Neural Networks สามารถฝึกได้เร็วขึ้น เสถียรมากขึ้น และแม่นยำสูงขึ้น แม้จะมีข้อจำกัดบางประการในกรณีพิเศษ แต่โดยรวมแล้ว BN ยังคงเป็นหนึ่งในเทคนิคที่สำคัญที่สุดที่ขับเคลื่อนความก้าวหน้าของ Deep Learning Model ขนาดใหญ่ในยุคปัจจุบัน.
    </p>
  </div>
</section>

<section id="batchnorm-practice" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. BatchNorm ใน Practice</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img6} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Batch Normalization (BatchNorm) ได้กลายเป็นหนึ่งในเทคนิคที่สำคัญที่สุดในการฝึก Neural Networks ขนาดใหญ่ในโลกยุคใหม่ โดยเฉพาะในงานที่มีโครงสร้างลึกหลายชั้น เช่น Convolutional Neural Networks (CNNs) และ Transformers การใช้ BatchNorm อย่างเหมาะสมสามารถช่วยให้โมเดลเรียนรู้ได้รวดเร็วขึ้น ลดความไวต่อการตั้งค่า Learning Rate และเพิ่มความเสถียรของกระบวนการฝึกทั้งหมด
    </p>

    <h3 className="text-xl font-semibold">ควรใส่ BatchNorm ที่ตำแหน่งใด?</h3>
    <p>
      คำถามคลาสสิกในการใช้งาน BatchNorm คือควรวางมันก่อนหรือหลัง Activation Function? งานวิจัย "Identity Mappings in Deep Residual Networks" โดย He et al. (2016) แนะนำให้แทรก BatchNorm <strong>ก่อน Activation</strong> ซึ่งเรียกว่า "Pre-Activation" เพราะช่วยให้การไหลของ Gradient มีความราบรื่นขึ้น และทำให้โมเดลลึก ๆ ง่ายต่อการฝึกมากขึ้น
    </p>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-6 rounded-xl">
        <h4 className="text-lg font-semibold mb-2">Before Activation</h4>
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>เป็นแนวทางที่นิยมที่สุดใน ResNet, Transformer</li>
          <li>ช่วยให้ Optimization Landscape มีความราบรื่น</li>
          <li>มักได้ผลดีกว่าในโมเดลลึก ๆ</li>
        </ul>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-6 rounded-xl">
        <h4 className="text-lg font-semibold mb-2">After Activation</h4>
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>นิยมในโมเดลเก่าเช่น VGGNet</li>
          <li>อาจทำให้ Gradient Flow ติดขัดในบางกรณี</li>
          <li>มีความเสี่ยงต่อ Saturation เมื่อใช้ Activation แบบ Sigmoid หรือ Tanh</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">การใช้ BatchNorm กับสถาปัตยกรรมต่าง ๆ</h3>
    <p>
      BatchNorm ไม่ได้ใช้เหมือนกันทุกสถาปัตยกรรม การเลือกตำแหน่งและวิธีการใช้ควรพิจารณาจากประเภทของเครือข่ายที่ใช้งาน
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>CNNs:</strong> ใช้ BatchNorm หลัง Convolution Layer และก่อน Activation</li>
      <li><strong>MLPs:</strong> ใช้ BatchNorm หลัง Linear Layer และก่อน Activation</li>
      <li><strong>RNNs:</strong> การใช้ BatchNorm ยากขึ้นเนื่องจากลักษณะ sequential; มักใช้ LayerNorm แทน</li>
      <li><strong>Transformers:</strong> นิยมใช้ LayerNorm เป็นหลัก แต่บางงานใช้ BatchNorm ในบาง Layer ของ Encoder</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้งาน BatchNorm</h3>
    <p>
      แม้ BatchNorm จะมีข้อดีมหาศาล แต่ก็มีข้อจำกัดที่ควรระวัง เช่น ขนาดของ Batch Size ที่เล็กเกินไปสามารถทำให้การประมาณค่า Mean และ Variance ไม่น่าเชื่อถือ ส่งผลให้ประสิทธิภาพลดลงอย่างมีนัยสำคัญ งานของ Ioffe (2017) เสนอแนวทางใช้ Virtual BatchNorm หรือ GroupNorm เพื่อแก้ปัญหานี้ในบางกรณี
    </p>

    <h3 className="text-xl font-semibold">Batch Size ที่เหมาะสม</h3>
    <p>
      คำแนะนำจาก Facebook AI Research (FAIR) คือควรใช้ Batch Size อย่างน้อย 32–64 ต่อ GPU เพื่อให้ BatchNorm มีการประมาณค่าที่มีความแม่นยำพอสมควร การใช้ Batch Size เล็กเกินไป เช่น 4 หรือ 8 อาจทำให้โมเดลมีการเรียนรู้ที่ไม่นิ่ง และ Validation Accuracy ลดลง
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="text-lg font-semibold mb-4">Insight Box: ข้อแนะนำจาก DeepMind</h4>
      <ul className="list-disc pl-6 text-sm">
        <li>ในงานที่มี Batch Size เล็ก เช่น Reinforcement Learning ให้พิจารณา LayerNorm หรือ GroupNorm แทน</li>
        <li>ใน Distributed Training, ต้อง Sync Mean/Variance ข้าม GPUs เพื่อให้ผลลัพธ์สม่ำเสมอ</li>
        <li>อย่าปล่อยให้ Gamma (scale parameter) ถูกตัดออกระหว่างการทำ Quantization</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">BatchNorm กับ Regularization</h3>
    <p>
      การใช้ BatchNorm ส่งผลเหมือน Regularization แบบอ่อน ๆ ซึ่งช่วยลดความจำเป็นในการใช้ Dropout ในบางกรณี งานวิจัยจาก Stanford (Goodfellow, Bengio) พบว่าการฝึกโมเดลลึกโดยมี BatchNorm สามารถใช้ Dropout น้อยลงหรือไม่ต้องใช้เลยในบางสถานการณ์
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้ BatchNorm ใน PyTorch</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
`}
    </pre>

    <p>
      จากตัวอย่างจะเห็นว่ามีการแทรก BatchNorm ระหว่าง Convolution และ Activation ซึ่งสอดคล้องกับแนวทางปฏิบัติที่แนะนำโดยงานวิจัยล่าสุด
    </p>

    <h3 className="text-xl font-semibold">ข้อแนะนำเพิ่มเติม</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สำหรับ Transfer Learning ควรทดลอง Freeze หรือ Fine-Tune BatchNorm Parameters</li>
      <li>เมื่อทำ Mixed Precision Training ควรใช้ BatchNorm ที่รองรับ FP16 เช่น NVIDIA Apex</li>
      <li>ในงานที่ต้องการ Real-Time Inference ควรใช้ Running Statistics ที่บันทึกไว้แทนการคำนวณใหม่ทุกครั้ง</li>
    </ul>

    <h3 className="text-xl font-semibold">สรุป</h3>
    <p>
      Batch Normalization เป็นเทคนิคที่เปลี่ยนโฉมวงการ Deep Learning อย่างแท้จริง การใช้ BatchNorm อย่างเหมาะสมช่วยเร่งความเร็วการฝึกโมเดล เพิ่มเสถียรภาพ และลดความไวต่อ Hyperparameters แต่ต้องใช้ด้วยความระมัดระวังโดยคำนึงถึงขนาด Batch Size, ตำแหน่งการวางในโมเดล, และผลกระทบต่อกระบวนการฝึกทั้งหมด
    </p>
  </div>
</section>

<section id="batchnorm-limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. ปัญหาของ Batch Normalization</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img7} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Batch Normalization (BN) ได้นำความก้าวหน้าที่สำคัญมาให้กับการฝึกโมเดล Deep Learning ขนาดใหญ่ โดยเฉพาะในด้านการเร่งความเร็วการเรียนรู้และการทำให้การฝึกเสถียรมากขึ้น อย่างไรก็ตาม BN มีข้อจำกัดที่ต้องเข้าใจและพิจารณาอย่างรอบคอบเพื่อให้การออกแบบระบบ Deep Learning มีประสิทธิภาพสูงสุด
    </p>

    <h3 className="text-xl font-semibold">5.1 ความต้องการ Batch Size ขนาดใหญ่</h3>
    <p>
      การทำงานของ BN ต้องการการประมาณค่า Mean และ Variance อย่างแม่นยำจากตัวอย่างใน Mini-batch ดังนั้นถ้า Batch Size มีขนาดเล็กเกินไป เช่น &lt; 32 ตัวอย่าง การประมาณนี้จะมีความไม่แน่นอนสูง ส่งผลให้การฝึกเกิดความไม่เสถียรหรือแม้กระทั่งไม่ลู่เข้า (Non-convergence)
    </p>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-lg">
        <h4 className="font-semibold mb-2">Batch Size ใหญ่</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Mean และ Variance มีความแม่นยำสูง</li>
          <li>ช่วยเร่งการลู่เข้าอย่างมีเสถียรภาพ</li>
          <li>เหมาะสำหรับงานที่ใช้ GPU/TPU ที่มีหน่วยความจำมาก</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-lg">
        <h4 className="font-semibold mb-2">Batch Size เล็ก</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>การประมาณ Mean และ Variance มี noise สูง</li>
          <li>ทำให้ Loss curve มีการแกว่งมาก</li>
          <li>เสี่ยงต่อการ Divergence ในช่วงฝึก</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">5.2 Dependence ข้ามตัวอย่างใน Batch</h3>
    <p>
      BN ใช้ข้อมูลจากตัวอย่างหลายตัวเพื่อคำนวณสถิติในแต่ละเลเยอร์ ซึ่งนำไปสู่ Dependence ระหว่างตัวอย่างภายใน Batch เดียวกัน สิ่งนี้ขัดแย้งกับสมมุติฐานของ Independence ของข้อมูล โดยเฉพาะในงานเช่น Reinforcement Learning หรือ Online Learning ที่ข้อมูลมาแบบสตรีมทีละตัวอย่าง
    </p>

    <h3 className="text-xl font-semibold">5.3 ความยุ่งยากในงาน Sequence Modeling</h3>
    <p>
      ในงานที่ใช้โมเดลประเภท Sequential เช่น RNN หรือ LSTM การใช้ BN กลายเป็นเรื่องท้าทาย เพราะ Sequence Data มีความยาวไม่แน่นอน และการสุ่ม Batch อาจตัดลำดับข้อมูลจนเสียความสัมพันธ์ทางเวลา
    </p>
    <p>
      Layer Normalization (LN) จึงมักถูกเลือกแทนในงานประเภทนี้ เนื่องจาก LN ไม่พึ่งพา Batch Size แต่คำนวณ Mean และ Variance ต่อหนึ่งตัวอย่างโดยตรง
    </p>

    <h3 className="text-xl font-semibold">5.4 ความไม่เหมาะสมใน Online Learning</h3>
    <p>
      BN ต้องการคำนวณค่าเฉลี่ยในระดับ Mini-batch ทำให้ไม่สามารถใช้งานตรง ๆ กับ Online Learning ที่มีข้อมูลเข้ามาทีละตัวได้ จำเป็นต้องพัฒนาเทคนิคพิเศษ เช่น Streaming Normalization หรือใช้ LayerNorm, GroupNorm แทน
    </p>

    <h3 className="text-xl font-semibold">5.5 ค่า Running Mean/Variance ไม่เหมาะกับทุกงาน</h3>
    <p>
      ใน Phase ของ Inference (Validation/Test) BN ใช้ค่า Running Mean และ Running Variance ที่ได้จาก Training ซึ่งอาจไม่สะท้อน Distribution ของข้อมูล Test เสมอไป โดยเฉพาะเมื่อ Distribution มีการเปลี่ยนแปลง เช่นใน Domain Adaptation หรือ Test Time Augmentation
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดแสดงปัญหาเมื่อใช้ BN กับ Batch Size เล็ก</h3>
    <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm">
{`import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

x_small_batch = torch.randn(2, 3, 32, 32)  # Batch Size = 2
output = model(x_small_batch)`}
    </pre>
    <p>
      ในตัวอย่างนี้ การใช้ Batch Size เพียง 2 ตัวอย่างทำให้ค่าที่ได้จาก BN มี noise สูงมาก และส่งผลต่อการเรียนรู้
    </p>

    <h3 className="text-xl font-semibold">ทางเลือกเพื่อแก้ไขข้อจำกัดของ BN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Layer Normalization (LN):</strong> ใช้ Mean/Variance ต่อหนึ่งตัวอย่าง แทนการใช้ข้ามตัวอย่าง</li>
      <li><strong>Group Normalization (GN):</strong> แบ่ง Feature Maps ออกเป็นกลุ่ม ๆ และทำ Normalization ภายในกลุ่ม</li>
      <li><strong>Instance Normalization (IN):</strong> ทำ Normalization สำหรับแต่ละตัวอย่างและแต่ละ Channel</li>
      <li><strong>Batch Renormalization (BRN):</strong> เทคนิคปรับปรุง BN ให้ทนต่อ Batch Size เล็ก</li>
    </ul>

    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-5 rounded-lg border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="font-semibold mb-2">Insight จากงานวิจัยชั้นนำ</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>
          งานวิจัยของ Ba, Kiros, และ Hinton (2016) แสดงให้เห็นว่า LayerNorm เหมาะกับ RNN และ Transformer มากกว่า BN
        </li>
        <li>
          งานของ Wu และ He (2018) นำเสนอ GroupNorm เป็นทางเลือกที่มีเสถียรภาพใน Batch ขนาดเล็ก
        </li>
        <li>
          งานของ Ioffe (2017) เสนอ Batch Renormalization เพื่อลด Dependence ต่อ Batch Size
        </li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      แม้ Batch Normalization จะเป็นหนึ่งในนวัตกรรมสำคัญที่สุดของ Deep Learning แต่การเข้าใจข้อจำกัดเหล่านี้ช่วยให้ออกแบบระบบที่มีความเหมาะสมกับข้อมูลและโมเดลในแต่ละบริบทมากขึ้น การเลือกเทคนิค Normalization ต้องพิจารณาทั้งด้านลักษณะข้อมูล, ขนาด Batch, และรูปแบบของโมเดล เพื่อให้การเรียนรู้มีความเสถียร รวดเร็ว และมีประสิทธิภาพสูงสุด
    </p>
  </div>
</section>

<section id="layer-normalization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Layer Normalization (LN)</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img8} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Layer Normalization (LN) คือเทคนิคการทำ Normalization ที่ออกแบบมาเพื่อแก้ไขข้อจำกัดของ Batch Normalization โดยเฉพาะในกรณีที่ Batch Size มีขนาดเล็ก หรือในงานที่ต้องการประมวลผลลำดับข้อมูล เช่น Recurrent Neural Networks (RNNs) และ Transformers ซึ่ง BatchNorm ทำงานได้ยากเพราะ Batch Statistic ไม่เสถียร
    </p>

    <h3 className="text-xl font-semibold">หลักการทำงานของ Layer Normalization</h3>
    <p>
      แตกต่างจาก Batch Normalization ที่ทำ Normalization ข้ามตัวอย่างหลายตัวใน Batch เดียวกัน Layer Normalization ทำ Normalization ภายในตัวอย่างเดียว โดยพิจารณา mean และ variance ของ feature dimensions ทั้งหมดในตัวอย่างนั้น
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`\\( \\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\)`}
</pre>

      <p className="text-sm mt-2">
        โดยที่ \( \mu \) และ \( \sigma^2 \) คำนวณจาก feature ภายในตัวอย่างเดียว
      </p>
    </div>

    <p>
      หลังการ Normalize จะมีการเรียนรู้พารามิเตอร์ Scale (γ) และ Shift (β) เพื่อคงความสามารถในการแทนข้อมูล:
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
    <code className="text-sm block">
  {"\\[ y = \\gamma \\hat{x} + \\beta \\]"}
</code>

    </div>

    <h3 className="text-xl font-semibold">ข้อดีหลักของ Layer Normalization</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ไม่ขึ้นอยู่กับ Batch Size ทำงานได้แม้ batch = 1</li>
      <li>เพิ่มเสถียรภาพของการฝึกใน Sequential Models เช่น RNNs และ Transformers</li>
      <li>ทำให้การเรียนรู้ในแต่ละตัวอย่างมีอิสระมากขึ้น</li>
      <li>ช่วยลด Internal Covariate Shift ในระดับ feature</li>
    </ul>


    <h3 className="text-xl font-semibold">การประยุกต์ใช้ Layer Normalization</h3>
    <p>
      Layer Normalization ถูกนำมาใช้อย่างแพร่หลายในสถาปัตยกรรมที่เป็น Sequential เช่น:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Transformer Models:</strong> เช่น BERT, GPT series ใช้ LN หลังทุก Sub-layer</li>
      <li><strong>Recurrent Neural Networks (RNNs):</strong> ปรับเสถียรภาพในระหว่างลำดับยาว ๆ</li>
      <li><strong>Reinforcement Learning Agents:</strong> ใช้ในสถานการณ์ที่ batch มีขนาดเล็ก</li>
    </ul>

    <h3 className="text-xl font-semibold">สูตรคำนวณ Mean และ Variance</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
    <pre className="text-sm overflow-x-auto">
{`Mean (\\mu) = \\frac{1}{H} \\sum_{i=1}^{H} x_i
Variance (\\sigma^2) = \\frac{1}{H} \\sum_{i=1}^{H} (x_i - \\mu)^2`}
</pre>

    </div>

    <p>
      โดยที่ H คือจำนวน feature ภายในตัวอย่างนั้น ๆ การคำนวณนี้ทำให้ LN ไม่มีการผูกมัดกับ Batch Statistic ทำให้เหมาะกับการประมวลผลแบบ online และลำดับข้อมูล
    </p>

    <h3 className="text-xl font-semibold">เปรียบเทียบ BatchNorm และ LayerNorm</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border-collapse border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-800">
            <th className="border px-4 py-2">Aspect</th>
            <th className="border px-4 py-2">BatchNorm</th>
            <th className="border px-4 py-2">LayerNorm</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Batch Size Dependency</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ไม่มี</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">เหมาะกับ CNN</td>
            <td className="border px-4 py-2">มาก</td>
            <td className="border px-4 py-2">น้อย</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">เหมาะกับ RNN/Transformer</td>
            <td className="border px-4 py-2">น้อย</td>
            <td className="border px-4 py-2">มาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Regularization Effect</td>
            <td className="border px-4 py-2">มี</td>
            <td className="border px-4 py-2">น้อย</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้ LayerNorm ใน PyTorch</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
      <pre className="text-sm overflow-x-auto">
{`import torch
import torch.nn as nn

# ตัวอย่าง: สร้าง LayerNorm สำหรับ Vector ขนาด 128
layer_norm = nn.LayerNorm(128)

# ข้อมูลตัวอย่าง
x = torch.randn(32, 128)  # batch size 32, feature size 128

# Apply LayerNorm
output = layer_norm(x)
print(output.shape)  # (32, 128)`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">Insight Box: ข้อสังเกตเชิงลึก</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>LN ช่วยลด Sharp Minima ทำให้การเรียนรู้มีเสถียรภาพมากขึ้นในโมเดลขนาดใหญ่</li>
        <li>LN ทำให้ hidden states ใน RNNs มีการกระจายที่สม่ำเสมอขึ้น</li>
        <li>งานเช่น "Attention is All You Need" ใช้ LN หลังทุก Sub-layer ของ Transformer</li>
        <li>LN ช่วยลดการพึ่งพา Initialization และ Batch Statistic</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ Layer Normalization</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ในการประมวลผลภาพ LN ไม่ได้ผลดีเท่า BN เนื่องจาก Spatial Correlation สูง</li>
      <li>LN เพิ่มภาระการคำนวณเล็กน้อยเมื่อเทียบกับ BN ในบางกรณี</li>
    </ul>

    <p>
      การเลือกใช้ Layer Normalization จึงควรพิจารณาตามบริบทของงานและลักษณะของข้อมูลเสมอ
    </p>
  </div>
</section>


<section id="bn-vs-ln" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. เปรียบเทียบ BatchNorm vs LayerNorm</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img9} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Batch Normalization (BatchNorm) และ Layer Normalization (LayerNorm) เป็นเทคนิคการทำ Normalization ที่สำคัญใน Neural Networks ซึ่งถูกพัฒนาขึ้นเพื่อแก้ไขปัญหา Internal Covariate Shift และปรับปรุงเสถียรภาพของการฝึกโมเดล เทคนิคทั้งสองมีลักษณะการทำงานที่แตกต่างกันและเหมาะสมกับประเภทของโมเดลและสถานการณ์ที่แตกต่างกันด้วย
    </p>

    <h3 className="text-xl font-semibold">หลักการทำงาน</h3>
    <p>
      BatchNorm ดำเนินการ Normalize Activation ของแต่ละ Feature ตามสถิติของ Mini-Batch ทั้ง Batch โดยคำนวณ Mean และ Variance ข้ามตัวอย่างภายใน Batch เดียวกัน ในขณะที่ LayerNorm ทำ Normalization บน Feature ของแต่ละตัวอย่างอย่างอิสระ โดยใช้ Mean และ Variance จาก Feature ภายในตัวอย่างนั้นเพียงตัวเดียว โดยไม่ขึ้นกับ Batch Size
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg">
        <h4 className="font-semibold mb-2">Batch Normalization</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>Normalize Across Batch Dimension</li>
          <li>ต้องการ Batch Size ที่ใหญ่พอเพื่อคำนวณสถิติได้แม่นยำ</li>
          <li>เหมาะกับ Convolutional Neural Networks (CNNs)</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg">
        <h4 className="font-semibold mb-2">Layer Normalization</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>Normalize Across Feature Dimension ภายในตัวอย่างเดียว</li>
          <li>ไม่ขึ้นอยู่กับ Batch Size</li>
          <li>เหมาะกับ Recurrent Neural Networks (RNNs) และ Transformers</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">สูตรคำนวณ</h3>
    <p className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-sm">
  BatchNorm: {"\\( \\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}} \\gamma + \\beta \\)"}
  <br />
  LayerNorm: {"\\( \\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta \\)"}
</p>


    <h3 className="text-xl font-semibold">ข้อดีและข้อจำกัด</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-green-50 dark:bg-green-900 p-6 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อดีของ BatchNorm</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>เร่งการลู่เข้าของโมเดลอย่างมีนัยสำคัญ</li>
          <li>ช่วยลดความจำเป็นในการตั้งค่า Learning Rate อย่างแม่นยำ</li>
          <li>มีผลเหมือน Regularization เล็กน้อย ช่วยลด Overfitting</li>
        </ul>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-6 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อจำกัดของ BatchNorm</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ไม่เหมาะกับสถานการณ์ที่ Batch Size เล็กมาก</li>
          <li>ขึ้นอยู่กับลำดับของตัวอย่างใน Batch</li>
          <li>ทำให้การใช้ใน Online Learning และ RNN มีความยุ่งยาก</li>
        </ul>
      </div>
    </div>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-green-50 dark:bg-green-900 p-6 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อดีของ LayerNorm</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ทำงานได้ดีแม้ Batch Size เล็กหรือเท่ากับ 1</li>
          <li>เหมาะสมกับ Sequential Models เช่น RNN และ Transformer</li>
          <li>ไม่ขึ้นอยู่กับลำดับตัวอย่างใน Batch</li>
        </ul>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-6 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อจำกัดของ LayerNorm</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>อาจไม่สามารถเพิ่มประสิทธิภาพได้ดีเท่า BatchNorm ใน CNNs</li>
          <li>มี Regularization Effect น้อยกว่าจาก Noise ภายใน Batch</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งาน</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-lg shadow border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-2">CNN (Convolutional Neural Network)</h4>
        <p className="text-sm">นิยมใช้ BatchNorm หลัง Convolution Layer เพื่อลด Internal Covariate Shift และเร่งการฝึกโมเดล</p>
      </div>
      <div className="bg-white dark:bg-gray-900 p-6 rounded-lg shadow border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-2">Transformer</h4>
        <p className="text-sm">นิยมใช้ LayerNorm ก่อนการคำนวณ Attention และ Feedforward เพื่อรักษาเสถียรภาพในแต่ละ Step</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-6 rounded-lg border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>BatchNorm เหมาะสมกับโครงข่ายเชิงภาพและการใช้ Batch ใหญ่</li>
        <li>LayerNorm เหมาะสำหรับงาน NLP, Time-Series และโครงข่ายลำดับ</li>
        <li>สำหรับ Tiny Batch (เช่น Batch Size น้อยกว่า 8) ควรเลือก LayerNorm หรือ GroupNorm</li>
        <li>BatchNorm สามารถช่วย Smooth Loss Landscape ซึ่งช่วยให้ Optimization ง่ายขึ้น</li>
        <li>การใช้ Pre-Activation ResNet มีการวาง BatchNorm ก่อน Nonlinearity เพื่อผลลัพธ์ที่เสถียรกว่า</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      BatchNorm และ LayerNorm มีบทบาทสำคัญในการเพิ่มเสถียรภาพและเร่งความเร็วการฝึก Neural Networks การเลือกใช้งานควรขึ้นอยู่กับประเภทของข้อมูลและสถาปัตยกรรมโมเดล เช่น CNNs มักได้ประโยชน์สูงสุดจาก BatchNorm ในขณะที่ RNNs และ Transformers จะได้ประสิทธิภาพที่ดีกว่าจาก LayerNorm การเข้าใจข้อดีข้อจำกัดของแต่ละเทคนิคมีผลอย่างมากต่อความสำเร็จของการฝึกโมเดลในงานจริง
    </p>
  </div>
</section>


<section id="other-normalizations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Other Normalization Techniques</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img10} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      นอกจาก Batch Normalization และ Layer Normalization ที่ถูกใช้อย่างแพร่หลายแล้ว ยังมีเทคนิคการทำ Normalization อื่น ๆ ที่พัฒนาขึ้นเพื่อตอบโจทย์ข้อจำกัดในงานลักษณะเฉพาะต่าง ๆ การเลือกใช้เทคนิคเหล่านี้สามารถเพิ่มประสิทธิภาพของโมเดลได้อย่างมีนัยสำคัญโดยเฉพาะในเครือข่ายที่ลึก หรือสถานการณ์ที่มี batch size เล็กมาก
    </p>

    <h3 className="text-xl font-semibold">8.1 Instance Normalization (IN)</h3>
    <p>
      Instance Normalization เป็นเทคนิคที่ได้รับความนิยมในงาน Style Transfer โดยเฉพาะผลงานของ Ulyanov et al. (2016) ในงานวิจัย "Instance Normalization: The Missing Ingredient for Fast Stylization" ได้เสนอแนวทางการ Normalize activation ต่อหนึ่งตัวอย่างในแต่ละ channel อย่างเป็นอิสระกัน โดยไม่สนใจค่าเฉลี่ยหรือความแปรปรวนของ batch ทั้งหมด
    </p>
    <p>
      วิธีการนี้เหมาะสำหรับงานที่ต้องการคงโครงสร้างของสไตล์ภาพและลดการพึ่งพาค่าทางสถิติระหว่างตัวอย่างใน batch
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
    <code className="text-sm block">
  {"\\( \\hat{x}_{i,j,k} = \\frac{x_{i,j,k} - \\mu_{i,j}}{\\sqrt{\\sigma_{i,j}^2 + \\epsilon}} \\)"}
</code>

      <p className="text-sm mt-2">โดยที่ i คือ index ของตัวอย่าง, j คือ channel และ k คือ spatial location</p>
    </div>

    <h3 className="text-xl font-semibold">8.2 Group Normalization (GN)</h3>
    <p>
      Group Normalization ถูกนำเสนอโดย Yuxin Wu และ Kaiming He (Facebook AI Research) ในปี 2018 โดยมีจุดประสงค์เพื่อแก้ไขปัญหาของ BatchNorm เมื่อ batch size มีขนาดเล็ก GN ทำงานโดยการแบ่ง channel ออกเป็นหลายกลุ่ม (groups) แล้วทำ normalization ภายในแต่ละกลุ่มแทนที่จะทำบน batch ทั้งหมด
    </p>
    <p>
      GN ได้พิสูจน์ว่ามีประสิทธิภาพในงานที่ batch มีขนาดเล็ก เช่น Object Detection และ Video Recognition ซึ่งไม่สามารถใช้ BatchNorm ได้อย่างมีประสิทธิภาพ
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
    <code className="text-sm block">
  {"\\( \\hat{x} = \\frac{x - \\mu_{group}}{\\sqrt{\\sigma_{group}^2 + \\epsilon}} \\)"}
</code>

    </div>

    <div className="flex flex-col md:flex-row gap-6">
      <div className="flex-1 bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อดีของ Group Normalization</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ไม่ขึ้นกับ batch size</li>
          <li>เสถียรในงาน Computer Vision ขนาดใหญ่</li>
          <li>เหมาะกับ Distributed Training ที่ batch เล็ก</li>
        </ul>
      </div>
      <div className="flex-1 bg-red-100 dark:bg-red-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อจำกัดของ Group Normalization</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>อาจต้องจูนจำนวน groups ให้เหมาะสม</li>
          <li>การเลือก group size มีผลต่อประสิทธิภาพ</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">8.3 Weight Normalization (WN)</h3>
    <p>
      Weight Normalization ได้รับการเสนอโดย Tim Salimans และ Diederik Kingma (2016) เพื่อลดการพึ่งพาค่าการกระจายของ input โดยตรงที่ layer WN ทำการ Normalize weight vector เองแทนที่จะ Normalize activation
    </p>
    <p>
      แนวทางนี้ช่วยให้การ Optimization เร็วขึ้น และลด dependency กับ Initialization
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
    <code className="text-sm block">
  {"\\( \\mathbf{w} = \\frac{g}{\\| \\mathbf{v} \\|} \\mathbf{v} \\)"}
</code>

<p className="text-sm mt-2">
  {"โดยที่ \\( \\mathbf{w} \\) คือ normalized weight, \\( g \\) คือ scalar parameter, \\( \\mathbf{v} \\) คือ unconstrained weight vector"}
</p>

    </div>

    <h3 className="text-xl font-semibold">8.4 Switchable Normalization (SN)</h3>
    <p>
      Switchable Normalization พัฒนาขึ้นโดย Ping Luo และทีมงานจาก Chinese University of Hong Kong (2019) เพื่อรวมข้อดีของ BatchNorm, InstanceNorm และ LayerNorm เข้าด้วยกัน ผ่านการเรียนรู้พารามิเตอร์ที่เลือกวิธี normalization ที่เหมาะสมที่สุดระหว่างการฝึก
    </p>
    <p>
      วิธีนี้ช่วยให้โมเดลสามารถปรับตัวได้ดีขึ้นตามลักษณะของข้อมูลและงาน
    </p>

    <h3 className="text-xl font-semibold">8.5 เปรียบเทียบ Normalization Techniques</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-800">
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">ขึ้นกับ Batch Size</th>
            <th className="border px-4 py-2">เหมาะกับ</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">BatchNorm</td>
            <td className="border px-4 py-2">ใช่</td>
            <td className="border px-4 py-2">CNNs, Large Batch</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">LayerNorm</td>
            <td className="border px-4 py-2">ไม่ใช่</td>
            <td className="border px-4 py-2">RNNs, Transformers</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">InstanceNorm</td>
            <td className="border px-4 py-2">ไม่ใช่</td>
            <td className="border px-4 py-2">Style Transfer</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GroupNorm</td>
            <td className="border px-4 py-2">ไม่ใช่</td>
            <td className="border px-4 py-2">Small Batch CNNs</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">WeightNorm</td>
            <td className="border px-4 py-2">ไม่เกี่ยวข้อง</td>
            <td className="border px-4 py-2">Optimization Stability</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">8.6 Insight Box: ข้อสังเกตจากงานวิจัย</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>BatchNorm ยังคงเป็นตัวเลือกที่ดีที่สุดสำหรับ CNNs ขนาดใหญ่เมื่อ batch size เพียงพอ</li>
        <li>GroupNorm เป็นตัวเลือกที่น่าเชื่อถือในงานที่ batch size มีข้อจำกัด</li>
        <li>LayerNorm เป็นแกนสำคัญของสถาปัตยกรรม Transformer และรุ่นใหญ่เช่น GPT-3, BERT</li>
        <li>InstanceNorm ช่วยให้ Style Transfer ประสบความสำเร็จในการแยก content และ style</li>
        <li>WeightNorm ช่วยให้โมเดล convergent ได้เร็วกว่าบน tasks บางประเภทโดยเฉพาะใน RL</li>
      </ul>
    </div>
  </div>
</section>

<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Insight เชิงลึก</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img11} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <p>
      การทำ Normalization ใน Neural Networks ไม่เพียงแค่ช่วยเร่งความเร็วในการฝึกและทำให้การเรียนรู้เสถียรมากขึ้นเท่านั้น แต่ยังมีผลเชิงลึกต่อ Landscape ของ Optimization และความสามารถในการ Generalization ของโมเดลอย่างมีนัยสำคัญ งานวิจัยของ Sergey Ioffe และ Christian Szegedy (2015) ที่นำเสนอแนวคิด Batch Normalization ได้เปลี่ยนแปลงแนวทางการฝึกโมเดลลึกอย่างสิ้นเชิง และกลายเป็นมาตรฐานในหลายโมเดลระดับโลก เช่น ResNet, Inception, Transformer
    </p>

    <h3 className="text-xl font-semibold">ผลกระทบของ Batch Normalization ต่อ Loss Landscape</h3>
    <p>
      งานวิจัยจาก MIT CSAIL และ Google Brain พบว่าการใช้ Batch Normalization ช่วยให้ Surface ของ Loss Function มีความ Smooth มากขึ้น ลดปัญหา Sharp Minima ซึ่งส่งผลให้โมเดลสามารถค้นหา Minima ที่มีคุณสมบัติในการ Generalize ดีกว่าได้มากขึ้น (Santurkar et al., 2018)
    </p>

    <div className="flex justify-center my-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-6 rounded-xl border border-gray-300 dark:border-gray-700 max-w-md w-full">
        <h4 className="text-lg font-semibold mb-2">Key Effects</h4>
        <ul className="list-disc pl-6 text-sm">
          <li>ลดความชันที่รุนแรงในพื้นผิวของ Loss</li>
          <li>ทำให้การอัปเดตพารามิเตอร์มีเสถียรภาพมากขึ้น</li>
          <li>ช่วยให้ Optimizer ก้าวไปได้ไกลขึ้นในแต่ละรอบ</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Normalization กับการลดปัญหา Vanishing/Exploding Gradients</h3>
    <p>
      ในเครือข่ายลึกมาก ๆ เช่น 100+ layers การไหลของ Gradient มักจะอ่อนลงอย่างรวดเร็ว (Vanishing) หรือเพิ่มขึ้นจนเกินควบคุม (Exploding) การทำ Normalization ช่วยทำให้ค่า Activation อยู่ในช่วงที่เหมาะสมตลอดการฝึก จึงสามารถลดปัญหาเหล่านี้ได้อย่างมีประสิทธิภาพ ซึ่งเป็นสิ่งสำคัญต่อการฝึกโมเดลลึกเช่น ResNet, DenseNet
    </p>

    <h3 className="text-xl font-semibold">การทำงานร่วมกับ Optimizer สมัยใหม่</h3>
    <p>
      งานวิจัยจาก FAIR (Facebook AI Research) แสดงให้เห็นว่าการผสาน Batch Normalization เข้ากับ Optimizers แบบ Adaptive เช่น Adam หรือ RMSProp ช่วยเพิ่มอัตราการลู่เข้าได้ดีขึ้นอย่างมีนัยสำคัญ โดยเฉพาะในช่วงเริ่มต้นการฝึกที่ค่า Gradient มีความผันผวนสูง
    </p>

    <h3 className="text-xl font-semibold">LayerNorm และการขยายขอบเขตของ Normalization</h3>
    <p>
      การออกแบบ Layer Normalization (LN) เพื่อรองรับโครงสร้างเชิงลำดับ เช่น RNN และ Transformer ได้แสดงให้เห็นว่าการทำ Normalization ต่อ 1 ตัวอย่างเป็นแนวทางที่มีประสิทธิภาพมากเมื่อ batch size มีขนาดเล็กหรือเปลี่ยนแปลงตลอดเวลา (Ba et al., 2016)
    </p>

    <div className="grid md:grid-cols-2 gap-6 my-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-5 rounded-xl">
        <h4 className="font-semibold mb-2">ข้อดีของ LayerNorm</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ไม่ขึ้นกับขนาดของ Batch</li>
          <li>รองรับโมเดลที่ต้องการ Sequential Dependency</li>
          <li>เสถียรในการฝึกโมเดลขนาดใหญ่ เช่น Transformer, BERT</li>
        </ul>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-5 rounded-xl">
        <h4 className="font-semibold mb-2">ข้อจำกัดของ BatchNorm</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Batch Size เล็กเกินไปทำให้สถิติไม่แม่นยำ</li>
          <li>ขึ้นกับข้อมูล Batch อื่น ทำให้ไม่เหมาะกับ Online Learning</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">การประยุกต์ในโมเดลระดับโลก</h3>
    <p>
      ในงานเช่น Vision Transformer (ViT) และ GPT series ของ OpenAI มีการใช้ LayerNorm แทน BatchNorm เพื่อเพิ่มความเสถียรในโมเดลลำดับยาว นอกจากนี้ งานอย่าง StyleGAN2 ของ Nvidia ใช้ Adaptive Instance Normalization เพื่อเพิ่มคุณภาพการสร้างภาพ โดยแสดงให้เห็นว่า Normalization ไม่ใช่แค่เรื่องความเร็ว แต่มีผลต่อการควบคุมลักษณะข้อมูล (Content vs Style) อย่างลึกซึ้ง
    </p>

    <h3 className="text-xl font-semibold text-center">Insight เชิงลึกจากงานวิจัยปี 2023-2024</h3>
    <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img12} />
        </div>
    <ul className="list-disc pl-6 space-y-2">
      <li>งานของ Liu et al. (ICLR 2023) เสนอ Dynamic Normalization ที่เปลี่ยนแปลงตาม Training Phase ช่วยเร่งการลู่เข้าของ ViT ได้ถึง 30%</li>
      <li>งานจาก DeepMind เสนอ Flexible Normalization ซึ่งเลือกประเภท Normalization อัตโนมัติตามรูปแบบของ Feature Map</li>
      <li>งานของ Microsoft Research ชี้ว่าการเลือกใช้ Group Normalization (GN) ในโมเดลขนาดใหญ่มักให้ผลดีกว่า BN ในงานที่มี batch size เล็ก</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="text-lg font-semibold mb-2">Best Practices สรุป</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>เริ่มจาก BatchNorm เมื่อ batch size มีขนาดพอเหมาะ (มากกว่า 32)</li>
        <li>ใช้ LayerNorm ใน RNN, Transformer, หรือเมื่อ batch size เล็กมาก</li>
        <li>พิจารณาใช้ GroupNorm เมื่อทำ Distributed Training ขนาดใหญ่</li>
        <li>ระวังเรื่องสถิติ (mean, var) ที่เก็บใน BatchNorm เมื่อนำโมเดลไปใช้ inference</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">สรุปภาพรวม</h3>
    <p>
      การทำความเข้าใจ Normalization อย่างลึกซึ้งช่วยให้สามารถออกแบบ Neural Networks ที่มีประสิทธิภาพและเสถียรสูงได้ดีขึ้นในงานจริง ไม่ว่าจะเป็น Image Recognition, Natural Language Processing, Reinforcement Learning หรือ Generative Modeling โดยการเลือกใช้ Normalization ที่เหมาะสมกับโครงสร้างของข้อมูลและสถาปัตยกรรมของโมเดลยังคงเป็นปัจจัยสำคัญในการผลักดันขีดความสามารถของ AI ในอนาคต
    </p>

  </div>
</section>


<section id="special-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">Special Box: Best Practices</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h3 className="text-xl font-semibold mb-4">แนวทางปฏิบัติที่ดีที่สุดในการทำ Normalization ใน Neural Networks</h3>

      <h4 className="text-lg font-semibold mt-6">1. ใช้ Batch Normalization สำหรับ CNNs โดยเริ่มจาก Pre-Activation</h4>
      <p>
        การศึกษาจาก He et al. (2016) แนะนำให้แทรก Batch Normalization ก่อน Activation Functions เช่น ReLU แทนที่จะวางหลังจากนั้น การจัดวางในลักษณะ Pre-Activation ช่วยปรับปรุงเสถียรภาพของการเรียนรู้และเพิ่มความเร็วในการฝึกในเครือข่ายลึกมาก เช่น ResNet-152 และ DenseNet-201
      </p>

      <h4 className="text-lg font-semibold mt-6">2. ใช้ Layer Normalization สำหรับ Sequential Models</h4>
      <p>
        งานวิจัยจาก Ba, Kiros, และ Hinton (2016) ชี้ให้เห็นว่า Layer Normalization มีประสิทธิภาพสูงกว่า Batch Normalization ใน Recurrent Neural Networks (RNNs) และ Transformer Models เนื่องจากไม่มีการพึ่งพา batch size และสามารถทำงานได้อย่างมีเสถียรภาพแม้ในกรณีที่ batch มีขนาดเล็กหรือเท่ากับหนึ่ง
      </p>

      <h4 className="text-lg font-semibold mt-6">3. ใช้ Group Normalization เมื่อ Batch Size มีขนาดเล็กมาก</h4>
      <p>
        Wu และ He (2018) เสนอ Group Normalization (GN) ซึ่งแบ่ง feature maps ออกเป็นกลุ่มย่อย ๆ แล้วทำ normalization ภายในแต่ละกลุ่ม GN ทำงานได้ดีใน CNNs ที่ batch size เล็ก เช่น 16 หรือน้อยกว่า โดยไม่สูญเสียประสิทธิภาพเมื่อเทียบกับ Batch Normalization
      </p>

      <h4 className="text-lg font-semibold mt-6">4. ตั้งค่าพารามิเตอร์ gamma และ beta อย่างระมัดระวัง</h4>
      <p>
        ใน Batch Normalization และ Layer Normalization, พารามิเตอร์ gamma (scale) และ beta (shift) มีผลต่อการรักษาความสามารถในการเรียนรู้ของโมเดล ควรอนุญาตให้ gamma และ beta ถูกฝึก (trainable) และเริ่มต้น gamma ใกล้ 1.0 และ beta ใกล้ 0.0 เพื่อให้สัญญาณเริ่มต้นมีลักษณะไม่เปลี่ยนแปลงจากอินพุตมากนัก
      </p>

      <h4 className="text-lg font-semibold mt-6">5. ใช้ epsilon (𝜖) ที่ไม่ต่ำเกินไป</h4>
      <p>
        เพื่อป้องกันการหารด้วยศูนย์ในการทำ normalization ค่าของ epsilon ควรอยู่ในช่วงประมาณ 1e-5 ถึง 1e-3 ตามคำแนะนำจากเอกสาร TensorFlow และ PyTorch ค่า epsilon ที่ต่ำเกินไปอาจทำให้การคำนวณมีความไม่เสถียรโดยไม่จำเป็น
      </p>

      <h4 className="text-lg font-semibold mt-6">6. ไม่ควรใช้ Normalization บน Output Layer</h4>
      <p>
        ในการออกแบบโมเดลสำหรับงานเช่น Classification หรือ Regression ควรหลีกเลี่ยงการใช้ Normalization บน Output Layer เนื่องจากจะเปลี่ยนลักษณะการกระจายของค่าผลลัพธ์ที่ต้องการให้อยู่ในรูปแบบเฉพาะ เช่น Softmax หรือ Linear Activation
      </p>

      <h4 className="text-lg font-semibold mt-6">7. ตรวจสอบการทำงานร่วมกับ Dropout</h4>
      <p>
        การใช้ Batch Normalization ร่วมกับ Dropout ควรทำด้วยความระมัดระวัง โดยแนะนำให้นำ Batch Normalization มาใช้ก่อน Dropout Layer เพื่อหลีกเลี่ยงการสร้าง Noise เกินจำเป็นในสถิติของ Mean และ Variance ในระหว่างการฝึก
      </p>

      <h4 className="text-lg font-semibold mt-6">8. ใช้ Weight Initialization ที่สอดคล้องกับ Normalization</h4>
      <p>
        เมื่อมีการใช้ Batch Normalization, การใช้ Weight Initialization แบบ He Initialization หรือ Xavier Initialization ยังคงมีประโยชน์ แต่ความเข้มงวดในการตั้งค่าเริ่มต้นจะลดลง เพราะ Normalization จะช่วยรักษาการกระจายของสัญญาณเอาไว้ระหว่างเลเยอร์ได้
      </p>

      <h4 className="text-lg font-semibold mt-6">9. ใช้ Moving Average อย่างระมัดระวังใน BatchNorm</h4>
      <p>
        ในระหว่างการฝึก, BatchNorm จะเก็บค่า Moving Average ของ Mean และ Variance สำหรับใช้ในโหมด Inference การเลือก Momentum สำหรับการอัปเดต Moving Average (เช่น 0.9 หรือ 0.99) มีผลโดยตรงต่อเสถียรภาพในการทำ Inference ภายหลัง
      </p>

      <h4 className="text-lg font-semibold mt-6">10. ติดตามสถิติ BatchNorm ในโหมดฝึกและประเมิน</h4>
      <p>
        ในการใช้งานเฟรมเวิร์ก เช่น PyTorch หรือ TensorFlow ควรระวังการสลับโหมด (training vs evaluation mode) อย่างถูกต้อง โดย BatchNorm ต้องทำงานต่างกันในสองโหมดนี้ เพื่อให้โมเดลสามารถใช้ค่า Moving Average ได้ในระหว่างการประเมินผลจริง
      </p>

      <div className="bg-blue-50 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-400 dark:border-blue-600 mt-8">
        <h4 className="text-lg font-semibold">Insight Box: แนวทางเพิ่มเติมจากงานวิจัย</h4>
        <ul className="list-disc pl-6 text-sm space-y-2">
          <li>Research จาก Stanford แนะนำให้ใช้ LayerNorm สำหรับงาน Transformer-based NLP Models เช่น BERT และ GPT</li>
          <li>Google Brain แสดงให้เห็นว่า BatchNorm สามารถลด Sharp Minima ทำให้โมเดล Generalize ได้ดีขึ้น</li>
          <li>DeepMind พบว่าใน Reinforcement Learning, LayerNorm ช่วยเพิ่มความเสถียรในโมเดล Actor-Critic ได้</li>
          <li>Facebook FAIR แนะนำ GroupNorm สำหรับ CNN ในงาน Object Detection และ Instance Segmentation ที่ต้องการใช้ batch ขนาดเล็ก</li>
        </ul>
      </div>

    </div>

  </div>
</section>


        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day20 theme={theme} />
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
        <ScrollSpy_Ai_Day20 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day20_BatchLayerNormalization;
