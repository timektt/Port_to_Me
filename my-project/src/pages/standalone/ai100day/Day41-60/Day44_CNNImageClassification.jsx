import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day44 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day44";
import MiniQuiz_Day44 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day44";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day44_CNNImageClassification = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day44_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day44_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day44_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day44_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day44_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day44_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day44_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day44_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day44_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day44_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day44_11").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 44: CNN for Image Classification</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

     <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ภาพหนึ่งภาพ เล่าข้อมูลพันคำได้อย่างไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="prose prose-lg max-w-none dark:prose-invert text-base leading-relaxed space-y-10">

    <p>
      การประมวลผลภาพ (Image Processing) คือศาสตร์ที่ใช้ข้อมูลภาพเป็นอินพุตเพื่อให้ระบบสามารถวิเคราะห์ จำแนก และตัดสินใจได้โดยอัตโนมัติ ซึ่ง Convolutional Neural Networks (CNNs) ได้กลายมาเป็นสถาปัตยกรรมหลักในการทำความเข้าใจภาพระดับลึก โดยเฉพาะในงานจำแนกภาพ (Image Classification) ซึ่งถือเป็นพื้นฐานของหลายระบบ AI ในยุคปัจจุบัน เช่น รถยนต์ไร้คนขับ, ระบบวินิจฉัยโรคจากภาพทางการแพทย์, และระบบแนะนำสินค้า
    </p>

    <h3>ภาพดิจิทัลในมุมมองของโมเดลเชิงคณิตศาสตร์</h3>
    <p>
      สำหรับโมเดลเชิงคณิตศาสตร์ ภาพถูกแทนด้วยเมทริกซ์ของค่าพิกเซล เช่น ภาพ RGB ขนาด 224x224 จะกลายเป็นเทนเซอร์ 3 มิติ ขนาด 224x224x3 ซึ่งทำให้สามารถใช้ Linear Algebra และการคำนวณแบบ Tensor เพื่อประมวลผลได้อย่างมีประสิทธิภาพ โดยเฉพาะผ่านฟังก์ชัน Convolution ที่ออกแบบมาเพื่อจับ pattern ภายในภาพ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded">
      <p className="font-semibold text-yellow-800">Insight Box:</p>
      <p>
        การแทนภาพเป็น Tensor ทำให้โมเดล Deep Learning สามารถเชื่อมโยงแนวคิดทางคณิตศาสตร์เข้ากับความเข้าใจในระดับมนุษย์ เช่น การรับรู้วัตถุ การจำแนกประเภท และการตีความเชิงนามธรรม ซึ่งสะท้อนผ่าน CNN ได้โดยตรง
      </p>
    </div>

    <h3>การสื่อสารเชิงนามธรรมผ่าน Feature</h3>
    <p>
      CNN สามารถแยกแยะและเรียนรู้ Feature ที่มีลำดับขั้นจากระดับพื้นฐานถึงนามธรรม เช่น:
    </p>
    <ul className="list-disc pl-6">
      <li>เลเยอร์ต้นเรียนรู้ edge, corner, และ texture</li>
      <li>เลเยอร์กลางแยกรูปทรง هندเรขาคณิต หรือ ลวดลาย</li>
      <li>เลเยอร์ลึกเข้าใจบริบทของภาพ เช่น วัตถุ หรืออารมณ์</li>
    </ul>

    <h3>เปรียบเทียบการตีความระหว่างมนุษย์กับ CNN</h3>
    <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-600">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">ระดับการรับรู้</th>
          <th className="border px-4 py-2">มนุษย์</th>
          <th className="border px-4 py-2">CNN</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">พื้นฐาน</td>
          <td className="border px-4 py-2">ขอบ, ความคม</td>
          <td className="border px-4 py-2">Edge Detection</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">กลาง</td>
          <td className="border px-4 py-2">รูปร่าง, ลวดลาย</td>
          <td className="border px-4 py-2">Shape/Texture Encoding</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">การตีความวัตถุ</td>
          <td className="border px-4 py-2">Object Classification</td>
        </tr>
      </tbody>
    </table>

    <h3>ความสำคัญของ CNN ใน Image Classification</h3>
    <p>
      ก่อนยุคของ CNN การจำแนกภาพใช้วิธี handcrafted feature เช่น SIFT หรือ HOG ร่วมกับ SVM หรือ Random Forest ซึ่งมักมีข้อจำกัดด้านความสามารถในการ generalize เมื่อเจอภาพใหม่ CNN ช่วยลดภาระนี้โดยให้โมเดลเรียนรู้ Feature โดยอัตโนมัติ ซึ่งถูกยืนยันในงานวิจัยระดับโลก เช่น LeNet, AlexNet, และ ResNet
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold text-blue-800">Highlight:</p>
      <p>
        งานของ Krizhevsky et al. (2012) ใน AlexNet เปลี่ยนภูมิทัศน์ของการจำแนกภาพ ด้วยการใช้ CNN แบบลึกบน ImageNet ซึ่งลด error rate ได้มากกว่า 10% เมื่อเทียบกับวิธีเดิม
      </p>
    </div>

    <h3>แนวโน้มและความท้าทาย</h3>
    <p>
      แม้ CNN จะเป็นแกนหลักของ Image Classification แต่แนวโน้มใหม่เช่น Vision Transformers (ViT) และการรวม Attention Mechanism เข้ามาในระบบภาพกำลังเปลี่ยนแปลงรูปแบบเดิม โดยการวิจัยต่อไปจะเน้นไปที่:
    </p>
    <ul className="list-disc pl-6">
      <li>การลดพารามิเตอร์และเพิ่มความเร็วในการประมวลผล</li>
      <li>ความสามารถในการตีความ feature ที่ลึกขึ้นและกว้างขึ้น</li>
      <li>การออกแบบสถาปัตยกรรมแบบ hybrid ระหว่าง CNN กับ Self-Attention</li>
    </ul>

    <h3>อ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NIPS*</li>
      <li>LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*</li>
      <li>Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*</li>
    </ul>

  </div>
</section>


        <section id="cnn-structure" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. โครงสร้างพื้นฐานของ CNN สำหรับ Image Classification</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>ภาพรวมของ Convolutional Neural Network</h3>
    <p>
      โครงสร้างพื้นฐานของ Convolutional Neural Network (CNN) สำหรับ Image Classification ได้รับแรงบันดาลใจมาจากกระบวนการมองเห็นของระบบประสาทในมนุษย์ โดยเลเยอร์ต่าง ๆ จะทำหน้าที่ดึงข้อมูลจากภาพแบบลำดับชั้น ตั้งแต่ขอบและพื้นผิว ไปจนถึงวัตถุที่ซับซ้อนขึ้นเรื่อย ๆ
    </p>

    <h3>เลเยอร์หลักของ CNN</h3>
    <ul className="list-disc pl-6">
      <li>
        <strong>Convolutional Layer:</strong> ทำหน้าที่ในการสกัด feature จากภาพต้นฉบับโดยใช้ kernel ที่เลื่อนไปทั่วภาพ เพื่อดึงข้อมูลเชิงพื้นที่ เช่น ลวดลาย ขอบ และรูปร่าง
      </li>
      <li>
        <strong>Activation Layer:</strong> มักใช้ฟังก์ชัน ReLU เพื่อเพิ่ม non-linearity ให้กับระบบ
      </li>
      <li>
        <strong>Pooling Layer:</strong> ลดขนาด spatial dimension และช่วยลดความซ้ำซ้อนของข้อมูล เช่น Max Pooling, Average Pooling
      </li>
      <li>
        <strong>Fully Connected Layer:</strong> เชื่อมโยงข้อมูลทั้งหมดเพื่อทำการคาดการณ์ (classification)
      </li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded-md">
      <p className="text-sm text-yellow-800">
        <strong>Insight:</strong> สถาปัตยกรรมของ CNN ถูกออกแบบให้เหมาะสมกับลักษณะของข้อมูลภาพ ซึ่งแตกต่างจาก MLP ทั่วไปที่ไม่สามารถรับข้อมูลเชิง spatial ได้อย่างมีประสิทธิภาพ
      </p>
    </div>

    <h3>สถาปัตยกรรมที่เรียบง่ายของ CNN</h3>
    <pre className="bg-gray-600 text-sm overflow-x-auto p-4 rounded-md">
{`
Input Image (32x32x3)
  ↓
Conv Layer (5x5 filters, stride 1, padding 2) → Output: 32x32x16
  ↓
ReLU Activation
  ↓
Max Pooling (2x2) → Output: 16x16x16
  ↓
Conv Layer (5x5 filters) → Output: 16x16x32
  ↓
ReLU Activation
  ↓
Max Pooling (2x2) → Output: 8x8x32
  ↓
Flatten
  ↓
Fully Connected Layer → Output: 64
  ↓
Softmax → Output: 10 classes
`}
    </pre>

    <h3>ข้อดีของโครงสร้าง CNN</h3>
    <ul className="list-disc pl-6">
      <li>สามารถสกัด feature ได้โดยไม่ต้องใช้ handcrafted feature</li>
      <li>มีพารามิเตอร์น้อยเมื่อเทียบกับ MLP เนื่องจากการใช้ weight sharing</li>
      <li>ปรับขนาดภาพได้ดีเนื่องจากมี layer ลด dimension</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded-md">
      <p className="text-sm text-blue-900">
        <strong>Highlight:</strong> หนึ่งในข้อได้เปรียบของ CNN คือความสามารถในการเรียนรู้ feature ที่มีลำดับชั้นจากข้อมูลภาพ โดยไม่ต้องใช้ feature engineer แบบดั้งเดิม
      </p>
    </div>

    <h3>ตารางเปรียบเทียบระหว่าง MLP และ CNN</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 text-sm text-left">
        <thead className="bg-gray-200">
          <tr>
            <th className="border px-4 py-2">ลักษณะ</th>
            <th className="border px-4 py-2">MLP</th>
            <th className="border px-4 py-2">CNN</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Input Shape</td>
            <td className="border px-4 py-2">Flatten vector</td>
            <td className="border px-4 py-2">Image matrix</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Weight Sharing</td>
            <td className="border px-4 py-2">ไม่มี</td>
            <td className="border px-4 py-2">มี</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">เหมาะกับข้อมูลภาพ</td>
            <td className="border px-4 py-2">ไม่ดี</td>
            <td className="border px-4 py-2">ดีมาก</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning Lecture Series (2021)</li>
      <li>Goodfellow, I., Bengio, Y., Courville, A. Deep Learning (MIT Press)</li>
      <li>LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"</li>
    </ul>
  </div>
</section>


     <section id="cnn-architectures" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. ตัวอย่าง CNN Architectures ที่ใช้จริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>Overview of Historical CNN Models</h3>
    <p>
      ตั้งแต่ปี 1998 ที่ Yann LeCun นำเสนอ LeNet-5 เป็นสถาปัตยกรรมพื้นฐานสำหรับการรู้จำตัวเลขจากภาพ การพัฒนา Convolutional Neural Networks (CNNs) ได้ก้าวหน้ามาอย่างต่อเนื่อง โดยมีโมเดลที่สำคัญต่อวงการหลายรุ่น
    </p>
    <ul>
      <li><strong>LeNet-5:</strong> โครงสร้างขนาดเล็ก มีเพียง 7 เลเยอร์ ใช้กับตัวเลข MNIST</li>
      <li><strong>AlexNet (2012):</strong> โมเดล CNN แรกที่ชนะการแข่งขัน ILSVRC ด้วยความแม่นยำสูงกว่าระบบเดิมอย่างมาก</li>
      <li><strong>VGGNet (2014):</strong> ใช้การเพิ่มความลึกโดยคงโครงสร้างแบบง่าย (3x3 filters)</li>
      <li><strong>GoogLeNet/Inception (2014):</strong> แนะนำแนวคิด inception module เพื่อรวมฟีเจอร์หลายระดับ</li>
      <li><strong>ResNet (2015):</strong> โมเดลที่แนะนำ residual connection เพื่อสร้างโครงข่ายลึกมากกว่า 600 เลเยอร์</li>
    </ul>

    <h3>ตารางเปรียบเทียบ Architectures</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 text-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="px-4 py-2 text-left">Architecture</th>
            <th className="px-4 py-2 text-left">Year</th>
            <th className="px-4 py-2 text-left">Top-5 Accuracy (ILSVRC)</th>
            <th className="px-4 py-2 text-left">Key Innovation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">LeNet-5</td>
            <td className="border px-4 py-2">1998</td>
            <td className="border px-4 py-2">N/A</td>
            <td className="border px-4 py-2">พื้นฐานสำหรับการแยกเลข</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">AlexNet</td>
            <td className="border px-4 py-2">2012</td>
            <td className="border px-4 py-2">84.7%</td>
            <td className="border px-4 py-2">ReLU + Dropout + GPU Training</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">VGGNet</td>
            <td className="border px-4 py-2">2014</td>
            <td className="border px-4 py-2">92.7%</td>
            <td className="border px-4 py-2">Simplicity + Deep Layers</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GoogLeNet</td>
            <td className="border px-4 py-2">2014</td>
            <td className="border px-4 py-2">93.3%</td>
            <td className="border px-4 py-2">Inception Module</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ResNet</td>
            <td className="border px-4 py-2">2015</td>
            <td className="border px-4 py-2">96.4%</td>
            <td className="border px-4 py-2">Residual Connections</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-600 border-l-4 border-blue-500 text-blue-900 p-4 rounded">
      <p className="font-semibold">Insight:</p>
      <p>
        ความสำเร็จของ CNN ส่วนหนึ่งมาจากการพัฒนาโครงสร้างที่ตอบโจทย์การเรียนรู้ฟีเจอร์ที่ลึกและมีความซับซ้อนมากขึ้นอย่างเป็นระบบ
      </p>
    </div>

    <h3>แนวโน้มใหม่ใน CNN Architectures</h3>
    <p>
      โมเดล CNN สมัยใหม่เช่น EfficientNet ใช้แนวคิด compound scaling เพื่อปรับขนาด depth, width และ resolution อย่างสมดุล นอกจากนี้ยังมี ConvNeXt ที่นำแนวคิดจาก Vision Transformer มาใช้ปรับ CNN ให้ทันสมัยมากขึ้น
    </p>
    <ul>
      <li><strong>EfficientNet:</strong> พัฒนาโดย Google AI, ใช้ compound scaling</li>
      <li><strong>ConvNeXt:</strong> ปรับปรุง CNN ให้เทียบเท่า Vision Transformers</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 text-yellow-900 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        แม้ CNN จะถูกท้าทายจาก Vision Transformer (ViT) แต่โครงสร้างที่เรียบง่ายและความเร็วในการ inference ยังทำให้ CNN เป็นตัวเลือกสำคัญในหลาย use case
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul>
      <li>LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proc. IEEE, 1998</li>
      <li>Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS 2012</li>
      <li>Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556</li>
      <li>Szegedy et al., "Going Deeper with Convolutions," CVPR 2015</li>
      <li>He et al., "Deep Residual Learning for Image Recognition," CVPR 2016</li>
      <li>Tan and Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019</li>
    </ul>
  </div>
</section>


     <section id="loss-metrics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Loss Function และ Evaluation Metrics</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>ความสำคัญของ Loss Function ใน Deep Learning</h3>
    <p>
      Loss function เป็นกลไกสำคัญในการวัดความผิดพลาดระหว่างค่าที่โมเดลทำนายกับค่าความจริง โดยมีบทบาทในการกำหนดทิศทางของการเรียนรู้ผ่านกระบวนการ backpropagation หากไม่มี loss function ระบบ deep learning จะไม่สามารถประเมินได้ว่าควรปรับ weight ไปในทิศทางใดเพื่อให้ผลลัพธ์ดีขึ้น
    </p>

    <h3>ประเภทของ Loss Function ที่ใช้ในงาน Classification</h3>
    <ul>
      <li><strong>Cross-Entropy Loss</strong>: เป็น loss function พื้นฐานในการจำแนกประเภท โดยเฉพาะใน softmax output</li>
      <li><strong>Hinge Loss</strong>: มักใช้ใน SVM แต่สามารถนำมาประยุกต์กับ deep networks ได้</li>
      <li><strong>Focal Loss</strong>: ออกแบบมาเพื่อลดผลกระทบจาก class imbalance</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded">
      <p className="text-sm">
        <strong>Insight:</strong> ในงานจำแนกภาพแบบ multi-class เช่น CIFAR-600 การเลือกใช้ Cross-Entropy loss ร่วมกับ weighted class sampling มีผลอย่างมากต่อประสิทธิภาพของโมเดลเมื่อ training data ไม่สมดุล
      </p>
    </div>

    <h3>Loss Function สำหรับ Regression</h3>
    <ul>
      <li><strong>Mean Squared Error (MSE)</strong>: ใช้สำหรับวัดระยะห่างระหว่างค่าจริงกับค่าทำนาย</li>
      <li><strong>Mean Absolute Error (MAE)</strong>: มีความ robust กว่า MSE ในกรณีมี outliers</li>
      <li><strong>Huber Loss</strong>: ผสมผสาน MSE กับ MAE เพื่อความเสถียรในการฝึก</li>
    </ul>

    <h3>Evaluation Metrics ที่ใช้ในการประเมินโมเดล</h3>
    <table className="table-auto border-collapse w-full text-sm">
      <thead>
        <tr className="bg-gray-600 dark:bg-gray-700">
          <th className="border px-4 py-2 text-left">Metric</th>
          <th className="border px-4 py-2 text-left">คำอธิบาย</th>
          <th className="border px-4 py-2 text-left">เหมาะสำหรับ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Accuracy</td>
          <td className="border px-4 py-2">วัดสัดส่วนของผลลัพธ์ที่ทำนายถูกต้อง</td>
          <td className="border px-4 py-2">Classification</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Precision / Recall</td>
          <td className="border px-4 py-2">ใช้วัดความแม่นยำและความครอบคลุม</td>
          <td className="border px-4 py-2">Imbalanced Classification</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">F1 Score</td>
          <td className="border px-4 py-2">ค่าเฉลี่ยเชิงเรขาคณิตระหว่าง precision และ recall</td>
          <td className="border px-4 py-2">Binary Classification</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ROC AUC</td>
          <td className="border px-4 py-2">วัดความสามารถของโมเดลในการแยกแยะ class</td>
          <td className="border px-4 py-2">Probabilistic Model</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="text-sm">
        <strong>Highlight:</strong> ค่าประเมินเช่น Accuracy อาจทำให้เข้าใจผิดใน dataset ที่มี class imbalance จำเป็นต้องใช้ precision/recall หรือ F1 score ควบคู่เพื่อความเข้าใจที่ถูกต้อง
      </p>
    </div>

    <h3>แนวโน้มล่าสุดในการเลือก Loss Function</h3>
    <p>
      งานวิจัยล่าสุดจาก Stanford และ CMU แนะนำการปรับปรุง loss function เช่น Label Smoothing, Contrastive Loss และ Self-supervised Objectives เพื่อให้เหมาะกับโมเดลที่ใหญ่ขึ้น เช่น Vision Transformers (ViTs) หรือ ResNet-152
    </p>

    <h3>อ้างอิง</h3>
    <ul>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>IEEE: Survey on Evaluation Metrics for Deep Learning Models</li>
      <li>He et al., Deep Residual Learning for Image Recognition, CVPR 2016</li>
      <li>Chen et al., Label Smoothing Regularization, arXiv:1906.02629</li>
    </ul>
  </div>
</section>


       <section id="training-pipeline" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Training Pipeline สำหรับ Classification</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <h3>ขั้นตอนหลักของ Training Pipeline</h3>
    <p>
      ในระบบการเรียนรู้เชิงลึกที่พัฒนาอย่างเต็มรูปแบบ การฝึกโมเดลเพื่อทำ Classification จำเป็นต้องมีขั้นตอนการทำงานที่เป็นระบบอย่างชัดเจน โดยทั่วไป Training Pipeline จะประกอบด้วยขั้นตอนดังต่อไปนี้:
    </p>
    <ul>
      <li>Data Collection & Preprocessing</li>
      <li>Dataset Splitting (Train/Validation/Test)</li>
      <li>Model Initialization & Configuration</li>
      <li>Training Loop: Forward Pass, Loss, Backpropagation</li>
      <li>Checkpointing & Logging</li>
      <li>Evaluation & Fine-tuning</li>
    </ul>

    <h3>1. Data Preprocessing และ Augmentation</h3>
    <p>
      ขั้นตอนการเตรียมข้อมูลมีผลต่อประสิทธิภาพของโมเดลอย่างยิ่ง เช่น การปรับขนาดภาพ, Normalization, การใช้เทคนิค Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล
    </p>
    <div className="bg-blue-600 p-4 rounded-lg border border-blue-300">
      <strong>Highlight:</strong> เทคนิคอย่าง Random Flip, Rotation และ Color Jitter ช่วยลด Overfitting ได้ดีมากโดยเฉพาะกับ Dataset ที่มีขนาดเล็ก
    </div>

    <h3>2. Batch Loader และ Optimization Strategy</h3>
    <p>
      Batch Size ที่เหมาะสม และการใช้ Optimizer เช่น Adam หรือ SGD พร้อม Learning Rate Scheduling เป็นหัวใจของกระบวนการ Optimization
    </p>
    <table className="table-auto w-full text-left text-sm">
      <thead>
        <tr>
          <th className="border px-4 py-2">Parameter</th>
          <th className="border px-4 py-2">Example Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Batch Size</td>
          <td className="border px-4 py-2">32 - 256</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Optimizer</td>
          <td className="border px-4 py-2">AdamW, SGD</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Learning Rate</td>
          <td className="border px-4 py-2">1e-3 → 1e-5</td>
        </tr>
      </tbody>
    </table>

    <h3>3. Training Loop และ Backpropagation</h3>
    <p>
      กระบวนการเรียนรู้ของโมเดลจะเกิดจากการทำ Forward Pass, คำนวณ Loss, แล้วจึงทำ Backpropagation เพื่อลดค่าความผิดพลาด โดยใช้ Optimizer ในการปรับค่าพารามิเตอร์
    </p>

    <div className="bg-yellow-600 p-4 rounded-lg border border-yellow-300">
      <strong>Insight:</strong> งานวิจัยจาก Stanford แสดงให้เห็นว่า Gradient Clipping และการใช้ Warm-up Phase ของ Learning Rate ช่วยลดปัญหา Gradient Explosion ใน ResNet ได้อย่างมีนัยสำคัญ
    </div>

    <h3>4. Early Stopping และ Model Checkpointing</h3>
    <p>
      การหยุดการฝึกเมื่อ Validation Loss ไม่ดีขึ้นเป็นวิธีที่มีประสิทธิภาพในการป้องกัน Overfitting และช่วยลดเวลาการฝึกได้มาก
    </p>

    <h3>5. Evaluation และ Model Deployment</h3>
    <p>
      หลังฝึกเสร็จสิ้น โมเดลต้องผ่านขั้นตอน Evaluation อย่างละเอียด โดยใช้ Metrics เช่น Precision, Recall, F1-score และ ROC-AUC ก่อนจะนำไปใช้งานจริงหรือ Deploy บนระบบ Production
    </p>

    <h3>แหล่งอ้างอิงที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016</li>
      <li>Goodfellow et al., "Deep Learning," MIT Press, 2016</li>
      <li>arXiv: "Training Deep Neural Networks with Batch Normalization and Adaptive Learning Rates"</li>
    </ul>
  </div>
</section>


      <section id="accuracy-techniques" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. เทคนิคที่ช่วยเพิ่มความแม่นยำ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>6.1 การใช้ Data Augmentation</h3>
    <p>
      Data augmentation เป็นเทคนิคการเพิ่มชุดข้อมูลเทียมจากข้อมูลจริงโดยการเปลี่ยนแปลงลักษณะของข้อมูล เช่น การหมุน การสลับสี หรือการครอปภาพ ซึ่งช่วยให้โมเดลมีความสามารถในการ generalize สูงขึ้นและลดปัญหา overfitting โดยเฉพาะในการเรียนรู้จากชุดข้อมูลที่มีขนาดจำกัด
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4">
      <p className="font-medium">Insight</p>
      <p>
        จากการศึกษาของ Stanford University (2020) พบว่าการใช้ augmentation ที่มีความซับซ้อนร่วมกับ regularization ช่วยเพิ่ม accuracy บนชุดข้อมูล CIFAR-10 ได้ถึง 4–7% เมื่อเทียบกับ baseline model ที่ไม่ใช้ augmentation
      </p>
    </div>

    <h3>6.2 การเลือก Optimizer ที่เหมาะสม</h3>
    <p>
      การเลือก optimizer ที่เหมาะสม เช่น Adam, SGD+Momentum หรือ RMSProp มีผลโดยตรงต่ออัตราการเรียนรู้และความแม่นยำของโมเดล โดยการ tuning learning rate, weight decay และ momentum parameters สามารถสร้างผลลัพธ์ที่ต่างกันอย่างมีนัยสำคัญในการฝึกโมเดล deep learning
    </p>

    <h3>6.3 การใช้ Pretrained Models</h3>
    <p>
      การใช้โมเดลที่ถูกฝึกมาก่อนจากชุดข้อมูลขนาดใหญ่ เช่น ImageNet แล้วนำมา fine-tune บน task-specific dataset เป็นเทคนิคที่มีประสิทธิภาพสูง โดยเฉพาะเมื่อชุดข้อมูลมีขนาดจำกัด เช่น ในการตรวจจับโรคจากภาพถ่ายทางการแพทย์
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4">
      <p className="font-medium">Highlight</p>
      <p>
        การใช้ pretrained model ไม่เพียงแต่ลดเวลาในการ train เท่านั้น แต่ยังช่วยให้โมเดลสามารถเรียนรู้ feature representation ที่มีความลึกจาก layer ต้นจนถึง layer ลึกได้ทันทีโดยไม่ต้องเริ่มจากศูนย์
      </p>
    </div>

    <h3>6.4 เทคนิค Ensemble Learning</h3>
    <p>
      Ensemble learning คือการผสมโมเดลหลายตัวเข้าด้วยกันเพื่อเพิ่มความแม่นยำและความมั่นคง เช่น การใช้ voting, bagging, boosting หรือ stacking ซึ่งเป็นเทคนิคที่ใช้ในโมเดลระดับแข่งขัน เช่น ImageNet Challenge
    </p>

    <table className="table-auto w-full border text-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border px-4 py-2 text-left">เทคนิค</th>
          <th className="border px-4 py-2 text-left">รายละเอียด</th>
          <th className="border px-4 py-2 text-left">ความนิยม</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Voting</td>
          <td className="border px-4 py-2">เฉลี่ยผลลัพธ์จากโมเดลหลายตัว</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Bagging</td>
          <td className="border px-4 py-2">ใช้ sampling และฝึกโมเดลหลายตัว</td>
          <td className="border px-4 py-2">ปานกลาง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Boosting</td>
          <td className="border px-4 py-2">ปรับน้ำหนักตัวอย่างเพื่อเน้นตัวอย่างที่ยาก</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stacking</td>
          <td className="border px-4 py-2">ใช้โมเดลชั้นที่สองเรียนจาก output ของโมเดลแรก</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
      </tbody>
    </table>

    <h3>6.5 การใช้ Learning Rate Scheduler</h3>
    <p>
      Learning rate ที่เปลี่ยนแปลงตามช่วงเวลา เช่น Step decay, Cosine Annealing หรือ ReduceLROnPlateau ช่วยให้โมเดลสามารถหลีกเลี่ยงการติด local minima และทำให้การเรียนรู้มีประสิทธิภาพมากขึ้นในช่วงท้ายของการฝึก
    </p>

    <h3>6.6 การ Regularization ด้วย Techniques เช่น Dropout และ Weight Decay</h3>
    <p>
      Dropout ช่วยป้องกันการ overfitting โดยสุ่มปิด neuron บางส่วนในระหว่างการฝึก ในขณะที่ weight decay เป็นการเพิ่มค่า penalty ต่อ weight ที่มีค่าสูงเกินไป ซึ่งช่วยให้โมเดลสามารถเรียนรู้ได้แบบ generalize มากขึ้น
    </p>

    <h3>6.7 การอ้างอิงจากแหล่งข้อมูลวิจัย</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>He, K. et al. "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385.</li>
      <li>Howard, A. G. et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861.</li>
      <li>Srivastava, N. et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR 2014.</li>
      <li>Yosinski, J. et al. "How Transferable Are Features in Deep Neural Networks?." NeurIPS 2014.</li>
      <li>Stanford CS231n Course Notes, Lecture 10: Neural Networks Tricks.</li>
    </ul>
  </div>
</section>


     <section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Visualization การทำงานของ CNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10 dark:prose-invert">
    <h3 className="text-xl font-semibold">7.1 ความสำคัญของการทำ Visualization</h3>
    <p>
      Visualization เป็นเครื่องมือสำคัญในการวิเคราะห์พฤติกรรมของ Convolutional Neural Networks (CNNs) ซึ่งช่วยให้เข้าใจว่าโมเดลให้ความสำคัญกับ feature ใดใน input และสามารถตรวจสอบว่าโมเดลเรียนรู้อย่างไรในแต่ละ layer
    </p>
    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-medium">Insight:</p>
      <p>การทำ Visualization ของ CNN ช่วยเพิ่มความเข้าใจและสามารถวิเคราะห์ข้อผิดพลาดหรือ bias ได้โดยไม่ต้องใช้ brute-force experiments.</p>
    </div>

    <h3 className="text-xl font-semibold">7.2 เทคนิค Visualization ที่นิยม</h3>
    <ul className="list-disc list-inside">
      <li><strong>Feature Map Visualization:</strong> แสดงผลลัพธ์จากแต่ละ convolution layer เพื่อดูว่า layer นั้นเน้น feature ใด</li>
      <li><strong>Filter Visualization:</strong> แสดง weight ของ kernel ในแต่ละ filter เพื่อดูว่าโมเดลเรียนรู้อะไร</li>
      <li><strong>Activation Maximization:</strong> สร้าง input ที่ maximize การตอบสนองของ neuron ที่สนใจ</li>
      <li><strong>Grad-CAM (Gradient-weighted Class Activation Mapping):</strong> แสดงพื้นที่ใน input ที่ส่งผลต่อ output มากที่สุด</li>
      <li><strong>Saliency Maps:</strong> คำนวณ gradient ของ output ต่อ input เพื่อดู pixel ที่สำคัญ</li>
    </ul>

    <h3 className="text-xl font-semibold">7.3 ตารางเปรียบเทียบเทคนิค Visualization</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-left border border-gray-300">
        <thead className="bg-gray-600 dark:bg-gray-800">
          <tr>
            <th className="px-4 py-2">เทคนิค</th>
            <th className="px-4 py-2">เป้าหมาย</th>
            <th className="px-4 py-2">ข้อดี</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-4 py-2">Feature Map</td>
            <td className="px-4 py-2">วิเคราะห์การทำงานของ layer</td>
            <td className="px-4 py-2">เข้าใจ feature hierarchy</td>
          </tr>
          <tr>
            <td className="px-4 py-2">Filter</td>
            <td className="px-4 py-2">เข้าใจ weight pattern</td>
            <td className="px-4 py-2">ดูว่า kernel เรียนรู้อะไร</td>
          </tr>
          <tr>
            <td className="px-4 py-2">Grad-CAM</td>
            <td className="px-4 py-2">หา pixel ที่ส่งผลต่อ decision</td>
            <td className="px-4 py-2">มี interpretability สูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">7.4 ตัวอย่าง Code: การสร้าง Saliency Map</h3>
<div className="overflow-auto bg-gray-700 text-white text-sm rounded-lg my-6 border ">
  <pre className="whitespace-pre p-4 min-w-full">
    <code className="language-python">
{`import torch
import torch.nn.functional as F
from torchvision import models

def compute_saliency(input_image, model):
    input_image.requires_grad_()
    output = model(input_image)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency = input_image.grad.abs().squeeze().max(dim=0)[0]
    return saliency`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold">7.5 การประยุกต์ใช้ในงานวิจัย</h3>
    <p>
      งานวิจัยจาก MIT และ Stanford ได้นำ Visualization มาใช้เพื่อวิเคราะห์ bias และ fairness ของ CNN ในงานเช่น facial recognition และ autonomous driving ซึ่งช่วยระบุได้ว่าโมเดลตัดสินใจผิดจากอะไร และสามารถใช้เพื่อปรับปรุง fairness
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-medium">Highlight:</p>
      <p>Visualization ไม่ได้เป็นเพียงเครื่องมือเชิงเทคนิค แต่เป็นกลไกสำคัญในการทำ interpretability และ explainability ของ AI ให้เข้าใจได้ง่ายและตรวจสอบได้จริง</p>
    </div>

    <h3 className="text-xl font-semibold">7.6 แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Olah, C. (2017). <em>Feature Visualization</em>, Distill. <a href="https://distill.pub/2017/feature-visualization/">distill.pub</a></li>
      <li>Selvaraju, R. R. et al. (2017). <em>Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization</em>, ICCV.</li>
      <li>Zeiler, M. D., & Fergus, R. (2014). <em>Visualizing and Understanding Convolutional Networks</em>, ECCV.</li>
    </ul>
  </div>
</section>


<section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Real-World Use Cases</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>การประยุกต์ใช้ CNN ในสาขาต่าง ๆ</h3>
    <p>
      Convolutional Neural Networks (CNNs) ได้รับการใช้งานในหลากหลายโดเมน เนื่องจากความสามารถในการเรียนรู้เชิงลึกจากข้อมูลภาพและโครงสร้างข้อมูลที่ซับซ้อน ทำให้เหมาะสมกับงานที่ต้องการการแยกแยะรูปแบบและบริบทในระดับสูง
    </p>

    <h3>1. ด้านการแพทย์</h3>
    <ul>
      <li><strong>Medical Imaging:</strong> ใช้สำหรับการตรวจจับมะเร็งเต้านมจากภาพแมมโมแกรม (mammography), การวินิจฉัยโรคจาก MRI หรือ CT Scan</li>
      <li><strong>Retinal Analysis:</strong> ใช้ตรวจหาภาวะเบาหวานขึ้นจอประสาทตา (diabetic retinopathy)</li>
    </ul>

    <h3>2. ด้านอุตสาหกรรมยานยนต์</h3>
    <ul>
      <li><strong>Autonomous Vehicles:</strong> ใช้ในระบบกล้องสำหรับตรวจจับวัตถุ การจำแนกถนน และระบบเตือนภัย</li>
      <li><strong>Driver Monitoring:</strong> ตรวจสอบการหลับในหรือการใช้สมาธิของผู้ขับขี่</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded">
      <p className="font-semibold">Insight:</p>
      <p>
        การประยุกต์ใช้ CNN ในภาพทางการแพทย์มีความแม่นยำถึงระดับเดียวกับผู้เชี่ยวชาญ และได้รับการรับรองจาก FDA ในหลายกรณี เช่น ระบบของ Google Health สำหรับการวินิจฉัยโรคตาจากภาพเรตินา
      </p>
    </div>

    <h3>3. การเกษตรและสิ่งแวดล้อม</h3>
    <ul>
      <li><strong>Crop Disease Detection:</strong> ใช้ตรวจสอบโรคพืชจากภาพถ่ายใบพืชในระบบ IoT</li>
      <li><strong>Environmental Monitoring:</strong> วิเคราะห์ภาพถ่ายดาวเทียมเพื่อประเมินการตัดไม้ทำลายป่า</li>
    </ul>

    <h3>4. ด้านความปลอดภัยและความมั่นคง</h3>
    <ul>
      <li><strong>Face Recognition:</strong> ใช้ในระบบตรวจสอบตัวตนที่สนามบิน, ธนาคาร และระบบสาธารณะ</li>
      <li><strong>Surveillance:</strong> ตรวจจับพฤติกรรมต้องสงสัยจากกล้องวงจรปิดแบบ real-time</li>
    </ul>

    <h3>5. ธุรกิจและ e-Commerce</h3>
    <ul>
      <li><strong>Visual Search:</strong> ใช้ค้นหาสินค้าจากภาพที่อัปโหลด</li>
      <li><strong>Product Categorization:</strong> แยกประเภทสินค้าโดยอัตโนมัติจากภาพ</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        Amazon, Alibaba, และ Shopee ได้ใช้ CNN ในการแนะนำสินค้าแบบ personal recommendation ผ่านการวิเคราะห์ภาพประกอบสินค้าและพฤติกรรมผู้ใช้งาน
      </p>
    </div>

    <h3>6. การสร้างสรรค์เชิงศิลป์</h3>
    <ul>
      <li><strong>Style Transfer:</strong> นำสไตล์ของภาพวาดศิลปินมาใช้กับภาพถ่าย</li>
      <li><strong>Generative Art:</strong> ใช้ CNN ร่วมกับ GAN เพื่อสร้างภาพใหม่ที่มีองค์ประกอบเชิงศิลป์</li>
    </ul>

    <h3>อ้างอิง</h3>
    <ul>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," arXiv 2017</li>
      <li>Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," Nature 2017</li>
      <li>Redmon et al., "YOLOv3: An Incremental Improvement," arXiv 2018</li>
      <li>Nguyen et al., "Automated Crop Disease Recognition Using CNNs," IEEE Access 2020</li>
    </ul>
  </div>
</section>


   <section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ข้อจำกัดที่ต้องพิจารณา</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>ปัญหาด้านข้อมูลและอคติที่แฝงอยู่</h3>
    <p>
      ความสำเร็จของ Deep Learning พึ่งพาข้อมูลปริมาณมากที่มีคุณภาพสูง อย่างไรก็ตาม ข้อมูลที่ใช้ฝึกโมเดลมักมีอคติทางวัฒนธรรม, เชื้อชาติ, หรือแม้แต่ปัญหาด้านการแทนค่าผิด ซึ่งอาจนำไปสู่การตัดสินใจที่ไม่ยุติธรรม โดยเฉพาะในระบบที่เกี่ยวข้องกับสุขภาพ, ความปลอดภัย หรือความยุติธรรมทางสังคม
    </p>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded">
      <strong>Insight:</strong> งานวิจัยจาก MIT และ Harvard พบว่าโมเดล AI ที่ฝึกจากชุดข้อมูลภาพถ่ายใบหน้ามีความแม่นยำลดลงอย่างมากเมื่อใช้กับกลุ่มชาติพันธุ์ที่ไม่ปรากฏในข้อมูลฝึก (Buolamwini & Gebru, 2018)
    </div>

    <h3>การตีความผลลัพธ์และ Explainability</h3>
    <p>
      แม้ Deep Learning จะมีความสามารถในการจำแนกรูปแบบที่ซับซ้อน แต่โมเดลกลับขาดความสามารถในการอธิบายว่าเหตุใดจึงให้ผลลัพธ์เช่นนั้น ทำให้เกิดความท้าทายในการนำไปใช้งานในบริบทที่ต้องการความโปร่งใส เช่น การแพทย์ การเงิน และการตัดสินคดี
    </p>
    <ul>
      <li>เทคนิคเช่น SHAP, LIME, Grad-CAM ถูกพัฒนาขึ้นเพื่อตีความการตัดสินใจของโมเดล</li>
      <li>อย่างไรก็ตาม ความน่าเชื่อถือของเทคนิคเหล่านี้ยังเป็นที่ถกเถียงในวงวิชาการ</li>
    </ul>

    <h3>ความเปราะบางของโมเดล (Adversarial Vulnerabilities)</h3>
    <p>
      งานวิจัยจำนวนมากยืนยันว่า Deep Neural Networks มีความเปราะบางอย่างยิ่งต่อการโจมตีแบบ Adversarial เช่น การเพิ่ม noise ที่ตามนุษย์ไม่สามารถสังเกตเห็นได้ แต่สามารถเปลี่ยนผลลัพธ์ของโมเดลอย่างสิ้นเชิง
    </p>
    <table className="table-auto w-full text-left border mt-6">
      <thead>
        <tr>
          <th className="border px-4 py-2">ประเภท</th>
          <th className="border px-4 py-2">ลักษณะ</th>
          <th className="border px-4 py-2">ตัวอย่าง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">White-box Attack</td>
          <td className="border px-4 py-2">รู้โครงสร้างของโมเดล</td>
          <td className="border px-4 py-2">FGSM, PGD</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Black-box Attack</td>
          <td className="border px-4 py-2">ไม่รู้โครงสร้างภายใน</td>
          <td className="border px-4 py-2">Transfer-based Attack</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded">
      <strong>Highlight:</strong> Adversarial Examples สามารถเปลี่ยนผลลัพธ์การจำแนกรูปภาพจาก "แพนด้า" เป็น "กอริลลา" ด้วยการเพิ่ม noise เพียงเล็กน้อย (Goodfellow et al., 2015)
    </div>

    <h3>ข้อจำกัดด้านพลังงานและทรัพยากร</h3>
    <p>
      โมเดล Deep Learning ที่มีขนาดใหญ่ เช่น GPT, BERT, และ ResNet ต้องการทรัพยากรการประมวลผลสูงมาก ส่งผลให้การฝึกและใช้งานมีค่าใช้จ่ายด้านพลังงานที่สูง
    </p>
    <ul>
      <li>การฝึก GPT-3 ใช้พลังงานไฟฟ้าประมาณ 1,287 MWh (Strubell et al., 2019)</li>
      <li>โมเดลขนาดใหญ่ยังต้องใช้เวลา inference ที่นานบน edge devices</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul>
      <li>Buolamwini & Gebru, "Gender Shades", Conference on Fairness, Accountability, and Transparency, 2018</li>
      <li>Strubell et al., "Energy and Policy Considerations for Deep Learning", ACL, 2019</li>
      <li>Goodfellow et al., "Explaining and Harnessing Adversarial Examples", arXiv preprint, 2015</li>
      <li>Samek et al., "Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models", arXiv, 2017</li>
    </ul>
  </div>
</section>


    <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">การมองเชิงลึกในวิวัฒนาการของสถาปัตยกรรม Deep Learning</h3>
    <p>
      การพัฒนาสถาปัตยกรรมในระบบ Deep Learning ไม่ได้เป็นเพียงแค่การออกแบบโครงสร้างเครือข่ายที่ซับซ้อนขึ้น แต่ยังเกี่ยวข้องกับการเปลี่ยนแปลงแนวคิดหลักในการเรียนรู้ การถ่ายทอดความรู้ การแยกแยะฟีเจอร์ และการประเมินความสามารถในการเรียนรู้ของระบบอัตโนมัติ ทั้งนี้ การปรับสถาปัตยกรรมเหล่านี้ช่วยให้โมเดลสามารถ generalize ได้ดีขึ้น และทำงานกับข้อมูลที่หลากหลายมากขึ้นในโลกจริง
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-medium">Insight:</p>
      <p>
        สถาปัตยกรรมเชิงลึกอย่าง ResNet และ Transformer ไม่เพียงแต่เพิ่มความลึก แต่ยังออกแบบกลไกการถ่ายทอด gradient และการประมวลผลแบบ attention เพื่อขยายศักยภาพของโมเดลในด้านความเข้าใจบริบท และการเรียนรู้ฟีเจอร์ที่มีลำดับซับซ้อน
      </p>
    </div>

    <h3 className="text-xl font-semibold">การนำไปใช้ในสาขาต่าง ๆ</h3>
    <ul className="list-disc pl-6">
      <li><strong>การแพทย์:</strong> ระบบ AI ในการวินิจฉัยภาพถ่ายรังสี (เช่น X-ray, CT, MRI) ด้วย CNN และ Vision Transformer</li>
      <li><strong>การเงิน:</strong> การใช้ RNN และ LSTM ในการทำนายแนวโน้มของตลาด หรือการตรวจจับ anomaly</li>
      <li><strong>พลังงาน:</strong> ใช้ deep forecasting networks ในการบริหารจัดการการผลิตพลังงานไฟฟ้าและพลังงานทดแทน</li>
      <li><strong>สิ่งแวดล้อม:</strong> วิเคราะห์ภาพดาวเทียมเพื่อทำนายการเปลี่ยนแปลงสภาพภูมิอากาศและการจัดการภัยพิบัติ</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-medium">Highlight:</p>
      <p>
        โมเดลเชิงลึกไม่สามารถทำงานได้ดีหากขาดการเข้าใจธรรมชาติของข้อมูลเฉพาะสาขา ดังนั้นการเลือกสถาปัตยกรรมต้องสัมพันธ์กับ task และลักษณะข้อมูลโดยตรง เพื่อให้ได้ผลลัพธ์ที่มีประสิทธิภาพสูงสุด
      </p>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบสถาปัตยกรรมยอดนิยม</h3>
    <table className="table-auto w-full text-left border border-gray-300">
      <thead>
        <tr className="bg-gray-500">
          <th className="px-4 py-2">Architecture</th>
          <th className="px-4 py-2">ข้อดี</th>
          <th className="px-4 py-2">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">CNN</td>
          <td className="border px-4 py-2">แม่นยำสูงในงานภาพ</td>
          <td className="border px-4 py-2">จำกัดในข้อมูลลำดับ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">RNN / LSTM</td>
          <td className="border px-4 py-2">เข้าใจบริบทข้อมูลเวลา</td>
          <td className="border px-4 py-2">Gradient Vanishing</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Transformer</td>
          <td className="border px-4 py-2">ประมวลผลแบบขนานได้ดี</td>
          <td className="border px-4 py-2">ใช้ทรัพยากรสูง</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Harvard NLP Group: The Annotated Transformer</li>
      <li>IEEE Transactions on Neural Networks and Learning Systems</li>
      <li>Nature: Deep Learning in Medical Diagnosis, 2020</li>
      <li>arXiv: "Attention Is All You Need" – Vaswani et al., 2017</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day44 theme={theme} />
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
        </div>
      </div>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day44 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day44_CNNImageClassification;
