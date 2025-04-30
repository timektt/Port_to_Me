import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day21 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day21";
import MiniQuiz_Day21 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day21";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day21_CNNIntro = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("cnn1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("cnn2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("cnn3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("cnn4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("cnn5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("cnn6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("cnn7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("cnn8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("cnn9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("cnn10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("cnn11").format("auto").quality("auto").resize(scale().width(600));
  const img12 = cld.image("cnn12").format("auto").quality("auto").resize(scale().width(600));
  const img13 = cld.image("cnn13").format("auto").quality("auto").resize(scale().width(600));



  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 21: Introduction to Convolutional Neural Networks (CNN)</h1>

        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img1} />
        </div>

        <div className="w-full flex justify-center my-12">
          <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
        </div>

        <section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้อง Convolutional Neural Networks</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img2} />
        </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      ในโลกของข้อมูลภาพ (image data) ซึ่งเป็นข้อมูลที่มีลักษณะเชิงพื้นที่สูง (spatial structure) การใช้โครงข่ายประสาทแบบดั้งเดิม เช่น Multilayer Perceptron (MLP) มักพบข้อจำกัดหลายประการที่ทำให้ประสิทธิภาพในการเรียนรู้และตีความข้อมูลลดลงอย่างมาก โดยเฉพาะในงานด้าน Computer Vision เช่น Image Classification, Object Detection, และ Image Segmentation ที่ต้องการการเข้าใจเชิงลึกเกี่ยวกับโครงสร้างของภาพ
    </p>

    <h3 className="text-xl font-semibold">ข้อจำกัดของ Multilayer Perceptron (MLP) ต่อข้อมูลภาพ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        ข้อมูลภาพ เช่น ขนาด 224x224x3 (RGB) เมื่อป้อนเข้า MLP ต้องทำการ flatten เป็น vector ขนาด 150,528 ซึ่งทำให้สูญเสียโครงสร้างเชิงพื้นที่
      </li>
      <li>
        การเชื่อมต่อแบบ Fully Connected ทำให้เกิดจำนวนพารามิเตอร์ที่มากเกินไป และส่งผลให้ต้องใช้ทรัพยากรในการฝึกโมเดลสูง
      </li>
      <li>
        ไม่มีความสามารถในการเรียนรู้ Feature ที่มีลักษณะ Local (เช่น ขอบภาพ, รูปร่าง) อย่างมีประสิทธิภาพ
      </li>
    </ul>

    <h3 className="text-xl font-semibold">แนวคิดเบื้องต้นของ Convolutional Neural Networks (CNN)</h3>
    <p>
      Convolutional Neural Networks (CNN) ได้รับแรงบันดาลใจจากระบบการมองเห็นของมนุษย์ (visual cortex) โดยใช้หลักการของการเรียนรู้ Feature จากภาพแบบลำดับชั้น (hierarchical features) ซึ่งช่วยลดจำนวนพารามิเตอร์และรักษาโครงสร้างเชิงพื้นที่ของข้อมูลไว้ได้อย่างมีประสิทธิภาพ
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>
        ใช้ <strong>Filter หรือ Kernel</strong> ซึ่งเลื่อนผ่านภาพเพื่อจับ Feature แบบเฉพาะจุด
      </li>
      <li>
        สร้าง <strong>Feature Map</strong> ที่แสดงคุณลักษณะต่าง ๆ เช่น ขอบ, รูปทรง, และลวดลายในภาพ
      </li>
      <li>
        ลดความซับซ้อนของข้อมูลผ่าน <strong>Pooling</strong> และใช้ Fully Connected Layer เพื่อทำการตัดสินผลลัพธ์สุดท้าย
      </li>
    </ul>

    <h3 className="text-xl font-semibold">วิวัฒนาการและการนำไปใช้จริง</h3>
    <p>
      CNN ได้รับความนิยมอย่างแพร่หลายภายหลังความสำเร็จของโมเดล <strong>AlexNet</strong> (Krizhevsky et al., 2012) ที่สามารถคว้ารางวัลชนะเลิศจากการแข่งขัน ImageNet Large Scale Visual Recognition Challenge (ILSVRC) ด้วยความแม่นยำที่เหนือกว่าทุกทีมในขณะนั้น และเป็นจุดเปลี่ยนสำคัญของวงการ Deep Learning
    </p>

    <p>
      ต่อมาโมเดลอย่าง <strong>VGGNet (Oxford)</strong>, <strong>ResNet (Microsoft Research)</strong>, และ <strong>Inception (Google)</strong> ได้พัฒนาโครงสร้างของ CNN ให้ลึกขึ้นและมีประสิทธิภาพยิ่งขึ้นโดยเน้นการใช้งานในงานที่มีความซับซ้อนมากขึ้น
    </p>

    <h3 className="text-xl font-semibold">การประยุกต์ใช้งาน CNN ที่หลากหลาย</h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Computer Vision</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Image Classification เช่น ResNet, EfficientNet</li>
          <li>Object Detection เช่น YOLO, Faster-RCNN</li>
          <li>Image Segmentation เช่น U-Net, DeepLab</li>
        </ul>
      </div>
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">สาขาอื่น ๆ</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Medical Imaging เช่น ตรวจหามะเร็งจาก MRI หรือ CT</li>
          <li>Autonomous Driving เช่น การตรวจจับเลนและสิ่งกีดขวาง</li>
          <li>Face Recognition และ Video Surveillance</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">ทำไม CNN ถึงสำคัญต่อยุคของ AI</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สามารถเรียนรู้ Feature โดยไม่ต้องใช้ Feature Engineering แบบแมนนวล</li>
      <li>รองรับข้อมูลเชิงภาพได้ดีกว่าโครงข่ายทั่วไป</li>
      <li>ลดความซับซ้อนและเพิ่มประสิทธิภาพในการฝึกโมเดล</li>
      <li>สนับสนุนการทำ Transfer Learning และ Fine-tuning ได้ดี</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight จากงานวิจัยระดับโลก</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm">
        <li>
          <strong>AlexNet (2012)</strong>: จุดเริ่มต้นของการฟื้นฟู Deep Learning ด้วยการใช้ GPU ในการฝึกโมเดลภาพขนาดใหญ่
        </li>
        <li>
          <strong>VGGNet (2014)</strong>: เน้นการใช้ Convolution 3x3 ซ้อนกันหลายชั้น ทำให้เข้าใจง่ายและขยายโมเดลได้ง่าย
        </li>
        <li>
          <strong>ResNet (2015)</strong>: เสนอ Residual Connection แก้ปัญหา vanishing gradient ในเครือข่ายลึก
        </li>
        <li>
          <strong>CoAtNet (Google Brain 2021)</strong>: ผสมผสานระหว่าง CNN และ Transformer เพื่อให้ได้ความสามารถทั้งแบบ Local และ Global
        </li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      Convolutional Neural Networks เป็นหนึ่งในสถาปัตยกรรมที่ทรงพลังที่สุดใน Deep Learning สำหรับงานด้าน Computer Vision และภาพในโลกจริง ด้วยความสามารถในการเรียนรู้ Feature แบบอัตโนมัติจากข้อมูลภาพโดยตรง CNN ได้เปลี่ยนแนวทางของ Machine Learning จากการพึ่งพาการออกแบบ Feature แบบแมนนวลไปสู่ระบบที่สามารถเรียนรู้ได้เองจากข้อมูลปริมาณมาก และยังคงเป็นแกนกลางของงานวิจัยที่สำคัญใน AI อย่างต่อเนื่อง
    </p>
  </div>
</section>

<section id="cnn-structure" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. โครงสร้างพื้นฐานของ CNN</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img3} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Convolutional Neural Network (CNN) ประกอบด้วยเลเยอร์หลักที่ออกแบบมาเพื่อจัดการข้อมูลเชิงภาพแบบมีโครงสร้าง ได้แก่ Convolutional Layer, Activation Function, Pooling Layer, และ Fully Connected Layer โดยมีการใช้งานอย่างแพร่หลายในงาน Image Classification, Object Detection และ Computer Vision ที่ต้องการให้โมเดลเข้าใจโครงสร้างเชิงพื้นที่ของภาพอย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">2.1 Input Layer</h3>
    <p>
      ข้อมูลที่ป้อนเข้าสู่ CNN มักอยู่ในรูปของ Tensor ขนาด 3 มิติ: ความสูง (H), ความกว้าง (W), และจำนวนช่อง (C) เช่น ภาพสี RGB จะมีขนาด 224 x 224 x 3 ในงาน ImageNet โดยไม่จำเป็นต้อง flatten ภาพเหมือนใน MLP
    </p>

    <h3 className="text-xl font-semibold">2.2 Convolutional Layer</h3>
    <p>
      เป็นเลเยอร์ที่ใช้ Kernel หรือ Filter ขนาดเล็ก (เช่น 3x3, 5x5) สไลด์บนภาพเพื่อดึงเอาคุณลักษณะท้องถิ่น (Local Features) โดยการคำนวณ Dot Product ระหว่าง Filter กับ Sub-region ของภาพ ผลลัพธ์จะได้เป็น Feature Map
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">
{`output(i,j) = ΣΣ input(i+m, j+n) * filter(m,n)`}
      </pre>
    </div>

    <p>
      งานของ LeCun et al. (1998) กับ LeNet-5 เป็นจุดเริ่มต้นของการใช้ Convolutional Layer ที่ช่วยลดจำนวนพารามิเตอร์ลงอย่างมากเมื่อเทียบกับ Fully Connected Layer
    </p>

    <h3 className="text-xl font-semibold">2.3 Activation Function</h3>
    <p>
      ฟังก์ชันที่นิยมใน CNN คือ ReLU (Rectified Linear Unit) ซึ่งช่วยเพิ่มความไม่เชิงเส้น (non-linearity) ให้กับโมเดล โดยนิยมนำมาใช้หลัง Convolution เพื่อให้โมเดลเรียนรู้ฟีเจอร์ที่ซับซ้อนได้ดีขึ้น
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">
{`ReLU(x) = max(0, x)`}
      </pre>
    </div>

    <p>
      จากงานของ He et al. (2015), การใช้ ReLU ช่วยลดปัญหา vanishing gradient และเร่งการลู่เข้าในการฝึกโมเดลลึก
    </p>

    <h3 className="text-xl font-semibold">2.4 Pooling Layer</h3>
    <p>
      Pooling ช่วยลดมิติของ Feature Map และเพิ่ม Invariance ต่อการเปลี่ยนแปลงของภาพ เช่น Max Pooling ใช้ค่าที่มากที่สุดในแต่ละ region โดยทั่วไปใช้ขนาด 2x2 และ stride = 2 เพื่อลดขนาดลงครึ่งหนึ่งในแต่ละ dimension
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Max Pooling</h4>
        <p className="text-sm">เลือกค่ามากสุดในแต่ละ window</p>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Average Pooling</h4>
        <p className="text-sm">คำนวณค่าเฉลี่ยในแต่ละ window</p>
      </div>
    </div>

    <p>
      จากการศึกษาในงานของ Zeiler & Fergus (2013), การใช้ Pooling Layer ช่วยลด noise และช่วยให้โมเดลสามารถ generalize ได้ดีขึ้น
    </p>

    <h3 className="text-xl font-semibold">2.5 Fully Connected Layer (FC)</h3>
    <p>
      เป็นเลเยอร์ที่เชื่อมต่อ Feature จาก Convolution และ Pooling Layer แบบ Flat แล้วป้อนเข้าสู่ Dense Layer เพื่อทำ Classification หรือ Regression ในขั้นตอนสุดท้าย
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">
{`model = nn.Sequential(
    nn.Conv2d(...),
    nn.ReLU(),
    nn.MaxPool2d(...),
    nn.Flatten(),
    nn.Linear(...),
    nn.Softmax(dim=1)
)`}
      </pre>
    </div>

    <p>
      ใน CNN สมัยใหม่ เช่น VGG และ ResNet มีการใช้ FC Layer น้อยลงและแทนที่ด้วย Global Average Pooling เพื่อรักษาความกะทัดรัดของโมเดล
    </p>

    <h3 className="text-xl font-semibold">2.6 Output Layer</h3>
    <p>
      เป็นชั้นสุดท้ายที่ใช้ฟังก์ชัน Softmax หรือ Sigmoid ขึ้นอยู่กับประเภทของปัญหา เช่น Softmax ใช้ใน Multi-class Classification เพื่อเปลี่ยนค่า logits ให้เป็นความน่าจะเป็นรวม 1
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">
{`Softmax(z_i) = exp(z_i) / Σ exp(z_j)`}
      </pre>
    </div>

    <p>
      สถาปัตยกรรม CNN ที่ประสบความสำเร็จมากที่สุดในประวัติศาสตร์ ได้แก่:
    </p>
    <ul className="list-disc pl-6 space-y-1">
      <li><strong>LeNet-5:</strong> พัฒนาโดย Yann LeCun สำหรับ Handwritten Digit Recognition</li>
      <li><strong>AlexNet:</strong> ใช้ ReLU และ Dropout กับ GPU Training เป็นรายแรก</li>
      <li><strong>VGG:</strong> ใช้ 3x3 Convolution ซ้ำ ๆ ในสถาปัตยกรรมที่ลึก</li>
      <li><strong>ResNet:</strong> นำ Residual Connection มาใช้เพื่อแก้ปัญหา Degradation</li>
    </ul>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="text-lg font-semibold mb-2">Insight จากงานวิจัย</h4>
      <ul className="list-disc pl-6 text-sm">
        <li>He et al. (2016) แสดงให้เห็นว่า Residual Learning ช่วยให้ฝึกโมเดลได้ลึกขึ้นโดยไม่เกิด vanishing gradient</li>
        <li>Simonyan & Zisserman (2014) พบว่าแม้ 3x3 Conv ขนาดเล็กก็สามารถสร้างโมเดลลึกที่ทรงพลังได้</li>
        <li>Szegedy et al. (Inception) พัฒนา Conv block ที่มีขนาดหลายแบบเพื่อเพิ่มความหลากหลายของ receptive field</li>
      </ul>
    </div>
  </div>
</section>

<section id="convolution-layer" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. หลักการทำงานของ Convolution Layer</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img4} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Convolution Layer คือหัวใจสำคัญของ Convolutional Neural Networks (CNNs) ซึ่งมีหน้าที่หลักในการเรียนรู้ Feature Map โดยการเลื่อน Filter (หรือ Kernel) บนข้อมูล Input เช่น รูปภาพ เพื่อดึง Feature ที่สำคัญในแต่ละพื้นที่ CNNs ได้รับแรงบันดาลใจจาก Visual Cortex ของสมองมนุษย์ที่ตอบสนองเฉพาะจุดในพื้นที่ภาพ ทำให้สามารถเรียนรู้ "local patterns" ได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">3.1 กลไกของ Convolution</h3>
    <p>
      การคำนวณ Convolution เกิดจากการเลื่อน Filter บนภาพ และคำนวณค่า Dot Product ระหว่าง Filter กับ Patch ของ Input ณ ตำแหน่งนั้น ค่า Dot Product ที่ได้จะเป็นค่าหนึ่งใน Feature Map ดังนั้น Output ของ Convolution Layer จะเป็น Tensor 3 มิติ (H', W', D) ซึ่ง H', W' คือขนาดใหม่ของภาพ และ D คือจำนวน Filters ที่ใช้
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">
{`Input Tensor: (H, W, C)
Filter: (fH, fW, C)
Stride: S
Padding: P

Output Height: H' = (H - fH + 2P) / S + 1
Output Width:  W' = (W - fW + 2P) / S + 1

Feature Map[i, j] = sum(Filter * Input[i:i+fH, j:j+fW])`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">3.2 Hyperparameters ที่สำคัญ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Filter Size:</strong> กำหนดขนาดของ Kernel เช่น 3x3, 5x5</li>
      <li><strong>Stride:</strong> ระยะการเลื่อนของ Filter บน Input (ค่าที่นิยม: 1 หรือ 2)</li>
      <li><strong>Padding:</strong> การเพิ่ม Border รอบ Input เช่น same (รักษาขนาดเดิม), valid (ไม่ Padding)</li>
      <li><strong>Number of Filters:</strong> กำหนดจำนวน Feature Maps ที่ต้องการเรียนรู้</li>
    </ul>

    <h3 className="text-xl font-semibold">3.3 คุณสมบัติเด่นของ Convolution Layer</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ช่วยลดจำนวนพารามิเตอร์เมื่อเทียบกับ Fully Connected Layer โดยเฉพาะกับ Input ขนาดใหญ่</li>
      <li>สามารถเรียนรู้ Local Patterns เช่น ขอบ (edge), เส้นตรง, สี หรือ texture</li>
      <li>สามารถ Reuse filter เดิมในหลายตำแหน่งของ Input ได้ (weight sharing)</li>
    </ul>

    <h3 className="text-xl font-semibold">3.4 โค้ดตัวอย่างการสร้าง Convolution Layer ด้วย PyTorch</h3>
    <div className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
      <pre>{`import torch
import torch.nn as nn

# Simple Conv Layer
conv_layer = nn.Conv2d(
    in_channels=3,      # RGB
    out_channels=16,    # Number of filters
    kernel_size=3,      # Filter size (3x3)
    stride=1,
    padding=1           # Same padding
)

input_tensor = torch.randn(8, 3, 224, 224)  # Batch of 8 images (224x224 RGB)
output_tensor = conv_layer(input_tensor)

print(output_tensor.shape)  # (8, 16, 224, 224)`}</pre>
    </div>

    <h3 className="text-xl font-semibold">3.5 Visualization: ตัวอย่างการทำงาน</h3>
    <p>
      จาก Paper โดย Zeiler & Fergus (2014) พบว่า Filters ในเลเยอร์ล่างของ CNN เรียนรู้ขอบ (edges), ส่วนที่เป็น texture หรือสี ในขณะที่เลเยอร์กลางเริ่มเข้าใจ pattern ที่ซับซ้อนขึ้น เช่น shape หรือ object parts ส่วนเลเยอร์สูงจะเรียนรู้ถึง semantic concept เช่น ใบหน้า หรือวัตถุทั้งชิ้น
    </p>

    <h3 className="text-xl font-semibold">3.6 ประโยชน์เชิงลึกที่ได้จาก CNN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Local Receptive Field:</strong> ทำให้โครงข่ายสามารถมองเห็นบริบทเฉพาะจุดใน Input</li>
      <li><strong>Hierarchical Feature Learning:</strong> ทำให้โมเดลสามารถเรียนรู้ลำดับชั้นของ Feature ได้</li>
      <li><strong>Translation Invariance:</strong> CNN ไม่ไวต่อการเลื่อนตำแหน่งของวัตถุเล็กน้อย</li>
    </ul>

    <h3 className="text-xl font-semibold">3.7 Insight จากงานวิจัย</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>Paper AlexNet (Krizhevsky et al., 2012) แสดงให้เห็นว่า CNN ชนะ ImageNet ด้วยความแม่นยำสูงเมื่อใช้ Convolution Layers ลึก</li>
        <li>Zeiler & Fergus (2014) ทำ Visualization แสดงให้เห็นว่าแต่ละเลเยอร์ของ CNN มีการเรียนรู้ Feature ที่ชัดเจนและซับซ้อนมากขึ้นเรื่อย ๆ</li>
        <li>VGGNet ใช้ Filter 3x3 อย่างต่อเนื่องเพื่อรักษาความละเอียดของ Feature และเพิ่มความลึกของเครือข่าย</li>
      </ul>
    </div>
  </div>
</section>


<section id="activation-functions" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. ฟังก์ชันกระตุ้น (Activation Functions)</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img5} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      ฟังก์ชันกระตุ้น (Activation Functions) เป็นองค์ประกอบสำคัญใน Convolutional Neural Networks (CNNs)
      โดยมีหน้าที่เพิ่มความไม่เชิงเส้น (non-linearity) ให้กับเครือข่าย ทำให้โมเดลสามารถเรียนรู้รูปแบบที่ซับซ้อนได้
      หากไม่มีฟังก์ชันกระตุ้น โมเดลจะเป็นเพียงฟังก์ชันเชิงเส้น แม้จะมีหลายเลเยอร์ก็ตาม
    </p>

    <h3 className="text-xl font-semibold">4.1 ReLU: Rectified Linear Unit</h3>
    <p>
      ReLU เป็นฟังก์ชันกระตุ้นที่ถูกใช้งานมากที่สุดในสถาปัตยกรรม CNN ยุคใหม่ เนื่องจากใช้งานง่ายและให้ผลลัพธ์ที่ดี
      โดยนิยามดังนี้:
    </p>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
      <code className="block text-sm">f(x) = max(0, x)</code>
    </div>
    <p>
      ReLU สร้างความเป็นสปาร์ส (sparsity) โดยทำให้ค่าที่น้อยกว่า 0 กลายเป็นศูนย์ ช่วยลดความซับซ้อนในการคำนวณ
      และยังช่วยลดปัญหา vanishing gradient ที่มักเกิดกับ sigmoid และ tanh
    </p>

    <h3 className="text-xl font-semibold">4.2 ทำไม ReLU ถึงเหมาะกับ CNN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>คำนวณได้รวดเร็ว → ส่งผลให้ฝึกโมเดลได้เร็วขึ้น</li>
      <li>ลดโอกาสเกิดปัญหา vanishing gradient</li>
      <li>เพิ่มความไม่เชิงเส้นในขณะที่ยังเรียบง่าย</li>
    </ul>

    <h3 className="text-xl font-semibold">4.3 ReLU แบบต่าง ๆ</h3>
    <p>
      นักวิจัยจากสถาบันชั้นนำ เช่น Stanford และ MIT ได้เสนอ ReLU หลากหลายรูปแบบ เพื่อแก้ข้อจำกัดของ ReLU แบบดั้งเดิม:
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">Leaky ReLU</h4>
        <p className="text-sm">อนุญาตให้ค่าลบมี slope เล็กน้อย เพื่อป้องกัน neuron ตาย:</p>
        <code className="text-sm block">f(x) = x ถ้า x มากกว่า 0 มิฉะนั้น αx (เช่น α = 0.01)</code>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">Parametric ReLU (PReLU)</h4>
        <p className="text-sm">คล้าย Leaky ReLU แต่เรียนรู้ค่า α ได้ระหว่างการฝึก</p>
      </div>
      <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">Exponential Linear Unit (ELU)</h4>
        <p className="text-sm">เปลี่ยนค่าลบแบบ smooth:</p>
        <code className="text-sm block">f(x) = x ถ้า x ≥ 0 มิฉะนั้น α(exp(x) - 1)</code>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">GELU (Gaussian Error Linear Unit)</h4>
        <p className="text-sm">ใช้ใน BERT และ Transformer; มีลักษณะ probabilistic gating:</p>
        <code className="text-sm block">f(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))</code>
      </div>
    </div>

    <h3 className="text-xl font-semibold">4.4 หลักฐานจากงานวิจัย</h3>
    <div className="bg-yellow-100 dark:bg-yellow-800 p-5 rounded-lg border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>He et al. (2015): ReLU ช่วยให้ฝึก CNN ที่ลึกมาก เช่น ResNet ได้รวดเร็วขึ้น</li>
        <li>Google Brain และ OpenAI: GELU ทำงานได้ดีกว่าในสถาปัตยกรรม Transformer</li>
        <li>Stanford CS231n: ReLU ให้ผลการลู่เข้า (convergence) ที่ดีกว่า sigmoid และ tanh</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">4.5 ปัญหา ReLU และวิธีแก้</h3>
    <p>
      ReLU มีปัญหาที่รู้จักกันดีคือ "dying ReLU" ซึ่ง neuron จะให้ค่าออกมาเป็นศูนย์เสมอและหยุดเรียนรู้ Leaky ReLU และ PReLU ถูกเสนอเพื่อให้ยังมี gradient เล็ก ๆ เมื่อ x &lt; 0
    </p>

    <h3 className="text-xl font-semibold">4.6 ตารางเปรียบเทียบ</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-800">
            <th className="border px-4 py-2">ฟังก์ชัน</th>
            <th className="border px-4 py-2">สูตร</th>
            <th className="border px-4 py-2">ข้อดี</th>
            <th className="border px-4 py-2">ข้อเสีย</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">ReLU</td>
            <td className="border px-4 py-2">max(0, x)</td>
            <td className="border px-4 py-2">รวดเร็ว เรียบง่าย</td>
            <td className="border px-4 py-2">neuron อาจหยุดทำงาน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Leaky ReLU</td>
            <td className="border px-4 py-2">x ถ้า x&gt;0 มิฉะนั้น αx</td>
            <td className="border px-4 py-2">แก้ dying ReLU</td>
            <td className="border px-4 py-2">ต้องปรับค่า α</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GELU</td>
            <td className="border px-4 py-2">Probabilistic</td>
            <td className="border px-4 py-2">Smooth, แข็งแกร่ง</td>
            <td className="border px-4 py-2">ช้ากว่าเล็กน้อย</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">4.7 แนวทางการใช้งาน</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ ReLU เป็น default สำหรับ CNN ทั่วไป</li>
      <li>โมเดลลึก เช่น ResNet อาจใช้ GELU หรือ PReLU เพื่อประสิทธิภาพที่ดีกว่า</li>
      <li>เมื่อพบปัญหา neuron ตาย → ลอง Leaky ReLU</li>
      <li>ทดสอบหลาย activation โดยใช้ cross-validation สำหรับงานใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold">4.8 ตัวอย่างการใช้ใน PyTorch</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.leaky = nn.LeakyReLU(0.01)

    def forward(self, x):
        x1 = self.relu(self.conv(x))
        x2 = self.leaky(self.conv(x))
        x3 = self.gelu(self.conv(x))
        return x1, x2, x3`}
    </pre>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      ฟังก์ชันกระตุ้นทำหน้าที่กำหนดขอบเขตการตัดสินใจเชิงไม่เชิงเส้นใน CNN โดยส่งผลต่อความเร็ว เสถียรภาพ และประสิทธิภาพในการฝึก การเข้าใจพฤติกรรมเชิงคณิตศาสตร์และผลลัพธ์จากงานวิจัยช่วยให้สามารถออกแบบโมเดล CNN ที่แข็งแกร่งและใช้งานได้จริงมากขึ้น
    </p>
  </div>
</section>


<section id="pooling-layers" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Pooling Layers</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img6} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <p>
      Pooling คือกระบวนการลดขนาดของ Feature Maps ใน Convolutional Neural Networks (CNN) เพื่อควบคุมจำนวนพารามิเตอร์และคอมพิวต์ เพิ่มความสามารถของโมเดลในการเรียนรู้แบบ Translation Invariant และลดการ Overfitting ผ่านการเลือก Feature ที่สำคัญเท่านั้น
    </p>

    <h3 className="text-xl font-semibold">5.1 หลักการทำงานของ Pooling</h3>
    <p>
      Pooling ทำงานโดยการเคลื่อนหน้าต่างเล็ก ๆ (เช่น 2x2, 3x3) ไปทั่ว Feature Map แล้วใช้ฟังก์ชันเฉพาะในการเลือกค่าจากภายในหน้าต่างนั้น เช่น การเลือกค่ามากที่สุด (Max) หรือค่าเฉลี่ย (Average)
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <pre className="text-sm overflow-x-auto">{`
Max Pooling:
    f(x) = max(x_1, x_2, ..., x_n)

Average Pooling:
    f(x) = (1/n) * Σ x_i
      `}</pre>
    </div>

    <h3 className="text-xl font-semibold">5.2 Max Pooling</h3>
    <p>
      Max Pooling คือการเลือกค่าสูงสุดภายในหน้าต่างย่อย ซึ่งช่วยให้โมเดลเน้น Feature ที่สำคัญที่สุด เช่น edge หรือ blob ที่ชัดเจนในภาพ และลด Feature ที่อ่อนกว่า
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>ลดขนาดของข้อมูลและความซับซ้อนในการคำนวณ</li>
      <li>เพิ่ม Robustness ต่อ Noise เล็กน้อยในข้อมูลภาพ</li>
      <li>ทำให้โมเดลมี Invariance ต่อการแปล (Translation Invariance)</li>
    </ul>

    <h3 className="text-xl font-semibold">5.3 Average Pooling</h3>
    <p>
      Average Pooling คำนวณค่าเฉลี่ยภายในหน้าต่าง ซึ่งจะช่วยลดความแตกต่างรุนแรงระหว่าง Feature และส่งผลให้โมเดลเข้าใจภาพรวมได้มากขึ้น แต่จะสูญเสีย Feature ที่เด่นชัด
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>เหมาะสำหรับงานที่ไม่ต้องการเน้น Feature เฉพาะจุด</li>
      <li>ใช้งานในบางงานเช่น Signal Processing หรือ Texture Recognition</li>
    </ul>

    <h3 className="text-xl font-semibold">5.4 Hyperparameters ของ Pooling Layer</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Window Size:</strong> กำหนดขนาดของกล่องที่ใช้ Pooling เช่น 2x2, 3x3</li>
      <li><strong>Stride:</strong> ระยะการเลื่อนของหน้าต่าง Pooling ในแต่ละแกน</li>
      <li><strong>Padding:</strong> การเติมขอบของ Feature Map เพื่อลดการสูญเสียข้อมูล</li>
    </ul>

    <h3 className="text-xl font-semibold text-center">5.5 ตัวอย่างภาพประกอบ</h3>
    <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img7} />
        </div>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-4 rounded-lg text-center">
        <p className="text-sm font-medium">ก่อนทำ Max Pooling</p>
        
      </div>
      <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-4 rounded-lg text-center">
        <p className="text-sm font-medium">หลังทำ Max Pooling (2x2, stride=2)</p>
        
      </div>
    </div>

    <h3 className="text-xl font-semibold">5.6 ข้อควรระวังในการใช้ Pooling</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การเลือกขนาด Stride ใหญ่เกินไป อาจทำให้ข้อมูลสูญหายมากเกิน</li>
      <li>Pooling หลายครั้งอาจทำให้ Spatial Resolution ต่ำจนเกินไป</li>
      <li>ในบางงาน เช่น Segmentation ควรเลือกวิธีอื่นแทน เช่น Dilated Convolution</li>
    </ul>

    <h3 className="text-xl font-semibold">5.7 เทคนิคสมัยใหม่: การแทน Pooling ด้วย Strided Convolution</h3>
    <p>
      โมเดลรุ่นใหม่ เช่น ResNet, MobileNet ใช้ Strided Convolution แทน Pooling เพื่อให้การเรียนรู้เกิดขึ้นในกระบวนการลดขนาด Feature Map โดยตรง ซึ่งช่วยให้โมเดลสามารถเรียนรู้การลดข้อมูลอย่างเหมาะสมมากขึ้น
    </p>

    <pre className="bg-gray-800 text-white p-4 rounded-xl text-sm overflow-x-auto">
{`# ตัวอย่างการใช้ Strided Convolution แทน Pooling ใน PyTorch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # ลดขนาดเหมือน Max Pooling
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
)`}
    </pre>

    <h3 className="text-xl font-semibold">5.8 งานวิจัยที่เกี่ยวข้อง</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>LeCun et al. (1998): LeNet ใช้ Average Pooling ในงาน Handwritten Digit Recognition</li>
        <li>Zeiler & Fergus (2014): งาน Visualization ชี้ว่า Max Pooling ช่วยรักษา Feature สำคัญไว้ได้ดี</li>
        <li>He et al. (2016): ResNet ใช้ Strided Convolution แทน Pooling บางส่วนเพื่อเรียนรู้ Feature โดยตรง</li>
        <li>Springenberg et al. (2015): All-Convolutional Network ใช้ Strided Convolution ทั้งหมดแทน Max Pooling</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">5.9 Insight เชิงลึก</h3>
    <p>
      การเลือกใช้ Pooling Layer มีผลโดยตรงต่อความสามารถในการ Generalize และประสิทธิภาพของโมเดล โดยเฉพาะในงาน Computer Vision เช่น Image Classification, Detection และ Segmentation ควรเลือกกลยุทธ์การ Pooling ให้เหมาะกับลักษณะของข้อมูลและความลึกของโมเดล
    </p>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-5 rounded-xl">
        <h4 className="font-semibold mb-2">ข้อดีของ Pooling</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ลด Overfitting โดยลดความละเอียดเกินจำเป็นของ Feature</li>
          <li>เพิ่ม Invariance ต่อการแปลและการเปลี่ยนขนาด</li>
          <li>ลดจำนวนพารามิเตอร์และเวลาในการฝึกโมเดล</li>
        </ul>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-5 rounded-xl">
        <h4 className="font-semibold mb-2">ข้อจำกัดของ Pooling</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ข้อมูลบางส่วนอาจสูญหายอย่างถาวร</li>
          <li>Spatial Relationship อาจถูกละเลย</li>
          <li>อาจไม่เหมาะกับงานที่ต้องการความละเอียดสูง เช่น Medical Imaging</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">5.10 บทสรุป</h3>
    <p>
      Pooling คือกลไกสำคัญในการลดขนาดข้อมูลเชิงพื้นที่ใน CNN และเพิ่มความสามารถในการจัดการภาพหลากหลายลักษณะ การเลือกใช้ Max หรือ Average Pooling ขึ้นอยู่กับลักษณะของงาน แต่ในสถาปัตยกรรมใหม่ที่ต้องการความยืดหยุ่นสูงขึ้น การแทน Pooling ด้วย Strided Convolution หรือ Adaptive Pooling จึงกลายเป็นทางเลือกที่น่าสนใจ
    </p>
  </div>
</section>

<section id="fully-connected" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. การเชื่อมต่อ Fully Connected</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img8} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การเชื่อมต่อ Fully Connected (FC) เป็นขั้นตอนสุดท้ายของสถาปัตยกรรม Convolutional Neural Networks (CNN) ที่ทำหน้าที่เปลี่ยน Feature Maps ซึ่งได้จากกระบวนการ Convolution และ Pooling ให้เป็นเวกเตอร์แบน (Flattened Vector) แล้วป้อนเข้าสู่ Dense Layers เพื่อทำการตัดสินใจสุดท้าย เช่น การจำแนกประเภท (Classification) หรือการทำนาย (Regression)
    </p>

    <h3 className="text-xl font-semibold">กระบวนการ Flatten</h3>
    <p>
      ขั้นตอนแรกของการเชื่อมต่อ FC คือการนำ Feature Map ที่มีหลาย Channel และมีลักษณะเป็น Tensor 3 มิติ (H x W x C) มาแปลงเป็นเวกเตอร์ 1 มิติ เพื่อให้สามารถป้อนเข้าสู่ Fully Connected Layer ได้
    </p>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`# PyTorch Example
import torch
import torch.nn as nn

x = torch.randn(16, 64, 7, 7)  # Batch of 16, 64 filters, 7x7 spatial dimensions
flatten = nn.Flatten()
x_flat = flatten(x)
print(x_flat.shape)  # Output: torch.Size([16, 3136])`}
    </pre>

    <h3 className="text-xl font-semibold">การเชื่อมต่อ Dense Layer</h3>
    <p>
      เมื่อได้เวกเตอร์แบนแล้ว จะถูกเชื่อมต่อกับ Dense Layer (Linear Layer) โดยแต่ละ Node ใน Layer ดังกล่าวเชื่อมกับทุกค่าในเวกเตอร์ Input ส่งผลให้การเรียนรู้มีลักษณะ Global และสามารถแยกแยะ Feature ได้ในระดับสูงขึ้น
    </p>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`# เชื่อมต่อกับ Fully Connected Layer
fc1 = nn.Linear(3136, 128)
fc2 = nn.Linear(128, 10)  # สมมุติว่าเป็นการจำแนกภาพออกเป็น 10 ประเภท

out = fc1(x_flat)
out = torch.relu(out)
out = fc2(out)`}
    </pre>

    <h3 className="text-xl font-semibold">ทำไม Fully Connected Layer ใน CNN มีจำนวนน้อย</h3>
    <p>
      เมื่อเปรียบเทียบกับ Multilayer Perceptron (MLP) จะพบว่า CNN มักใช้ Fully Connected Layers เพียง 1-3 ชั้นเท่านั้น เนื่องจากขั้นตอน Convolution และ Pooling ได้เรียนรู้ Feature ในเชิงลึกไปแล้ว การเพิ่ม FC มากเกินไปจะเพิ่มพารามิเตอร์จำนวนมากโดยไม่จำเป็น และเพิ่มความเสี่ยงต่อ Overfitting
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">MLP (Fully Connected Network)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ทุกชั้นเชื่อมต่อแบบ Dense</li>
          <li>เรียนรู้ Feature จากข้อมูลแบบแบน</li>
          <li>มีจำนวนพารามิเตอร์มาก</li>
        </ul>
      </div>
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">CNN + FC Layer</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Convolution Layer ดึง Feature</li>
          <li>Fully Connected เฉพาะตอนจบ</li>
          <li>พารามิเตอร์น้อยกว่า MLP มาก</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">การใช้ Softmax เพื่อทำ Classification</h3>
    <p>
      ในงาน Classification มักจะใช้ Softmax Layer เป็นชั้นสุดท้ายเพื่อแปลงค่า Logits ที่ได้จาก Fully Connected Layer ให้กลายเป็นความน่าจะเป็นในแต่ละคลาส:
    </p>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`# ตัวอย่างการใช้ Softmax
softmax = nn.Softmax(dim=1)
probs = softmax(out)
print(probs.shape)  # torch.Size([16, 10])`}
    </pre>

    <h3 className="text-xl font-semibold">Insight เชิงลึกจากงานวิจัย</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>LeNet-5 (LeCun, 1998) ใช้ FC 2 ชั้นสุดท้ายเพื่อทำ Classification บน MNIST</li>
        <li>AlexNet (2012) ใช้ FC 3 ชั้น โดยมี Dropout เพื่อป้องกัน Overfitting</li>
        <li>VGGNet ใช้ FC Layer 4096 Neurons ก่อนสุดท้ายและใช้ Softmax เป็น Output</li>
        <li>ResNet ลดขนาดของ FC Layer เหลือเพียง 1 ชั้นก่อน Softmax เพื่อลดพารามิเตอร์</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ Fully Connected Layers</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>จำนวนพารามิเตอร์เพิ่มขึ้นอย่างรวดเร็วตามขนาดของ Input Vector</li>
      <li>ทำให้โมเดล Overfit ได้ง่ายหากไม่มีการ Regularization เช่น Dropout หรือ Weight Decay</li>
      <li>ต้องแน่ใจว่า Flatten แล้วมีจำนวน Node ที่เหมาะสมก่อนเข้าสู่ FC Layer</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      Fully Connected Layers ทำหน้าที่เปลี่ยน Feature ที่ CNN เรียนรู้ให้กลายเป็นการตัดสินใจขั้นสุดท้ายในโมเดล Deep Learning โดยมักจะเชื่อมต่อแบบ Dense เพียง 1–2 ชั้น เพื่อหลีกเลี่ยง Overfitting และลดจำนวนพารามิเตอร์ โดยเฉพาะในงานที่ข้อมูลมีขนาดใหญ่หรือจำนวนคลาสมาก การออกแบบ FC Layer ที่เหมาะสมยังคงเป็นหัวใจสำคัญในการเชื่อมโยง Feature เข้ากับเป้าหมายของโมเดล
    </p>
  </div>
</section>


<section id="cnn-learns" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. ความเข้าใจเชิงลึก: CNN เรียนรู้อะไร</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img9} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Convolutional Neural Networks (CNNs) สามารถเรียนรู้ลำดับลักษณะ (hierarchical features) ที่เพิ่มความซับซ้อนจากชั้นล่างไปยังชั้นบน โดยที่ไม่จำเป็นต้องมีการออกแบบ feature extraction แบบ manual ซึ่งเป็นการเปลี่ยนแปลงแนวทางของ Computer Vision จากยุคก่อนปี 2010 อย่างสิ้นเชิง งานของ Zeiler & Fergus (2013) และ Yosinski et al. (2014) แสดงให้เห็นภาพชัดเจนว่าแต่ละเลเยอร์ของ CNN มีลักษณะการตอบสนองที่แตกต่างกัน
    </p>

    <h3 className="text-xl font-semibold">7.1 ชั้นล่าง: Feature เบื้องต้น (Edges, Textures)</h3>
    <p>
      ชั้นล่างของ CNN มักเรียนรู้ลักษณะที่เรียบง่าย เช่น ขอบ (edges), เส้นตรง, และ texture patterns การเรียนรู้ลักษณะเหล่านี้สอดคล้องกับการทำงานของ V1 ในสมองมนุษย์ (Hubel & Wiesel, 1962) ซึ่งมี neuron ที่ตอบสนองต่อเส้นตรงในทิศทางต่าง ๆ
    </p>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">ตัวอย่างฟิลเตอร์</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Edge Detector (Sobel, Prewitt)</li>
          <li>Texture Filters เช่น Gabor, DoG</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">การตอบสนองต่อภาพ</h4>
        <p className="text-sm">เส้นขอบ แนวตั้ง แนวนอน สีที่ต่างกันอย่างชัดเจน</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">7.2 ชั้นกลาง: Shape & Part Detectors</h3>
    <p>
      เลเยอร์กลาง ๆ ของ CNN เริ่มเรียนรู้ shape ที่ซับซ้อนขึ้น เช่น วงกลม, เส้นโค้ง, หรือลวดลายเฉพาะในบริบทของวัตถุ งานของ Zeiler (2014) แสดงว่า layer กลางสามารถตรวจจับ features ที่สื่อถึงส่วนหนึ่งของวัตถุ เช่น หู ตา หรือขาของสัตว์ เป็นต้น
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-xl">
      <ul className="list-disc pl-6 text-sm">
        <li>Neurons ตอบสนองต่อ pattern แบบกำหนดเฉพาะ</li>
        <li>Filters เริ่มมีลักษณะของการรวม edge หลายมุมเข้าด้วยกัน</li>
        <li>เรียนรู้บริบทของพื้นผิวและ shape โดยไม่จำเป็นต้องสั่งให้รู้จักล่วงหน้า</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">7.3 ชั้นบน: Semantic Features</h3>
    <p>
      ในชั้นบนสุดของ CNN (ก่อน fully connected layer) โมเดลจะเริ่มเรียนรู้ semantic features เช่น วัตถุทั้งตัว หรือภาพที่สื่อความหมาย เช่น “แมว”, “รถยนต์”, หรือ “ใบหน้า” โดย neuron ในเลเยอร์นี้จะตอบสนองต่อภาพทั้งภาพที่มีลักษณะเฉพาะ งานของ Yosinski และทีมงาน DeepDream ของ Google ได้แสดงให้เห็นว่า CNN สามารถสร้างภาพที่กระตุ้น neuron ให้ตอบสนองสูงสุด ซึ่งชี้ให้เห็นว่ามีการเรียนรู้เชิงแนวคิด (concept-level learning)
    </p>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-2">ตัวอย่าง Feature Map</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ตรวจจับวัตถุที่คุ้นเคย</li>
          <li>ตัดคลาสโดยใช้ feature ที่สื่อความหมาย</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-2">Visualization Techniques</h4>
        <p className="text-sm">DeepDream, Grad-CAM, Feature Inversion</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">7.4 งานวิจัยสนับสนุน</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>Zeiler & Fergus (2014): ใช้ DeconvNet แสดงให้เห็นว่า CNN มีลำดับ feature hierarchy ชัดเจน</li>
        <li>Yosinski et al. (2015): แสดงให้เห็นว่า feature สามารถถ่ายโอน (transfer) ไปยังงานอื่นได้</li>
        <li>Olah et al. (Distill.pub): เปิดเผย feature ในทุก layer ด้วยภาพ Visualization แบบ high-res</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">7.5 ข้อสังเกตเชิงลึก</h3>
    <p>
      CNN ไม่ได้ถูกออกแบบให้เข้าใจภาพโดยตรง แต่เรียนรู้จากตัวอย่างซ้ำ ๆ จนเกิดการ encode ความสัมพันธ์ระหว่าง pattern กับ class โดยอัตโนมัติ ข้อมูลที่ผ่าน layer แรกมีรายละเอียดเยอะและ abstract น้อย แต่เมื่อผ่านไปแต่ละ layer feature จะ abstract มากขึ้นและเฉพาะเจาะจงกับงานมากขึ้น
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-green-100 dark:bg-green-900 p-5 rounded-lg">
        <h4 className="font-semibold mb-2">Hierarchical Feature Learning</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Edge ➝ Part ➝ Object</li>
          <li>Simple ➝ Complex ➝ Semantic</li>
        </ul>
      </div>
      <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-lg">
        <h4 className="font-semibold mb-2">ประโยชน์</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ไม่ต้องทำ manual feature engineering</li>
          <li>เหมาะกับข้อมูลภาพขนาดใหญ่</li>
          <li>สามารถ generalize ไปยังงานอื่นได้</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">7.6 สรุป</h3>
    <p>
      CNN สามารถเรียนรู้ feature ได้ในลักษณะลำดับชั้น ตั้งแต่ edge ที่เรียบง่ายไปจนถึง semantic representation ที่ซับซ้อนในชั้นบนสุด ความสามารถนี้ช่วยให้ CNN ประสบความสำเร็จอย่างกว้างขวางในงาน Image Classification, Object Detection, และอื่น ๆ อีกมาก โดยไม่ต้องพึ่ง handcrafted features เหมือนในอดีต
    </p>
  </div>
</section>

<section id="cnn-strengths" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. จุดแข็งของ CNN</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img10} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Convolutional Neural Networks (CNNs) ได้กลายเป็นรากฐานของงานด้าน Computer Vision และ Image Processing เนื่องจากมีโครงสร้างที่สามารถเรียนรู้ข้อมูลเชิงภาพอย่างมีประสิทธิภาพ งานวิจัยของ LeCun, Bengio และ Hinton (Nature, 2015) รวมถึงสถาบันชั้นนำอย่าง Stanford และ MIT ได้ยืนยันว่า CNN มีความสามารถพิเศษในการวิเคราะห์ข้อมูลเชิงพื้นที่ที่โมเดลทั่วไปไม่สามารถทำได้อย่างมีประสิทธิภาพเท่าเทียม
    </p>

    <h3 className="text-xl font-semibold">8.1 Weight Sharing</h3>
    <p>
      CNNs ใช้แนวคิด Weight Sharing ในการประมวลผลข้อมูลภาพ กล่าวคือ ฟิลเตอร์เดียวกันถูกใช้เลื่อนผ่านภาพทั้งภาพ ซึ่งแตกต่างจาก MLP ที่แต่ละพิกเซลมีน้ำหนักเฉพาะของตนเอง การทำ Weight Sharing นี้ช่วยลดจำนวนพารามิเตอร์ลงอย่างมหาศาล และลดปัญหาการ Overfitting ในขณะที่ยังสามารถเรียนรู้ Feature ได้อย่างมีประสิทธิภาพ
    </p>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <pre>{`
# เปรียบเทียบจำนวนพารามิเตอร์ระหว่าง MLP กับ CNN บน input 32x32x3
# MLP แบบ Fully Connected:
params_mlp = (32*32*3) * 1000  # สมมติว่า hidden layer มี 1000 units
print("MLP Parameters:", params_mlp)

# CNN ที่ใช้ 32 filters ขนาด 3x3:
params_cnn = (3*3*3)*32  # 3 channels, 32 filters
print("CNN Parameters:", params_cnn)
      `}</pre>
    </div>

    <h3 className="text-xl font-semibold">8.2 Local Receptive Fields</h3>
    <p>
      CNNs ออกแบบให้แต่ละนิวรอนในเลเยอร์ถัดไปเชื่อมต่อกับเฉพาะบริเวณเล็ก ๆ ของเลเยอร์ก่อนหน้า เรียกว่า Receptive Field ซึ่งช่วยให้โมเดลเรียนรู้ Feature ที่เป็นบริบทท้องถิ่นของภาพ เช่น ขอบ เส้น รูปร่างพื้นฐาน โดยไม่จำเป็นต้องเรียนรู้ภาพทั้งหมดในครั้งเดียว
    </p>

    <div className="flex flex-col md:flex-row gap-6">
      <div className="bg-green-50 dark:bg-green-900 p-5 rounded-xl flex-1">
        <h4 className="font-semibold mb-2">ข้อดี</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ประหยัดหน่วยความจำ</li>
          <li>โฟกัสที่บริบทเฉพาะจุด → ช่วยให้เข้าใจโครงสร้างภาพได้ดีขึ้น</li>
          <li>ลดความซับซ้อนของการเรียนรู้</li>
        </ul>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-5 rounded-xl flex-1">
        <h4 className="font-semibold mb-2">ข้อจำกัด</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ไม่เข้าใจความสัมพันธ์ระยะไกลโดยตรง</li>
          <li>ต้องใช้การซ้อนเลเยอร์หลายชั้นเพื่อรวม global context</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">8.3 Hierarchical Feature Learning</h3>
    <p>
      โครงสร้างของ CNN อนุญาตให้โมเดลเรียนรู้ลำดับชั้นของ Features จากภาพในลักษณะคล้ายกับวิธีการรับรู้ของสมองมนุษย์:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>เลเยอร์ต้น → edge, texture</li>
      <li>เลเยอร์กลาง → parts of object</li>
      <li>เลเยอร์ลึก → semantic object (face, cat, car)</li>
    </ul>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="font-semibold mb-2">Insight จาก AlexNet (2012)</h4>
      <p className="text-sm">
        จากการวิเคราะห์เลเยอร์ของ AlexNet พบว่าเลเยอร์ล่าง ๆ จะตอบสนองต่อลักษณะพื้นฐาน เช่น เส้นตรง ขอบ และสี ขณะที่เลเยอร์บนจะตอบสนองต่อ pattern ที่ซับซ้อนขึ้น เช่น ใบหน้าหรือวัตถุเฉพาะ ทำให้ CNNs มีความสามารถในการแยกแยะ Feature แบบลำดับชั้นได้อย่างยอดเยี่ยม
      </p>
    </div>

    <h3 className="text-xl font-semibold">8.4 Translation Invariance</h3>
    <p>
      การเลื่อนฟิลเตอร์ผ่านภาพทำให้ CNNs สามารถตรวจจับวัตถุในตำแหน่งต่าง ๆ ของภาพได้โดยไม่สูญเสียความแม่นยำ ซึ่งต่างจาก MLP ที่จะต้องเห็นข้อมูลในตำแหน่งนั้นเป๊ะ ๆ CNNs จึงสามารถ generalize ได้ดีในภาพที่มีการเลื่อนหรือปรับตำแหน่งของวัตถุ
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 p-6 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <h4 className="font-semibold mb-2">ตัวอย่างเชิงภาพ</h4>
      <p className="text-sm">
        เมื่อแสดงภาพแมวไว้ตรงกลางและมุมของรูป CNN สามารถจับ Feature ได้ทั้งสองกรณีได้อย่างแม่นยำ ด้วยการแชร์ฟิลเตอร์ทั่วทั้งภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold">8.5 Parameter Efficiency</h3>
    <p>
      การใช้ Weight Sharing และ Local Connectivity ทำให้ CNNs ใช้พารามิเตอร์น้อยกว่าโมเดลทั่วไปหลายเท่า ตัวอย่างเช่น ResNet-50 มีพารามิเตอร์ประมาณ 25 ล้านตัว ในขณะที่ MLP ที่มีจำนวน layer เท่ากันจะมีพารามิเตอร์มากกว่า 100 ล้านตัว
    </p>

    <h3 className="text-xl font-semibold">8.6 Generalization ที่แข็งแกร่ง</h3>
    <p>
      CNNs สามารถเรียนรู้จากชุดข้อมูลจำนวนมาก และสามารถนำไปใช้กับภาพที่ไม่เคยเห็นมาก่อนได้ดี นี่เป็นสาเหตุหลักที่ทำให้ CNN กลายเป็นหัวใจของระบบเช่น Self-driving Cars, Medical Imaging และ Face Recognition
    </p>

    <h3 className="text-xl font-semibold">8.7 Robust to Noise</h3>
    <p>
      ด้วยความสามารถในการเรียนรู้ Feature ที่ทนต่อการเปลี่ยนแปลง CNNs จึงสามารถทำงานได้ดีแม้มี Noise หรือ Distortion ในภาพ ซึ่งช่วยเพิ่มความน่าเชื่อถือในระบบที่ใช้งานในสภาพแวดล้อมจริง
    </p>

    <h3 className="text-xl font-semibold">สรุปภาพรวม</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Weight Sharing ลดพารามิเตอร์และเพิ่มประสิทธิภาพการเรียนรู้</li>
      <li>Local Receptive Fields ช่วยให้เข้าใจโครงสร้างภาพ</li>
      <li>Hierarchical Learning ทำให้แยกแยะ Feature ตั้งแต่พื้นฐานถึง semantics ได้</li>
      <li>Translation Invariance ส่งเสริมการ generalize ข้ามตำแหน่ง</li>
      <li>ความทนทานต่อ Noise และความสามารถในการใช้งานในโลกจริง</li>
    </ul>
  </div>
</section>

<section id="cnn-limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ข้อจำกัดของ CNN</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img11} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    
    <p>
      แม้ Convolutional Neural Networks (CNNs) จะเป็นเทคโนโลยีหลักในด้านการประมวลผลภาพสมัยใหม่ แต่ก็มีข้อจำกัดที่ต้องพิจารณาอย่างรอบคอบเมื่อนำไปประยุกต์ใช้กับงานที่ซับซ้อนหรือมีความต้องการเฉพาะด้าน ข้อจำกัดเหล่านี้มีทั้งด้านการประมวลผล ความสามารถในการเรียนรู้ระยะยาว ไปจนถึงข้อจำกัดของโครงสร้างพื้นฐานเอง
    </p>

    <h3 className="text-xl font-semibold">9.1 ความอ่อนไหวต่อ Spatial Transformation</h3>
    <p>
      CNN ไม่สามารถจัดการกับการหมุน (rotation), การเลื่อน (translation), หรือการเปลี่ยนมุมมอง (perspective change) ได้ดีหากไม่ได้มีการสอนอย่างเพียงพอ แม้ว่า pooling จะช่วยให้เกิด invariance บางส่วน แต่ก็ไม่เพียงพอต่อการ generalize กับข้อมูลที่มีการเปลี่ยนแปลงลักษณะนี้
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-red-100 dark:bg-red-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ตัวอย่างความอ่อนไหว</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>โมเดลจำแนกรูปแมวหมุน 45 องศาไม่ถูกต้อง</li>
          <li>กล้องเปลี่ยนมุมเพียงเล็กน้อย ส่งผลให้การตรวจจับวัตถุล้มเหลว</li>
        </ul>
      </div>
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">แนวทางแก้ไข</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Data Augmentation เพื่อจำลองการเปลี่ยนแปลงมุมมอง</li>
          <li>ใช้ Spatial Transformer Networks (Jaderberg et al., 2015)</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">9.2 ขาดความสามารถในการเรียนรู้ Long-Range Dependencies</h3>
    <p>
      ด้วยลักษณะการประมวลผลแบบ local receptive fields ทำให้ CNN มีข้อจำกัดในการเข้าใจความสัมพันธ์เชิงลึกระหว่างส่วนที่อยู่ไกลกันในภาพ การใช้ kernel ขนาดเล็กแม้จะลดพารามิเตอร์แต่ก็ลดบริบทเชิงภาพด้วย ส่งผลให้ไม่สามารถวิเคราะห์ภาพที่ต้องอาศัย global context ได้ดี
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="font-semibold text-lg mb-2">Insight จากงานวิจัย</h4>
      <p className="text-sm">
        งานของ Dosovitskiy et al. (2020) ใน Vision Transformers ระบุว่า CNN มี receptive field ที่จำกัดและต้องการชั้นลึกเพื่อรวบรวมข้อมูล global context ซึ่งส่งผลต่อทั้งเวลาและหน่วยความจำในการฝึกโมเดล
      </p>
    </div>

    <h3 className="text-xl font-semibold">9.3 ข้อจำกัดด้านข้อมูลและการ Overfitting</h3>
    <p>
      CNN ต้องการข้อมูลจำนวนมากเพื่อเรียนรู้ feature representation อย่างมีประสิทธิภาพ หากข้อมูลไม่เพียงพอ หรือมีลักษณะ imbalanced จะนำไปสู่ overfitting ได้ง่าย แม้จะมีเทคนิค Regularization เช่น Dropout หรือ Data Augmentation แต่ก็ไม่สามารถทดแทนข้อมูลที่หลากหลายได้ทั้งหมด
    </p>

    <div className="grid md:grid-cols-3 gap-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">เมื่อข้อมูลน้อย</h4>
        <p className="text-sm">โมเดลจดจำ noise แทน pattern ที่แท้จริง</p>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">เมื่อข้อมูล unbalanced</h4>
        <p className="text-sm">โมเดลลำเอียงไปหากลุ่มข้อมูลที่มีมาก</p>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">แนวทางแก้ไข</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Pretrained Model</li>
          <li>ใช้ Transfer Learning</li>
          <li>เพิ่ม Synthetic Data</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">9.4 ความซับซ้อนด้านการออกแบบโครงสร้าง</h3>
    <p>
      การเลือกจำนวนเลเยอร์, kernel size, stride, padding, จำนวน filters ฯลฯ ต้องอาศัยความเชี่ยวชาญหรือการทดลองซ้ำหลายรอบ ไม่มีสูตรตายตัว ทำให้เสียเวลาและทรัพยากรจำนวนมากเพื่อ fine-tune โมเดลให้มีประสิทธิภาพ
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <h4 className="font-semibold text-lg mb-2">Insight: Neural Architecture Search</h4>
      <p className="text-sm">
        งานของ Google AutoML และ Facebook RegNet ได้เสนอแนวทางอัตโนมัติในการออกแบบโครงสร้าง CNN โดยใช้เทคนิคเช่น Reinforcement Learning และ Evolutionary Algorithms เพื่อลดภาระของนักวิจัย
      </p>
    </div>

    <h3 className="text-xl font-semibold">9.5 อ่อนไหวต่อ Adversarial Attacks</h3>
    <p>
      CNN สามารถถูกหลอกได้ง่ายโดยการเพิ่ม noise เล็กน้อยที่มนุษย์มองไม่เห็นแต่สามารถเปลี่ยนผลลัพธ์ของโมเดลได้อย่างสิ้นเชิง งานของ Goodfellow et al. (2015) แสดงให้เห็นว่าเพียงแค่การ perturb input image บางพิกเซลก็สามารถทำให้โมเดลจำแนกผิดได้
    </p>

    <div className="bg-red-50 dark:bg-red-900 p-5 rounded-xl border-l-4 border-red-400 dark:border-red-600">
      <h4 className="font-semibold text-lg mb-2">ตัวอย่างโค้ดการโจมตี Fast Gradient Sign Method (FGSM)</h4>
      <pre className="text-sm overflow-x-auto bg-gray-900 text-white p-4 rounded-lg">
{`import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# ตัวอย่างการใช้งาน
model.eval()
image.requires_grad = True
output = model(image)
loss = F.nll_loss(output, label)
model.zero_grad()
loss.backward()
data_grad = image.grad.data
perturbed = fgsm_attack(image, 0.03, data_grad)`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">9.6 ใช้พลังการคำนวณสูง</h3>
    <p>
      การฝึก CNN โดยเฉพาะเครือข่ายที่มีขนาดใหญ่ เช่น ResNet-152, EfficientNet-B7 ต้องใช้ GPU ที่มีหน่วยความจำสูงและใช้เวลาหลายชั่วโมงถึงหลายวัน ทำให้ไม่เหมาะกับการนำไปใช้ในอุปกรณ์ฝังตัว หรืออุปกรณ์ที่มีทรัพยากรจำกัด
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>Training Time สูงขึ้นเมื่อเพิ่มเลเยอร์หรือ resolution</li>
      <li>Deployment ต้องใช้การ Optimize ด้วย TensorRT, ONNX หรือ Pruning</li>
      <li>โมเดลใหญ่ไม่เหมาะกับ real-time inference</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุปข้อจำกัดของ CNN</h3>
    <p>
      แม้ CNN จะประสบความสำเร็จอย่างมากในหลายงาน แต่การเข้าใจข้อจำกัดเหล่านี้มีความสำคัญต่อการออกแบบโมเดลที่เหมาะสมกับข้อมูลและข้อจำกัดของงานจริง รวมถึงการเลือกใช้เทคโนโลยีเสริม เช่น Attention, Capsule Networks หรือ Transformers เพื่อขยายขอบเขตความสามารถของโมเดล
    </p>

  </div>
</section>

<section id="research-insight" className="mb-16 scroll-mt-32 min-h-[400px]">
    <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight จากงานวิจัย</h2>
    <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img12} />
        </div>
    <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
      <p>
        การพัฒนา Convolutional Neural Networks (CNN) ได้รับการขับเคลื่อนจากงานวิจัยระดับโลกในช่วงกว่าทศวรรษที่ผ่านมา โดยเริ่มต้นจากการปฏิวัติวงการด้วย AlexNet (2012) จนถึงการสร้างโมเดลที่สามารถแข่งขันกับมนุษย์ในด้านการจดจำภาพได้ งานวิจัยต่อไปนี้แสดงให้เห็นถึงพัฒนาการของสถาปัตยกรรม CNN และแนวคิดเชิงลึกที่ส่งผลต่ออุตสาหกรรม AI โดยรวม
      </p>

      <h3 className="text-xl font-semibold">10.1 AlexNet (2012): จุดเปลี่ยนของ Deep Learning</h3>
      <p>
        AlexNet โดย Krizhevsky et al. ได้รับรางวัลชนะเลิศ ImageNet Challenge ด้วยความแม่นยำเหนือกว่าผู้เข้าแข่งขันรายอื่นถึงกว่า 10% ซึ่งเป็นการยืนยันว่า CNN เมื่อจับคู่กับ GPU สามารถเรียนรู้ feature ที่ซับซ้อนจากข้อมูลภาพจำนวนมหาศาลได้อย่างมีประสิทธิภาพ
      </p>
      <ul className="list-disc pl-6">
        <li>ใช้ ReLU แทน Sigmoid ทำให้การฝึกเร็วขึ้น</li>
        <li>ใช้ Dropout เป็นครั้งแรกเพื่อลด Overfitting</li>
        <li>ใช้ Data Augmentation อย่างจริงจัง เช่น การ flip, crop</li>
        <li>แบ่ง training ไปยัง 2 GPUs เพื่อความเร็ว</li>
      </ul>

      <h3 className="text-xl font-semibold">10.2 VGGNet (2014): เรียบง่ายแต่ทรงพลัง</h3>
      <p>
        VGGNet จาก Oxford Robotics Institute นำเสนอแนวทางที่เรียบง่ายแต่มีประสิทธิภาพสูง โดยใช้ 3x3 convolution แบบติดกันหลายชั้นเพื่อเพิ่ม non-linearity และเรียนรู้ pattern ที่ลึกมากขึ้น
      </p>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">จุดเด่น</h4>
          <ul className="list-disc pl-5 text-sm">
            <li>โมเดลโครงสร้างเป็นลำดับชั้นที่เข้าใจง่าย</li>
            <li>ใช้ Convolution kernel เล็ก (3x3) ซ้ำ ๆ</li>
            <li>เพิ่ม depth โดยไม่เพิ่มพารามิเตอร์มาก</li>
          </ul>
        </div>
        <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">ข้อสังเกต</h4>
          <ul className="list-disc pl-5 text-sm">
            <li>แม้จะมี performance สูง แต่พารามิเตอร์จำนวนมาก</li>
            <li>ต้องการทรัพยากรมากในการ train และ deploy</li>
          </ul>
        </div>
      </div>

      <h3 className="text-xl font-semibold">10.3 ResNet (2015): แก้ปัญหา Degradation ด้วย Residual Learning</h3>
      <p>
        ResNet (He et al., Microsoft Research) เสนอแนวคิด Residual Connection ที่ช่วยให้สามารถสร้าง Neural Network ที่มีความลึกหลายร้อยชั้นได้ โดยไม่สูญเสียความสามารถในการเรียนรู้
      </p>
      <pre className="bg-gray-900 text-white text-sm p-4 rounded-lg overflow-x-auto">
{`def residual_block(x):
    residual = x
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = ReLU()(x)
    return x`}
      </pre>
      <ul className="list-disc pl-6">
        <li>Residual block ช่วยให้ gradient ไหลผ่านได้ดีในเครือข่ายลึก</li>
        <li>ได้รับรางวัล ImageNet 2015 ด้วย Accuracy สูงสุด</li>
        <li>แนวคิดนี้ถูกนำไปใช้ในทุกสถาปัตยกรรมลึกสมัยใหม่</li>
      </ul>

      <h3 className="text-xl font-semibold">10.4 EfficientNet (2019): การสเกลแบบ Multi-Dimensional</h3>
      <p>
        งานของ Tan & Le (Google Brain) เสนอ Compound Scaling ที่สเกลทั้งความลึก, ความกว้าง, และขนาดของภาพพร้อมกันอย่างเป็นระบบ ส่งผลให้ได้โมเดลที่ประสิทธิภาพดีแต่ใช้ทรัพยากรต่ำกว่าสถาปัตยกรรมเดิม
      </p>
      <ul className="list-disc pl-6">
        <li>ใช้ Neural Architecture Search (NAS) ในการค้นหาโครงสร้าง</li>
        <li>มีความยืดหยุ่นในการ deploy บนอุปกรณ์ที่มีทรัพยากรจำกัด</li>
        <li>กลายเป็น backbone ที่ได้รับความนิยมในการทำ Fine-Tuning</li>
      </ul>

      <h3 className="text-xl font-semibold">10.5 จาก CNN สู่ Vision Transformer (ViT)</h3>
      <p>
        งานจาก Google Research ปี 2020 เสนอว่า CNN แม้จะมี inductive bias ที่ดีต่อการประมวลผลภาพ แต่เมื่อมีข้อมูลขนาดใหญ่มากพอ Transformer ก็สามารถ outperform CNN ได้ งาน Vision Transformer แสดงให้เห็นว่า self-attention สามารถเรียนรู้ spatial structure ได้โดยไม่ต้อง convolution
      </p>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">ข้อค้นพบหลัก</h4>
          <ul className="list-disc pl-5 text-sm">
            <li>ไม่ต้องใช้ convolution ก็สามารถเรียนรู้ representation ได้</li>
            <li>Performance ดีกว่า CNN เมื่อ pre-train บน dataset ใหญ่</li>
          </ul>
        </div>
        <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">ข้อควรระวัง</h4>
          <ul className="list-disc pl-5 text-sm">
            <li>ต้องการ compute สูงมากใน training phase</li>
            <li>ไม่เหมาะกับ dataset ขนาดเล็ก</li>
          </ul>
        </div>
      </div>

      <h3 className="text-xl font-semibold">10.6 การบูรณาการระหว่าง CNN + Transformer</h3>
      <p>
        ในงานล่าสุดปี 2023 เช่น CoAtNet, ConvNeXt, และ MaxViT มีการผสานข้อดีของ CNN กับ Self-Attention เข้าด้วยกัน โดยใช้ convolution เพื่อ capture local feature และ attention เพื่อเรียนรู้ long-range dependencies
      </p>
      <ul className="list-disc pl-6">
        <li>CoAtNet ใช้ convolution ใน layer แรก ๆ และ attention ใน layer ลึก</li>
        <li>ConvNeXt ปรับสถาปัตยกรรม CNN ให้มี behavior ใกล้กับ Transformer</li>
        <li>MaxViT รวม local และ global receptive field ผ่าน parallel attention</li>
      </ul>

      <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
        <h4 className="text-lg font-semibold mb-2">Insight Box: บทเรียนจากงานวิจัย</h4>
        <ul className="list-disc pl-6 text-sm space-y-2">
          <li>Depth สำคัญ → แต่ต้องมีวิธีจัดการ gradient เช่น Residual</li>
          <li>Kernel เล็กติดกันหลายชั้นมีประสิทธิภาพสูงกว่า kernel ใหญ่</li>
          <li>Convolution เหมาะกับข้อมูลน้อย, Attention เหมาะกับ Big Data</li>
          <li>Mixing CNN กับ Transformer กำลังเป็นแนวทางที่แข็งแกร่งที่สุดใน Image Recognition</li>
        </ul>
      </div>
    </div>
  </section>

  <section id="cnn-vs-transformer" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. เปรียบเทียบ CNN vs Transformer</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img13} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Convolutional Neural Networks (CNNs) และ Transformer-based Models เป็นสองแนวทางหลักในงาน Deep Learning ที่ใช้สำหรับประมวลผลข้อมูลที่มีโครงสร้าง เช่น ภาพ วิดีโอ และข้อความ ทั้งสองแนวทางมีจุดเด่นและข้อจำกัดที่แตกต่างกันอย่างชัดเจน งานวิจัยจากมหาวิทยาลัย Stanford, MIT, และ Google Research ได้ทำการเปรียบเทียบระหว่าง CNN และ Transformer เพื่อนำเสนอแนวทางการเลือกใช้ที่เหมาะสมกับบริบทของงาน
    </p>

    <h3 className="text-xl font-semibold">โครงสร้างพื้นฐาน: Local vs Global</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-xl">
        <h4 className="font-semibold mb-2">CNNs</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Convolution Filters ที่มี Receptive Fields แบบ Local</li>
          <li>เหมาะกับการเรียนรู้ Local Patterns เช่น edge, texture</li>
          <li>พัฒนาลำดับ Hierarchical Features โดยอัตโนมัติ</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-xl">
        <h4 className="font-semibold mb-2">Transformers</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Self-Attention ที่มี Global Receptive Field</li>
          <li>สามารถเชื่อมโยงข้อมูลตำแหน่งใด ๆ ภายใน Input</li>
          <li>มีความยืดหยุ่นในการจัดการข้อมูลลำดับและโครงสร้าง</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">ประสิทธิภาพในด้านต่าง ๆ</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-200 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">มิติเปรียบเทียบ</th>
            <th className="border px-4 py-2">CNN</th>
            <th className="border px-4 py-2">Transformer</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Data Efficiency</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำ (ต้อง Pre-train)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Training Time</td>
            <td className="border px-4 py-2">รวดเร็วกว่าในขนาดเล็ก</td>
            <td className="border px-4 py-2">ช้ากว่าแต่ scale ได้ดี</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Parallelization</td>
            <td className="border px-4 py-2">จำกัดโดยลำดับการคำนวณ</td>
            <td className="border px-4 py-2">ดีมาก (ใช้ Attention Matrix)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Inductive Bias</td>
            <td className="border px-4 py-2">Strong (เช่น translation invariance)</td>
            <td className="border px-4 py-2">Low (ต้องเรียนรู้เอง)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Interpretability</td>
            <td className="border px-4 py-2">ดี (visualize filter/activation)</td>
            <td className="border px-4 py-2">สูงขึ้นด้วย attention map</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">การผสาน CNN กับ Transformer</h3>
    <p>
      งานวิจัยล่าสุด เช่น CoAtNet จาก Google Brain ได้ผสานความสามารถของ CNN (เช่น การเรียนรู้ pattern เชิงพื้นที่ที่มีประสิทธิภาพ) เข้ากับ Transformer (การเรียนรู้ความสัมพันธ์ระยะไกล) เพื่อสร้างโมเดลที่มีประสิทธิภาพทั้งในด้านความแม่นยำและประสิทธิภาพการประมวลผล ทำให้เกิดคลาสของโมเดลใหม่ที่เรียกว่า Hybrid Models
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="text-lg font-semibold mb-2">Insight จาก CoAtNet (Google, 2021)</h4>
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>เริ่มด้วย Convolution Layers ในชั้นล่างเพื่อดึง Low-level Features</li>
        <li>ใช้ Self-Attention ในชั้นบนเพื่อเข้าใจ High-level Global Semantics</li>
        <li>ช่วยลดจำนวนพารามิเตอร์เมื่อเทียบกับ pure Transformer</li>
        <li>สามารถฝึกได้รวดเร็วกว่า Vision Transformer (ViT) บน dataset ขนาดกลาง</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ข้อสังเกตจากสถาปัตยกรรมต่าง ๆ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>ResNet:</strong> ใช้ CNN อย่างมีประสิทธิภาพเพื่อเรียนรู้ hierarchical representation</li>
      <li><strong>ViT (Vision Transformer):</strong> แบ่งภาพเป็น patches แล้วทำ Attention ระหว่าง patch</li>
      <li><strong>Swin Transformer:</strong> เพิ่ม inductive bias แบบ CNN โดยใช้ window-based attention</li>
      <li><strong>ConvNeXt:</strong> พัฒนา CNN ให้มีคุณสมบัติเทียบเท่า Transformer โดยคง structure แบบ convolution</li>
    </ul>

    <h3 className="text-xl font-semibold">สรุปภาพรวม</h3>
    <p>
      CNN ยังคงเป็นโมเดลที่เหมาะกับงานที่มีข้อมูลน้อย และต้องการเรียนรู้ feature ที่มีโครงสร้างเฉพาะเช่น edge หรือ texture ได้อย่างมีประสิทธิภาพ ในขณะที่ Transformer มีศักยภาพสูงมากในงานที่ต้องการความเข้าใจเชิงลึกและสัมพันธ์ระยะไกล โดยเฉพาะเมื่อมีข้อมูลขนาดใหญ่เพียงพอ การเลือกใช้โมเดลควรพิจารณาจากลักษณะข้อมูล และความสามารถของระบบในการประมวลผลเชิงขนานและขนาด batch ที่ใช้
    </p>
  </div>
</section>


<section id="special-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">Special Box: ข้อควรระวัง</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h3 className="text-xl font-semibold mb-4">ข้อควรระวังในการออกแบบและใช้งาน CNN</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li><strong>Stride ใหญ่เกินไป:</strong> อาจทำให้รายละเอียดของภาพหายไปเร็วเกินควรในเลเยอร์ต้น ๆ</li>
        <li><strong>Padding ไม่เพียงพอ:</strong> ทำให้ข้อมูลที่อยู่บริเวณขอบภาพไม่ถูกพิจารณา</li>
        <li><strong>Filter น้อยเกินไป:</strong> จำกัดความสามารถในการเรียนรู้ Feature ตั้งแต่ต้น</li>
        <li><strong>ใช้ Pooling มากเกินไป:</strong> ทำให้ขนาดของ Feature Map ลดลงเร็วเกินไป</li>
        <li><strong>จำนวนพารามิเตอร์สูงเกินไป:</strong> เพิ่มโอกาส Overfitting โดยเฉพาะเมื่อข้อมูลไม่เพียงพอ</li>
        <li><strong>Initialization ไม่เหมาะสม:</strong> ทำให้โมเดลไม่สามารถลู่เข้าสู่ค่าที่ดีได้</li>
        <li><strong>Activation Function แบบมี Saturation:</strong> เช่น Sigmoid ส่งผลให้เกิด Vanishing Gradient</li>
        <li><strong>ใช้ Fully Connected Layer มากเกินไป:</strong> เพิ่มภาระพารามิเตอร์โดยไม่จำเป็น</li>
        <li><strong>Input Image Size ไม่เหมาะสม:</strong> ส่งผลให้ Output Tensor ไม่สามารถเชื่อมต่อกับ FC Layer ได้</li>
        <li><strong>ไม่แยก Training และ Inference Mode ชัดเจน:</strong> ทำให้ BatchNorm และ Dropout ทำงานผิดพลาด</li>
      </ul>
      <div className="bg-blue-100 dark:bg-blue-900 p-6 rounded-lg border-l-4 border-blue-400 dark:border-blue-600 mt-6">
        <h4 className="font-semibold mb-2">แหล่งอ้างอิงที่ใช้ในการจัดทำ:</h4>
        <ul className="list-disc pl-6 text-sm">
          <li>Stanford CS231n</li>
          <li>MIT 6.S191</li>
          <li>Deep Learning Book - Ian Goodfellow et al.</li>
          <li>He et al. 2015 - Deep Residual Learning</li>
          <li>Ioffe & Szegedy - Batch Normalization</li>
          <li>Wu & He - Group Normalization</li>
        </ul>
      </div>
    </div>
  </div>
</section>


        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day21 theme={theme} />
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
        <ScrollSpy_Ai_Day21 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day21_CNNIntro;
