import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day22 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day22";
import MiniQuiz_Day22 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day22";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day22_CNNArchitecture = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("cnn_architecture_intro1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("cnn_architecture_intro2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("cnn_architecture_intro3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("cnn_architecture_intro4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("cnn_architecture_intro5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("cnn_architecture_intro6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("cnn_architecture_intro7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("cnn_architecture_intro8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("cnn_architecture_intro9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("cnn_architecture_intro10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("cnn_architecture_intro11").format("auto").quality("auto").resize(scale().width(600));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 22: CNN Architecture & Filters</h1>

        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img1} />
        </div>

        <div className="w-full flex justify-center my-12">
          <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
        </div>

        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: โครงสร้าง CNN ไม่ได้มีแค่เลเยอร์ซ้อนกัน</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img2} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Convolutional Neural Networks (CNNs) เป็นสถาปัตยกรรมหลักใน Deep Learning โดยเฉพาะในด้าน Computer Vision แต่แทนที่จะเป็นเพียงการวางเลเยอร์แบบเรียงกันธรรมดา สถาปัตยกรรมของ CNN ได้พัฒนาไปเป็นโครงสร้างที่ซับซ้อนและออกแบบมาอย่างมีแบบแผน ซึ่งส่งผลต่อประสิทธิภาพในการฝึก ความสามารถในการ generalize และความเร็วในการประมวลผล
    </p>

    <h3 className="text-xl font-semibold">1.1 CNN ไม่ใช่แค่ “Layer Stack”</h3>
    <p>
      การเรียงเลเยอร์ Convolution → Activation → Pooling เป็นเพียงโครงสร้างพื้นฐาน แต่สถาปัตยกรรมระดับสูงประกอบด้วยแนวคิด เช่น Residual Connection, Multi-branching, Depth-wise Separable Convolutions, และ Attention Mechanisms ที่ช่วยยกระดับความสามารถของโมเดล
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">โครงสร้างเรียบง่าย</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Conv → ReLU → Pool → FC → Softmax</li>
          <li>ใช้งานใน LeNet, AlexNet</li>
        </ul>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">โครงสร้างเชิงลึก</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Residual Path, Inception Block</li>
          <li>ใช้ใน ResNet, GoogLeNet</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">1.2 การออกแบบ Architecture ส่งผลต่อผลลัพธ์</h3>
    <p>
      สถาปัตยกรรมที่ดีช่วยให้โมเดลสามารถฝึกได้ลึกขึ้น ลดปัญหา Vanishing Gradient และเพิ่ม Accuracy อย่างมีนัยสำคัญ จากการศึกษาโดย He et al. (2015) พบว่า ResNet ที่ใช้ Residual Connection สามารถสร้างโมเดลลึก 152 ชั้นได้โดยไม่สูญเสียความสามารถในการเรียนรู้
    </p>

    <div className="overflow-x-auto bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <pre className="text-sm">
{`# เปรียบเทียบความลึกของโมเดลกับความแม่นยำ (ImageNet Top-5 Accuracy)
Depth     Top-5 Accuracy
-------------------------
8-layer     92.0%
34-layer    92.5%
152-layer   93.8%  ← ResNet-152
`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">1.3 Case Study: เปรียบเทียบ CNN Architectures</h3>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold mb-2">AlexNet (2012)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ ReLU แทน sigmoid → เพิ่ม speed</li>
          <li>แบ่ง training เป็น 2 GPUs</li>
          <li>ใช้ Dropout → ลด overfitting</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold mb-2">VGGNet (2014)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Conv 3x3 เท่านั้น → สร้างลำดับชั้น</li>
          <li>เพิ่ม depth แต่คงความเรียบง่าย</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold mb-2">ResNet (2015)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Residual Blocks → ฝึกโมเดลลึก</li>
          <li>แก้ปัญหา Vanishing Gradient</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">1.4 สถาปัตยกรรมระดับสูงที่ใช้งานจริง</h3>
    <p>
      งานวิจัยจาก Google, Microsoft, และ Meta ได้พัฒนาสถาปัตยกรรมขั้นสูงที่ผสมผสานระหว่าง CNN และ Self-Attention เพื่อเพิ่มประสิทธิภาพของโมเดลในระดับ production เช่น EfficientNet, ConvNeXt, และ CoAtNet
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li><strong>EfficientNet:</strong> ใช้ Compound Scaling เพื่อเพิ่ม depth, width และ resolution อย่างเป็นระบบ</li>
      <li><strong>CoAtNet:</strong> รวม CNN กับ Transformer ใน block เดียว</li>
      <li><strong>ConvNeXt:</strong> ปรับ CNN ให้มี behavior ใกล้กับ Vision Transformer</li>
    </ul>

    <h3 className="text-xl font-semibold">1.5 การเลือกสถาปัตยกรรมให้เหมาะสมกับงาน</h3>
    <p>
      การเลือกสถาปัตยกรรมควรพิจารณา Dataset, Task, ความเร็วที่ต้องการ, และขนาดของ Hardware:
    </p>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-200 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Architecture</th>
            <th className="border px-4 py-2">เหมาะกับงาน</th>
            <th className="border px-4 py-2">ข้อดี</th>
            <th className="border px-4 py-2">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">VGG</td>
            <td className="border px-4 py-2">งานที่ต้องการความง่ายในการปรับปรุง</td>
            <td className="border px-4 py-2">เข้าใจง่าย</td>
            <td className="border px-4 py-2">มีพารามิเตอร์มาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ResNet</td>
            <td className="border px-4 py-2">งานที่ต้องการโมเดลลึก</td>
            <td className="border px-4 py-2">แก้ gradient ได้</td>
            <td className="border px-4 py-2">ซับซ้อนขึ้น</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">EfficientNet</td>
            <td className="border px-4 py-2">ระบบ edge/mobile</td>
            <td className="border px-4 py-2">พลังสูง น้ำหนักเบา</td>
            <td className="border px-4 py-2">ต้องใช้ NAS หรือ pretrained</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">1.6 สรุปภาพรวม</h3>
    <p>
      สถาปัตยกรรมของ CNN คือหัวใจในการออกแบบโมเดลที่มีประสิทธิภาพ ไม่ว่าจะเป็นโมเดลที่มีโครงสร้างง่ายสำหรับงานพื้นฐาน หรือโมเดลลึกที่ต้องการเรียนรู้ feature ที่ซับซ้อน การเข้าใจแนวทางการออกแบบที่มาจากงานวิจัยระดับโลกช่วยให้สามารถเลือกใช้สถาปัตยกรรมที่เหมาะสมและปรับให้เข้ากับระบบได้อย่างแม่นยำ
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600 text-sm space-y-2">
      <h4 className="font-semibold">Insight Box</h4>
      <ul className="list-disc pl-6">
        <li>โมเดลที่ดีไม่ได้แปลว่าต้องลึกเสมอ แต่ต้องเหมาะกับ task</li>
        <li>Residual Learning คือการปฏิวัติ Deep Learning ที่แท้จริง</li>
        <li>การเลือก architecture ที่ไม่เหมาะสม อาจทำให้โมเดลล้มเหลวทั้งที่มีข้อมูลเพียงพอ</li>
      </ul>
    </div>
  </div>
</section>


<section id="building-blocks" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Basic Building Blocks ของ CNN Architecture</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img3} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      โครงสร้างพื้นฐานของ Convolutional Neural Networks (CNNs) ประกอบด้วยชุดเลเยอร์ที่ออกแบบมาเพื่อประมวลผลข้อมูลภาพอย่างมีประสิทธิภาพ โดยแต่ละเลเยอร์มีหน้าที่เฉพาะตัวในการเรียนรู้ ลำดับของเลเยอร์และวิธีเชื่อมโยงมีผลต่อความสามารถในการเรียนรู้ของโมเดล ซึ่งเป็นหัวใจของงานวิจัยในสาขา Deep Learning และได้รับการพัฒนาจากมหาวิทยาลัยชั้นนำ เช่น Stanford, MIT และ Oxford
    </p>

    <h3 className="text-xl font-semibold">2.1 ลำดับขั้นพื้นฐานของ CNN</h3>
    <p>โมเดล CNN โดยทั่วไปจะมีลำดับเลเยอร์หลักดังนี้:</p>
    <div className="overflow-x-auto bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <pre className="text-sm">{`[Conv → Activation → BatchNorm → Pooling] → [Dropout] → [Fully Connected → Softmax]`}</pre>
    </div>

    <h3 className="text-xl font-semibold">2.2 Convolutional Layer</h3>
    <p>
      เป็นเลเยอร์ที่สำคัญที่สุด ใช้ Filter เล็ก ๆ เลื่อนผ่านข้อมูลภาพเพื่อดึง Feature เฉพาะจุด เช่น ขอบ เส้น รูปร่าง โดยค่าของ filter จะถูกเรียนรู้จากข้อมูลผ่าน backpropagation
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">หน้าที่</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>เรียนรู้ Feature แบบ Local</li>
          <li>ลดจำนวนพารามิเตอร์เมื่อเทียบกับ Fully Connected</li>
        </ul>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ตัวอย่างโค้ด (PyTorch)</h4>
        <pre className="text-sm overflow-x-auto">
{`import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3,       # RGB
    out_channels=32,     # 32 filters
    kernel_size=3,       # 3x3 filter
    stride=1,
    padding=1
)`}
        </pre>
      </div>
    </div>

    <h3 className="text-xl font-semibold">2.3 Activation Function (ReLU)</h3>
    <p>
      เพิ่มความไม่เชิงเส้นให้โมเดลเพื่อให้สามารถเรียนรู้ฟังก์ชันซับซ้อนได้ นิยมใช้ ReLU (Rectified Linear Unit) เพราะคำนวณเร็วและลดปัญหา vanishing gradient
    </p>
    <div className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
      <pre className="text-sm">{`f(x) = max(0, x)`}</pre>
    </div>

    <h3 className="text-xl font-semibold">2.4 Batch Normalization</h3>
    <p>
      ช่วยให้โมเดลลู่เข้าสู่ค่าที่เหมาะสมเร็วขึ้น โดยลด internal covariate shift และรักษาค่าการกระจายของ output ให้อยู่ในช่วงคงที่
    </p>
    <div className="overflow-x-auto bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <pre className="text-sm">{`nn.BatchNorm2d(num_features=32)`}</pre>
    </div>

    <h3 className="text-xl font-semibold">2.5 Pooling Layer</h3>
    <p>
      ลดขนาดของ Feature Map เพื่อควบคุมจำนวนพารามิเตอร์และลด overfitting โดยทั่วไปใช้ MaxPooling หรือ AveragePooling
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Max Pooling</h4>
        <pre className="text-sm overflow-x-auto">{`nn.MaxPool2d(kernel_size=2, stride=2)`}</pre>
      </div>
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Average Pooling</h4>
        <pre className="text-sm overflow-x-auto">{`nn.AvgPool2d(kernel_size=2, stride=2)`}</pre>
      </div>
    </div>

    <h3 className="text-xl font-semibold">2.6 Dropout</h3>
    <p>
      เทคนิค Regularization เพื่อป้องกัน overfitting โดยการสุ่มปิดบาง neuron ในแต่ละรอบ training
    </p>
    <div className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
      <pre className="text-sm">{`nn.Dropout(p=0.5)`}</pre>
    </div>

    <h3 className="text-xl font-semibold">2.7 Fully Connected Layer</h3>
    <p>
      ชั้นสุดท้ายที่ใช้แปลง Feature ที่ดึงมาได้ให้เป็นผลลัพธ์ เช่น Class Score โดยแต่ละ neuron เชื่อมกับทุก input
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-sm overflow-x-auto">
{`nn.Linear(in_features=128, out_features=10)`}</pre>

    <h3 className="text-xl font-semibold">2.8 Softmax Output</h3>
    <p>
      แปลง logits ที่ได้จาก Fully Connected ให้เป็นความน่าจะเป็นรวม 1 ใช้ใน classification ที่มีหลาย class
    </p>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-lg overflow-x-auto">
{`import torch.nn.functional as F

output = F.softmax(logits, dim=1)`}
    </pre>

    <h3 className="text-xl font-semibold">2.9 สรุป Flow ของ CNN</h3>
    <pre className=" dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
{`model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(2),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)`}
    </pre>

    <h3 className="text-xl font-semibold">2.10 ความเชื่อมโยงกับงานวิจัยระดับโลก</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600 text-sm space-y-2">
      <p><strong>Stanford CS231n:</strong> สอนหลักการ convolution, activation, normalization และ pooling อย่างเป็นระบบ</p>
      <p><strong>MIT 6.S191:</strong> นำเสนอการจัดลำดับเลเยอร์และการ fine-tune CNN</p>
      <p><strong>Deep Learning Book:</strong> โดย Ian Goodfellow อธิบายกลไกของ CNN อย่างลึกซึ้ง</p>
      <p><strong>He et al. (2016):</strong> ยืนยันว่าการใช้ ReLU และ BatchNorm ช่วยให้โมเดลเรียนรู้ได้ดีในโครงสร้างลึก</p>
    </div>
  </div>
</section>


<section id="filters" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. การออกแบบ Filter (Kernel)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      ใน Convolutional Neural Networks (CNNs) ฟิลเตอร์ หรือที่เรียกว่า Kernel คือหัวใจหลักที่ใช้ดึงลักษณะเฉพาะของข้อมูลภาพ การออกแบบและการเรียนรู้ฟิลเตอร์เหล่านี้มีผลโดยตรงต่อประสิทธิภาพของโมเดล ทั้งในแง่การรู้จำวัตถุ การแยกแยะลวดลาย และความสามารถในการเรียนรู้ฟีเจอร์ที่มีความซับซ้อน
    </p>

    <h3 className="text-xl font-semibold">3.1 ฟังก์ชันของ Filter</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ตรวจจับขอบ (edges), texture, รูปร่าง และลักษณะเฉพาะที่ปรากฏในภาพ</li>
      <li>ลด noise และแยก feature สำคัญออกจาก background</li>
      <li>เปลี่ยนข้อมูลดิบให้กลายเป็น feature map ที่สามารถใช้ในการเรียนรู้ลำดับชั้น</li>
    </ul>

    <h3 className="text-xl font-semibold">3.2 ประเภทของ Filter ที่ใช้ในงานคอมพิวเตอร์วิทัศน์</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Edge Detectors</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Sobel</li>
          <li>Prewitt</li>
          <li>Laplacian</li>
        </ul>
        <p className="text-sm">ใช้ในการค้นหาขอบหรือเส้นขอบที่มีการเปลี่ยนแปลงความเข้มของพิกเซล</p>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Blur / Sharpening</h4>
        <p className="text-sm">ใช้ลด noise หรือเพิ่มความคมชัดของวัตถุในภาพ</p>
      </div>
      <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Gabor Filters</h4>
        <p className="text-sm">ใช้ในงานแยกความถี่เฉพาะเชิงพื้นที่ เหมาะสำหรับตรวจจับลวดลายเฉพาะในภาพ</p>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Trainable Filters</h4>
        <p className="text-sm">เรียนรู้จากข้อมูลโดยตรงระหว่างการฝึก ไม่ต้องกำหนดเองแบบฟิลเตอร์ดั้งเดิม</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">3.3 การคำนวณ Convolution</h3>
    <p>
      ฟิลเตอร์ขนาดเล็ก (เช่น 3x3 หรือ 5x5) จะเลื่อนผ่าน input image เพื่อคำนวณ dot product กับแต่ละ sub-region ของภาพ ผลลัพธ์ที่ได้จะถูกรวบรวมเป็น feature map
    </p>
    <pre className="bg-gray-900 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`output(i,j) = ΣΣ input(i+m, j+n) * filter(m,n)`}
    </pre>

    <h3 className="text-xl font-semibold">3.4 ตัวอย่างฟิลเตอร์แบบดั้งเดิม</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-200 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">ประเภท</th>
            <th className="border px-4 py-2">Kernel 3x3</th>
            <th className="border px-4 py-2">ผลลัพธ์</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Sobel (X)</td>
            <td className="border px-4 py-2">[-1 0 1;<br/>-2 0 2;<br/>-1 0 1]</td>
            <td className="border px-4 py-2">ตรวจจับขอบแนวตั้ง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Sharpen</td>
            <td className="border px-4 py-2">[0 -1 0;<br/>-1 5 -1;<br/>0 -1 0]</td>
            <td className="border px-4 py-2">เพิ่มความคมชัด</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Gaussian Blur</td>
            <td className="border px-4 py-2">[1 2 1;<br/>2 4 2;<br/>1 2 1] ÷ 16</td>
            <td className="border px-4 py-2">ลด noise โดยรักษาโครงสร้างภาพ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">3.5 การเรียนรู้ Filter ด้วย CNN</h3>
    <p>
      แทนที่จะใช้ฟิลเตอร์แบบกำหนดล่วงหน้า CNN ใช้ฟิลเตอร์ที่เรียนรู้จากข้อมูลภาพระหว่างการฝึก โดยการอัปเดตพารามิเตอร์ผ่าน backpropagation ซึ่งช่วยให้โมเดลสามารถสกัดฟีเจอร์ที่มีความเหมาะสมกับงานนั้น ๆ ได้อย่างอัตโนมัติ
    </p>
    <pre className="bg-gray-900 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import torch
import torch.nn as nn

# สร้าง convolutional layer ที่เรียนรู้ได้
conv = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)

input_image = torch.randn(1, 3, 224, 224)  # รูปภาพ RGB ขนาด 224x224
output_feature = conv(input_image)
print(output_feature.shape)  # (1, 32, 224, 224)`}
    </pre>

    <h3 className="text-xl font-semibold">3.6 Feature Map และ Kernel Visualization</h3>
    <p>
      นักวิจัยเช่น Zeiler & Fergus (2014) ได้พัฒนาเทคนิค visualization ที่สามารถแสดงให้เห็นว่า filter ต่าง ๆ ภายใน CNN เรียนรู้อะไร เช่น ในเลเยอร์แรกจะเน้น edge และ texture ในขณะที่เลเยอร์ลึกจะเข้าใจรูปร่างหรือวัตถุแบบรวม
    </p>

    <h3 className="text-xl font-semibold">3.7 ผลกระทบของขนาด Kernel</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>3x3:</strong> ใช้บ่อยใน VGGNet ช่วยลดพารามิเตอร์และเน้น feature เฉพาะจุด</li>
      <li><strong>5x5 / 7x7:</strong> เพิ่ม receptive field สำหรับ feature ที่มีบริบทกว้างขึ้น</li>
      <li><strong>1x1:</strong> ใช้ลดมิติ (dimension reduction) หรือผสมข้อมูลข้าม channel</li>
    </ul>

    <h3 className="text-xl font-semibold">3.8 Insight จากงานวิจัยชั้นนำ</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li><strong>Zeiler & Fergus (2014)</strong>: พัฒนา DeconvNet เพื่อ visualize ฟิลเตอร์ในแต่ละเลเยอร์</li>
        <li><strong>Yosinski et al. (2015, Cornell & Google)</strong>: แสดงว่า filter ลึกสามารถเรียนรู้ semantic concept</li>
        <li><strong>Simonyan & Zisserman (Oxford, 2014)</strong>: ใช้ conv 3x3 ซ้อนกันหลายชั้นใน VGGNet แทน 5x5</li>
        <li><strong>Coates et al. (Stanford)</strong>: พิสูจน์ว่าการสุ่ม filter ก็สามารถให้ performance สูงถ้ามีจำนวนมากพอ</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">3.9 สรุป</h3>
    <p>
      ฟิลเตอร์ใน CNN คือเครื่องมือที่ทำให้โมเดลสามารถเข้าใจและแยกแยะลักษณะต่าง ๆ ในภาพได้อย่างมีประสิทธิภาพ ไม่ว่าจะเป็นขอบ เส้น รูปทรง หรือ semantic pattern การออกแบบ filter ที่ดีสามารถยกระดับความสามารถของโมเดลได้อย่างมีนัยสำคัญ ทั้งในด้านความแม่นยำ ความเร็ว และความสามารถในการ generalize
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-6 rounded-xl border-l-4 border-blue-400 dark:border-blue-600 mt-6">
      <h4 className="font-semibold mb-2 text-center">Special Insight</h4>
      <p className="text-sm text-center italic\">Filters are the eyes of a CNN. The better they are trained, the clearer the vision of the model becomes.</p>
    </div>
  </div>
</section>

<section id="hyperparameters" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Layer Hyperparameters</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img5} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <p>
      Hyperparameters ภายในแต่ละเลเยอร์ของ Convolutional Neural Networks (CNN) มีผลโดยตรงต่อพฤติกรรมการเรียนรู้ของโมเดล เช่น ขนาดฟิลเตอร์ การเลื่อน (stride) การ padding และจำนวนฟิลเตอร์ ส่งผลต่อความสามารถในการตรวจจับลักษณะเฉพาะ ความละเอียดของข้อมูล และการใช้ทรัพยากรระหว่างการฝึก
    </p>

    <h3 className="text-xl font-semibold">4.1 Filter Size</h3>
    <p>
      Filter หรือ Kernel คือหน้าต่างเล็ก ๆ ที่เลื่อนผ่าน Input เพื่อตรวจจับ Feature เฉพาะ เช่น ขอบ (edge), รูปร่าง, หรือ texture โดยขนาดที่นิยม ได้แก่ 3x3, 5x5 และ 7x7
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>3x3: ขนาดมาตรฐานที่ให้ receptive field เพียงพอเมื่อ stack หลายชั้น (VGGNet ใช้ต่อเนื่องหลายชั้น)</li>
      <li>5x5 หรือ 7x7: ใช้ในโมเดลรุ่นเก่า เช่น AlexNet แต่ถูกแทนด้วย stack 3x3 ในโมเดลใหม่เพื่อ efficiency</li>
    </ul>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-xl text-sm">
        <p><strong>ตัวอย่าง:</strong> ใช้ 2 ชั้นของ 3x3 convolution แทน 1 ชั้นของ 5x5 convolution เพื่อเพิ่ม non-linearity</p>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-xl text-sm">
        <p>CS231n (Stanford): แนะนำให้ใช้ 3x3 ซ้อนกันมากกว่าการใช้ filter ใหญ่เพื่อควบคุมพารามิเตอร์</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">4.2 Stride</h3>
    <p>
      Stride คือจำนวน pixel ที่ filter ขยับเมื่อสไลด์ผ่าน input ถ้า stride = 1 ภาพ output จะมีความละเอียดใกล้เคียง input ถ้า stride = 2 output จะเล็กลงครึ่งหนึ่งในแต่ละ dimension
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>Stride 1: ใช้เพื่อคง spatial resolution</li>
      <li>Stride 2: ลด resolution เพื่อเร่งความเร็วและลดพารามิเตอร์</li>
    </ul>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto text-sm">
      <pre>{`# PyTorch Example
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
input = torch.randn(1, 3, 224, 224)
output = conv(input)
print(output.shape)  # Output: torch.Size([1, 64, 112, 112])`}</pre>
    </div>

    <h3 className="text-xl font-semibold">4.3 Padding</h3>
    <p>
      Padding คือการเพิ่ม pixel รอบขอบของภาพเพื่อรักษาขนาดภาพ output ให้อยู่ในระดับที่ต้องการ และไม่ทำให้ภาพเล็กลงเร็วเกินไป
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Valid:</strong> ไม่มี padding → output เล็กลงทุกครั้ง</li>
      <li><strong>Same:</strong> padding เพื่อให้ขนาด output เท่ากับ input (ใช้ padding ตามสูตร)</li>
    </ul>

    <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-xl text-sm">
      <p>สูตรคำนวณ Padding (P) สำหรับ same:</p>
      <pre className="overflow-x-auto">{`P = ((S - 1) * W - S + F) / 2`}</pre>
      <p className="mt-2">โดย S = stride, W = width, F = filter size</p>
    </div>

    <h3 className="text-xl font-semibold">4.4 Number of Filters</h3>
    <p>
      จำนวน filters ในแต่ละเลเยอร์มีผลโดยตรงต่อจำนวน channel ของ output และความสามารถของโมเดลในการเรียนรู้ feature หลากหลาย
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>Filters มากขึ้น → ความสามารถในการจำแนกลักษณะเฉพาะดีขึ้น</li>
      <li>Filters มากไป → เพิ่มพารามิเตอร์และอาจเกิด overfitting</li>
    </ul>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-4 rounded-lg text-center">
        <p className="text-sm font-medium">VGG16 ใช้ filters: 64 → 128 → 256 → 512</p>
      </div>
      <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-4 rounded-lg text-center">
        <p className="text-sm font-medium">ResNet เริ่มจาก 64 และค่อย ๆ เพิ่มเป็น 512, 1024</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">4.5 การคำนวณ Output Shape</h3>
    <p>
      ขนาดของ output จาก convolution layer สามารถคำนวณได้จากสูตร:
    </p>
    <pre className="bg-gray-800 text-white p-4 rounded-xl text-sm overflow-x-auto">
{`Output_Width = (W - F + 2P) / S + 1
Output_Height = (H - F + 2P) / S + 1`}
    </pre>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Example 1</h4>
        <p className="text-sm">Input = 224x224, Filter = 3x3, Stride = 1, Padding = 1 → Output = 224x224</p>
      </div>
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Example 2</h4>
        <p className="text-sm">Input = 32x32, Filter = 5x5, Stride = 2, Padding = 0 → Output = 14x14</p>
      </div>
    </div>

    <h3 className="text-xl font-semibold">4.6 การตั้งค่า Hyperparameters ให้เหมาะสม</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เริ่มต้นด้วย filter 3x3, stride 1, padding 1 เป็น baseline</li>
      <li>เพิ่มจำนวน filters ทุก 2–3 blocks ตามความลึกของ network</li>
      <li>คำนวณ resolution หลังแต่ละ layer เพื่อไม่ให้ spatial dimension ลดเร็วเกินไป</li>
      <li>ปรับ padding และ stride ให้เหมาะกับ task เช่น segmentation ต้องการขนาด output คงที่</li>
    </ul>

    <h3 className="text-xl font-semibold">4.7 Insight จากงานวิจัย</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li><strong>Simonyan & Zisserman (2014):</strong> ใช้ 3x3 filters ซ้ำเพื่อเพิ่ม depth และ non-linearity</li>
        <li><strong>He et al. (2015):</strong> ResNet ใช้ fixed filter size 3x3 พร้อม stride ที่ควบคุม resolution ได้ดี</li>
        <li><strong>Google EfficientNet (2019):</strong> ปรับขนาด filter, depth, width อย่างมีระบบด้วย Compound Scaling</li>
        <li><strong>MIT & Facebook (RegNet):</strong> เสนอการเลือก hyperparameter ที่สมดุลตามความลึกของ architecture</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">4.8 สรุป</h3>
    <p>
      การกำหนดค่าของ Layer Hyperparameters ส่งผลโดยตรงต่อความสามารถในการเรียนรู้, ความลึกของข้อมูลเชิงภาพ และประสิทธิภาพโดยรวมของโมเดล CNN การเลือกค่าเหล่านี้ควรผ่านการวิเคราะห์ทั้งในเชิงคณิตศาสตร์และประสบการณ์จากงานวิจัย เพื่อออกแบบ architecture ที่มีความยืดหยุ่นและประสิทธิภาพสูงสุดในสภาพแวดล้อมจริง
    </p>
  </div>
</section>

<section id="layer-arrangement" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. การจัดลำดับ Layers: Depth & Width</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img6} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      โครงสร้างของ Convolutional Neural Networks (CNNs) มีความยืดหยุ่นในการออกแบบ โดยเฉพาะในด้านจำนวนชั้น (depth) และความกว้างของแต่ละชั้น (width) การออกแบบลำดับชั้นของเลเยอร์อย่างเหมาะสมมีผลโดยตรงต่อความสามารถของโมเดลในการเรียนรู้ข้อมูล การทำ generalization และประสิทธิภาพโดยรวมของระบบ CNN ทั้งหมด
    </p>

    <h3 className="text-xl font-semibold">5.1 ความหมายของ Depth และ Width</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Depth:</strong> จำนวนเลเยอร์ในเครือข่าย เช่น Conv, Pooling, BatchNorm, Fully Connected เป็นต้น</li>
      <li><strong>Width:</strong> จำนวนฟิลเตอร์ในแต่ละเลเยอร์ หรือจำนวน neurons ต่อ layer</li>
    </ul>
    <p>
      งานวิจัยจาก <strong>Stanford CS231n</strong> และ <strong>Deep Residual Learning by Microsoft Research (He et al., 2015)</strong> ได้แสดงให้เห็นว่าการเพิ่ม depth ช่วยให้โมเดลสามารถเรียนรู้ฟีเจอร์ที่มีความซับซ้อนได้มากขึ้น แต่การเพิ่ม depth อย่างไม่มีหลักการอาจทำให้เกิดปัญหา vanishing gradient ได้
    </p>

    <h3 className="text-xl font-semibold">5.2 ทำไม Layer Arrangement จึงสำคัญ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>โมเดลลึกสามารถเรียนรู้จาก low-level → high-level features ได้อย่างมีประสิทธิภาพ</li>
      <li>การจัดเรียงฟิลเตอร์แบบกว้างจะเพิ่มความสามารถในการจับหลายลักษณะพร้อมกันในแต่ละเลเยอร์</li>
      <li>การออกแบบ layer ที่สมดุลช่วยลด overfitting และเร่งการลู่เข้าสู่ค่า optimum</li>
    </ul>

    <h3 className="text-xl font-semibold">5.3 ตัวอย่าง Layer Arrangement ในโมเดลจริง</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">AlexNet (2012)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>มี 8 เลเยอร์หลัก: 5 Conv + 3 FC</li>
          <li>ใช้ ReLU และ Dropout เพื่อเพิ่ม non-linearity</li>
          <li>แบ่ง training ไปยัง 2 GPUs</li>
        </ul>
      </div>
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">VGGNet (2014)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Conv 3x3 ติดกันหลายชั้นเพื่อเพิ่ม depth</li>
          <li>ทุก Pooling ใช้ขนาด 2x2 stride=2</li>
          <li>เชื่อว่า depth = performance</li>
        </ul>
      </div>
      <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">Inception (GoogLeNet)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ parallel filters: 1x1, 3x3, 5x5</li>
          <li>ปรับ dimension ด้วย 1x1 convolution</li>
          <li>จัดเลเยอร์เป็น block เพื่อลด parameter</li>
        </ul>
      </div>
      <div className="bg-red-100 dark:bg-red-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">ResNet (2015)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ residual connection เพื่อข้ามบางเลเยอร์</li>
          <li>แก้ปัญหา vanishing gradient</li>
          <li>สามารถฝึกโมเดลได้ลึกถึง 152 เลเยอร์</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">5.4 เทคนิคการจัดลำดับเลเยอร์แบบคลาสสิก</h3>
    <p>ลำดับที่มักใช้ใน CNN แบบพื้นฐาน:</p>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto text-sm">
      <pre>{`[Conv → ReLU → BatchNorm → Pool] → [FC → Softmax]`}</pre>
    </div>
    <p>ในงานจริงจะมีการซ้อนหลาย block ขึ้นอยู่กับขนาดของข้อมูลและเป้าหมายของ task</p>

    <h3 className="text-xl font-semibold">5.5 โค้ดตัวอย่างใน PyTorch</h3>
    <pre className="bg-gray-800 text-white p-4 rounded-xl overflow-x-auto text-sm">
{`import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = DeepCNN()
print(model)`}
    </pre>

    <h3 className="text-xl font-semibold">5.6 Insight จากงานวิจัยของ Google & Facebook</h3>
    <ul className="list-disc pl-6 space-y-2 text-sm">
      <li>Google AutoML ใช้ NAS (Neural Architecture Search) เพื่อจัดลำดับเลเยอร์โดยอัตโนมัติ</li>
      <li>Facebook ConvNeXt ปรับ Conv blocks ให้มีลักษณะเหมือน Transformer</li>
      <li>งานวิจัยจาก MIT-IBM Watson แสดงว่า width มากเกินไปอาจลด generalization</li>
    </ul>

    <h3 className="text-xl font-semibold">5.7 Best Practice</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เพิ่ม depth เมื่อ dataset มีขนาดใหญ่และ pattern ซับซ้อน</li>
      <li>เพิ่ม width เมื่อข้อมูลต้องการจับหลายลักษณะพร้อมกันในแต่ละเลเยอร์</li>
      <li>ใช้ residual block เมื่อโมเดลลึกมากเกิน 20 เลเยอร์</li>
      <li>ควรมีการ normalize (BatchNorm) หลัง Conv เสมอ</li>
      <li>เริ่มด้วยเลเยอร์เล็กแล้วเพิ่มความซับซ้อนทีละขั้น</li>
    </ul>

    <h3 className="text-xl font-semibold">5.8 บทสรุป</h3>
    <p>
      การจัดลำดับเลเยอร์ใน CNN คือหัวใจของการออกแบบโมเดล deep learning ที่มีประสิทธิภาพ การเข้าใจความสัมพันธ์ระหว่าง depth และ width กับความสามารถของโมเดลในเชิงข้อมูลเชิงภาพมีผลโดยตรงต่อความสามารถในการเรียนรู้แบบลำดับชั้น และการทำ generalization ได้อย่างแข็งแกร่งในงานจริง
    </p>
  </div>
</section>


<section id="channels" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. การจัด Group & Channel</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      ใน Convolutional Neural Networks (CNNs) ข้อมูลภาพและฟีเจอร์จะถูกจัดเก็บในรูปแบบ Tensor ซึ่งมีมิติที่สำคัญ ได้แก่ ความสูง (Height), ความกว้าง (Width), และจำนวนช่อง (Channels) โดยเฉพาะมิติ Channel มีความสำคัญต่อการจัดการฟีเจอร์หลายระดับในแต่ละเลเยอร์ของโมเดล
    </p>

    <h3 className="text-xl font-semibold">6.1 Channel คืออะไรในภาพและ CNN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        ภาพ RGB มี 3 Channels (Red, Green, Blue)
      </li>
      <li>
        ฟีเจอร์ที่ได้จากเลเยอร์ Convolution จะเพิ่มจำนวน Channels ตามจำนวน Filters ที่ใช้
      </li>
      <li>
        Tensor ขนาดทั่วไปใน CNN: (Batch Size, Channels, Height, Width)
      </li>
    </ul>

    <h3 className="text-xl font-semibold">6.2 Input และ Output Channels ใน Convolution Layer</h3>
    <p>
      แต่ละ Convolution Layer จะรับ Input Channels และสร้าง Output Channels ตามจำนวน Filters ที่กำหนด ตัวอย่างเช่น ถ้าเลเยอร์รับภาพ RGB (3 Channels) และมี Filters จำนวน 64 ตัว Output จะมีขนาด (Batch, 64, H, W)
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">{`# PyTorch Example
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
input_tensor = torch.randn(32, 3, 224, 224)  # Batch size 32
output = conv(input_tensor)
print(output.shape)  # torch.Size([32, 64, 224, 224])`}</pre>
    </div>

    <h3 className="text-xl font-semibold">6.3 Group Convolution คืออะไร</h3>
    <p>
      Group Convolution เป็นแนวคิดที่นำมาใช้เพื่อลดจำนวนพารามิเตอร์ โดยแบ่ง Channels ออกเป็นกลุ่มย่อยแล้วทำ Convolution แยกกันในแต่ละกลุ่ม ก่อนนำผลลัพธ์มารวมเป็น Output Channels กลับมา
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>
        เมื่อ <strong>groups=1</strong>: เท่ากับ Convolution ปกติ
      </li>
      <li>
        เมื่อ <strong>groups=จำนวนช่อง</strong>: เรียกว่า Depthwise Convolution
      </li>
      <li>
        ใช้ในสถาปัตยกรรมเช่น <strong>MobileNet, ShuffleNet</strong> เพื่อเพิ่มประสิทธิภาพและลด computational cost
      </li>
    </ul>

    <div className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
      <pre>{`# Group Convolution
conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, groups=4)
# แบ่ง input 16 channels ออกเป็น 4 กลุ่ม (แต่ละกลุ่มมี 4 channels)
# ทำ conv แยกกันในแต่ละกลุ่ม แล้วรวม output ได้ 32 channels
`}</pre>
    </div>

    <h3 className="text-xl font-semibold">6.4 Depthwise และ Pointwise Convolution</h3>
    <p>
      ใน Depthwise Separable Convolution ซึ่งนิยมใน MobileNet มี 2 ขั้นตอน:
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">Depthwise Convolution</h4>
        <p className="text-sm">ใช้ Filter คนละตัวกับแต่ละ Channel</p>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-4 rounded-xl">
        <h4 className="font-semibold mb-2">Pointwise Convolution</h4>
        <p className="text-sm">ใช้ 1x1 Convolution รวม Channels เข้าด้วยกัน</p>
      </div>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">{`# Depthwise Separable Conv (simplified)
depthwise = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)
pointwise = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
output = pointwise(depthwise(input_tensor))`}</pre>
    </div>

    <h3 className="text-xl font-semibold">6.5 เทคนิค Channel Reduction</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ 1x1 Convolution เพื่อลดจำนวน Channel ก่อนทำ Convolution ใหญ่</li>
      <li>ใช้ Squeeze-and-Excitation block เพื่อเรียนรู้น้ำหนักของแต่ละ Channel</li>
    </ul>

    <div className="bg-gray-800 text-white p-4 rounded-xl overflow-x-auto text-sm">
      <pre>{`# Channel Reduction ด้วย 1x1 Conv
conv1x1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
reduced = conv1x1(input_tensor)`}</pre>
    </div>

    <h3 className="text-xl font-semibold">6.6 ตัวอย่างการคำนวณ Output Shape</h3>
    <p>
      การคำนวณขนาด Output หลัง Convolution เมื่อใช้ Padding และ Stride:
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto text-sm">
      <pre>{`Input: (Batch, 3, 224, 224)
Filter: 3x3, Stride=1, Padding=1, Filters=64
Output: (Batch, 64, 224, 224)

Formula:
H_out = (H_in + 2P - K) / S + 1
W_out = (W_in + 2P - K) / S + 1`}</pre>
    </div>

    <h3 className="text-xl font-semibold">6.7 งานวิจัยและสถาปัตยกรรมที่ใช้ Group/Channel Strategies</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>MobileNetV1:</strong> ใช้ Depthwise Separable Conv เพื่อลด latency</li>
      <li><strong>ResNeXt:</strong> ใช้ Group Convolution เพื่อเพิ่ม cardinality</li>
      <li><strong>ShuffleNet:</strong> ใช้ Group + Channel Shuffle เพื่อปรับปรุง flow</li>
    </ul>

    <h3 className="text-xl font-semibold">6.8 Insight จากงานวิจัยระดับโลก</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600 space-y-2 text-sm">
      <ul className="list-disc pl-6">
        <li>
          <strong>Howard et al. (Google MobileNet):</strong> การใช้ Depthwise Conv ลด FLOPs ได้กว่า 90% โดยไม่สูญเสีย accuracy มากนัก
        </li>
        <li>
          <strong>Xie et al. (ResNeXt):</strong> การเพิ่ม Group (cardinality) มีประสิทธิภาพมากกว่าการเพิ่ม depth หรือ width
        </li>
        <li>
          <strong>Ma et al. (ShuffleNet):</strong> Channel Shuffle เป็นวิธีที่คุ้มค่าในการทำให้ข้อมูลระหว่างกลุ่มมีปฏิสัมพันธ์กัน
        </li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">6.9 สรุป</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Channel เป็นองค์ประกอบสำคัญที่กำหนดความสามารถในการเรียนรู้ feature ของ CNN</li>
      <li>การใช้ Group และ Depthwise Convolution เป็นวิธีที่มีประสิทธิภาพในการลดพารามิเตอร์</li>
      <li>สถาปัตยกรรมสมัยใหม่ผสานเทคนิค Channel/Group เพื่อเพิ่มประสิทธิภาพทั้ง training และ deployment</li>
    </ul>
  </div>
</section>


<section id="transfer-learning" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Pretrained Filters & Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      ในการฝึก Convolutional Neural Networks (CNN) จากศูนย์มักต้องใช้ข้อมูลขนาดใหญ่และทรัพยากรคอมพิวเตอร์ที่สูง การใช้โมเดลที่ผ่านการฝึกมาก่อน (Pretrained Model) ช่วยให้สามารถนำฟีเจอร์ที่ได้จากการเรียนรู้ก่อนหน้า มาใช้ในงานใหม่ผ่านกระบวนการ Transfer Learning ซึ่งช่วยลดเวลาในการฝึกและปรับโมเดลให้เข้ากับงานที่มีข้อมูลจำกัด
    </p>

    <h3 className="text-xl font-semibold">7.1 แนวคิดพื้นฐานของ Transfer Learning</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>นำโมเดลที่ฝึกด้วย dataset ใหญ่ เช่น ImageNet มาใช้เป็นฐาน</li>
      <li>ปรับ fine-tune เลเยอร์บางส่วนกับ dataset ใหม่</li>
      <li>ลดจำนวนพารามิเตอร์ที่ต้องเรียนรู้ใหม่ → ฝึกเร็วขึ้น</li>
      <li>สามารถใช้ได้แม้ dataset ใหม่จะมีข้อมูลน้อย</li>
    </ul>

    <h3 className="text-xl font-semibold">7.2 โครงสร้างฟีเจอร์ที่เรียนรู้จาก ImageNet</h3>
    <p>
      งานวิจัยจาก Stanford CS231n พบว่าเลเยอร์ล่างของ CNN ที่ฝึกด้วย ImageNet สามารถเรียนรู้ฟีเจอร์พื้นฐาน เช่น ขอบ (edges), เส้นตรง, และ texture ซึ่งสามารถใช้ซ้ำได้ในงานอื่นโดยไม่ต้องฝึกใหม่
    </p>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ฟีเจอร์ในเลเยอร์ล่าง</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Edge detection</li>
          <li>Corner patterns</li>
          <li>Color blob filtering</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ฟีเจอร์ในเลเยอร์บน</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Shape composition</li>
          <li>Object parts</li>
          <li>Semantic encoding</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">7.3 ขั้นตอนการทำ Transfer Learning</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`# ตัวอย่างใน PyTorch
import torchvision.models as models
import torch.nn as nn

# โหลดโมเดล Pretrained
base_model = models.resnet50(pretrained=True)

# Freeze เลเยอร์ที่ไม่ต้องการปรับ
for param in base_model.parameters():
    param.requires_grad = False

# แทนที่ Fully Connected Layer
base_model.fc = nn.Sequential(
    nn.Linear(base_model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10)  # จำนวนคลาสใหม่
)`}
    </pre>

    <h3 className="text-xl font-semibold">7.4 เทคนิคการเลือกเลเยอร์ที่ควร Fine-Tune</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Fine-tune เฉพาะเลเยอร์บนสุดหาก dataset มีขนาดเล็ก</li>
      <li>Fine-tune มากขึ้นหาก dataset มีลักษณะต่างจาก dataset ดั้งเดิม</li>
      <li>ใช้ Learning Rate ต่ำเมื่อ fine-tune เพื่อรักษา weights เดิม</li>
    </ul>

    <h3 className="text-xl font-semibold">7.5 งานวิจัยสนับสนุนแนวคิด</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm">
        <li>Yosinski et al. (2014): ฟีเจอร์ที่เรียนรู้จาก ImageNet สามารถโอนย้ายได้ดีแม้เปลี่ยน domain</li>
        <li>Donahue et al. (2014): CNN Pretrained บน ImageNet สามารถใช้เป็น feature extractor ได้หลากหลายงาน</li>
        <li>He et al. (2016): ResNet ช่วยให้ Transfer Learning มีประสิทธิภาพมากขึ้นด้วย Residual Block</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">7.6 ประโยชน์ของการใช้ Pretrained Filters</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ลดเวลาฝึกโมเดล</li>
      <li>ลดการใช้พลังงานและทรัพยากร GPU</li>
      <li>เพิ่ม Accuracy โดยเฉพาะใน dataset ขนาดเล็ก</li>
      <li>สามารถใช้กับงาน Classification, Detection, Segmentation</li>
    </ul>

    <h3 className="text-xl font-semibold text-center">7.7 Visualization ของ Filters ที่เรียนรู้</h3>
    <p className=" text-center">
      การดูภาพของ Filters ที่เรียนรู้สามารถแสดงให้เห็นว่าโมเดลตอบสนองต่อ pattern แบบใด ตัวอย่างภาพจากงานของ Zeiler & Fergus แสดงให้เห็นว่า layer แรกของ CNN มักเรียนรู้ edge และ shape
    </p>
    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

    <h3 className="text-xl font-semibold">7.8 Fine-tuning vs Feature Extraction</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Feature Extraction</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Freeze ทุกเลเยอร์ของ base model</li>
          <li>Train เฉพาะ classifier head</li>
          <li>เร็วกว่า แต่จำกัด generalization</li>
        </ul>
      </div>
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Fine-Tuning</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>Train บางหรือทุกเลเยอร์ของโมเดลเดิม</li>
          <li>ต้องใช้เวลาและพลังงานมากขึ้น</li>
          <li>ผลลัพธ์ดีกว่าหาก dataset มีความต่างจากต้นฉบับ</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">7.9 ตัวอย่างการนำไปใช้ในงานจริง</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Medical Imaging: ใช้ฟีเจอร์จาก ImageNet → classify เนื้องอกจาก MRI</li>
      <li>Wildlife Monitoring: ใช้ ResNet + Transfer Learning ตรวจจับสัตว์หายากจากกล้องในป่า</li>
      <li>Autonomous Driving: ใช้ pretrained CNN บน COCO dataset เพื่อตรวจจับคนและรถ</li>
    </ul>

    <h3 className="text-xl font-semibold">7.10 สรุปแนวทางที่ดีที่สุด</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เริ่มต้นจาก pretrained model ที่เหมาะกับงาน</li>
      <li>วิเคราะห์ลักษณะของ dataset ใหม่เพื่อปรับเลเยอร์ที่ fine-tune</li>
      <li>ใช้ learning rate scheduler เพื่อฝึกอย่างมีประสิทธิภาพ</li>
      <li>ทดลองทั้ง feature extraction และ fine-tuning</li>
      <li>ตรวจสอบผลลัพธ์ด้วย validation set และ visualization</li>
    </ul>

    <blockquote className="border-l-4 border-yellow-500 pl-4 italic text-sm text-gray-700 dark:text-gray-300">
      “Filters are the eyes of a CNN. The better they are trained, the clearer the vision of the model becomes.”
    </blockquote>
  </div>
</section>


<section id="real-architectures" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Insight จาก Architecture จริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การออกแบบสถาปัตยกรรมของ Convolutional Neural Networks (CNNs) มีความสำคัญอย่างยิ่งต่อความสามารถในการเรียนรู้ การประมวลผล และการ generalize ของโมเดล งานวิจัยจากสถาบันชั้นนำ เช่น University of Toronto, Oxford, Microsoft Research และ Google Brain ได้สร้างสถาปัตยกรรมต้นแบบที่เป็นรากฐานของ Deep Learning ยุคใหม่
    </p>

    <h3 className="text-xl font-semibold">8.1 AlexNet (2012) - จุดเริ่มต้นของ Deep CNN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>พัฒนาโดย Krizhevsky, Sutskever และ Hinton จาก University of Toronto</li>
      <li>คว้าชัยชนะ ImageNet ILSVRC 2012 ด้วย error rate ที่ลดลงมากกว่า 10%</li>
      <li>ใช้ GPU ในการฝึกโมเดลที่มีเลเยอร์ลึกอย่างเต็มรูปแบบ</li>
      <li>แนะนำการใช้ ReLU แทน Sigmoid/Tanh เพื่อเร่งการลู่เข้า</li>
      <li>ใช้ Dropout เพื่อป้องกัน Overfitting ใน Fully Connected Layers</li>
    </ul>

    <h3 className="text-xl font-semibold">8.2 VGGNet (2014) - เรียบง่ายแต่ลึกมาก</h3>
    <p>
      VGGNet จาก Oxford Robotics Institute โดย Simonyan และ Zisserman เน้นการใช้ convolution ขนาดเล็ก (3x3) ซ้ำหลายครั้งเพื่อลึกแบบลำดับขั้น ทำให้โมเดลเข้าใจ pattern ซับซ้อนขึ้นโดยไม่เพิ่มพารามิเตอร์แบบระเบิด
    </p>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">จุดเด่น</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ Conv 3x3 หลายชั้นติดกัน</li>
          <li>รองรับการเพิ่มความลึกของโมเดลได้ง่าย</li>
          <li>ใช้งานได้ดีใน Transfer Learning</li>
        </ul>
      </div>
      <div className="bg-red-50 dark:bg-red-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ข้อควรระวัง</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>มีพารามิเตอร์จำนวนมาก</li>
          <li>ใช้ทรัพยากรสูงทั้งในการฝึกและ inference</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">8.3 GoogLeNet (Inception) - การคำนวณแบบขนาน</h3>
    <p>
      จากงานของ Szegedy et al. (Google Research, 2015) เสนอแนวคิดการใช้ filter หลายขนาด (1x1, 3x3, 5x5) ภายในบล็อกเดียวเพื่อจับ feature ที่หลากหลาย ลดพารามิเตอร์ด้วย 1x1 convolution
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>แนะนำ Inception Module → เลือก filter แบบขนาน</li>
      <li>ใช้ 1x1 Conv เพื่อลดความซับซ้อนของ computation</li>
      <li>ลดจำนวนพารามิเตอร์เมื่อเทียบกับโมเดลที่ลึกเท่ากัน</li>
    </ul>

    <h3 className="text-xl font-semibold">8.4 ResNet (2015) - การเรียนรู้เชิงลึกที่แท้จริง</h3>
    <p>
      ResNet จาก Microsoft Research โดย Kaiming He et al. เป็นสถาปัตยกรรมที่ได้รับความนิยมสูงสุดในปัจจุบัน ด้วยการใช้ Residual Connections แก้ปัญหา vanishing gradient และช่วยให้ฝึกโมเดลที่ลึกมากได้
    </p>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
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
    <ul className="list-disc pl-6 space-y-2">
      <li>Residual Block ช่วยให้ gradient ไหลผ่านเครือข่ายลึกได้ดี</li>
      <li>ฝึกโมเดลลึกถึง 152 เลเยอร์ได้สำเร็จ</li>
      <li>เป็น backbone ของโมเดลในงานภาพทุกประเภท</li>
    </ul>

    <h3 className="text-xl font-semibold">8.5 DenseNet (2017) - การเชื่อมต่อทุกชั้น</h3>
    <p>
      DenseNet เชื่อมทุกเลเยอร์เข้าด้วยกันโดยตรง (Dense Connectivity) ช่วยให้เกิดการ reuse feature และเพิ่มความลู่เข้าอย่างรวดเร็ว
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>ทุกเลเยอร์ได้รับข้อมูลจากทุกเลเยอร์ก่อนหน้า</li>
      <li>ลดการซ้ำซ้อนในการเรียนรู้ feature</li>
      <li>มีจำนวนพารามิเตอร์น้อยกว่าที่คาด</li>
    </ul>

    <h3 className="text-xl font-semibold">8.6 EfficientNet (2019) - สเกลโมเดลอย่างมีหลักการ</h3>
    <p>
      งานจาก Google Brain โดย Tan & Le เสนอการใช้ Compound Scaling เพื่อเพิ่มขนาดโมเดลในทุกมิติ (ความลึก, ความกว้าง, ขนาดภาพ) อย่างสมดุล
    </p>
    <div className="bg-green-100 dark:bg-green-900 p-4 rounded-xl text-sm">
      <p>สูตร Compound Scaling:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto">
{`depth: d = α^ϕ
width: w = β^ϕ
resolution: r = γ^ϕ

โดยที่ α * β^2 * γ^2 ≈ 2 (ภายใต้ constraint ของ resource)`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">8.7 ConvNeXt (2022) - CNN ที่เทียบเท่า Transformer</h3>
    <p>
      ConvNeXt จาก Facebook AI ปรับโครงสร้าง CNN ให้ใกล้เคียง Transformer มากที่สุด ทั้งในแง่ normalization, activation, และ architecture design
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ LayerNorm แทน BatchNorm</li>
      <li>ใช้ GELU แทน ReLU</li>
      <li>Conv block คล้าย Transformer block</li>
    </ul>

    <h3 className="text-xl font-semibold">8.8 CoAtNet (2021) - การผสาน CNN + Transformer</h3>
    <p>
      CoAtNet ผสมผสานข้อดีของ CNN (local inductive bias) และ Transformer (global attention) เพื่อให้ได้ model ที่ทั้ง generalize และมี efficiency
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>เลเยอร์ต้นใช้ Convolution</li>
      <li>เลเยอร์ลึกใช้ Self-Attention</li>
      <li>เหมาะกับ dataset ทั้งขนาดเล็กและใหญ่</li>
    </ul>

    <h3 className="text-xl font-semibold">8.9 เปรียบเทียบโมเดลหลัก</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-200 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Architecture</th>
            <th className="border px-4 py-2">Year</th>
            <th className="border px-4 py-2">Key Feature</th>
            <th className="border px-4 py-2">Institution</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">AlexNet</td>
            <td className="border px-4 py-2">2012</td>
            <td className="border px-4 py-2">ReLU + GPU + Dropout</td>
            <td className="border px-4 py-2">U. Toronto</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">VGGNet</td>
            <td className="border px-4 py-2">2014</td>
            <td className="border px-4 py-2">3x3 Conv + Depth</td>
            <td className="border px-4 py-2">Oxford</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ResNet</td>
            <td className="border px-4 py-2">2015</td>
            <td className="border px-4 py-2">Residual Block</td>
            <td className="border px-4 py-2">Microsoft</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">EfficientNet</td>
            <td className="border px-4 py-2">2019</td>
            <td className="border px-4 py-2">Compound Scaling</td>
            <td className="border px-4 py-2">Google</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">8.10 Insight ปิดท้าย</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600 text-sm">
      <p>
        การเลือกสถาปัตยกรรมมีผลโดยตรงต่อความสามารถของโมเดลในการตีความข้อมูลภาพอย่างมีประสิทธิภาพ สถาปัตยกรรมที่ดีควร balance ระหว่าง ความลึก, ความกว้าง, พลังการคำนวณ และขนาดของข้อมูล งานจากมหาวิทยาลัยและบริษัทระดับโลกยังคงขับเคลื่อนวิวัฒนาการของ CNN อย่างต่อเนื่อง
      </p>
    </div>
  </div>
</section>

<section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Visualization: ดูว่า Filter เรียนรู้อะไร</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การทำ Visualization ของ Filters และ Feature Maps เป็นเครื่องมือที่สำคัญในการตีความว่าโครงข่ายประสาทเทียมโดยเฉพาะ CNN มีการเรียนรู้อะไรจากข้อมูลบ้าง เทคนิคนี้ถูกใช้อย่างแพร่หลายในงานวิจัยของมหาวิทยาลัย Stanford, MIT, และ Google Research เพื่อตรวจสอบว่าโมเดลเข้าใจบริบทของข้อมูลอย่างถูกต้องหรือไม่ และเพื่อปรับปรุงความสามารถในการอธิบายของโมเดล (Model Interpretability)
    </p>

    <h3 className="text-xl font-semibold">9.1 Feature Map คืออะไร?</h3>
    <p>
      เมื่อภาพอินพุตผ่านเลเยอร์ convolutional แต่ละชั้น ผลลัพธ์จะเป็นชุดของ Feature Maps ซึ่งแสดงถึงการตอบสนองของฟิลเตอร์ต่อ pattern เฉพาะในภาพ เช่น ขอบ รูปร่าง หรือ texture การ visualize feature map เหล่านี้ช่วยให้เข้าใจว่าฟิลเตอร์แต่ละตัวจับสิ่งใดในภาพ
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto">
      <pre className="text-sm">
{`# ตัวอย่างการดึง Feature Map จากโมเดล PyTorch
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# โหลดโมเดลที่ผ่านการเทรนแล้ว
model = models.resnet18(pretrained=True)
model.eval()

# โหลดภาพและแปลงให้อยู่ในรูป tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("cat.jpg")
input_tensor = transform(img).unsqueeze(0)

# ดึง feature map ชั้นแรก
with torch.no_grad():
    features = model.conv1(input_tensor)

# แสดงภาพ feature map ช่องแรก
plt.imshow(features[0, 0].numpy(), cmap='viridis')
plt.title("Feature Map from Conv1")
plt.axis('off')
plt.show()`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">9.2 เทคนิค Activation Maximization</h3>
    <p>
      Activation Maximization เป็นเทคนิคที่ใช้ค้นหาภาพที่ทำให้ neuron ในโมเดลตอบสนองสูงสุด โดยเริ่มจาก noise และปรับค่าพิกเซลผ่าน Gradient Ascent ให้ neuron เฉพาะจุดมีค่ามากที่สุด แนวคิดนี้ถูกนำเสนอโดย DeepDream และวิจัยโดย Google Brain
    </p>

    <div className="bg-gray-900 text-white p-4 rounded-xl overflow-x-auto text-sm">
      <pre>{`# PyTorch Activation Maximization: สร้างภาพที่กระตุ้น neuron
from torch.nn.functional import mse_loss

# เลือกเลเยอร์และ neuron
layer = model.layer1[0].conv1
target_neuron = 0

# เริ่มจาก noise image
input_img = torch.randn((1, 3, 224, 224), requires_grad=True)

optimizer = torch.optim.Adam([input_img], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    activation = layer(input_img)
    loss = -activation[0, target_neuron].mean()
    loss.backward()
    optimizer.step()`}</pre>
    </div>

    <h3 className="text-xl font-semibold">9.3 Grad-CAM (Gradient-weighted Class Activation Mapping)</h3>
    <p>
      Grad-CAM เป็นเทคนิคที่ใช้ Gradient ของคลาสเป้าหมายกับ feature map ของเลเยอร์ convolutional เพื่อสร้าง heatmap ที่แสดงว่าพื้นที่ใดในภาพส่งผลมากที่สุดต่อการตัดสินใจของโมเดล เทคนิคนี้ได้รับความนิยมสูงในงานวิจัยจาก MIT CSAIL และ Google Research เนื่องจากตีความได้ชัดเจนและใช้งานง่าย
    </p>

    <div className="overflow-x-auto bg-gray-100 dark:bg-gray-800 p-4 rounded-xl text-sm">
      <pre>{`# Grad-CAM ตัวอย่างง่าย (ใช้โมดูลจาก torchvision)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ระบุเลเยอร์เป้าหมาย
target_layers = [model.layer4[-1]]

# สร้าง Grad-CAM object
cam = GradCAM(model=model, target_layers=target_layers)

# คำนวณ heatmap
grayscale_cam = cam(input_tensor=input_tensor)[0]
visualization = show_cam_on_image(img_tensor, grayscale_cam, use_rgb=True)`}</pre>
    </div>

    <h3 className="text-xl font-semibold">9.4 เทคนิค DeepDream</h3>
    <p>
      DeepDream เป็นการต่อยอดจาก activation maximization โดยให้โมเดลขยาย pattern ที่เรียนรู้ขึ้นเรื่อย ๆ จนกลายเป็นภาพเหนือจริง เทคนิคนี้สร้างโดย Google เพื่อวิเคราะห์ว่าโมเดลมีการเรียนรู้สิ่งใดอยู่ภายใน และเผยให้เห็นโครงสร้าง pattern ที่ละเอียดและซับซ้อน
    </p>

    <h3 className="text-xl font-semibold">9.5 ตัวอย่างการตีความในแต่ละเลเยอร์</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ชั้นล่าง</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>จับขอบและสีพื้นฐาน</li>
          <li>ตอบสนองต่อ edge แนวตั้ง แนวนอน</li>
        </ul>
      </div>
      <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ชั้นกลาง</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ตรวจจับรูปร่าง เช่น วงกลม รูปตัว T</li>
          <li>เริ่มเข้าใจ pattern ย่อยของวัตถุ</li>
        </ul>
      </div>
      <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">ชั้นบน</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>เข้าใจบริบททั้งภาพ เช่น ใบหน้า, สิ่งของ</li>
          <li>สร้าง concept-level activation</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">9.6 งานวิจัยและแหล่งอ้างอิง</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>Zeiler & Fergus (2014) — Visualizing and Understanding CNNs</li>
        <li>Yosinski et al. (2015) — Deep Visualization Toolbox (MIT)</li>
        <li>Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks</li>
        <li>Google Brain — DeepDream Project</li>
        <li>Stanford CS231n Lecture 4 — Feature Visualization</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">9.7 บทสรุป</h3>
    <p>
      การทำ Visualization ของ CNN ช่วยเปิดเผยการทำงานภายในของโมเดลในเชิงลึก และยกระดับความเข้าใจต่อสิ่งที่โมเดลเรียนรู้ในแต่ละเลเยอร์ เทคนิคอย่าง Activation Maximization, Grad-CAM และ DeepDream ล้วนเป็นเครื่องมือที่จำเป็นในการออกแบบ ตรวจสอบ และอธิบายพฤติกรรมของโมเดลให้เข้าใจง่ายและเชื่อถือได้มากขึ้น โดยเฉพาะในงานที่ต้องการความโปร่งใสเช่นการแพทย์หรือความปลอดภัย
    </p>
  </div>
</section>

<section id="problems" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. ปัญหาและการแก้ไขในการออกแบบ CNN</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การออกแบบ Convolutional Neural Networks (CNNs) จำเป็นต้องเผชิญกับปัญหาหลายด้านที่มีผลต่อประสิทธิภาพของโมเดลทั้งในด้านการเรียนรู้ ความสามารถในการ generalize และการนำไปใช้งานจริง การเข้าใจปัญหาเหล่านี้และแนวทางการแก้ไขที่ได้จากงานวิจัยระดับโลก เช่น จาก Stanford CS231n, MIT 6.S191, และ DeepMind ช่วยให้สามารถพัฒนาโมเดลที่มีเสถียรภาพและมีประสิทธิภาพสูงได้
    </p>

    <h3 className="text-xl font-semibold">10.1 Overfitting</h3>
    <p>
      โมเดลเรียนรู้ข้อมูล training ได้ดีเกินไปจนไม่สามารถ generalize กับข้อมูลใหม่ได้ สาเหตุหลักมาจากจำนวนพารามิเตอร์สูงเมื่อเทียบกับขนาดของข้อมูล หรือข้อมูลไม่หลากหลายพอ
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ Dropout เพื่อลดการพึ่งพา neuron ใด neuron หนึ่งมากเกินไป</li>
      <li>ทำ Data Augmentation เช่น random crop, flip, rotate</li>
      <li>ใช้ Early Stopping หยุดการฝึกเมื่อ validation loss เพิ่มขึ้น</li>
      <li>Regularization เช่น L1/L2 Weight Decay</li>
    </ul>

    <h3 className="text-xl font-semibold">10.2 Vanishing/Exploding Gradient</h3>
    <p>
      ในโมเดลที่ลึกมาก ค่า gradient อาจหายไปหรือระเบิดระหว่างการ backpropagation ส่งผลให้โมเดลไม่สามารถเรียนรู้ได้
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ ReLU แทน Sigmoid หรือ Tanh เพื่อลดปัญหา vanishing</li>
      <li>ใช้ Batch Normalization เพื่อควบคุม distribution ภายใน layer</li>
      <li>ใช้ Residual Connection (จาก ResNet) เพื่อให้ gradient ไหลผ่านได้ดี</li>
      <li>Initialize weights ด้วย He Initialization</li>
    </ul>

    <h3 className="text-xl font-semibold">10.3 การใช้พารามิเตอร์มากเกินไป (Overparameterization)</h3>
    <p>
      การใช้ filter จำนวนมากหรือ kernel ใหญ่โดยไม่จำเป็น ทำให้โมเดลมีขนาดใหญ่เกินความจำเป็น เพิ่มเวลาในการฝึกและความเสี่ยงต่อ overfitting
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ 1x1 Convolution เพื่อปรับขนาด channel โดยไม่เพิ่ม spatial dimension</li>
      <li>เลือกจำนวน filter ตามความซับซ้อนของข้อมูล ไม่ใช่ค่า default เสมอไป</li>
      <li>ใช้ Depthwise Separable Convolution เช่นใน MobileNet</li>
    </ul>

    <h3 className="text-xl font-semibold">10.4 การเลือก Hyperparameters ผิดพลาด</h3>
    <p>
      การเลือกค่า stride, kernel size, padding และจำนวนเลเยอร์ที่ไม่เหมาะสมอาจทำให้โมเดลทำงานได้ไม่ดี
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ Grid Search หรือ Random Search เพื่อลองค่าหลายแบบ</li>
      <li>ใช้ Hyperparameter Optimization Libraries เช่น Optuna หรือ Ray Tune</li>
      <li>พิจารณา validation loss อย่างละเอียดเพื่อเทียบผล</li>
    </ul>

    <h3 className="text-xl font-semibold">10.5 Input Size และ Resolution</h3>
    <p>
      ขนาด input ที่ไม่เหมาะสมอาจทำให้เกิดการลดขนาดเร็วเกินไปผ่าน pooling หรือ convolution ทำให้ feature หายไป
    </p>
    <ul className="list-disc pl-6">
      <li>เพิ่ม padding เพื่อรักษาขนาด spatial</li>
      <li>ออกแบบ network ให้ match กับ input size</li>
      <li>ใช้ Global Average Pooling แทน Flatten เพื่อลดปัญหาจาก spatial mismatch</li>
    </ul>

    <h3 className="text-xl font-semibold">10.6 การจัดลำดับ Layer ที่ไม่มีเหตุผลรองรับ</h3>
    <p>
      การวาง layer แบบสุ่ม เช่น BatchNorm หลัง Pooling หรือ Activation หลัง Dropout อาจส่งผลลบต่อการเรียนรู้
    </p>
    <ul className="list-disc pl-6">
      <li>โครงสร้างมาตรฐาน: Conv → BatchNorm → ReLU → Pooling</li>
      <li>Activation ควรอยู่หลัง BatchNorm และก่อน Dropout</li>
    </ul>

    <h3 className="text-xl font-semibold">10.7 ปัญหาเรื่อง Resource</h3>
    <p>
      โมเดลที่ใหญ่เกินไปอาจไม่สามารถใช้งานได้จริงบนอุปกรณ์จำกัด เช่น มือถือหรือ embedded systems
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ Model Pruning เพื่อลดขนาดโมเดล</li>
      <li>Quantization เพื่อลด precision เช่น จาก float32 → int8</li>
      <li>เลือก backbone ที่เบา เช่น EfficientNet, MobileNet, SqueezeNet</li>
    </ul>

    <h3 className="text-xl font-semibold">10.8 ความอ่อนไหวต่อ Adversarial Attack</h3>
    <p>
      CNN สามารถถูกหลอกได้ง่ายจากการเปลี่ยนพิกเซลเพียงเล็กน้อยใน input
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ Adversarial Training ด้วย input ที่มีการโจมตี</li>
      <li>ใช้ Defensive Distillation ลดความไวต่อ perturbation</li>
      <li>เพิ่ม Robustness ด้วย Data Augmentation และ Noise Injection</li>
    </ul>

    <h3 className="text-xl font-semibold">10.9 ข้อมูลไม่สมดุล (Imbalanced Data)</h3>
    <p>
      โมเดลอาจ bias ไปยัง class ที่มีจำนวนข้อมูลมาก
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ Weighted Loss Function เช่น CrossEntropyLoss(weights)</li>
      <li>ทำ Oversampling/Undersampling</li>
      <li>ใช้ Focal Loss สำหรับปัญหา class imbalance รุนแรง</li>
    </ul>

    <h3 className="text-xl font-semibold">10.10 การเรียนรู้ไม่เสถียร</h3>
    <p>
      Validation loss แกว่งหรือ training ไม่ converge เป็นปัญหาที่พบบ่อย
    </p>
    <ul className="list-disc pl-6">
      <li>ลด learning rate</li>
      <li>ใช้ Learning Rate Scheduler เช่น ReduceLROnPlateau</li>
      <li>เปลี่ยน optimizer เช่น จาก SGD เป็น Adam หรือ RMSProp</li>
    </ul>

    <h3 className="text-xl font-semibold">สรุป Insight</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 text-sm space-y-2">
        <li>Stanford CS231n: แนะนำการใช้ ReLU + BatchNorm + Dropout เป็น default</li>
        <li>MIT 6.S191: ย้ำเรื่องการใช้ Transfer Learning เพื่อลด overfitting</li>
        <li>Facebook AI Research: ใช้ AutoAugment และ NAS เพื่อลด human bias ในการออกแบบ</li>
        <li>Google Brain: แสดงให้เห็นว่ Regularization + Architecture Optimization ช่วยให้โมเดลมีความ robust</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold text-center mt-10">Special Insight</h3>
    <div className="bg-blue-50 dark:bg-blue-800 p-6 rounded-xl border-l-4 border-blue-400 dark:border-blue-600 text-center text-lg italic font-medium">
      “Filters are the eyes of a CNN. The better they are trained, the clearer the vision of the model becomes.”
    </div>
  </div>
</section>

        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day22 theme={theme} />
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
        <ScrollSpy_Ai_Day22 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day22_CNNArchitecture;
