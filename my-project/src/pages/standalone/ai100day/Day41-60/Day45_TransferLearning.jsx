import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day45 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day45";
import MiniQuiz_Day45 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day45";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day45_TransferLearning = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day45_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day45_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day45_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day45_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day45_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day45_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day45_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day45_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day45_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day45_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day45_11").format("auto").quality("auto").resize(scale().width(501));

   return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 45: Transfer Learning with Pretrained CNNs</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

      <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    1. บทนำ: ทำไมต้องใช้ Transfer Learning?
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <h3>ปัญหาในการฝึกโมเดล Deep Learning ตั้งแต่ต้น</h3>
    <p>
      การฝึกโมเดล Deep Learning ตั้งแต่เริ่มต้น (from scratch) ต้องการชุดข้อมูลขนาดใหญ่และใช้เวลาฝึกจำนวนมาก ซึ่งอาจใช้ทรัพยากรคอมพิวเตอร์จำนวนมหาศาลโดยเฉพาะอย่างยิ่งเมื่อใช้ Convolutional Neural Networks (CNNs) ที่มีความลึกและซับซ้อน ตัวอย่างเช่น การฝึกโมเดล ImageNet ตั้งแต่ต้นใช้เวลาหลายสัปดาห์บน GPU หลายตัว
    </p>
    
    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยจาก <strong>Stanford</strong> และ <strong>Google Research</strong> แสดงให้เห็นว่าโมเดลที่ผ่านการฝึกจากชุดข้อมูลใหญ่ ๆ แล้วสามารถนำมาใช้งานกับงานเฉพาะทาง (specific task) ได้อย่างมีประสิทธิภาพโดยไม่ต้องฝึกใหม่ทั้งหมด
      </p>
    </div>

    <h3>แนวคิดพื้นฐานของ Transfer Learning</h3>
    <p>
      Transfer Learning คือแนวทางในการใช้โมเดลที่ได้รับการฝึกไว้แล้ว (pretrained model) บนชุดข้อมูลขนาดใหญ่ และนำความรู้ที่ได้มาใช้กับงานใหม่ ซึ่งมักมีชุดข้อมูลน้อยกว่า โดยโมเดลที่ถูกใช้บ่อย เช่น ResNet, VGG, EfficientNet จะถูกตัด layer สุดท้ายออกและต่อด้วย layer ใหม่ที่เหมาะกับงานเป้าหมาย
    </p>

    <ul>
      <li>ลดเวลาการฝึกโมเดล</li>
      <li>ลดปัญหา overfitting ในชุดข้อมูลขนาดเล็ก</li>
      <li>เพิ่มความแม่นยำโดยเฉพาะในงานที่มีข้อมูลน้อย</li>
    </ul>

    <h3>การเปรียบเทียบกับการฝึกใหม่จากศูนย์</h3>
    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-700 mt-6">
  <table className="table-auto min-w-[640px] w-full">
    <thead>
      <tr className="bg-gray-200 dark:bg-gray-700 text-black dark:text-white">
        <th className="border px-4 py-2 border-gray-300 dark:border-gray-600">แนวทาง</th>
        <th className="border px-4 py-2 border-gray-300 dark:border-gray-600">Training Time</th>
        <th className="border px-4 py-2 border-gray-300 dark:border-gray-600">Data Requirement</th>
        <th className="border px-4 py-2 border-gray-300 dark:border-gray-600">Accuracy (ทั่วไป)</th>
      </tr>
    </thead>
    <tbody className="text-sm bg-white dark:bg-gray-900">
      <tr>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Train from scratch</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">สูงมาก</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">ชุดข้อมูลใหญ่</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">ขึ้นกับคุณภาพข้อมูล</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800">
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Transfer Learning</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">ต่ำ</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">ชุดข้อมูลเล็ก</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">สูงในหลายกรณี</td>
      </tr>
    </tbody>
  </table>
</div>

    <h3>ตัวอย่างงานวิจัยที่ใช้ Transfer Learning อย่างได้ผล</h3>
    <ul className="list-disc ml-6">
      <li>Image classification บน medical image (เช่น MRI, CT Scan)</li>
      <li>Plant disease detection ในภาพถ่ายจากโทรศัพท์มือถือ</li>
      <li>Face recognition ในระบบรักษาความปลอดภัย</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4">
      <p className="font-semibold">Highlight:</p>
      <p>
        Transfer Learning กลายเป็นแนวทางหลักในหลายงานของ Deep Learning เนื่องจากช่วยลดต้นทุนและเพิ่มความสามารถในการใช้งานจริงโดยเฉพาะในสาขาเชิงอุตสาหกรรม เช่น Smart Farming, HealthTech และ Remote Sensing
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc ml-6">
      <li>Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014</li>
      <li>Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019</li>
      <li>Stanford CS231n Course Notes: Transfer Learning</li>
      <li>Oxford VGG Research Group</li>
    </ul>
  </div>
</section>


<section id="concept" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. แนวคิดหลักของ Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="text-base leading-relaxed space-y-8">
    <h3 className="text-xl font-semibold">แนวคิดพื้นฐานของ Transfer Learning</h3>
    <p>
      Transfer Learning เป็นแนวทางหนึ่งในการเรียนรู้ของโมเดลที่นำความรู้จาก task หนึ่งที่เคยเรียนรู้มาก่อนมาใช้กับ task ใหม่ที่เกี่ยวข้อง โดยเฉพาะอย่างยิ่งเมื่อ dataset ใหม่มีขนาดเล็กหรือขาดแคลนข้อมูล ทำให้การฝึกโมเดลจากศูนย์ (training from scratch) ไม่ได้ผลดีเท่าที่ควร
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-medium">Insight:</p>
      <p>
        จากงานวิจัยของ Stanford (Donahue et al., 2014) พบว่า features ที่เรียนรู้จาก CNN ที่ฝึกบน ImageNet สามารถนำมาใช้ในงานใหม่ได้หลากหลายโดยไม่จำเป็นต้องฝึกใหม่ทั้งหมด
      </p>
    </div>

    <h3 className="text-xl font-semibold">องค์ประกอบสำคัญของ Transfer Learning</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Source Domain: โดเมนต้นทางที่มีข้อมูลมาก เช่น ImageNet</li>
      <li>Target Domain: โดเมนปลายทางที่ข้อมูลมีจำกัด เช่น ชุดข้อมูลทางการแพทย์</li>
      <li>Shared Representations: ฟีเจอร์ที่เรียนรู้สามารถแปลงใช้ซ้ำในโดเมนใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold">ประเภทของ Transfer Learning</h3>
    <table className="table-auto border border-gray-300 w-full text-sm">
      <thead>
        <tr className="bg-gray-600">
          <th className="border border-gray-300 px-4 py-2">ประเภท</th>
          <th className="border border-gray-300 px-4 py-2">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Inductive</td>
          <td className="border border-gray-300 px-4 py-2">Target task แตกต่างจาก source task แต่ใช้โมเดลที่ฝึกจาก source</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Transductive</td>
          <td className="border border-gray-300 px-4 py-2">Task เดียวกัน แต่โดเมนต่างกัน เช่น ภาษาอังกฤษ → ภาษาฝรั่งเศส</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Unsupervised</td>
          <td className="border border-gray-300 px-4 py-2">ไม่มี label ในทั้งสองโดเมน เช่น feature extraction จาก text</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-medium">Highlight:</p>
      <p>
        แนวคิดหลักของ Transfer Learning คือการ “ลดต้นทุนข้อมูล” และ “ลดระยะเวลาในการฝึกโมเดล” โดยเฉพาะอย่างยิ่งใน domain-specific task เช่น การวิเคราะห์ภาพถ่ายทางการแพทย์
      </p>
    </div>

    <h3 className="text-xl font-semibold">โมเดลที่นิยมใช้สำหรับ Transfer Learning</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ResNet: โครงสร้าง deep residual ที่สามารถ reuse ได้ดี</li>
      <li>Inception: โมเดลที่มี multi-scale feature extraction</li>
      <li>VGG: โครงสร้างเรียบง่าย เหมาะกับการปรับ fine-tune</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบเมื่อเทียบกับการฝึกจากศูนย์</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ลดเวลาการฝึกโมเดลลงอย่างมาก</li>
      <li>เพิ่มประสิทธิภาพเมื่อ dataset ปลายทางมีขนาดเล็ก</li>
      <li>ประหยัดทรัพยากรทางคอมพิวเตอร์และพลังงาน</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">แหล่งอ้างอิง</h3>
    <ul className="list-decimal list-inside space-y-1 text-sm">
      <li>Yosinski et al. (2014). How transferable are features in deep neural networks? – NIPS.</li>
      <li>Donahue et al. (2014). DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition – ICML.</li>
      <li>Pan & Yang (2010). A Survey on Transfer Learning – IEEE Transactions on Knowledge and Data Engineering.</li>
    </ul>
  </div>
</section>


       <section id="pipeline" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. โครงสร้างพื้นฐานของ Transfer Learning Pipeline</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="text-base leading-relaxed space-y-6">
    <h3 className="text-xl font-semibold">ลำดับขั้นตอนหลักของ Transfer Learning</h3>
    <p>
      Transfer Learning pipeline ในการประยุกต์ใช้โมเดล Convolutional Neural Networks (CNNs) ที่ผ่านการฝึกมาก่อน (pretrained) ประกอบด้วยขั้นตอนที่เป็นระบบ ตั้งแต่การเตรียมข้อมูล การโหลดโมเดล การแช่ค่าพารามิเตอร์ ไปจนถึงการ fine-tune และประเมินผล การจัดวาง pipeline ที่ดีช่วยให้การนำโมเดลไปใช้งานใหม่เป็นไปอย่างมีประสิทธิภาพสูงสุด
    </p>

    <h3 className="text-xl font-semibold">โครงสร้างทั่วไปของ Pipeline</h3>
    <table className="table-auto w-full border border-gray-300 text-sm">
      <thead>
        <tr className="bg-gray-600 dark:bg-gray-800">
          <th className="border px-4 py-2">ลำดับ</th>
          <th className="border px-4 py-2">ขั้นตอน</th>
          <th className="border px-4 py-2">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">1</td>
          <td className="border px-4 py-2">โหลดชุดข้อมูลใหม่</td>
          <td className="border px-4 py-2">เตรียมข้อมูลให้เหมาะสมกับ input shape ของโมเดล pretrained เช่น 224x224x3 สำหรับ ResNet</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">2</td>
          <td className="border px-4 py-2">โหลด pretrained model</td>
          <td className="border px-4 py-2">เลือกโมเดลจากแหล่งมาตรฐาน เช่น ImageNet หรือ OpenAI CLIP และ freeze layers บางส่วน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">3</td>
          <td className="border px-4 py-2">เพิ่ม layer สำหรับ task ใหม่</td>
          <td className="border px-4 py-2">มักจะเพิ่ม fully connected layer ใหม่ด้านบนของ feature extractor</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">4</td>
          <td className="border px-4 py-2">ฝึกเฉพาะส่วนที่เพิ่ม</td>
          <td className="border px-4 py-2">ฝึกเฉพาะ layer ด้านบน โดยไม่เปลี่ยนค่าพารามิเตอร์ของ pretrained model เดิม</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">5</td>
          <td className="border px-4 py-2">Fine-tuning</td>
          <td className="border px-4 py-2">ปลดล็อกบาง layer ของ pretrained model แล้วฝึกพร้อมกับ layer ใหม่</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">6</td>
          <td className="border px-4 py-2">ประเมินผล</td>
          <td className="border px-4 py-2">ใช้ชุดทดสอบ (test set) เพื่อตรวจสอบ performance ของโมเดลใหม่</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-600 dark:bg-yellow-800 text-black dark:text-white border-l-4 border-yellow-500 p-4 mt-6">
      <p className="font-semibold">Insight:</p>
      <p>
        โครงสร้าง pipeline ที่ดีสามารถลดเวลาในการพัฒนาโมเดลใหม่ได้หลายเท่า โดยเฉพาะใน domain ที่มีข้อมูลจำกัด การแยกขั้นตอนอย่างชัดเจนยังช่วยให้สามารถ debug และปรับแต่งโมเดลได้สะดวกยิ่งขึ้น
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8">ตัวอย่างจาก PyTorch</h3>
    <pre className="bg-gray-600 dark:bg-gray-900 text-sm overflow-x-auto p-4 rounded-md">
      <code className="language-python">
import torch
import torchvision.models as models
from torch import nn

# Load pretrained ResNet
resnet = models.resnet50(pretrained=True)

# Freeze base layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # Assume 10 classes

# Now train resnet on new dataset...
      </code>
    </pre>

    <h3 className="text-xl font-semibold mt-8">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS</li>
      <li>Huh et al., "What makes ImageNet good for transfer learning?", CVPR</li>
      <li>He et al., "Deep Residual Learning for Image Recognition", CVPR 2016</li>
    </ul>
  </div>
</section>


     <section id="model-selection" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การเลือก Pretrained Model</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวทางการเลือกโมเดลที่ฝึกไว้ล่วงหน้า</h3>
    <p>
      การเลือกใช้โมเดลที่ผ่านการฝึกมาแล้ว (pretrained model) เป็นขั้นตอนสำคัญของ Transfer Learning ซึ่งขึ้นอยู่กับปัจจัยหลายด้าน เช่น โครงสร้างสถาปัตยกรรม, ประสิทธิภาพที่รายงานไว้บน benchmark datasets, ขนาดของโมเดล, ความสามารถในการปรับใช้บน edge devices, และการรองรับจาก community ในงานวิจัยหรืออุตสาหกรรม.
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded-md">
      <p className="font-semibold">Insight:</p>
      <p>
        จากการศึกษาของ Stanford (2021) และการรีวิวใน NeurIPS พบว่าโมเดลที่ผ่านการ pretrain บนชุดข้อมูลขนาดใหญ่ เช่น ImageNet หรือ JFT-300M มักให้ผลลัพธ์ที่ generalize ได้ดีแม้ถูก fine-tune บน domain ที่ต่างกัน.
      </p>
    </div>

    <h3 className="text-xl font-semibold">ประเภทของ Pretrained Models</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>ResNet:</strong> โครงสร้าง residual blocks ที่ช่วยให้ train ได้ลึกและมีความเสถียร เหมาะสำหรับงาน general image classification.
      </li>
      <li>
        <strong>VGG:</strong> สถาปัตยกรรมที่เข้าใจง่ายและถูกใช้เพื่อการวิเคราะห์ feature map เป็นหลัก.
      </li>
      <li>
        <strong>EfficientNet:</strong> ใช้ compound scaling ที่สามารถ balance ระหว่างความเร็วและความแม่นยำ เหมาะกับ deployment บน edge devices.
      </li>
      <li>
        <strong>Vision Transformer (ViT):</strong> สถาปัตยกรรมใหม่ที่ใช้ self-attention แทน convolution เหมาะกับข้อมูลที่มีโครงสร้างไม่ตายตัว.
      </li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบโมเดลเบื้องต้น</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 text-sm">
        <thead className="bg-gray-600">
          <tr>
            <th className="border px-4 py-2 text-left">Model</th>
            <th className="border px-4 py-2">Top-1 Accuracy</th>
            <th className="border px-4 py-2">Parameters</th>
            <th className="border px-4 py-2">Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">ResNet50</td>
            <td className="border px-4 py-2">76.2%</td>
            <td className="border px-4 py-2">25.6M</td>
            <td className="border px-4 py-2">12</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">EfficientNet-B0</td>
            <td className="border px-4 py-2">77.1%</td>
            <td className="border px-4 py-2">5.3M</td>
            <td className="border px-4 py-2">8</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ViT-B/16</td>
            <td className="border px-4 py-2">84.0%</td>
            <td className="border px-4 py-2">86M</td>
            <td className="border px-4 py-2">25</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">การเลือกโมเดลที่เหมาะกับ Task</h3>
    <p>
      การเลือกโมเดลไม่ได้ขึ้นอยู่กับ accuracy เพียงอย่างเดียว แต่ควรพิจารณา domain ของข้อมูลเป้าหมายและ resource ที่มี เช่น สำหรับงาน segmentation มักใช้ UNet หรือ DeepLabV3+, ขณะที่ NLP ใช้ BERT, RoBERTa หรือ LLaMA และในงาน multi-modal อาจต้องเลือก CLIP หรือ Flamingo.
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded-md">
      <p className="font-semibold">Highlight:</p>
      <p>
        การเลือกโมเดลที่ถูกฝึกมาบน domain ที่ใกล้เคียงจะช่วยลดปัญหา domain shift และ fine-tune ได้เร็วกว่าโมเดลที่ไม่ได้ถูกฝึกในลักษณะเดียวกัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดการโหลด Pretrained Model (PyTorch)</h3>
    <pre className="overflow-x-auto bg-gray-900 text-white text-sm rounded-md p-4">
      <code>
{`import torchvision.models as models
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, num_classes)  # ปรับตาม task
model.eval()`}
      </code>
    </pre>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-1">
      <li>He et al. “Deep Residual Learning for Image Recognition.” CVPR 2016.</li>
      <li>Tan and Le. “EfficientNet: Rethinking Model Scaling.” ICML 2019.</li>
      <li>Dosovitskiy et al. “An Image is Worth 16x16 Words.” ICLR 2021.</li>
      <li>Stanford CS231n Lecture Notes (2023 Edition)</li>
    </ul>
  </div>
</section>


       <section id="resnet-example" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. ตัวอย่าง: ใช้ ResNet50 กับชุดข้อมูลใหม่</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวทางการปรับใช้ ResNet50 ในบริบทใหม่</h3>
    <p>
      ResNet50 เป็นหนึ่งในสถาปัตยกรรม convolutional neural network ที่ถูกใช้อย่างแพร่หลายในงานด้าน image classification เนื่องจากโครงสร้าง residual blocks ที่ช่วยลดปัญหา vanishing gradient และทำให้สามารถฝึกโมเดลที่ลึกได้อย่างมีประสิทธิภาพ โดยทั่วไป ResNet50 ถูกฝึกบนชุดข้อมูล ImageNet แต่สามารถนำมาใช้กับ domain ใหม่ได้ผ่านการทำ fine-tuning.
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded-md">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยของ He et al. (2016) จาก Microsoft Research แสดงให้เห็นว่า residual learning ช่วยให้โมเดลลึกสามารถฝึกได้เร็วและมี performance สูงกว่าโมเดลดั้งเดิม แม้มี layer หลายสิบชั้นก็ตาม.
      </p>
    </div>

    <h3 className="text-xl font-semibold">โครงสร้างของ ResNet50</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Input: 224x224x3 image</li>
      <li>Convolution layer + BatchNorm + ReLU</li>
      <li>Residual Block x 16</li>
      <li>Global Average Pooling</li>
      <li>Fully Connected Layer → Softmax</li>
    </ul>

    <h3 className="text-xl font-semibold">ขั้นตอนการนำ ResNet50 มาใช้งานกับชุดข้อมูลใหม่</h3>
    <ol className="list-decimal pl-6 space-y-2">
      <li>โหลดโมเดลที่ pretrained บน ImageNet</li>
      <li>เปลี่ยน fully connected layer ท้ายสุดให้ตรงกับจำนวน class ของ task ใหม่</li>
      <li>Freeze layer ต้น ๆ ไว้ (optional) เพื่อเก็บ feature จาก domain เดิม</li>
      <li>Fine-tune layer ท้าย ๆ ด้วยชุดข้อมูลใหม่</li>
    </ol>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ด: Fine-Tune ด้วย PyTorch</h3>
    <pre className="overflow-x-auto bg-gray-700 text-white text-sm rounded-md p-4">
      <code>
{`import torchvision.models as models
import torch.nn as nn

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze all layers

model.fc = nn.Linear(2048, NUM_CLASSES)  # เปลี่ยน layer ท้ายสุด
model = model.to(device)`}
      </code>
    </pre>

    <h3 className="text-xl font-semibold">การประเมินประสิทธิภาพ</h3>
    <p>
      การประเมินโมเดลควรใช้ metric ที่เหมาะสมกับบริบท เช่น accuracy, precision, recall, F1-score หรือ confusion matrix. ในกรณีของ multi-class classification, ความแม่นยำ (accuracy) อาจไม่เพียงพอหาก class distribution ไม่สมดุล.
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded-md">
      <p className="font-semibold">Highlight:</p>
      <p>
        การ fine-tune เฉพาะ layer ท้าย ๆ จะช่วยลดความซับซ้อนในการฝึก แต่หากต้องการประสิทธิภาพสูงสุด อาจต้อง unfreeze layer บางส่วนเพื่อเรียนรู้ feature ใหม่ใน domain ปัจจุบัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบการฝึก Fine-Tuning vs Training from Scratch</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 text-sm">
        <thead className="bg-gray-600">
          <tr>
            <th className="border px-4 py-2">Metric</th>
            <th className="border px-4 py-2">Fine-tuned ResNet50</th>
            <th className="border px-4 py-2">Train from Scratch</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Training Time</td>
            <td className="border px-4 py-2">2 hours</td>
            <td className="border px-4 py-2">10+ hours</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Accuracy</td>
            <td className="border px-4 py-2">88%</td>
            <td className="border px-4 py-2">84%</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Required Data</td>
            <td className="border px-4 py-2">10K images</td>
            <td className="border px-4 py-2">มากกว่า 50K images</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ ResNet50</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>โมเดลขนาดใหญ่ อาจไม่เหมาะกับอุปกรณ์ที่มีหน่วยความจำน้อย</li>
      <li>หากชุดข้อมูลใหม่แตกต่างจาก ImageNet มาก การ fine-tune อาจไม่เพียงพอ</li>
      <li>ต้องระวัง overfitting โดยเฉพาะเมื่อมีข้อมูลน้อย</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-1">
      <li>He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.</li>
      <li>Stanford CS231n Lecture Series (2023 Edition)</li>
      <li>Howard & Gugger, "Fastai: A Layered API for Deep Learning." ArXiv, 2020.</li>
      <li>PyTorch Official Tutorials: Transfer Learning (pytorch.org)</li>
    </ul>

  </div>
</section>


       <section id="fine-tune" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    6. เทคนิคการ Fine-Tune อย่างมีประสิทธิภาพ
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">พื้นฐานของการ Fine-Tuning</h3>
    <p>
      Fine-tuning คือกระบวนการปรับพารามิเตอร์ของโมเดล deep learning ที่ถูกฝึกมาก่อนแล้ว (pretrained model)
      ให้เข้ากับงานเฉพาะ โดยไม่เริ่มฝึกจากศูนย์ ช่วยให้ได้ผลลัพธ์ที่ดีขึ้นแม้มีข้อมูลจำกัด เทคนิคนี้ได้รับความนิยมในงานเช่น
      computer vision, NLP และ speech recognition โดยเฉพาะในกรณีที่ต้องการลดต้นทุนเวลาและทรัพยากรในการฝึกโมเดลใหม่ทั้งหมด
    </p>

    <h3 className="text-xl font-semibold">กลยุทธ์หลักในการ Fine-Tune</h3>
    <ul className="list-disc pl-6">
      <li>Unfreezing Layers: เริ่มจากการฝึกเฉพาะ output layer แล้วค่อย ๆ unfreeze layer ลึกขึ้นทีละขั้น</li>
      <li>Discriminative Learning Rate: ใช้อัตราเรียนรู้ (learning rate) ที่ต่างกันในแต่ละ layer ตามระดับ abstraction</li>
      <li>Gradual Unfreezing: ค่อย ๆ เปิดให้ layer ต่าง ๆ ถูกฝึกใหม่ตามลำดับเวลา</li>
    </ul>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <p className="font-medium">
        Insight:
      </p>
      <p>
        งานวิจัยจาก University of Washington พบว่า การ fine-tune เฉพาะ output layer ก่อน แล้วจึงค่อย ๆ ปรับ layer ลึกลงไป
        ส่งผลให้ลด overfitting ได้อย่างมีนัยสำคัญ โดยเฉพาะเมื่อใช้โมเดลที่ใหญ่เกินกว่าขนาดข้อมูล (Howard & Ruder, 2018)
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ด Fine-Tuning ด้วย PyTorch</h3>
    <div className="bg-gray-500 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
  <pre className="whitespace-pre text-sm">
    <code className="language-python">
{`import torch
import torch.nn as nn
from torchvision import models

# Load pretrained model
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace output layer for new task
model.fc = nn.Linear(model.fc.in_features, 10)  # สมมุติว่าเป็น task classification 10 classes

# Unfreeze บาง layer
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# Define optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Train loop (ย่อ)
model.train()
for input, target in dataloader:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold">ข้อควรระวังในการ Fine-Tune</h3>
    <ul className="list-disc pl-6">
      <li>หาก unfreeze เร็วเกินไป อาจทำให้โมเดลจำค่าที่ไม่จำเป็นและเกิด overfitting</li>
      <li>ควรใช้ validation set เพื่อตรวจสอบผลลัพธ์ทุกขั้นตอนของการปรับพารามิเตอร์</li>
      <li>เทคนิคเช่น Early Stopping และ Weight Decay ควรถูกรวมในการควบคุมการฝึก</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุปทางเทคนิค</h3>
    <p>
      Fine-tuning คือเครื่องมือสำคัญในการนำโมเดล pretrained ไปใช้ใน domain ใหม่ โดยเฉพาะเมื่อไม่มีทรัพยากรเพียงพอสำหรับฝึกจากศูนย์
      เทคนิคเช่น Discriminative Learning Rate, Gradual Unfreezing และการประเมิน performance อย่างละเอียด คือหัวใจของความสำเร็จ
    </p>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL.</li>
      <li>Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NIPS.</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Oxford Visual Geometry Group (VGG) - Transfer Learning Resources</li>
    </ul>
  </div>
</section>


    <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Use Cases ที่ได้ผลลัพธ์สูงจาก Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">การนำ Transfer Learning ไปใช้ในงานด้านคอมพิวเตอร์วิทัศน์</h3>
    <p>
      ในงานด้านการรู้จำภาพ (image recognition) และการจำแนกวัตถุ (object detection)
      การใช้ Transfer Learning ช่วยลดเวลาในการฝึกโมเดลอย่างมีนัยสำคัญ โดยเฉพาะเมื่อใช้โครงข่ายที่ได้รับการฝึกมาก่อน
      เช่น ResNet, VGG หรือ EfficientNet กับชุดข้อมูลเฉพาะทาง เช่นภาพทางการแพทย์หรืออุตสาหกรรมการผลิต
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded">
      <p className="font-semibold">Insight:</p>
      <p>
        การปรับใช้โมเดล pretrained ในงานทางการแพทย์ เช่น การวินิจฉัยมะเร็งจากภาพ CT หรือ MRI มีความแม่นยำสูงขึ้นอย่างชัดเจน
        จากการฝึกแบบ fine-tuning บน dataset ขนาดเล็กที่ curated อย่างดี
      </p>
    </div>

    <h3 className="text-xl font-semibold">การใช้งานใน Natural Language Processing (NLP)</h3>
    <p>
      โมเดลภาษาอย่าง BERT, RoBERTa และ T5 ได้รับการนำไปใช้ในงานหลากหลาย เช่นการจัดหมวดหมู่เอกสาร, การตอบคำถาม,
      การสกัดข้อมูลจากข้อความ (Named Entity Recognition) โดยใช้เพียงตัวอย่างฝึกไม่มาก
    </p>
    <ul className="list-disc list-inside">
      <li>Document classification บนข้อมูลภายในองค์กร</li>
      <li>การวิเคราะห์อารมณ์ใน Social Media</li>
      <li>การดึงข้อมูลเฉพาะจากเอกสารทางกฎหมาย</li>
    </ul>

    <h3 className="text-xl font-semibold">การประยุกต์ใน Edge AI และงาน IoT</h3>
    <p>
      การใช้ Transfer Learning บนอุปกรณ์ปลายทาง (Edge Devices) เช่นกล้องวงจรปิด, หุ่นยนต์อุตสาหกรรม,
      หรือระบบ Smart Farming ช่วยให้การประมวลผลภาพและเสียงสามารถเกิดขึ้นภายในอุปกรณ์ได้เลย
      โดยไม่ต้องพึ่งพา cloud ตลอดเวลา
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยของ Stanford และ CMU ระบุว่าการฝึกแบบ fine-tune บนอุปกรณ์ IoT ขนาดเล็กทำได้จริงผ่าน model compression
        และการแช่ weights ของ layer บางส่วนไว้ไม่ให้เรียนรู้ใหม่
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งานที่ประสบความสำเร็จในอุตสาหกรรม</h3>
    <table className="table-auto w-full text-left border mt-6">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="p-2 border">องค์กร</th>
          <th className="p-2 border">Use Case</th>
          <th className="p-2 border">โมเดลที่ใช้</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border">Google Health</td>
          <td className="p-2 border">วินิจฉัยโรคทางจักษุจากภาพเรตินา</td>
          <td className="p-2 border">InceptionV3 pretrained + fine-tune</td>
        </tr>
        <tr>
          <td className="p-2 border">Amazon</td>
          <td className="p-2 border">ตรวจจับ defect ในสินค้าด้วยภาพ</td>
          <td className="p-2 border">ResNet50 pretrained</td>
        </tr>
        <tr>
          <td className="p-2 border">Hugging Face</td>
          <td className="p-2 border">Named Entity Recognition ในระบบ search</td>
          <td className="p-2 border">BERT fine-tuned</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ข้อควรระวังและข้อจำกัด</h3>
    <p>
      แม้ Transfer Learning จะช่วยลดภาระในการฝึกโมเดลใหม่จากศูนย์ แต่หากเลือกโมเดลต้นทางที่แตกต่างจากลักษณะข้อมูลจริงมากเกินไป
      อาจทำให้เกิด overfitting หรือเรียนรู้ผิดพลาดได้ นอกจากนี้บาง architecture มีขนาดใหญ่เกินกว่าจะ deploy บนอุปกรณ์เล็กได้
    </p>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
      <li>arXiv: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</li>
      <li>Nature: Transfer learning with convolutional neural networks for cancer detection</li>
    </ul>
  </div>
</section>


      <section id="caution" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ข้อควรระวังในการใช้ Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">ความเข้าใจผิดเกี่ยวกับการนำโมเดลมาใช้ซ้ำ</h3>
    <p>
      หนึ่งในความเข้าใจผิดที่พบบ่อยคือ การเชื่อว่าการใช้ Transfer Learning เป็นวิธีลัดที่สามารถใช้ได้กับทุกปัญหาโดยไม่ต้องปรับแต่งใด ๆ อย่างไรก็ตาม ในบริบททางวิชาการและการใช้งานจริง การนำโมเดลที่ถูกฝึกมาบนชุดข้อมูลหนึ่งไปใช้กับปัญหาอื่น ต้องพิจารณาหลายมิติ เช่น ความเหมาะสมของ feature, domain shift, และความเสี่ยงต่อ overfitting
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4">
      <p className="font-medium">Insight:</p>
      <p>
        จากรายงานของ Stanford (2021) การนำโมเดล pre-trained ที่มีความซับซ้อนสูงมาใช้กับชุดข้อมูลขนาดเล็กอาจทำให้เกิด overfitting แทนที่จะได้ประโยชน์จากการเรียนรู้ล่วงหน้า
      </p>
    </div>

    <h3 className="text-xl font-semibold">Domain Mismatch: ความแตกต่างของบริบทข้อมูล</h3>
    <p>
      เมื่อโมเดลถูกฝึกมาบนชุดข้อมูลหนึ่ง เช่น ImageNet ซึ่งประกอบด้วยภาพจากบริบททั่วไป การนำไปใช้กับงานเฉพาะทาง เช่น การวิเคราะห์ภาพถ่ายทางการแพทย์ อาจเกิดความไม่สอดคล้องด้าน distribution ซึ่งลดประสิทธิภาพของโมเดล และทำให้ผลลัพธ์เบี่ยงเบนอย่างมาก
    </p>

    <h3 className="text-xl font-semibold">การ Overfit จาก Layer ที่เรียนรู้มากเกินไป</h3>
    <p>
      ในหลายกรณี การ fine-tune layer ลึก ๆ โดยไม่จำกัด learning rate หรือไม่ใช้ regularization อาจทำให้โมเดล "จำ" ข้อมูลใหม่เกินไป และสูญเสีย generalization ที่มีจากโมเดลต้นทาง
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4">
      <p className="font-medium">Highlight:</p>
      <p>
        บทความของ Yosinski et al. (2014, arXiv:1411.1792) แสดงให้เห็นว่า layer ที่อยู่ลึกขึ้นในโมเดลมีความเฉพาะเจาะจงต่อ task เดิมมากขึ้น ซึ่งทำให้การ fine-tune โดยไม่ระวังอาจทำลาย feature representation ที่ดีเดิมไปโดยไม่ตั้งใจ
      </p>
    </div>

    <h3 className="text-xl font-semibold">การเลือกโมเดลที่ไม่สอดคล้องกับลักษณะข้อมูล</h3>
    <p>
      Transfer Learning ไม่ได้หมายความว่าโมเดลทุกแบบจะใช้ได้ดีในทุกสถานการณ์ เช่น การเลือกใช้ ResNet สำหรับงานที่มีข้อมูล temporal อย่าง audio หรือ video อาจไม่มีประสิทธิภาพเท่ากับโมเดลที่ถูกออกแบบมาเฉพาะ เช่น Conformer หรือ Temporal Convolutional Network (TCN)
    </p>

    <h3 className="text-xl font-semibold">การประเมินประสิทธิภาพผิดพลาด</h3>
    <p>
      บางครั้งการปรับแต่ง Transfer Learning ให้มี performance ที่ดูดีบน validation set อาจเกิดจาก data leakage หรือ bias ภายในชุดข้อมูล ซึ่งไม่สามารถสะท้อนประสิทธิภาพในสถานการณ์จริงได้
    </p>

    <table className="table-auto w-full border mt-8">
      <thead>
        <tr className="bg-gray-600">
          <th className="border px-4 py-2">ข้อควรระวัง</th>
          <th className="border px-4 py-2">ผลกระทบที่อาจเกิดขึ้น</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">เลือกโมเดลไม่ตรง domain</td>
          <td className="border px-4 py-2">ประสิทธิภาพต่ำ, ขาด generalization</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ไม่ freeze layer ที่เหมาะสม</td>
          <td className="border px-4 py-2">เกิดการเรียนรู้ซ้ำซ้อน, overfitting</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ใช้ learning rate เดิมจากต้นแบบ</td>
          <td className="border px-4 py-2">ไม่สามารถปรับตัวให้เข้ากับข้อมูลใหม่ได้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ไม่ตรวจสอบ domain shift</td>
          <td className="border px-4 py-2">โมเดลไม่สามารถ generalize ได้</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-5 space-y-2">
      <li>Yosinski et al., "How transferable are features in deep neural networks?", arXiv:1411.1792</li>
      <li>Stanford CS231n Lecture Notes, 2021: Transfer Learning</li>
      <li>IEEE Transactions on Neural Networks and Learning Systems, Vol. 30, No. 9, 2019</li>
      <li>MIT Deep Learning for Self-Driving Cars, Transfer Learning Module</li>
    </ul>
  </div>
</section>


  <section id="research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Research & References</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="text-base leading-relaxed space-y-10">
    <p>
      ความก้าวหน้าในสถาปัตยกรรม Deep Learning ได้รับการขับเคลื่อนจากงานวิจัยจำนวนมหาศาลในช่วงทศวรรษที่ผ่านมา โดยมีบทบาทสำคัญต่อความสามารถในการประมวลผลภาพ เสียง ข้อความ และข้อมูลเชิงซ้อนในระดับที่ไม่เคยเป็นไปได้มาก่อน งานวิจัยเหล่านี้ได้วางรากฐานเชิงทฤษฎี และผลักดันการนำไปประยุกต์ใช้งานจริงในระบบปัญญาประดิษฐ์ระดับโลก
    </p>

    <h3 className="text-xl font-semibold">9.1 งานวิจัยพื้นฐานที่กำหนดทิศทาง</h3>
    <p>
      งานวิจัยที่ได้รับการอ้างถึงมากที่สุด เช่น AlexNet, ResNet, Transformer ได้เปลี่ยนวิธีการสร้างโมเดล AI อย่างถาวร ตัวอย่างเช่น AlexNet ได้พิสูจน์ว่า CNN สามารถเอาชนะระบบ handcrafted features ได้อย่างเด็ดขาด ขณะที่ ResNet แสดงให้เห็นว่าความลึกของโมเดลสามารถเพิ่มขึ้นได้อย่างมีเสถียรภาพผ่าน residual connections
    </p>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-lg">
      <strong>Highlight:</strong>
      <p>
        งานวิจัย "Attention Is All You Need" โดย Vaswani et al. (2017) ถือเป็นหมุดหมายสำคัญในการสร้างกระแส Transformer ที่กลายเป็นสถาปัตยกรรมหลักใน NLP และ multimodal learning
      </p>
    </div>

    <h3 className="text-xl font-semibold">9.2 ขอบเขตของงานวิจัย</h3>
    <ul className="list-disc list-inside">
      <li>Image Classification: AlexNet, VGG, ResNet, EfficientNet</li>
      <li>Object Detection: R-CNN, YOLO, DETR</li>
      <li>Natural Language Processing: Transformer, BERT, GPT</li>
      <li>Multi-modal Models: CLIP, Flamingo, Gemini</li>
      <li>Optimization & Regularization: Dropout, BatchNorm, LayerNorm, Adam</li>
    </ul>

    <h3 className="text-xl font-semibold">9.3 แหล่งอ้างอิงที่แนะนำ</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Deep Learning for Self-Driving Cars</li>
      <li>Harvard NLP: The Annotated Transformer</li>
      <li>CMU 11-785: Introduction to Deep Learning</li>
      <li>Oxford Visual Geometry Group (VGG): Deep Learning Resources</li>
      <li>arXiv preprints: <a href="https://arxiv.org" className="underline text-blue-600">https://arxiv.org</a></li>
      <li>Google Research Blog: <a href="https://ai.googleblog.com" className="underline text-blue-600">https://ai.googleblog.com</a></li>
    </ul>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-lg">
      <strong>Insight:</strong>
      <p>
        การเข้าใจบริบทของงานวิจัยแต่ละฉบับไม่เพียงแต่ช่วยให้สามารถนำโมเดลไปประยุกต์ใช้ได้ดียิ่งขึ้น แต่ยังเปิดโอกาสในการวางแผนสถาปัตยกรรมใหม่ ๆ ที่เหมาะกับโจทย์เฉพาะทาง
      </p>
    </div>
  </div>
</section>


     <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className=" max-w-none text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">การเปลี่ยนแปลงเชิงโครงสร้างของ Deep Learning</h3>
    <p>
      ตลอดทศวรรษที่ผ่านมา โครงสร้างของ Deep Learning ได้เปลี่ยนแปลงจากโครงข่ายแบบเรียบง่าย เช่น MLP (Multilayer Perceptron) ไปสู่สถาปัตยกรรมที่ซับซ้อนมากขึ้น เช่น CNNs, RNNs, Transformers และ Graph Neural Networks (GNNs) โดยการพัฒนาเหล่านี้ตอบโจทย์ข้อมูลที่มีลักษณะแตกต่างกัน เช่น ภาพ เสียง ข้อความ หรือโครงสร้างแบบกราฟ
    </p>

    <div className="bg-yellow-600 p-4 rounded-lg border border-yellow-300">
      <strong>Highlight:</strong> การใช้ Insight Box ในเนื้อหาเชิงเทคนิค มีประโยชน์อย่างยิ่งในการสรุปใจความสำคัญ ช่วยให้ผู้อ่านเห็นแนวโน้มและความสัมพันธ์ได้ชัดเจนยิ่งขึ้น
    </div>

    <h3 className="text-xl font-semibold">บทบาทของ Transformer ในการพลิกวงการ</h3>
    <p>
      Transformer ถือเป็นสถาปัตยกรรมที่สร้างการเปลี่ยนแปลงครั้งใหญ่ในวงการ AI โดยเฉพาะใน Natural Language Processing (NLP) และ Vision Transformer (ViT) ได้ขยายแนวคิดนี้สู่สายงาน Computer Vision อีกด้วย จุดเด่นคือการใช้ Self-Attention แทน Convolution หรือ Recurrence ทำให้สามารถขนานการประมวลผลและเรียนรู้ long-range dependency ได้ดียิ่งขึ้น
    </p>

    <h3 className="text-xl font-semibold">Insight Box ในการออกแบบระบบ AI</h3>
    <p>
      ในการออกแบบระบบ AI โดยเฉพาะในระดับ Production จำเป็นต้องมีมุมมองทั้งเชิงเทคนิคและเชิงกลยุทธ์ Insight Box ทำหน้าที่เป็นองค์ประกอบสำคัญที่สื่อสารมุมมองระดับสูง และช่วยให้นักพัฒนาสามารถตัดสินใจได้อย่างมีประสิทธิภาพ โดยเฉพาะในการออกแบบโครงสร้างที่สามารถปรับตัวต่อข้อมูลขนาดใหญ่ การเปลี่ยนแปลงอย่างรวดเร็ว และข้อจำกัดด้านทรัพยากร
    </p>

    <div className="bg-blue-600 p-4 rounded-lg border border-blue-300">
      <strong>Insight Box:</strong> โมเดลที่ประสบความสำเร็จสูงสุดในแต่ละยุค มักมาพร้อมกับแนวคิดใหม่ในระดับสถาปัตยกรรม เช่น AlexNet (CNN), LSTM (RNN), BERT (Transformer)
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบโมเดลโดยใช้ Insight Box</h3>
    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-700">
  <table className="table-auto min-w-[600px] w-full">
    <thead>
      <tr className="bg-gray-600 dark:bg-gray-700 text-white">
        <th className="px-4 py-2 border border-gray-300 dark:border-gray-600">Architecture</th>
        <th className="px-4 py-2 border border-gray-300 dark:border-gray-600">Strengths</th>
        <th className="px-4 py-2 border border-gray-300 dark:border-gray-600">Use Cases</th>
      </tr>
    </thead>
    <tbody className="bg-white dark:bg-gray-800 text-sm">
      <tr>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">CNN</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Feature locality, spatial invariance</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Image classification, segmentation</td>
      </tr>
      <tr>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">RNN / LSTM</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Temporal modeling, sequential data</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Speech, time series, NLP (before 2017)</td>
      </tr>
      <tr>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Transformer</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Scalability, long-range dependencies</td>
        <td className="border px-4 py-2 border-gray-300 dark:border-gray-700">Language modeling, vision, multi-modal</td>
      </tr>
    </tbody>
  </table>
</div>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดเชิงเทคนิคที่ใช้ Insight Box</h3>
<div className="bg-gray-600 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
  <pre className="whitespace-pre text-sm">
    <code className="language-python">
{`def print_model_architecture(model):
    for idx, (name, layer) in enumerate(model.named_children()):
        print(f"{idx}: {name} - {layer.__class__.__name__}")

print_model_architecture(resnet50)`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc ml-6">
      <li>Vaswani et al., "Attention is All You Need", NeurIPS 2017</li>
      <li>Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021</li>
      <li>LeCun et al., "Deep Learning", Nature, 2015</li>
      <li>Karpathy CS231n Stanford Lectures</li>
      <li>CMU Neural Network Course Notes</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day45 theme={theme} />
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
          <div className="mb-20" />
        </div>
      </div>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day45 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day45_TransferLearning;
