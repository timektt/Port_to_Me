"use client";
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day59 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day59";
import MiniQuiz_Day59 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day59";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day59_GANs = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day59_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day59_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day59_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day59_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day59_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day59_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day59_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day59_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day59_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day59_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day59_11").format("auto").quality("auto").resize(scale().width(500));
  const img12 = cld.image("Day59_12").format("auto").quality("auto").resize(scale().width(500));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 59: Generative Adversarial Networks (GANs)</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

    <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: GAN คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      <strong>Generative Adversarial Networks (GANs)</strong> เป็นสถาปัตยกรรมหนึ่งในแขนง Deep Learning ที่มีความสามารถในการ "สร้างข้อมูลใหม่" (Data Generation) โดยการเรียนรู้จากตัวอย่างข้อมูลจริง
      ถูกเสนอครั้งแรกในปี 2014 โดย Ian Goodfellow และคณะจาก University of Montreal และถือเป็นหนึ่งในก้าวกระโดดสำคัญของการพัฒนา AI ในด้านความคิดสร้างสรรค์ (Creative AI)
    </p>

    <p>
      GANs ประกอบด้วยโมเดลหลัก 2 ส่วน ที่ทำงานในลักษณะ "แข่งขันกัน" (Adversarial Training) ได้แก่:
    </p>

    <ul className="list-disc pl-6 space-y-3">
      <li><strong>Generator (G):</strong> ทำหน้าที่สร้างข้อมูลปลอม (Fake Data) จาก distribution ที่เรียนรู้มา</li>
      <li><strong>Discriminator (D):</strong> ทำหน้าที่แยกแยะว่าข้อมูลที่ได้รับมาเป็นข้อมูลจริง (Real) หรือปลอม (Fake) ที่สร้างโดย Generator</li>
    </ul>

    <p>
      เป้าหมายของ Generator คือพยายาม "หลอก" Discriminator ให้เชื่อว่าข้อมูลปลอมเป็นของจริง ในขณะที่ Discriminator ก็พยายามพัฒนาความสามารถในการตรวจจับความแตกต่าง
      ซึ่งการฝึกแบบ adversarial นี้นำไปสู่การเรียนรู้ Feature ที่ลึกซึ้งขึ้น และสามารถสร้างข้อมูลที่มีความสมจริงสูงได้
    </p>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: การปฏิวัติการสร้างภาพ</h3>
      <p>
        GANs ได้เปลี่ยนโฉมวงการ Computer Vision อย่างมาก ตัวอย่างเช่นการสร้างภาพใบหน้าคนที่ไม่เคยมีอยู่จริง (เช่นโครงการ ThisPersonDoesNotExist), 
        การเพิ่มความละเอียด (Super Resolution), การสร้างภาพสไตล์ศิลปะ (Art Style Transfer), และแม้กระทั่งการจำลองข้อมูลทางการแพทย์
      </p>
    </div>

    <h3>วิวัฒนาการของ GANs ในรอบทศวรรษ</h3>
    <p>
      นับตั้งแต่เปิดตัว GANs ได้มีการต่อยอดออกมาเป็น Variants จำนวนมาก เช่น DCGAN, WGAN, StyleGAN ฯลฯ
      ซึ่งเพิ่มขีดความสามารถให้กับโมเดล ทั้งในแง่ความเสถียรของการฝึก ความสมจริงของผลลัพธ์ และการนำไปประยุกต์ในงานที่หลากหลายยิ่งขึ้น
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">เวอร์ชัน</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">จุดเด่น</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ตัวอย่างการใช้งาน</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">DCGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ปรับใช้ CNN แทน MLP</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">การสร้างภาพเบื้องต้น</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">WGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ใช้ Wasserstein Distance ลด mode collapse</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Image-to-Image Translation</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">StyleGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ความละเอียดสูง / ควบคุม Style ได้</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Synthetic Human Faces</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ความสำคัญของ GANs ใน AI สมัยใหม่</h3>
    <p>
      GANs ได้กลายมาเป็นหัวใจสำคัญของ AI ที่มีความสามารถเชิงสร้างสรรค์ (Creative AI) ทั้งในด้านภาพ เสียง วิดีโอ และเนื้อหาแบบ multimodal
      โดยมีบทบาทในอุตสาหกรรมหลากหลาย ตั้งแต่บันเทิง แฟชั่น ไปจนถึงวิทยาศาสตร์พื้นฐาน
    </p>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: ทำไม GAN ถึงปฏิวัติวงการ?</h3>
      <p>
        แตกต่างจากโมเดล Generative ก่อนหน้า GANs ใช้แนวทางแบบ Game Theory ทำให้กระบวนการเรียนรู้มีความยืดหยุ่นและสมจริงมากขึ้น
        ผลลัพธ์ที่ได้มีความใกล้เคียงกับ Distribution จริงในระดับสูง
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets", NeurIPS.</li>
      <li>Radford et al. (2016). "Unsupervised Representation Learning with Deep Convolutional GANs", arXiv.</li>
      <li>Gulrajani et al. (2017). "Improved Training of Wasserstein GANs", NeurIPS.</li>
      <li>Karras et al. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks", CVPR.</li>
    </ul>

  </div>
</section>


   <section id="architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. โครงสร้างพื้นฐานของ GAN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      โครงสร้างพื้นฐานของ Generative Adversarial Networks (GAN) ประกอบด้วยโมดูล 2 ส่วนหลักที่ทำงานแบบ adversarial คือ <strong>Generator (G)</strong> และ <strong>Discriminator (D)</strong>
      โดยทั้งสองถูกออกแบบให้เรียนรู้ร่วมกันผ่านการแข่งขันในเชิงคณิตศาสตร์
    </p>

    <h3>โครงสร้างของ Generator (G)</h3>
    <p>
      Generator มีหน้าที่แปลงเวกเตอร์สุ่ม z ซึ่งสุ่มมาจาก prior distribution เช่น Normal(0,1) ให้กลายเป็นข้อมูลใหม่ เช่น ภาพ เสียง ข้อความ
      โดยโครงข่ายที่ใช้ใน Generator มักจะเป็นแบบ Deep Neural Networks ที่มี layer ของการ upsampling และ transformation
    </p>

    <h3>โครงสร้างของ Discriminator (D)</h3>
    <p>
      Discriminator ทำหน้าที่ประเมินความน่าจะเป็นว่า ตัวอย่างข้อมูลหนึ่ง ๆ เป็นข้อมูลจริง (real) จาก dataset หรือข้อมูลปลอม (fake) ที่สร้างโดย Generator
      โดยทั่วไป Discriminator จะใช้ Convolutional Neural Networks (CNN) ในงานภาพ หรือ Transformer-based architecture ในงานอื่น ๆ
    </p>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: การเรียนรู้เชิงแข่งขัน</h3>
      <p>
        ความสำคัญของ GANs คือการทำให้ Generator และ Discriminator เรียนรู้ไปพร้อมกัน — เมื่อ Generator สร้างตัวอย่างปลอมได้ดีขึ้น Discriminator ก็ต้องฝึกให้แยกแยะได้เก่งขึ้นเช่นกัน
        ส่งผลให้เกิดการปรับปรุงร่วมกันอย่างต่อเนื่องจนข้อมูลที่สร้างมีความสมจริงสูง
      </p>
    </div>

    <h3>การนิยามฟังก์ชันเป้าหมาย (Objective Function)</h3>
    <p>
      เป้าหมายการฝึก GAN คือการทำให้ Discriminator ไม่สามารถแยกความแตกต่างระหว่างข้อมูลจริงกับข้อมูลปลอมได้สำเร็จ ซึ่งสามารถนิยามได้เป็นฟังก์ชันค่าคาดหวัง (Expected Value) ดังนี้:
    </p>

    <div className="overflow-x-auto mt-6">
      <pre className="bg-gray-500 dark:bg-gray-800 text-sm p-4 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 overflow-x-auto">
<code>
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
</code>
      </pre>
    </div>

    <h3>ตัวอย่างโครงสร้างทางสถาปัตยกรรม (Architecture Diagram)</h3>
    <p>
      โครงสร้าง GAN พื้นฐานสามารถสรุปได้ดังนี้:
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">โมดูล</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Input</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Output</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generator (G)</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Noise Vector z ~ N(0,1)</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Fake Data (e.g. Image)</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Discriminator (D)</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Data (Real or Fake)</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Probability Real/Fake</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: ทำไมต้องเรียน GAN architecture?</h3>
      <p>
        การเข้าใจโครงสร้างพื้นฐานของ GAN เป็นรากฐานสำคัญก่อนจะต่อยอดไปยัง GAN variants ที่มีความซับซ้อนมากขึ้น เช่น Conditional GAN, CycleGAN หรือ StyleGAN
        โดยทุกสถาปัตยกรรมจะยังคงแนวคิด Generator vs Discriminator เป็นแกนกลาง
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets", NeurIPS.</li>
      <li>Radford et al. (2016). "Unsupervised Representation Learning with Deep Convolutional GANs", arXiv.</li>
      <li>Nowozin et al. (2016). "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization", NeurIPS.</li>
      <li>Arjovsky et al. (2017). "Wasserstein GAN", arXiv.</li>
    </ul>

  </div>
</section>


<section id="adversarial-loss" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Adversarial Loss & Game Theoretic View</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      หนึ่งในหัวใจสำคัญที่สุดของ GAN คือแนวคิด **Adversarial Loss** ซึ่งนิยามกระบวนการเรียนรู้ในลักษณะของ "เกม" ระหว่าง Generator (G) และ Discriminator (D)
      โดย Generator มีเป้าหมายเพื่อผลิตตัวอย่างที่เหมือนจริงมากที่สุด ในขณะที่ Discriminator มีเป้าหมายเพื่อตรวจจับตัวอย่างปลอมให้แม่นยำที่สุด
    </p>

    <h3>ฟังก์ชัน Loss แบบ Adversarial</h3>
    <p>
      ฟังก์ชัน Loss พื้นฐานของ GAN สะท้อนกระบวนการเรียนรู้เชิงแข่งขัน โดยนิยามเป็น:
    </p>

    <div className="overflow-x-auto mt-6">
      <pre className="bg-gray-500 dark:bg-gray-800 text-sm p-4 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 overflow-x-auto">
<code>
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
</code>
      </pre>
    </div>

    <p>
      โดยที่:
      <ul className="list-disc pl-6 space-y-2">
        <li>E_x[log D(x)] คือความน่าจะเป็นที่ Discriminator ประเมินตัวอย่างจริงได้ถูกต้อง</li>
        <li>E_z[log(1 - D(G(z)))] คือความน่าจะเป็นที่ Discriminator ประเมินตัวอย่างปลอมที่สร้างโดย Generator ได้ถูกต้อง</li>
      </ul>
    </p>

    <h3>มุมมองในเชิง Game Theory</h3>
    <p>
      เมื่อมองในเชิง Game Theory, การเรียนรู้ของ GAN เป็น **Minimax Game** ที่มีเป้าหมายหาจุดดุลยภาพ (Equilibrium) ที่ Generator และ Discriminator ไม่สามารถ "เอาชนะ" อีกฝ่ายได้อีกต่อไป
      โดยทั้งสองฝ่ายจะปรับเปลี่ยนน้ำหนักเพื่อเอาชนะอีกฝ่ายเสมอ ทำให้เกิดวงจรการเรียนรู้ต่อเนื่อง
    </p>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: Nash Equilibrium ใน GANs</h3>
      <p>
        เมื่อ GAN ฝึกจนถึง Nash Equilibrium จะได้ Generator ที่สามารถผลิตตัวอย่างปลอมที่เหมือนจริงจน Discriminator ไม่สามารถแยกแยะได้ (Predict ~ 0.5) — นี่คือเป้าหมายสูงสุดของการฝึก GAN
      </p>
    </div>

    <h3>การวิเคราะห์เสถียรภาพ (Stability)</h3>
    <p>
      หนึ่งในปัญหาที่พบได้บ่อยของการฝึก GAN คือ **Training Instability** หรือการที่ Loss ฟลูฟลัก (Oscillation) ซึ่งอาจทำให้โมเดลไม่สามารถหาค่า Equilibrium ได้
      งานวิจัยหลายฉบับจึงได้เสนอวิธีการแก้ไข เช่น:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>Wasserstein GAN (WGAN): ใช้ Earth-Mover distance แทน cross-entropy เพื่อเสถียรภาพที่ดีขึ้น</li>
      <li>Gradient Penalty: ป้องกัน Discriminator มี gradient ที่ชันเกินไป</li>
      <li>Label Smoothing: ลด overconfidence ของ Discriminator</li>
    </ul>

    <h3>ตัวอย่าง Loss Function ใน Variants ของ GAN</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Variant</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Loss Function</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Standard GAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-entropy</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">WGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Wasserstein Distance</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">LSGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Least-Squares</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: รูปแบบ Loss ที่เลือกใช้มีผลต่อคุณภาพของภาพที่สร้าง</h3>
      <p>
        Loss function ที่แตกต่างกันส่งผลต่อความคมชัด, ความเสถียรของการฝึก และความสามารถในการสร้างตัวอย่างใหม่ เช่น WGAN มักให้คุณภาพที่ดีและเสถียรกว่า GAN แบบดั้งเดิม
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets", NeurIPS.</li>
      <li>Arjovsky et al. (2017). "Wasserstein GAN", arXiv.</li>
      <li>Mao et al. (2017). "Least Squares Generative Adversarial Networks", ICCV.</li>
      <li>Mescheder et al. (2018). "Which Training Methods for GANs Actually Converge?", ICML.</li>
    </ul>

  </div>
</section>

     <section id="training-challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การฝึกและความท้าทาย</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      การฝึก Generative Adversarial Networks (GANs) มีความซับซ้อนและแตกต่างจากการฝึกโมเดล Deep Learning ทั่วไป เนื่องจากต้องฝึกโมเดลสองตัวพร้อมกัน (Generator และ Discriminator)
      โดยมีวัตถุประสงค์ที่ขัดแย้งกัน (Adversarial Objective) ซึ่งทำให้เกิดความท้าทายด้านเสถียรภาพ และความยากในการหาค่าดุลยภาพ (Equilibrium)
    </p>

    <h3>กระบวนการฝึก GAN</h3>
    <p>
      กระบวนการฝึก GAN โดยทั่วไปประกอบด้วยสองขั้นตอนหลักในแต่ละรอบ:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>อัปเดต Discriminator: ให้สามารถแยกแยะระหว่างตัวอย่างจริง (Real) กับตัวอย่างปลอม (Fake) ที่ Generator สร้างขึ้น</li>
      <li>อัปเดต Generator: เพื่อปรับปรุงการสร้างตัวอย่างปลอม ให้สามารถหลอก Discriminator ได้ดีขึ้น</li>
    </ul>

    <h3>ปัญหาหลักในการฝึก GAN</h3>
    <p>
      ถึงแม้แนวคิดของ GAN จะเรียบง่าย แต่ในการฝึกจริงกลับพบปัญหาหลายประการที่ท้าทายการนำไปใช้งาน:
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ปัญหา</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คำอธิบาย</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Mode Collapse</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generator ผลิตตัวอย่างคล้ายกันซ้ำๆ แทนที่จะครอบคลุมความหลากหลายของข้อมูลจริง</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Vanishing Gradient</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Gradient ของ Generator จางลงเมื่อ Discriminator เก่งเกินไป ทำให้ Generator หยุดเรียนรู้</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Instability</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เกิดการแกว่งของ Loss ไม่สามารถหาค่าดุลยภาพได้</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: Mode Collapse ใน GAN</h3>
      <p>
        หนึ่งในปัญหาคลาสสิกของ GAN คือ Mode Collapse — เมื่อ Generator สร้างตัวอย่างที่หลากหลายไม่ได้ และเลือกสร้างตัวอย่างเฉพาะรูปแบบหนึ่งซ้ำๆ
        วิธีการแก้ไขเช่น Mini-batch Discrimination หรือ Unrolled GAN ถูกนำมาใช้เพื่อลดปัญหานี้
      </p>
    </div>

    <h3>เทคนิคที่ใช้ปรับปรุงการฝึก</h3>
    <p>
      เพื่อเพิ่มเสถียรภาพในการฝึกและหลีกเลี่ยง Mode Collapse นักวิจัยได้พัฒนาเทคนิคต่างๆ ขึ้นมา เช่น:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ Wasserstein Distance แทน Cross-entropy (WGAN)</li>
      <li>เพิ่ม Gradient Penalty</li>
      <li>ใช้ Spectral Normalization</li>
      <li>ใช้ Two Time-Scale Update Rule (TTUR) แยก learning rate สำหรับ G และ D</li>
      <li>การใช้ Progressive Growing ในการเทรน</li>
    </ul>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: เทคนิค TTUR</h3>
      <p>
        การกำหนด learning rate ให้ Discriminator และ Generator ต่างกัน (TTUR) ช่วยให้การฝึกสมดุลมากขึ้น ลดโอกาสที่ D จะเก่งเกินไป หรือ G จะไล่ตามไม่ทัน
      </p>
    </div>

    <h3>ปัจจัยด้าน Hyperparameters ที่สำคัญ</h3>
    <p>
      การเลือก Hyperparameters อย่างเหมาะสมมีผลอย่างมากต่อการฝึก GAN:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>Learning Rate ของ G และ D</li>
      <li>Batch Size</li>
      <li>Optimization Algorithm (เช่น Adam, RMSProp)</li>
      <li>การเลือก Activation Functions (เช่น LeakyReLU)</li>
    </ul>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets", NeurIPS.</li>
      <li>Arjovsky et al. (2017). "Wasserstein GAN", arXiv.</li>
      <li>Miyato et al. (2018). "Spectral Normalization for GANs", ICLR.</li>
      <li>Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS.</li>
    </ul>

  </div>
</section>


  <section id="gan-variants" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Variants ของ GANs</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      ตั้งแต่การเปิดตัว Generative Adversarial Networks (GANs) โดย Ian Goodfellow และคณะในปี 2014 โครงสร้างพื้นฐานนี้ได้รับการพัฒนาและขยายออกไปอย่างรวดเร็ว
      ทำให้เกิด Variants ของ GANs จำนวนมาก โดยแต่ละ Variant พัฒนาขึ้นเพื่อตอบโจทย์ข้อจำกัดบางประการของ GAN ดั้งเดิม เช่น การเพิ่มเสถียรภาพ, การเพิ่มคุณภาพของภาพ, และการขยายขีดความสามารถของโมเดล
    </p>

    <h3>หมวดหมู่หลักของ Variants</h3>
    <p>
      สามารถจัดกลุ่ม Variants ของ GANs ได้เป็นหมวดหมู่ต่างๆ ตามเป้าหมายของการพัฒนา:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>เสริมความเสถียรในการฝึก (Stabilization)</li>
      <li>พัฒนา loss function (Alternative Objectives)</li>
      <li>เพิ่มความละเอียดของภาพ (High-resolution Generation)</li>
      <li>ควบคุมคุณลักษณะของ output (Conditional GANs)</li>
      <li>นำไปใช้ใน application เฉพาะทาง (Domain-specific GANs)</li>
    </ul>

    <h3>ตารางเปรียบเทียบ Variants ที่สำคัญ</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Variant</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คุณสมบัติเด่น</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Publication</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">WGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ใช้ Wasserstein Distance แทน cross-entropy เพื่อความเสถียร</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Arjovsky et al., 2017</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">WGAN-GP</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม Gradient Penalty เพื่อแก้ปัญหา Lipschitz constraint</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Gulrajani et al., 2017</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Progressive GAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สร้างภาพความละเอียดสูงด้วยการฝึกแบบ progressive</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Karras et al., 2018</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">StyleGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ควบคุม style ของ output ได้ละเอียดมาก</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Karras et al., 2019</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Conditional GAN (cGAN)</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สามารถกำหนดเงื่อนไขของ output ได้ (เช่น label หรือ style)</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Mirza and Osindero, 2014</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: ทำไม WGAN จึงสำคัญ?</h3>
      <p>
        Wasserstein GAN (WGAN) เปลี่ยนมุมมองการฝึก GAN จากการใช้ cross-entropy loss มาเป็นการคำนวณระยะทางระหว่าง distribution (Wasserstein distance)
        ทำให้การฝึกมี gradient ที่มีความต่อเนื่องและเสถียรมากขึ้น ลดปัญหา vanishing gradient และช่วยให้การเรียนรู้ของ Generator มีคุณภาพสูงขึ้น
      </p>
    </div>

    <h3>ประโยชน์ของ Variants ต่าง ๆ</h3>
    <p>
      การมี Variants ของ GANs ทำให้นักวิจัยสามารถเลือกใช้โครงสร้างและ objective function ที่เหมาะสมกับงานแต่ละประเภท เช่น:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>งานสร้างภาพความละเอียดสูง เช่น Face Synthesis → Progressive GAN, StyleGAN</li>
      <li>งาน Text-to-Image → cGAN ร่วมกับ text embeddings</li>
      <li>งาน domain transfer → CycleGAN</li>
      <li>การควบคุม style และ attribute → StyleGAN2/StyleGAN3</li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: การใช้ Conditional GANs ในการควบคุม style</h3>
      <p>
        Conditional GANs (cGANs) ช่วยให้นักพัฒนา AI สามารถ “ใส่เงื่อนไข” ลงไปในกระบวนการสร้างภาพ ตัวอย่างเช่น การสร้างภาพใบหน้าที่มีอายุหรืออารมณ์ตามต้องการ,
        หรือการสร้างภาพใน style ศิลปะที่เจาะจง
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Arjovsky et al. (2017). "Wasserstein GAN". arXiv preprint arXiv:1701.07875.</li>
      <li>Gulrajani et al. (2017). "Improved Training of Wasserstein GANs". arXiv preprint arXiv:1704.00028.</li>
      <li>Karras et al. (2018). "Progressive Growing of GANs for Improved Quality, Stability, and Variation". ICLR.</li>
      <li>Karras et al. (2019). "StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks". CVPR.</li>
      <li>Mirza & Osindero (2014). "Conditional Generative Adversarial Nets". arXiv preprint arXiv:1411.1784.</li>
    </ul>

  </div>
</section>


    <section id="visual-results" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ตัวอย่าง Visual Results จาก GANs</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      หนึ่งในปัจจัยที่ทำให้ Generative Adversarial Networks (GANs) ได้รับความนิยมอย่างรวดเร็วในวงการวิจัยและอุตสาหกรรม คือความสามารถในการสร้างภาพสังเคราะห์ที่มีความสมจริงสูง
      โดยเฉพาะในช่วงปี 2017–2020 GANs ได้สร้างความก้าวหน้าครั้งสำคัญด้านคุณภาพภาพที่สร้างขึ้น (image fidelity) และความหลากหลาย (diversity)
    </p>

    <h3>Evolution ของคุณภาพภาพ</h3>
    <p>ตัวอย่างต่อไปนี้แสดงให้เห็นการพัฒนาความสามารถของ GANs ผ่านงานวิจัยสำคัญ:</p>

     <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>


    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: ปัจจัยสำคัญต่อคุณภาพของภาพ</h3>
      <p>
        ปัจจัยที่มีผลต่อความสมจริงของภาพจาก GAN ได้แก่:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>ขนาดของ latent space และการควบคุม style</li>
        <li>วิธีการ training (เช่น progressive growing)</li>
        <li>ชนิดของ loss function (Wasserstein, hinge loss)</li>
        <li>คุณภาพและปริมาณของ training dataset</li>
      </ul>
    </div>

    <h3>การวัดประสิทธิภาพด้วย Inception Score และ FID</h3>
    <p>
      ประสิทธิภาพของ GAN ในการสร้างภาพมักวัดด้วยตัวชี้วัดอย่าง Inception Score (IS) และ Fréchet Inception Distance (FID):
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Model</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Inception Score (IS)</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">FID Score (↓)</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">DCGAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~6.4</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~50–60</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Progressive GAN</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~8.7</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~10–15</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">StyleGAN2</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~9.3</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~2.8–4.0</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: FID score ที่ดีหมายถึงอะไร?</h3>
      <p>
        ค่า FID (Fréchet Inception Distance) ที่ต่ำ แสดงถึงการสร้างภาพที่มี distribution ใกล้เคียงกับข้อมูลจริง ยิ่ง FID ต่ำมากเท่าไร
        ภาพจาก GAN ก็ยิ่งมีความสมจริงสูงและหลีกเลี่ยง visual artifacts ได้ดีขึ้น
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets". NIPS.</li>
      <li>Radford et al. (2015). "Unsupervised Representation Learning with Deep Convolutional GANs". arXiv.</li>
      <li>Karras et al. (2018). "Progressive Growing of GANs for Improved Quality, Stability, and Variation". ICLR.</li>
      <li>Karras et al. (2019). "StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks". CVPR.</li>
      <li>Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium". NIPS.</li>
    </ul>

  </div>
</section>

    <section id="global-research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. GAN ในงานวิจัยระดับโลก</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      ตั้งแต่ปี 2014 เป็นต้นมา Generative Adversarial Networks (GANs) ได้กลายเป็นหัวข้อสำคัญในงานวิจัยปัญญาประดิษฐ์ทั่วโลก
      มีการนำ GANs ไปใช้ในโจทย์หลากหลาย ทั้งภาพ เสียง ข้อความ วิดีโอ และโมเดล 3 มิติ โดยสถาบันวิจัยชั้นนำ เช่น Stanford, MIT, CMU, Google Research, DeepMind และ Facebook AI Research (FAIR)
      ต่างผลักดันขอบเขตของ GANs ให้ก้าวหน้าขึ้นเรื่อย ๆ
    </p>

    <h3>GANs ใน Computer Vision</h3>
    <p>
      งานวิจัยด้าน Computer Vision ถือเป็นสนามหลักที่ GANs ได้แสดงศักยภาพสูงสุด ตัวอย่างหัวข้อสำคัญ ได้แก่:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>Image-to-Image Translation (Pix2Pix, CycleGAN)</li>
      <li>Super-Resolution (SRGAN, ESRGAN)</li>
      <li>Face Generation & Editing (StyleGAN, StyleGAN2, StyleGAN3)</li>
      <li>Domain Adaptation</li>
      <li>Unsupervised Representation Learning</li>
    </ul>

    <h3>GANs ใน Natural Language Processing</h3>
    <p>
      แม้ว่า NLP จะพึ่งโมเดลเชิงลำดับ เช่น Transformers เป็นหลัก แต่ GANs ก็มีบทบาทสำคัญใน:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>Text-to-Image Generation (DALL·E, CogView)</li>
      <li>Style Transfer ของข้อความ</li>
      <li>การสร้างข้อความที่มีความหลากหลาย (diverse text generation)</li>
    </ul>

    <h3>GANs ในงานวิจัย 3 มิติ และเสียง</h3>
    <p>หัวข้อใหม่ที่มาแรงในช่วง 2020–2024 ได้แก่:</p>
    <ul className="list-disc pl-6 space-y-2">
      <li>3D Shape Generation (3D-GAN, voxel-based GANs)</li>
      <li>Neural Radiance Fields (NeRF) + GANs</li>
      <li>Audio Synthesis (WaveGAN, MelGAN)</li>
      <li>Cross-Modal Generation (Text → 3D Model, Text → Audio)</li>
    </ul>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: อิทธิพลของ GAN ต่อวงการวิจัย</h3>
      <p>
        การมาของ GAN เปลี่ยน paradigm ของการสร้างข้อมูลในปัญญาประดิษฐ์ไปอย่างสิ้นเชิง
        จากที่เคยพึ่งพา data augmentation แบบ rule-based → ไปสู่ความสามารถในการ “สร้าง” ข้อมูลใหม่ที่สมจริงมากพอจะใช้เทรนหรือ fine-tune โมเดลใน downstream task
      </p>
    </div>

    <h3>การยอมรับในเวทีวิจัยระดับโลก</h3>
    <p>GAN เป็นหัวข้อที่มีการตีพิมพ์จำนวนมากในงานประชุมวิชาการชั้นนำ เช่น:</p>
    <ul className="list-disc pl-6 space-y-2">
      <li>NeurIPS (Conference on Neural Information Processing Systems)</li>
      <li>ICLR (International Conference on Learning Representations)</li>
      <li>CVPR (Computer Vision and Pattern Recognition)</li>
      <li>ICML (International Conference on Machine Learning)</li>
      <li>ECCV, ICCV</li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: Citation Impact ของ GANs</h3>
      <p>
        Paper ต้นฉบับของ GAN โดย Goodfellow et al. (2014) เป็นหนึ่งใน paper ที่มียอด citation สูงที่สุดในประวัติศาสตร์ AI สมัยใหม่
        มีงานต่อยอดมากกว่า 40,000+ paper ภายในระยะเวลาไม่ถึง 10 ปี
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets". NIPS.</li>
      <li>Isola et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks". CVPR.</li>
      <li>Zhu et al. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". ICCV.</li>
      <li>Karras et al. (2019). "StyleGAN: A Style-Based Generator Architecture". CVPR.</li>
      <li>Donahue et al. (2019). "Large Scale Adversarial Representation Learning". NeurIPS.</li>
    </ul>

  </div>
</section>


     <section id="practical-use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Practical Use Cases</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      Generative Adversarial Networks (GANs) ได้ถูกนำไปใช้งานจริงในหลายอุตสาหกรรม ทั้งเชิงสร้างสรรค์ วิทยาศาสตร์ และเชิงพาณิชย์
      ตัวอย่าง Use Cases เหล่านี้แสดงถึงความยืดหยุ่นและศักยภาพของ GAN ในการขยายขอบเขตของ AI
    </p>

    <h3>1. งานสร้างสรรค์ภาพถ่าย (Creative Image Synthesis)</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การสร้างภาพใบหน้าสมจริง (StyleGAN / StyleGAN3): ใช้ในวงการศิลปะและการโฆษณา</li>
      <li>การ generate ภาพผลิตภัณฑ์ต้นแบบ (product concept art)</li>
      <li>การแปลงภาพร่างเป็นภาพสมจริง (sketch-to-real)</li>
    </ul>

    <h3>2. การเพิ่มคุณภาพของภาพ (Super-Resolution)</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การเพิ่มความละเอียดของภาพเก่า หรือภาพถ่ายจากกล้องความละเอียดต่ำ</li>
      <li>การกู้คืนรายละเอียดในภาพถ่ายดาวเทียม</li>
      <li>การปรับปรุงคุณภาพวิดีโอ (Video Super-Resolution)</li>
    </ul>

    <h3>3. Synthetic Data Generation</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สร้างข้อมูลภาพปลอมสำหรับ train โมเดล classification ในอุตสาหกรรมที่มี data scarcity เช่น การแพทย์</li>
      <li>สร้างข้อมูลสังเคราะห์สำหรับงาน autonomous driving (traffic scenarios, corner cases)</li>
      <li>Augmentation สำหรับงาน security & surveillance</li>
    </ul>

    <h3>4. Text-to-Image Applications</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สร้างภาพจากคำอธิบายข้อความ (DALL·E, CogView)</li>
      <li>AI art generation ในแพลตฟอร์มต่าง ๆ</li>
      <li>สร้างสื่อการเรียนรู้เชิง visual โดยไม่ต้องใช้ stock photo</li>
    </ul>

    <h3>5. การแพทย์และชีววิทยา (Medical & Biomedical)</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Image translation เช่น MRI → CT หรือ PET → CT</li>
      <li>การกู้คืนข้อมูล missing region จากภาพทางการแพทย์</li>
      <li>การสร้างภาพ medical synthetic data เพื่อ train AI diagnostic model</li>
    </ul>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: GANs ช่วยขับเคลื่อน AI เชิงพาณิชย์</h3>
      <p>
        ปัจจุบัน GAN เป็น backbone หลักของหลาย solution เชิงพาณิชย์ ตั้งแต่ platform การสร้าง digital avatar, AI art generator, AI content creator
        ไปจนถึง automated video upscaling — ช่วยลดต้นทุนและเร่งวงจรผลิตเนื้อหาดิจิทัลให้รวดเร็วกว่าที่เคย
      </p>
    </div>

    <h3>6. การประยุกต์ในวงการเกมและภาพยนตร์</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>AI-driven texture generation</li>
      <li>การเพิ่มความสมจริงให้กับ character animation</li>
      <li>style transfer ในวิดีโอเกม</li>
      <li>สร้างเนื้อหาภาพยนตร์แบบ photorealistic pre-visualization</li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: GANs → โมดูลสำคัญของ production pipeline</h3>
      <p>
        สตูดิโอเกมและ VFX ระดับโลก เช่น Electronic Arts, Ubisoft, Pixar และ ILM ได้นำ GANs เข้าไปอยู่ใน production pipeline อย่างจริงจัง
        ช่วยลดเวลา manual design จากหลายสัปดาห์เหลือเพียงไม่กี่วัน
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Karras et al. (2021). "Alias-Free Generative Adversarial Networks (StyleGAN3)". NeurIPS.</li>
      <li>Ledig et al. (2017). "Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)". CVPR.</li>
      <li>Frid-Adar et al. (2018). "GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification". Neurocomputing.</li>
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets". NIPS.</li>
    </ul>

  </div>
</section>


    <section id="gan-best-practice" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. การนำ GAN มาใช้จริง: อะไรคือ best practice?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      การนำ Generative Adversarial Networks (GANs) มาใช้งานจริงในระดับ production มีความซับซ้อน
      ไม่เพียงแต่ด้านการออกแบบ model architecture เท่านั้น แต่ยังรวมไปถึง data pipeline,
      การควบคุมคุณภาพ (quality control), และ resource optimization ซึ่งมีความสำคัญต่อความสำเร็จของโปรเจกต์
    </p>

    <h3>1. เริ่มต้นจาก Dataset ที่มีคุณภาพสูง</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เลือกใช้ dataset ที่มีความสม่ำเสมอ (uniformity) ของ distribution</li>
      <li>ทำความสะอาดข้อมูลอย่างรัดกุมก่อนการฝึก (data cleaning)</li>
      <li>จัด balance ระหว่างจำนวน class ต่าง ๆ ใน dataset</li>
    </ul>

    <h3>2. การเลือก Loss Function ที่เหมาะสม</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ Wasserstein loss (WGAN) เพื่อลดปัญหา mode collapse</li>
      <li>กำหนด gradient penalty สำหรับ stability</li>
      <li>พิจารณาใช้ hinge loss ในการเพิ่ม sharpness ของ output</li>
    </ul>

    <h3>3. Hyperparameter Tuning ที่มีประสิทธิภาพ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ระมัดระวัง learning rate ของ Generator และ Discriminator ไม่ควรตั้งเท่ากันเสมอไป</li>
      <li>กำหนด batch size ที่สัมพันธ์กับ memory ของ GPU</li>
      <li>ใช้ Adam optimizer (β₁≈0.5, β₂≈0.999) เป็น baseline</li>
    </ul>

    <h3>4. ใช้ Techniques สำหรับเพิ่ม stability</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Spectral normalization ใน Discriminator</li>
      <li>ใช้ label smoothing สำหรับ discriminator</li>
      <li>apply Instance Normalization / Adaptive Instance Normalization (AdaIN)</li>
    </ul>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: ปัญหาหลักในการนำ GAN มาใช้จริง</h3>
      <p>
        ความไม่เสถียรระหว่าง training generator/discriminator ถือเป็นจุดท้าทายสำคัญ หากไม่มีแนวทางป้องกัน จะเกิด mode collapse, poor diversity,
        และ difficulty in convergence ซึ่งส่งผลโดยตรงต่อการนำ GAN ไปใช้งานใน production
      </p>
    </div>

    <h3>5. การประเมินผล (Evaluation) ของโมเดล GAN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ FID (Fréchet Inception Distance) เป็น metric หลัก</li>
      <li>พิจารณา Inception Score (IS) ร่วมด้วยในบาง context</li>
      <li>ใช้ human evaluation (perceptual study) สำหรับงานที่เกี่ยวข้องกับ aesthetics</li>
    </ul>

    <h3>6. การ deploy GAN models อย่างปลอดภัย</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ export ผ่าน TensorRT / ONNX สำหรับ inference บน edge devices</li>
      <li>ปรับแต่ง model ให้ latency ต่ำเพียงพอในกรณี real-time generation</li>
      <li>ตรวจสอบ bias หรือ ethical implications ของเนื้อหาที่สร้าง</li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: Best practice ของ Studio ระดับโลก</h3>
      <p>
        Studio ชั้นนำ เช่น NVIDIA, DeepMind, RunwayML นิยมใช้ training pipeline ที่มีการ monitor FID อย่างต่อเนื่อง,
        deploy ผ่าน optimized inference engine และ integrate เข้ากับ CI/CD pipeline
        เพื่อสามารถอัปเดต model version ใหม่ได้อย่างปลอดภัยใน production environment
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Gulrajani et al. (2017). "Improved Training of Wasserstein GANs". NeurIPS.</li>
      <li>Karras et al. (2020). "Analyzing and Improving the Image Quality of StyleGAN". CVPR.</li>
      <li>Miyato et al. (2018). "Spectral Normalization for Generative Adversarial Networks". ICLR.</li>
      <li>Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium". NeurIPS.</li>
    </ul>

  </div>
</section>


    <section id="future-directions" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Future Direction ของ GANs</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      แม้ว่า Generative Adversarial Networks (GANs) ได้แสดงศักยภาพอันสูงในด้านการสร้างข้อมูลจำลอง 
      แต่เส้นทางการพัฒนายังมีความท้าทายและโอกาสอีกมากมาย ซึ่งถูกขับเคลื่อนโดยความต้องการของอุตสาหกรรมและการค้นคว้าในระดับแนวหน้า
    </p>

    <h3>1. เสถียรภาพในการฝึก (Stability of Training)</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การพัฒนา loss function ที่สามารถลด mode collapse อย่างมีประสิทธิภาพ</li>
      <li>การออกแบบ optimization technique ที่ช่วยให้ training มีความเสถียรสูงขึ้น</li>
      <li>การปรับโครงสร้าง discriminator และ generator เพื่อหลีกเลี่ยงการ overfitting</li>
    </ul>

    <h3>2. Interpretable GANs</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สร้างโมเดลที่สามารถอธิบายได้ว่า feature ใดถูกเรียนรู้จาก latent space</li>
      <li>พัฒนา techniques เช่น disentangled representation learning</li>
    </ul>

    <h3>3. Low-Resource GANs</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การพัฒนา GAN ที่สามารถฝึกด้วย dataset ขนาดเล็กได้</li>
      <li>ใช้ transfer learning และ meta-learning สำหรับ fine-tuning ใน domain ใหม่</li>
    </ul>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Insight Box: ทำไม Future Direction ของ GANs จึงสำคัญ?</h3>
      <p>
        แม้ว่า GAN จะสร้างผลงานที่โดดเด่นในหลายด้าน แต่ความไม่เสถียรของ training, ความเข้าใจยากของ latent space,
        และการใช้ resource สูง ยังคงเป็นข้อจำกัดสำคัญ การวิจัยในอนาคตจึงเน้นที่การลดข้อจำกัดเหล่านี้
        เพื่อให้ GAN เป็นเครื่องมือที่ยืดหยุ่นและใช้งานได้กว้างขึ้น
      </p>
    </div>

    <h3>4. Multi-Modal Generative GANs</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>พัฒนา GAN ที่สามารถ generate ข้อมูลหลาย modal พร้อมกัน เช่น ภาพ + เสียง + ข้อความ</li>
      <li>เรียนรู้การ map จาก latent space สู่ multi-modal space</li>
    </ul>

    <h3>5. Application-Specific GANs</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การสร้าง GAN เฉพาะทาง เช่น ใน biomedical imaging, satellite imagery, 3D generation</li>
      <li>การใช้ GAN ในการเร่ง discovery ของวัสดุใหม่ หรือยารักษาโรค</li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="font-semibold mb-2">Highlight Box: เทรนด์สำคัญจากงานประชุมระดับโลก</h3>
      <p>
        แนวโน้มล่าสุดจาก NeurIPS และ CVPR ปีล่าสุดชี้ว่า GANs รุ่นใหม่มักถูกผนวกรวมกับ Large Language Models (LLMs)
        และใช้ attention-based mechanisms เพื่อเพิ่มความแม่นยำในการ generate cross-domain content
      </p>
    </div>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Mescheder et al. (2018). "Which Training Methods for GANs Actually Converge?" ICML.</li>
      <li>Donahue & Simonyan (2019). "Large Scale Adversarial Representation Learning". NeurIPS.</li>
      <li>Karras et al. (2021). "Alias-Free Generative Adversarial Networks". NeurIPS.</li>
      <li>Arjovsky et al. (2017). "Wasserstein GAN". ICML.</li>
    </ul>

  </div>
</section>


    <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Box</h2>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <div className="bg-yellow-500 border border-yellow-300 p-6 rounded-xl">
      <h3 className="font-semibold mb-4 text-xl">Insight Box: บทสรุปสำคัญของ GANs ในการปฏิวัติ Generative AI</h3>
      <p className="mb-4">
        จากจุดเริ่มต้นของ GANs ในปี 2014 โดย Ian Goodfellow จนถึงปัจจุบัน GANs ได้กลายเป็นหนึ่งในสถาปัตยกรรมหลักของโลก AI ที่มีผลกระทบต่อวงการอย่างมหาศาล
        ทั้งในเชิงวิชาการและอุตสาหกรรม ความสามารถของ GANs ในการสร้างข้อมูลใหม่จาก latent space ได้ปฏิวัติหลายแขนง ไม่ว่าจะเป็นศิลปะดิจิทัล การแพทย์
        วิทยาศาสตร์ข้อมูล ไปจนถึงความบันเทิง
      </p>
      <p className="mb-4">
        ที่สำคัญที่สุด GANs ได้เปิดทางให้นักวิจัยเริ่มตั้งคำถามกับขอบเขตของ AI creativity ว่า AI สามารถ "สร้างสรรค์" (creative) ได้จริงหรือไม่
        และได้จุดประกายความก้าวหน้าของโมเดลยุคใหม่ เช่น Diffusion Models และ Latent Transformer Models ที่นำแนวคิดพื้นฐานจาก GANs มาขยายผลต่อ
      </p>
      <p>
        แนวทางในอนาคตจะมุ่งไปสู่ความเข้าใจเชิงลึกของ latent space, เสถียรภาพในการฝึก และ ethical AI — เพื่อให้ GANs กลายเป็นเครื่องมือที่ทรงพลัง 
        แต่ยังคงมีความรับผิดชอบในสังคม
      </p>
    </div>

    <h3>Key Lessons Learned</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>GANs คือจุดเริ่มต้นสำคัญของการปฏิวัติ Generative AI</li>
      <li>โมเดล GANs รุ่นใหม่มีแนวโน้มถูกพัฒนาในบริบท Multi-modal และ Multi-agent systems</li>
      <li>ความเสถียรของ training และ interpretability ของ latent space เป็นหัวข้อวิจัยสำคัญในปัจจุบัน</li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-6 rounded-xl">
      <h3 className="font-semibold mb-4 text-xl">Highlight Box: บทเรียนจากงานประชุมวิจัยชั้นนำ</h3>
      <p>
        ในปีที่ผ่านมา งานประชุม NeurIPS, ICML, และ ICLR ได้เน้นย้ำบทบาทของ GANs ในการสร้าง Realistic synthetic data 
        ที่สามารถนำมาใช้เพิ่มความหลากหลายของ training datasets, ลด bias, และช่วยในงาน research หลายสาขา โดยเฉพาะใน domain ที่มี data scarcity
      </p>
    </div>

    <h3>ตัวอย่างการประยุกต์ใช้งานจริงที่โดดเด่น</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border">Domain</th>
          <th className="px-4 py-3 border">Use Case</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800">
          <td className="px-4 py-3 border">Healthcare</td>
          <td className="px-4 py-3 border">สร้างภาพ MRI ปลอมเพื่อเพิ่ม dataset สำหรับ training</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700">
          <td className="px-4 py-3 border">Art & Design</td>
          <td className="px-4 py-3 border">สร้างงานศิลปะจาก latent representation ที่ไม่เคยมีมาก่อน</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800">
          <td className="px-4 py-3 border">E-commerce</td>
          <td className="px-4 py-3 border">ปรับแต่งภาพสินค้าแบบ personalized บนหน้าร้านออนไลน์</td>
        </tr>
      </tbody>
    </table>

    <h3>Academic References</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al. (2014). "Generative Adversarial Nets". NeurIPS.</li>
      <li>Karras et al. (2019). "StyleGAN: A Style-Based Generator Architecture for GANs". CVPR.</li>
      <li>Radford et al. (2015). "Unsupervised Representation Learning with Deep Convolutional GANs". arXiv.</li>
      <li>Brock et al. (2019). "Large Scale GAN Training for High Fidelity Natural Image Synthesis". ICLR.</li>
    </ul>

  </div>
</section>


      <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Academic References</h2>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <div className="bg-blue-500 border border-blue-300 p-6 rounded-xl">
      <h3 className="font-semibold mb-4 text-xl">บทนำ</h3>
      <p>
        ส่วนนี้รวบรวมเอกสารวิชาการและแหล่งอ้างอิงที่สำคัญเกี่ยวกับ Generative Adversarial Networks (GANs) ที่ได้รับการตีพิมพ์ในวารสารและงานประชุมระดับนานาชาติ
        ซึ่งช่วยขับเคลื่อนการพัฒนาทางทฤษฎีและการประยุกต์ใช้งาน GANs ตลอดทศวรรษที่ผ่านมา
      </p>
    </div>

    <h3>กลุ่มเอกสารตั้งต้น (Foundational Papers)</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). <strong>"Generative Adversarial Nets"</strong>. In *NeurIPS 2014*.</li>
      <li>Radford, A., Metz, L., & Chintala, S. (2015). <strong>"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"</strong>. arXiv preprint arXiv:1511.06434.</li>
      <li>Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). <strong>"Improved Techniques for Training GANs"</strong>. In *NeurIPS 2016*.</li>
    </ul>

    <h3>กลุ่มงานวิจัยด้านสถาปัตยกรรมและการขยายตัว (Architectural Advances & Extensions)</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). <strong>"Progressive Growing of GANs for Improved Quality, Stability, and Variation"</strong>. In *ICLR 2018*.</li>
      <li>Karras, T., Laine, S., & Aila, T. (2019). <strong>"A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)"</strong>. In *CVPR 2019*.</li>
      <li>Brock, A., Donahue, J., & Simonyan, K. (2019). <strong>"Large Scale GAN Training for High Fidelity Natural Image Synthesis"</strong>. In *ICLR 2019*.</li>
    </ul>

    <h3>กลุ่มงานวิจัยด้านการประยุกต์ใช้ (Applications of GANs)</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Bowles, C., Chen, L., Guerrero, R., Bentley, P., Gunn, R., Hammers, A., ... & Rueckert, D. (2018). <strong>"GAN Augmentation: Augmenting training data using generative adversarial networks"</strong>. arXiv preprint arXiv:1810.10863.</li>
      <li>Frid-Adar, M., Klang, E., Amitai, M., Goldberger, J., & Greenspan, H. (2018). <strong>"Synthetic data augmentation using GAN for improved liver lesion classification"</strong>. *IEEE Transactions on Medical Imaging*, 38(3), 677-685.</li>
    </ul>

    <div className="bg-yellow-500 border border-yellow-300 p-6 rounded-xl">
      <h3 className="font-semibold mb-4 text-xl">Highlight Box: ข้อคิดเห็นจากงานประชุมวิจัยชั้นนำ</h3>
      <p>
        ในงานประชุม NeurIPS, CVPR, และ ICLR ช่วง 5 ปีที่ผ่านมา ได้มีการเน้นย้ำว่าการพัฒนาของ GANs ในแง่ architectural improvements และ training stability
        ยังคงเป็นประเด็นสำคัญ นักวิจัยจำนวนมากได้เสนอแนวทาง hybrid model ระหว่าง GANs และ Diffusion models เพื่อเพิ่มความเสถียร
        และลด mode collapse ซึ่งเป็นหนึ่งในปัญหาหลักของ GANs
      </p>
    </div>

    <h3>กลุ่มงานวิจัยด้านความปลอดภัยและจริยธรรม (Ethics and Safety in GANs)</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Obermeyer, Z., & Mullainathan, S. (2019). <strong>"Dissecting racial bias in an algorithm used to manage the health of populations"</strong>. *Science*, 366(6464), 447-453.</li>
      <li>Mirsky, Y., & Lee, W. (2021). <strong>"The Creation and Detection of Deepfakes: A Survey"</strong>. *ACM Computing Surveys*, 54(1), 1-41.</li>
    </ul>

    <h3>แหล่งความรู้เพิ่มเติม (Further Reading)</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Ian Goodfellow’s GAN Tutorial: <a href="https://github.com/goodfeli/adversarial" className="text-blue-700 underline">https://github.com/goodfeli/adversarial</a></li>
      <li>OpenAI Blog: <a href="https://openai.com/research" className="text-blue-700 underline">https://openai.com/research</a></li>
      <li>MIT Deep Learning Lecture: <a href="http://introtodeeplearning.com/" className="text-blue-700 underline">http://introtodeeplearning.com/</a></li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-6 rounded-xl">
      <p>
        เอกสารเหล่านี้ไม่เพียงแต่มีบทบาทสำคัญต่อการพัฒนา Generative AI
        แต่ยังเป็นแนวทางสำหรับนักวิจัยและนักพัฒนารุ่นใหม่ในการต่อยอดสถาปัตยกรรม GANs
        ไปสู่ระบบ AI ที่มีความสามารถสูงขึ้นและมีความปลอดภัยมากขึ้นในอนาคต
      </p>
    </div>

  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day59 theme={theme} />
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
        <ScrollSpy_Ai_Day59 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day59_GANs;
