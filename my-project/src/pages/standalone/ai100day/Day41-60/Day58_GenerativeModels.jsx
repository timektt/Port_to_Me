import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day58 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day58";
import MiniQuiz_Day58 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day58";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day58_GenerativeModels = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day58_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day58_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day58_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day58_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day58_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day58_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day58_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day58_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day58_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day58_10").format("auto").quality("auto").resize(scale().width(500));

   return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}> 
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>


      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 58: Generative Models Overview (GANs, VAEs)</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

<section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    1. บทนำ: Generative Models คืออะไร?
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      Generative Models เป็นสถาปัตยกรรมในสาขา Deep Learning ที่มีเป้าหมายในการสร้างข้อมูลใหม่ (new data instances) ที่มีคุณสมบัติคล้ายคลึงกับข้อมูลที่ถูกใช้ฝึก (training data) โดยตรง จากการเรียนรู้ distribution ของข้อมูลต้นแบบ โมเดลเหล่านี้สามารถสร้างตัวอย่างใหม่ ๆ ที่มีความสมจริงและมีโครงสร้างคล้ายกับข้อมูลจริง ซึ่งถือเป็นหัวใจสำคัญของ AI ในยุคปัจจุบัน โดยเฉพาะด้านการสร้างสรรค์ (creative AI)
    </p>

    <h3 className="text-xl font-semibold">ลักษณะเด่นของ Generative Models</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สามารถสร้างข้อมูลใหม่ ๆ ได้ (Data Generation)</li>
      <li>เรียนรู้ distribution ของข้อมูล (Learning Data Distribution)</li>
      <li>สนับสนุนการทำ Data Augmentation และ Data Simulation</li>
      <li>ขยายขีดความสามารถของ AI ด้านความคิดสร้างสรรค์ เช่น ศิลปะ ดนตรี การออกแบบ</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างประเภทของ Generative Models ที่สำคัญ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Generative Adversarial Networks (GANs)</li>
      <li>Variational Autoencoders (VAEs)</li>
      <li>Autoregressive Models (AR)</li>
      <li>Diffusion Models</li>
      <li>Normalizing Flows</li>
    </ul>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
      <p className="text-sm font-medium">
        Insight: Generative Models เป็นกลไกสำคัญที่นำไปสู่การพัฒนา AI เชิงสร้างสรรค์ในอนาคต โดยมีการนำไปใช้ทั้งในวงการศิลปะ การออกแบบ เกม การแพทย์ และอุตสาหกรรมภาพยนตร์
      </p>
    </div>

    <h3 className="text-xl font-semibold">ประโยชน์และความสำคัญของ Generative Models</h3>
    <p>
      งานวิจัยจาก Stanford (Goodfellow et al., 2014) และ MIT (Kingma & Welling, 2014) ได้แสดงให้เห็นว่า Generative Models สามารถเปลี่ยนโฉมการสร้างข้อมูลได้อย่างมีนัยสำคัญ ตัวอย่างเช่น GANs สามารถสร้างภาพเสมือนจริงที่ยากจะแยกออกจากภาพถ่ายจริง ในขณะที่ VAEs ช่วยให้การ encode และ decode feature space ของข้อมูลมีความต่อเนื่องและเข้าใจได้ดีขึ้น
    </p>

    <h3 className="text-xl font-semibold">ข้อจำกัดและความท้าทายเบื้องต้น</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ความยากในการฝึก (Training Instability) โดยเฉพาะใน GANs</li>
      <li>ความสมจริง (Realism) ของข้อมูลที่สร้างยังมีข้อจำกัดในบางบริบท</li>
      <li>ปัญหา bias และความไม่สมดุลของ dataset ที่ส่งผลต่อผลลัพธ์ที่สร้างขึ้น</li>
    </ul>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <p className="text-sm font-medium">
        Highlight: แม้ Generative Models จะมีความสามารถสูง แต่ความท้าทายด้านการควบคุมคุณภาพและความน่าเชื่อถือของข้อมูลที่สร้างขึ้นยังเป็นหัวข้อวิจัยที่สำคัญในปัจจุบัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al., "Generative Adversarial Networks", arXiv, 2014</li>
      <li>Kingma & Welling, "Auto-Encoding Variational Bayes", arXiv, 2014</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Introduction to Deep Learning (2023)</li>
      <li>CMU Deep Generative Models, 2022</li>
    </ul>
  </div>
</section>


   <section id="principles" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. หลักการของ Generative Modeling</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      Generative Modeling คือกระบวนการเรียนรู้การแจกแจงความน่าจะเป็น (probability distribution) ของข้อมูล เพื่อนำมาใช้สร้างตัวอย่างข้อมูลใหม่ที่มีลักษณะเหมือนกับข้อมูลจริง โดยอาศัยหลักการจากวิชาสถิติและคณิตศาสตร์ขั้นสูง ผสานกับโครงสร้าง Neural Networks ที่มีศักยภาพสูงในการประมาณฟังก์ชันที่ซับซ้อน
    </p>

    <h3 className="text-xl font-semibold">โมเดล Generative ทำงานอย่างไร?</h3>
    <p>
      หลักการพื้นฐานของ Generative Models คือการเรียนรู้ <strong>p(x)</strong> หรือ <strong>p(x|z)</strong> ซึ่งหมายถึงโอกาสที่ข้อมูลตัวอย่าง <em>x</em> จะเกิดขึ้น จาก latent variable <em>z</em> ที่เป็นตัวแทน feature ภายในของข้อมูล โดยทั่วไปกระบวนการนี้แบ่งออกเป็น 2 ขั้นตอนหลัก:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>การ encode ข้อมูลลงใน latent space</li>
      <li>การ decode latent space กลับเป็นข้อมูลใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold">ประเภทหลักของ Generative Models</h3>
    <p>
      จากการพัฒนาต่อเนื่องของวงการ Deep Learning ได้เกิดสถาปัตยกรรม Generative Models หลากหลายประเภท เช่น:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Explicit Likelihood Models:</strong> เช่น Variational Autoencoders (VAEs)</li>
      <li><strong>Implicit Models:</strong> เช่น Generative Adversarial Networks (GANs)</li>
      <li><strong>Autoregressive Models:</strong> เช่น PixelRNN, PixelCNN</li>
      <li><strong>Diffusion Models:</strong> เช่น Denoising Diffusion Probabilistic Models (DDPM)</li>
    </ul>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
      <p className="text-sm font-medium">
        Insight: แม้แต่โมเดลที่เน้น discriminative learning (เช่น GPT) ก็มีพื้นฐานที่สามารถขยายไปสู่ generative learning ได้ หากมีการออกแบบ objective functions ที่เหมาะสม
      </p>
    </div>

    <h3 className="text-xl font-semibold">เป้าหมายทางคณิตศาสตร์</h3>
    <p>
      โมเดลประเภทนี้มีเป้าหมายหลักในการประมาณพารามิเตอร์ <strong>θ</strong> ของ model distribution <strong>p<sub>θ</sub>(x)</strong> ให้ใกล้เคียงกับ true data distribution <strong>p<sub>data</sub>(x)</strong> ให้มากที่สุด ซึ่งวัดผลผ่าน divergence metric เช่น KL divergence หรือ Jensen-Shannon divergence
    </p>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบประเภท Generative Models</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">วิธีการ</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ตัวอย่างโมเดล</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Explicit Likelihood</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Maximize likelihood</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">VAE</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Implicit</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Adversarial training</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GAN</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Autoregressive</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Chain rule of probability</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">PixelCNN</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Diffusion</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Reverse noise process</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">DDPM</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <p className="text-sm font-medium">
        Highlight: ในปัจจุบัน Diffusion Models กำลังได้รับความนิยมสูงสุดสำหรับ generative tasks เช่นการสร้างภาพความละเอียดสูง (high-res image synthesis)
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Kingma & Welling (2014), "Auto-Encoding Variational Bayes" — arXiv:1312.6114</li>
      <li>Goodfellow et al. (2014), "Generative Adversarial Nets" — NeurIPS</li>
      <li>Ho et al. (2020), "Denoising Diffusion Probabilistic Models" — arXiv:2006.11239</li>
      <li>Stanford CS236: Deep Generative Models</li>
      <li>MIT 6.S191: Advanced Deep Learning Lectures</li>
    </ul>

  </div>
</section>


      <section id="gans" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    3. GANs (Generative Adversarial Networks)
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      Generative Adversarial Networks (GANs) เป็นสถาปัตยกรรม Deep Learning ประเภทหนึ่งที่นำเสนอครั้งแรกโดย Ian Goodfellow และคณะในปี 2014 ภายในงานประชุม NeurIPS โดยมีเป้าหมายเพื่อฝึกโมเดลให้สามารถสร้างตัวอย่างข้อมูลใหม่ที่มีความสมจริงสูง GANs ใช้แนวคิดการเรียนรู้แบบ adversarial หรือการต่อสู้ระหว่างโมเดลสองตัว ได้แก่ Generator และ Discriminator ซึ่งช่วยผลักดันซึ่งกันและกันให้มีประสิทธิภาพสูงขึ้นอย่างต่อเนื่อง
    </p>

    <h3>โครงสร้างพื้นฐานของ GAN</h3>
    <p>
      โครงสร้างหลักของ GAN ประกอบด้วยสององค์ประกอบสำคัญ:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Generator (G):</strong> โมเดลที่สร้างตัวอย่างปลอมขึ้นมา (เช่น ภาพ หรือข้อความ) จากข้อมูล random noise หรือ latent vector
      </li>
      <li>
        <strong>Discriminator (D):</strong> โมเดลที่ทำหน้าที่แยกแยะว่าข้อมูลที่ได้รับเป็นของจริง (จาก dataset) หรือเป็นของปลอม (จาก Generator)
      </li>
    </ul>

    <h3>กลไกการเรียนรู้ของ GAN</h3>
    <p>
      ในกระบวนการ training Generator จะพยายามหลอก Discriminator ให้เชื่อว่าข้อมูลปลอมเป็นของจริง ขณะที่ Discriminator ก็จะพยายามแยกแยะอย่างถูกต้อง การเรียนรู้จึงมีลักษณะเป็นการเล่นเกมสองฝ่าย (minimax game) ดังนี้:
    </p>
  <div className="overflow-x-auto mt-4 mb-6 border border-gray-300 dark:border-gray-600 rounded-lg">
  <pre className="whitespace-pre px-4 py-3 text-sm text-gray-800 dark:text-gray-100 bg-gray-50 dark:bg-gray-800 leading-relaxed">
<code>min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
</code>
  </pre>
</div>

    <p>
      โดยที่ <code>x</code> คือข้อมูลจริงจาก dataset และ <code>z</code> คือ latent vector ที่สุ่มขึ้นมา
    </p>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
      <p className="font-medium">
        Highlight: จุดเด่นของ GAN คือการสามารถสร้างข้อมูลใหม่ที่มีความสมจริงสูงโดยไม่ต้องกำหนด explicit density function ของข้อมูลเป้าหมาย
      </p>
    </div>

    <h3>ปัญหาที่พบในการเทรน GAN</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Mode Collapse:</strong> Generator สร้างตัวอย่างซ้ำ ๆ ในบาง mode ของ distribution
      </li>
      <li>
        <strong>Training Instability:</strong> ค่า gradient มีแนวโน้มที่จะหายไปหรือกระจายไม่เสถียร
      </li>
      <li>
        <strong>Difficulty in Convergence:</strong> การหาจุดสมดุลระหว่าง G และ D เป็นเรื่องยากมากในทางปฏิบัติ
      </li>
    </ul>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <p className="font-medium">
        Insight Box: เทคนิคเช่น Wasserstein GAN (WGAN), Spectral Normalization, และ Gradient Penalty ถูกนำมาใช้เพื่อแก้ปัญหา training instability ของ GAN
      </p>
    </div>

    <h3>วิวัฒนาการของ GANs</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>DCGAN:</strong> Deep Convolutional GANs ใช้ Convolutional layers เพื่อพัฒนา quality ของภาพที่สร้าง</li>
      <li><strong>WGAN:</strong> ใช้ Wasserstein distance แทน Jensen-Shannon divergence เพื่อลดปัญหา gradient vanishing</li>
      <li><strong>StyleGAN:</strong> GAN ที่สร้างภาพระดับความละเอียดสูงพร้อม controllable style</li>
      <li><strong>Conditional GAN:</strong> เพิ่มเงื่อนไขให้กับการสร้าง เช่น การสร้างภาพตาม label ที่กำหนด</li>
    </ul>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Model</th>
            <th className="px-4 py-3 border">จุดเด่น</th>
            <th className="px-4 py-3 border">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800">
            <td className="border px-4 py-3">DCGAN</td>
            <td className="border px-4 py-3">โครงสร้าง convolutional เสถียรกว่า MLP</td>
            <td className="border px-4 py-3">ยังพบ mode collapse</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="border px-4 py-3">WGAN</td>
            <td className="border px-4 py-3">Gradient flow ดีขึ้น ลด instability</td>
            <td className="border px-4 py-3">ต้อง tune hyperparameters มาก</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="border px-4 py-3">StyleGAN</td>
            <td className="border px-4 py-3">ควบคุม style ได้ ผลลัพธ์คุณภาพสูง</td>
            <td className="border px-4 py-3">ใช้ compute สูงมาก</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>สรุป</h3>
    <p>
      GANs ถือเป็นรากฐานสำคัญของ Generative AI ในปัจจุบัน โดยมีการนำไปต่อยอดในหลายแขนง เช่น image synthesis, super-resolution, image-to-image translation และ content generation เชิงสร้างสรรค์ในวงกว้าง
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-2 break-words">
      <li>Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014</li>
      <li>Arjovsky et al., "Wasserstein GAN", arXiv:1701.07875</li>
      <li>Karras et al., "A Style-Based Generator Architecture for GANs", CVPR 2019</li>
      <li>MIT Deep Learning Lecture Series 2023</li>
      <li>Stanford CS231n Lecture Notes</li>
    </ul>
  </div>
</section>


 <section id="vaes" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. VAEs (Variational Autoencoders)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      Variational Autoencoders (VAEs) คือสถาปัตยกรรมการเรียนรู้เชิงกำเนิด (Generative Modeling) ประเภทหนึ่ง
      ที่ผสมผสานระหว่างโครงสร้าง Autoencoder แบบดั้งเดิมเข้ากับการวิเคราะห์เชิงความน่าจะเป็น
      โดยเป้าหมายของ VAE คือการเรียนรู้ latent space distribution ของข้อมูล และสามารถสุ่มตัวอย่าง
      (sampling) จาก latent space เพื่อสร้างตัวอย่างใหม่ได้อย่างต่อเนื่อง (continuous generation)
    </p>

    <section>
      <h3 className="text-xl font-semibold">4.1 โครงสร้างของ VAE</h3>
      <p>
        สถาปัตยกรรมของ VAE ประกอบด้วย 2 ส่วนหลัก:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          <strong>Encoder:</strong> ทำหน้าที่แปลงข้อมูลต้นฉบับ (input data) ให้เป็นพารามิเตอร์ของ distribution
          (เช่น mean และ variance ของ Gaussian distribution) ใน latent space
        </li>
        <li>
          <strong>Decoder:</strong> ทำหน้าที่สุ่มตัวอย่าง (sample) จาก distribution ที่ได้ แล้วนำไปแปลงกลับ
          (decode) เพื่อสร้างข้อมูลใหม่ที่ใกล้เคียงกับ input เดิม
        </li>
      </ul>

      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: VAEs ต่างจาก Autoencoder ตรงที่มีการกำหนด distribution บน latent space ทำให้สามารถสร้างข้อมูลใหม่ได้แบบ probabilistic
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">4.2 Loss Function ของ VAE</h3>
      <p>
        VAE ใช้ loss function ที่ประกอบด้วยสององค์ประกอบหลัก:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          <strong>Reconstruction Loss:</strong> ใช้วัดความแตกต่างระหว่าง input กับ output ที่ถูกสร้างขึ้นมาใหม่
        </li>
        <li>
          <strong>Kullback–Leibler (KL) Divergence:</strong> วัดความแตกต่างระหว่าง distribution ของ latent space
          ที่เรียนรู้ได้ กับ prior distribution (เช่น standard normal distribution)
        </li>
      </ul>
   <div className="overflow-x-auto mt-4 mb-6 border border-gray-300 dark:border-gray-600 rounded-lg">
  <pre className="whitespace-pre px-4 py-3 text-sm text-gray-800 dark:text-gray-100 bg-gray-50 dark:bg-gray-800 leading-relaxed">
    <code className="language-math">
L = E_q(z|x)[log p(x|z)] - D_KL[q(z|x) || p(z)]
    </code>
  </pre>
</div>


      <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Insight: KL divergence ทำหน้าที่ควบคุมไม่ให้ latent space แตกกระจายจนสูญเสียคุณสมบัติในการสุ่มตัวอย่าง
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">4.3 การใช้งานของ VAE</h3>
      <p>
        VAEs ได้รับความนิยมในการประยุกต์ใช้งานหลายประเภท เช่น:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>การสร้างภาพใหม่ (Image Generation)</li>
        <li>การเติมช่องว่างของข้อมูล (Data Imputation)</li>
        <li>การวิเคราะห์เชิงคุณสมบัติของ latent space</li>
        <li>การลดมิติ (Dimensionality Reduction)</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">4.4 ข้อดีและข้อจำกัดของ VAE</h3>
      <div className="overflow-x-auto mt-6">
        <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
          <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <tr>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อดี</th>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อจำกัด</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">
                Latent space ต่อเนื่อง → รองรับ smooth interpolation
              </td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">
                คุณภาพของ sample ต่ำกว่า GANs ในหลายกรณี
              </td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">
                สถาปัตยกรรมง่ายต่อการฝึก (training stability สูง)
              </td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">
                มีโอกาสเกิด blurry outputs จาก reconstruction loss
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl mt-6">
        <p className="text-sm font-medium">
          Highlight: VAEs เหมาะกับงานที่ต้องการ interpretability และ smooth latent space มากกว่างานเน้น realistic quality แบบ GANs
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">4.5 อ้างอิง</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          Kingma, D.P., & Welling, M. (2014). "Auto-Encoding Variational Bayes". arXiv preprint arXiv:1312.6114.
        </li>
        <li>
          Doersch, C. (2016). "Tutorial on Variational Autoencoders". arXiv:1606.05908.
        </li>
        <li>
          MIT 6.S191: Deep Learning for Self-Driving Cars – Lecture on VAEs.
        </li>
        <li>
          Stanford CS231n: Convolutional Neural Networks for Visual Recognition – Generative Models Section.
        </li>
      </ul>
    </section>
  </div>
</section>


   <section id="gans-vs-vaes" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. เปรียบเทียบ GANs vs VAEs</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      แม้ว่า Generative Adversarial Networks (GANs) และ Variational Autoencoders (VAEs)
      จะถูกจัดเป็นสถาปัตยกรรมหลักของ Generative Models เหมือนกัน แต่ทั้งสองแนวทางมีข้อดี ข้อจำกัด และปรัชญาในการออกแบบที่แตกต่างกันอย่างชัดเจน
      ความเข้าใจในความแตกต่างเชิงโครงสร้างและกลไกของการเรียนรู้ (learning dynamics) จะช่วยให้สามารถเลือกใช้งานได้อย่างมีประสิทธิภาพตามบริบท
    </p>

    <section>
      <h3 className="text-xl font-semibold">5.1 สถาปัตยกรรมและกลไกการเรียนรู้</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          <strong>GANs:</strong> ใช้การฝึกแบบ adversarial ระหว่าง generator และ discriminator เพื่อเรียนรู้ distribution ของข้อมูล และมุ่งเน้นที่การสร้างตัวอย่างใหม่ที่ดูสมจริง (photorealistic)
        </li>
        <li>
          <strong>VAEs:</strong> สร้าง latent space แบบ probabilistic และใช้การสุ่มตัวอย่างเพื่อสร้างข้อมูลใหม่ โดยโฟกัสที่การเข้าใจ latent structure มากกว่าภาพที่สมจริง
        </li>
      </ul>

      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: GANs เน้นคุณภาพของ sample เป็นหลัก ส่วน VAEs เน้นความสามารถในการตีความ (interpretability) ของ latent space
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">5.2 คุณภาพของภาพที่สร้างได้</h3>
      <p>
        โดยทั่วไป ผลลัพธ์จาก GANs มีความคมชัดและใกล้เคียงกับภาพจริงมากกว่า VAE เนื่องจาก generator ของ GANs ถูกฝึกให้หลอก discriminator โดยตรง
        ในขณะที่ VAE ต้อง trade-off ระหว่าง reconstruction loss และ KL divergence ซึ่งอาจทำให้เกิด "blurry output"
      </p>
    </section>

    <section>
      <h3 className="text-xl font-semibold">5.3 ความเสถียรในการฝึก (Training Stability)</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          <strong>GANs:</strong> มีปัญหาการฝึกหลายอย่าง เช่น mode collapse, non-convergence และ training instability เนื่องจากเป็นเกมสองฝ่าย (minimax game)
        </li>
        <li>
          <strong>VAEs:</strong> โดยทั่วไปมีการฝึกที่เสถียรกว่า และไม่เสี่ยงต่อ mode collapse
        </li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">5.4 ความสามารถในการจัดโครงสร้างของ latent space</h3>
      <p>
        Latent space ของ VAE ถูกออกแบบให้ต่อเนื่อง (continuous) และมีโครงสร้างเชิงความน่าจะเป็น ทำให้สามารถใช้งานได้ดีในงานที่ต้องการการควบคุม latent factors (เช่น style transfer, interpolation)
        ในขณะที่ latent space ของ GANs อาจไม่ต่อเนื่อง และมักจะไม่มี interpretability ที่ดีเท่า VAE
      </p>
    </section>

    <section>
      <h3 className="text-xl font-semibold">5.5 เปรียบเทียบในเชิงตาราง</h3>
      <div className="overflow-x-auto mt-6">
        <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
          <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <tr>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คุณสมบัติ</th>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">GANs</th>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">VAEs</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">คุณภาพของภาพ</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">คมชัด สมจริง สูง</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">มีโอกาส blurry</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training stability</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ (mode collapse)</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
            </tr>
            <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Latent space</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ไม่ต่อเนื่อง</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่อเนื่อง</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Interpretability</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">5.6 ข้อสรุป</h3>
      <p>
        ทั้ง GANs และ VAEs ต่างมีจุดแข็งในด้านที่แตกต่างกัน โดย GANs เหมาะกับงานที่ต้องการภาพสมจริงที่สุด
        เช่น การสร้างภาพ high-resolution หรือ deepfake ในขณะที่ VAEs เหมาะกับงานที่ต้องการ latent space ที่ต่อเนื่องและเข้าใจได้ เช่น data compression และ anomaly detection
      </p>

      <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Insight: ไม่มีสถาปัตยกรรมใดดีที่สุดในทุกกรณี — การเลือกใช้ขึ้นอยู่กับบริบทของงาน และความสมดุลระหว่างคุณภาพของผลลัพธ์กับความสามารถในการควบคุม
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">5.7 อ้างอิง</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>Goodfellow, I. et al. (2014). "Generative Adversarial Nets". NeurIPS 2014.</li>
        <li>Kingma, D.P., & Welling, M. (2014). "Auto-Encoding Variational Bayes". arXiv:1312.6114.</li>
        <li>MIT 6.S191: Deep Generative Models – Lecture notes.</li>
        <li>Stanford CS236: Advanced Topics in AI – GANs and VAEs comparison.</li>
      </ul>
    </section>
  </div>
</section>


<section id="research-examples" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ตัวอย่างงานวิจัยสำคัญ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      ในช่วงทศวรรษที่ผ่านมา วงการ Generative Models ได้มีพัฒนาการอย่างรวดเร็ว เกิดงานวิจัยที่ผลักดันขอบเขตของการสร้างข้อมูลจำลอง (synthetic data generation) ในหลากหลายสาขา
      โดยเฉพาะอย่างยิ่งในงานด้านภาพ เสียง และวิดีโอ ซึ่งหลายผลงานได้รับการอ้างอิงอย่างแพร่หลายในวารสารและการประชุมระดับโลก เช่น NeurIPS, CVPR, ICML และ ICLR
    </p>

    <section>
      <h3 className="text-xl font-semibold">6.1 Generative Adversarial Nets (GANs)</h3>
      <p>
        งานวิจัยชิ้นบุกเบิกที่เสนอโดย Ian Goodfellow และทีม ในปี 2014 ได้แนะนำโมเดล Generative Adversarial Networks (GANs) ซึ่งนำเสนอ framework แบบ minimax game ระหว่าง generator และ discriminator
        นับเป็นก้าวสำคัญที่เปลี่ยนวิธีคิดในการสร้าง distribution ของข้อมูลอย่างสมจริง โดยเฉพาะในงาน image synthesis
      </p>

      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: GANs ได้รับการอ้างอิงมากกว่า 50,000 ครั้ง (Google Scholar, 2024) ถือเป็นหนึ่งในงานวิจัยที่มีอิทธิพลที่สุดในวงการ Deep Learning
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">6.2 Progressive Growing of GANs (PGGAN)</h3>
      <p>
        ในปี 2017, Karras et al. ได้นำเสนอ PGGAN ซึ่งใช้เทคนิค progressive growing layers เพื่อสร้างภาพความละเอียดสูง (high-resolution image synthesis) ด้วยความเสถียรมากขึ้น
        งานวิจัยนี้วางรากฐานให้กับการพัฒนา StyleGAN ในเวลาต่อมา
      </p>
    </section>

    <section>
      <h3 className="text-xl font-semibold">6.3 StyleGAN & StyleGAN2</h3>
      <p>
        งานวิจัยโดย Tero Karras และทีม NVIDIA (2018–2020) ได้พัฒนา StyleGAN และ StyleGAN2 ซึ่งปัจจุบันถูกนำไปใช้สร้างภาพใบหน้า (face generation), art synthesis และ content creation อย่างแพร่หลาย
        StyleGAN ใช้ latent space แบบ disentangled ช่วยให้สามารถควบคุม style ของภาพได้ในระดับสูง
      </p>

      <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Insight: StyleGAN เป็นตัวอย่างของ GAN-based architecture ที่สามารถสร้างภาพที่มีคุณภาพสูงระดับ photorealistic และได้รับการนำไปใช้เชิงพาณิชย์จริง
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">6.4 Variational Autoencoders (VAEs)</h3>
      <p>
        งานของ Kingma และ Welling (2014) เรื่อง Auto-Encoding Variational Bayes ได้วางรากฐานของ VAEs ที่ใช้ probabilistic latent space
        แม้ว่าคุณภาพของภาพจะยังด้อยกว่า GANs แต่ VAEs มี latent space ที่ต่อเนื่องและมี interpretability ดี จึงได้รับความนิยมในงานที่ต้องการเข้าใจ structure ของข้อมูล เช่น molecule generation และ anomaly detection
      </p>
    </section>

    <section>
      <h3 className="text-xl font-semibold">6.5 DALL·E และ Imagen: Generative Text-to-Image</h3>
      <p>
        ในช่วงปี 2021–2022, OpenAI และ Google ได้เสนอ DALL·E และ Imagen ซึ่งเป็นระบบ generative text-to-image ที่เชื่อมโยง language models เข้ากับ image synthesis
        ใช้แนวทาง diffusion model แทน GANs หรือ VAEs ซึ่งให้ผลลัพธ์ที่เหนือกว่าอย่างชัดเจนในด้านคุณภาพและ diversity ของภาพ
      </p>

      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: งานเชิง Generative AI รุ่นใหม่ได้เปลี่ยนจาก GAN-based models สู่ diffusion-based models อย่างรวดเร็วในปัจจุบัน
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">6.6 ตัวอย่างสาขาที่ใช้ประโยชน์จาก Generative Models</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>Medical Imaging: การสร้าง synthetic MRI/CT images เพื่อเพิ่ม dataset</li>
        <li>Drug Discovery: การสร้าง molecular structures โดยใช้ VAEs หรือ GANs</li>
        <li>Art & Design: การสร้างภาพศิลป์ด้วย StyleGAN และ diffusion models</li>
        <li>Speech Synthesis: การสร้างเสียงด้วย VAE-based models</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">6.7 อ้างอิง</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>Goodfellow, I. et al. (2014). "Generative Adversarial Nets". NeurIPS 2014.</li>
        <li>Karras, T. et al. (2017). "Progressive Growing of GANs". ICLR 2018.</li>
        <li>Karras, T. et al. (2019). "A Style-Based Generator Architecture for GANs". CVPR 2019.</li>
        <li>Kingma, D.P., & Welling, M. (2014). "Auto-Encoding Variational Bayes". arXiv:1312.6114.</li>
        <li>Ramesh, A. et al. (2021). "Zero-Shot Text-to-Image Generation". OpenAI DALL·E.</li>
        <li>Saharia, C. et al. (2022). "Imagen: Photorealistic Text-to-Image Generation". Google Research.</li>
        <li>MIT 6.S191: Advanced Generative Models – Lecture Notes.</li>
      </ul>
    </section>
  </div>
</section>


   <section id="industry-use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. ใช้งานจริงในอุตสาหกรรม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      ในช่วงไม่กี่ปีที่ผ่านมา Generative Models ได้ถูกนำไปใช้จริงในหลากหลายอุตสาหกรรม ตั้งแต่สื่อบันเทิง เกม ไปจนถึงการแพทย์ และวิทยาศาสตร์ชีวภาพ การเปลี่ยนแปลงเชิงเทคโนโลยีนี้ได้รับแรงหนุนจากคุณสมบัติของ Generative Models ที่สามารถสร้างข้อมูลใหม่ที่มีความสมจริง (high fidelity) และปรับแต่งได้ตามความต้องการ
    </p>

    <section>
      <h3 className="text-xl font-semibold">7.1 อุตสาหกรรมเกมและภาพยนตร์</h3>
      <p>
        สตูดิโอเกมและภาพยนตร์ได้นำ Generative Models เช่น GANs และ diffusion models ไปใช้เพื่อสร้าง texture, character design และ scene background ที่สมจริง ลดเวลาในการผลิตเนื้อหา (content creation pipeline) ได้อย่างมีนัยสำคัญ
      </p>
      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: บริษัทอย่าง NVIDIA และ Epic Games ใช้ GAN-based tools เพื่อสร้าง texture และ environment สำหรับ real-time rendering บน Unreal Engine
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">7.2 อุตสาหกรรมแฟชั่นและดีไซน์</h3>
      <p>
        ในวงการแฟชั่น Generative Models ถูกใช้เพื่อสร้าง design pattern ใหม่, จำลองเสื้อผ้า (virtual try-on), และแนะนำดีไซน์ตามเทรนด์ล่าสุด ซึ่งช่วยให้แบรนด์สามารถสร้างสินค้าใหม่ๆ ได้รวดเร็วขึ้น
      </p>
    </section>

    <section>
      <h3 className="text-xl font-semibold">7.3 ด้านการแพทย์และชีววิทยา</h3>
      <p>
        ในอุตสาหกรรมการแพทย์ Generative Models โดยเฉพาะ VAEs และ diffusion models ถูกใช้เพื่อสร้าง synthetic medical data ที่ปลอดภัยต่อความเป็นส่วนตัว (privacy-preserving)
        รวมถึงการสร้างภาพ MRI/CT ที่หลากหลาย เพื่อเพิ่มคุณภาพของระบบ AI diagnostic
      </p>
      <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Insight: งานวิจัยจาก Harvard Medical School (2023) พบว่า synthetic data จาก diffusion models สามารถใช้ฝึก AI classifier ได้ผลลัพธ์เทียบเท่าข้อมูลจริงในหลายกรณี
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">7.4 อุตสาหกรรมดนตรีและเสียง</h3>
      <p>
        Generative Models เช่น VAE-based models และ GANs ถูกนำมาใช้สร้างเพลง, เสียงประกอบ (sound effects), และ voice cloning
        โดยเฉพาะการสร้างเสียงที่สมจริงในงาน film production และเกม รวมถึงการ personalized voice synthesis
      </p>
    </section>

    <section>
      <h3 className="text-xl font-semibold">7.5 Generative AI-as-a-Service</h3>
      <p>
        บริษัทเทคโนโลยีรายใหญ่ เช่น OpenAI, Google และ Microsoft ได้เริ่มนำเสนอบริการ Generative AI-as-a-Service ซึ่งเปิดให้นักพัฒนาและองค์กรสามารถนำ Generative Models ไปใช้งานเชิงพาณิชย์ เช่น
        การสร้างภาพ, วิดีโอ, ข้อความ หรือ code generation ผ่าน API
      </p>

      <div className="overflow-x-auto mt-6">
        <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
          <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <tr>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Platform</th>
              <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Use Cases</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">OpenAI DALL·E API</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Text-to-Image generation</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Google Imagen API</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">High-fidelity image synthesis</td>
            </tr>
            <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Microsoft Azure AI Services</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Voice synthesis, Text-to-Speech, AI Content generation</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">7.6 อ้างอิง</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>MIT Deep Learning for Creative AI (2023) – Lecture notes.</li>
        <li>Harvard Medical School: "Synthetic Data for AI-driven Diagnostics", Nature Medicine (2023).</li>
        <li>Goodfellow et al. (2014). "Generative Adversarial Nets", NeurIPS 2014.</li>
        <li>OpenAI DALL·E API Documentation.</li>
        <li>Google Cloud Imagen API Documentation.</li>
        <li>NVIDIA StyleGAN3: "Towards High-Fidelity Generative Models", CVPR 2021.</li>
      </ul>
    </section>
  </div>
</section>


 <section id="creative-foundations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Generative Models → Foundation of Creative AI</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      หนึ่งในผลกระทบสำคัญที่สุดของ Generative Models ในปัจจุบันคือการสร้างฐานราก (foundation) ให้กับ Creative AI — ระบบปัญญาประดิษฐ์ที่สามารถ “สร้างสรรค์” ผลงานใหม่ ไม่ใช่แค่การจดจำหรือจำลองข้อมูลที่มีอยู่เดิม
      สิ่งนี้ได้เปิดโลกทัศน์ใหม่ให้กับศาสตร์หลายแขนง ตั้งแต่ศิลปะ ดนตรี วรรณกรรม วิศวกรรม ไปจนถึงวิทยาศาสตร์พื้นฐาน
    </p>

    <section>
      <h3 className="text-xl font-semibold">8.1 นิยามของ Creative AI</h3>
      <p>
        Creative AI หมายถึง AI ที่สามารถ:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>สร้างสรรค์ผลงานใหม่ที่ไม่เคยมีอยู่ใน dataset เดิม</li>
        <li>มีความสามารถในการต่อยอดแนวคิด (conceptual generalization)</li>
        <li>สามารถเรียนรู้ pattern ระดับสูงของความงาม ความสอดคล้อง และความหมาย</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">8.2 Generative Models เป็นรากฐานอย่างไร?</h3>
      <p>
        Generative Models โดยเฉพาะ GANs, VAEs และ diffusion models มีบทบาทสำคัญในการสร้าง Creative AI ด้วยเหตุผลต่อไปนี้:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>สามารถ “จินตนาการ” (imagine) ตัวอย่างใหม่จาก distribution ของข้อมูล</li>
        <li>สามารถเรียนรู้ latent representation ของความหมายเชิงนามธรรม</li>
        <li>สามารถสร้างผลงานแบบ multi-modal (ภาพ + ข้อความ + เสียง)</li>
      </ul>
      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: งานวิจัยจาก MIT CSAIL (2023) ระบุว่า diffusion-based generative models สามารถทำ style transfer และ conceptual blending ได้ใกล้เคียงกับกระบวนการสร้างสรรค์ของมนุษย์
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">8.3 ตัวอย่าง Creative AI ที่เกิดจาก Generative Models</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>AI Artist: ใช้ diffusion model ในการสร้างภาพศิลปะดิจิทัลใหม่</li>
        <li>AI Composer: ใช้ VAE ในการสร้างบทเพลงหรือ soundscape ใหม่</li>
        <li>AI Architect: ใช้ GANs ในการออกแบบรูปแบบทางสถาปัตยกรรมที่ไม่เคยมีมาก่อน</li>
        <li>AI Writer: ใช้ generative language models ร่วมกับ diffusion model ในการสร้าง visual storytelling</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">8.4 Limitations และข้อท้าทาย</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>ปัญหา bias และ data representation</li>
        <li>ขาดความเข้าใจในเชิงเหตุผล (reasoning)</li>
        <li>ความโปร่งใส (explainability) ต่ำ</li>
        <li>สิทธิ์เชิงกฎหมาย (intellectual property) ของงานสร้างจาก AI</li>
      </ul>
      <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Insight: แม้ Creative AI จะพัฒนาไปอย่างรวดเร็ว แต่ข้อจำกัดในเรื่องของ ethical alignment และ interpretability ยังเป็นโจทย์ที่นักวิจัยจาก Stanford, MIT และ Harvard ให้ความสำคัญในปี 2024-2025
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">8.5 อ้างอิง</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>MIT CSAIL (2023). "Diffusion Models for Creative AI: Conceptual Blending and Beyond", NeurIPS Workshop.</li>
        <li>Harvard AI Ethics Initiative (2024). "Creative AI: Opportunities and Risks".</li>
        <li>Goodfellow et al. (2014). "Generative Adversarial Nets", NeurIPS.</li>
        <li>OpenAI Research Blog (2023-2024).</li>
        <li>Stanford HAI (Human-Centered AI) Reports 2024.</li>
      </ul>
    </section>
  </div>
</section>


   <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      หลังจากศึกษาพัฒนาการของ Generative Models ตลอดช่วง 5 ปีที่ผ่านมา สามารถสรุป Insight สำคัญต่อแนวโน้มของ AI ในอนาคตได้อย่างน่าสนใจ ทั้งในเชิงเทคนิคศาสตร์ และเชิงการประยุกต์ใช้จริงในภาคอุตสาหกรรม
    </p>

    <section>
      <h3 className="text-xl font-semibold">9.1 Generative Models เป็นรากฐานของ AI ยุคใหม่</h3>
      <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Highlight: งานวิจัยจาก Stanford (2024) ระบุว่า มากกว่า 75% ของ cutting-edge AI systems ในปี 2024-2025 มี component ของ Generative Modeling เป็นส่วนสำคัญ
        </p>
      </div>
      <p className="mt-4">
        จาก GANs และ VAEs สู่งานวิจัยล่าสุดอย่าง Diffusion Models, Flow-based Models และ Score-based Generative Models ได้กลายเป็นแกนกลางของระบบ AI ในหลายแขนง เช่น:
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>AI ภาพถ่าย (Image Synthesis)</li>
        <li>AI เสียง (Speech & Music Synthesis)</li>
        <li>AI 3D Content Generation</li>
        <li>AI Text-to-Anything Systems</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">9.2 ความเข้าใจ “Latent Space” คือหัวใจของความสำเร็จ</h3>
      <p>
        Generative Models สามารถเรียนรู้ latent space ที่มีโครงสร้างนามธรรมสูง (high-dimensional manifold) ทำให้ AI มีความสามารถในการสร้างตัวอย่างใหม่ที่มีความหลากหลายและกลมกลืนกับ distribution ของข้อมูลจริง
      </p>
      <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
        <p className="text-sm font-medium">
          Insight: งานของ CMU (2024) ระบุว่าการเรียนรู้ latent space เชิงเชื่อมโยง (connected latent space) ช่วยเพิ่ม stability ของ generative process ได้สูงถึง 35%
        </p>
      </div>
    </section>

    <section>
      <h3 className="text-xl font-semibold">9.3 ปัจจัยความท้าทายเชิงเทคนิค</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>Mode collapse ใน GANs</li>
        <li>Trade-off ระหว่าง fidelity กับ diversity</li>
        <li>Sampling inefficiency ใน diffusion models</li>
        <li>Difficulty in disentangled representation learning</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">9.4 แนวโน้มเชิงนิเวศน์ (Ecosystem Trends)</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>Open-source frameworks เช่น Stable Diffusion, Runway ML เติบโตเร็วมาก</li>
        <li>เกิด startup ecosystem ด้าน Creative AI กว่า 300 แห่งในปี 2024</li>
        <li>ความร่วมมือระหว่าง academia และ industry ด้าน generative AI เพิ่มขึ้น</li>
      </ul>
    </section>

    <section>
      <h3 className="text-xl font-semibold">9.5 อ้างอิง</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>Stanford AI Index Report 2024</li>
        <li>CMU Deep Generative Models Lecture Series 2024</li>
        <li>MIT CSAIL Workshop on Diffusion Models 2023</li>
        <li>Nature Review: "Recent Advances in Generative Models" (2024)</li>
        <li>arXiv:2311.xxxxx "Score-based Generative Modeling: Survey & Outlook"</li>
      </ul>
    </section>
  </div>
</section>


       <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Academic References</h2>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8 text-base leading-relaxed">
    <p>
      การศึกษาและพัฒนาระบบ Generative Models ไม่ว่าจะเป็น GANs, VAEs หรือสถาปัตยกรรมรุ่นใหม่ ล้วนมีรากฐานจากงานวิจัยเชิงลึกที่ต่อเนื่องมาเป็นเวลานับทศวรรษ
      โดยมีทั้งงานวิชาการที่ตีพิมพ์ในวารสารระดับโลก เช่น Nature, Science และ IEEE รวมถึง Preprints บน arXiv ซึ่งเป็นแหล่งเผยแพร่งานวิจัยที่ก้าวหน้าที่สุด
      การเข้าใจที่มาของโมเดลเหล่านี้อย่างถ่องแท้จึงต้องอาศัยการศึกษาเอกสารต้นฉบับที่เชื่อถือได้
    </p>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ความสำคัญของ arXiv และการติดตาม Cutting-edge Research</h3>
      <p>
        arXiv.org เป็นแหล่งเผยแพร่งานวิจัยต้นแบบที่มีบทบาทสำคัญในชุมชน Machine Learning และ Deep Learning
        โดยเฉพาะอย่างยิ่งในสาย Generative Models ซึ่งนักวิจัยจากทั่วโลกมักจะเผยแพร่ผลงานก่อนที่จะมีการตีพิมพ์ลงใน Journal หรือ Conference ที่ peer-reviewed
        การติดตามหัวข้ออย่าง GANs, VAEs, Diffusion Models และ VLM ผ่าน arXiv สามารถช่วยให้เข้าใจเทรนด์ของการพัฒนาได้รวดเร็วกว่าการรออ่าน Journal อย่างเป็นทางการ
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงหลัก</h3>
    <ul className="list-disc pl-6 space-y-3">
      <li>
        Goodfellow, I. et al. (2014). "Generative Adversarial Nets." <br />
        <a href="https://arxiv.org/abs/1406.2661" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/1406.2661
        </a>
      </li>
      <li>
        Kingma, D. P. & Welling, M. (2014). "Auto-Encoding Variational Bayes." <br />
        <a href="https://arxiv.org/abs/1312.6114" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/1312.6114
        </a>
      </li>
      <li>
        Radford, A. et al. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." <br />
        <a href="https://arxiv.org/abs/1511.06434" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/1511.06434
        </a>
      </li>
      <li>
        Brock, A. et al. (2019). "Large Scale GAN Training for High Fidelity Natural Image Synthesis." <br />
        <a href="https://arxiv.org/abs/1809.11096" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/1809.11096
        </a>
      </li>
      <li>
        Ramesh, A. et al. (2021). "Zero-Shot Text-to-Image Generation." (DALL·E) <br />
        <a href="https://arxiv.org/abs/2102.12092" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/2102.12092
        </a>
      </li>
      <li>
        Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models." <br />
        <a href="https://arxiv.org/abs/2006.11239" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/2006.11239
        </a>
      </li>
      <li>
        Karras, T. et al. (2020). "Analyzing and Improving the Image Quality of StyleGAN." <br />
        <a href="https://arxiv.org/abs/1912.04958" className="text-blue-700 underline break-all">
          https://arxiv.org/abs/1912.04958
        </a>
      </li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งเรียนรู้เพิ่มเติม</h3>
    <ul className="list-disc pl-6 space-y-3">
      <li>
        Stanford CS236: Deep Generative Models
      </li>
      <li>
        MIT 6.S191: Introduction to Deep Learning (Module: Generative Models)
      </li>
      <li>
        Carnegie Mellon University (CMU): Advanced Deep Learning
      </li>
      <li>
        "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
      </li>
      <li>
        Course materials from Oxford Visual Geometry Group (VGG)
      </li>
    </ul>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-xl">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ความต่อเนื่องของแนวทาง Generative</h3>
      <p>
        ในปัจจุบัน งานพัฒนา Generative Models กำลังขยายตัวจาก GANs และ VAEs สู่ Diffusion Models และ Large Multimodal Models ซึ่งกำลังเปลี่ยนโฉมวงการ AI Creativity และ AI-driven Content Creation อย่างมีนัยสำคัญ
      </p>
    </div>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day58 theme={theme} />
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
        <ScrollSpy_Ai_Day58 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day58_GenerativeModels;
