"use client";
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day60 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day60";
import MiniQuiz_Day60 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day60";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day60_StableDiffusion = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day60_1").format("auto").quality("auto").resize(scale().width(600));
  const img2 = cld.image("Day60_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day60_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day60_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day60_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day60_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day60_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day60_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day60_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day60_10").format("auto").quality("auto").resize(scale().width(500));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 60: Stable Diffusion & Text-to-Image Generation</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

   <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: Text-to-Image Generation คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      Text-to-Image Generation คือขอบเขตหนึ่งของงานวิจัยในสาขา Deep Generative Models ที่มีเป้าหมายสร้างภาพ (Image) ที่มีความสมจริงและสอดคล้องกับข้อความที่เป็นตัวบรรยาย (Text Prompt) ที่ให้ไว้ โดยอาศัยกระบวนการเรียนรู้จากชุดข้อมูลที่จับคู่ระหว่างข้อความและภาพ ในรูปแบบ <code>(text, image)</code> pair ทำให้โมเดลสามารถเรียนรู้การเชื่อมโยงระหว่างความหมายของภาษาและองค์ประกอบในภาพ
    </p>

    <p>
      โมเดลประเภทนี้จัดอยู่ในกลุ่ม <strong>Cross-Modal Generative Learning</strong> ซึ่งครอบคลุมการนำ Modal หนึ่ง (Text) ไปใช้สร้างอีก Modal หนึ่ง (Image) ได้ โดยไม่จำกัดเพียงการจำแนกหรือเข้าใจภาพเท่านั้น แต่เป็นการ <strong>สร้างเนื้อหาใหม่</strong> ที่ไม่เคยมีมาก่อน
    </p>

    <div className="bg-blue-500 p-4 rounded-lg border border-blue-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        จุดเด่นของ Text-to-Image Generation อยู่ที่ความสามารถในการ “สังเคราะห์ภาพใหม่” (Novel Image Synthesis) บนเงื่อนไขของภาษา ซึ่งเปิดประตูสู่งานสร้างสรรค์ในสาขาต่าง ๆ เช่น ศิลปะ, สื่อโฆษณา, อุตสาหกรรมภาพยนตร์, เกม, Virtual World รวมถึงการออกแบบผลิตภัณฑ์เชิงอุตสาหกรรม
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">วิวัฒนาการของ Text-to-Image Generation</h3>
    <p>
      งานวิจัยด้านนี้มีพัฒนาการต่อเนื่องตั้งแต่ปี 2016 จากโมเดล GANs (เช่น StackGAN, AttnGAN) มาจนถึง Diffusion Models ในปัจจุบัน โดยในช่วงแรก ๆ มีความยากในการจับคู่ความหมายของภาษาและรายละเอียดภาพ แต่ด้วยความก้าวหน้าของโมเดลขนาดใหญ่ (Foundation Models) และเทคนิคใหม่ ๆ เช่น Latent Diffusion ทำให้ปัจจุบันสามารถสร้างภาพที่มีคุณภาพสูงอย่างน่าทึ่งได้
    </p>

    <h3 className="text-xl font-semibold mt-10">ความสำคัญทางวิชาการ</h3>
    <p>
      จากมุมมองของมหาวิทยาลัยชั้นนำ เช่น Stanford และ MIT, งาน Text-to-Image Generation ถือเป็นการผสานระหว่างความก้าวหน้าใน <strong>Vision-Language Pretraining</strong> กับ <strong>Probabilistic Generative Models</strong> และถือเป็นแกนสำคัญใน AI Creative Applications ในอนาคต
    </p>

    <h3 className="text-xl font-semibold mt-10">เป้าหมายหลักของระบบ</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>สร้างภาพใหม่ที่ตรงตามความหมายของข้อความ</li>
      <li>สามารถควบคุมเนื้อหาของภาพด้วยระดับภาษา (Prompt Engineering)</li>
      <li>รองรับความหลากหลายและความคิดสร้างสรรค์</li>
      <li>มีความสามารถทั่วไปที่ถ่ายโอนไปใช้กับหลายโดเมน</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ความท้าทายหลัก</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>การเข้าใจความซับซ้อนเชิงบริบทของภาษา</li>
      <li>การสังเคราะห์ภาพที่มีความต่อเนื่องทางโครงสร้าง</li>
      <li>การจัดการ Bias ในข้อมูลต้นทาง</li>
      <li>การควบคุมระดับ fine-detail ของภาพ</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        งาน Stable Diffusion (2022) เป็นตัวอย่างที่ทำให้ Text-to-Image Generation เข้าสู่ระดับการใช้งานจริง และส่งผลกระทบเชิงพาณิชย์อย่างแพร่หลาย โดยมีจุดเด่นที่ประสิทธิภาพ ความคุ้มค่าเชิงคำนวณ และการเปิดโอเพ่นซอร์สที่ช่วยเร่งนวัตกรรมในวงการ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">การเชื่อมโยงกับ Foundation Models</h3>
    <p>
      Text-to-Image Generation นับเป็นกรณีตัวอย่างของ Foundation Models ที่สามารถประยุกต์ข้าม Modal ได้จริง โดยโมเดลขนาดใหญ่ที่ฝึกบน Vision-Language Datasets ขนาดมหาศาล (เช่น LAION, COCO Captions) ช่วยให้เกิดความเข้าใจแบบ <em>semantic alignment</em> ระหว่างภาษาและภาพในระดับที่ละเอียดขึ้น
    </p>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>"High-Resolution Image Synthesis with Latent Diffusion Models", Rombach et al., CVPR 2022</li>
      <li>"Hierarchical Text-Conditional Image Generation with CLIP Latents", arXiv 2022</li>
      <li>MIT Deep Learning for Computer Vision (6.S191), 2023 edition</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
    </ul>

  </div>
</section>


    <section id="evolution" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Evolution ของ Generative Models สำหรับภาพ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      พัฒนาการของ Generative Models สำหรับการสร้างภาพมีจุดเริ่มต้นจากแนวคิดพื้นฐานในเชิงสถิติและความน่าจะเป็น โดยมีเป้าหมายเพื่อให้เครื่องสามารถเรียนรู้ distribution ของข้อมูลภาพและสังเคราะห์ตัวอย่างใหม่ที่มีความสมจริงและหลากหลาย งานวิจัยตั้งแต่ยุคแรกจนถึงปัจจุบันสะท้อนให้เห็นถึงความก้าวหน้าอย่างก้าวกระโดดทั้งด้านคุณภาพของภาพ ความเข้าใจเชิง semantic และความสามารถในการควบคุมเนื้อหา
    </p>

    <h3 className="text-xl font-semibold mt-10">ยุค Early Generative Models</h3>
    <p>
      ก่อนปี 2014 งานวิจัยส่วนใหญ่ใน Generative Models สำหรับภาพเน้นไปที่ <strong>Probabilistic Graphical Models</strong> เช่น Restricted Boltzmann Machines (RBMs), Deep Belief Networks (DBNs) และ Variational Autoencoders (VAEs) ซึ่งสามารถสร้างภาพระดับ low-resolution ได้ แต่ยังขาดความสามารถเชิงโครงสร้างและไม่สามารถควบคุมคุณภาพของภาพได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold mt-10">การมาของ GANs (2014)</h3>
    <p>
      งาน GANs ที่เสนอโดย Ian Goodfellow et al. (2014) ถือเป็นก้าวกระโดดสำคัญ โดยเปลี่ยนกรอบการคิดจาก probabilistic likelihood-based models มาสู่การ <strong>adversarial learning</strong> ที่ใช้เกมเชิงกลยุทธ์ระหว่าง Generator และ Discriminator ส่งผลให้สามารถสร้างภาพที่มีความสมจริงสูงขึ้นอย่างมาก
    </p>
    <div className="bg-blue-500 p-4 rounded-lg border border-blue-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        จุดเปลี่ยนของยุค GANs คือความสามารถในการสร้างภาพความละเอียดสูง (High-Resolution Synthesis) ที่มีรายละเอียดซับซ้อน และกลายเป็นแกนกลางของงานวิจัยหลายแขนง เช่น Super-Resolution, Image-to-Image Translation และ Conditional Generation
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">ยุค Transformer-based Models</h3>
    <p>
      ตั้งแต่ปี 2020 เป็นต้นมา Transformer ได้เริ่มเข้ามามีบทบาทใน Generative Models เช่น DALL·E ของ OpenAI ที่ใช้ Transformer ในการจับคู่ลำดับของ Token จาก Text และ Image ทำให้สามารถควบคุมการสร้างภาพตามความหมายของข้อความได้อย่างยืดหยุ่นและหลากหลาย
    </p>

    <h3 className="text-xl font-semibold mt-10">การถือกำเนิดของ Diffusion Models</h3>
    <p>
      ปี 2021 เป็นยุคเปลี่ยนผ่านอีกครั้งเมื่อ <strong>Diffusion Probabilistic Models</strong> ถูกพัฒนาจนสามารถแซงหน้า GANs ทั้งด้านคุณภาพของภาพและความเสถียรในการฝึก โดยใช้กระบวนการสลาย noise (denoising) แบบ iterative ที่สามารถควบคุมการสังเคราะห์ภาพได้อย่างละเอียด
    </p>
    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Stable Diffusion (2022) เป็นโมเดลที่แสดงให้เห็นว่าด้วยความก้าวหน้าของ Diffusion Models ผสานกับ Latent Space Learning และ Vision-Language Pretraining จะสามารถสร้างภาพคุณภาพสูงในเชิง open domain ได้อย่างมีประสิทธิภาพสูงสุด
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">สรุปวิวัฒนาการในเชิงเปรียบเทียบ</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">ยุค</th>
            <th className="px-4 py-3 border">แนวทางหลัก</th>
            <th className="px-4 py-3 border">ข้อดี</th>
            <th className="px-4 py-3 border">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">ก่อน 2014</td>
            <td className="px-4 py-3 border">RBM, DBN, VAE</td>
            <td className="px-4 py-3 border">แบบจำลองเชิงสถิติ</td>
            <td className="px-4 py-3 border">ภาพไม่สมจริง, low-res</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">2014–2020</td>
            <td className="px-4 py-3 border">GANs</td>
            <td className="px-4 py-3 border">High-Res, realistic image synthesis</td>
            <td className="px-4 py-3 border">Instability ในการฝึก</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">2020–2021</td>
            <td className="px-4 py-3 border">Transformer-based (DALL·E)</td>
            <td className="px-4 py-3 border">Cross-modal flexibility</td>
            <td className="px-4 py-3 border">Resource-heavy</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">2021–ปัจจุบัน</td>
            <td className="px-4 py-3 border">Diffusion Models</td>
            <td className="px-4 py-3 border">ภาพคมชัด เสถียร ควบคุมได้</td>
            <td className="px-4 py-3 border">เวลา inference นาน</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Goodfellow et al., "Generative Adversarial Networks", NeurIPS 2014</li>
      <li>Ramesh et al., "Zero-shot Text-to-Image Generation", DALL·E, arXiv 2021</li>
      <li>Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020</li>
      <li>Rombach et al., "Latent Diffusion Models", CVPR 2022</li>
      <li>Stanford CS236: Deep Generative Models (2023 Edition)</li>
    </ul>

  </div>
</section>


  <section id="diffusion-basics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. แนวคิดพื้นฐานของ Diffusion Models</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      แนวคิดพื้นฐานของ Diffusion Models มีจุดกำเนิดจากงานวิจัยในด้าน <strong>probabilistic generative modeling</strong> ที่พยายามจำลองกระบวนการเปลี่ยนแปลงแบบ stochastic ของข้อมูลจาก distribution ที่ซับซ้อน (complex data distribution) ไปสู่ distribution แบบง่าย (เช่น Gaussian noise) และย้อนกระบวนการนี้กลับมาเพื่อสร้างข้อมูลใหม่
    </p>

    <h3 className="text-xl font-semibold mt-10">Forward Process: การเติม Noise แบบลำดับ</h3>
    <p>
      ใน phase แรก หรือที่เรียกว่า <strong>forward diffusion process</strong> โมเดลจะทำการเติม noise เข้าไปในข้อมูลจริง (เช่น ภาพ) ทีละ step จนกระทั่งข้อมูลถูกแปลงให้มีลักษณะเป็น pure Gaussian noise
    </p>

<div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`x_0 → x_1 → x_2 → ... → x_T ~ N(0, I)`}
    </code>
  </pre>
</div>


    <p>
      โดยกระบวนการนี้สามารถนิยามได้ด้วย Markov chain ที่มี transition kernel แบบ Gaussian ซึ่งทำให้สามารถคำนวณ probability ของทุก step ได้อย่าง explicit
    </p>

    <div className="bg-blue-500 p-4 rounded-lg border border-blue-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ข้อได้เปรียบสำคัญของการใช้ diffusion คือการเรียนรู้ distribution ของข้อมูลอย่างค่อยเป็นค่อยไป (gradual learning) ซึ่งมีความเสถียรมากกว่า adversarial training และสามารถกำหนด likelihood ได้อย่างชัดเจน
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">Reverse Process: การคืนค่าข้อมูล</h3>
    <p>
      กระบวนการที่สองคือ <strong>reverse diffusion process</strong> ซึ่งโมเดลจะเรียนรู้วิธีการลบ noise ทีละ step เพื่อสร้างภาพที่มีความหมายขึ้นมาใหม่ โดยใช้ Neural Network (เช่น U-Net หรือ Transformer) เป็นตัวประมาณค่าพารามิเตอร์ของ distribution ย้อนกลับนี้
    </p>

<div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`x_T → x_{T-1} → ... → x_0 (reconstructed image)`}
    </code>
  </pre>
</div>


    <p>
      โมเดลจะถูกฝึกด้วย objective function เช่น MSE loss ระหว่าง noise ที่เติมจริงกับ noise ที่โมเดลประมาณ ซึ่งทำให้การฝึกมี stability สูง และสามารถสร้างภาพที่มีคุณภาพสูงได้เมื่อทำ sampling ย้อนกลับครบทุก step
    </p>

    <h3 className="text-xl font-semibold mt-10">เปรียบเทียบกับ Generative Approaches อื่น ๆ</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-200 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Approach</th>
            <th className="px-4 py-3 border">ตัวอย่างโมเดล</th>
            <th className="px-4 py-3 border">ข้อดี</th>
            <th className="px-4 py-3 border">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">GANs</td>
            <td className="px-4 py-3 border">StyleGAN, BigGAN</td>
            <td className="px-4 py-3 border">ภาพคม, เร็ว</td>
            <td className="px-4 py-3 border">Mode collapse, training instability</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">VAEs</td>
            <td className="px-4 py-3 border">VAE, β-VAE</td>
            <td className="px-4 py-3 border">Latent control, explicit likelihood</td>
            <td className="px-4 py-3 border">Blurry outputs</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Diffusion Models</td>
            <td className="px-4 py-3 border">DDPM, Stable Diffusion</td>
            <td className="px-4 py-3 border">High quality, stable training</td>
            <td className="px-4 py-3 border">Slow inference</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020</li>
      <li>Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021</li>
      <li>Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022</li>
      <li>Stanford CS236: Deep Generative Models (2023 Edition)</li>
      <li>MIT 6.S191: Introduction to Deep Generative Modeling (2022)</li>
    </ul>

  </div>
</section>


   <section id="stable-diffusion-architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Stable Diffusion: สถาปัตยกรรม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      <strong>Stable Diffusion</strong> เป็นหนึ่งใน generative model ที่ทรงพลังที่สุดในยุคปัจจุบัน โดยพัฒนาบนพื้นฐานของ latent diffusion models (LDM) ซึ่งถูกออกแบบมาเพื่อเพิ่มประสิทธิภาพทั้งด้านคุณภาพและความเร็วในการสร้างภาพ โดยใช้เทคนิคการนำ image space ไป encode อยู่ใน latent space เพื่อลด computational cost อย่างมาก
    </p>

    <h3 className="text-xl font-semibold mt-10">สถาปัตยกรรมหลัก</h3>
    <p>
      สถาปัตยกรรมของ Stable Diffusion ประกอบด้วย 3 องค์ประกอบหลัก ได้แก่:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li><strong>1. Variational Autoencoder (VAE)</strong>: ใช้ในการ encode ภาพจาก pixel space → latent space และ decode กลับเป็นภาพในภายหลัง</li>
      <li><strong>2. U-Net Backbone</strong>: เป็น core neural network ที่ใช้ในการเรียนรู้ reverse diffusion process ใน latent space</li>
      <li><strong>3. Text Encoder (เช่น CLIP Text Encoder)</strong>: ใช้แปลงข้อความ prompt ให้เป็น text embedding เพื่อนำมา conditioning ในการสร้างภาพ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">การทำงานแบบ Latent Diffusion</h3>
    <p>
      จุดแตกต่างสำคัญของ Stable Diffusion จาก diffusion model แบบเดิม คือการใช้ latent space ซึ่งทำให้การประมวลผลมีประสิทธิภาพสูงขึ้นกว่าเดิมอย่างมาก ตัวอย่างเช่นการใช้ latent vector ขนาด 64x64 แทนภาพ 512x512 pixels
    </p>

  <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre-wrap text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`Pixel space → Latent space (VAE) → Diffusion Process (U-Net) → Latent vector → Decoded image`}
    </code>
  </pre>
</div>


    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การใช้ latent space ช่วยลด computational cost ได้อย่างมาก ทำให้สามารถ generate ภาพระดับความละเอียดสูง (512x512 หรือมากกว่า) บน GPU ที่มี memory จำกัด เช่น consumer GPUs
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">Text-to-Image Conditioning</h3>
    <p>
      จุดแข็งของ Stable Diffusion คือความสามารถในการ control output ด้วย text prompt โดยใช้ embedding vector ที่ได้จาก pre-trained CLIP Text Encoder ซึ่ง embedding นี้จะถูก inject เข้าไปใน U-Net ผ่าน cross-attention layer
    </p>

  <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`[Text Prompt] → [Text Embedding] → Cross-attention in U-Net → Image Generation`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold mt-10">Pipeline โดยรวมของ Stable Diffusion</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Stage</th>
            <th className="px-4 py-3 border">Component</th>
            <th className="px-4 py-3 border">Function</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">1</td>
            <td className="px-4 py-3 border">VAE Encoder</td>
            <td className="px-4 py-3 border">แปลงภาพไป latent space</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">2</td>
            <td className="px-4 py-3 border">Noise Scheduler</td>
            <td className="px-4 py-3 border">ควบคุมกระบวนการเติม noise</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">3</td>
            <td className="px-4 py-3 border">U-Net Backbone</td>
            <td className="px-4 py-3 border">เรียนรู้การลด noise แบบ conditioned</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">4</td>
            <td className="px-4 py-3 border">VAE Decoder</td>
            <td className="px-4 py-3 border">แปลง latent กลับมาเป็นภาพ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">ความแตกต่างกับ diffusion model แบบเต็ม</h3>
    <p>
      โดยปกติ diffusion model อย่าง DDPM (Denoising Diffusion Probabilistic Models) ทำงานใน image space ตรง ซึ่งต้องใช้เวลาและ memory มาก ในขณะที่ latent diffusion สามารถเร่งความเร็วขึ้นหลายเท่าตัว และใช้พลังงาน GPU ต่ำลง
    </p>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022</li>
      <li>Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020</li>
      <li>Stanford CS236: Deep Generative Models (2023)</li>
      <li>MIT 6.S191: Introduction to Deep Learning 2023</li>
      <li>CompVis & LAION official Stable Diffusion repositories</li>
    </ul>

  </div>
</section>


<section id="prompt-engineering" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Prompt Engineering</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      การสร้างภาพด้วย generative models แบบ text-to-image เช่น <strong>Stable Diffusion</strong> มีความซับซ้อนในระดับที่ต้องอาศัยความเข้าใจอย่างลึกซึ้งในกระบวนการ <strong>Prompt Engineering</strong> เพื่อควบคุมผลลัพธ์ให้ออกมาตรงตามความต้องการ เพราะตัว model ต้องแปลง embedding ของข้อความมาเป็น visual representation ที่แม่นยำ
    </p>

    <h3 className="text-xl font-semibold mt-10">หลักการของ Prompt Engineering</h3>
    <p>
      Prompt Engineering หมายถึงการออกแบบโครงสร้างของข้อความ (prompt) ที่ป้อนเข้าไปใน model โดยมีเป้าหมายเพื่อควบคุมลักษณะของ output ทั้งด้านเนื้อหา (content), สไตล์ (style), และองค์ประกอบต่าง ๆ ของภาพ
    </p>

    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        แม้ว่า Stable Diffusion จะมี latent capacity สูงมาก แต่ความสามารถของ model ก็ยังคงขึ้นอยู่กับ precision ของ prompt ที่ออกแบบมา ซึ่งสามารถเปลี่ยนผลลัพธ์ได้อย่างมีนัยสำคัญ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">องค์ประกอบของ Prompt ที่มีผลต่อ Output</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Content descriptor</strong>: คำบรรยายเนื้อหาหลัก เช่น "a cat", "mountain landscape"</li>
      <li><strong>Style modifier</strong>: คำบรรยายลักษณะศิลปะ เช่น "digital painting", "photorealistic", "watercolor"</li>
      <li><strong>Artist reference</strong>: การอ้างอิงศิลปิน เช่น "in the style of Hokusai"</li>
      <li><strong>Lighting condition</strong>: คำบรรยายแสง เช่น "golden hour", "studio lighting"</li>
      <li><strong>Detail level</strong>: คำเสริมความละเอียด เช่น "high detail", "8k", "ultra realistic"</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ตัวอย่าง Prompt</h3>

  <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre-wrap text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200 break-words">
    <code>
{`"A majestic lion standing on a cliff during sunset, ultra realistic, 8k, in the style of National Geographic photography"`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold mt-10">ผลกระทบของคำต่าง ๆ ใน Prompt</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">องค์ประกอบ</th>
            <th className="px-4 py-3 border">ตัวอย่างคำ</th>
            <th className="px-4 py-3 border">ผลกระทบต่อ Output</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Content descriptor</td>
            <td className="px-4 py-3 border">"lion", "castle", "robot"</td>
            <td className="px-4 py-3 border">กำหนดวัตถุหลักของภาพ</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Style modifier</td>
            <td className="px-4 py-3 border">"oil painting", "sketch", "anime style"</td>
            <td className="px-4 py-3 border">เปลี่ยนแนวทางทางศิลปะของภาพ</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Lighting</td>
            <td className="px-4 py-3 border">"sunset", "neon lighting"</td>
            <td className="px-4 py-3 border">ควบคุมสภาพแสงและบรรยากาศ</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Detail level</td>
            <td className="px-4 py-3 border">"highly detailed", "4k"</td>
            <td className="px-4 py-3 border">ส่งผลต่อความคมชัดและความซับซ้อนของภาพ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">Negative Prompting</h3>
    <p>
      Stable Diffusion 2.x และเวอร์ชันใหม่ ๆ รองรับการใช้ negative prompt เพื่อบอก model ว่าไม่ต้องการสิ่งใดใน output เช่น:
    </p>

  <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`"A cat, photorealistic, 8k", negative_prompt="blurry, text, watermark"`}
    </code>
  </pre>
</div>

    <h3 className="text-xl font-semibold mt-10">กลยุทธ์ในการออกแบบ Prompt</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>เริ่มจาก prompt ที่ simple และค่อย ๆ เพิ่ม layer ของ detail</li>
      <li>จัดลำดับความสำคัญของคำตามลำดับความต้องการ</li>
      <li>ใช้ keyword ที่ model ผ่านการเทรนมาอย่างแพร่หลาย</li>
      <li>ปรับแต่ง prompt แบบ iterative ตามผลลัพธ์ที่ได้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022</li>
      <li>OpenAI DALL·E 2 research paper</li>
      <li>Google Imagen: Photorealistic Text-to-Image Generation, ICML 2022</li>
      <li>Stanford CS236: Deep Generative Models (2023)</li>
      <li>MIT 6.S191: Introduction to Deep Learning 2023</li>
    </ul>

  </div>
</section>


  <section id="variants" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Variants & Improvements</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      แม้ว่า <strong>Stable Diffusion</strong> รุ่นแรกจะสามารถสร้างภาพคุณภาพสูงได้อย่างน่าทึ่ง แต่การพัฒนาต่อเนื่องในวงการ diffusion models ได้ผลักดันให้เกิดหลากหลาย variants และงานวิจัยที่มุ่งปรับปรุงข้อจำกัดเดิม เพิ่มขีดความสามารถ และยืดหยุ่นต่อ use case ที่หลากหลายมากขึ้น
    </p>

    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การพัฒนาสายพันธุ์ใหม่ของ diffusion models ไม่ได้มีเป้าหมายแค่เพิ่ม resolution แต่ยังเน้นด้าน efficiency, controllability, multi-modal capability และความปลอดภัยทางจริยธรรม
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">Variants สำคัญของ Stable Diffusion</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>Stable Diffusion v1.5</strong>: ปรับปรุง latent space ให้แม่นยำยิ่งขึ้น โดยใช้ checkpoint ที่ fine-tuned กับ aesthetic dataset
      </li>
      <li>
        <strong>Stable Diffusion v2</strong>: ขยาย resolution native เป็น 768x768, ปรับปรุง architecture ของ text encoder เป็น OpenCLIP และฝึกบน dataset ที่สะอาดขึ้น
      </li>
      <li>
        <strong>Stable Diffusion XL (SDXL)</strong>: รองรับ multi-prompt conditioning, มี latent space ที่ซับซ้อนยิ่งขึ้น, สามารถ generate ภาพในระดับ 1024x1024 อย่างมีคุณภาพ
      </li>
      <li>
        <strong>InstructPix2Pix</strong>: ปรับแต่งภาพตามคำสั่งที่เป็นข้อความ
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">เปรียบเทียบ Variants ต่าง ๆ</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Variant</th>
            <th className="px-4 py-3 border">Resolution</th>
            <th className="px-4 py-3 border">Text Encoder</th>
            <th className="px-4 py-3 border">จุดเด่น</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">SD v1.5</td>
            <td className="px-4 py-3 border">512x512</td>
            <td className="px-4 py-3 border">CLIP</td>
            <td className="px-4 py-3 border">ประสิทธิภาพดี, inference เร็ว</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">SD v2.1</td>
            <td className="px-4 py-3 border">768x768</td>
            <td className="px-4 py-3 border">OpenCLIP</td>
            <td className="px-4 py-3 border">ภาพคมขึ้น, dataset สะอาด</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">SDXL</td>
            <td className="px-4 py-3 border">1024x1024</td>
            <td className="px-4 py-3 border">OpenCLIP+</td>
            <td className="px-4 py-3 border">multi-prompt, rich latent space</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">Improvements ที่สำคัญในสายวิจัย</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>Latent Diffusion Refinement:</strong> ลดจำนวน denoising steps โดยใช้ adaptive sampling techniques (เช่น DDIM, DPM++)
      </li>
      <li>
        <strong>Classifier-Free Guidance (CFG):</strong> ปรับ trade-off ระหว่าง faithfulness กับ creativity
      </li>
      <li>
        <strong>Noise Schedules:</strong> การออกแบบ noise schedule ใหม่เพื่อให้ model สามารถเรียนรู้ latent space ได้ลึกขึ้น
      </li>
      <li>
        <strong>Controllability:</strong> งานวิจัยเช่น T2I-Adapter เพิ่ม conditioning module ให้ควบคุมองค์ประกอบเชิงโครงสร้าง เช่น pose, depth
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ตัวอย่างโค้ด: การใช้งาน CFG</h3>

   <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe.to("cuda")

prompt = "a futuristic cityscape at night, cyberpunk, ultra-detailed"
negative_prompt = "blurry, low quality, text"

image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]
image.save("output.png")`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022</li>
      <li>Stability AI Stable Diffusion XL Technical Report</li>
      <li>Stanford CS236: Deep Generative Models (2023)</li>
      <li>OpenAI "Guided Diffusion Models", ICML 2022</li>
      <li>arXiv:2305.08891 "T2I-Adapter: Learning Adapters to Align Text-to-Image Diffusion Models"</li>
    </ul>

  </div>
</section>


<section id="real-world-applications" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Applications ในโลกจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      ตั้งแต่การเปิดตัวของ <strong>Stable Diffusion</strong> จนถึงปัจจุบัน ระบบ <strong>Text-to-Image Generation</strong> ได้ถูกนำไปใช้งานจริงในหลากหลายอุตสาหกรรม ทั้งเชิงสร้างสรรค์, ธุรกิจ, การแพทย์ และอุตสาหกรรมบันเทิง โดยไม่ใช่เพียงแค่ "การสร้างภาพใหม่" เท่านั้น แต่ยังถูกนำไปใช้ในการเร่งการออกแบบ, automation pipeline, และ augmentation ข้อมูล
    </p>

    <div className="bg-blue-500 p-4 rounded-lg border border-blue-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ความสามารถของ diffusion models ในการสร้างภาพระดับ photorealistic จากข้อความเพียงอย่างเดียว ได้เปลี่ยนแปลงวงการ AI image generation อย่างสิ้นเชิง และเริ่มกลายเป็นส่วนหนึ่งของ production pipeline ในหลากหลายอุตสาหกรรม
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1️⃣ การออกแบบเชิงสร้างสรรค์ (Creative Design)</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>ผลิตภาพ artwork, concept art สำหรับเกมและภาพยนตร์</li>
      <li>สร้าง mockup ของสินค้าใหม่ เช่น เสื้อผ้า, เฟอร์นิเจอร์</li>
      <li>ใช้ในการพัฒนา <em>visual language</em> สำหรับ brand identity</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2️⃣ อุตสาหกรรมโฆษณาและสื่อ</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>สร้างภาพโฆษณาเฉพาะกลุ่มเป้าหมาย (personalized ad images)</li>
      <li>เร่ง production ของ visual content สำหรับ marketing campaign</li>
      <li>เพิ่มความหลากหลายของภาพสำหรับ social media content</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">3️⃣ การแพทย์ และ Bio-imaging</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>สร้างภาพ synthetic ของโครงสร้างเซลล์ / เนื้อเยื่อ เพื่อ augment dataset</li>
      <li>ทำ image-to-image translation ระหว่าง modality (CT → MRI)</li>
      <li>การสร้างตัวอย่างข้อมูลที่หายากเพื่อช่วยฝึกโมเดลทางการแพทย์</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">4️⃣ E-commerce & Fashion</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>สร้างภาพตัวอย่างเสื้อผ้าใหม่โดยใช้ <em>prompt-driven fashion generation</em></li>
      <li>สร้างภาพ “try-on” เสมือน (virtual try-on)</li>
      <li>สร้างภาพ product packshot โดยไม่ต้องถ่ายจริง</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">5️⃣ Gaming & Metaverse</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>สร้าง texture, background, environment assets จากคำบรรยาย</li>
      <li>สร้าง avatar, props, world items สำหรับ metaverse และเกม</li>
      <li>เร่ง pipeline ของ game design โดยไม่ต้องสร้าง manual asset ทีละชิ้น</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ตาราง: การประยุกต์ใช้งานในอุตสาหกรรมหลัก</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Industry</th>
            <th className="px-4 py-3 border">Use Case</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Media & Entertainment</td>
            <td className="px-4 py-3 border">Concept art, film pre-visualization</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Fashion & Retail</td>
            <td className="px-4 py-3 border">Product visualization, fashion mockup</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Healthcare</td>
            <td className="px-4 py-3 border">Synthetic data generation for training</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Marketing</td>
            <td className="px-4 py-3 border">Personalized advertising content</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Gaming & Metaverse</td>
            <td className="px-4 py-3 border">Asset generation for virtual worlds</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">ตัวอย่างโค้ด: การสร้างภาพด้วย Stable Diffusion XL</h3>

   <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
  <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <code>
{`from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to("cuda")

prompt = "a photorealistic portrait of a futuristic robot barista in a coffee shop, 4K"
image = pipe(prompt, guidance_scale=8.5).images[0]

image.save("robot_barista.png")`}
    </code>
  </pre>
</div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022</li>
      <li>Stability AI Stable Diffusion XL Technical Report</li>
      <li>Stanford CS236: Deep Generative Models (2023)</li>
      <li>Meta AI "Make-A-Scene: Scene-based text-to-image generation", 2022</li>
      <li>arXiv:2212.07412 "T2I-Adapter: Learning Adapters for Conditional Text-to-Image Generation"</li>
    </ul>

  </div>
</section>

   <section id="ethics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Ethical Considerations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      แม้ว่าเทคโนโลยี <strong>Text-to-Image Generation</strong> และ <strong>Stable Diffusion</strong> จะสร้างโอกาสใหม่มหาศาลในเชิงเศรษฐกิจและศิลปะ แต่ก็มีประเด็นทางจริยธรรม (ethics) ที่ต้องถูกพิจารณาอย่างรอบคอบ ทั้งในระดับเทคโนโลยี, สังคม และกฎหมาย
    </p>

    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การพัฒนา Generative AI ในปัจจุบันก้าวหน้าเร็วกว่ากรอบกฎหมายและแนวปฏิบัติทางจริยธรรมที่มีอยู่ในหลายประเทศ ทำให้จำเป็นต้องมีการถกเถียงเชิงสังคมและวิชาการเพื่อกำหนด "ขอบเขต" ของการใช้ที่เหมาะสม
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1️⃣ Copyright & Fair Use</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>การสร้างภาพที่มี style หรือองค์ประกอบใกล้เคียงกับงานศิลปะต้นฉบับ อาจละเมิดลิขสิทธิ์ของศิลปินเดิม</li>
      <li>Dataset ขนาดใหญ่ที่ใช้ train Stable Diffusion มัก scraping จากเว็บโดยไม่ได้รับ consent จากเจ้าของงาน</li>
      <li>ยังไม่มีข้อยุติทางกฎหมายที่ชัดเจนว่าผลงานจาก AI model จัดเป็น “derivative work” หรือไม่</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2️⃣ Bias & Stereotype Reinforcement</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>โมเดลมีแนวโน้มสร้างผลลัพธ์ที่ reinforce stereotype เช่น gender, race, culture</li>
      <li>Bias เหล่านี้สะท้อนจาก dataset ที่ใช้ฝึก ซึ่งอาจไม่เป็นกลางหรือหลากหลาย</li>
      <li>ตัวอย่าง: prompt “CEO” อาจให้ผลลัพธ์เป็นภาพของเพศชายผิวขาวในอัตราสูงกว่าความเป็นจริง</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">3️⃣ Deepfake & Misinformation</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>เทคโนโลยี diffusion สามารถใช้สร้างภาพปลอม (deepfake) ที่สมจริงยิ่งขึ้น</li>
      <li>เสี่ยงต่อการถูกใช้สร้าง misinformation, fake news, หรือเนื้อหาปลอมที่ทำลายความน่าเชื่อถือของข้อมูล</li>
      <li>ยังไม่มีระบบตรวจจับ deepfake ที่มีความแม่นยำในระดับ production</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">4️⃣ Privacy & Consent</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Stable Diffusion สามารถสร้างภาพของบุคคลจริง (celebrity หรือบุคคลทั่วไป) ได้จากการ prompt</li>
      <li>อาจละเมิดสิทธิ privacy ของบุคคล หากสร้างภาพในบริบทที่ sensitive หรือไม่พึงประสงค์</li>
      <li>ยังไม่มีมาตรฐานในระดับนานาชาติในการควบคุมการนำภาพบุคคลมา train โมเดล</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ตาราง: Ethical Concerns & Potential Risks</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">ประเด็นจริยธรรม</th>
            <th className="px-4 py-3 border">ตัวอย่างความเสี่ยง</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Copyright Infringement</td>
            <td className="px-4 py-3 border">สร้างภาพที่คล้ายงานศิลปะต้นฉบับโดยไม่ได้รับอนุญาต</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Bias Reinforcement</td>
            <td className="px-4 py-3 border">สร้างภาพ stereotype ต่อเพศหรือเชื้อชาติ</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Deepfake</td>
            <td className="px-4 py-3 border">ใช้สร้าง fake news หรือภาพบิดเบือน</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Privacy Violation</td>
            <td className="px-4 py-3 border">สร้างภาพบุคคลในบริบทไม่เหมาะสม</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แนวทางการพัฒนาอย่างรับผิดชอบ</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ใช้ curated dataset ที่ตรวจสอบความเป็นกลางและมีความหลากหลาย</li>
      <li>พัฒนา “content moderation tools” สำหรับผู้ใช้</li>
      <li>กำหนด policy การใช้งานที่เข้มงวด เช่น ไม่ใช้ในบริบทที่ละเมิดสิทธิบุคคล</li>
      <li>สนับสนุนการวิจัยเชิงจริยธรรมใน Generative AI (AI ethics research)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>Harvard Kennedy School - AI and Ethics in Generative Models, 2023</li>
      <li>Oxford Internet Institute - The Ethical Risks of AI-Generated Images, 2023</li>
      <li>IEEE Spectrum - “Who Owns AI Art?”, IEEE 2023</li>
      <li>Meta AI - "Responsible AI for Generative Systems", Meta Research Report, 2023</li>
      <li>Stanford HAI Policy Brief - “AI Art, Copyright, and Fair Use”, 2022</li>
    </ul>

  </div>
</section>


   <section id="future-trends" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Future Trends</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <p>
      การพัฒนา <strong>Text-to-Image Generation</strong> โดยเฉพาะเทคโนโลยี <strong>Stable Diffusion</strong> และ <strong>Diffusion Models</strong> กำลังเข้าสู่ยุคใหม่ที่มีความซับซ้อนและความสามารถสูงขึ้นในอัตราเร่ง ภาพรวมของแนวโน้มในอนาคตสะท้อนถึงทิศทางที่นำไปสู่การเปลี่ยนแปลงทั้งในเชิงเทคนิค, สังคม และเศรษฐกิจ
    </p>

    <div className="bg-blue-500 p-4 rounded-lg border border-blue-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        เทคโนโลยี diffusion กำลังก้าวข้ามขอบเขตของ "Text-to-Image" ไปสู่ <strong>Multi-Modal Generation</strong> ที่สามารถสร้างสื่อรูปแบบผสม (Mixed Modality) เช่น วิดีโอ, 3D, เสียง, และอินเทอร์แอคทีฟคอนเทนต์
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1️⃣ Advanced Multi-Modality</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>การวิจัยกำลังก้าวสู่การสร้าง <strong>Text → Video</strong> และ <strong>Text → 3D Object</strong> ที่มี resolution สูงขึ้น</li>
      <li>โมเดลเช่น <strong>Imagen Video</strong>, <strong>Make-A-Video</strong>, <strong>VideoCrafter</strong> เริ่มเปิดทางสู่การสร้าง "Moving Images"</li>
      <li>การเชื่อมโยง Latent Space ของ Visual / Auditory / Language Models</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2️⃣ Real-Time & Interactive Generation</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>แนวโน้มของ Generative AI ที่มี Latency ต่ำ (Low Latency) เพื่อรองรับการสร้างภาพในเชิง real-time</li>
      <li>ใช้ในระบบ <strong>AI Copilot</strong> หรือ <strong>Generative UX/UI</strong> ในแอปพลิเคชัน</li>
      <li>Prompt → Response ที่แสดงผลแบบ Interactive และ On-Demand</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">3️⃣ Personalization & Style Adaptation</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>การฝึก Fine-Tune หรือ LoRA ที่เร็วขึ้นสำหรับ Personal Style</li>
      <li>ผู้ใช้ทั่วไปสามารถสร้างภาพใน style ส่วนตัว โดยใช้ <strong>Personal Dataset</strong> ขนาดเล็ก</li>
      <li>AI Art Generator ในเชิง Consumer-Level Platform</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">4️⃣ Responsible AI & Governance</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>เริ่มมีมาตรฐานใหม่ เช่น <strong>Content Authenticity Initiative</strong> ของ Adobe</li>
      <li>AI Policy ในระดับรัฐบาลสหรัฐและสหภาพยุโรปกำลังร่างกฎหมายควบคุม Generative AI</li>
      <li>การตรวจสอบและ Traceability ของภาพที่ถูกสร้างด้วย AI</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ตาราง: Future Trends Roadmap</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">ทิศทาง</th>
            <th className="px-4 py-3 border">ตัวอย่างเทคโนโลยี</th>
            <th className="px-4 py-3 border">เวลาโดยประมาณ</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Text → Video Generation</td>
            <td className="px-4 py-3 border">Imagen Video, Make-A-Video</td>
            <td className="px-4 py-3 border">2024-2025</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Real-Time Diffusion</td>
            <td className="px-4 py-3 border">Latent Consistency Models, Instant Stable Diffusion</td>
            <td className="px-4 py-3 border">2025+</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Personal AI Art</td>
            <td className="px-4 py-3 border">DreamBooth, LoRA</td>
            <td className="px-4 py-3 border">2024+</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">AI Policy & Governance</td>
            <td className="px-4 py-3 border">EU AI Act, CAI</td>
            <td className="px-4 py-3 border">2024-2026</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>Stanford HAI - "Foundation Models & the Future of AI Creativity", 2023</li>
      <li>MIT CSAIL - "Emerging Trends in Generative AI Research", 2024</li>
      <li>Google DeepMind - "Scaling Diffusion for Multi-Modal AI", 2024</li>
      <li>Oxford Internet Institute - "AI Governance for Next-Gen Models", 2024</li>
      <li>Meta FAIR - "Low-Latency Generative Systems", 2024</li>
    </ul>

  </div>
</section>


       <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight Box</h2>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        เทคโนโลยี <strong>Stable Diffusion</strong> และ <strong>Text-to-Image Generation</strong> ไม่เพียงแต่เปลี่ยนโฉมวงการ "AI Art" เท่านั้น แต่ยังสะท้อนให้เห็นถึงการบรรจบกันของสาขา <strong>Computer Vision</strong>, <strong>Natural Language Processing</strong> และ <strong>Deep Generative Modeling</strong> อย่างลึกซึ้ง
      </p>
      <p className="mt-2">
        ในแง่สังคม แนวโน้มของ "Creative AI" กำลังขยายขอบเขตความเข้าใจต่อบทบาทของ <strong>Human-in-the-Loop</strong> ในกระบวนการสร้างสรรค์ เนื่องจาก AI เปลี่ยนจาก "Tool" สู่ "Collaborative Partner" ของศิลปิน, นักออกแบบ และผู้ใช้งานทั่วไป
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1️⃣ ความหมายเชิงระบบของ AI Art</h3>
    <p>
      เทคโนโลยี <strong>Text-to-Image Diffusion</strong> กำลังเปลี่ยนแนวคิดพื้นฐานของ "ศิลปะ" และ "การออกแบบ" โดยเปิดโอกาสให้ AI สามารถเข้าใจและตีความสัญลักษณ์ทางภาษา สู่การสร้างสรรค์ visual representation ที่มีความหลากหลายสูง
    </p>
    <ul className="list-disc list-inside space-y-2 mt-4">
      <li>เส้นแบ่งระหว่าง "ผู้สร้าง" และ "ผู้กำกับการสร้าง" (Prompt Engineer) เริ่มเลือนราง</li>
      <li>บทบาทของศิลปินจะเปลี่ยนจาก "ผู้วาด" เป็น "ผู้กำหนด narrative และ conceptual guidance"</li>
      <li>การพัฒนา LoRA & Personalization จะผลักดัน AI Art ให้กลายเป็น personalized creative tool</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2️⃣ ความเสี่ยงและโอกาส</h3>
    <div className="bg-blue-500 p-4 rounded-lg border border-blue-300 shadow-sm">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ความเสี่ยงหลักที่ต้องจับตามอง คือ <strong>Ethical Use & Societal Impact</strong> เช่น Deepfake, Misinformation, Visual Manipulation ในสื่อสาธารณะ
      </p>
    </div>

    <ul className="list-disc list-inside space-y-2 mt-4">
      <li>การใช้งานเชิงอุตสาหกรรม: การโฆษณา, เกม, Virtual World Creation</li>
      <li>ความเสี่ยงเชิงข้อมูล: Training Data Bias, Copyright Infringement</li>
      <li>โอกาสใน Creative AI: On-Demand Visual Content, Democratization of Art</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ตาราง: Key Implications ของ Stable Diffusion</h3>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Impact Area</th>
            <th className="px-4 py-3 border">Positive Impact</th>
            <th className="px-4 py-3 border">Risks</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Art & Design</td>
            <td className="px-4 py-3 border">Democratization of creation tools</td>
            <td className="px-4 py-3 border">Loss of traditional artist roles</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Media Industry</td>
            <td className="px-4 py-3 border">Faster content production</td>
            <td className="px-4 py-3 border">Potential for fake news imagery</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Education</td>
            <td className="px-4 py-3 border">Accessible tools for teaching visual arts</td>
            <td className="px-4 py-3 border">Plagiarism risks</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Research</td>
            <td className="px-4 py-3 border">Foundation for Multi-Modal AI</td>
            <td className="px-4 py-3 border">Data privacy issues</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>

    <ul className="list-disc list-inside space-y-2">
      <li>Stanford HAI - "Generative AI and Art", 2024</li>
      <li>MIT CSAIL - "AI Creativity and Visual Synthesis", 2023</li>
      <li>Oxford Internet Institute - "Ethical Futures of Generative Models", 2024</li>
      <li>Meta FAIR - "Societal Risks of AI Image Generation", 2023</li>
      <li>DeepMind - "Towards Responsible AI Art Generation", 2024</li>
    </ul>

  </div>
</section>

 <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Academic References</h2>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">

    <div className="bg-yellow-500 p-4 rounded-lg border border-yellow-300 shadow-sm">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        งานวิจัยในสาขา <strong>Diffusion-based Generative Models</strong> ได้รับความสนใจอย่างล้นหลามในวงการ Deep Learning ตั้งแต่ปี 2021 เป็นต้นมา โดยมีการตีพิมพ์ทั้งในระดับ <strong>Conference ชั้นนำ</strong> อาทิ <strong>NeurIPS, ICML, ICLR, CVPR</strong> รวมถึงวารสารวิชาการอย่าง <strong>Nature Machine Intelligence</strong> และ <strong>Science</strong>
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1️⃣ บทความพื้นฐานของ Diffusion Models</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        Ho, J., Jain, A., & Abbeel, P. (2020). <em>Denoising Diffusion Probabilistic Models</em>. <strong>arXiv preprint arXiv:2006.11239</strong>.
      </li>
      <li>
        Song, Y., & Ermon, S. (2019). <em>Generative Modeling by Estimating Gradients of the Data Distribution</em>. <strong>NeurIPS</strong>.
      </li>
      <li>
        Dhariwal, P., & Nichol, A. (2021). <em>Diffusion Models Beat GANs on Image Synthesis</em>. <strong>NeurIPS</strong>.
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2️⃣ งานวิจัยด้าน Stable Diffusion</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). <em>High-Resolution Image Synthesis with Latent Diffusion Models</em>. <strong>CVPR</strong>.
      </li>
      <li>
        OpenAI (2023). <em>Exploring the capabilities of DALL·E 2</em>. OpenAI Research.
      </li>
      <li>
        Saharia, C. et al. (2022). <em>Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding</em>. Google Research.
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">3️⃣ Conference ที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Neural Information Processing Systems (NeurIPS)</li>
      <li>International Conference on Learning Representations (ICLR)</li>
      <li>International Conference on Machine Learning (ICML)</li>
      <li>Conference on Computer Vision and Pattern Recognition (CVPR)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">4️⃣ วารสารที่มีบทความสำคัญ</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Nature Machine Intelligence</li>
      <li>IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)</li>
      <li>Science Robotics (เมื่อกล่าวถึง Generative Models สำหรับ Robotics)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">5️⃣ ตาราง: ตัวอย่าง Citation ที่ใช้อ้างอิงบ่อย</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Paper</th>
            <th className="px-4 py-3 border">Authors</th>
            <th className="px-4 py-3 border">Citation</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-500">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Denoising Diffusion Probabilistic Models</td>
            <td className="px-4 py-3 border">Ho et al., 2020</td>
            <td className="px-4 py-3 border">~9,000+</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Latent Diffusion Models</td>
            <td className="px-4 py-3 border">Rombach et al., 2022</td>
            <td className="px-4 py-3 border">~5,000+</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Diffusion Models Beat GANs</td>
            <td className="px-4 py-3 border">Dhariwal & Nichol, 2021</td>
            <td className="px-4 py-3 border">~4,000+</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">DALL·E 2</td>
            <td className="px-4 py-3 border">OpenAI, 2023</td>
            <td className="px-4 py-3 border">~3,000+</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">สรุป</h3>
    <p>
      แหล่งข้อมูลวิชาการในหัวข้อ <strong>Text-to-Image Diffusion Models</strong> และ <strong>Stable Diffusion</strong> กำลังเติบโตอย่างรวดเร็วในช่วง 3 ปีที่ผ่านมา โดยมีทั้งสาย <strong>Deep Generative Modeling</strong>, <strong>AI Art</strong>, <strong>Language & Vision Alignment</strong> และ <strong>Ethics in AI</strong> ที่ถูกตีพิมพ์ในระดับโลก
    </p>

  </div>
</section>

          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day60 theme={theme} />
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
        <ScrollSpy_Ai_Day60 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day60_StableDiffusion;
