import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day57 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day57";
import MiniQuiz_Day57 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day57";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day57_MultiModalModels = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day57_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day57_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day57_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day57_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day57_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day57_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day57_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day57_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day57_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day57_10").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}> 
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 57: Multi-Modal Models – CLIP, Flamingo, and the Rise of Unified AI</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

    <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: Multi-Modal Models คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">
    <p>
      โมเดลแบบ Multi-Modal คือโมเดลที่สามารถประมวลผลข้อมูลจากหลาย Modalities พร้อมกัน เช่น ข้อความ, ภาพ, เสียง หรือวิดีโอ โดยมีเป้าหมายเพื่อให้เข้าใจความหมายที่หลากหลายและซับซ้อนจากข้อมูลที่มีหลายรูปแบบได้อย่างลึกซึ้ง
    </p>

    <h3 className="text-xl font-semibold">1.1 ความสำคัญของ Multi-Modal AI</h3>
    <p>
      ในโลกความเป็นจริง มนุษย์ประมวลผลข้อมูลจากหลายแหล่งพร้อมกัน เช่น การดูภาพประกอบพร้อมฟังเสียงอธิบาย แนวคิดของ Multi-Modal AI จึงมีเป้าหมายเพื่อเลียนแบบความสามารถนี้ โดยเฉพาะในการพัฒนาระบบ AI ที่เข้าใจทั้งบริบทภาพและข้อความร่วมกัน เช่น ระบบสรุปภาพอัตโนมัติ หรือ Chatbot ที่สามารถเข้าใจภาพพร้อมตอบคำถามได้
    </p>

    <h3 className="text-xl font-semibold">1.2 ความแตกต่างจากโมเดลแบบ Single-Modality</h3>
    <p>
      โมเดลทั่วไป เช่น CNN หรือ Transformer แบบดั้งเดิม มักได้รับข้อมูลเพียงชนิดเดียว (ภาพหรือข้อความเท่านั้น) ในขณะที่โมเดล Multi-Modal จะต้องมีโครงสร้างที่สามารถเรียนรู้ representation จากข้อมูลหลายประเภทได้พร้อมกัน ซึ่งต้องมีการออกแบบวิธีการ encode ข้อมูลแต่ละประเภทให้เหมาะสม และมีวิธีการผสาน (fusion) ที่มีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">1.3 ตัวอย่างของโมเดล Multi-Modal</h3>
    <ul className="list-disc list-inside">
      <li>CLIP ของ OpenAI – เชื่อมโยงภาพกับข้อความโดยใช้ Contrastive Learning</li>
      <li>Flamingo ของ DeepMind – รองรับ Visual Reasoning แบบ few-shot learning</li>
      <li>BLIP ของ Salesforce – ใช้ dual-stream encoder-decoder ในการประมวลผลข้อมูลแบบ cross-modal</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-400 dark:border-blue-600 p-4 rounded-lg">
      <p className="font-medium text-blue-900 dark:text-blue-200">
        Insight:
      </p>
      <p className="text-blue-800 dark:text-blue-100">
        Multi-Modal Learning ไม่เพียงแต่เพิ่มศักยภาพของโมเดล AI ในการเข้าใจโลกจริงที่ซับซ้อนมากขึ้น แต่ยังช่วยให้โมเดลสามารถทำงานได้ดีกว่าใน task ที่เกี่ยวข้องกับการตีความ context จากหลายมุมมองพร้อมกัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">1.4 ตารางเปรียบเทียบแนวคิดหลัก</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภทโมเดล</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Modalities</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">วิธีเชื่อมโยงข้อมูล</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ภาพ + ข้อความ</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Contrastive Learning</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Flamingo</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ภาพ + ข้อความ + บริบท</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-attention Fusion</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">BLIP</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ภาพ + ข้อความ</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Dual-Stream Transformer</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">1.5 สรุปแนวโน้ม</h3>
    <p>
      แนวทางของ Multi-Modal Learning กำลังกลายเป็นแกนหลักของการพัฒนา AI สมัยใหม่ โดยเฉพาะอย่างยิ่งกับการเกิดขึ้นของ Foundation Models ที่สามารถเรียนรู้จากข้อมูลหลากหลายและนำไปประยุกต์ใช้ได้กับ task ที่หลากหลาย เช่น VLM (Vision-Language Models) และ Generalist AI อย่าง Gemini หรือ GPT-4V ที่สามารถรับ multimodal input ได้ในระดับเดียวกับมนุษย์
    </p>

    <h3 className="text-xl font-semibold">1.6 อ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”, arXiv:2103.00020</li>
      <li>Alayrac et al., “Flamingo: A Visual Language Model for Few-Shot Learning”, DeepMind, 2022</li>
      <li>Jia et al., “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision”, ICML 2021</li>
    </ul>
  </div>
</section>


    <section id="concept" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. แนวคิดเบื้องหลัง Multi-Modal Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">2.1 การบูรณาการระหว่าง Modalities</h3>
    <p>
      Multi-Modal Learning หมายถึงการฝึกโมเดล AI ให้สามารถเรียนรู้จากข้อมูลหลายประเภท (modalities) เช่น ภาพ เสียง และข้อความ โดยมีเป้าหมายเพื่อให้โมเดลสามารถเชื่อมโยงข้อมูลที่แตกต่างกันในลักษณะที่เหมือนการรับรู้ของมนุษย์ ซึ่งสามารถเข้าใจได้ผ่านการรับรู้จากหลายประสาทสัมผัส
    </p>

    <h3 className="text-xl font-semibold">2.2 ประเภทของการผสานข้อมูล</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Early Fusion:</strong> รวมข้อมูลจากหลาย modalities ตั้งแต่ต้น เช่น การรวม pixel กับ embedding ของข้อความ</li>
      <li><strong>Late Fusion:</strong> ประมวลผลแต่ละ modality แยก แล้วรวมผลลัพธ์ที่ปลายทาง</li>
      <li><strong>Joint Representation:</strong> สร้าง embedding ที่รวมคุณลักษณะร่วมของหลาย modalities เช่น ใน CLIP</li>
    </ul>

    <h3 className="text-xl font-semibold">2.3 โมเดลพื้นฐานที่ใช้ในการรวมข้อมูล</h3>
    <p>
      แนวทางทั่วไปสำหรับ Multi-Modal Learning ใช้โครงสร้างโมเดลประเภทต่อไปนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Encoder-Decoder Framework:</strong> เช่น ใช้ CNN เป็น Encoder สำหรับภาพ และ Transformer Decoder สำหรับข้อความ</li>
      <li><strong>Dual Encoder:</strong> ใช้ encoder แยกกันสำหรับแต่ละ modality แล้วรวมกันผ่าน similarity function</li>
      <li><strong>Cross-Attention Mechanism:</strong> ใช้ attention layer เชื่อมโยงระหว่าง feature ของ modal ต่าง ๆ</li>
    </ul>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border dark:border-gray-600">Fusion Type</th>
            <th className="px-4 py-3 border dark:border-gray-600">วิธีการรวม</th>
            <th className="px-4 py-3 border dark:border-gray-600">จุดเด่น</th>
          </tr>
        </thead>
        <tbody className="text-gray-800 dark:text-gray-100">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border dark:border-gray-600">Early Fusion</td>
            <td className="px-4 py-3 border dark:border-gray-600">รวมข้อมูลก่อนการเรียนรู้</td>
            <td className="px-4 py-3 border dark:border-gray-600">ใช้โครงสร้างง่าย เหมาะกับข้อมูลที่มี temporal alignment</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border dark:border-gray-600">Late Fusion</td>
            <td className="px-4 py-3 border dark:border-gray-600">รวมผลลัพธ์สุดท้ายของแต่ละโมเดล</td>
            <td className="px-4 py-3 border dark:border-gray-600">ยืดหยุ่น สามารถใช้โมเดลที่เหมาะกับแต่ละ modality</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border dark:border-gray-600">Joint Representation</td>
            <td className="px-4 py-3 border dark:border-gray-600">เรียนรู้ latent space ร่วมกัน</td>
            <td className="px-4 py-3 border dark:border-gray-600">ใช้ในงาน retrieval, captioning ที่ต้องการ understanding ร่วม</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-500 text-blue-900 dark:text-blue-100 p-4 mt-10 rounded-lg shadow">
      <p className="font-semibold">Insight:</p>
      <p>
        แนวคิดของ Multi-Modal Learning กำลังกลายเป็นรากฐานของ AI ยุคใหม่ โดยเฉพาะในโมเดลขนาดใหญ่ที่ต้องรองรับบริบทจากหลายแหล่งข้อมูลพร้อมกัน เช่น CLIP, Flamingo และ GPT-4 ที่มีความสามารถ multi-modal โดยธรรมชาติ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">2.4 อ้างอิงงานวิจัยที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2018). Multimodal Machine Learning: A Survey. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence</em>.</li>
      <li>Tsai, Y. H. H., et al. (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. <em>ACL</em>.</li>
      <li>Li, J., et al. (2021). Align before Fuse: Vision and Language Representation Learning with Momentum Distillation. <em>NeurIPS</em>.</li>
    </ul>
  </div>
</section>


  <section id="clip" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    3. CLIP (Contrastive Language-Image Pretraining)
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">3.1 แนวคิดพื้นฐานของ CLIP</h3>
    <p>
      CLIP (Contrastive Language-Image Pretraining) เป็นโมเดลจาก OpenAI ที่ใช้แนวคิดการเรียนรู้แบบ contrastive เพื่อนำ representation ของข้อความและภาพเข้าไปอยู่ใน latent space เดียวกัน โดยเรียนรู้จากการจับคู่อธิบายภาพที่สอดคล้องกันในระดับ semantically อย่างแม่นยำ CLIP ใช้ข้อมูลภาพและข้อความจากอินเทอร์เน็ตจำนวนมากแบบ weakly-supervised ทำให้สามารถ generalize ได้ดีกับ downstream tasks โดยไม่ต้อง fine-tune เพิ่มเติม
    </p>

    <h3 className="text-xl font-semibold">3.2 โครงสร้างของโมเดล CLIP</h3>
    <p>
      โครงสร้างของ CLIP ประกอบด้วยโมเดลสองส่วน ได้แก่ Visual Encoder และ Text Encoder โดย Visual Encoder ใช้ ViT หรือ ResNet ขนาดใหญ่ และ Text Encoder ใช้ Transformer ที่รับ tokenized sentence เป็น input หลังจากเข้ารหัสข้อมูลแล้ว ทั้งสองฝั่งจะถูก map ไปยัง latent space เดียวกัน จากนั้นใช้ cosine similarity เพื่อคำนวณว่า image กับ text ใดตรงกันมากที่สุด
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">Component</th>
            <th className="px-4 py-3 border">Architecture</th>
            <th className="px-4 py-3 border">Function</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Visual Encoder</td>
            <td className="px-4 py-3 border">ViT / ResNet</td>
            <td className="px-4 py-3 border">แปลงภาพเป็น embedding</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">Text Encoder</td>
            <td className="px-4 py-3 border">Transformer</td>
            <td className="px-4 py-3 border">แปลงข้อความเป็น embedding</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Contrastive Objective</td>
            <td className="px-4 py-3 border">Cosine Similarity + Softmax</td>
            <td className="px-4 py-3 border">จับคู่ภาพ-ข้อความที่ถูกต้อง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">3.3 จุดแข็งของ CLIP</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถ zero-shot classification ได้โดยไม่ต้อง fine-tune</li>
      <li>ปรับใช้กับ downstream task ได้หลากหลาย</li>
      <li>มี robustness ต่อข้อมูล noisy จากโลกจริง</li>
    </ul>

    <h3 className="text-xl font-semibold">3.4 ข้อจำกัดและความท้าทาย</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ยังมี bias จาก dataset ขนาดใหญ่ที่ไม่ได้คัดกรอง</li>
      <li>การใช้ ViT ขนาดใหญ่ต้องการ compute สูง</li>
      <li>ความสามารถ zero-shot บางกรณียังสู้ fine-tuned model ไม่ได้</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
      <strong className="block font-semibold mb-2">Insight:</strong>
      จุดเด่นของ CLIP คือความสามารถในการประเมินภาพกับข้อความที่ไม่เคยเห็นมาก่อน ซึ่งทำให้มันเป็นฐานสำหรับงาน multimodal รุ่นถัดไป เช่น Flamingo และ DALL·E
    </div>

    <h3 className="text-xl font-semibold mt-10">3.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." arXiv preprint arXiv:2103.00020 (2021).</li>
      <li>Stanford CS231n 2023, Lecture 11: Vision-Language Models.</li>
      <li>MIT 6.S191 Deep Learning for AI, Spring 2022.</li>
    </ul>
  </div>
</section>


   <section id="flamingo" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Flamingo (DeepMind, 2022)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">4.1 ภาพรวมของโมเดล Flamingo</h3>
    <p>
      Flamingo คือโมเดล multi-modal ขนาดใหญ่จาก DeepMind ที่ถูกออกแบบมาเพื่อให้สามารถเรียนรู้จากหลาย modal พร้อมกัน เช่น ภาพและข้อความ โดยไม่ต้อง fine-tune เฉพาะ task ใด ๆ โดยเฉพาะ โมเดลนี้พัฒนาขึ้นในปี 2022 และกลายเป็นอีกหนึ่งหมุดหมายสำคัญของการรวมข้อมูลหลาย modal เข้าด้วยกันภายใน framework เดียว
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
      <strong>Highlight:</strong> Flamingo แสดงให้เห็นว่าโมเดลขนาดใหญ่สามารถ generalize ไปยัง downstream tasks ใหม่ ๆ โดยไม่ต้อง retrain ได้ดีมากผ่านการใช้ <em>few-shot prompting</em>
    </div>

    <h3 className="text-xl font-semibold">4.2 สถาปัตยกรรมของ Flamingo</h3>
    <p>
      Flamingo ผสานโมเดล Vision Transformer (ViT) เข้ากับโมเดล Language Model ขนาดใหญ่ที่ผ่านการ pretrain แล้ว เช่น Chinchilla โดยมีการใช้ Perceiver Resampler เป็นตัวแปลง representation ของภาพให้เหมาะสมกับการป้อนเข้าไปใน language model
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600">องค์ประกอบ</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600">คำอธิบาย</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Vision Encoder (ViT)</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">ใช้ ViT ในการแปลงภาพเป็น token เพื่อป้อนให้กับ Perceiver</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Perceiver Resampler</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">เลือกและแปลง representation ของภาพให้เหมาะกับการต่อกับ language model</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Frozen Language Model</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">ใช้ LM ที่ถูก train ไว้แล้วโดยไม่ update พารามิเตอร์</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">4.3 จุดเด่นของ Flamingo</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถเรียนรู้ได้จากตัวอย่างจำนวนน้อย (few-shot learning) โดยไม่ต้อง fine-tune</li>
      <li>รับรู้บริบทของภาพและข้อความพร้อมกัน</li>
      <li>แยก pipeline ของ vision และ language ได้อย่างยืดหยุ่น</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-800 text-yellow-900 dark:text-yellow-100 p-4 rounded-lg border border-yellow-300 dark:border-yellow-700">
      <strong>Insight:</strong> งานของ Flamingo แสดงให้เห็นถึงศักยภาพของโมเดล multi-modal ที่สามารถ adapt กับ task ใหม่โดยแทบไม่ต้องปรับพารามิเตอร์ — ซึ่งเป็นแนวคิดที่ต่อยอดมาสู่โมเดลอย่าง Gemini และ GPT-4V
    </div>

    <h3 className="text-xl font-semibold">4.4 ผลลัพธ์จากงานวิจัย</h3>
    <p>
      ใน benchmark ต่าง ๆ เช่น OKVQA และ ScienceQA, Flamingo แสดงให้เห็นว่าแม้ไม่มีการ fine-tune แต่ก็สามารถ outperform โมเดลที่ผ่านการฝึกเฉพาะทางได้ในหลายกรณี โดยใช้ตัวอย่างเพียงไม่กี่ตัวอย่างเท่านั้นในระหว่าง prompting
    </p>

    <h3 className="text-xl font-semibold mt-8">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 break-words">
  <li>
    DeepMind Flamingo:{" "}
    <a
      href="https://www.deepmind.com/publications/flamingo-a-visual-language-model-for-few-shot-learning"
      className="text-blue-700 underline break-all"
      target="_blank"
      rel="noopener noreferrer"
    >
      https://www.deepmind.com/publications/flamingo-a-visual-language-model-for-few-shot-learning
    </a>
  </li>
  <li>
    "Perceiver: General Perception with Iterative Attention", arXiv 2021
  </li>
  <li>
    "Scaling Laws for Neural Language Models", Hoffmann et al., 2022 (Chinchilla)
  </li>
</ul>

  </div>
</section>


      <section id="clip-vs-flamingo" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. เปรียบเทียบ CLIP vs Flamingo</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">
    <h3 className="text-xl font-semibold">หลักการทำงานพื้นฐาน</h3>
    <p>
      ทั้ง CLIP และ Flamingo ต่างเป็นโมเดลมัลติโมดัลที่ได้รับการออกแบบมาเพื่อรวมภาพและภาษาธรรมชาติเข้าด้วยกัน แต่ใช้หลักการและกลยุทธ์ที่แตกต่างกันในการเรียนรู้และประมวลผลข้อมูล:
    </p>
    <ul className="list-disc list-inside">
      <li><strong>CLIP:</strong> ใช้ contrastive learning เพื่อจับคู่ภาพและข้อความโดยใช้ cosine similarity</li>
      <li><strong>Flamingo:</strong> ใช้ transformer แบบ decoder-only พร้อม cross-attention สำหรับการรวมข้อมูลข้าม modality</li>
    </ul>

    <h3 className="text-xl font-semibold">โครงสร้างโมเดล</h3>
    <div className="overflow-x-auto">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border">คุณลักษณะ</th>
            <th className="px-4 py-3 border">CLIP</th>
            <th className="px-4 py-3 border">Flamingo</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">Backbone</td>
            <td className="px-4 py-3 border">ViT + Transformer</td>
            <td className="px-4 py-3 border">Perceiver Resampler + Decoder Transformer</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">วิธีการเรียนรู้</td>
            <td className="px-4 py-3 border">Contrastive (image-text pairs)</td>
            <td className="px-4 py-3 border">Few-shot, In-context Learning</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border">การตอบสนอง</td>
            <td className="px-4 py-3 border">Matching (ภาพ ↔ ข้อความ)</td>
            <td className="px-4 py-3 border">Generation (สร้างคำตอบแบบต่อเนื่อง)</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border">ความสามารถด้าน Context</td>
            <td className="px-4 py-3 border">จำกัด</td>
            <td className="px-4 py-3 border">เข้าใจลำดับหลาย modal</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-100 dark:bg-blue-900 dark:text-white p-4 rounded-lg border border-blue-300 dark:border-blue-700">
      <p className="font-semibold">Insight:</p>
      <p>
        CLIP เหมาะกับงานที่ต้องการการจับคู่อย่างรวดเร็วและมีขนาดเล็กกว่า ส่วน Flamingo เหมาะกับงานที่ต้องการการทำ reasoning ที่ลึกซึ้งข้าม modal โดยเฉพาะ task ที่ต้องตอบสนองในบริบทที่ซับซ้อน เช่น VQA และ instruction following
      </p>
    </div>

    <h3 className="text-xl font-semibold">การใช้งานตามบริบท</h3>
    <ul className="list-disc list-inside">
      <li><strong>CLIP:</strong> ใช้งานอย่างแพร่หลายในระบบ recommendation, zero-shot classification, และ image search</li>
      <li><strong>Flamingo:</strong> ใช้ในระบบ AI assistant แบบมัลติโมดัลที่สามารถรับภาพ วิดีโอ และข้อความได้พร้อมกัน เช่น DeepMind’s multimodal assistants</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”, arXiv:2103.00020</li>
      <li>Alayrac et al., “Flamingo: a Visual Language Model for Few-Shot Learning”, DeepMind, 2022, arXiv:2204.14198</li>
      <li>Stanford CS231n Lecture Notes, 2023</li>
    </ul>
  </div>
</section>


 <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. การใช้งานจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      โมเดลแบบ Multi-Modal อย่าง CLIP และ Flamingo ได้รับความสนใจอย่างกว้างขวางในวงการ AI เนื่องจากสามารถประมวลผลและเข้าใจข้อมูลจากหลาย modality ได้อย่างมีประสิทธิภาพ
      การใช้งานจริงของโมเดลเหล่านี้กระจายตัวในหลากหลายอุตสาหกรรม ตั้งแต่งานค้นหาเชิงภาพ (image retrieval), การทำ caption อัตโนมัติ, การวิเคราะห์วิดีโอ, ไปจนถึงการพัฒนา AI ที่สามารถสนทนาแบบเข้าใจโลกจริง
    </p>

    <h3 className="text-xl font-semibold">6.1 ระบบค้นหาข้อมูลที่อิงทั้งภาพและข้อความ</h3>
    <p>
      CLIP ถูกนำมาใช้ในระบบ search engine ที่สามารถค้นหาภาพตามคำอธิบาย หรือค้นหาคำอธิบายจากภาพได้ เช่นในระบบค้นหาภายในฐานข้อมูลสินค้าหรือคลังภาพขนาดใหญ่
      โดยโมเดลสามารถแปลงทั้งภาพและข้อความให้อยู่ใน latent space เดียวกัน ซึ่งช่วยให้การค้นหามีความแม่นยำสูงขึ้น
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 border border-blue-300 dark:border-blue-700 p-4 rounded-lg">
      <p className="font-semibold">Highlight:</p>
      <p>
        การใช้ CLIP ในระบบ search engine ทำให้สามารถค้นหาได้แบบ "zero-shot" โดยไม่ต้อง fine-tune โมเดลเพิ่ม เช่นการพิมพ์คำว่า “a cat wearing sunglasses” แล้วระบบสามารถค้นหาภาพที่ตรงตามนั้นได้ทันที
      </p>
    </div>

    <h3 className="text-xl font-semibold">6.2 AI Assistant ที่สามารถเข้าใจภาพและข้อความร่วมกัน</h3>
    <p>
      Flamingo ซึ่งพัฒนาโดย DeepMind ได้ถูกนำมาใช้ในการสร้าง assistant แบบ Multi-Modal ที่สามารถตอบคำถามเกี่ยวกับเนื้อหาในภาพ เช่น ถามว่า “มีอะไรผิดปกติในภาพนี้” หรือ “คนในภาพกำลังทำอะไรอยู่”
      โดยไม่ต้องสอนโมเดลเพิ่มเติมด้วยตัวอย่างเฉพาะเจาะจง
    </p>

    <h3 className="text-xl font-semibold">6.3 การวิเคราะห์เนื้อหาวิดีโอแบบ multi-modal</h3>
    <p>
      Flamingo สามารถใช้ในการสรุปเนื้อหาวิดีโอ หรืออธิบายภาพเคลื่อนไหว (frame-by-frame reasoning) โดยการนำลำดับของภาพจากวิดีโอเข้าไปในโมเดลร่วมกับ prompt แบบข้อความ
      ทำให้สามารถสร้าง caption, ตอบคำถาม, หรือเข้าใจ context ของเหตุการณ์ในวิดีโอได้อย่างลึกซึ้ง
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-800 border border-yellow-300 dark:border-yellow-700 p-4 rounded-lg">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานของ Flamingo เปิดประตูสู่การใช้ AI เพื่อเข้าใจ “เวลาต่อเนื่อง” ผ่านข้อมูล multimodal ซึ่งอาจมีผลอย่างมากต่อวงการ healthcare, security surveillance, และ content moderation
      </p>
    </div>

    <h3 className="text-xl font-semibold">6.4 การใช้งานในด้าน Healthcare และการแพทย์</h3>
    <p>
      Multi-Modal AI ถูกนำมาใช้ในการจับคู่ระหว่างภาพทางการแพทย์ (เช่น X-ray, MRI) กับบันทึกการวินิจฉัยของแพทย์ เพื่อช่วยตรวจหาโรคหรือสนับสนุนการวินิจฉัย
      งานวิจัยของ Stanford (RadImageNet + text reports) ได้แสดงว่าโมเดลสามารถจับคู่ความสัมพันธ์ระหว่าง lesion บนภาพและคำบรรยายได้แม่นยำขึ้น
    </p>

    <h3 className="text-xl font-semibold">6.5 การเรียนรู้ภาษาด้วยบริบทจากโลกจริง</h3>
    <p>
      โมเดลอย่าง Flamingo สามารถถูกนำมาใช้เป็นระบบช่วยสอนภาษาสำหรับเด็กหรือผู้ใหญ่ โดยสามารถสร้าง prompt จากภาพในโลกจริงเพื่อฝึก vocabulary, grammar, หรือ reasoning ผ่านคำถามที่มีคำตอบหลากหลายรูปแบบ
    </p>

    <h3 className="text-xl font-semibold">6.6 การใช้งานในงานศิลปะและความคิดสร้างสรรค์</h3>
    <p>
      เมื่อรวมกับ generative models อย่าง DALL·E หรือ Stable Diffusion, โมเดล CLIP ถูกใช้เพื่อสร้างงานศิลปะ, ออกแบบ UX/UI, หรือแม้แต่ในเกมที่สร้างโลกตามคำสั่งของผู้เล่นได้ (text-to-scene generation)
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 border border-blue-300 dark:border-blue-700 p-4 rounded-lg">
      <p className="font-semibold">Highlight:</p>
      <p>
        โมเดล Multi-Modal กำลังกลายเป็นแกนกลางของ “Foundation Models” ที่สามารถปรับตัวและ fine-tune ไปสู่ task ใด ๆ ในโลกจริง โดยใช้ข้อมูลไม่จำกัดแค่ภาพหรือข้อความเพียงอย่างเดียว
      </p>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" - arXiv:2103.00020</li>
      <li>Alayrac, J.-B., et al. (2022). "Flamingo: a Visual Language Model for Few-Shot Learning" - DeepMind, arXiv:2204.14198</li>
      <li>Stanford ML Group. (2022). "RadImageNet: An Open Radiologic Dataset" - Stanford Research Archive</li>
      <li>Brown, T.B., et al. (2020). "Language Models are Few-Shot Learners" - arXiv:2005.14165</li>
      <li>OpenAI. (2022). "CLIP + DALL·E: Compositional Visual Creativity" - OpenAI Blog</li>
    </ul>
  </div>
</section>


       <section id="vl-foundations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Vision-Language Foundation Models</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      Vision-Language Foundation Models (VL-FMs) คือกลุ่มของโมเดล AI ที่ได้รับการฝึกฝนล่วงหน้าบนข้อมูลภาพและข้อความขนาดใหญ่อย่างครอบคลุม
      โมเดลเหล่านี้มีศักยภาพในการปรับใช้กับหลากหลายงาน downstream โดยไม่ต้องฝึกฝนใหม่ตั้งแต่ต้น (zero-shot หรือ few-shot learning)
      คล้ายกับแนวคิดของ Foundation Models ใน Natural Language Processing เช่น GPT, T5 หรือ BERT
    </p>

    <h3 className="text-xl font-semibold">7.1 แนวคิดของ Foundation Models</h3>
    <p>
      Foundation Models เป็นโมเดลที่ถูกฝึกแบบ self-supervised บนข้อมูลที่หลากหลายและมหาศาล ด้วยเป้าหมายเพื่อให้โมเดลเข้าใจรูปแบบเชิงโครงสร้างของโลก
      แนวคิดนี้ขยายจากโมเดลภาษาไปสู่ Multi-Modal โดย VL-FMs คือการต่อยอดให้ AI เข้าใจทั้งภาพและภาษาอย่างลึกซึ้งใน latent space เดียวกัน
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 border border-yellow-300 dark:border-yellow-700 p-4 rounded-lg">
      <p className="font-semibold">Insight Box:</p>
      <p>
        Vision-Language Foundation Models เปรียบเสมือน “pre-trained brain” ที่สามารถต่อยอดไปทำ task ใด ๆ โดยใช้การ prompt หรือ fine-tune เพียงเล็กน้อย—ลดการพึ่งพาข้อมูล labeled จำนวนมาก
      </p>
    </div>

    <h3 className="text-xl font-semibold">7.2 ตัวอย่าง Vision-Language Foundation Models</h3>
    <ul className="list-disc list-inside space-y-1">
      <li><strong>CLIP (OpenAI):</strong> จับคู่ภาพและข้อความใน latent space เดียว โดยใช้ contrastive loss</li>
      <li><strong>ALIGN (Google):</strong> ขยายแนวคิดของ CLIP ไปยัง dataset ขนาดใหญ่ขึ้นกว่า 1.8 พันล้านคู่ image-text</li>
      <li><strong>Flamingo (DeepMind):</strong> ออกแบบ decoder-only ที่สามารถทำ visual question answering แบบ few-shot โดยใช้ frozen vision backbone</li>
      <li><strong>BLIP / BLIP-2:</strong> โมเดลที่ผสาน vision encoder กับ language model ผ่าน query transformer เพื่อรองรับ tasks ทั้ง generation และ classification</li>
    </ul>

    <h3 className="text-xl font-semibold">7.3 โครงสร้างพื้นฐานของ VL-FMs</h3>
    <p>
      โดยทั่วไป VL-FMs จะประกอบด้วย component หลัก 2 ส่วน ได้แก่:
    </p>

    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border-b border-gray-300 dark:border-gray-600">Component</th>
            <th className="px-4 py-3 border-b border-gray-300 dark:border-gray-600">Function</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <td className="px-4 py-3 border-b border-gray-300 dark:border-gray-600">Vision Encoder</td>
            <td className="px-4 py-3 border-b border-gray-300 dark:border-gray-600">แปลงภาพให้เป็น representation แบบ vector เช่นผ่าน ResNet, ViT</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100">
            <td className="px-4 py-3 border-b border-gray-300 dark:border-gray-600">Text Encoder / Decoder</td>
            <td className="px-4 py-3 border-b border-gray-300 dark:border-gray-600">สร้าง embedding จากข้อความ หรือใช้เป็น output สำหรับ caption generation</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">7.4 การปรับใช้งานในระบบจริง</h3>
    <p>
      VL-FMs ถูกนำไปใช้ในงานเชิงพาณิชย์อย่างแพร่หลาย เช่น การทำ content moderation บนโซเชียลมีเดีย, ระบบช่วยอ่านภาพในการแพทย์, การช่วยเหลือผู้พิการด้วย AI ที่อธิบายภาพผ่านเสียง และระบบตรวจจับ context ใน smart home
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 border border-blue-300 dark:border-blue-700 p-4 rounded-lg">
      <p className="font-semibold">Highlight:</p>
      <p>
        ความสามารถของ VL-FMs ในการประมวลผล “ภาพ-คำพูด-ความเข้าใจ” อย่างต่อเนื่อง ทำให้โมเดลเหล่านี้กลายเป็นรากฐานของยุคใหม่ของ AGI (Artificial General Intelligence)
      </p>
    </div>

    <h3 className="text-xl font-semibold">7.5 ความสามารถแบบ Zero-shot และ Few-shot</h3>
    <p>
      หนึ่งในจุดเด่นของ VL-FMs คือความสามารถในการทำ task ใหม่โดยไม่ต้อง fine-tune เช่น CLIP สามารถจัดหมวดหมู่ภาพที่ไม่เคยเห็นมาก่อน หรือ Flamingo สามารถตอบคำถามเกี่ยวกับภาพแม้จะเห็นเพียงตัวอย่างเดียว
    </p>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>Bommasani, R., et al. (2021). “On the Opportunities and Risks of Foundation Models” - Stanford CRFM</li>
      <li>Radford, A., et al. (2021). “Learning Transferable Visual Models From Natural Language Supervision” - arXiv:2103.00020</li>
      <li>Jia, C., et al. (2021). “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision” - arXiv:2102.05918</li>
      <li>Alayrac, J.-B., et al. (2022). “Flamingo: A Visual Language Model for Few-Shot Learning” - DeepMind</li>
      <li>Salesforce AI Research. (2022). “BLIP-2: Bootstrapping Language-Image Pre-training” - arXiv:2301.12597</li>
    </ul>
  </div>
</section>


   <section id="challenges-future" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ความท้าทาย & อนาคต</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      แม้ว่า Multi-Modal Foundation Models จะมีศักยภาพมหาศาลในการเชื่อมโยงข้อมูลจากหลาย modality แต่ก็ยังคงมีความท้าทายเชิงเทคนิค จริยธรรม และสังคมที่สำคัญ
      การพัฒนาและปรับใช้โมเดลเหล่านี้ในระดับ production จำเป็นต้องตระหนักถึงขอบเขตข้อจำกัด ความเสี่ยง และโอกาสในระยะยาว
    </p>

    <h3 className="text-xl font-semibold">8.1 ข้อจำกัดด้านข้อมูลและ Bias</h3>
    <p>
      หนึ่งในอุปสรรคหลักของโมเดลแบบ multi-modal คือการได้มาซึ่งข้อมูลที่ครอบคลุมทั้งภาพและข้อความในปริมาณมากและหลากหลายทางวัฒนธรรม
      ชุดข้อมูลส่วนใหญ่มักกระจุกอยู่ในภาษาอังกฤษและวัฒนธรรมตะวันตก ซึ่งอาจนำไปสู่ bias ที่ขยายตัวเมื่อโมเดลถูกนำไปใช้งานในบริบทที่ต่างออกไป
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 border border-yellow-300 dark:border-yellow-700 p-4 rounded-lg">
      <p className="font-semibold">Insight Box:</p>
      <p>
        โมเดลอย่าง CLIP และ Flamingo แสดงให้เห็นถึง bias ที่แฝงอยู่ในการแมปความหมายระหว่างภาพและคำ เช่น การแมปภาพผู้หญิงกับอาชีพในเชิง stereotype หรือการตอบคำถามที่สะท้อนมุมมองตะวันตก
      </p>
    </div>

    <h3 className="text-xl font-semibold">8.2 ความท้าทายด้านการตีความ</h3>
    <p>
      Multi-modal models ยังเผชิญกับปัญหา “semantic grounding” ที่การตีความภาพหรือข้อความอาจไม่มีความสอดคล้องกันเสมอไป โดยเฉพาะในงานที่ต้องใช้ common sense หรือ reasoning เชิงบริบท
    </p>

    <h3 className="text-xl font-semibold">8.3 ปัญหาด้านพลังงานและการฝึกฝน</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>การฝึกโมเดลระดับ Flamingo ต้องใช้ GPU cluster ขนาดใหญ่ ซึ่งทำให้ต้นทุนการเข้าถึงสูง</li>
      <li>การ fine-tune โมเดลบน task เฉพาะยังมีต้นทุนสูงและต้องการทรัพยากรที่ยืดหยุ่น</li>
      <li>การประเมิน performance ของ multi-modal model ยังไม่มีมาตรฐานกลางที่เป็นที่ยอมรับทั่วโลก</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 border border-blue-300 dark:border-blue-700 p-4 rounded-lg">
      <p className="font-semibold">Highlight:</p>
      <p>
        อนาคตของ multi-modal models จะขึ้นอยู่กับการพัฒนาเทคนิค low-resource fine-tuning เช่น parameter-efficient tuning, adapter layers, และ prompt-based tuning ซึ่งลดภาระด้านพลังงานและข้อมูล
      </p>
    </div>

    <h3 className="text-xl font-semibold">8.4 ทิศทางอนาคตของ Unified AI</h3>
    <p>
      นักวิจัยจาก Stanford, DeepMind และ MIT กำลังพัฒนาโมเดลแบบ unified ที่สามารถประมวลผลทุก modality พร้อมกัน ทั้งภาพ วิดีโอ เสียง ข้อความ และคำสั่ง (instruction)
      โดยเป้าหมายคือการสร้าง “Generalist Agents” ที่สามารถเข้าใจโลกแบบ multimodal ได้อย่างเป็นธรรมชาติ เหมือนมนุษย์
    </p>

    <ul className="list-disc list-inside space-y-1">
      <li><strong>Gemini (Google DeepMind):</strong> แพลตฟอร์ม AI ที่ผสานโมเดล language และ vision เข้าด้วยกัน</li>
      <li><strong>Gato:</strong> โมเดลจาก DeepMind ที่สามารถเล่นเกม Atari, ทำ robotic control และอ่านภาพได้ในโมเดลเดียว</li>
      <li><strong>PaLI-X:</strong> โมเดล multilingual multi-modal ที่รองรับ image captioning และ visual reasoning ข้ามภาษา</li>
    </ul>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>Ramesh, A., et al. (2022). “Hierarchical Text-Conditional Image Generation with CLIP Latents” - arXiv:2204.06125</li>
      <li>Bommasani, R., et al. (2021). “On the Opportunities and Risks of Foundation Models” - Stanford CRFM</li>
      <li>Alayrac, J.-B., et al. (2022). “Flamingo: A Visual Language Model for Few-Shot Learning” - DeepMind</li>
      <li>Reed, S., et al. (2022). “A Generalist Agent” - DeepMind</li>
      <li>Chen, T., et al. (2023). “PaLI-X: Scaling Language-Image Pre-training” - Google Research</li>
    </ul>
  </div>
</section>


   <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">

    <p>
      ในช่วงเวลาที่เทคโนโลยีด้านปัญญาประดิษฐ์พัฒนาอย่างรวดเร็ว แนวทางในการเชื่อมโยงระหว่าง Modalities ต่าง ๆ เช่น ภาพ เสียง และข้อความ
      ได้กลายเป็นแนวโน้มสำคัญของงานวิจัยด้าน Deep Learning โดยเฉพาะในบริบทของ Vision-Language Models ซึ่งรวมเอาความสามารถในการเข้าใจภาพและภาษาธรรมชาติเข้าด้วยกันอย่างลึกซึ้ง
    </p>

    <h3 className="text-xl font-semibold">สรุปบทเรียนสำคัญจากโมเดล CLIP และ Flamingo</h3>
    <p>
      การพัฒนาโมเดล CLIP โดย OpenAI และ Flamingo โดย DeepMind ไม่เพียงแค่สร้างผลกระทบเชิงเทคนิคเท่านั้น แต่ยังเปลี่ยนรูปแบบความเข้าใจ
      ในการเรียนรู้แบบ Cross-modal อย่างสิ้นเชิง การฝึกแบบ contrastive ใน CLIP ทำให้โมเดลสามารถเข้าใจความสัมพันธ์ระหว่างภาพและข้อความ
      ส่วน Flamingo พัฒนาแนวทางการทำ In-Context Learning บนหลาย Modalities ได้อย่างยืดหยุ่น
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 dark:text-white border border-blue-300 dark:border-blue-600 p-4 rounded-xl">
      <h4 className="font-semibold mb-2">Insight: พลังของ Multimodal Alignment</h4>
      <p className="mb-0">
        ความสำเร็จของ CLIP และ Flamingo สะท้อนให้เห็นถึงความสามารถของโมเดลที่สามารถ Mapping ข้อมูลต่าง Modalities ไปสู่ Latent Space ร่วมกันได้
        โดยไม่จำเป็นต้องพึ่งพาการ fine-tune อย่างหนักในแต่ละงาน ส่งผลให้โมเดลมีความสามารถในการ Generalize ที่สูงกว่าโมเดลทั่วไปในอดีต
      </p>
    </div>

    <h3 className="text-xl font-semibold">ความสำคัญเชิงกลยุทธ์ของ Vision-Language Models</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>เพิ่มความยืดหยุ่นของโมเดลในบริบทจริง เช่น สามารถเข้าใจ prompt ที่ประกอบด้วยภาพและข้อความ</li>
      <li>ลดความจำเป็นในการฝึก Fine-tuning แยกในแต่ละโดเมน</li>
      <li>ขยายขอบเขตของการประยุกต์ใช้โมเดล AI เช่น การวิเคราะห์ Medical Imaging ควบคู่กับรายงานทางคลินิก</li>
      <li>สนับสนุนการพัฒนา AI แบบ Foundation Model ที่สามารถทำงานข้ามงานได้</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบความสามารถ: Generalization & In-Context</h3>
    <div className="overflow-x-auto mt-6">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600">โมเดล</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600">Generalization</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600">In-Context Learning</th>
          </tr>
        </thead>
        <tbody className="text-gray-900 dark:text-gray-100">
          <tr className="bg-white dark:bg-gray-800">
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">CLIP</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">สูงมาก (Zero-shot)</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">จำกัด</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700">
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Flamingo</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">สูง</td>
            <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">โดดเด่น (Multimodal Prompt)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 dark:text-white border border-yellow-300 dark:border-yellow-600 p-4 rounded-xl mt-6">
      <h4 className="font-semibold mb-2">Highlight: สู่ยุคของ Multimodal Reasoning</h4>
      <p className="mb-0">
        แนวโน้มใหม่ใน AI คือการรวมความสามารถในการวิเคราะห์เหตุผล (Reasoning) เข้ากับ Multimodal Input เช่น ภาพ วิดีโอ และภาษา
        ซึ่งจะนำไปสู่โมเดลที่มีความสามารถคล้ายมนุษย์มากขึ้นทั้งในการเรียนรู้และสื่อสาร
      </p>
    </div>

    <h3 className="text-xl font-semibold">ประเด็นที่ต้องระวัง: ความซับซ้อนและความโปร่งใส</h3>
    <p>
      แม้ว่าโมเดลเช่น CLIP และ Flamingo จะมีศักยภาพสูง แต่ก็ยังมีประเด็นด้าน Interpretability และการควบคุม Output
      โดยเฉพาะเมื่อโมเดลถูกใช้ในบริบทที่ต้องการความโปร่งใส เช่น ระบบสนับสนุนการตัดสินใจทางการแพทย์ หรือกฎหมาย
    </p>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", OpenAI, 2021 (arXiv:2103.00020)</li>
      <li>Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", DeepMind, 2022 (arXiv:2204.14198)</li>
      <li>Stanford CS231n, Lecture Notes on Multimodal Learning, 2023</li>
      <li>Harvard NLP Group: Vision-Language Pretraining</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day57 theme={theme} />
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
        <ScrollSpy_Ai_Day57 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day57_MultiModalModels;
