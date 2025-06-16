import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day56 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day56";
import MiniQuiz_Day56 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day56";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day56_ImageCaptioning = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day56_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day56_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day56_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day56_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day56_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day56_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day56_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day56_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day56_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day56_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day56_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day56_12").format("auto").quality("auto").resize(scale().width(501));

 return (
      <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>


      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 56: Image Captioning & Vision-Language Models</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

         <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Vision-Language จึงสำคัญ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="space-y-10 text-base leading-relaxed">
    <p>
      ในช่วงทศวรรษที่ผ่านมา การรวมความสามารถด้านการประมวลผลภาพ (vision) และภาษาธรรมชาติ (language) ได้กลายเป็นหนึ่งในความท้าทายหลักของวงการปัญญาประดิษฐ์. แนวคิดของ Vision-Language Models นั้นมุ่งหมายให้ระบบสามารถเข้าใจโลกทั้งในเชิงภาพและเชิงข้อความ พร้อมทั้งตีความ, อธิบาย, และสื่อสารแบบมนุษย์ได้อย่างเป็นธรรมชาติ. โมเดลเหล่านี้มีบทบาทสำคัญในการเชื่อมช่องว่างระหว่างการรับรู้และความเข้าใจเชิงสัญลักษณ์ ซึ่งเป็นรากฐานของ AI เชิงทั่วไป (Artificial General Intelligence).
    </p>

    <h3 className="text-xl font-semibold">บริบทของ Vision-Language ในระบบ AI สมัยใหม่</h3>
    <p>
      ในงานวิจัยจาก Stanford และ CMU พบว่าการจับคู่ภาพและข้อความอย่างมีประสิทธิภาพ ไม่เพียงช่วยให้ระบบสามารถทำงานอย่าง image captioning หรือ visual question answering (VQA) ได้เท่านั้น แต่ยังเป็นการเตรียมพื้นฐานสำหรับงานที่ต้องใช้การ reasoning แบบมัลติโหมด เช่น navigation, robotics และ content generation.
    </p>

    <div className="bg-yellow-500 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ระบบอย่าง GPT-4V, CLIP, Flamingo และ BLIP แสดงให้เห็นถึงพลังของ vision-language fusion ซึ่งทำให้โมเดลสามารถ "เข้าใจ" ภาพในระดับภาษาธรรมชาติ และอธิบาย context หรือ caption ได้แม่นยำกว่าโมเดลเดี่ยวในอดีต.
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งานจริง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ระบบผู้ช่วยด้าน accessibility ที่อธิบายภาพให้ผู้มีความบกพร่องทางการมองเห็น</li>
      <li>ระบบอัตโนมัติใน e-commerce ที่สร้างคำอธิบายสินค้าแบบภาพ</li>
      <li>การใช้ในระบบอัตโนมัติด้านการแพทย์ เช่น อธิบายผล CT scan ด้วย caption ทางคลินิก</li>
    </ul>

    <h3 className="text-xl font-semibold">ภาพรวมของการพัฒนาโมเดล</h3>
    <p>
      วิวัฒนาการของโมเดล Vision-Language เริ่มต้นจาก encoder-decoder architectures แบบพื้นฐาน ไปจนถึงการ pretrain ด้วย contrastive loss (เช่นใน CLIP), multimodal transformers (เช่น Flamingo), และ fine-tuning บน task-specific datasets เช่น COCO, Conceptual Captions, VQA2.0.
    </p>

  <div className="overflow-x-auto mt-6">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Model</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Architecture</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Training Objective</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Dual Encoder</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Contrastive Loss</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">BLIP</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Vision Transformer + Language Decoder</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Captioning + Retrieval</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Flamingo</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multimodal Transformer</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Prompt-based Generation</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", OpenAI, arXiv 2021</li>
      <li>Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation", Salesforce Research, arXiv 2022</li>
      <li>Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", DeepMind, arXiv 2022</li>
      <li>Fei-Fei Li et al., "ImageNet Large Scale Visual Recognition Challenge", Stanford Vision Lab</li>
    </ul>
  </div>
</section>

    <section id="captioning-definition" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Image Captioning คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold mt-8">ความหมายและพื้นฐานของ Image Captioning</h3>
    <p>
      Image Captioning คือกระบวนการสร้างคำอธิบายเป็นข้อความที่เข้าใจได้ของภาพที่กำหนด โดยรวมความสามารถของระบบ Computer Vision ในการเข้าใจภาพ และความสามารถของ Natural Language Processing (NLP) ในการสร้างภาษามนุษย์ที่มีความหมาย. ระบบ Image Captioning ที่มีประสิทธิภาพจะสามารถสรุปสิ่งที่เกิดขึ้นในภาพได้อย่างแม่นยำและกระชับ.
    </p>

    <h3 className="text-xl font-semibold mt-8">ส่วนประกอบหลักของระบบ Image Captioning</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Encoder:</strong> โดยมากเป็น Convolutional Neural Network (CNN) เช่น ResNet หรือ EfficientNet ที่ใช้แปลงภาพเป็น feature vector</li>
      <li><strong>Decoder:</strong> เป็น Recurrent Neural Network (RNN), LSTM หรือ Transformer ที่แปลง feature vector ไปเป็นลำดับของคำ</li>
      <li><strong>Attention Mechanism:</strong> ช่วยให้ decoder โฟกัสในบริเวณภาพที่เกี่ยวข้องขณะสร้างแต่ละคำ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">ตัวอย่างประโยคคำอธิบายที่สร้างโดยระบบ Image Captioning</h3>
    <div className="bg-gray-500 dark:bg-gray-800 p-4 rounded-xl">
<div className="bg-gray-500 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto text-sm">
  <pre>
    <code className="whitespace-pre-wrap break-words text-gray-800 dark:text-gray-200">
"A black dog is running through the grass."
"Two people are riding bicycles on a street."
"A group of children are playing in a park."
    </code>
  </pre>
</div>

    </div>

    <h3 className="text-xl font-semibold mt-8">ประเภทของการเรียนรู้ที่ใช้ใน Image Captioning</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Supervised Learning:</strong> ใช้ dataset เช่น MSCOCO ที่จับคู่ภาพกับคำอธิบายที่มนุษย์เขียนไว้</li>
      <li><strong>Reinforcement Learning:</strong> ปรับปรุงคุณภาพของคำอธิบายโดยใช้ reward function เช่น BLEU หรือ CIDEr</li>
      <li><strong>Pretraining + Finetuning:</strong> ใช้โมเดลที่ pretrain บน multimodal data ก่อน แล้วปรับให้เหมาะกับ captioning task</li>
    </ul>

    <div className="bg-yellow-500 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Image Captioning ถือเป็นหนึ่งใน task แรก ๆ ที่ผลักดันการพัฒนา Vision-Language Models อย่างมีนัยสำคัญ โดยเฉพาะงานวิจัยจากกลุ่ม Google Brain และ Microsoft Research ได้พิสูจน์ว่าระบบ captioning ที่ดีสามารถใช้เป็นรากฐานสำหรับ multimodal reasoning ได้ในอนาคต.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8">อ้างอิงจากแหล่งข้อมูลระดับโลก</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vinyals et al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015</li>
      <li>Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML 2015</li>
      <li>Hossain et al., "Comprehensive Survey of Deep Learning for Image Captioning", ACM Computing Surveys 2019</li>
      <li>Karpathy & Fei-Fei, "Deep Visual-Semantic Alignments for Generating Image Descriptions", CVPR 2015</li>
    </ul>
  </div>
</section>


<section id="early-architectures" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Early Architectures</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="text-base leading-relaxed space-y-6">
    <h3 className="text-xl font-semibold">3.1 จุดเริ่มต้นของระบบ Vision-to-Text</h3>
    <p>
      ระบบ image captioning รุ่นแรกถูกพัฒนาขึ้นโดยใช้แนวทางที่แยกระหว่างโมดูลประมวลผลภาพและโมดูลสร้างภาษา โดยทั่วไปจะประกอบด้วย CNN (เช่น AlexNet หรือ VGGNet) สำหรับแปลงภาพเป็นเวกเตอร์ฟีเจอร์ และ RNN หรือ LSTM สำหรับแปลงเวกเตอร์นั้นเป็นลำดับของคำบรรยาย
    </p>

    <h3 className="text-xl font-semibold">3.2 สถาปัตยกรรม Show and Tell (Google, 2015)</h3>
    <p>
      งานวิจัย <em>Show and Tell</em> โดย Vinyals et al. (Google Brain, 2015) เป็นโมเดลสำคัญยุคแรกที่เสนอแนวคิด "End-to-End" สำหรับ Image Captioning โดยใช้ CNN (Inception) ร่วมกับ LSTM แบบ encoder-decoder
    </p>

    <div className="bg-yellow-500 dark:bg-yellow-900 border border-yellow-400 dark:border-yellow-700 p-4 rounded-lg">
      <h4 className="font-bold mb-2">Insight Box:</h4>
      <p>
        จุดเด่นของ Show and Tell คือความสามารถในการเรียนรู้คำบรรยายจากภาพแบบไม่ต้องพึ่งพา template หรือ rules เชิงภาษาศาสตร์ใด ๆ ซึ่งถือเป็นก้าวสำคัญของการนำ deep learning มาใช้สร้างระบบสื่อความหมายข้าม modality
      </p>
    </div>

    <h3 className="text-xl font-semibold">3.3 การพัฒนาหลังจาก Show and Tell</h3>
    <ul className="list-disc list-inside">
      <li><strong>Show, Attend and Tell (Xu et al., 2015):</strong> เพิ่มกลไก attention บนฟีเจอร์ภาพ เพื่อโฟกัสส่วนที่สำคัญของภาพในแต่ละ timestep ของการสร้างคำ</li>
      <li><strong>Dense Captioning:</strong> สร้างคำบรรยายแบบหลายตำแหน่งในภาพ พร้อมแสดงตำแหน่ง bounding box ของ object แต่ละจุด</li>
      <li><strong>Bottom-Up and Top-Down (Anderson et al., 2018):</strong> ผสาน object detection กับ attention mechanism เพื่อควบคุมการ focus ภาพได้ละเอียดขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">3.4 ข้อจำกัดของสถาปัตยกรรมยุคแรก</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อจำกัด</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด contextual reasoning</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ไม่สามารถเข้าใจความสัมพันธ์เชิงแนวคิดหรือความหมายลึกของภาพ</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ไม่มี pretraining</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">พึ่งพา dataset ขนาดเล็กโดยตรง เช่น MS-COCO</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ผลลัพธ์มักจำกัดความ</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">คำบรรยายที่ได้มักซ้ำกับตัวอย่างฝึก</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-10">3.5 สรุป: จุดเปลี่ยนสู่ Vision-Language Pretraining</h3>
    <p>
      แม้ว่าสถาปัตยกรรมยุคแรกจะประสบความสำเร็จในระดับหนึ่ง แต่ยังขาดความสามารถในการเรียนรู้บริบทกว้าง การประมวลผลความรู้ภายนอก และความสามารถด้าน transfer learning จึงนำไปสู่การพัฒนาแนวทางใหม่อย่าง Vision-Language Pretraining (VLP)
    </p>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Vinyals et al., "Show and Tell: A Neural Image Caption Generator," CVPR 2015</li>
      <li>Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention," ICML 2015</li>
      <li>Anderson et al., "Bottom-Up and Top-Down Attention for Image Captioning and VQA," CVPR 2018</li>
      <li>Stanford CS231n Lecture Notes on Image Captioning</li>
      <li>arXiv:1705.00487 – Review of Deep Learning Architectures in Captioning</li>
    </ul>
  </div>
</section>


 <section id="vlp" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Vision-Language Pretraining (VLP)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      Vision-Language Pretraining (VLP) คือกระบวนการฝึกโมเดลลึกให้เข้าใจความสัมพันธ์ระหว่างภาพและข้อความอย่างกว้างขวาง
      โดยอาศัยชุดข้อมูลขนาดใหญ่ที่มีการจับคู่ระหว่างภาพและข้อความ เช่น caption, alt text หรือแม้กระทั่ง metadata ที่สร้างโดยผู้ใช้
    </p>

    <h3 className="text-xl font-semibold">4.1 นิยามและแนวคิดพื้นฐานของ VLP</h3>
    <p>
      แนวคิดของ VLP เริ่มต้นจากการตระหนักว่าโมเดลที่เรียนรู้เฉพาะภาพหรือเฉพาะข้อความ ไม่สามารถเข้าใจโลกในเชิงมัลติโมดัลได้อย่างสมบูรณ์
      ดังนั้นจึงมีการออกแบบโมเดลที่สามารถรับข้อมูลทั้งภาพและข้อความพร้อมกันเพื่อให้เกิด "cross-modal understanding"
    </p>

    <div className="bg-blue-500 dark:bg-blue-900 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยจาก MIT และ Google Research ชี้ว่าโมเดล VLP เช่น CLIP และ ALIGN มีประสิทธิภาพสูงในการ generalize ไปยัง downstream task ที่ไม่เคยฝึกมาก่อน เช่น zero-shot image classification
      </p>
    </div>

    <h3 className="text-xl font-semibold">4.2 ประเภทข้อมูลที่ใช้ใน VLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Image-Caption Pairs:</strong> ข้อมูลที่พบได้ทั่วไป เช่น MS-COCO, Flickr30k</li>
      <li><strong>Alt-text & Web-scale Data:</strong> ดึงมาจากอินเทอร์เน็ต เช่น YFCC500M, LAION</li>
      <li><strong>Instruction-based:</strong> คำอธิบายที่ยาวและละเอียด เช่นใน datasets ของ Flamingo</li>
    </ul>

    <h3 className="text-xl font-semibold">4.3 เทคนิคการฝึกแบบ Pretraining</h3>
    <p>
      การฝึกโมเดล VLP มักใช้เทคนิค contrastive learning หรือ masked prediction โดยตัวอย่างกลยุทธ์ที่นิยม ได้แก่:
    </p>

  <div className="overflow-x-auto mt-6">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-700 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-700 font-semibold">วิธีการ</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-700 font-semibold">รายละเอียด</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-700">Image-Text Contrastive (ITC)</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-700">
          เรียนรู้ให้ embedding ของภาพและข้อความที่จับคู่กันอยู่ใกล้กัน
        </td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-700">Masked Language Modeling (MLM)</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-700">
          ซ่อนคำใน caption แล้วให้โมเดลเดา โดยมีภาพเป็น context
        </td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-700">Multimodal Fusion</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-700">
          ใช้ attention ในการรวม feature ของทั้งภาพและข้อความในระดับ token
        </td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-yellow-500 dark:bg-yellow-900 p-4 rounded-lg border border-yellow-300 dark:border-yellow-700">
      <p className="font-semibold">Highlight:</p>
      <p>
        การ pretrain ด้วยข้อมูล multimodal ขนาดใหญ่ช่วยให้โมเดลสามารถตอบคำถามที่เกี่ยวกับภาพ, สร้างคำอธิบาย, และเข้าใจบริบทได้ดีกว่าโมเดลที่เรียนแบบ unimodal อย่างมีนัยสำคัญ
      </p>
    </div>

    <h3 className="text-xl font-semibold">4.4 โมเดลตัวอย่างที่ใช้ VLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>CLIP (OpenAI):</strong> Contrastive Language–Image Pretraining</li>
      <li><strong>ALIGN (Google):</strong> Large-scale Image–Text Contrastive Learning</li>
      <li><strong>BLIP/BLIP-2:</strong> Bootstrapped Language-Image Pretraining จาก Salesforce Research</li>
      <li><strong>Flamingo:</strong> โมเดลของ DeepMind ที่มี visual context memory</li>
    </ul>

    <h3 className="text-xl font-semibold">4.5 ความแตกต่างจาก ImageNet Pretraining</h3>
    <p>
      ต่างจาก ImageNet ที่ฝึกด้วย label เดียวต่อภาพ โมเดล VLP ต้องเข้าใจบริบทของข้อความที่เป็นคำบรรยายที่หลากหลายและยาว
    </p>

    <div className="bg-blue-500 dark:bg-blue-900 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
      <p className="font-semibold">Insight:</p>
      <p>
        การรวมภาพและข้อความในการฝึกฝน ไม่เพียงช่วยให้โมเดลเข้าใจ object แต่ยังเข้าใจ context, ความหมาย และความสัมพันธ์เชิงแนวคิด
      </p>
    </div>

    <h3 className="text-xl font-semibold">4.6 ข้อจำกัดและการออกแบบ Dataset</h3>
    <p>
      Dataset ขนาดใหญ่บางชุดมี noise สูงจากการรวบรวมจากเว็บโดยไม่มีการตรวจสอบ การเลือก dataset ที่มีคุณภาพและการปรับ balance จึงเป็นหัวใจสำคัญของ VLP
    </p>

    <h3 className="text-xl font-semibold">4.7 แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021</li>
      <li>Jia et al., "ALIGN: Scaling Up Visual and Vision-Language Pretraining", arXiv:2102.05918</li>
      <li>Li et al., "BLIP: Bootstrapped Language-Image Pretraining", ECCV 2022</li>
      <li>Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", DeepMind 2022</li>
    </ul>
  </div>
</section>


   <section id="core-architectures" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Core Architectures in Modern Models</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-10 text-base leading-relaxed max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
    <p>
      ในยุคปัจจุบัน การพัฒนาโมเดล Vision-Language ที่สามารถเข้าใจภาพและข้อความได้พร้อมกันต้องพึ่งพาสถาปัตยกรรมที่มีความซับซ้อนสูง บทนี้จะสำรวจโครงสร้างแกนหลักของโมเดลสมัยใหม่ที่ถูกออกแบบมาเพื่อรองรับงานที่ใช้ทั้งภาพและภาษาอย่างมีประสิทธิภาพ โดยจะเน้นไปที่โมเดลเชิง Transformer-based และการเชื่อมโยงภาพกับ embedding ทางภาษาที่ลึกซึ้งขึ้น
    </p>

    <h3 className="text-xl font-semibold">5.1 Encoder-Decoder Architecture</h3>
    <p>
      โครงสร้างแบบ Encoder-Decoder ถูกใช้ครั้งแรกในงาน Machine Translation แต่ได้รับการปรับให้เหมาะสมกับงาน Vision-Language โดยให้ Encoder รับภาพ (ผ่าน CNN หรือ ViT) และ Decoder รับภาษา (ผ่าน Transformer หรือ LSTM)
    </p>

   <div className="bg-gray-500 dark:bg-gray-800 p-4 rounded-xl overflow-x-auto text-sm mt-4">
  <pre className="whitespace-pre-wrap break-words text-gray-800 dark:text-gray-200">
    <code>
{`# Pseudo structure of encoder-decoder
image_features = CNN(image_input)
text_output = TransformerDecoder(image_features, text_input)`}
    </code>
  </pre>
</div>


    <h3 className="text-xl font-semibold">5.2 Multimodal Transformer</h3>
    <p>
      โมเดลเช่น ViLBERT, VisualBERT และ UNITER ได้พัฒนาให้ภาพและภาษาถูก encode พร้อมกันในสถาปัตยกรรมแบบเดียว โดยมีการใช้ Attention ระหว่าง modality เพื่อเสริมบริบทซึ่งกันและกัน
    </p>

    <div className="bg-blue-500 p-4 rounded-xl border-l-4 border-blue-500">
      <p className="font-semibold">Highlight:</p>
      <p>
        ViLBERT แยก encoding ของภาพและข้อความ แล้วใช้ co-attentional transformer เชื่อมต่อสอง embedding เข้าด้วยกัน ทำให้สามารถจับคู่ feature ของภาพกับคำที่เกี่ยวข้องได้ลึกยิ่งขึ้น
      </p>
    </div>

    <h3 className="text-xl font-semibold">5.3 CLIP-based Architecture</h3>
    <p>
      Contrastive Language–Image Pretraining (CLIP) จาก OpenAI เป็นตัวอย่างของ zero-shot model ที่ใช้ cosine similarity ระหว่าง image embeddings กับ text embeddings ในการเลือกคำอธิบายภาพที่เหมาะสม
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>ไม่ใช้ decoder แต่ใช้การจับคู่ vector space</li>
      <li>ใช้ contrastive loss ในการ pretrain</li>
      <li>สามารถ generalize ไปยัง task ที่ไม่เคยเห็น</li>
    </ul>

    <h3 className="text-xl font-semibold">5.4 Vision Encoder Language Decoder (VELD)</h3>
    <p>
      โมเดลบางตัว เช่น Flamingo และ GIT ใช้ภาพเป็น input ผ่าน vision encoder และข้อความเป็น target ของ language decoder ที่ได้รับ conditioning จาก embedding ของภาพ
    </p>

    <h3 className="text-xl font-semibold">5.5 Multimodal Bottleneck Transformer</h3>
    <p>
      งานของ CMU และ MIT นำเสนอแนวคิดการใช้ bottleneck token ในการรวม multimodal context โดยมี token กลางที่เรียนรู้จากทั้งภาพและข้อความและทำหน้าที่เป็นตัวกลางของ attention flow
    </p>

    <div className="bg-yellow-500 p-4 rounded-xl border-l-4 border-yellow-500">
      <p className="font-semibold">Insight:</p>
      <p>
        การเพิ่ม token กลาง (bottleneck) ช่วยลด computational cost และยังช่วยให้เกิดการบูรณาการข้อมูลภาพและภาษาได้ดีขึ้น โดยเฉพาะในการฝึกโมเดลขนาดใหญ่ที่มีจำนวน parameter สูง
      </p>
    </div>

    <h3 className="text-xl font-semibold">5.6 เปรียบเทียบโมเดลหลัก</h3>
    <div className="overflow-x-auto mt-6">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Architecture</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Image Processing</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Language Processing</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Interaction</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Encoder-Decoder</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CNN / ViT</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer / LSTM</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Conditional decoding</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multimodal Transformer</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Patch embedding</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Shared transformer</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross attention</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ViT</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cosine similarity</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">VELD</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CNN / ViT</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Decoder only</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Conditioned decoding</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold mt-10">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Tan, H., & Bansal, M. (2019). LXMERT: Learning Cross-Modality Encoder Representations from Transformers. arXiv:1908.07490</li>
      <li>Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.</li>
      <li>Alayrac, J. B., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. arXiv:2204.14198</li>
      <li>Yu, J., et al. (2022). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. CVPR.</li>
    </ul>
  </div>
</section>


     <section id="pretraining-objectives" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Pretraining Objectives</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="max-w-4xl mx-auto text-base leading-relaxed space-y-10">
    <p>
      เป้าหมายหลักของการ Pretraining ใน Vision-Language Models (VLMs) คือการฝึกโมเดลให้เข้าใจการจับคู่อินพุตภาพและข้อความ โดยไม่ต้องใช้ข้อมูลที่มีการกำกับอย่างเข้มข้น (weak supervision) ซึ่งคล้ายกับแนวทางใน NLP ที่โมเดลสามารถเรียนรู้จากการเติมคำหรือการคาดการณ์ประโยคได้ล่วงหน้า
    </p>

    <h3 className="text-xl font-semibold">Masked Language Modeling (MLM)</h3>
    <p>
      เป้าหมายนี้ดัดแปลงมาจาก BERT โดยจะปิดบังบางคำในประโยคอธิบายภาพ และให้โมเดลทำนายคำที่หายไป โดยพิจารณาทั้งจากบริบทของข้อความและภาพที่สัมพันธ์กัน
    </p>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-500 shadow-sm">
      <strong className="block mb-2">Insight:</strong>
      โมเดลอย่าง UNITER และ ViLT ใช้เทคนิค MLM เป็นเป้าหมายหลัก เพื่อให้เข้าใจภาพและภาษาร่วมกันผ่าน attention layers แบบ shared encoder
    </div>

    <h3 className="text-xl font-semibold">Image-Text Matching (ITM)</h3>
    <p>
      ITM เป็นการฝึกให้โมเดลตัดสินว่า ข้อความอธิบายภาพหนึ่งนั้น “ตรง” กับภาพที่ให้มาหรือไม่ มักใช้โครงสร้าง contrastive โดยใส่ negative samples (เช่น ภาพที่ไม่เกี่ยวกับคำอธิบาย) ให้โมเดลเรียนรู้ความแตกต่าง
    </p>

    <table className="table-auto border-collapse w-full text-left text-sm my-6">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-800">
          <th className="p-2 border">Objective</th>
          <th className="p-2 border">อธิบาย</th>
          <th className="p-2 border">ตัวอย่างโมเดล</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border">MLM</td>
          <td className="p-2 border">ทำนาย token ที่หายไปในข้อความโดยดูจากภาพ</td>
          <td className="p-2 border">ViLT, UNITER</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-900">
          <td className="p-2 border">ITM</td>
          <td className="p-2 border">พิจารณาว่าข้อความตรงกับภาพหรือไม่</td>
          <td className="p-2 border">CLIP, ALIGN</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">Contrastive Learning</h3>
    <p>
      โมเดลเรียนรู้โดยการดึง embeddings ของคู่ที่ตรงกัน (เช่น ภาพกับคำอธิบาย) ให้ใกล้กันใน latent space และผลัก embeddings ของคู่ที่ไม่ตรงให้ห่างกัน เช่นที่ใช้ในโมเดล CLIP ของ OpenAI และ ALIGN ของ Google
    </p>

    <pre className="bg-gray-500 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <code>
        L = -log( exp(sim(I, T⁺)/τ) / Σ exp(sim(I, Tᵢ)/τ) ) {"\n"}
        // where sim is cosine similarity, τ is temperature
      </code>
    </pre>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-500 shadow-sm">
      <strong className="block mb-2">Highlight:</strong>
      การใช้ contrastive learning ช่วยให้โมเดลเรียนรู้ cross-modal representation ได้ดีขึ้น และเป็น foundation ของ multimodal retrieval และ zero-shot classification
    </div>

    <h3 className="text-xl font-semibold">Cross-Modal Matching Objectives</h3>
    <p>
      แนวทางใหม่ใน VLP เช่น METER และ FLAVA ใช้ objective ที่ออกแบบเฉพาะสำหรับการเชื่อมโยง modal ต่างกัน เช่น Text-Image Alignment Loss, Masked Patch Prediction และ Multi-Modal Consistency Loss
    </p>

    <ul className="list-disc list-inside mt-4">
      <li>Multi-task Learning: ผสม MLM, ITM, และ contrastive learning ในโมเดลเดียว</li>
      <li>Cross-modal Alignment: การสร้าง latent space ที่รวมทั้ง text และ image</li>
      <li>Knowledge Distillation: ใช้โมเดลที่ pretrain มาก่อนเป็น teacher</li>
    </ul>

    <h3 className="text-xl font-semibold">สรุปอ้างอิง</h3>
    <ul className="list-disc list-inside text-sm mt-2">
      <li>Tan, H. and Bansal, M. “LXMERT: Learning Cross-Modality Encoder Representations from Transformers.” arXiv:1908.07490</li>
      <li>Radford, A. et al. “Learning Transferable Visual Models From Natural Language Supervision.” arXiv:2103.00020 (CLIP)</li>
      <li>Chen, X. et al. “UNITER: UNiversal Image-TExt Representation Learning.” ECCV 2020</li>
      <li>Jia, C. et al. “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision.” ICML 2021 (ALIGN)</li>
    </ul>
  </div>
</section>


      <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Use Cases & Real-World Applications</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-8">
    <h3 className="text-xl font-semibold">7.1 การใช้งาน Image Captioning ในระบบจริง</h3>
    <p>
      Image Captioning เป็นเทคโนโลยีที่ช่วยให้ระบบสามารถสร้างคำอธิบายจากภาพโดยอัตโนมัติ
      ซึ่งถูกนำไปใช้อย่างกว้างขวางในระบบที่ต้องการเข้าใจบริบทของภาพในเชิงลึก โดยเฉพาะในระบบที่ต้องแปลภาษาภาพเป็นข้อความ
      หรือสื่อสารกับมนุษย์ผ่านข้อความที่อธิบายภาพ
    </p>

    <h3 className="text-xl font-semibold">7.2 ตัวอย่างแอปพลิเคชัน</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ระบบคำบรรยายภาพอัตโนมัติสำหรับผู้พิการทางสายตา</li>
      <li>แพลตฟอร์มโซเชียลที่แนะนำ caption อัตโนมัติให้กับผู้ใช้งาน</li>
      <li>การค้นหาภาพจากคำอธิบาย (Reverse Caption Retrieval)</li>
      <li>การอธิบายภาพในระบบ Smart Surveillance (กล้องวงจรปิด)</li>
      <li>การสร้างคำบรรยายภาพใน e-Commerce เพื่อ SEO</li>
    </ul>

    <h3 className="text-xl font-semibold">7.3 การประยุกต์ใช้ในระบบมัลติโมดัล (Multimodal Systems)</h3>
    <p>
      ระบบ Vision-Language ไม่ได้จำกัดอยู่แค่การสร้างคำบรรยาย แต่ยังรวมถึงการประมวลผลภาพร่วมกับข้อความ เช่น VQA (Visual Question Answering),
      Visual Dialog, และ Multimodal Retrieval ซึ่งพบในโมเดลชั้นนำอย่าง Flamingo, GIT, BLIP-2 เป็นต้น
    </p>

    <div className="bg-blue-500 dark:bg-blue-900 p-4 rounded-lg border border-blue-300 dark:border-blue-700 shadow-md">
      <h4 className="font-bold text-lg mb-2">Insight Box: ความก้าวหน้าของ Vision-Language Systems</h4>
      <p>
        ระบบอย่าง OpenAI CLIP และ Google Flamingo เป็นตัวอย่างของการรวม Vision กับ Language
        ทำให้ระบบสามารถเข้าใจภาพผ่านความหมายของภาษาได้ลึกยิ่งขึ้น
        การฝึกแบบ contrastive learning และ prompt tuning ก็กลายเป็นเทคนิคสำคัญในงานวิจัย
      </p>
    </div>

    <h3 className="text-xl font-semibold">7.4 การเปรียบเทียบระบบในอุตสาหกรรม</h3>
   <div className="overflow-x-auto mt-6">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Model</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">จุดเด่น</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">จุดอ่อน</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">BLIP</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Pretraining แบบ Bootstrapping + Captioning</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">ต้องการภาพที่มี annotation คุณภาพ</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">GIT (Grounded Image Text)</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">สามารถ caption ได้แม่นยำและเร็ว</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">ต้องใช้ compute สูงสำหรับ training</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Flamingo</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">Multimodal few-shot learning</td>
        <td className="px-4 py-3 border border-gray-300 dark:border-gray-600">ยังคงอยู่ในช่วงการวิจัย</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">7.5 บทสรุปและมุมมองเชิงกลยุทธ์</h3>
    <p>
      การนำโมเดลที่ผสานภาพและข้อความมาใช้ในระดับอุตสาหกรรมสามารถเพิ่มประสิทธิภาพของระบบ recommendation,
      content understanding, และ personalized user experience ได้อย่างมีนัยสำคัญ
      วิสัยทัศน์ในอนาคตจะนำไปสู่การใช้งานแบบ cross-modal ที่ผสมข้อมูลเสียง, วิดีโอ และเซ็นเซอร์เข้าด้วยกัน
    </p>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Radford et al. “Learning Transferable Visual Models From Natural Language Supervision.” arXiv:2103.00020</li>
      <li>Li et al. “BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.” arXiv:2201.12086</li>
      <li>Alayrac et al. “Flamingo: a Visual Language Model for Few-Shot Learning.” DeepMind 2022</li>
      <li>Salesforce Research. “GIT: A Generative Image-to-Text Transformer.” 2022</li>
    </ul>
  </div>
</section>


   <section id="real-systems" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ตัวอย่างระบบจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="text-base leading-relaxed space-y-10 px-4 lg:px-0 max-w-4xl mx-auto">
    <p>
      ระบบจริงที่ใช้ Vision-Language Models ได้ถูกนำไปใช้ในหลายแวดวง ตั้งแต่การช่วยเหลือผู้พิการทางสายตา ไปจนถึงการทำงานร่วมกับหุ่นยนต์ในสภาพแวดล้อมที่ซับซ้อน ตัวอย่างเหล่านี้แสดงให้เห็นถึงความสามารถของโมเดลที่สามารถตีความภาพและแสดงผลในรูปแบบข้อความได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">8.1 CLIP จาก OpenAI</h3>
    <p>
      CLIP (Contrastive Language–Image Pretraining) เป็นหนึ่งในระบบที่ได้รับการกล่าวถึงอย่างมาก โดยสามารถจับคู่ภาพและข้อความผ่านการเรียนรู้แบบ contrastive ซึ่งเปิดโอกาสให้ระบบสามารถ "เข้าใจ" ความหมายของภาพในระดับภาษาธรรมชาติ
    </p>

    <div className="bg-blue-500 border-l-4 border-blue-500 p-4 rounded-lg">
      <p className="font-semibold">Highlight:</p>
      <p>CLIP ไม่ได้ฝึกโมเดลให้สร้างคำอธิบายภาพโดยตรง แต่เรียนรู้ mapping ระหว่าง image embedding และ text embedding จาก dataset ขนาดใหญ่กว่า 400 ล้านคู่</p>
    </div>

    <ul className="list-disc list-inside mt-4 space-y-2">
      <li>สามารถใช้ text prompt เพื่อค้นหาภาพที่เกี่ยวข้องได้ทันที</li>
      <li>ประสิทธิภาพสูงมากเมื่อใช้ zero-shot classification</li>
      <li>นำไปต่อยอดในการค้นหาภาพ, สรุปเนื้อหา, หรือควบคุมหุ่นยนต์จากคำสั่งภาษา</li>
    </ul>

    <h3 className="text-xl font-semibold">8.2 Flamingo จาก DeepMind</h3>
    <p>
      Flamingo เป็นโมเดล vision-language แบบ few-shot ที่สามารถใช้เพียงตัวอย่างไม่กี่ตัวก็สามารถเข้าใจ task ได้ทันที โดยเฉพาะในงานที่ต้องการให้ AI ตอบคำถามจากภาพ เช่น VQA (Visual Question Answering)
    </p>

    <table className="w-full text-left border mt-4">
      <thead>
        <tr className="bg-gray-500 dark:bg-gray-800">
          <th className="p-2 border">ระบบ</th>
          <th className="p-2 border">ลักษณะเด่น</th>
          <th className="p-2 border">กรณีใช้งาน</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border">CLIP</td>
          <td className="p-2 border">Contrastive Training ระหว่างภาพและข้อความ</td>
          <td className="p-2 border">Zero-shot image classification, semantic search</td>
        </tr>
        <tr>
          <td className="p-2 border">Flamingo</td>
          <td className="p-2 border">Multimodal few-shot learning</td>
          <td className="p-2 border">Visual QA, captioning, interactive agents</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-500 border-l-4 border-yellow-500 p-4 rounded-lg mt-6">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ระบบจริงอย่าง CLIP และ Flamingo ได้สร้างมาตรฐานใหม่ในด้านความสามารถในการ generalize ของโมเดล AI โดยใช้ text-prompt เพียงเล็กน้อย และไม่ต้อง fine-tune เพิ่ม
      </p>
    </div>

    <h3 className="text-xl font-semibold">8.3 แอปพลิเคชันในภาคสังคม</h3>
    <p>
      Vision-Language Models ยังถูกใช้ในการพัฒนาแอปสำหรับผู้พิการ เช่น Seeing AI ของ Microsoft ซึ่งสามารถบรรยายสิ่งที่กล้องมองเห็นออกมาเป็นเสียงแบบเรียลไทม์
    </p>

    <p>ฟังก์ชันที่สนับสนุน:</p>
    <ul className="list-disc list-inside space-y-2">
      <li>อ่านฉลากอาหาร</li>
      <li>บอกสีของวัตถุ</li>
      <li>ตรวจจับใบหน้าคนรอบตัว</li>
    </ul>

    <p>สิ่งเหล่านี้ชี้ให้เห็นว่า Image Captioning และ Vision-Language ไม่ใช่เพียงแนวคิดเชิงทฤษฎี แต่ถูกใช้งานจริงเพื่อช่วยชีวิตในโลกจริง</p>

    <h3 className="text-xl font-semibold">8.4 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 mt-4">
      <li>Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", arXiv:2103.00020</li>
      <li>Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", arXiv:2204.14198</li>
      <li>Microsoft Seeing AI: https://www.microsoft.com/en-us/ai/seeing-ai</li>
    </ul>
  </div>
</section>

    <section id="challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ความท้าทาย</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="text-base leading-relaxed space-y-10 px-4 lg:px-0 max-w-4xl mx-auto">

    <p>
      แม้ว่าระบบ Vision-Language (VL) และ Image Captioning จะได้รับความสนใจและพัฒนาอย่างรวดเร็ว แต่ยังคงมีความท้าทายเชิงเทคนิคและจริยธรรมหลายด้านที่ต้องได้รับการแก้ไข โดยเฉพาะอย่างยิ่งในประเด็นของ bias, ความเข้าใจเชิงเหตุผล และการควบคุมผลลัพธ์
    </p>

    <h3 className="text-xl font-semibold">9.1 ปัญหาเรื่อง Bias และความไม่สมดุลของข้อมูล</h3>
    <p>
      เนื่องจากโมเดล VL ส่วนมากได้รับการฝึกจากชุดข้อมูลที่ดึงมาจากอินเทอร์เน็ต เช่น LAION หรือ COCO จึงมีความเสี่ยงในการเรียนรู้ bias ทางเชื้อชาติ เพศ และวัฒนธรรมอย่างไม่ได้ตั้งใจ
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ภาพของบางเพศหรือเชื้อชาติอาจถูกแทนด้วยคำอธิบายที่มีอคติ</li>
      <li>ความถี่ของวัตถุบางประเภทใน dataset ส่งผลต่อความแม่นยำในการพรรณนาภาพ</li>
      <li>ระบบอาจเลือกใช้คำไม่เหมาะสมในบริบททางสังคมหรือวัฒนธรรม</li>
    </ul>

    <div className="bg-yellow-500 border-l-4 border-yellow-500 p-4 rounded-lg">
      <p className="font-semibold">Insight:</p>
      <p>
        โมเดล VL ที่แม่นยำอาจไม่เท่ากับ “เป็นธรรม” — จึงมีความจำเป็นต้องออกแบบ pipeline เพื่อตรวจสอบและลด bias อย่างเป็นระบบ
      </p>
    </div>

    <h3 className="text-xl font-semibold">9.2 ความเข้าใจที่เกินข้อมูลภาพ (Compositional Reasoning)</h3>
    <p>
      หนึ่งในความท้าทายหลักคือการให้โมเดลเข้าใจ "บริบท" ที่ซับซ้อน เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>การตีความเหตุการณ์จากลำดับการกระทำ</li>
      <li>การเชื่อมโยงระหว่างวัตถุหลายชนิดในภาพเดียว</li>
      <li>การเข้าใจมุขตลกหรือสิ่งแฝงนัยที่ไม่ได้แสดงตรง ๆ</li>
    </ul>

    <div className="bg-blue-500 border-l-4 border-blue-500 p-4 rounded-lg">
      <p className="font-semibold">Highlight:</p>
      <p>
        แม้ว่าโมเดลอย่าง Flamingo หรือ GPT-4V จะทำได้ดีในบาง task แต่การสรุปภาพรวมที่ต้องใช้ตรรกะหรือ common sense reasoning ยังคงเป็นจุดอ่อน
      </p>
    </div>

    <h3 className="text-xl font-semibold">9.3 ความยากในการประเมินผลลัพธ์ (Evaluation Metrics)</h3>
    <p>
      การประเมินประสิทธิภาพของ Image Captioning มักใช้ BLEU, METEOR หรือ CIDEr ซึ่งเน้นการเปรียบเทียบคำกับคำอ้างอิง แต่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ไม่สามารถสะท้อนความ "ถูกต้องตามตรรกะ" ได้เสมอ</li>
      <li>บาง caption ที่ใช้คำต่างกัน อาจให้ความหมายที่เหมือนกัน</li>
      <li>การประเมินด้วยมนุษย์มีต้นทุนสูงและไม่สามารถ scale ได้</li>
    </ul>

    <p>
      ส่งผลให้การประเมินระบบ VL ยังเป็นความท้าทายในการวิจัย ซึ่งต้องการการออกแบบ metric ที่สะท้อน “ความเข้าใจ” มากกว่าแค่การ “จำคำ”
    </p>

    <h3 className="text-xl font-semibold">9.4 ข้อจำกัดด้าน computation และ energy</h3>
    <p>
      โมเดลขนาดใหญ่ต้องการ GPU ที่มี memory สูง (เช่น A500) และพลังงานจำนวนมากในการฝึก ซึ่งนำมาสู่ปัญหาทางสิ่งแวดล้อมและการเข้าถึงของประเทศกำลังพัฒนา
    </p>

    <table className="w-full text-left border mt-4">
      <thead>
        <tr className="bg-gray-500 dark:bg-gray-800">
          <th className="p-2 border">โมเดล</th>
          <th className="p-2 border">จำนวนพารามิเตอร์</th>
          <th className="p-2 border">GPU Training Time</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-2 border">CLIP</td>
          <td className="p-2 border">400M+</td>
          <td className="p-2 border">มากกว่า 256 GPUs × หลายวัน</td>
        </tr>
        <tr>
          <td className="p-2 border">Flamingo</td>
          <td className="p-2 border">80B+</td>
          <td className="p-2 border">TPU Pods (Google scale)</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-500 border-l-4 border-yellow-500 p-4 rounded-lg mt-6">
      <p className="font-semibold">Insight:</p>
      <p>
        การ democratize เทคโนโลยี AI ยังเป็นเป้าหมายที่ไกล — โดยเฉพาะถ้าทรัพยากรการฝึกยังคงกระจุกตัวที่องค์กรขนาดใหญ่
      </p>
    </div>

    <h3 className="text-xl font-semibold">9.5 อ้างอิง</h3>
    <ul className="list-disc list-inside mt-4 space-y-2">
      <li>Li et al., "Aligning Pretrained Models via Dataset Bias Analysis", arXiv:2303.05050</li>
      <li>Hendricks et al., "Women also Snowboard: Overcoming Bias in Captioning Models", arXiv:1803.09797</li>
      <li>Shekhar et al., "Evaluating Multimodal Reasoning in VL Models", ACL 2023</li>
      <li>Google DeepMind, "Flamingo Model Card", 2022</li>
    </ul>

  </div>
</section>


  <section id="future-trends" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    10. Future Trends
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <div className="space-y-10">
      <div>
        <h3 className="text-xl font-semibold">ความก้าวหน้าทาง Multimodal AI</h3>
        <p>
          การบูรณาการระหว่างการมองเห็น (Vision) และการเข้าใจภาษาธรรมชาติ (Language Understanding) จะเป็นรากฐานสำคัญของระบบ AI ที่สามารถคิด วิเคราะห์ และสื่อสารกับมนุษย์ได้ในลักษณะใกล้เคียงกับความเข้าใจของมนุษย์มากขึ้น โดยเฉพาะโมเดลที่รองรับ input หลากหลาย เช่น ข้อความ + รูปภาพ + วิดีโอ + เสียง จะมีความสามารถในการให้คำอธิบายหรือแนะนำที่ซับซ้อนมากขึ้น เช่น GPT-4V หรือ Gemini
        </p>
        <div className="bg-blue-500 dark:bg-blue-900/40 border border-blue-300 dark:border-blue-700 p-4 rounded-xl mt-4">
          <p className="text-sm font-medium">
            Highlight: ระบบ AI ที่รองรับหลาย modality พร้อมกัน เช่น Visual Question Answering (VQA), Document Understanding และ AI Copilot แบบ AR/VR กำลังถูกพัฒนาอย่างต่อเนื่อง
          </p>
        </div>
      </div>

      <div>
        <h3 className="text-xl font-semibold">การเรียนรู้จากข้อมูลที่มีความซับซ้อนสูง</h3>
        <p>
          การวิจัยล่าสุดได้มุ่งไปที่การสอนโมเดลให้เรียนรู้จากข้อมูลที่มีโครงสร้างซับซ้อนมากขึ้น เช่น องค์ความรู้จากคู่มือทางการแพทย์ (multimodal reports), ชุดข้อมูลด้านการศึกษา (ed-tech), และเอกสารที่รวมทั้ง schema, diagram และภาพประกอบ
        </p>
      </div>

      <div>
        <h3 className="text-xl font-semibold">Real-Time Image Captioning ในแอปพลิเคชัน</h3>
        <ul className="list-disc pl-6 space-y-2">
          <li>เทคโนโลยี Assistive AI เช่น Be My Eyes ใช้ Image Captioning แบบเรียลไทม์เพื่อช่วยผู้พิการทางสายตา</li>
          <li>การแปลภาพถ่ายทันทีในแอปกล้องแบบ Google Lens</li>
          <li>ระบบ Auto ALT-text บน Social Media สำหรับคนตาบอด</li>
        </ul>
      </div>

      <div>
        <h3 className="text-xl font-semibold">Ethics และ Explainability ของ Vision-Language Models</h3>
        <p>
          ความท้าทายที่สำคัญในอนาคตคือการทำให้โมเดล Vision-Language มีความสามารถในการอธิบายเหตุผล (Reasoning) และการอธิบายคำตอบได้อย่างโปร่งใส (Explainable AI)
        </p>
        <div className="bg-yellow-500 dark:bg-yellow-900/30 border border-yellow-300 dark:border-yellow-700 p-4 rounded-xl mt-4">
          <p className="text-sm font-medium">
            Insight: ในอนาคต โมเดลจำเป็นต้อง "อธิบายได้" ว่าทำไมถึงตอบเช่นนั้น ไม่ใช่เพียงแค่ให้คำตอบที่ถูกต้องแต่ไม่มีเหตุผลรองรับ
          </p>
        </div>
      </div>

      <div>
        <h3 className="text-xl font-semibold">แนวทางการผสานกับ AGI</h3>
        <p>
          โมเดล Vision-Language จะเป็นแกนสำคัญของการพัฒนา Artificial General Intelligence (AGI) ในอนาคต เนื่องจากการเข้าใจภาพ + ภาษาพร้อมกันคือรากฐานของการเข้าใจโลกแบบมนุษย์
        </p>
      </div>

      <div>
        <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
        <ul className="list-disc pl-6 space-y-2">
          <li>Stanford Vision-Language Initiative, 2024</li>
          <li>OpenAI GPT-4V Technical Report</li>
          <li>"Multimodal Learning with Transformers" – Oxford Deep Learning 2023</li>
          <li>"VisualGPT: Data-Efficient Adaptation for Vision-Language" – arXiv:2205.01580</li>
          <li>"Be My Eyes powered by GPT-4V" – OpenAI & Accessibility Journal, 2024</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Summary</h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      การเรียนรู้เชิงลึก (Deep Learning) ได้เปลี่ยนวิธีการที่มนุษย์ใช้คอมพิวเตอร์วิเคราะห์ภาพ เสียง และภาษาธรรมชาติ โดยเฉพาะเมื่อสถาปัตยกรรมแบบ Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), และ Transformer ถูกนำมาผสมผสานเข้ากับข้อมูลหลายมิติทั้งภาพและข้อความ ทำให้เกิดความสามารถใหม่ที่ล้ำหน้ากว่าเดิมในทุกด้านของ AI
    </p>

    <p>
      โมเดลอย่าง CLIP, BLIP และ Flamingo ชี้ให้เห็นถึงพลังของระบบที่สามารถเข้าใจข้อมูลแบบ Multimodal ได้อย่างแท้จริง โดยการฝึกจาก caption ภาพแบบ massive scale ทำให้เกิด embedding space ที่สามารถแมปภาพและข้อความในบริบทเดียวกันได้อย่างมีประสิทธิภาพ
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-800 border border-yellow-300 dark:border-yellow-600 p-4 rounded-xl">
      <h4 className="font-semibold mb-2">Highlight: การบูรณาการภาพกับภาษาคือหัวใจของ AI สมัยใหม่</h4>
      <p>
        ความสามารถของโมเดลในการเข้าใจทั้งภาพและข้อความ ไม่ได้เป็นเพียงการจำแนกวัตถุหรือแปลภาษาอีกต่อไป แต่เป็นการสร้างความเข้าใจแบบบริบทข้าม modality ซึ่งจะเป็นรากฐานของ AI ทั่วไปในอนาคต
      </p>
    </div>

    <p>
      อย่างไรก็ตาม ความท้าทายยังคงอยู่ในเรื่องของ reasoning, bias จากข้อมูลฝึก, และการนำไปใช้อย่างรับผิดชอบ ซึ่งเป็นโจทย์สำคัญที่นักวิจัยต้องแก้ไขอย่างต่อเนื่อง ไม่ใช่เพียงเพื่อประสิทธิภาพ แต่เพื่อความปลอดภัยและจริยธรรมของเทคโนโลยี
    </p>

    <p>
      การพัฒนาระบบ AI ที่มีความเข้าใจลึกซึ้งและมีความสามารถในการ generalize ได้ดีในโลกจริง จำเป็นต้องอาศัยการผสมผสานระหว่างแนวทางวิศวกรรม, ปรัชญาการเรียนรู้, และความเข้าใจที่ลึกซึ้งต่อข้อมูลทั้งในเชิงสถิติและภาษาธรรมชาติ
    </p>

    <div className="bg-blue-100 dark:bg-blue-800 border border-blue-300 dark:border-blue-600 p-4 rounded-xl">
      <h4 className="font-semibold mb-2">Insight Box: เส้นทางต่อไปของ AI คือการเรียนรู้แบบเข้าใจโลก</h4>
      <p>
        การพัฒนาโมเดลที่ไม่เพียงแค่ “ทำนายถูก” แต่ “เข้าใจอย่างมีบริบท” คือหัวใจของการสร้าง AI ที่สามารถใช้ชีวิตร่วมกับมนุษย์ในระบบนิเวศแห่งอนาคตได้อย่างกลมกลืน
      </p>
    </div>
  </div>
</section>


   <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Research Foundations & Global Contributions</h2>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-base leading-relaxed space-y-10">
    <p>
      ความก้าวหน้าในสถาปัตยกรรม Deep Learning ไม่ได้เกิดขึ้นอย่างโดดเดี่ยว แต่เป็นผลลัพธ์จากการบูรณาการระหว่างงานวิจัยเชิงทฤษฎีจากมหาวิทยาลัยระดับโลก และการผลักดันขีดความสามารถในระดับอุตสาหกรรมโดยองค์กรวิจัยเทคโนโลยีชั้นนำ การเข้าใจต้นน้ำของความรู้เหล่านี้คือรากฐานสำคัญในการพัฒนาโมเดล AI ที่ลึกซึ้งและมีความสามารถสูง
    </p>

    <h3 className="text-xl font-semibold">13.1 มหาวิทยาลัยผู้ผลักดันแนวคิดหลัก</h3>
    <ul className="list-disc pl-6 space-y-3">
      <li><strong>Stanford University:</strong> จุดเริ่มของความเข้าใจ CNN เชิงลึก (CS231n)</li>
      <li><strong>MIT:</strong> ศูนย์กลางของการสอน Transformer, RL และ Multimodal AI (6.S191)</li>
      <li><strong>CMU & Oxford:</strong> วิจัยเรื่อง LSTM, GNN และ VGGNet ที่เป็นรากฐานของงานวิจัยยุคใหม่</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-800 border border-yellow-300 dark:border-yellow-600 p-4 rounded-xl">
      <h4 className="font-semibold mb-2">Highlight: สถาบันเหล่านี้ไม่เพียงผลิตนักวิจัย แต่สร้าง "ภาษากลางของ AI สมัยใหม่"</h4>
      <p>หลักสูตรจาก MIT และ Stanford ถูกนำไปประยุกต์ใช้จริงโดยวิศวกร AI ในองค์กรระดับโลก</p>
    </div>

    <h3 className="text-xl font-semibold">13.2 งานวิจัยต้นแบบและสถาปัตยกรรมเปลี่ยนโลก</h3>
    <ul className="list-decimal pl-6 space-y-3">
      <li>“Attention Is All You Need” (Vaswani et al., 2017) — วางรากฐาน Transformer</li>
      <li>“Deep Residual Learning” (He et al., 2016) — เสนอ ResNet แก้ vanishing gradient</li>
      <li>“Long Short-Term Memory” (Hochreiter & Schmidhuber, 1997) — เปลี่ยนการเรียนรู้ลำดับใน RNN</li>
      <li>“AlphaFold” (Jumper et al., 2021) — แสดงศักยภาพของ AI ในชีววิทยา</li>
      <li>“CLIP” (Radford et al., 2021) — เชื่อมภาพกับภาษาผ่าน contrastive learning</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-800 border border-blue-300 dark:border-blue-600 p-4 rounded-xl">
      <h4 className="font-semibold mb-2">Insight Box: arXiv ไม่ใช่แค่ที่เผยแพร่ แต่เป็นสนามทดลองไอเดียแห่งอนาคต</h4>
      <p>งานวิจัยหลายชิ้นกลายเป็นจุดเปลี่ยนของวงการ AI ก่อนจะเข้าสู่วารสารทางการ เช่น NeurIPS, CVPR, Nature</p>
    </div>

    <h3 className="text-xl font-semibold">13.3 องค์กรวิจัยที่ผลักดันโลก AI</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>OpenAI:</strong> GPT, CLIP, DALL·E — ความสามารถแบบ Foundation Models</li>
      <li><strong>DeepMind:</strong> AlphaGo, AlphaFold — ต้นแบบ AI เชิงกลยุทธ์และวิทยาศาสตร์</li>
      <li><strong>FAIR (Meta):</strong> Detectron2, ResNeXt — นำ CNN สู่ production scale</li>
      <li><strong>Google Brain:</strong> BERT, ViT, EfficientNet — ผสาน Vision และ NLP อย่างมีประสิทธิภาพ</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-800 border border-yellow-300 dark:border-yellow-600 p-4 rounded-xl">
      <h4 className="font-semibold mb-2">Highlight: การติดตาม Open Source และ paper จากองค์กรเหล่านี้ คือการเรียนรู้จาก “แนวหน้าของโลก”</h4>
      <p>โมเดลและโค้ดที่เผยแพร่จาก Google, Meta และ OpenAI มักถูกนำไปต่อยอดในงานวิจัยใหม่และ AI startup ทั่วโลก</p>
    </div>
  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day56 theme={theme} />
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
        <ScrollSpy_Ai_Day56 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day56_ImageCaptioning;
