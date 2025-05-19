import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day36 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day36";
import MiniQuiz_Day36 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day36";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day36_FineTuning = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

const img1 = cld.image("Day36_1").format("auto").quality("auto").resize(scale().width(660));
const img2 = cld.image("Day36_2").format("auto").quality("auto").resize(scale().width(500));
const img3 = cld.image("Day36_3").format("auto").quality("auto").resize(scale().width(500));
const img4 = cld.image("Day36_4").format("auto").quality("auto").resize(scale().width(500));
const img5 = cld.image("Day36_5").format("auto").quality("auto").resize(scale().width(500));
const img6 = cld.image("Day36_6").format("auto").quality("auto").resize(scale().width(500));
const img7 = cld.image("Day36_7").format("auto").quality("auto").resize(scale().width(500));
const img8 = cld.image("Day36_8").format("auto").quality("auto").resize(scale().width(500));
const img9 = cld.image("Day36_9").format("auto").quality("auto").resize(scale().width(500));
const img10 = cld.image("Day36_10").format("auto").quality("auto").resize(scale().width(500));
const img11 = cld.image("Day36_11").format("auto").quality("auto").resize(scale().width(500));
const img12 = cld.image("Day36_12").format("auto").quality("auto").resize(scale().width(500));
const img13 = cld.image("Day36_13").format("auto").quality("auto").resize(scale().width(500));
const img14 = cld.image("Day36_14").format("auto").quality("auto").resize(scale().width(500));
const img15 = cld.image("Day36_15").format("auto").quality("auto").resize(scale().width(500));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 36: Fine-tuning Pretrained Models</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

         <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: Fine-tuning คือหัวใจของ Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>นิยามของ Fine-tuning ในบริบท Deep Learning</h3>
    <p>
      Fine-tuning คือกระบวนการนำโมเดลที่ผ่านการ pretrain มาแล้ว เช่น BERT, GPT หรือ ResNet มาปรับให้เหมาะกับงานเฉพาะ โดยการฝึกต่อบนชุดข้อมูลใหม่ที่มีขนาดเล็กกว่าขั้น pretraining
      แนวทางนี้ช่วยลดเวลาในการฝึก ลดความต้องการข้อมูล และยังช่วยเพิ่มประสิทธิภาพเมื่อมีข้อมูล labeled จำกัด
    </p>

    <h3>บริบททางประวัติศาสตร์และวิวัฒนาการ</h3>
    <p>
      แนวคิดของ Fine-tuning เริ่มได้รับความสนใจในช่วงปี 2015 จากงานวิจัยเกี่ยวกับ Computer Vision และ NLP ซึ่งพบว่า features ที่เรียนรู้จากโมเดล pretrain สามารถ transfer ไปยัง task อื่นได้ดีมาก
      โดยเฉพาะหลังการเปิดตัว BERT (Devlin et al., 2018) และ GPT-2 (Radford et al., 2019) ทำให้แนวทางนี้กลายเป็นมาตรฐานของวงการ
    </p>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        ในงานของ Raghu et al. (2019) พบว่า ในงานด้าน Medical Imaging การ fine-tune เพียงไม่กี่ layer ก็สามารถสร้างผลลัพธ์ที่ดีเกินคาดเมื่อเทียบกับการฝึกใหม่จากศูนย์
      </p>
    </div>

    <h3>แนวคิดเบื้องหลังการลดต้นทุนการเรียนรู้</h3>
    <ul className="list-disc pl-6">
      <li>Pretraining บน dataset ขนาดใหญ่ เช่น C4, Wikipedia ทำให้โมเดลเรียนรู้ representation ทั่วไป</li>
      <li>Fine-tuning ช่วยให้โมเดล specialize กับ task เฉพาะ เช่น sentiment analysis, medical diagnosis</li>
      <li>ช่วยลด overfitting โดยเฉพาะใน low-resource domains</li>
    </ul>

    <h3>ตารางเปรียบเทียบ: Pretraining vs Fine-tuning</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">มิติ</th>
            <th className="border px-4 py-2">Pretraining</th>
            <th className="border px-4 py-2">Fine-tuning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Dataset Size</td>
            <td className="border px-4 py-2">ใหญ่ (หลายล้านตัวอย่าง)</td>
            <td className="border px-4 py-2">เล็ก (หลักพันถึงหมื่น)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Objective</td>
            <td className="border px-4 py-2">Self-supervised</td>
            <td className="border px-4 py-2">Supervised (ส่วนมาก)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Parameter Update</td>
            <td className="border px-4 py-2">ทั้งหมด</td>
            <td className="border px-4 py-2">บางส่วนหรือทั้งหมด</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Goal</td>
            <td className="border px-4 py-2">เรียนรู้ representation</td>
            <td className="border px-4 py-2">ปรับให้เหมาะกับ task</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805</li>
      <li>Raghu, M. et al. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. NeurIPS</li>
      <li>Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Technical Report</li>
      <li>Stanford CS224N Lecture Notes on Transfer Learning</li>
    </ul>
  </div>
</section>


<section id="why" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ทำไมต้อง Fine-tune?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>เหตุผลหลักของการ Fine-tune โมเดลที่ผ่าน Pretraining</h3>
    <p>
      แม้ว่าโมเดล Pretrained เช่น BERT, GPT, หรือ ResNet จะมีศักยภาพสูงในการเรียนรู้ representation จากข้อมูลจำนวนมาก แต่การนำโมเดลเหล่านี้มาใช้งานจริงในบริบทเฉพาะ เช่น การวิเคราะห์ทางการแพทย์ หรือระบบแชตภาษาไทย จำเป็นต้องปรับจูน (Fine-tuning) เพื่อให้เข้าใจ domain ใหม่ที่แตกต่างจาก pretraining corpus เดิม
    </p>

    <h3>ความจำเป็นในมุมของ Domain Adaptation</h3>
    <p>
      ข้อมูลในโลกความเป็นจริงมักมีลักษณะเฉพาะของตนเอง เช่น รูปแบบทางภาษา ความแปรปรวนของภาพ หรือรูปแบบคำถามเฉพาะกลุ่ม การ fine-tune ช่วยให้โมเดลปรับ representation ให้เข้ากับ domain เหล่านี้ได้ดียิ่งขึ้น
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ในงานของ Gururangan et al. (ACL 2020) พบว่า Domain-Adaptive Pretraining (DAPT) และ Task-Adaptive Pretraining (TAPT) ช่วยเพิ่มความแม่นยำของ BERT อย่างมีนัยสำคัญใน domain เช่น biomedical และ legal text
      </p>
    </div>

    <h3>เปรียบเทียบ: ใช้งาน Pretrained โดยตรง vs. Fine-tuning</h3>
    <ul className="list-disc pl-6">
      <li>Zero-shot: ใช้โมเดล pretrain โดยไม่ปรับ weights มักจะได้ผลลัพธ์ต่ำในงานเฉพาะ</li>
      <li>Few-shot: ใช้ prompt หรือ in-context learning ช่วยได้บ้าง แต่ยังมีข้อจำกัด</li>
      <li>Fine-tuning: ปรับ weights ให้เหมาะกับข้อมูลใหม่โดยตรง มีความแม่นยำสูงสุด</li>
    </ul>

    <h3>ความคุ้มค่าเชิงต้นทุนและประสิทธิภาพ</h3>
    <p>
      การ fine-tune บน dataset ขนาดเล็กสามารถสร้างโมเดลที่มีประสิทธิภาพเทียบเท่ากับการฝึกโมเดลใหม่จากศูนย์บนข้อมูลขนาดใหญ่ โดยใช้ทรัพยากร compute ต่ำกว่าหลายเท่าตัว
    </p>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        ในการศึกษาโดย Phang et al. (2018) พบว่าโมเดล BERT ที่ fine-tune ด้วยข้อมูลเพียง 1,000 ตัวอย่าง ยังสามารถ outperform โมเดลที่ train from scratch ด้วยข้อมูลขนาดใหญ่กว่า
      </p>
    </div>

    <h3>ตัวอย่างการใช้งานจริง</h3>
    <ul className="list-disc pl-6">
      <li>การปรับ BERT เพื่อใช้ในระบบตรวจจับข่าวปลอมในภาษาไทย</li>
      <li>Fine-tune CLIP สำหรับแยกประเภทภาพทางการแพทย์</li>
      <li>นำ GPT-2 มาปรับใช้ในระบบสนทนาเฉพาะสายการเงิน</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Gururangan, S. et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. ACL.</li>
      <li>Phang, J. et al. (2018). Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks. arXiv:1811.01088</li>
      <li>Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL.</li>
      <li>Stanford CS224N Lecture Notes – Fine-tuning and Transfer</li>
    </ul>
  </div>
</section>


<section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. ประเภทของการ Fine-tune</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>การแบ่งประเภทตามระดับการปรับพารามิเตอร์</h3>
    <p>Fine-tuning สามารถจำแนกได้หลายวิธี แต่วิธีหนึ่งที่สำคัญคือการแบ่งตามขอบเขตที่ปรับพารามิเตอร์ของโมเดล โดยแบ่งได้เป็น:</p>
    <ul className="list-disc pl-6">
      <li><strong>Full Fine-tuning:</strong> ปรับพารามิเตอร์ทั้งหมดของโมเดล ซึ่งให้ flexibility สูงสุด แต่ต้องใช้ข้อมูลและ compute มาก</li>
      <li><strong>Partial Fine-tuning:</strong> ปรับเฉพาะบางชั้น เช่น decoder หรือ classifier head</li>
      <li><strong>Adapter-based Tuning:</strong> แทรก layer ขนาดเล็กเข้าไปในโมเดลเดิมแล้ว train เฉพาะส่วนนั้น</li>
      <li><strong>Prefix-tuning / Prompt-tuning:</strong> เพิ่ม token หรือ embedding ที่ปรับได้ไว้ที่ input โดยไม่เปลี่ยนพารามิเตอร์หลัก</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>Parameter-efficient tuning เช่น LoRA และ AdapterFusion ช่วยให้ fine-tuning ทำได้บน resource ที่จำกัด โดยไม่เสีย accuracy มากนัก</p>
    </div>

    <h3>การเลือกประเภทที่เหมาะสมกับบริบท</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">กลยุทธ์</th>
          <th className="border px-4 py-2">ต้องการ compute</th>
          <th className="border px-4 py-2">เหมาะกับ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Full Fine-tuning</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">งานที่ต้องการ performance สูงสุด</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Adapter Tuning</td>
          <td className="border px-4 py-2">ปานกลาง</td>
          <td className="border px-4 py-2">multi-task, low-resource</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Prompt Tuning</td>
          <td className="border px-4 py-2">ต่ำ</td>
          <td className="border px-4 py-2">zero/few-shot</td>
        </tr>
      </tbody>
    </table>

    <h3>เทคนิคใหม่ที่กำลังเติบโต</h3>
    <ul className="list-disc pl-6">
      <li><strong>BitFit:</strong> ปรับแค่ bias ของ layer ต่าง ๆ เท่านั้น</li>
      <li><strong>LoRA (Low-Rank Adaptation):</strong> เพิ่ม matrix ขนาดเล็กเข้าไปใน weight matrix</li>
      <li><strong>IA3:</strong> ปรับแค่ scalar คูณกับ activations</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400">
      <p className="font-semibold">Insight:</p>
      <p>ในงานของ Hu et al. (2021) พบว่า LoRA สามารถลดจำนวน parameter ที่ต้องปรับลงได้กว่า 90% ขณะที่ยังคง performance ใกล้เคียงกับ full fine-tuning</p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Houlsby et al. (2019). Parameter-Efficient Transfer Learning for NLP. arXiv:1902.00751</li>
      <li>Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685</li>
      <li>Li and Liang (2021). Prefix-Tuning: Optimizing Continuous Prompts. arXiv:2101.00190</li>
      <li>Stanford CS224N 2023 Lecture Slides</li>
    </ul>
  </div>
</section>


<section id="best-practices" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Best Practices ก่อน Fine-tune</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>การเตรียมข้อมูลที่เหมาะสม</h3>
    <p>
      ความสำเร็จของการ fine-tune ขึ้นอยู่กับคุณภาพของข้อมูลเป็นหลัก ก่อนเริ่มต้นการฝึก ควรตรวจสอบว่าข้อมูลใหม่มีความสอดคล้องกับ domain เป้าหมายและผ่านการ preprocess อย่างเหมาะสม เช่น tokenization, normalization และ filtering ที่ลด noise
    </p>
    <ul className="list-disc pl-6">
      <li>ลบข้อมูลซ้ำหรือข้อมูลผิดพลาดที่อาจทำให้โมเดลเรียนรู้ผิด</li>
      <li>จัดสมดุลระหว่างแต่ละ class เพื่อป้องกัน bias</li>
      <li>ใช้เทคนิค data augmentation ในกรณีที่ข้อมูลมีจำนวนน้อย</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ในงานของ Hendrycks et al. (2020) แสดงให้เห็นว่าการใช้ benchmark ที่มีการจัดโครงสร้างข้อมูลที่ดี เช่น Robustness Gym มีผลโดยตรงต่อประสิทธิภาพของการ fine-tune
      </p>
    </div>

    <h3>การเลือก learning rate และ optimization strategy</h3>
    <p>
      Learning rate คือหนึ่งใน hyperparameter ที่มีอิทธิพลสูงที่สุด ควรเลือกแบบ warm-up และ cosine decay หรือใช้ scheduler ที่มี adaptive learning
    </p>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">เทคนิค</th>
          <th className="border px-4 py-2">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Learning Rate Warm-up</td>
          <td className="border px-4 py-2">ค่อย ๆ เพิ่มค่า learning rate ช่วงต้นการฝึก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Cosine Decay</td>
          <td className="border px-4 py-2">ลด learning rate ตามรูปแบบโคไซน์</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">AdamW</td>
          <td className="border px-4 py-2">optimizer ที่นิยมที่สุดสำหรับ LLM</td>
        </tr>
      </tbody>
    </table>

    <h3>ควบคุมการ overfitting ด้วย regularization</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ dropout ที่เหมาะสม (0.1–0.3 สำหรับ LLM)</li>
      <li>ใช้ early stopping หาก validation loss หยุดนิ่ง</li>
      <li>ไม่ควร fine-tune ทุก layer หาก dataset เล็ก</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        ในงานของ Raffel et al. (2020) ที่พัฒนา T5 พบว่า การปรับการ regularization และ label smoothing ช่วยเพิ่ม generalization ได้อย่างมีนัยสำคัญ แม้ใน task ที่มีข้อมูลน้อย
      </p>
    </div>

    <h3>การตรวจสอบผลระหว่างฝึก</h3>
    <p>
      การ monitor ค่า loss และ metric เช่น accuracy, F1 score เป็นประจำช่วยให้ปรับ hyperparameter ได้ทันก่อนโมเดลจะ overfit
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR</li>
      <li>Hendrycks, D. et al. (2020). Measuring Massive Multitask Language Understanding. arXiv:2009.03300</li>
      <li>Stanford CS224N: Best Practices for Fine-tuning Pretrained Models</li>
    </ul>
  </div>
</section>


<section id="architectures" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. สถาปัตยกรรมที่นิยม Fine-tune</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>โมเดลภาษาขนาดใหญ่ (LLMs)</h3>
    <p>
      โมเดลประเภท Transformer เป็นแกนหลักของสถาปัตยกรรมที่ถูก fine-tune มากที่สุดในยุคปัจจุบัน โดยเฉพาะ Large Language Models (LLMs) อย่าง GPT-2, GPT-3, BERT, RoBERTa และ T5 ซึ่งมี encoder, decoder หรือ encoder-decoder architecture ขึ้นอยู่กับลักษณะ task
    </p>

    <h3>สถาปัตยกรรมที่เหมาะกับ Natural Language Processing</h3>
    <ul className="list-disc pl-6">
      <li><strong>BERT:</strong> Encoder-only ใช้สำหรับ classification, QA</li>
      <li><strong>GPT:</strong> Decoder-only เหมาะกับ generative task</li>
      <li><strong>T5:</strong> Encoder-decoder เหมาะกับ translation, summarization</li>
    </ul>

    <h3>สถาปัตยกรรมสำหรับ Vision และ Multimodal</h3>
    <ul className="list-disc pl-6">
      <li><strong>ResNet:</strong> Backbone สำหรับ vision task ทั่วไป</li>
      <li><strong>ViT (Vision Transformer):</strong> ใช้ transformer แบบ pure</li>
      <li><strong>CLIP:</strong> ใช้ encoder สองฝั่ง (ภาพ-ข้อความ) เรียนรู้ร่วมกัน</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        โมเดลที่มีความยืดหยุ่นสูง เช่น T5 หรือ BART มักเป็นที่นิยมในงานที่ต้องการทั้งการ encode และ generate ข้อความ เพราะสามารถใช้งานได้ทั้งใน supervised และ unsupervised context
      </p>
    </div>

    <h3>สถาปัตยกรรม Lightweight และ Efficient</h3>
    <p>
      เมื่อ resource จำกัด โมเดลเช่น DistilBERT, MobileBERT หรือ TinyCLIP ถูกออกแบบมาเพื่อการใช้งานที่มีข้อจำกัดด้านหน่วยความจำและ latency โดยยังคงรักษาประสิทธิภาพระดับหนึ่งของต้นฉบับไว้
    </p>

    <h3>เปรียบเทียบโครงสร้างโมเดลหลัก</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">Model</th>
            <th className="border px-4 py-2">Architecture</th>
            <th className="border px-4 py-2">Objective</th>
            <th className="border px-4 py-2">Pretraining Corpus</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">BERT</td>
            <td className="border px-4 py-2">Transformer Encoder</td>
            <td className="border px-4 py-2">MLM, NSP</td>
            <td className="border px-4 py-2">BookCorpus, Wikipedia</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GPT</td>
            <td className="border px-4 py-2">Transformer Decoder</td>
            <td className="border px-4 py-2">Causal LM</td>
            <td className="border px-4 py-2">WebText</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">T5</td>
            <td className="border px-4 py-2">Encoder-Decoder</td>
            <td className="border px-4 py-2">Span Corruption</td>
            <td className="border px-4 py-2">C4</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805</li>
      <li>Brown, T. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165</li>
      <li>Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR</li>
      <li>Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML</li>
      <li>Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition. ICLR</li>
    </ul>
  </div>
</section>


<section id="techniques" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. เทคนิค Fine-tune ให้ได้ผลดี</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>การเลือก Learning Rate และการใช้ Scheduler</h3>
    <p>
      การเลือก learning rate ที่เหมาะสมเป็นปัจจัยสำคัญในการ fine-tune โมเดล โดยค่า learning rate ที่สูงเกินไปอาจทำให้โมเดลเรียนรู้ผิดพลาดจากข้อมูลใหม่ ในขณะที่ค่าที่ต่ำเกินไปอาจทำให้โมเดลเรียนรู้ช้าและไม่สามารถปรับตัวกับ task ใหม่ได้ดี ตัวอย่างของ scheduler ที่นิยมใช้งานคือ cosine decay, linear warm-up และ step decay
    </p>

    <h3>การใช้ Layer Freezing อย่างมีกลยุทธ์</h3>
    <p>
      ในหลายกรณี ไม่จำเป็นต้อง fine-tune พารามิเตอร์ทั้งหมดของโมเดล pretrain โดยสามารถแช่ (freeze) layer ด้านล่างไว้ แล้ว fine-tune เฉพาะ layer ด้านบนที่เกี่ยวข้องกับ task ใหม่ วิธีนี้ช่วยลดโอกาส overfitting และลดความต้องการหน่วยความจำ
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ในการวิจัยของ Howard and Ruder (2018) ในงาน ULMFiT พบว่าการ unfreeze layer ทีละชั้น (discriminative fine-tuning) ช่วยให้ได้ performance ที่เสถียรกว่าการ fine-tune ทั้งหมดพร้อมกัน
      </p>
    </div>

    <h3>การ Normalize ข้อมูลใหม่</h3>
    <p>
      ก่อนการ fine-tune ข้อมูลใหม่ควรถูก normalize ให้อยู่ในช่วงและ distribution ที่ใกล้เคียงกับข้อมูล pretraining เพื่อป้องกัน feature drift หรือ domain shift
    </p>

    <h3>การ Regularize ด้วย Dropout และ Weight Decay</h3>
    <ul className="list-disc pl-6">
      <li>Dropout ถูกใช้เพื่อลดความเสี่ยงของ overfitting ใน task ขนาดเล็ก</li>
      <li>Weight decay ช่วยจำกัดการเปลี่ยนแปลงของพารามิเตอร์ ทำให้โมเดลไม่เบี่ยงเบนจาก representation เดิมมากเกินไป</li>
    </ul>

    <h3>การใช้ Early Stopping</h3>
    <p>
      การใช้ early stopping บน validation set ช่วยป้องกันไม่ให้ fine-tune เกินจุดที่ performance ดีที่สุด โดยเฉพาะในงานที่มีข้อมูล labeled จำกัด
    </p>

    <h3>การใช้ Mixed Precision Training</h3>
    <p>
      เทคนิค mixed precision ช่วยเพิ่มความเร็วในการเทรนและลดการใช้ GPU memory โดยใช้ float16 แทน float32 สำหรับการคำนวณบางส่วน ซึ่งไม่ส่งผลกระทบต่อความแม่นยำของโมเดลในทางปฏิบัติ
    </p>

    <h3>ตารางเปรียบเทียบเทคนิคยอดนิยม</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">ข้อดี</th>
            <th className="border px-4 py-2">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Discriminative Fine-tuning</td>
            <td className="border px-4 py-2">ลดการ overfit ใน layer ล่าง</td>
            <td className="border px-4 py-2">ต้องกำหนด LR หลายค่า</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Gradual Unfreezing</td>
            <td className="border px-4 py-2">ฝึกได้เสถียร</td>
            <td className="border px-4 py-2">เทรนช้าขึ้น</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Layer-wise LR Decay</td>
            <td className="border px-4 py-2">ควบคุมการอัปเดตได้ละเอียด</td>
            <td className="border px-4 py-2">ต้อง tune เพิ่ม</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Howard, J. & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification (ULMFiT). ACL.</li>
      <li>Sun, C., et al. (2019). How to Fine-Tune BERT for Text Classification? arXiv:1905.05583</li>
      <li>Li, X. et al. (2022). Layer-wise Learning Rate Decay for BERT Fine-tuning. arXiv:2206.01731</li>
      <li>Stanford CS224N: NLP with Deep Learning — Lecture on Fine-tuning</li>
    </ul>
  </div>
</section>


<section id="peft" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. การใช้ Parameter-efficient Fine-tuning (PEFT)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>แนวคิดของ Parameter-efficient Fine-tuning (PEFT)</h3>
    <p>
      PEFT คือเทคนิคที่พัฒนาเพื่อให้การปรับ fine-tune โมเดลขนาดใหญ่เกิดขึ้นอย่างมีประสิทธิภาพ โดยไม่จำเป็นต้องอัปเดตพารามิเตอร์ทั้งหมดของโมเดล โดยเฉพาะในยุคของ LLMs (Large Language Models) เช่น GPT, T5 หรือ LLaMA ซึ่งมีขนาดใหญ่มากเกินกว่าจะ fine-tune ได้เต็มรูปแบบภายใต้ข้อจำกัดของหน่วยความจำและพลังประมวลผล
    </p>

    <h3>กลุ่มเทคนิคหลักใน PEFT</h3>
    <ul className="list-disc pl-6">
      <li><strong>Adapters:</strong> เพิ่ม layer เล็ก ๆ แทรกในแต่ละ block ของ Transformer และ train เฉพาะ adapter เหล่านั้น</li>
      <li><strong>LoRA (Low-Rank Adaptation):</strong> ปรับ weight ผ่านการแยก matrix ที่มี rank ต่ำ ช่วยลดพารามิเตอร์ที่ต้องฝึก</li>
      <li><strong>Prefix Tuning:</strong> เพิ่ม context vector เฉพาะ task เข้าไปที่ input โดยไม่แก้ model weight เลย</li>
      <li><strong>BitFit:</strong> ปรับแค่ bias ของ network แทนที่จะปรับทุก weight</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        LoRA กลายเป็นหนึ่งในเทคนิค PEFT ที่ได้รับความนิยมสูงสุดในปี 2022–2023 โดยเฉพาะในการ fine-tune LLaMA และ StableLM บนเครื่องที่ใช้ GPU ระดับ consumer ได้
      </p>
    </div>

    <h3>ข้อดีของ PEFT</h3>
    <ul className="list-disc pl-6">
      <li>ใช้หน่วยความจำน้อยมากเมื่อเทียบกับ full fine-tuning</li>
      <li>สามารถฝึกหลาย task พร้อมกันได้ (multi-task adaptation)</li>
      <li>ลดเวลาในการฝึกและค่าใช้จ่ายโดยรวม</li>
      <li>ง่ายต่อการแชร์ adapter model โดยไม่ต้องแชร์ทั้งโมเดลหลัก</li>
    </ul>

    <h3>ตารางเปรียบเทียบ: PEFT เทียบกับ Full Fine-tuning</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">มิติ</th>
            <th className="border px-4 py-2">Full Fine-tuning</th>
            <th className="border px-4 py-2">PEFT (e.g., LoRA)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">พารามิเตอร์ที่ฝึก</td>
            <td className="border px-4 py-2">100%</td>
            <td className="border px-4 py-2">น้อยกว่า 1%</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">หน่วยความจำที่ใช้</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำมาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความเร็วในการฝึก</td>
            <td className="border px-4 py-2">ช้า</td>
            <td className="border px-4 py-2">เร็ว</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การแชร์ model</td>
            <td className="border px-4 py-2">ต้องแชร์ทั้งโมเดล</td>
            <td className="border px-4 py-2">แชร์แค่ adapter</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ข้อควรระวังและข้อจำกัด</h3>
    <ul className="list-disc pl-6">
      <li>บาง task ที่มี distribution ต่างจาก pretraining อาจต้อง full fine-tuning</li>
      <li>LoRA และ Adapter อาจไม่เพียงพอสำหรับ tasks ที่ต้องการ reasoning ลึก ๆ</li>
      <li>Hyperparameter ของ PEFT (เช่น rank, dropout) มีผลอย่างมากต่อ performance</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Hu, E. et al. (2021). <i>LoRA: Low-Rank Adaptation of Large Language Models</i>. arXiv:2106.09685</li>
      <li>He, P. et al. (2021). <i>DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing</i>. arXiv:2111.09543</li>
      <li>Houlsby, N. et al. (2019). <i>Parameter-Efficient Transfer Learning for NLP</i>. ICML</li>
      <li>Stanford CS25. Lecture 6: Efficient Adaptation of LLMs</li>
    </ul>
  </div>
</section>


<section id="evaluation" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. การประเมินผลหลัง Fine-tuning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>เป้าหมายของการประเมินผลหลังการ Fine-tuning</h3>
    <p>
      หลังการปรับแต่งโมเดลผ่านขั้นตอน Fine-tuning การประเมินผลมีบทบาทสำคัญในการวัดว่าโมเดลสามารถ generalize ได้ดีแค่ไหนบน task เป้าหมาย เช่น การจำแนกข้อความ, การแปลภาษา หรือการวิเคราะห์ภาพ
    </p>

    <h3>เกณฑ์การประเมิน</h3>
    <ul className="list-disc pl-6">
      <li><strong>Accuracy:</strong> เหมาะกับ task แบบ classification ที่มี label ชัดเจน</li>
      <li><strong>F1 Score:</strong> เหมาะสำหรับ class imbalance</li>
      <li><strong>BLEU / ROUGE:</strong> ใช้ใน task ด้าน NLP เช่น machine translation หรือ summarization</li>
      <li><strong>MAE / RMSE:</strong> สำหรับ regression task</li>
    </ul>

    <h3>เทคนิค Evaluation ที่แนะนำ</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ holdout set หรือ cross-validation สำหรับ low-resource task</li>
      <li>วัด performance บน unseen domains เพื่อทดสอบความสามารถของโมเดลในการ generalize</li>
      <li>ใช้ confusion matrix และ error analysis เพื่อวิเคราะห์ข้อผิดพลาดเฉพาะจุด</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ไม่ควรประเมินเฉพาะค่าความแม่นยำ แต่ควรวิเคราะห์ error patterns และค่า metric อื่น ๆ เพื่อให้เข้าใจพฤติกรรมของโมเดลอย่างลึกซึ้ง
      </p>
    </div>

    <h3>แนวปฏิบัติจากงานวิจัยระดับโลก</h3>
    <p>
      งานของ Hendrycks et al. (2020) แนะนำให้ประเมินโมเดลบนชุดข้อมูลที่หลากหลายเพื่อวัดความสามารถในการ generalize ขณะที่ T5 และ BERT ใช้ evaluation benchmark อย่าง SuperGLUE และ SQuAD สำหรับ NLP
    </p>

    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">Metric</th>
            <th className="border px-4 py-2">ใช้ใน Task</th>
            <th className="border px-4 py-2">จุดเด่น</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">F1 Score</td>
            <td className="border px-4 py-2">NER, Classification</td>
            <td className="border px-4 py-2">จัดการ class imbalance</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">BLEU</td>
            <td className="border px-4 py-2">Translation</td>
            <td className="border px-4 py-2">วัดความใกล้เคียงกับคำตอบจริง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">RMSE</td>
            <td className="border px-4 py-2">Regression</td>
            <td className="border px-4 py-2">ลดทอน error ที่สูงเกิน</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Hendrycks, D., et al. (2020). Measuring Massive Multitask Language Understanding. arXiv:2009.03300</li>
      <li>Wang, A., et al. (2019). SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. arXiv:1905.00537</li>
      <li>Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. arXiv:1606.05250</li>
      <li>Stanford CS224N: Evaluation Metrics for NLP</li>
    </ul>
  </div>
</section>


<section id="challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ความท้าทายและข้อผิดพลาดที่พบบ่อย</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>1. Catastrophic Forgetting</h3>
    <p>
      หนึ่งในปัญหาที่พบบ่อยคือการที่โมเดลลืมความรู้ที่ได้จากการ pretraining เมื่อ fine-tune บน dataset ขนาดเล็กหรือ task ที่แตกต่างมาก วิธีการแก้ปัญหานี้อาจรวมถึงการใช้ learning rate ที่ต่ำลง การ freeze layer และการใช้ regularization แบบ L2 หรือ Elastic Weight Consolidation (EWC)
    </p>

    <h3>2. Overfitting บนข้อมูลใหม่</h3>
    <p>
      ในหลายกรณี โดยเฉพาะใน domain ที่มีข้อมูลน้อย การ fine-tune อาจทำให้โมเดล overfit ได้ง่าย เนื่องจากจำนวน parameter มีมากกว่าจำนวนตัวอย่างอย่างมีนัยสำคัญ
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ dropout และ data augmentation</li>
      <li>เลือกใช้ early stopping เพื่อหยุด training ก่อน overfitting</li>
    </ul>

    <h3>3. Distribution Shift</h3>
    <p>
      หากข้อมูลใน task ใหม่แตกต่างจาก pretraining อย่างมาก เช่น เปลี่ยนจากภาษาอังกฤษเป็นภาษาจีน หรือจากภาพถ่ายเป็นภาพดาวเทียม อาจทำให้ performance ต่ำลงแม้ fine-tune อย่างถูกต้อง
    </p>

    <h3>4. Hyperparameter Sensitivity</h3>
    <p>
      Fine-tuning ต้องการการปรับค่าพารามิเตอร์หลายตัว เช่น learning rate, batch size, weight decay ซึ่งมีผลโดยตรงต่อคุณภาพของผลลัพธ์
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        การใช้ learning rate ที่เหมาะสมมีผลโดยตรงต่อความสำเร็จของการ fine-tune โดยเฉพาะเมื่อต้องการ preserve ความรู้เดิมจาก pretraining
      </p>
    </div>

    <h3>5. ความซับซ้อนของ Infrastructure</h3>
    <p>
      การ fine-tune โมเดลขนาดใหญ่ เช่น GPT-3, PaLM, LLaMA ต้องการทรัพยากรอย่างมาก ทั้ง GPU/TPU ขนาดใหญ่, memory จำนวนมาก, และระบบจัดการ training pipeline ที่มีประสิทธิภาพ
    </p>

    <h3>6. ความเข้าใจผิดเกี่ยวกับ Transferability</h3>
    <p>
      บางกรณี การ assume ว่า knowledge จาก pretraining สามารถ transfer ได้เสมออาจไม่เป็นจริง เช่น task เฉพาะทางอย่าง medical diagnosis ที่ pretraining อาจไม่มี representation ที่เกี่ยวข้อง
    </p>

    <h3>7. ตัวอย่างกรณีศึกษา</h3>
    <ul className="list-disc pl-6">
      <li>งานของ He et al. (2021) แสดงว่า fine-tuning บน ImageNet subset ที่มีเพียง 1% ของข้อมูลยังให้ performance ดี แต่ต้องปรับการ normalize ใหม่ทั้งหมด</li>
      <li>Facebook AI Research พบว่าการ fine-tune RoBERTa บนข้อมูลข่าวที่มี bias ส่งผลให้โมเดลทำนายผิดพลาดเมื่อข่าวมาจากแหล่งใหม่</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Goodfellow, I. et al. (2016). Deep Learning. MIT Press.</li>
      <li>Howard, J. and Gugger, S. (2020). Fastai: A layered API for deep learning. arXiv:2002.04688</li>
      <li>Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.</li>
      <li>He, K. et al. (2021). An Empirical Study of Training Self-Supervised Vision Transformers. arXiv:2104.02057</li>
    </ul>
  </div>
</section>


<section id="multilingual" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. การ Fine-tune สำหรับ Multilingual หรือ Multimodal</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>Multilingual Fine-tuning: การจัดการภาษาหลายภาษา</h3>
    <p>
      Multilingual fine-tuning หมายถึงการฝึกโมเดล pre-trained บนภาษาหลายภาษา เพื่อปรับให้เหมาะสมกับ task เฉพาะ เช่น translation, question answering หรือ named entity recognition (NER) ที่รองรับได้หลายภาษาในเวลาเดียวกัน โดยทั่วไปโมเดลจะได้รับ pretraining บน corpus ที่มีความหลากหลายของภาษา เช่น mC4, CC100 หรือ Wikipedia หลายภาษา
    </p>

    <h3>Multimodal Fine-tuning: การรวมหลาย modality</h3>
    <p>
      Multimodal fine-tuning คือการฝึกโมเดลให้สามารถประมวลผลข้อมูลที่มาจากหลาย modality พร้อมกัน เช่น ภาพ + ข้อความ (Vision-Language), เสียง + ข้อความ (Audio-Text) โดยต้องมี encoder เฉพาะสำหรับแต่ละ modality และการออกแบบ cross-modal fusion layer
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยของ Google (mT5, mBART) และ Meta (M2M-100, FLORET) ชี้ให้เห็นว่าการ pretrain บนภาษาที่หลากหลายช่วยเพิ่ม zero-shot performance บนภาษาที่ไม่เคยเห็นมาก่อน โดยเฉพาะภาษาท้องถิ่นที่มี data น้อย
      </p>
    </div>

    <h3>ตารางเปรียบเทียบ: Multilingual vs Multimodal</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">มิติ</th>
            <th className="border px-4 py-2">Multilingual</th>
            <th className="border px-4 py-2">Multimodal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Input Format</td>
            <td className="border px-4 py-2">Text หลายภาษา</td>
            <td className="border px-4 py-2">ภาพ, ข้อความ, เสียง ฯลฯ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Pretraining Corpus</td>
            <td className="border px-4 py-2">mC4, CC100, X-Wiki</td>
            <td className="border px-4 py-2">CLIP Dataset, AudioSet, HowTo100M</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Encoder Design</td>
            <td className="border px-4 py-2">Shared or per-language</td>
            <td className="border px-4 py-2">Per modality + fusion</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Use Cases</td>
            <td className="border px-4 py-2">Translation, NER, QA</td>
            <td className="border px-4 py-2">Image captioning, Speech QA</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ข้อควรพิจารณาในการ Fine-tune</h3>
    <ul className="list-disc pl-6">
      <li>การใช้ shared encoder อาจส่งผลต่อ quality หากภาษา/โมดาลิตีต่างกันมาก</li>
      <li>Multimodal fusion ที่ไม่เหมาะสมอาจเกิด attention collapse หรือ modality dominance</li>
      <li>ควรใช้ normalization และ alignment layer เพื่อให้ feature space เชื่อมโยงกันได้ดี</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        ในโมเดลอย่าง Flamingo และ GIT-2 (Google DeepMind) พบว่า “in-context few-shot multimodal learning” ให้ performance สูงมากเมื่อ fine-tune บน task เฉพาะ เช่น VQA, captioning, video-text retrieval
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Xue et al. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. arXiv:2010.11934</li>
      <li>Fan et al. (2020). Beyond English-centric multilingual machine translation. arXiv:2010.11125</li>
      <li>Alayrac et al. (2022). Flamingo: A Visual Language Model for Few-Shot Learning. DeepMind</li>
      <li>Gadre et al. (2023). GIT-2: Scaling Up Vision-Language Pretraining. arXiv:2303.07916</li>
      <li>Stanford CS330: Multimodal Learning Notes</li>
    </ul>
  </div>
</section>


<section id="case-study" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. กรณีศึกษาจริงจากวงการ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>กรณีศึกษาจาก Google Research: BERT กับ Question Answering</h3>
    <p>
      Google Research ได้นำโมเดล BERT มาทำ fine-tuning บนชุดข้อมูล SQuAD (Stanford Question Answering Dataset) เพื่อให้สามารถตอบคำถามจากบริบทที่กำหนดได้อย่างแม่นยำ โดยการ fine-tune ใช้เวลาเพียงไม่กี่ชั่วโมง แต่ผลลัพธ์แซงหน้ามนุษย์ใน benchmark นี้ในปี 2018
    </p>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        การ fine-tune BERT บน task ที่เฉพาะเจาะจง สามารถเพิ่ม performance ได้อย่างมีนัยสำคัญ โดยไม่ต้องเปลี่ยนสถาปัตยกรรมโมเดลหลัก
      </p>
    </div>

    <h3>การนำ Fine-tuning มาใช้ในงานทางการแพทย์</h3>
    <p>
      งานวิจัยจาก Stanford ใช้ ResNet ที่ผ่านการ pretrain บน ImageNet แล้วนำมาปรับให้เหมาะสมกับการวินิจฉัยภาพถ่าย X-ray ของปอด (CheXNet project) ผลลัพธ์คือโมเดลสามารถตรวจจับภาวะปอดบวมได้แม่นยำกว่าแพทย์ทั่วไปในหลายกรณี
    </p>

    <h3>การใช้ Fine-tuning ในโมเดล Multilingual</h3>
    <ul className="list-disc pl-6">
      <li>Facebook AI นำ XLM-R มา fine-tune กับงาน classification หลายภาษา</li>
      <li>สามารถ maintain performance ได้แม้กับภาษาที่ไม่มีในชุด pretrain</li>
      <li>รองรับ zero-shot cross-lingual transfer ได้ดี</li>
    </ul>

    <h3>เปรียบเทียบผลลัพธ์จากกรณีศึกษา</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">โมเดล</th>
            <th className="border px-4 py-2">Domain</th>
            <th className="border px-4 py-2">ผลลัพธ์</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">BERT</td>
            <td className="border px-4 py-2">NLP / QA</td>
            <td className="border px-4 py-2">F1 Score บน SQuAD มากกว่า มนุษย์</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ResNet</td>
            <td className="border px-4 py-2">Medical Imaging</td>
            <td className="border px-4 py-2">แม่นยำกว่าแพทย์ในบาง task</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">XLM-R</td>
            <td className="border px-4 py-2">Multilingual NLP</td>
            <td className="border px-4 py-2">Zero-shot transfer สำเร็จ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805</li>
      <li>Rajpurkar, P. et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension. arXiv:1606.05250</li>
      <li>Irvin, J. et al. (2019). CheXpert: A Large Chest Radiograph Dataset. arXiv:1901.07031</li>
      <li>Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. arXiv:1911.02116</li>
    </ul>
  </div>
</section>


<section id="code" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Code Example (HuggingFace Transformers)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>การเตรียมสภาพแวดล้อม</h3>
    <p>
      ก่อนเริ่มต้นการ Fine-tune โมเดลจาก HuggingFace Transformers จำเป็นต้องติดตั้งไลบรารีหลักดังนี้:
    </p>
    <pre><code className="language-bash">pip install transformers datasets evaluate</code></pre>

    <h3>โหลด Dataset และ Tokenizer</h3>
    <p>
      ตัวอย่างการโหลดข้อมูลและ Tokenizer สำหรับ GLUE benchmark:
    </p>
    <pre><code className="language-python">
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
    </code></pre>

    <h3>สร้างโมเดลและ TrainingArguments</h3>
    <p>
      ตัวอย่างโมเดลและ configuration สำหรับ fine-tuning:
    </p>
    <pre><code className="language-python">
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
    </code></pre>

    <h3>เริ่มการฝึกด้วย Trainer</h3>
    <pre><code className="language-python">
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
    </code></pre>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        HuggingFace Trainer API ช่วยลดความซับซ้อนของการ Fine-tune และประหยัดเวลาในการจัดการ training loop ด้วยการตั้งค่าแบบ declarative
      </p>
    </div>

    <h3>การประเมินผลและบันทึกโมเดล</h3>
    <pre><code className="language-python">
eval_results = trainer.evaluate()
print(eval_results)

trainer.save_model("./fine-tuned-bert")
    </code></pre>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>HuggingFace Docs: https://huggingface.co/docs/transformers/training</li>
      <li>Stanford CS224N - NLP with Deep Learning (Transformer Fine-tuning)</li>
      <li>Wolf et al., (2020). Transformers: State-of-the-Art Natural Language Processing. arXiv:1910.03771</li>
    </ul>
  </div>
</section>


<section id="tools" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">13. งานวิจัยและเครื่องมือที่แนะนำ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>บทนำ: ความสำคัญของเครื่องมือและงานวิจัยในยุคของ Fine-tuning</h3>
    <p>
      การเลือกใช้เครื่องมือและอ้างอิงจากงานวิจัยที่ผ่านการ peer-review ถือเป็นปัจจัยสำคัญในการพัฒนาโมเดลที่มีประสิทธิภาพในการ fine-tune ให้เหมาะสมกับงานเฉพาะ โดยเฉพาะเมื่อขนาดโมเดลเพิ่มขึ้น และการจัดการทรัพยากร เช่น GPU, RAM และเวลาเริ่มเป็นข้อจำกัด
    </p>

    <h3>เครื่องมือยอดนิยมสำหรับการ Fine-tuning</h3>
    <ul className="list-disc pl-6">
      <li><strong>HuggingFace Transformers:</strong> ไลบรารีที่รองรับการโหลดโมเดลจากหลายสถาปัตยกรรม เช่น BERT, RoBERTa, T5, GPT</li>
      <li><strong>Accelerate:</strong> ใช้สำหรับการจัดการ multi-GPU และ mixed-precision training</li>
      <li><strong>Weights & Biases (WandB):</strong> ระบบติดตามการทดลอง (experiment tracking) ที่ช่วยให้ visualize ได้แบบ real-time</li>
      <li><strong>Optuna / Ray Tune:</strong> สำหรับการ hyperparameter tuning แบบอัตโนมัติ</li>
      <li><strong>LoRA / PEFT (Parameter Efficient Fine-tuning):</strong> ไลบรารีเสริมที่ช่วยให้ fine-tune ได้โดยใช้ parameter น้อยลง</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400">
      <p className="font-semibold">Highlight:</p>
      <p>
        HuggingFace PEFT library ร่วมกับ LoRA ทำให้การ fine-tune โมเดลขนาดใหญ่บน GPU เดียวเป็นไปได้ แม้ในสภาพแวดล้อมที่มีทรัพยากรจำกัด
      </p>
    </div>

    <h3>เครื่องมือในระดับองค์กรและงานขนาดใหญ่</h3>
    <ul className="list-disc pl-6">
      <li><strong>Kubeflow:</strong> สำหรับการจัดการ training pipeline บน Kubernetes</li>
      <li><strong>MLFlow:</strong> บริหาร lifecycle ทั้ง training, deployment และ monitoring</li>
      <li><strong>Amazon SageMaker / Google Vertex AI:</strong> แพลตฟอร์ม Cloud ที่รวมทุกขั้นตอนของ Machine Learning เข้าด้วยกัน</li>
    </ul>

    <h3>ตัวอย่างงานวิจัยที่ควรศึกษาเพิ่มเติม</h3>
    <ul className="list-disc pl-6">
      <li>Houlsby et al. (2019). <em>Parameter-Efficient Transfer Learning for NLP</em>. arXiv:1902.00751</li>
      <li>Hu et al. (2021). <em>LoRA: Low-Rank Adaptation of Large Language Models</em>. arXiv:2106.09685</li>
      <li>Raffel et al. (2020). <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</em>. JMLR</li>
      <li>Gururangan et al. (2020). <em>Don't Stop Pretraining: Adapt Language Models to Domains and Tasks</em>. ACL</li>
    </ul>

    <h3>ข้อควรระวังในการเลือกใช้เครื่องมือ</h3>
    <ul className="list-disc pl-6">
      <li>เครื่องมือที่ไม่ผ่านการตรวจสอบอาจทำให้เกิด data leakage</li>
      <li>ควรตรวจสอบ compatibility กับ GPU / TPU / M1 chip</li>
      <li>การ fine-tune แบบ black-box บน Cloud อาจมีค่าใช้จ่ายสูงหากไม่วางแผนล่วงหน้า</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400">
      <p className="font-semibold">Insight:</p>
      <p>
        นักวิจัยจาก Stanford และ Oxford แนะนำว่า การวางโครงสร้างการทดลองและการติดตามอย่างเป็นระบบช่วยลดเวลา debug และเพิ่ม reproducibility ได้อย่างมาก โดยเฉพาะในงาน production
      </p>
    </div>
  </div>
</section>


<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">14. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img15} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>แนวโน้มการเปลี่ยนผ่านจาก Fine-tuning แบบเดิม</h3>
    <p>
      ในช่วงไม่กี่ปีที่ผ่านมา วงการ Machine Learning ได้เคลื่อนผ่านจุดเปลี่ยนของการใช้ Fine-tuning แบบเต็มโมเดล (full fine-tuning) ไปสู่แนวทางใหม่ที่ประหยัดพารามิเตอร์และคำนวณได้อย่างมีประสิทธิภาพมากขึ้น เช่น PEFT, Adapter Tuning, LoRA และ Prompt-based Learning
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานของ Stanford (2022) ชี้ให้เห็นว่าการใช้ PEFT ช่วยลดค่าใช้จ่ายในการฝึกได้กว่า 90% ในขณะที่ performance บาง task ยังคงใกล้เคียงกับการ fine-tune แบบเต็มโมเดล
      </p>
    </div>

    <h3>ผลกระทบในงานอุตสาหกรรม</h3>
    <ul className="list-disc pl-6">
      <li>องค์กรสามารถ deploy LLMs ได้ง่ายขึ้นด้วยการโหลดแค่ adapter layers แทน full model</li>
      <li>Edge deployment กลายเป็นจริง: บางโมเดลสามารถฝึกและรันบนมือถือหรือ embedded device ได้</li>
      <li>เสริมความสามารถเฉพาะ domain ได้อย่างยืดหยุ่น เช่น medical, finance หรือ legal</li>
    </ul>

    <h3>เปรียบเทียบความคุ้มค่า: Full vs Efficient Fine-tuning</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">มิติ</th>
            <th className="border px-4 py-2">Full Fine-tuning</th>
            <th className="border px-4 py-2">PEFT/Adapter</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">พารามิเตอร์</td>
            <td className="border px-4 py-2">ทั้งหมด (~100%)</td>
            <td className="border px-4 py-2">1-5%</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความเร็วในการฝึก</td>
            <td className="border px-4 py-2">ช้า</td>
            <td className="border px-4 py-2">เร็วขึ้นอย่างมาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">หน่วยความจำ</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความสามารถในการ generalize</td>
            <td className="border px-4 py-2">ดีเยี่ยม</td>
            <td className="border px-4 py-2">ดี (บางกรณีดีกว่า)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>กรณีศึกษาจาก HuggingFace และ Meta</h3>
    <p>
      HuggingFace ได้เผยแพร่ AdapterHub และ PEFT Library เพื่อให้ชุมชนนักพัฒนาเข้าถึงการฝึกแบบประหยัดพารามิเตอร์อย่างแพร่หลาย ในขณะที่ Meta แนะนำ LoRA ใน LLaMA ที่ช่วยให้โมเดลสามารถเรียนรู้ task-specific ได้โดยไม่ต้องฝึก weights หลัก
    </p>

    <h3>ข้อควรระวังในการเลือกวิธี Fine-tuning</h3>
    <ul className="list-disc pl-6">
      <li>งานที่ต้องใช้ representation ลึก ๆ เช่น reasoning ยังอาจต้องใช้ full fine-tuning</li>
      <li>บาง method เช่น prompt-tuning อาจขึ้นกับ prompt quality อย่างมาก</li>
      <li>เทคนิคใหม่ต้องทดสอบกับ validation ที่หลากหลายเสมอ</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ในหลายกรณีที่ใช้ LLMs เพื่อสร้างระบบ NLP เฉพาะ domain เช่น clinical QA หรือ law summarization การใช้ PEFT ไม่เพียงช่วยลดต้นทุน แต่ยังสามารถ deploy และ update ได้รวดเร็วกว่าแนวทางเดิมหลายเท่า
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>He, Z. et al. (2022). Towards Efficient Fine-tuning of Pretrained Language Models. arXiv:2202.08906</li>
      <li>Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685</li>
      <li>Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR</li>
      <li>Stanford CS25 Lecture Notes on Efficient Transfer</li>
    </ul>
  </div>
</section>




          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day36 theme={theme} />
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
        <ScrollSpy_Ai_Day36 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day36_FineTuning;
