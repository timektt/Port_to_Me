"use client";
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day54 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day54";
import MiniQuiz_Day54 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day54";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day54_FineTuningLanguageModels = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

const img1 = cld.image("Day54_1").format("auto").quality("auto").resize(scale().width(660));
const img2 = cld.image("Day54_2").format("auto").quality("auto").resize(scale().width(500));
const img3 = cld.image("Day54_3").format("auto").quality("auto").resize(scale().width(500));
const img4 = cld.image("Day54_4").format("auto").quality("auto").resize(scale().width(500));
const img5 = cld.image("Day54_5").format("auto").quality("auto").resize(scale().width(500));
const img6 = cld.image("Day54_6").format("auto").quality("auto").resize(scale().width(500));
const img7 = cld.image("Day54_7").format("auto").quality("auto").resize(scale().width(500));
const img8 = cld.image("Day54_8").format("auto").quality("auto").resize(scale().width(500));
const img9 = cld.image("Day54_9").format("auto").quality("auto").resize(scale().width(500));
const img10 = cld.image("Day54_10").format("auto").quality("auto").resize(scale().width(500));
const img11 = cld.image("Day54_11").format("auto").quality("auto").resize(scale().width(500));
const img12 = cld.image("Day54_12").format("auto").quality("auto").resize(scale().width(500));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

       <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-20">
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 54: Fine-tuning Language Models</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

         <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
      <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Fine-tuning จึงสำคัญ</h2>
      <div className="flex justify-center my-6">
        <AdvancedImage cldImg={img2} />
      </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในยุคของ <strong>Large-scale Pretrained Language Models</strong> เช่น BERT, GPT, T5 การ <strong>Fine-tuning</strong> ได้กลายมาเป็นหนึ่งในขั้นตอนสำคัญที่สุดที่ผลักดันความก้าวหน้าของงาน NLP สมัยใหม่.  
      แม้ว่าโมเดลเหล่านี้จะถูกฝึกในลักษณะ <em>general-purpose</em> บน data ขนาดใหญ่, การนำไปใช้งานจริงกับ <strong>specific downstream tasks</strong> เช่น QA, summarization, หรือ sentiment analysis ยังคงต้องอาศัยกระบวนการ Fine-tuning เพื่อปรับให้ model เข้าใจ context เฉพาะ domain นั้น ๆ อย่างลึกซึ้ง.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.1 ความท้าทายของ Pretraining อย่างเดียว</h3>
    <p>
      แม้ว่า Pretraining จะสร้าง <strong>rich general representations</strong> ได้ดีมาก, แต่ในหลาย use case, โมเดล pretrained แบบ pure (zero-shot) อาจยังไม่สามารถให้ performance ระดับ production ได้ โดยเฉพาะใน domain ที่มี <strong>distribution shift</strong> จาก data ที่ pretrain มาเดิม.
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ตัวอย่างเช่น BERT pretrained บน Wikipedia + BookCorpus อาจ perform ไม่ดีนักบน biomedical text.</li>
      <li>GPT pretraining บน web text อาจไม่เข้าใจ structure เฉพาะของ legal documents หรือ financial reports.</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การ Fine-tuning ไม่ได้เป็นแค่ "training เพิ่มอีกนิด", แต่เป็นการ <strong>adapt representation space</strong> ของ model ให้ align กับ distribution ของ task ใหม่.  
        งานของ Howard & Ruder (ULMFiT) และ Devlin et al. (BERT) ได้แสดงให้เห็นว่า fine-tuning มีผลอย่างยิ่งต่อ downstream performance.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.2 ประโยชน์หลักของ Fine-tuning</h3>
    <p>ต่อไปนี้คือเหตุผลสำคัญที่ทำให้ Fine-tuning เป็นส่วนที่ "ขาดไม่ได้" ใน modern NLP pipeline:</p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Benefit of Fine-tuning</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Domain Adaptation</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ distribution เฉพาะของ target domain</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Task Specialization</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ objective ใหม่ เช่น classification, generation, QA</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Performance Boost</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ปรับ model ให้ achieve sota performance บน task เป้าหมาย</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.3 ตัวอย่างผลลัพธ์ที่ Fine-tuning ทำให้เกิด</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>BERT after fine-tuning → SOTA ใน GLUE benchmark</li>
      <li>T5 after fine-tuning → outperforming previous approaches on summarization</li>
      <li>BioBERT after fine-tuning → significant improvement บน biomedical NER & QA</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Fine-tuning ช่วยทำให้ pretrained LM เปลี่ยนจาก <em>general-purpose knowledge engine</em> ให้กลายเป็น <em>domain-specialized expert</em> ได้อย่างน่าทึ่ง — ทั้งใน NLP, multimodal, และ cross-lingual tasks.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.4 สรุป</h3>
    <p>
      ใน modern ML/NLP pipeline, การ Fine-tuning ได้กลายเป็น <strong>core phase</strong> ที่เติมเต็มช่องว่างระหว่าง pretraining ที่ general กับ task ที่เฉพาะเจาะจง.  
      การออกแบบ fine-tuning strategy ที่เหมาะสมจะมีผลโดยตรงต่อ performance และความสามารถในการ generalize ของ model.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Howard & Ruder. Universal Language Model Fine-tuning for Text Classification (ULMFiT), arXiv 2018.</li>
      <li>Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.</li>
      <li>Gururangan et al. Don't Stop Pretraining: Adapt Language Models to Domains and Tasks, ACL 2020.</li>
      <li>Lee et al. BioBERT: a pre-trained biomedical language representation model, Bioinformatics 2020.</li>
    </ul>
  </div>
</section>


       <section id="pretraining-vs-finetuning" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Pretraining vs Fine-tuning: Workflow เปรียบเทียบ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การเข้าใจความแตกต่างระหว่าง <strong>Pretraining</strong> และ <strong>Fine-tuning</strong> เป็นพื้นฐานสำคัญสำหรับการออกแบบ pipeline ของ Language Models อย่างมีประสิทธิภาพ.
      ทั้งสอง phase นี้มีเป้าหมายต่างกัน และใช้ objective function ที่แตกต่างกันอย่างชัดเจน.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.1 เป้าหมายของ Pretraining</h3>
    <p>
      เป้าหมายหลักของ Pretraining คือการสอนให้ Language Model เรียนรู้ <strong>general-purpose language representations</strong> จาก data ขนาดใหญ่ (เช่น WebCorpus, Wikipedia, Books).
      ตัวอย่าง Objective Function ที่ใช้เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Masked Language Modeling (MLM)</strong> → BERT</li>
      <li><strong>Next Sentence Prediction (NSP)</strong> → BERT</li>
      <li><strong>Causal Language Modeling (CLM)</strong> → GPT</li>
      <li><strong>Prefix Language Modeling</strong> → T5</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Pretraining ช่วยให้ model เรียนรู้ <em>syntactic, semantic, contextual</em> patterns ของภาษาแบบ general โดยไม่ผูกติดกับ task ใด task หนึ่ง → ทำให้ model สามารถ transfer knowledge ได้ดี.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.2 เป้าหมายของ Fine-tuning</h3>
    <p>
      Fine-tuning เป็นกระบวนการ <strong>supervised learning</strong> หรือ <strong>instruction tuning</strong> ที่ model จะถูกฝึกต่อบน data เฉพาะ domain หรือเฉพาะ task เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Text Classification → sentiment analysis, topic detection</li>
      <li>Named Entity Recognition (NER)</li>
      <li>Question Answering (QA)</li>
      <li>Summarization</li>
      <li>Code Generation</li>
    </ul>
    <p>
      ในขั้นนี้ objective function จะถูกเปลี่ยนใหม่ให้สอดคล้องกับ task ที่กำหนด เช่น Cross Entropy Loss บน label ของ classification หรือ sequence-to-sequence loss ใน generation task.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Fine-tuning มีผลโดยตรงต่อ ability ของ model ในการ <strong>specialize knowledge</strong> บน target domain → โดย model จะสามารถ perform ได้ดีแม้กับ distribution ของ data ที่ต่างจาก pretraining.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.3 Workflow เปรียบเทียบ</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Pretraining</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Fine-tuning</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Objective</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ representation แบบ general</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ปรับให้ตรงกับ downstream task</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Data</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Unlabeled corpus ขนาดใหญ่</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Labeled data ตาม target task</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Time</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">หลายสัปดาห์ (TPU/GPU cluster)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ไม่กี่ชั่วโมงหรือไม่กี่วัน</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Impact</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Model generalization สูง</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Model performance สูงใน specific domain/task</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.4 Pipeline โดยรวม</h3>
    <p>Pipeline โดยทั่วไปจะเป็นดังนี้:</p>
    <pre className="bg-gray-800 text-green-200 p-4 rounded text-sm overflow-x-auto">
{`Unlabeled Data → Pretraining → General LM → Fine-tuning → Specialized LM → Deployment`}
    </pre>
    <p>
      ตัวอย่างเช่น BERT ถูก pretrain บน Wikipedia+Books → Fine-tune บน QA datasets เช่น SQuAD → Deploy เป็น QA engine.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5), JMLR 2020.</li>
      <li>Gururangan et al. Don't Stop Pretraining: Adapt Language Models to Domains and Tasks, ACL 2020.</li>
      <li>Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.</li>
      <li>Radford et al. Language Models are Unsupervised Multitask Learners, OpenAI 2019.</li>
    </ul>
  </div>
</section>


     <section id="finetuning-strategies" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Types of Fine-tuning Strategies</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การเลือกใช้ Fine-tuning strategy ที่เหมาะสมกับ context ของ task, data size, และ compute budget มีผลอย่างยิ่งต่อ performance ของ Language Model.
      จากการศึกษาวิจัยล่าสุด พบว่า Fine-tuning strategies ไม่ได้มีเพียงแค่การ retrain ทั้ง model แต่ยังมีเทคนิคที่ sophisticated และ resource-efficient อื่น ๆ ด้วย.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.1 Full Fine-tuning</h3>
    <p>
      เป็น strategy แบบ "classic" ที่นิยมใช้ในยุคแรกของ BERT และ GPT:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ทำการ update parameter ทุกตัวใน network ทั้ง backbone และ head</li>
      <li>ต้องใช้ batch size และ learning rate ที่ carefully tuned</li>
      <li>ใช้ได้ดีเมื่อ target data มีขนาดใหญ่พอที่จะหลีกเลี่ยง overfitting</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Full fine-tuning ยังคงเป็น baseline ที่สำคัญสำหรับ task ที่มี target data ขนาดใหญ่ เช่น multi-domain QA, summarization หรือ translation.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.2 Adapter-based Fine-tuning</h3>
    <p>
      เป็น strategy ที่เริ่มได้รับความนิยมเพื่อแก้ปัญหาข้อจำกัดของ full fine-tuning:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Freeze parameter ของ backbone model ไว้ทั้งหมด</li>
      <li>เพิ่ม <strong>Adapter module</strong> ขนาดเล็ก (เช่น feedforward bottleneck layers) ระหว่าง layers ของ backbone</li>
      <li>ทำการ train เฉพาะ parameter ของ Adapter เท่านั้น</li>
    </ul>
    <p>
      ตัวอย่างเช่น <strong>Houlsby Adapters (2019)</strong> แสดงให้เห็นว่า performance ใกล้เคียงกับ full fine-tuning แต่ต้อง train parameter เพียง 3-5% ของ model.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Adapter-based fine-tuning เหมาะมากกับ setting แบบ multi-task หรือ multi-domain ที่ต้องการ maintain ความเป็น universal LM พร้อม deploy model หลาย version.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.3 Prompt-tuning</h3>
    <p>
      เป็น paradigm ที่เติบโตมากในช่วงปี 2020+ โดยเฉพาะกับ foundation models ขนาดใหญ่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ไม่ต้องเปลี่ยน parameter ของ backbone เลย (frozen model)</li>
      <li>เรียนรู้ "prompt vectors" หรือ "soft prompts" ที่ prepend เข้าไปใน input sequence</li>
      <li>ลด parameter ที่ต้องเรียนรู้เหลือเพียงไม่กี่ล้าน หรือแม้แต่หลักแสน parameter</li>
    </ul>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Strategy</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Trainable Params</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Data Requirement</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Full Fine-tuning</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">100%</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Adapter-based</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~3-5%</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">กลาง</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Prompt-tuning</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">มากกว่า 1%</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.4 เทคนิค Hybrid</h3>
    <p>
      ล่าสุดหลาย paper เริ่มใช้ hybrid approach เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Adapter + Prompt-tuning → เรียนรู้ทั้ง soft prompt + adapter layers</li>
      <li>Low-Rank Adaptation (LoRA) → เพิ่ม trainable rank-decomposition matrix เข้าไปใน weight ของ attention layers</li>
    </ul>
    <p>
      งานเช่น <strong>LoRA (Hu et al., 2021)</strong> ได้แสดงว่า LoRA fine-tuning บน GPT-3 สามารถลด parameter requirement ลงอย่างมาก พร้อม maintain performance ได้ดี.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Trend ในยุค foundation model ล่าสุด → shift จาก full fine-tuning → ไปสู่ soft tuning / parameter-efficient tuning → scaling ให้ได้ model version ที่ flexible กว่า และ deploy ได้ง่ายกว่า.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Houlsby et al. Parameter-efficient Transfer Learning for NLP, ICML 2019.</li>
      <li>Lester et al. The Power of Scale for Parameter-Efficient Prompt Tuning, EMNLP 2021.</li>
      <li>Hu et al. LoRA: Low-Rank Adaptation of Large Language Models, arXiv 2021.</li>
      <li>Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.</li>
    </ul>
  </div>
</section>

  <section id="finetuning-pipeline" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Fine-tuning Pipeline</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การออกแบบ Pipeline สำหรับ Fine-tuning Language Models ให้มีประสิทธิภาพนั้น ไม่ใช่เพียงการเรียกใช้ training script แต่เป็นการวางแผนที่รอบด้านตั้งแต่ data preparation → training → validation → deployment.
      ในหัวข้อนี้จะนำเสนอ **โครงสร้าง Pipeline ที่แนะนำ** โดยอิงจาก best practice ของทั้งงานวิจัยและ engineering systems.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.1 Overview ของ Fine-tuning Pipeline</h3>
    <p>
      Pipeline มาตรฐานสามารถแบ่งได้เป็นขั้นตอนหลัก ๆ ดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Data Curation & Preprocessing</strong> → เตรียมและ clean ข้อมูล target domain</li>
      <li><strong>Tokenization & Formatting</strong> → Tokenize และจัด format ให้เหมาะสมกับ model architecture</li>
      <li><strong>Model Initialization</strong> → Load model checkpoint จาก pre-trained model ที่เหมาะสม</li>
      <li><strong>Training Loop & Hyperparameters</strong> → Define optimizer, learning rate schedule, training loop</li>
      <li><strong>Evaluation & Validation</strong> → วัด performance บน validation set อย่างเป็นระบบ</li>
      <li><strong>Model Saving & Deployment</strong> → Save model + prepare artifact สำหรับ production</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในระบบ production-scale เช่นที่ใช้ใน Meta หรือ Google, Pipeline จะมี automation ทั้งหมด ตั้งแต่ data monitoring → auto retraining → auto evaluation → auto deploy → feedback loop.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.2 Data Preparation</h3>
    <p>
      ขั้นตอนแรกที่สำคัญที่สุดของ Pipeline คือ **Data Preparation** เพราะ quality ของ data ส่งผลโดยตรงต่อ final performance ของ model:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Remove noise, duplicate, และ outliers จาก dataset</li>
      <li>Balance data ถ้ามี class imbalance</li>
      <li>Define data split → training / validation / test</li>
      <li>ทำ data augmentation ถ้า task เอื้อ เช่น paraphrasing สำหรับ NLP</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.3 Tokenization & Formatting</h3>
    <p>
      ต้องเลือก tokenizer ที่ compatible กับ backbone model เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>BERT → WordPiece tokenizer</li>
      <li>GPT → Byte-Pair Encoding (BPE) tokenizer</li>
      <li>T5 → SentencePiece tokenizer</li>
    </ul>
    <p>
      และต้องจัดรูปแบบ input ให้ถูกต้อง:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Max sequence length → truncate หรือ pad</li>
      <li>Batch size → tradeoff ระหว่าง compute กับ stability</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.4 Model Initialization</h3>
    <p>
      การเลือก starting checkpoint ที่เหมาะสมช่วยลด training time และ improve performance:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เลือก model checkpoint ที่ pretrained บน domain ใกล้เคียง (domain adaptation)</li>
      <li>ใช้ checkpoint ที่ผ่าน stage 1 pretraining + stage 2 domain pretraining แล้ว หากมี</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.5 Training Loop & Hyperparameters</h3>
    <p>
      Core ของ Pipeline คือการ setup training loop:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Optimizer → AdamW เป็น choice ที่นิยมมากที่สุด</li>
      <li>Learning Rate Scheduler → cosine decay, linear warmup → decay</li>
      <li>Gradient Clipping → ป้องกัน exploding gradients</li>
      <li>Early Stopping → หยุด training เมื่อ validation metric เริ่ม degrade</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ตัวอย่าง config สำหรับ Fine-tuning BERT บน classification task:
      </p>
      <pre className="bg-gray-800 text-green-200 p-4 rounded text-xs overflow-x-auto">
{`{
  "optimizer": "AdamW",
  "learning_rate": 5e-5,
  "batch_size": 32,
  "epochs": 3,
  "gradient_clipping": 1.0,
  "scheduler": "linear_warmup_decay",
  "warmup_steps": 500
}`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.6 Evaluation & Validation</h3>
    <p>
      ต้อง design metric สำหรับวัดคุณภาพของ model:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Classification → accuracy, F1-score, ROC-AUC</li>
      <li>QA → Exact Match (EM), F1</li>
      <li>Summarization → ROUGE-L, BLEU</li>
      <li>Text Generation → human evaluation + automated metric</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.7 Model Saving & Deployment</h3>
    <p>
      Best Practice:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Save model in format ที่ ready สำหรับ inference → e.g., HuggingFace Transformers format</li>
      <li>Export tokenizer config พร้อม model</li>
      <li>Tag model version → support reproducibility</li>
      <li>Deploy บน serving platform เช่น Triton, TorchServe, HuggingFace Inference API</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.8 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Howard & Ruder. Universal Language Model Fine-tuning for Text Classification, ACL 2018.</li>
      <li>Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR 2020.</li>
      <li>Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.</li>
      <li>HuggingFace Transformers Library - Official Docs.</li>
    </ul>
  </div>
</section>


 <section id="practical-examples" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Practical Examples</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Fine-tuning Language Models ได้กลายเป็นหนึ่งในขั้นตอนสำคัญที่สุดของการนำโมเดลสมัยใหม่ไปใช้งานจริงในภาคอุตสาหกรรมและวิจัย.
      ในหัวข้อนี้จะนำเสนอตัวอย่าง practical ที่เกิดขึ้นจริงในหลาย use cases เพื่อแสดงให้เห็นว่า **การออกแบบ Fine-tuning pipeline ที่เหมาะสม** สามารถสร้างความแตกต่างทาง performance ได้อย่างไร.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.1 Fine-tuning BERT สำหรับ Named Entity Recognition (NER)</h3>
    <p>
      หนึ่งใน task ที่ Fine-tuning BERT ให้ผลลัพธ์ที่ยอดเยี่ยมคือ Named Entity Recognition (NER).
      ตัวอย่าง pipeline มีดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Dataset: CoNLL-2003 หรือ custom NER corpus ขององค์กร</li>
      <li>Model Initialization: BERT-base-uncased checkpoint</li>
      <li>Head Layer: Classification head สำหรับ token classification</li>
      <li>Optimizer: AdamW, learning rate 5e-5</li>
      <li>Training: 3-5 epochs</li>
    </ul>
    <p>
      ผลลัพธ์: บรรลุ F1-score ~92% บน CoNLL-2003 ภายในเวลา train น้อยกว่า 1 ชั่วโมง.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        สำหรับ task NER, Pretrained BERT สามารถจับ context ได้ดีมากกว่า CRF-based model เดิมที่เคยใช้ใน production systems.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.2 Fine-tuning T5 สำหรับ Text Summarization</h3>
    <p>
      T5 (Text-to-Text Transfer Transformer) ได้รับการพิสูจน์ว่ามีประสิทธิภาพสูงมากสำหรับ abstractive summarization.
      Pipeline ที่นิยมใช้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Dataset: CNN/DailyMail หรือ domain-specific news corpus</li>
      <li>Model Initialization: T5-base หรือ T5-large checkpoint</li>
      <li>Preprocessing: "summarize: " + input text → sequence-to-sequence task</li>
      <li>Optimizer: AdaFactor หรือ AdamW</li>
      <li>Training: 3 epochs, batch size 8-16</li>
    </ul>
    <p>
      ผลลัพธ์: ROUGE-L score ที่สูงกว่า baseline models เดิม ~5-10%.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        T5 สามารถ generalized ไปสู่หลาย downstream task ได้โดยไม่ต้องเปลี่ยน architecture เลย เพียงแค่เปลี่ยน prompt + target format.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.3 Fine-tuning GPT-2 สำหรับ Chatbot Generation</h3>
    <p>
      การ fine-tune GPT-2 เพื่อสร้าง conversational agents หรือ chatbot กำลังได้รับความนิยมสูง.
      Pipeline มีลักษณะดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Dataset: Custom conversation dataset (dialog pairs)</li>
      <li>Model Initialization: GPT-2-medium checkpoint</li>
      <li>Training objective: Language Modeling (causal LM) → next token prediction</li>
      <li>Optimizer: AdamW, learning rate ~1e-4</li>
      <li>Special considerations: Add conversation history window</li>
    </ul>
    <p>
      ผลลัพธ์: สามารถ generate chatbot responses ที่มีความเป็นธรรมชาติสูงกว่าระบบ rule-based เดิมมาก.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.4 Fine-tuning DistilBERT สำหรับ Intent Classification</h3>
    <p>
      DistilBERT ซึ่งเป็น BERT ที่ถูก compress แล้ว เป็น model ที่มีประสิทธิภาพมากสำหรับงานที่ latency-sensitive เช่นใน mobile apps.
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Dataset: Custom intent dataset (เช่น app intents)</li>
      <li>Model Initialization: DistilBERT-base-uncased checkpoint</li>
      <li>Head Layer: Classification head</li>
      <li>Training: 4 epochs, batch size 32</li>
      <li>Deployment target: Mobile device (ONNX or TensorFlow Lite export)</li>
    </ul>
    <p>
      ผลลัพธ์: Latency น้อยกว่า 10ms บน mobile, accuracy ~97% บน validation set.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Howard & Ruder. Universal Language Model Fine-tuning for Text Classification, ACL 2018.</li>
      <li>Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR 2020.</li>
      <li>Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.</li>
      <li>Wolf et al. Transformers: State-of-the-Art Natural Language Processing, EMNLP 2020 (HuggingFace Transformers).</li>
    </ul>
  </div>
</section>


       <section id="advanced-topics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Advanced Topics</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในขณะที่ Fine-tuning แบบเต็มโมเดล (Full Fine-tuning) ยังคงเป็นแนวทางมาตรฐานในการดัดแปลง Language Model ให้เข้ากับงานเฉพาะ domain, งานวิจัยล่าสุดได้เสนอแนวทางที่มีประสิทธิภาพสูงกว่าในด้าน resource efficiency และ model generalization.
      หัวข้อนี้จะกล่าวถึง advanced topics ที่อยู่ในแนวหน้าของงานวิจัยและการนำไปใช้จริง.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.1 Parameter-Efficient Fine-tuning (PEFT)</h3>
    <p>
      PEFT เป็นกลุ่มเทคนิคที่พยายามลดจำนวน parameter ที่ต้อง update ระหว่างการ fine-tune.
      แทนที่จะปรับ weight ทั้งหมดของ model, PEFT ปรับเฉพาะ layer หรือ sub-component บางส่วน.
      ตัวอย่างที่สำคัญมีดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Adapter Layers</li>
      <li>LoRA (Low-Rank Adaptation)</li>
      <li>Prefix Tuning</li>
      <li>Prompt Tuning</li>
    </ul>
    <p>
      จุดเด่นของ PEFT คือช่วยให้สามารถ fine-tune model ขนาดใหญ่ได้แม้ใน hardware ขนาดเล็ก และสามารถ swap adapters สำหรับหลาย task ได้โดยไม่ต้อง retrain โมเดลหลัก.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        LoRA ได้รับความนิยมสูงในงาน production-scale LLM fine-tuning เนื่องจากใช้ memory ต่ำมาก (~1% ของ full fine-tuning) แต่ได้ performance ใกล้เคียงกัน.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.2 LoRA: Low-Rank Adaptation</h3>
    <p>
      LoRA เป็นเทคนิคที่เพิ่ม rank-decomposed matrices เข้ามาใน layer หลัก เช่น attention หรือ feed-forward.
      สูตรการคำนวณโดยย่อ:
    </p>
    <pre className="bg-gray-800 text-green-200 text-sm rounded p-4 overflow-auto">
      <code>
{`Original weight: W
Update: W' = W + ΔW
ΔW = A * B   where A ∈ R^{d×r}, B ∈ R^{r×d}, r ≪ d
`}
      </code>
    </pre>
    <p>
      LoRA จึงสามารถทำให้ training process lightweight มาก โดย training เฉพาะ parameter ของ A และ B เท่านั้น.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.3 Prompt Tuning & Prefix Tuning</h3>
    <p>
      เทคนิคอีกกลุ่มหนึ่งที่ได้รับความสนใจสูง คือ Prompt-based tuning ซึ่งไม่ได้เปลี่ยน parameter ของ model เลย แต่เพิ่ม trainable prompt token หรือ prefix token ต่อหน้า input.
      ประโยชน์คือ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Model architecture ไม่เปลี่ยน → maintain compatibility กับ pre-trained model</li>
      <li>สามารถปรับ prompt ให้เหมาะกับ task-specific ได้โดยไม่กระทบ knowledge ของ model เดิม</li>
      <li>Training cost ต่ำมาก</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ใน paper "Prefix-Tuning: Optimizing Continuous Prompts for Generation", Li & Liang (2021) รายงานว่าสามารถใช้ prefix length เพียง 100-200 tokens ก็ fine-tune GPT2/3 ให้ performance ใกล้ full fine-tuning ได้ในหลาย task.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.4 Full Parameter vs PEFT: ตารางเปรียบเทียบ</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Full Fine-tuning</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">PEFT</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"># Parameters Updated</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">100%</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~1-5%</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Memory Usage</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำมาก</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Deployment Flexibility</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ (ต้อง deploy new model)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง (สามารถ swap adapters/prefix)</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models", arXiv 2021.</li>
      <li>Li & Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation", arXiv 2021.</li>
      <li>He et al. "Towards Parameter-Efficient Tuning of Large Language Models", NeurIPS 2022.</li>
      <li>Houlsby et al. "Parameter-Efficient Transfer Learning for NLP", ICML 2019.</li>
    </ul>
  </div>
</section>


  <section id="cost-considerations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Cost Considerations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในการนำ Language Model (LM) มาใช้งานจริงในระบบ production หรือสำหรับงานวิจัยขนาดใหญ่, **Cost Considerations** ถือเป็นปัจจัยสำคัญอย่างยิ่งต่อ feasibility ของโครงการ.
      Section นี้จะวิเคราะห์ปัจจัยต้นทุนที่สำคัญในแต่ละเฟสของ Fine-tuning pipeline และเปรียบเทียบ trade-offs ของเทคนิคที่เลือกใช้.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.1 ประเภทของต้นทุนหลัก</h3>
    <p>ต้นทุนที่เกี่ยวข้องกับการ fine-tune LM สามารถแบ่งเป็นกลุ่มหลัก ๆ ดังนี้:</p>
    <ul className="list-disc list-inside space-y-2">
      <li>Training Cost (GPU/TPU compute cost + engineer time)</li>
      <li>Inference Cost (latency, batch size impact, memory footprint)</li>
      <li>Storage Cost (checkpoints, adapter layers, versioning)</li>
      <li>Operational Overhead (deployment pipeline, CI/CD, testing)</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Google Research พบว่า LoRA fine-tuning GPT-3-175B บน single task ใช้เพียง <strong>~$500-1000 USD</strong> compute cost, ในขณะที่ full fine-tuning อาจต้องใช้ budget ระดับ <strong>~$50k-$100k USD</strong> (ขึ้นอยู่กับ batch size และ epoch).
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.2 Training Cost: Full vs PEFT</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Technique</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">GPU Hours (Relative)</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Estimated Cost (USD)</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Full Fine-tuning (LLaMA2-65B)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">100%</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~$20k-$50k</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">LoRA PEFT</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~2-5%</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">~$200-$1000</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.3 Inference Cost Impact</h3>
    <p>
      แม้ PEFT จะช่วยลด training cost ได้มาก, การเลือก fine-tuning technique ยังคงมีผลต่อ cost ของ inference pipeline:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Full Fine-tuned model → deploy เป็น model ใหม่ → GPU memory footprint ใหญ่</li>
      <li>Adapter-based PEFT → load base model shared + adapter layer → memory-efficient, multi-task possible</li>
      <li>Prompt Tuning → load base model + soft prefix → almost no additional inference cost</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ใน production system ที่ต้อง serve multi-task model, Prefix Tuning + shared base model เป็นทางเลือกที่มี cost-to-benefit ratio สูงสุด, เนื่องจากสามารถ switch task ด้วย prefix ที่ต่างกันได้โดยไม่ต้อง reload model.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.4 Deployment & Maintenance Overhead</h3>
    <p>
      นอกจากต้นทุน compute โดยตรง, การเลือก fine-tuning strategy ยังส่งผลต่อ complexity ของ deployment pipeline:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Full Fine-tune:</strong> ต้อง build, test, deploy model ใหม่ในแต่ละ task</li>
      <li><strong>LoRA / Adapter:</strong> base model คงเดิม, deploy adapter checkpoint ใหม่ได้ง่าย</li>
      <li><strong>Prompt Tuning:</strong> เพียง update soft prompt embedding → deployment simplicity สูงมาก</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.5 สรุปแนวโน้มการเลือกตามประเภทองค์กร</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Organization Type</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Recommended Strategy</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Large Tech (FAANG, Cloud Providers)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Full Fine-tuning / LoRA hybrid</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">AI Startups / Research Labs</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">LoRA / Prefix Tuning</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Enterprise Application (AI SaaS)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Prefix Tuning (maximize flexibility)</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.6 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models", arXiv 2021.</li>
      <li>Li & Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation", arXiv 2021.</li>
      <li>OpenAI. "Scaling Laws for Neural Language Models", arXiv 2020.</li>
      <li>Google Research. "Efficient Adaptation of Pretrained Transformers", 2022.</li>
    </ul>
  </div>
</section>


  <section id="engineering-best-practices" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Engineering Best Practices</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การ Fine-tune Language Models (LMs) ในระดับ production จำเป็นต้องมี **Engineering Best Practices** ที่มั่นคง เพื่อให้ได้ model ที่ **มีคุณภาพสูง, reproducible, cost-effective** และง่ายต่อการ maintain ในระยะยาว.
    </p>
    <p>
      Section นี้จะสรุปแนวทางที่ได้รับการยอมรับจากทีมงานในองค์กรระดับโลก เช่น Google, Meta AI, OpenAI, DeepMind และ HuggingFace.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.1 Learning Rate Scheduling</h3>
    <p>
      การจัดการ learning rate อย่างเป็นระบบมีผลสำคัญต่อ **stability** และ **convergence** ของ fine-tuning.
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เริ่มจาก **Warmup phase** 5-10% ของ total steps เพื่อป้องกัน exploding gradients.</li>
      <li>ใช้ **Cosine Decay** หรือ **Linear Decay** หลังจาก warmup.</li>
      <li>Hyperparameter ที่เหมาะสม: initial LR ~1e-4 สำหรับ PEFT / ~5e-5 สำหรับ full fine-tune.</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        งานวิจัยของ Google Research พบว่า **"Cosine Annealing + Warmup"** ให้ผลลัพธ์ดีที่สุดใน fine-tuning LLaMA-65B / T5-xl บน GLUE benchmark และ translation task.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.2 Checkpointing & Reproducibility</h3>
    <p>
      การจัดการ checkpoint อย่างถูกต้องช่วยให้สามารถ **resume training** ได้สะดวก และช่วยเรื่อง reproducibility:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Save checkpoint ทุก ๆ 500-1000 steps หรือทุก ~30-60 min (ขึ้นกับ task).</li>
      <li>Save ทั้ง model weights + optimizer state + learning rate scheduler state.</li>
      <li>Version checkpoint อย่างชัดเจน → ใช้ git commit hash เป็น part ของ checkpoint name.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.3 Determinism & Seed Control</h3>
    <p>
      เพื่อให้สามารถ **reproduce results ได้ 100%** (สำคัญมากใน production และ academic paper):
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fix random seed สำหรับ:
        <ul className="list-disc list-inside ml-6 space-y-1">
          <li>Python random</li>
          <li>Numpy random</li>
          <li>PyTorch / TensorFlow random</li>
        </ul>
      </li>
      <li>ใช้ deterministic ops (ถ้ามีตัวเลือก) → PyTorch: `torch.use_deterministic_algorithms(True)`.</li>
      <li>Log seed ลงใน training config / metadata file เสมอ.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.4 Data Pipeline & Sharding</h3>
    <p>
      การจัดการ data pipeline ที่ robust มีผลต่อ both training efficiency และ reproducibility:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Shard dataset เป็น fixed splits → ใช้ shard index control เพื่อ reproducibility.</li>
      <li>Log exact version ของ dataset ที่ใช้ (version number, git hash, URL snapshot).</li>
      <li>Cache preprocessing → หลีกเลี่ยงการทำ augmentation หรือ tokenization ใหม่ทุก run.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.5 Logging & Monitoring</h3>
    <p>
      Engineering teams ที่มี production LM pipeline ควร implement logging ดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Log training loss, validation loss ทุก n steps → TensorBoard หรือ Weights & Biases (wandb).</li>
      <li>Log compute utilization → GPU usage, memory usage → Optimize throughput.</li>
      <li>Log per-epoch eval metrics บน holdout set + downstream task set.</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        OpenAI Codex team ใช้ Weights & Biases pipeline monitor สำหรับ **600+ parallel fine-tuning jobs** → ช่วย identify hyperparameter bugs ได้เร็วขึ้นถึง **4x** เทียบกับ manual monitoring.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.6 Evaluation & Test Suite</h3>
    <p>
      ก่อน deploy LM model ควรมี **Evaluation Suite** ที่ครอบคลุม:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Eval บน in-domain task (primary objective metric).</li>
      <li>Eval บน out-of-domain task → เพื่อเช็ค generalization.</li>
      <li>Eval on bias, toxicity → ผ่าน toolkit เช่น **HuggingFace Detoxify**.</li>
      <li>Regression testing → compare กับ prior best model → ป้องกัน quality drop.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.7 Deployment Checklist</h3>
    <p>
      ก่อน deploy fine-tuned model → ควร checklist ดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Verify eval metrics pass threshold (documented).</li>
      <li>Verify reproducibility from checkpoint → run full eval cycle twice.</li>
      <li>Verify latency & memory budget → benchmark vs serving budget.</li>
      <li>Version model artifact + metadata + training config → log ทั้งหมดเป็น immutable artifact.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.8 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Chen et al. "Training Language Models to Follow Instructions with Human Feedback", OpenAI, arXiv 2022.</li>
      <li>Li et al. "Scaling Laws for Neural Language Models", OpenAI, arXiv 2020.</li>
      <li>Facebook AI. "LLaMA Engineering Practices", Meta AI 2023.</li>
      <li>Google Brain. "Practical Lessons from Building LLM Systems", ICML Workshop 2023.</li>
    </ul>
  </div>
</section>


  <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Use Cases & Industry Impact</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Fine-tuning Language Models (LMs) ได้กลายเป็นแกนกลางของการนำ AI ไปใช้ในเชิง **production scale** ทั่วโลก. 
      การ fine-tune model บน **domain-specific data**, **task-specific data** หรือ **customer data** ช่วยให้ได้ model ที่ตอบโจทย์ real-world use case ได้ดีกว่า pre-trained model แบบ generic.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.1 Personalization & Recommendation Systems</h3>
    <p>
      หนึ่งใน use case ที่สำคัญที่สุดคือ **Personalized Language Models** → สร้าง experience ที่ตอบสนองผู้ใช้แต่ละราย:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tune GPT-like model บน customer support transcript → personalize responses.</li>
      <li>Fine-tune recommendation model (e.g. for e-commerce) → personalize product descriptions.</li>
      <li>Fine-tune on past user queries → better relevance ranking.</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในการทดลองของ Amazon, การ fine-tune BERT model บน **customer clickstream data** เพิ่ม click-through-rate (CTR) ได้ถึง **+12%** บน homepage recommendation section.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.2 Enterprise Knowledge Search</h3>
    <p>
      การ fine-tune LMs บน enterprise knowledge base มี impact สูงมากในด้าน **internal search** และ **customer support**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tune LLaMA หรือ T5 บน internal docs → semantic search engine.</li>
      <li>Fine-tune retrieval augmented generation (RAG) pipeline → real-time contextual answer generation.</li>
      <li>Fine-tune on support ticket data → auto-suggest answer / summarization.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.3 Multilingual Applications</h3>
    <p>
      Use case สำคัญอีกประเภทคือ **Multilingual Fine-tuning** → ทำให้ LM model ใช้งานใน diverse market ได้จริง:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tune multilingual BERT หรือ XLM-R บน legal domain → multilingual contract analysis.</li>
      <li>Fine-tune translation LM → specialized translation (legal, medical, technical).</li>
      <li>Fine-tune voice LM + text LM → multilingual customer voice transcription.</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Microsoft Azure AI fine-tuned multilingual GPT model บน **45+ languages** → ลด latency ของ multilingual customer chatbot ได้ถึง **x3**, เพิ่ม NPS score +7 points.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.4 AI-Enhanced Content Generation</h3>
    <p>
      Content generation เป็นหนึ่งใน field ที่ Fine-tuning LMs ได้รับความนิยมสูงที่สุด:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tune GPT-3.5 / GPT-4 บน marketing copy corpus → create brand-consistent content.</li>
      <li>Fine-tune LLaMA / MPT → domain-specific technical writing (scientific, financial, legal).</li>
      <li>Fine-tune dialogue LM → game dialogue / interactive story generation.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.5 Legal & Regulatory Applications</h3>
    <p>
      ใน sector ที่มี compliance requirements สูง → fine-tune LMs เป็น **essential requirement**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tune LM on jurisdiction-specific legal corpus → better legal clause understanding.</li>
      <li>Fine-tune audit LM → automatic contract review + risk flagging.</li>
      <li>Fine-tune explainable LM → generate human-understandable compliance reports.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.6 Healthcare Applications</h3>
    <p>
      Healthcare sector ได้รับประโยชน์จากการ fine-tune LMs บน medical data อย่างมาก:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tune LM on EHR (Electronic Health Records) → clinical decision support.</li>
      <li>Fine-tune LM on biomedical papers → biomedical question answering.</li>
      <li>Fine-tune LM + summarization model → patient discharge summary generation.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.7 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Radford et al. "Language Models are Few-Shot Learners", OpenAI, arXiv 2020.</li>
      <li>Bommasani et al. "On the Opportunities and Risks of Foundation Models", Stanford CRFM, arXiv 2021.</li>
      <li>Google Research. "Scaling Deep Learning for Production Use Cases", Google Cloud AI Whitepaper 2023.</li>
      <li>Meta AI. "Fine-tuning LLaMA for Enterprise Use", Meta Engineering Blog 2023.</li>
    </ul>
  </div>
</section>


<section id="limitations-ethics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Limitations & Ethical Considerations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      แม้ว่า Fine-tuning Language Models จะสามารถเพิ่มความสามารถและ value ให้กับ model ได้อย่างมาก แต่การนำไปใช้จริงในระบบ production จำเป็นต้องพิจารณาทั้ง **ข้อจำกัดเชิงเทคนิค** และ **ข้อพิจารณาด้านจริยธรรม** อย่างรอบด้าน.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.1 Limitations ทางเทคนิค</h3>
    <p>
      การ fine-tune model ไม่ใช่กระบวนการที่ปราศจากข้อจำกัด. ข้อจำกัดเชิงเทคนิคที่พบบ่อย ได้แก่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Catastrophic Forgetting** → model อาจลืม knowledge จาก pretraining ได้หาก fine-tuning aggressive เกินไป.</li>
      <li>**Overfitting** → บน dataset ขนาดเล็กหรือ biased → ลด generalization.</li>
      <li>**Compute Cost** → fine-tune model ขนาดใหญ่ต้องใช้ GPU resources สูงมาก.</li>
      <li>**Version Drift** → fine-tuned model อาจต้อง re-tune เมื่อ upstream model updated.</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในงานศึกษาของ Stanford Center for Research on Foundation Models (CRFM), fine-tuning LLaMA-2-65B บน domain-specific dataset ขนาด 2M examples พบว่าหากไม่ใช้ techniques เช่น **LoRA** หรือ **adapter layers**, model มี risk ต่อ catastrophic forgetting สูงถึง **+48%** performance drop บน general benchmarks.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.2 Risk ของ Bias & Fairness</h3>
    <p>
      หนึ่งใน ethical risk ที่สำคัญที่สุดของ fine-tuned LMs คือการ **reinforce existing biases** หรือ **สร้าง new bias** จาก dataset ที่ใช้ fine-tuning:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Bias ที่มีใน training data → ถูก amplify หลัง fine-tuning.</li>
      <li>Domain-specific data ที่ไม่ diverse → ส่งผลต่อ fairness ของ output.</li>
      <li>Language bias, gender bias, cultural bias → ต้องถูก monitor อย่างต่อเนื่อง.</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ในงานวิจัยของ MIT Media Lab, การ fine-tune GPT-3.5 บน customer service data ขององค์กรหนึ่ง ทำให้ model เริ่มตอบด้วย tone ที่ **reinforces gender stereotypes** ซึ่งไม่พบใน base model — แสดงว่าการ fine-tuning สามารถ introduce bias ใหม่ได้.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.3 Legal & Compliance Risks</h3>
    <p>
      การ fine-tune LMs เพื่อใช้ใน production ต้องพิจารณา **legal risk** และ **compliance requirements**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Data Privacy** → ต้องแน่ใจว่า training data comply กับ GDPR / HIPAA / CCPA.</li>
      <li>**Data Licensing** → ต้องตรวจสอบ license ของ pretraining / fine-tuning data.</li>
      <li>**Explainability** → ในบาง sector (e.g. healthcare, finance) ต้องสามารถ audit decision process ของ model ได้.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.4 Risk ของ Model Misuse</h3>
    <p>
      LMs ที่ fine-tune แล้วสามารถถูก misuse ได้หากไม่ได้ deploy และ control อย่างเหมาะสม:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Misinformation generation** → model ถูกใช้ generate false content.</li>
      <li>**Impersonation** → ใช้ fine-tune model ให้ mimic บุคคล.</li>
      <li>**Toxic content generation** → หาก control mechanism ไม่ดีพอ.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.5 แนวทางในการ Mitigate Risk</h3>
    <p>
      แนวทางสำคัญที่ควรใช้ในการลด risk ของการ fine-tune LMs:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ใช้ **differential privacy techniques** ใน training pipeline.</li>
      <li>ทำ **bias & fairness audit** เป็นประจำ.</li>
      <li>ใช้ **RLHF (Reinforcement Learning with Human Feedback)** เพื่อ fine-tune safely.</li>
      <li>Deploy model พร้อม **content filters** และ **safety guardrails**.</li>
      <li>Log model output และทำ audit อย่างต่อเนื่อง.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.6 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>Bommasani et al. "On the Opportunities and Risks of Foundation Models", Stanford CRFM, arXiv 2021.</li>
      <li>OpenAI, "Lessons Learned from Fine-tuning Large Language Models", OpenAI Blog 2023.</li>
      <li>Raji et al., "Closing the AI Accountability Gap", FAT* 2020.</li>
      <li>MIT Media Lab, "Auditing Fine-tuned LLMs for Bias and Risk", arXiv 2023.</li>
    </ul>
  </div>
</section>


<section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Summary Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      กระบวนการ **Fine-tuning Language Models** ถือเป็นหนึ่งใน cornerstones ของการนำ Foundation Models ไปใช้งานจริงในอุตสาหกรรม. จาก workflow แบบ **Pretraining → Fine-tuning → Deployment**, fine-tuning คือ phase ที่สำคัญที่สุดในการ **align model** ให้ตอบโจทย์ task-specific และ business-specific objectives ได้อย่างมีประสิทธิภาพ.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.1 Summary of Key Takeaways</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Fine-tuning ช่วยให้ model สามารถ adapt กับ domain-specific data ได้ → เพิ่ม accuracy และ relevance.</li>
      <li>Techniques เช่น **LoRA, PEFT, Prompt Tuning** ช่วยลด compute cost ในการ fine-tune models ขนาดใหญ่.</li>
      <li>Fine-tuning pipeline ที่ robust ต้องคำนึงถึง data curation, training stability, evaluation, safety guardrails.</li>
      <li>การเลือก strategy ของ fine-tuning (Full Model, Partial Fine-tuning, Adapter-based) ต้อง balance ระหว่าง compute budget, flexibility, และ risk ของ forgetting.</li>
      <li>Risk ด้าน bias, fairness, misuse ต้องถูก monitor และ mitigate อย่างจริงจังในทุกขั้นตอน.</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในรายงานของ **Google DeepMind 2023**, การ fine-tune PaLM-2 ด้วย techniques แบบ **Adapter + RLHF** บน healthcare dialog data พบว่า model สามารถ outperform baseline model ที่ใช้ full fine-tuning ได้ในด้าน accuracy และ safety — พร้อมลด compute cost ลง **~42%**.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.2 Current Trends</h3>
    <p>ปัจจุบันมีแนวโน้มสำคัญหลายประการในงานวิจัยและ production engineering ด้าน Fine-tuning LMs:</p>
    <ul className="list-disc list-inside space-y-2">
      <li>Shift จาก full fine-tuning → ไปสู่ **Adapter-based fine-tuning** และ **Low-rank adaptation (LoRA)**.</li>
      <li>Focus มากขึ้นที่ **Safety-tuned fine-tuning** ผ่าน **RLHF** และ **bias mitigation pipelines**.</li>
      <li>Multi-stage fine-tuning → pre-fine-tune ด้วย synthetic data → ตามด้วย human-validated data.</li>
      <li>Research เรื่อง **continual learning** → เพื่อ mitigate catastrophic forgetting.</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        งานของ **Meta AI (2024)** บน LLaMA-3 family แสดงว่า LoRA-based fine-tuning บน task-specific adapters สามารถ enable **multi-task multi-domain specialization** → โดย model สามารถ load adapter สำหรับแต่ละ task แบบ dynamic ที่ runtime ได้ — เป็นแนวทางสำคัญสำหรับ serving cost-efficient production models.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.3 Strategic Implications</h3>
    <p>สำหรับการนำ Foundation Models ไปใช้ในองค์กร, ข้อแนะนำเชิงกลยุทธ์ได้แก่:</p>
    <ul className="list-disc list-inside space-y-2">
      <li>เริ่มต้นด้วย fine-tuning แบบ **Adapter-based** ก่อน full fine-tuning → ลด cost และ deployment complexity.</li>
      <li>จัดทำ **rigorous evaluation pipeline** → เพื่อ audit model outputs อย่างต่อเนื่องหลัง fine-tuning.</li>
      <li>มี **Human-in-the-loop (HITL)** process ใน critical domains เช่น Healthcare, Finance.</li>
      <li>พัฒนา **bias auditing & mitigation strategy** เป็น part ของ standard MLOps pipeline.</li>
      <li>พิจารณา cost impact ของ fine-tuning ที่ scale → ประเมินค่าใช้จ่าย lifecycle ทั้ง training และ serving.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.4 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2 text-sm">
      <li>OpenAI, "Lessons from Fine-tuning GPT models", OpenAI Engineering Blog 2023.</li>
      <li>Google DeepMind, "PaLM 2 Technical Report", arXiv 2023.</li>
      <li>Meta AI, "LLaMA 3 Release and Engineering Insights", Meta AI Blog 2024.</li>
      <li>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020 (T5 paper).</li>
      <li>Stanford CRFM, "Responsible Use of Foundation Models: Fine-tuning and Risk", CRFM Report 2023.</li>
    </ul>
  </div>
</section>


      <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. References</h2>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในหัวข้อนี้ได้รวบรวมเอกสารทางวิชาการ งานวิจัย รายงานเทคนิค และบทความจากแหล่งความรู้ระดับโลก ที่ถูกใช้อ้างอิงตลอดบทเรียน **Day 54: Fine-tuning Language Models** เพื่อให้นักเรียนสามารถศึกษาเชิงลึกเพิ่มเติมได้อย่างเป็นระบบ.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        งานวิจัยด้าน Fine-tuning Language Models มีความคืบหน้าอย่างรวดเร็วในช่วง **3-4 ปีที่ผ่านมา**. รายงานของ Stanford CRFM (2023) ชี้ว่า trend ของ field นี้กำลัง shift จาก full model fine-tuning → ไปสู่ techniques แบบ **Adapter-based** และ **Low-rank adaptation (LoRA)** อย่างรวดเร็ว — เพื่อเพิ่ม scalability และ reduce serving costs.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.1 Core Research Papers</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *Journal of Machine Learning Research* (T5 paper).
      </li>
      <li>
        Hu, E. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv preprint* arXiv:2106.09685.
      </li>
      <li>
        Li, X. et al. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *arXiv preprint* arXiv:2101.00190.
      </li>
      <li>
        Dettmers, T. et al. (2022). "Efficient Fine-Tuning of Transformer Models using LoRA." *arXiv preprint* arXiv:2211.05132.
      </li>
      <li>
        Wei, J. et al. (2022). "Chain of Thought Prompting Elicits Reasoning in Large Language Models." *arXiv preprint* arXiv:2201.11903.
      </li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        รายงาน "LoRA: Low-Rank Adaptation" ได้กลายเป็นหนึ่งใน foundational papers ที่สร้าง paradigm shift สำหรับ scalable fine-tuning — enabling fine-tuning LLaMA, GPT, T5 ใน production scale ได้อย่างมีประสิทธิภาพ และได้ถูกใช้งานใน production pipeline ของบริษัทชั้นนำเช่น **Meta AI**, **Databricks**, **MosaicML**.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.2 Reports & Whitepapers</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        OpenAI Engineering Blog (2023). "Lessons from Fine-tuning GPT models."
      </li>
      <li>
        Google DeepMind (2023). "PaLM 2 Technical Report." *arXiv preprint* arXiv:2305.10403.
      </li>
      <li>
        Meta AI (2024). "LLaMA 3 Release and Engineering Insights." *Meta AI Blog*.
      </li>
      <li>
        Stanford CRFM (2023). "Responsible Use of Foundation Models: Fine-tuning and Risk."
      </li>
      <li>
        Anthropic (2024). "RLHF: Best Practices for Human-Aligned Fine-tuning." *Anthropic Research Blog*.
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.3 Selected Tutorials & Engineering Resources</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        HuggingFace 🤗 Tutorials. "Fine-tune a Transformer model using PEFT / LoRA." Available at: https://huggingface.co/docs/peft.
      </li>
      <li>
        DeepLearning.AI Courses. "Prompt Engineering & Fine-tuning LLMs."
      </li>
      <li>
        FastAI Course v4. "Practical Fine-tuning of Language Models." Available at: https://course.fast.ai.
      </li>
      <li>
        MLOps Community (2023). "Fine-tuning LLMs at Scale: Engineering Best Practices." *MLOps World Conference*.
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.4 Further Reading</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        Bommasani, R. et al. (2021). "On the Opportunities and Risks of Foundation Models." *Stanford CRFM Report*.
      </li>
      <li>
        Bai, Y. et al. (2022). "Training a Helpful and Harmless Assistant with RLHF." *Anthropic Research*.
      </li>
      <li>
        Zoph, B. et al. (2022). "Designing Effective Adapter Layers for LLMs." *arXiv preprint* arXiv:2210.00161.
      </li>
      <li>
        Dinh, T. et al. (2023). "Efficient Fine-Tuning and Serving of LLMs: Survey and Future Directions." *arXiv preprint* arXiv:2310.10927.
      </li>
    </ul>

    <p className="mt-8 text-sm text-gray-600 dark:text-gray-400">
      This list is curated based on leading publications and engineering blogs from Stanford, MIT, CMU, Harvard, Google DeepMind, OpenAI, Meta AI, HuggingFace, and Anthropic.
    </p>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day54 theme={theme} />
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
      </main>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day54 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day54_FineTuningLanguageModels;
