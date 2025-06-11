import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day53 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day53";
import MiniQuiz_Day53 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day53";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day53_TransformerArchitectures = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day53_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day53_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day53_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day53_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day53_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day53_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day53_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day53_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day53_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day53_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day53_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day53_12").format("auto").quality("auto").resize(scale().width(501));

 return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 53: Transformer-based Architectures (BERT, GPT)</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

       <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม BERT & GPT เปลี่ยนวงการ NLP?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในช่วงปี 2018-2019 วงการ Natural Language Processing (NLP) ได้เกิดการเปลี่ยนแปลงครั้งใหญ่ด้วยการเปิดตัวของ
      <strong> BERT (Bidirectional Encoder Representations from Transformers)</strong> และ <strong>GPT (Generative Pre-trained Transformer)</strong>.
      ทั้งสองสถาปัตยกรรมนี้ได้ขยายขีดความสามารถของโมเดล NLP แบบดั้งเดิมอย่างก้าวกระโดด
      ทำลายสถิติในหลาย benchmark สำคัญ เช่น GLUE, SQuAD, และ SuperGLUE.
    </p>

    <p>
      ปัจจัยสำคัญที่ทำให้ BERT และ GPT เปลี่ยน landscape ของ NLP ได้แก่:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>การใช้สถาปัตยกรรม Transformer ที่สามารถ model ความสัมพันธ์ระยะไกล (long-range dependency) ได้ดี</li>
      <li>การนำแนวคิด <strong>Pre-training + Fine-tuning</strong> มาใช้เป็นกระบวนการเรียนรู้หลัก</li>
      <li>ความสามารถในการเรียนรู้ contextual representation ของ token ได้อย่างลึกซึ้ง</li>
      <li>ความยืดหยุ่นในการปรับใช้กับ downstream task ที่หลากหลายโดยใช้ transfer learning</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.1 NLP ก่อนยุค Transformer</h3>
    <p>
      ก่อนการมาถึงของ Transformer-based Architectures วงการ NLP ต้องพึ่งพาโมเดลประเภท RNN, LSTM หรือ GRU ซึ่งมีข้อจำกัดหลายประการ:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>ความสามารถในการ parallelization ต่ำ → training ช้า</li>
      <li>ประสิทธิภาพในการจับ long-range dependency ไม่ดี (ปัญหา vanishing gradient)</li>
      <li>contextual representation ที่ได้มีข้อจำกัด (unidirectional หรือ limited context window)</li>
    </ul>

    <p>
      แม้ Word Embedding แบบ static (เช่น Word2Vec, GloVe) จะช่วยพัฒนา NLP ได้ระดับหนึ่ง แต่ก็ยังไม่สามารถเรียนรู้ representation แบบ context-sensitive ได้ดีพอ.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การเปลี่ยน paradigm จาก static embedding → contextualized embedding เป็น <strong>จุดเปลี่ยนสำคัญที่สุดในรอบทศวรรษ</strong> ของ NLP. 
        BERT และ GPT เป็นตัวอย่างแรกที่พิสูจน์ว่า pre-training บน corpora ขนาดใหญ่สามารถสร้าง universal language representation ที่ powerful ได้จริง.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.2 กระบวนการ Pre-training → Fine-tuning</h3>
    <p>
      BERT และ GPT ใช้แนวคิด <strong>Pre-training → Fine-tuning</strong> ที่ปัจจุบันกลายเป็นกระบวนการมาตรฐานใน NLP:
    </p>

    <ol className="list-decimal list-inside space-y-2">
      <li>
        <strong>Pre-training</strong>: โมเดลถูกฝึกบน unlabeled data ขนาดใหญ่ด้วย task แบบ self-supervised เช่น Masked Language Modeling (BERT) หรือ Next Token Prediction (GPT)
      </li>
      <li>
        <strong>Fine-tuning</strong>: โมเดล pre-trained ถูกนำมาปรับ (fine-tune) บน task-specific dataset ขนาดเล็ก เช่น Question Answering, Sentiment Analysis, Text Classification
      </li>
    </ol>

    <p>
      แนวทางนี้ช่วยให้สามารถสร้างโมเดล general-purpose NLP ที่ทรงพลังมากขึ้นกว่าเดิมหลายระดับ ทั้งในด้าน accuracy และ efficiency.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.3 Impact ต่อวงการ NLP และ AI โดยรวม</h3>
    <p>
      ความสำเร็จของ BERT และ GPT ทำให้เกิดผลกระทบในวงกว้าง:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>ทำให้ Transfer Learning กลายเป็นกระแสหลักใน NLP</li>
      <li>เร่งการพัฒนา Foundation Model → โมเดลใหญ่ข้าม task เช่น GPT-3, PaLM, LLaMA</li>
      <li>นำไปสู่ cross-modal Transformer (เช่น CLIP, Flamingo)</li>
      <li>เปลี่ยนความเข้าใจของวงการเกี่ยวกับ representation learning</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        BERT และ GPT เปลี่ยน paradigm ของ NLP จาก "feature engineering + task-specific model" → "pre-trained universal model + minimal fine-tuning". 
        ปัจจุบัน technique นี้ขยายไปยังทุก domain เช่น Vision, Speech, Robotics, Multimodal AI.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.4 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J. et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv, 2018.</li>
      <li>Radford, A. et al., "Improving Language Understanding by Generative Pre-Training", OpenAI, 2018.</li>
      <li>Vaswani, A. et al., "Attention Is All You Need", NeurIPS, 2017.</li>
      <li>Howard, J. & Ruder, S., "Universal Language Model Fine-tuning for Text Classification", arXiv, 2018.</li>
      <li>Bommasani, R. et al., "On the Opportunities and Risks of Foundation Models", Stanford CRFM, 2021.</li>
    </ul>
  </div>
</section>


    <section id="transformer-recap" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Transformer Architecture Recap</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การเข้าใจสถาปัตยกรรมของ Transformer อย่างลึกซึ้งเป็นสิ่งจำเป็นก่อนที่จะศึกษา BERT และ GPT เนื่องจากโมเดลเหล่านี้ต่อยอดจากแนวคิดพื้นฐานของ Transformer โดยตรง. สถาปัตยกรรม Transformer ถูกเสนอครั้งแรกในงานวิจัย <strong>"Attention Is All You Need"</strong> (Vaswani et al., 2017) โดยมีเป้าหมายเพื่อแทนที่ RNN และ CNN ในการประมวลผลลำดับข้อมูล (sequence modeling).
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.1 องค์ประกอบหลักของ Transformer</h3>
    <p>
      Transformer มีองค์ประกอบหลักดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Embedding Layer</strong>: แปลง token เป็น dense vector</li>
      <li><strong>Positional Encoding</strong>: เพิ่มข้อมูลลำดับเข้าไปใน embedding</li>
      <li><strong>Multi-Head Self-Attention</strong>: เรียนรู้ความสัมพันธ์ระหว่าง token ทั้งหมดในลำดับ</li>
      <li><strong>Feedforward Neural Network (FFN)</strong>: Layer fully connected ที่เรียนรู้ representation ต่อเนื่อง</li>
      <li><strong>Residual Connection & Layer Normalization</strong>: ช่วยให้ gradient flow ดีขึ้นและ training เสถียร</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        โครงสร้าง Transformer แยก <strong>Encoder</strong> และ <strong>Decoder</strong> ออกจากกัน. โมเดล BERT ใช้เฉพาะ Encoder, ในขณะที่ GPT ใช้เฉพาะ Decoder แบบ causal masked self-attention. การเลือก sub-architecture นี้มีผลต่อ behavior และ use case ของโมเดลอย่างมาก.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.2 Encoder vs Decoder Structure</h3>
    <p>
      Transformer ดั้งเดิมประกอบด้วย Encoder-Decoder architecture:
    </p>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-x-auto mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Component</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Encoder</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Decoder</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Bidirectional</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Masked (causal)</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">N/A</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Uses encoder outputs</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Use Case</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Representation learning</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sequence generation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.3 Self-Attention Mechanism</h3>
    <p>
      หัวใจสำคัญของ Transformer คือ <strong>Self-Attention</strong>, ซึ่งช่วยให้โมเดลสามารถเรียนรู้ context ของแต่ละ token แบบ global:
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <code>
        Attention(Q, K, V) = softmax(Q * Kᵀ / √dₖ) * V
      </code>
    </pre>

    <p>
      โดย Q (Query), K (Key), และ V (Value) ถูกสร้างจาก linear projection ของ embedding เดิม. การ scaling โดย √dₖ ช่วยให้ softmax มีเสถียรภาพมากขึ้น.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        <strong>Multi-Head Self-Attention</strong> ช่วยให้ Transformer สามารถเรียนรู้ pattern ต่าง ๆ ได้ในหลาย subspace พร้อมกัน — นี่คือสาเหตุที่โมเดลเช่น BERT และ GPT มีความสามารถในการ generalize ที่สูงมาก.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.4 Positional Encoding</h3>
    <p>
      เนื่องจาก Self-Attention ไม่รับรู้ลำดับเวลาโดยธรรมชาติ, Transformer จึงต้องเพิ่ม <strong>Positional Encoding</strong> เข้าไปใน input embedding เพื่อแทรกข้อมูลลำดับ:
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <code>
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model)){'\n'}
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
      </code>
    </pre>

    <p>
      วิธีนี้ทำให้ Transformer สามารถ encode both relative และ absolute position ของ token ได้อย่างมีประสิทธิภาพ.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.5 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al., "Attention Is All You Need", NeurIPS 2017.</li>
      <li>Devlin, J. et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv 2018.</li>
      <li>Radford, A. et al., "Language Models are Unsupervised Multitask Learners", OpenAI, 2019.</li>
      <li>Alammar, J., "The Illustrated Transformer", 2018.</li>
    </ul>
  </div>
</section>


<section id="bert" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. BERT: Bidirectional Encoder Representations from Transformers</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      <strong>BERT (Bidirectional Encoder Representations from Transformers)</strong> ได้เปลี่ยนโฉมหน้าวงการ Natural Language Processing (NLP) อย่างสิ้นเชิง ตั้งแต่เปิดตัวในปี 2018 โดย <strong>Devlin et al.</strong> จาก Google Research. จุดเด่นสำคัญคือการใช้ <strong>bidirectional Transformer encoder</strong> เพื่อเรียนรู้ contextual representation ของคำในบริบทแบบรอบด้าน.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.1 แนวคิดหลักของ BERT</h3>
    <p>
      ก่อน BERT โมเดลภาษาแบบ Unidirectional (เช่น GPT-1 หรือ RNN-based language model) ประมวลผลข้อมูลจากซ้ายไปขวาหรือขวาไปซ้ายเพียงทิศทางเดียว ส่งผลให้ representation ของคำมีข้อจำกัด.  
      BERT แก้ปัญหานี้โดยใช้ <strong>Masked Language Modeling (MLM)</strong> ทำให้ model สามารถมองเห็น context จากทั้งสองทิศทางพร้อมกัน.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        จุดเปลี่ยนสำคัญของ BERT คือการเปลี่ยนจาก <strong>left-to-right/auto-regressive learning → bidirectional representation learning</strong> ทำให้ BERT เป็นโมเดลแรกที่ได้ **deep bidirectional context** แบบ fully pre-trained สำหรับ downstream NLP tasks.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.2 โครงสร้างของ BERT</h3>
    <p>
      BERT ใช้เฉพาะ **Encoder stack ของ Transformer** (ไม่มี Decoder). โครงสร้างหลักประกอบด้วย:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Input Embedding: WordPiece Token Embedding + Segment Embedding + Positional Encoding</li>
      <li>Multi-layer Transformer Encoder (เช่น BERT-base: 12 layers, BERT-large: 24 layers)</li>
      <li>Output: Contextual embedding ของ token แต่ละตัว</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.3 Pre-training Objectives</h3>
    <p>
      BERT ถูก pre-train ด้วย 2 objective หลัก:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>Masked Language Model (MLM)</strong>: Randomly mask token ใน input แล้วให้ model ทำนาย token นั้น
      </li>
      <li>
        <strong>Next Sentence Prediction (NSP)</strong>: ให้ model ทำนายว่า sentence B เป็น sentence ถัดจาก sentence A หรือไม่
      </li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        แม้ว่า <strong>NSP objective</strong> จะถูกนำมาใช้ใน BERT รุ่นแรก, งานวิจัยภายหลัง (เช่น RoBERTa) พบว่าการตัด NSP ออกไปสามารถช่วยเพิ่มประสิทธิภาพของโมเดลในบาง task ได้.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.4 Fine-tuning สำหรับ Downstream Tasks</h3>
    <p>
      หลังจาก pre-training เสร็จ BERT สามารถนำมา fine-tune กับ task ต่าง ๆ ได้อย่างยืดหยุ่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Question Answering (เช่น SQuAD)</li>
      <li>Sentence Classification (เช่น sentiment analysis)</li>
      <li>Named Entity Recognition (NER)</li>
      <li>Textual Entailment / Natural Language Inference (NLI)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.5 ตัวอย่าง Masked Language Modeling Objective</h3>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <code>
        Input:  The man went to the [MASK] to buy milk.{'\n'}
        Target: store
      </code>
    </pre>
    <p>
      ตัวอย่างนี้แสดงให้เห็นว่า BERT ต้องใช้ **bidirectional context** ทั้งซ้ายและขวาเพื่อทำนาย token ที่ถูก mask.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.6 ผลกระทบต่อวงการ NLP</h3>
    <p>
      BERT ได้กลายเป็น <strong>pre-training paradigm</strong> มาตรฐานใน NLP และเป็นจุดเริ่มต้นของการพัฒนาโมเดลขนาดใหญ่ (large-scale pre-trained language models).  
      หลังการเปิดตัว BERT โมเดลใหม่ ๆ ที่ต่อยอดจากแนวคิดนี้ได้ถูกพัฒนาขึ้นจำนวนมาก เช่น RoBERTa, ALBERT, DistilBERT, ELECTRA.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.7 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.</li>
      <li>Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.</li>
      <li>Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.</li>
      <li>Lan, Z., et al. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. arXiv preprint arXiv:1909.11942.</li>
    </ul>
  </div>
</section>


       <section id="gpt" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. GPT: Generative Pre-trained Transformer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      <strong>GPT (Generative Pre-trained Transformer)</strong> เป็นโมเดลภาษาเชิง generative ที่ถูกพัฒนาโดย <strong>OpenAI</strong> โดยอาศัยสถาปัตยกรรม <strong>Transformer decoder</strong> เพียงอย่างเดียว.  
      แนวคิดหลักคือการ **pre-train บน unlabeled text corpus ขนาดใหญ่** แล้วนำมา fine-tune สำหรับ downstream tasks หรือใช้แบบ zero-shot / few-shot learning.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.1 แนวคิดการออกแบบ GPT</h3>
    <p>
      GPT แตกต่างจาก BERT ตรงที่เป็น **unidirectional** หรือ **left-to-right** language model โดยเรียนรู้ความน่าจะเป็นร่วม (joint probability) ของลำดับ token:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <code>
        P(x₁, x₂, ..., xₙ) = Π P(xₖ | x₁, ..., xₖ₋₁)
      </code>
    </pre>
    <p>
      สิ่งนี้ทำให้ GPT เหมาะอย่างยิ่งสำหรับ **text generation** และ **autoregressive tasks**.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การออกแบบ GPT เน้นการ train ด้วย **next-token prediction** ซึ่งต่างจาก BERT ที่ใช้ masked language modeling.  
        การเรียนรู้แบบ autoregressive ช่วยให้ GPT สามารถ generate text ต่อเนื่องได้อย่างเป็นธรรมชาติ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.2 โครงสร้างของ GPT</h3>
    <p>
      GPT ใช้เฉพาะ **Transformer decoder stack** ซึ่งประกอบด้วย:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Input Embedding: Token Embedding + Positional Encoding</li>
      <li>Transformer Decoder Blocks (เช่น GPT-2: 12-48 layers, GPT-3: 96 layers)</li>
      <li>Masked Multi-Head Self-Attention เพื่อป้องกันการเห็น token ข้างหน้า (future tokens)</li>
      <li>Final Linear + Softmax layer สำหรับทำนาย token ถัดไป</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.3 Pre-training Objective</h3>
    <p>
      GPT ใช้ objective เดียว คือ **language modeling** แบบ autoregressive:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto text-sm">
      <code>
        Loss = - Σ log P(xₖ | x₁, ..., xₖ₋₁)
      </code>
    </pre>
    <p>
      โมเดลเรียนรู้ที่จะทำนาย token ถัดไปบนพื้นฐานของ token ที่เห็นมาแล้ว.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        GPT ไม่ต้องการ objective เสริม เช่น Next Sentence Prediction (NSP).  
        การ train ด้วย next-token prediction เพียงอย่างเดียว ช่วยให้ model scale ได้ดีมากบน corpus ขนาดใหญ่.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.4 Evolution ของ GPT</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>GPT (2018)</strong>: Proof-of-concept ใช้ Transformer decoder stack + unsupervised pre-training</li>
      <li><strong>GPT-2 (2019)</strong>: 1.5B parameters, สร้าง text ที่สมจริงจน OpenAI เลือกจะยังไม่ปล่อย model เต็มในตอนแรก</li>
      <li><strong>GPT-3 (2020)</strong>: 175B parameters, สามารถทำ zero-shot และ few-shot learning ได้ดีมาก</li>
      <li><strong>GPT-4 (2023)</strong>: Multimodal (text + image), scaling ที่ดีขึ้น, performance บน wide range of tasks</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.5 ตัวอย่าง Use Case ของ GPT</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Text Generation: การเขียนบทความอัตโนมัติ</li>
      <li>Dialogue Systems: Chatbots เช่น ChatGPT</li>
      <li>Code Generation: โมเดลเช่น Codex</li>
      <li>Summarization, Translation, Question Answering แบบ few-shot</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.6 การเปรียบเทียบกับ BERT</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">BERT</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">GPT</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Architecture</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer Encoder</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer Decoder</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Objective</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Masked Language Model + NSP</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Autoregressive Language Model</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Best For</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Understanding Tasks (e.g., classification, QA)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Text Generation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.7 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI.</li>
      <li>Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.</li>
      <li>Brown, T.B., et al. (2020). Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165.</li>
      <li>OpenAI (2023). GPT-4 Technical Report. https://openai.com/research/gpt-4</li>
    </ul>
  </div>
</section>


<section id="bert-vs-gpt" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. BERT vs GPT: เปรียบเทียบ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      แม้ว่า <strong>BERT</strong> และ <strong>GPT</strong> จะมีรากฐานมาจากสถาปัตยกรรม Transformer เช่นเดียวกัน  
      แต่ทั้งสองได้รับการออกแบบมาเพื่อ <strong>จุดประสงค์ที่แตกต่างกัน</strong> อย่างชัดเจน.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.1 ความแตกต่างในด้าน Architecture</h3>
    <p>
      สถาปัตยกรรมของ BERT และ GPT มีความต่างหลักใน <strong>ลักษณะการใช้ Transformer stack</strong>:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>BERT:</strong> ใช้ <strong>Transformer Encoder stack</strong> แบบ bidirectional</li>
      <li><strong>GPT:</strong> ใช้ <strong>Transformer Decoder stack</strong> แบบ unidirectional (left-to-right)</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        BERT มีข้อได้เปรียบในการเข้าใจบริบทแบบ **bidirectional** ในขณะที่ GPT ได้เปรียบในการสร้าง sequence อย่างมีลำดับแบบ **autoregressive**.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.2 วัตถุประสงค์การฝึก (Training Objectives)</h3>
    <p>
      <strong>Objective function</strong> ที่ใช้ในการ train ของแต่ละ model ส่งผลโดยตรงต่อประเภทของงานที่ทำได้ดี:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>BERT:</strong> Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)</li>
      <li><strong>GPT:</strong> Autoregressive Language Modeling (next-token prediction)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.3 ประเภทของ Tasks ที่เหมาะสม</h3>
    <p>
      ด้วย nature ของ architecture และ objective ที่แตกต่างกัน BERT และ GPT จึงเหมาะกับงานที่ต่างกันดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>BERT:</strong> เหมาะสำหรับ **understanding tasks** เช่น text classification, named entity recognition, QA</li>
      <li><strong>GPT:</strong> เหมาะสำหรับ **generation tasks** เช่น text generation, story writing, dialogue generation</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        GPT มีความสามารถโดดเด่นในงาน **creative generation** ที่ต้องสร้าง sequence ใหม่ต่อเนื่อง,  
        ขณะที่ BERT excels ในงาน **comprehension** และ **representation learning**.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.4 เปรียบเทียบเชิงโครงสร้าง</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">BERT</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">GPT</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Architecture</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer Encoder</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer Decoder</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Objective</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Masked Language Model + NSP</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Autoregressive Language Model</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Best For</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Understanding Tasks (e.g., classification, QA)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Text Generation</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Contextual Modeling</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Bidirectional</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Left-to-right (unidirectional)</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.5 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.</li>
      <li>Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI.</li>
      <li>Brown, T.B., et al. (2020). Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165.</li>
    </ul>
  </div>
</section>



<section id="key-innovations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Key Innovations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การพัฒนา <strong>BERT</strong> และ <strong>GPT</strong> นำไปสู่ **นวัตกรรมสำคัญ** ที่เปลี่ยนแนวทางการออกแบบโมเดล NLP สมัยใหม่อย่างสิ้นเชิง.  
      ส่วนนี้จะสรุป <strong>Key Innovations หลัก</strong> ที่ทำให้โมเดลเหล่านี้มีผลกระทบระดับโลก.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.1 Pre-training + Fine-tuning Paradigm</h3>
    <p>
      แนวทาง <strong>Pre-training + Fine-tuning</strong> เป็นหนึ่งในนวัตกรรมที่สำคัญที่สุดที่ BERT และ GPT แสดงให้เห็น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ใช้ **large-scale unsupervised pre-training** บน corpus ขนาดใหญ่ (Wikipedia, BookCorpus, WebText ฯลฯ)</li>
      <li>จากนั้นนำมา **fine-tune แบบ supervised** บน task-specific datasets เพียงเล็กน้อย</li>
      <li>ลดความต้องการ data annotation แบบมหาศาลในแต่ละ task ลงได้อย่างมาก</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Paradigm นี้ทำให้ NLP เข้าสู่ยุค **transfer learning** อย่างเต็มตัวในระดับ model ecosystem ทั้งหมด — จาก task-specific → ไปสู่ pre-trained → fine-tuned.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.2 Bidirectional vs Autoregressive Modeling</h3>
    <p>
      การเลือก <strong>directionality ของ modeling</strong> ถือเป็น innovation ที่แตกต่างกันระหว่าง BERT และ GPT:
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Model</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Directionality</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Effect</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">BERT</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Bidirectional</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Better understanding of context from both sides</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GPT</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Autoregressive (left-to-right)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Enables high-quality text generation and continuation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.3 Scalability and Layer Stacking</h3>
    <p>
      อีกหนึ่ง innovation ที่สำคัญคือ **scalability ของ Transformer-based models**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถ stack layers ของ Transformer encoder หรือ decoder ได้ลึกหลายสิบระดับ</li>
      <li>Self-attention mechanism scale ได้ดีใน terms ของ **compute efficiency** และ **representational capacity**</li>
      <li>เป็นรากฐานสำคัญของโมเดลสมัยใหม่อย่าง GPT-3, GPT-4, PaLM, LLaMA ฯลฯ</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Scaling laws ของ Transformer-based models ชี้ให้เห็นว่าการเพิ่ม model size (parameters, layers) จะเพิ่ม performance  
        อย่างต่อเนื่องหาก training และ data มีขนาดเหมาะสม (Kaplan et al., 2020).
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.4 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.</li>
      <li>Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI.</li>
      <li>Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.</li>
      <li>Brown, T.B., et al. (2020). Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165.</li>
    </ul>
  </div>
</section>


     <section id="architectural-variants" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Architectural Variants & Evolutions</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การพัฒนา <strong>Transformer-based architectures</strong> ได้ก่อให้เกิด **variants** และ **evolutions** จำนวนมากที่ปรับปรุงเพื่อประสิทธิภาพ, scalability และ flexibility ในงานหลากหลายประเภท.  
      ส่วนนี้จะสรุปสถาปัตยกรรมที่ต่อยอดจาก BERT และ GPT ที่สำคัญในเชิงโครงสร้าง.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.1 Variants ที่สืบทอดจาก BERT</h3>
    <p>
      BERT นำไปสู่การพัฒนา variants จำนวนมากในงาน **understanding-focused NLP**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>RoBERTa</strong>: Robustly optimized BERT (Liu et al., 2019), เพิ่ม training data + training steps + dynamic masking → performance สูงขึ้น</li>
      <li><strong>ALBERT</strong>: A Lite BERT (Lan et al., 2020), parameter sharing เพื่อลด model size</li>
      <li><strong>DistilBERT</strong>: Knowledge distillation จาก BERT → model ขนาดเล็กลงแต่ยังคงคุณภาพสูง</li>
      <li><strong>SpanBERT</strong>: ปรับ pre-training objective ของ BERT ให้สามารถเรียนรู้ span-level representation ได้ดีขึ้น</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        RoBERTa เป็น variant ของ BERT ที่ outperform BERT บน GLUE benchmark ในทุก task โดยไม่ต้องเปลี่ยน architecture — ใช้เพียง better training protocol.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.2 Variants ที่สืบทอดจาก GPT</h3>
    <p>
      GPT เปิดแนวทางใหม่ใน **generative language modeling**, โดยมี evolutions สำคัญดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>GPT-2</strong>: เพิ่มขนาด model → 1.5B parameters → สร้าง text ได้ realistic มากขึ้น</li>
      <li><strong>GPT-3</strong>: 175B parameters → แสดง emergence ของ few-shot learning capabilities</li>
      <li><strong>GPT-4</strong>: เพิ่ม capability ทั้งทางด้าน cross-modal และ reasoning (details not fully disclosed)</li>
      <li><strong>CTRL</strong>: Conditional Transformer Language model → เพิ่ม control token เพื่อควบคุม style, domain ของ output</li>
      <li><strong>DialoGPT</strong>: Fine-tune GPT-2 สำหรับ conversational agents</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        GPT evolution ตั้งแต่ GPT-2 → GPT-3 → GPT-4 ได้เปลี่ยนแนวทางของ **NLP** จาก task-specific → ไปสู่ **universal language models** ที่รองรับหลากหลาย task ผ่าน in-context learning.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.3 Structural Innovations ใน Variants</h3>
    <p>
      Variants ของ Transformer ยังได้นำเสนอ **นวัตกรรมเชิงโครงสร้าง** สำคัญ:
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">นวัตกรรม</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">เป้าหมาย</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ตัวอย่าง</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parameter Sharing</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ลด model size</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ALBERT</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Conditional Control</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ควบคุม output style/domain</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CTRL</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-modal Modeling</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้จาก multimodal data (text + image)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GPT-4, Flamingo</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.4 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.</li>
      <li>Lan, Z., et al. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. arXiv preprint arXiv:1909.11942.</li>
      <li>Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.</li>
      <li>Keskar, N.S., et al. (2019). CTRL: A Conditional Transformer Language Model for Controllable Generation. arXiv preprint arXiv:1909.05858.</li>
    </ul>
  </div>
</section>


  <section id="practical-engineering" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Practical Engineering Considerations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การนำ **Transformer-based architectures** เช่น BERT และ GPT ไปใช้งานในระบบ production นั้น ไม่ได้เป็นเพียงแค่การ fine-tune โมเดลเท่านั้น แต่ยังต้องพิจารณาประเด็นด้าน **engineering** อย่างรอบด้าน.
    </p>
    <p>
      ส่วนนี้จะกล่าวถึง **considerations สำคัญ** ที่ทีม engineering ในองค์กรระดับโลก เช่น Google, Meta, OpenAI และ HuggingFace ใช้เมื่อ deploy Transformer models ใน production-scale systems.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.1 Scaling & Hardware Considerations</h3>
    <p>
      เนื่องจาก **Transformers เป็น compute-intensive architecture** การเลือก hardware infrastructure ที่เหมาะสมมีความสำคัญสูง:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**GPU/TPU**: จำเป็นสำหรับ training + inference ที่เร็วเพียงพอ → NVIDIA A100, H100, Google TPU v4</li>
      <li>**Memory**: Transformer ต้องการ memory สูงเพราะ self-attention complexity คือ O(n²)</li>
      <li>**Batch size tuning**: ต้องหา sweet spot ระหว่าง memory utilization กับ training throughput</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Inference cost ของ GPT-3 scale ขึ้นแบบ linear กับ sequence length → practical system ต้อง implement optimizations เช่น **KV caching** และ **attention pruning**.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.2 Quantization & Model Compression</h3>
    <p>
      การ deploy โมเดล Transformer แบบ full precision (FP32) อาจทำให้ inference latency สูงเกินไปใน production. กลยุทธ์สำคัญ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Quantization** → แปลง weights ไปเป็น INT8 หรือ mixed precision → ลด memory footprint & latency</li>
      <li>**Knowledge distillation** → training student model จาก teacher GPT/BERT → model ขนาดเล็กลง ~50-80%</li>
      <li>**Pruning** → ตัด attention heads หรือ layers ที่ contribution ต่ำที่สุดออก</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การใช้งาน **DistilBERT** ใน mobile NLP systems เช่นบน Edge devices ทำให้ latency ต่ำกว่า 50 ms ต่อ query โดยใช้ quantized INT8 inference.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.3 Serving Infrastructure</h3>
    <p>
      การออกแบบ serving infrastructure ของ Transformer models มี trade-offs ที่ต้องพิจารณา:
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Approach</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อดี</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อเสีย</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Batching Inference Requests</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม GPU utilization, ลด cost</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม latency สำหรับ real-time users</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Async Request Handling</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">รองรับ throughput สูง</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">complexity ของ system สูงขึ้น</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.4 Monitoring & Safety</h3>
    <p>
      การ deploy Transformer-based models โดยเฉพาะ generative models (GPT) ต้องมีการ monitor แบบ proactive:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Latency monitoring** → track response time + tail latency</li>
      <li>**Content safety filters** → ต้องติดตั้ง filtering pipeline สำหรับ output ของ GPT models</li>
      <li>**Monitoring drift** → language drift เมื่อ distribution ของ user input เปลี่ยน → retraining จำเป็น</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.5 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Rajbhandari, S., et al. (2021). ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. arXiv preprint arXiv:2104.07857.</li>
      <li>Jiao, X., et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. arXiv preprint arXiv:1909.10351.</li>
      <li>Patil, D.J., et al. (2021). Production-Ready Large Language Models: Engineering for Performance & Safety. Stanford CS224N Lecture Notes.</li>
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
      ตั้งแต่การเปิดตัว **BERT** และ **GPT**, สถาปัตยกรรม Transformer ได้เปลี่ยน landscape ของอุตสาหกรรม NLP และ AI applications อย่างสิ้นเชิง.
      เทคโนโลยีเหล่านี้ได้กลายเป็น core engine เบื้องหลัง **AI products ระดับโลก** และถูกนำไปใช้ใน domain ต่าง ๆ ที่หลากหลาย.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.1 NLP Applications</h3>
    <p>
      **Natural Language Processing** เป็น domain ที่ได้รับผลกระทบมากที่สุดจากการพัฒนา BERT และ GPT:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Search Engines** → Google Search ใช้ BERT ใน ranking pipeline → ปรับปรุง contextual understanding ของ queries</li>
      <li>**Chatbots & Virtual Assistants** → GPT-based chatbots เช่น ChatGPT, Google Bard, Anthropic Claude</li>
      <li>**Text Classification** → sentiment analysis, topic detection, toxic content detection</li>
      <li>**Named Entity Recognition (NER)** → BERT fine-tuned สำหรับ extraction ของ entities ในเอกสารขนาดใหญ่</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        BERT-based models ช่วยเพิ่ม accuracy ของ Google Search บน long-tail queries กว่า 10% ในปีแรกที่ deploy ตามรายงานจาก Google AI Blog.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.2 Content Generation & Creative AI</h3>
    <p>
      **GPT** เป็น engine หลักใน domain **Generative AI** ซึ่งกำลัง disrupt หลายอุตสาหกรรม:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Text Generation** → GPT ถูกใช้ใน marketing copywriting, automated email writing, news generation</li>
      <li>**Code Generation** → GitHub Copilot ใช้ Codex (fine-tuned GPT) → ช่วย generate code snippets</li>
      <li>**Dialogue Systems** → multi-turn conversations ใน AI assistants และ customer service bots</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        รายงานจาก GitHub (2023) พบว่า 46% ของ code ที่เขียนโดย developers ที่ใช้ GitHub Copilot มาจาก AI-generated suggestions.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.3 Cross-Modal AI & Multimodal Applications</h3>
    <p>
      ล่าสุด BERT และ GPT ได้ถูกนำไป extend สู่งาน **Multimodal AI**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**CLIP** → multimodal Transformer จาก OpenAI ที่ map text ↔ image representations</li>
      <li>**Visual Question Answering (VQA)** → ใช้ BERT + Vision encoders → answer questions about images</li>
      <li>**Video Understanding** → Transformer-based models ถูกใช้ในการทำ video captioning, summarization</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.4 Enterprise Adoption & Business Impact</h3>
    <p>
      หลายบริษัทใหญ่ได้นำ BERT และ GPT models เข้าไปใน production systems อย่างจริงจัง:
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">บริษัท</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Use Case</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Impact</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Google</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Search ranking, Smart Compose</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม quality ของ search results และ productivity ของ GMail</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Meta</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Content moderation, Ads relevance</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ลด policy violations และเพิ่ม revenue per impression</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">OpenAI</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GPT-based APIs</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สร้าง ecosystem ของ LLM apps ทั่วโลก</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.5 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.</li>
      <li>Brown, T.B., et al. (2020). Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165.</li>
      <li>Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020.</li>
      <li>GitHub Copilot Report 2023.</li>
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
      แม้ว่าสถาปัตยกรรม **BERT** และ **GPT** จะมีประสิทธิภาพสูงมากและเป็นที่ยอมรับในวงกว้าง แต่ก็มีข้อจำกัดที่สำคัญและประเด็นด้านจริยธรรมที่จำเป็นต้องพิจารณาอย่างรอบคอบก่อนการนำไปใช้งานจริง.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.1 Computational Cost & Environmental Impact</h3>
    <p>
      การฝึกสอนโมเดลขนาดใหญ่อย่าง GPT-3 หรือ BERT-large ต้องใช้ทรัพยากรคำนวณมหาศาล:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ต้องใช้ GPU clusters หรือ TPU pods ขนาดใหญ่</li>
      <li>ใช้เวลา training หลายสัปดาห์ → สิ้นเปลืองพลังงาน</li>
      <li>มี carbon footprint สูง ซึ่งส่งผลกระทบต่อสิ่งแวดล้อม</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        งานวิจัยจาก Strubell et al. (2019) พบว่า training BERT-base มี carbon emissions เทียบเท่ากับการเดินทางด้วยรถยนต์ประมาณ 700 กิโลเมตร.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.2 Data Bias & Fairness</h3>
    <p>
      โมเดล BERT และ GPT ได้รับการฝึกสอนจาก data ขนาดใหญ่มากจากอินเทอร์เน็ต ซึ่งมักมี bias และข้อมูลที่ไม่สมดุล:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>อคติด้านเชื้อชาติ เพศ และชาติพันธุ์</li>
      <li>คำพูดที่แฝง hate speech หรือ misinformation</li>
      <li>Representation ที่ไม่เท่าเทียมของกลุ่มที่มีบทบาทน้อยใน data</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        GPT-3 ถูกพบว่าสามารถ generate text ที่มีความ bias สูงในบาง use case หากไม่มีการใช้ debiasing mechanisms อย่างเหมาะสม.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.3 Explainability & Interpretability</h3>
    <p>
      แม้ว่า Transformer จะเป็นสถาปัตยกรรมที่มีความยืดหยุ่นสูง แต่การทำความเข้าใจว่าทำไม model ถึงให้คำตอบแบบหนึ่งยังคงเป็นเรื่องยาก:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Attention weights ให้ interpretability ได้ระดับหนึ่ง แต่ไม่เพียงพอสำหรับ use case ที่มีข้อกำหนดด้านความโปร่งใส</li>
      <li>Model ขนาดใหญ่เช่น GPT-4 มีความซับซ้อนเกินกว่าที่มนุษย์จะตีความได้แบบ deterministic</li>
      <li>Explainability มีความสำคัญมากใน domain เช่น healthcare, finance และ law</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.4 Potential for Misuse</h3>
    <p>
      Generative models อย่าง GPT มีศักยภาพสูงมาก แต่ก็เปิดโอกาสให้เกิด misuse ได้ง่าย:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>การสร้าง fake news, misinformation อย่างรวดเร็วและแม่นยำ</li>
      <li>Deepfake text ที่ใช้ในการหลอกลวงหรือโจมตีทางสังคม</li>
      <li>การทำ spam generation อัตโนมัติใน social platforms</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.5 Governance & Regulation</h3>
    <p>
      ยังไม่มีมาตรฐานกลางหรือแนวทางปฏิบัติที่เป็นที่ยอมรับในระดับสากลสำหรับ governance ของ LLM:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ขาดแนวทางการ audit โมเดลอย่างมีประสิทธิภาพ</li>
      <li>ขาด transparency ใน model training และ data pipeline ของโมเดลขนาดใหญ่ในระดับ commercial</li>
      <li>ความไม่สมดุลของการเข้าถึง model capabilities ระหว่าง tech giants กับสถาบันอื่น ๆ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.6 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and Policy Considerations for Deep Learning in NLP. arXiv preprint arXiv:1906.02243.</li>
      <li>Bender, E.M., et al. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? arXiv preprint arXiv:2102.02503.</li>
      <li>Brown, T.B., et al. (2020). Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165.</li>
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
      หลังจากได้สำรวจ **สถาปัตยกรรม Transformer-based** โดยเฉพาะ BERT และ GPT อย่างละเอียด  
      จะเห็นได้ว่าการออกแบบเหล่านี้ได้สร้างผลกระทบที่ลึกซึ้งต่อวงการ **Natural Language Processing (NLP)** และขยายอิทธิพลไปยังสาขาอื่น ๆ ของ Machine Learning อย่างรวดเร็ว.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.1 Key Takeaways</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>**BERT** และ **GPT** เป็นตัวอย่างสำคัญของวิธีการนำสถาปัตยกรรม Transformer มาประยุกต์ใช้แตกต่างกัน:
        <ul className="list-disc list-inside ml-6 space-y-1">
          <li>BERT: Focus on bidirectional encoding → ใช้ context จากทั้งสองด้านในทุก token</li>
          <li>GPT: Focus on autoregressive decoding → ใช้ context จากอดีตเพื่อ generate token ถัดไป</li>
        </ul>
      </li>
      <li>**Self-Attention Mechanism** เป็นหัวใจสำคัญที่ช่วยให้โมเดลเข้าใจ **global dependencies** ภายใน sequence ได้อย่างมีประสิทธิภาพ</li>
      <li>**Pre-training + Fine-tuning paradigm** ที่ริเริ่มโดย BERT และ GPT ได้กลายเป็นมาตรฐานใหม่ใน NLP และขยายสู่ domains อื่นเช่น Vision (ViT), Multimodal learning</li>
      <li>ความสำเร็จของ BERT และ GPT ช่วยจุดประกายให้เกิด **scaling laws** และความพยายามในการสร้าง **Foundation Models** ขนาดใหญ่อย่างต่อเนื่อง</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        วิวัฒนาการของ BERT และ GPT ได้แสดงให้เห็นว่า "การขยายขนาดโมเดล (scaling)" ร่วมกับ **pre-training on large unlabelled data**  
        เป็นกุญแจสำคัญที่ผลักดันขอบเขตของสิ่งที่เป็นไปได้ใน AI ขึ้นไปอย่างต่อเนื่อง.  
        สิ่งนี้นำไปสู่แนวโน้ม **Universal Foundation Models** ที่สามารถทำงานได้หลากหลาย task ด้วยโมเดลเดียว.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.2 Future Outlook</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ขยาย **scaling** ไปสู่ multi-trillion parameter models → ยังคงเป็น research frontier</li>
      <li>บูรณาการความสามารถของ Transformer กับ **multimodal data** อย่างมีประสิทธิภาพ</li>
      <li>เพิ่มความสามารถด้าน **reasoning**, **planning** และ **world modeling**</li>
      <li>การพัฒนาเครื่องมือ **alignment** และ **safety** สำหรับ LLM อย่างจริงจัง → เพื่อรองรับ deployment ใน real-world applications</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        สถาปัตยกรรม Transformer-based ไม่เพียงแค่ "เปลี่ยน NLP" อีกต่อไป  
        แต่ได้กลายเป็น "backbone" ของ **AI สมัยใหม่ในหลายสาขา** ตั้งแต่ Vision, Speech, Code generation ไปจนถึง Scientific discovery.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">11.3 References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.</li>
      <li>Radford, A., et al. (2018-2023). Improving Language Understanding by Generative Pre-training. OpenAI research papers.</li>
      <li>Bommasani, R., et al. (2021). On the Opportunities and Risks of Foundation Models. arXiv preprint arXiv:2108.07258.</li>
      <li>Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.</li>
    </ul>
  </div>
</section>


       <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. References</h2>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การทำความเข้าใจวิวัฒนาการของสถาปัตยกรรม BERT และ GPT รวมถึงผลกระทบในเชิงอุตสาหกรรม จำเป็นต้องอ้างอิงจากแหล่งความรู้ระดับโลกที่ได้รับการยอมรับในวงการวิชาการและอุตสาหกรรม AI อย่างกว้างขวาง.  
      ด้านล่างนี้คือรายการเอกสารและบทความที่ใช้เป็นฐานข้อมูลและกรอบแนวคิดสำหรับบทเรียนนี้.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        References ของ BERT และ GPT เป็นหนึ่งในกลุ่ม paper ที่ถูกอ้างอิงมากที่สุดในประวัติศาสตร์ AI.  
        บทเรียนนี้ได้อิงโครงสร้างและองค์ความรู้จากทั้ง paper ต้นทาง และงานต่อยอดที่มีผลกระทบต่อการพัฒนา AI สมัยใหม่.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.1 Key Papers</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). <em>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.</em> arXiv preprint arXiv:1810.04805.</li>
      <li>Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). <em>Improving Language Understanding by Generative Pre-training.</em> OpenAI.</li>
      <li>Radford, A., et al. (2019). <em>Language Models are Unsupervised Multitask Learners.</em> OpenAI Blog.</li>
      <li>Brown, T. B., et al. (2020). <em>Language Models are Few-Shot Learners.</em> arXiv preprint arXiv:2005.14165. (GPT-3)</li>
      <li>Bommasani, R., et al. (2021). <em>On the Opportunities and Risks of Foundation Models.</em> arXiv preprint arXiv:2108.07258.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.2 Related Research on Transformer Scaling</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Kaplan, J., et al. (2020). <em>Scaling Laws for Neural Language Models.</em> arXiv preprint arXiv:2001.08361.</li>
      <li>Henighan, T., et al. (2020). <em>Scaling Laws for Autoregressive Generative Modeling.</em> arXiv preprint arXiv:2010.14701.</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.3 Key References on Engineering and Deployment</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>OpenAI Engineering Blog. <em>Scaling Autoregressive Language Models with Stable Layer Norm.</em> https://openai.com/blog/scaling-autoregressive-language-models</li>
      <li>Google AI Blog. <em>Lessons from BERT’s Deployment at Scale.</em> https://ai.googleblog.com/2020/02/transformer-qa-bert-in-practice.html</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">12.4 Additional Readings</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A., et al. (2017). <em>Attention is All You Need.</em> arXiv preprint arXiv:1706.03762. (Transformer paper ต้นฉบับ)</li>
      <li>Alammar, J. (2019). <em>The Illustrated Transformer.</em> Online Blog Post. https://jalammar.github.io/illustrated-transformer/</li>
      <li>Wolf, T., et al. (2020). <em>Transformers: State-of-the-Art Natural Language Processing.</em> arXiv preprint arXiv:1910.03771. (Hugging Face paper)</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การติดตาม literature ล่าสุดและ best practices จาก paper, blog engineering และ open-source community เช่น Hugging Face มีความสำคัญต่อการเข้าใจ evolution ของ Transformer-based architectures อย่างครบถ้วน.
      </p>
    </div>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day53 theme={theme} />
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
        <ScrollSpy_Ai_Day53 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day53_TransformerArchitectures;
