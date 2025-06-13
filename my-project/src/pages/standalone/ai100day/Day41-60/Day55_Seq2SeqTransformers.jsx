"use client";
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day55 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day55";
import MiniQuiz_Day55 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day55";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day55_Seq2SeqTransformers = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day55_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day55_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day55_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day55_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day55_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day55_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day55_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day55_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day55_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day55_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day55_11").format("auto").quality("auto").resize(scale().width(500));
  const img12 = cld.image("Day55_12").format("auto").quality("auto").resize(scale().width(500));

  return (
      <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

       <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 ">
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 55: Sequence-to-Sequence with Transformers</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

<section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Seq2Seq ถึงสำคัญ</h2>

  <div className="flex justify-center my-6 max-w-full overflow-x-auto">
    <AdvancedImage cldImg={img2} className="max-w-full h-auto" />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Sequence-to-Sequence (Seq2Seq) models เป็นหนึ่งในรูปแบบที่มีความสำคัญอย่างยิ่งในงานด้าน Deep Learning โดยเฉพาะในงานที่มีลักษณะ <strong>input และ output เป็นลำดับของข้อมูล</strong> เช่น Machine Translation, Speech Recognition, Summarization, Image Captioning และ Dialogue Systems เป็นต้น การเรียนรู้แบบ Seq2Seq ได้เปลี่ยนขีดความสามารถของโมเดล Deep Learning ให้รองรับ task ที่มีความซับซ้อนเชิงโครงสร้างลำดับมากขึ้น.
    </p>

    <p>
      ก่อนการพัฒนา Seq2Seq โมเดลแบบดั้งเดิม เช่น RNN และ LSTM นั้นยังมีข้อจำกัดมากในด้านการจับ dependency ระยะไกล และการถ่ายทอด context ตลอด sequence โดยเฉพาะเมื่อ sequence มีความยาวมาก. แนวคิด Seq2Seq นำเสนอวิธีการแก้ปัญหานี้ โดยการแยกโมเดลออกเป็น 2 ส่วนสำคัญคือ <strong>Encoder</strong> และ <strong>Decoder</strong> ซึ่งทำให้สามารถ map ลำดับ input ไปยังลำดับ output ได้อย่างยืดหยุ่นมากขึ้น.
    </p>

    <p>
      เมื่อเวลาผ่านไป เทคโนโลยี Seq2Seq ได้ถูกต่อยอดอย่างมากโดยเฉพาะเมื่อรวมเข้ากับสถาปัตยกรรม Transformer ซึ่งทำให้ประสิทธิภาพในการจับ context และ dependency ของ sequence ดีขึ้นอย่างก้าวกระโดด และลดข้อจำกัดเรื่อง parallelization ของโมเดล RNN-based เดิม.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">Seq2Seq ในบริบทของ NLP และ AI สมัยใหม่</h3>
    <p>
      ในบริบทของ NLP และ AI สมัยใหม่ โมเดล Seq2Seq ถือเป็น backbone ของหลายระบบที่ใช้กันอย่างแพร่หลาย ไม่ว่าจะเป็น:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li><strong>Machine Translation</strong> เช่น Google Translate, DeepL</li>
      <li><strong>Speech Recognition</strong> เช่น ระบบ ASR ของ Google, Amazon, Microsoft</li>
      <li><strong>Text Summarization</strong> สำหรับเอกสารขนาดใหญ่ในงานด้านกฎหมายและธุรกิจ</li>
      <li><strong>Dialogue Systems</strong> เช่น Chatbot ที่ตอบสนองคำถามและสนทนาแบบ multi-turn</li>
      <li><strong>Image Captioning</strong> การอธิบายภาพด้วยข้อความอัตโนมัติ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">วิวัฒนาการจาก RNN → LSTM → Transformer-based Seq2Seq</h3>
    <p>
      ความก้าวหน้าของ Seq2Seq เริ่มจากการใช้ RNN ซึ่งตามมาด้วย LSTM/GRU ที่แก้ปัญหา vanishing gradient แต่เมื่อเจอปัญหา scaling ใน sequence ที่ยาวมาก ก็มีการนำ Transformer เข้ามา ซึ่งช่วยแก้ปัญหาเหล่านี้ได้อย่างโดดเด่น.
    </p>

    <div className="overflow-x-auto">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
        <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <tr>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold break-words">Architecture</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold break-words">Strengths</th>
            <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold break-words">Limitations</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">RNN-based Seq2Seq</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Simple architecture, suitable for small datasets</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Difficulty with long-range dependencies</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">LSTM/GRU-based Seq2Seq</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Better handling of long sequences</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Limited parallelization capability</td>
          </tr>
          <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Transformer-based Seq2Seq</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Highly parallelizable, excellent long-range dependency modeling</td>
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 break-words">Higher compute cost</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        งานวิจัยจาก Google Brain และ DeepMind แสดงให้เห็นว่า Transformer-based Seq2Seq model เช่น T5 และ BART outperform RNN/LSTM models ได้อย่างมีนัยสำคัญในหลาย task เช่น Translation, Summarization และ Question Answering บน benchmark dataset ชั้นนำ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., <em>Attention Is All You Need</em>, NeurIPS 2017</li>
      <li>Raffel et al., <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</em>, JMLR 2020</li>
      <li>Lewis et al., <em>BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</em>, ACL 2020</li>
      <li>Cho et al., <em>Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation</em>, arXiv 2014</li>
    </ul>
  </div>
</section>



     <section id="what-is-seq2seq" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. What is Sequence-to-Sequence?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Sequence-to-Sequence (Seq2Seq) เป็นกรอบการเรียนรู้เชิงลึก (deep learning framework) สำหรับงานที่มีรูปแบบ **mapping ข้อมูลจากลำดับหนึ่งไปยังอีกลำดับหนึ่ง** โดยที่ความยาวของ input sequence และ output sequence อาจแตกต่างกันได้. โมเดล Seq2Seq มีรากฐานจากสถาปัตยกรรม Neural Networks ที่พัฒนาเพื่อรองรับ task ประเภทนี้ ซึ่งได้รับความนิยมอย่างมากในงานด้าน Natural Language Processing (NLP), Speech Recognition และ Image Captioning.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">องค์ประกอบหลักของ Seq2Seq</h3>
    <p>
      โดยทั่วไป โมเดล Seq2Seq ประกอบด้วย 2 องค์ประกอบหลัก:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>Encoder:</strong> มีหน้าที่อ่านข้อมูล input sequence และเข้ารหัส (encode) เป็น vector representation หรือ context vector ที่สรุปสาระสำคัญของ input ทั้งหมด.
      </li>
      <li>
        <strong>Decoder:</strong> ใช้ context vector ที่ได้จาก encoder เพื่อสร้าง (decode) output sequence ทีละ token.
      </li>
    </ul>

    <p>
      ในทางปฏิบัติ context vector จะถูกปรับให้รองรับลำดับข้อมูลที่ซับซ้อนมากขึ้น โดยเฉพาะในงานที่ต้องจับความสัมพันธ์ระยะไกลระหว่าง input และ output.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">Formal Definition</h3>
    <p>
      ให้ \( X = (x_1, x_2, ..., x_n) \) เป็น input sequence และ \( Y = (y_1, y_2, ..., y_m) \) เป็น target output sequence.
    </p>
    <p>
      เป้าหมายของโมเดล Seq2Seq คือการเรียนรู้การแจกแจงแบบมีเงื่อนไข \( P(Y | X) \) โดยการ factorize ผ่าน conditional probability ต่อเนื่อง:
    </p>

 <ul className="list-disc list-inside space-y-3 mt-6">
  <li>
    Vaswani, A. et al., <em>Attention is All You Need</em>, NeurIPS 2017.{" "}
    <a
      href="https://arxiv.org/abs/1706.03762"
      className="text-blue-500 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      arXiv:1706.03762
    </a>
  </li>
  <li>
    Raffel, C. et al., <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)</em>, JMLR 2020.{" "}
    <a
      href="https://arxiv.org/abs/1910.10683"
      className="text-blue-500 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      arXiv:1910.10683
    </a>
  </li>
  <li>
    Lewis, M. et al., <em>BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</em>, ACL 2020.{" "}
    <a
      href="https://arxiv.org/abs/1910.13461"
      className="text-blue-500 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      arXiv:1910.13461
    </a>
  </li>
  <li>
    Dong, L. et al., <em>Unified Language Model Pre-training for Natural Language Understanding and Generation</em>, NeurIPS 2019.{" "}
    <a
      href="https://arxiv.org/abs/1905.03197"
      className="text-blue-500 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      arXiv:1905.03197
    </a>
  </li>
  <li>
    Zhang, Y. et al., <em>Transformer-based Sequence-to-Sequence Models for Automatic Speech Recognition</em>, arXiv 2020.{" "}
    <a
      href="https://arxiv.org/abs/2005.08100"
      className="text-blue-500 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      arXiv:2005.08100
    </a>
  </li>
  <li>
    Aharoni, R. et al., <em>Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges</em>, ACL 2019.
  </li>
  <li>
    Stanford CS224N Course Notes, <em>Lecture 12: Machine Translation and Advanced Seq2Seq Models</em>, Stanford NLP Group, 2021.
  </li>
  <li>
    MIT 6.S191, <em>Introduction to Deep Learning</em>, Lecture 9: <em>Sequence Modeling with Transformers</em>, 2021.
  </li>
  <li>
    Wang, W. et al., <em>Fine-tuning Pre-trained Language Models with Noisy Data for Text Generation</em>, ACL Findings 2021.
  </li>
</ul>

    <p>
      ซึ่งหมายความว่า ในแต่ละ timestep, decoder จะพยายาม generate token ถัดไปโดยอิงจาก tokens ก่อนหน้าและ context จาก encoder.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">ข้อดีของ Seq2Seq</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>รองรับการ map ลำดับ input → output ที่มีความยาวแตกต่างกันได้</li>
      <li>สามารถเรียนรู้ dependency ที่ซับซ้อนภายใน sequence</li>
      <li>รองรับ dynamic output length (ไม่มีการจำกัดความยาว output ล่วงหน้า)</li>
      <li>สามารถ generalize ไปยัง task ที่หลากหลาย</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        งานวิจัยของ <em>Sutskever et al. (2014)</em> เป็นหนึ่งในก้าวสำคัญที่แสดงให้เห็นว่า RNN-based Seq2Seq models
        สามารถใช้งานได้อย่างมีประสิทธิภาพสำหรับ **machine translation** และเป็นรากฐานสำคัญของงานต่อเนื่องในสาขานี้. 
        ต่อมาสถาปัตยกรรม Transformer ได้ต่อยอดแนวคิด Seq2Seq นี้ไปสู่อีกระดับที่มี performance สูงกว่า.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">ตัวอย่าง Use Case</h3>
   <div className="overflow-x-auto">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Use Case</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Input</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Output</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Machine Translation</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sentence in English</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sentence in French</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Summarization</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Full Document</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Short Summary</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Dialogue Generation</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">User utterance</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">System response</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014</li>
      <li>Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv 2014</li>
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Luong et al., *Effective Approaches to Attention-based Neural Machine Translation*, arXiv 2015</li>
    </ul>
  </div>
</section>


     <section id="transformer-seq2seq" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Transformer as a Seq2Seq Model</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      แม้ว่าโมเดล Sequence-to-Sequence (Seq2Seq) แบบดั้งเดิมจะอิงกับ RNN และ LSTM เป็นหลัก แต่ <strong>สถาปัตยกรรม Transformer</strong> ซึ่งนำเสนอโดย Vaswani et al. (2017) ได้เปลี่ยนโฉมวงการ Seq2Seq อย่างสิ้นเชิง ด้วยความสามารถในการเรียนรู้ **long-range dependencies** และประมวลผลแบบขนาน (parallelizable).
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">โครงสร้าง Seq2Seq ของ Transformer</h3>
    <p>
      ในการประยุกต์ใช้ Transformer กับ Seq2Seq, โมเดลประกอบด้วยสององค์ประกอบหลัก:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>Encoder Stack:</strong> รับ input sequence และแปลงเป็น contextualized representations ผ่านการใช้ self-attention layers และ feed-forward layers.
      </li>
      <li>
        <strong>Decoder Stack:</strong> รับ encoded representation และประมวลผลแบบ auto-regressive เพื่อสร้าง output sequence ทีละ token.
      </li>
    </ul>

    <p>
      ในระดับการ implement, encoder และ decoder มีโครงสร้างที่คล้ายคลึงกัน แต่ decoder จะมี **masked self-attention** layer เพื่อป้องกันการเข้าถึง token ที่ยังไม่ได้ generate.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">การไหลของข้อมูล (Data Flow)</h3>
    <p>
      โครงสร้าง Transformer แบบ Seq2Seq สามารถสรุปการไหลของข้อมูลได้ดังนี้:
    </p>

    <pre className="bg-gray-800 text-green-300 p-4 rounded-lg overflow-x-auto text-sm">
      Input Sequence → Encoder → Encoded Representations → Decoder → Output Sequence
    </pre>

    <p>
      โดยที่ใน decoder แต่ละ timestep:
    </p>

    <pre className="bg-gray-800 text-green-300 p-4 rounded-lg overflow-x-auto text-sm">
      Decoder Input = [Generated Tokens so far] + Encoded Representations
    </pre>

    <h3 className="text-xl font-semibold mt-8 mb-4">ความแตกต่างจาก RNN-based Seq2Seq</h3>
  <div className="overflow-x-auto">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Feature</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN-based Seq2Seq</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Transformer Seq2Seq</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelism</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Long-range Dependencies</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีมาก</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Stability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต้องใช้ gradient clipping</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เสถียรกว่า</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold mt-8 mb-4">ตัวอย่างการนำไปใช้จริง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Neural Machine Translation (NMT) เช่น Google Translate</li>
      <li>Text Summarization</li>
      <li>Dialogue Systems / Chatbots</li>
      <li>Image Captioning (เมื่อใช้ Visual Transformer เป็น encoder)</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        สถาปัตยกรรม Transformer แบบ Seq2Seq ใน paper <em>Attention is All You Need</em> ได้แสดงให้เห็นว่า **ไม่จำเป็นต้องมี recurrent component** ก็สามารถสร้าง state-of-the-art performance ใน Machine Translation ได้ และได้กลายเป็นมาตรฐานสำหรับงาน Seq2Seq ในปัจจุบัน.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Luong et al., *Effective Approaches to Attention-based Neural Machine Translation*, arXiv 2015</li>
      <li>Ott et al., *Scaling Neural Machine Translation*, arXiv 2018</li>
      <li>Popel & Bojar, *Training Tips for the Transformer Model*, arXiv 2018</li>
    </ul>
  </div>
</section>


<section id="seq2seq-applications" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Applications of Seq2Seq Transformers</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      โมเดล **Sequence-to-Sequence (Seq2Seq)** ที่ใช้ **Transformer** ได้รับความนิยมอย่างแพร่หลายในหลายสาขา ทั้งในงานเชิงอุตสาหกรรมและการวิจัย เนื่องจากสามารถสร้าง **contextualized output sequence** ที่มีความแม่นยำสูง และสามารถประมวลผลได้แบบขนาน จึงรองรับการใช้งานในระดับ production ได้เป็นอย่างดี.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Neural Machine Translation (NMT)</h3>
    <p>
      หนึ่งใน **use case** ที่ขับเคลื่อนการพัฒนา Transformer คือ **Neural Machine Translation** เช่น **Google Translate**, **DeepL**, ซึ่งใช้สถาปัตยกรรม **Transformer Seq2Seq** เป็นแกนกลาง โดยมีประสิทธิภาพเหนือกว่าสถาปัตยกรรม RNN-based เดิมอย่างมาก.
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถประมวลผล context ทั้งประโยคได้พร้อมกัน (global context)</li>
      <li>รองรับการแปลหลายภาษา (Multilingual NMT)</li>
      <li>ลด latency เมื่อ deploy ใน production</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Text Summarization</h3>
    <p>
      งาน **Text Summarization** ทั้งแบบ **Extractive** และ **Abstractive** เป็นอีกหนึ่ง application สำคัญ โดยเฉพาะในสื่อข่าว บทความวิชาการ และรายงานทางธุรกิจ เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Summarization บทความข่าวอัตโนมัติ</li>
      <li>สร้าง executive summary สำหรับรายงานธุรกิจ</li>
      <li>สร้าง TL;DR สำหรับ forum และ social media</li>
    </ul>
    <p>
      โมเดลอย่าง **BART**, **PEGASUS** ได้รับการออกแบบมาเฉพาะสำหรับ task นี้บนสถาปัตยกรรม Seq2Seq Transformer.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Conversational AI / Dialogue Systems</h3>
    <p>
      ระบบ **Chatbot** และ **Conversational AI** ที่ทันสมัยใช้สถาปัตยกรรม Seq2Seq Transformer เป็นพื้นฐาน โดยเฉพาะในการสร้าง **response generation** ที่มี context-aware และ fluent:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Customer service chatbots</li>
      <li>AI assistants (เช่น Google Assistant, Alexa)</li>
      <li>Open-domain dialogue generation</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Code Generation & Code Translation</h3>
    <p>
      โมเดล Seq2Seq Transformer ยังสามารถใช้ในการ **Code Generation** และ **Code Translation** ได้ด้วย โดยแปลง sequence ของคำสั่งภาษาโปรแกรมหนึ่งไปยังอีกภาษา:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>แปลง Python → C++ หรือ Java</li>
      <li>Auto-completion และ code synthesis</li>
      <li>Refactoring code</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Speech Recognition & Speech-to-Text</h3>
    <p>
      เมื่อจับคู่ Transformer encoder กับ decoder ที่ออกแบบมาเฉพาะ **Speech-to-Text** task ก็สามารถสร้างระบบ **Automatic Speech Recognition (ASR)** ที่มีประสิทธิภาพสูงได้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Transcribe speech เป็น text</li>
      <li>Multilingual ASR systems</li>
      <li>End-to-end ASR pipelines</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Transformer-based Seq2Seq ได้กลายเป็นแกนกลางของงาน NLP สมัยใหม่เกือบทุกด้าน — จาก **Machine Translation** สู่ **Text Summarization**, จาก **Conversational AI** สู่ **Speech Recognition**. งานวิจัยจาก Google, Facebook AI Research, และ HuggingFace ได้แสดงให้เห็นว่า Seq2Seq architecture มีความยืดหยุ่นสูง และสามารถ fine-tune ได้กับ domain-specific task อย่างกว้างขวาง.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation*, arXiv 2020</li>
      <li>Zhang et al., *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization*, arXiv 2020</li>
      <li>Chen et al., *The Microsoft Speech AI Toolkit for End-to-End Speech Recognition*, Interspeech 2021</li>
    </ul>
  </div>
</section>


      <section id="key-techniques" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Key Techniques in Transformer-based Seq2Seq</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การพัฒนาโมเดล **Transformer-based Seq2Seq** ในปัจจุบันได้รับการปรับปรุงอย่างต่อเนื่องผ่านเทคนิคที่สำคัญหลายประการ ทั้งในเชิงสถาปัตยกรรม (architecture) และในกระบวนการเทรน (training process). เทคนิคเหล่านี้ช่วยให้โมเดลสามารถสร้าง sequence ที่มี **fluency**, **coherence**, และ **domain-adaptation** ได้ดีขึ้น.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Teacher Forcing</h3>
    <p>
      เทคนิค **Teacher Forcing** เป็นเทคนิคพื้นฐานในการเทรนโมเดล Seq2Seq โดยใช้ target sequence จริงในขั้นตอนการ decode เพื่อช่วยให้โมเดลเรียนรู้การจัดลำดับ token อย่างมีประสิทธิภาพ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ใน training phase: ใช้ ground-truth token ใน step ถัดไป</li>
      <li>ใน inference phase: ใช้ prediction ของโมเดลแทน</li>
      <li>ช่วยลดปัญหา error accumulation ใน sequence generation</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Beam Search Decoding</h3>
    <p>
      **Beam Search** เป็น algorithm สำหรับหา sequence ที่ดีที่สุดในขั้นตอน inference ของ Transformer Seq2Seq:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>แทนที่จะเลือก token ที่มี probability สูงสุดทันที (greedy), Beam Search จะเก็บ top-k path</li>
      <li>ทำให้ sequence ที่ได้มี fluency สูงกว่า greedy search</li>
      <li>ค่า beam width ที่ใช้มีผลต่อ tradeoff ระหว่าง quality กับ latency</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Label Smoothing</h3>
    <p>
      **Label Smoothing** เป็น regularization technique ที่ช่วยลด overconfidence ของโมเดล:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>แทนที่จะกำหนด target probability = 1 สำหรับ correct label, จะกระจาย probability เล็กน้อยไปยัง class อื่น</li>
      <li>ช่วยให้โมเดล generalize ได้ดีขึ้น</li>
      <li>นิยมใช้ในงาน translation, summarization</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Positional Encoding & Relative Positional Encoding</h3>
    <p>
      ใน Seq2Seq, การแทรก **Positional Encoding** ที่เหมาะสมมีผลต่อความสามารถของโมเดลในการเข้าใจโครงสร้าง sequence:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Absolute Positional Encoding ใช้ sinusoidal function แบบดั้งเดิม</li>
      <li>Relative Positional Encoding ช่วย capture ความสัมพันธ์แบบ pairwise ที่ยืดหยุ่นกว่า</li>
      <li>งานสมัยใหม่เช่น T5, Transformer-XL ใช้ Relative Encoding เป็นหลัก</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Scheduled Sampling</h3>
    <p>
      เทคนิค **Scheduled Sampling** เป็นการผสมผสานระหว่าง Teacher Forcing และ Autoregressive Decoding:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ในระหว่าง training จะมีโอกาสสุ่มใช้ token ที่โมเดลทำนายได้แทน ground-truth token</li>
      <li>ช่วยให้โมเดล robust ต่อ prediction error ที่สะสม</li>
      <li>เหมาะสำหรับ task ที่ sequence ยาวมาก เช่น dialogue</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">6. Pretraining & Transfer Learning</h3>
    <p>
      การใช้โมเดล Transformer ที่ pretrained บน large corpus แล้ว fine-tune บน Seq2Seq task เฉพาะ เป็นแนวทางมาตรฐาน:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ตัวอย่างเช่น BART, T5 pretrained บน text reconstruction, denoising autoencoding</li>
      <li>ช่วยให้โมเดลเรียนรู้ syntax และ semantics มาก่อน fine-tune</li>
      <li>ประหยัด data และ compute สำหรับ downstream task</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การเลือกเทคนิคที่เหมาะสม เช่น **Beam Search**, **Label Smoothing**, และ **Relative Positional Encoding** มีผลอย่างมากต่อ performance ของ Transformer-based Seq2Seq models ใน production จริง. งานวิจัยล่าสุดยังแนะนำให้ใช้ **pretrained seq2seq transformers** เป็น baseline ที่แข็งแกร่งที่สุดในปัจจุบัน.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training*, arXiv 2020</li>
      <li>Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR 2020</li>
      <li>He et al., *Understanding and Improving Layer Normalization*, NeurIPS 2020</li>
    </ul>
  </div>
</section>


  <section id="advanced-architectures" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Advanced Architectures</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในช่วงหลายปีที่ผ่านมา สถาปัตยกรรม **Transformer-based Seq2Seq** ได้ถูกพัฒนาต่อยอดอย่างรวดเร็ว เพื่อตอบโจทย์ทั้งด้าน **ประสิทธิภาพการประมวลผล**, **quality ของ sequence generation**, และ **scalability** สำหรับงานใน production จริง. งานวิจัยและวิศวกรรมระดับโลกได้ผลักดันเทคนิคใหม่ ๆ ที่สำคัญ ซึ่งกลายเป็นแกนหลักของสถาปัตยกรรม **advanced Seq2Seq models** ในปัจจุบัน.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Pre-LayerNorm Transformers</h3>
    <p>
      การใช้ **Layer Normalization (LayerNorm)** ก่อน **Multi-Head Attention** และ **FeedForward Network (FFN)** ช่วยเพิ่ม **training stability** อย่างมีนัยสำคัญ โดยเฉพาะในโมเดลขนาดใหญ่ (T5, GPT-3 ใช้เทคนิคนี้เป็นหลัก):
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ช่วยลดปัญหา exploding gradients</li>
      <li>ทำให้สามารถใช้ learning rate ที่สูงขึ้นได้</li>
      <li>เร่ง convergence ของ training process</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Relative Positional Encoding</h3>
    <p>
      แม้ว่าตัว Transformer ดั้งเดิมจะใช้ **absolute positional encoding**, งานวิจัยล่าสุดแสดงให้เห็นว่า **relative positional encoding** ให้ performance ที่ดีกว่าในหลาย task:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ช่วย capture relation แบบ pairwise ระหว่าง token ได้ดีกว่า</li>
      <li>generalize better สำหรับ sequence length ที่แตกต่างจาก training</li>
      <li>ใช้ในโมเดลเช่น Transformer-XL, T5</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Sparse Transformers & Long-Range Attention</h3>
    <p>
      โมเดล Seq2Seq แบบดั้งเดิมมี **quadratic attention complexity (O(n²))** ซึ่งไม่เหมาะสำหรับ sequence ยาวมาก:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Sparse Transformers** ใช้ sparse attention pattern เพื่อลด complexity เป็น O(n log n)</li>
      <li>**Longformer, BigBird** ออกแบบมาเพื่อจัดการ long sequences โดยเฉพาะ</li>
      <li>เหมาะสำหรับงานเช่น document summarization, genome modeling</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Prefix-Tuning & Prompt Tuning</h3>
    <p>
      **Prefix-Tuning** เป็นเทคนิค parameter-efficient ที่สามารถนำโมเดล pretrained มาใช้กับ downstream task ได้โดยไม่ต้อง fine-tune ทุก parameter:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เรียนรู้เพียง prefix vector ที่ prepended เข้าไปใน attention layers</li>
      <li>ช่วยลด compute cost และ storage overhead</li>
      <li>ใช้ได้ดีใน setting ที่มี compute resource จำกัด</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Denoising Autoencoder Architectures</h3>
    <p>
      โมเดลเช่น **BART** และ **mBART** ใช้สถาปัตยกรรม **Denoising Seq2Seq** เพื่อ pretrain:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เรียนรู้จาก corrupted input → reconstruct target sequence</li>
      <li>ช่วยให้ encoder-decoder model สามารถ generalize ได้ดีขึ้น</li>
      <li>เหมาะสำหรับ task ที่ต้องการ robust output เช่น summarization, translation</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">6. Multi-Modal Seq2Seq Architectures</h3>
    <p>
      งานวิจัยล่าสุดพัฒนา **Multi-Modal Transformer-based Seq2Seq** สำหรับ task เช่น **image captioning**, **video summarization**, **speech-to-text**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ใช้ **cross-modal attention** layers</li>
      <li>Encoder สามารถ encode multimodal inputs → Decoder generate target sequence</li>
      <li>ตัวอย่างเช่น Flamingo, BLIP</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        สถาปัตยกรรม **advanced Seq2Seq** ได้รับการพัฒนาทั้งในด้าน **scalability** และ **robustness**. ใน production setting, แนวทางเช่น **Prefix-Tuning**, **Sparse Attention**, และ **Relative Positional Encoding** เป็นเทคนิคหลักที่ถูกนำมาใช้ในงานระดับ enterprise เช่น translation API ของ Google, DeepMind, และ AI startups.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR 2020</li>
      <li>Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training*, arXiv 2020</li>
      <li>Beltagy et al., *Longformer: The Long-Document Transformer*, arXiv 2020</li>
      <li>Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, ACL 2021</li>
    </ul>
  </div>
</section>


<section id="training-optimization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Training & Optimization Tips</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การฝึกและปรับแต่งโมเดล <strong>Transformer-based Seq2Seq</strong> ให้อยู่ในระดับ production-grade จำเป็นต้องพิจารณาเทคนิคด้าน <strong>training efficiency</strong>, <strong>generalization capability</strong>, และ <strong>robustness</strong> อย่างรอบด้าน. Section นี้นำเสนอแนวปฏิบัติที่ดีที่สุดซึ่งได้รับการยอมรับในงานวิจัยและการใช้งานในอุตสาหกรรม.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Learning Rate Scheduling</h3>
    <p>
      <strong>Learning rate scheduling</strong> เป็นปัจจัยสำคัญในการฝึก Transformer:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Warmup</strong> phase ช่วย stabilize training ในช่วงเริ่มต้น</li>
      <li><strong>Inverse square root decay</strong> นิยมใช้ในงาน translation (เช่นใน Fairseq)</li>
      <li><strong>Cosine decay</strong> เป็นอีกทางเลือกหนึ่งที่ช่วยปรับ optimization dynamics</li>
    </ul>

    <div className="overflow-x-auto">
      <pre className="bg-gray-800 text-green-300 p-4 rounded-lg text-sm">
        <code className="language-python">
{`optimizer = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_inverse_sqrt_scheduler(optimizer, warmup_steps=4000)`}
        </code>
      </pre>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Gradient Clipping</h3>
    <p>
      การฝึกโมเดล Seq2Seq ขนาดใหญ่มีความเสี่ยงที่จะเกิด <strong>gradient explosion</strong>, ดังนั้นการใช้ <strong>gradient clipping</strong> เป็นวิธีมาตรฐานที่จำเป็น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ช่วยป้องกัน gradient overflow ใน training batch ขนาดใหญ่</li>
      <li>ปรับปรุง stability โดยเฉพาะใน task ที่ sequence มี length variance สูง</li>
      <li>ค่าที่นิยมใช้คือ clip ที่ <strong>norm ≤ 1.0</strong></li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Label Smoothing</h3>
    <p>
      เทคนิค <strong>label smoothing</strong> ลดปัญหา <strong>over-confidence</strong> ของ decoder และช่วย generalization:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เพิ่ม regularization ให้โมเดลไม่ฟิตกับ target distribution ที่ noise มากเกินไป</li>
      <li>ค่า smoothing factor ที่พบบ่อยคือ <strong>ε = 0.1 ~ 0.2</strong></li>
      <li>ช่วยให้ BLEU score ใน translation task ดีขึ้นอย่างมีนัยสำคัญ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Mixed Precision Training</h3>
    <p>
      <strong>Mixed precision training</strong> (FP16) ช่วยลด <strong>training time</strong> และ <strong>memory footprint</strong> อย่างมาก โดยไม่มีผลกระทบต่อ performance:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>นิยมใช้ใน library เช่น PyTorch AMP, NVIDIA Apex</li>
      <li>เหมาะกับ training บน GPU เช่น A100, V100, RTX 3090</li>
      <li>ประหยัด memory ได้ ~50%</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Curriculum Learning & Progressive Training</h3>
    <p>
      งานวิจัยล่าสุด (Google T5, OpenAI) พบว่า <strong>Curriculum Learning</strong> สามารถเพิ่ม quality ของ training:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เริ่มฝึกจาก sequence ที่ง่าย → ค่อยๆ เพิ่ม complexity</li>
      <li>เหมาะสำหรับ multilingual training, summarization, document-level translation</li>
      <li>ช่วยเร่ง convergence และลด training instability</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">6. Checkpointing & Resume Training</h3>
    <p>
      ใน training ที่ใช้เวลานาน (เช่นหลายสัปดาห์), <strong>robust checkpointing</strong> system เป็นสิ่งจำเป็น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>บันทึก model state + optimizer state + scheduler state</li>
      <li>รองรับ resume training อย่างต่อเนื่องโดยไม่มี performance drop</li>
      <li>ควรใช้ framework เช่น <strong>PyTorch Lightning</strong>, <strong>Fairseq</strong> ที่รองรับระบบนี้ในตัว</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">7. Regularization Techniques</h3>
    <p>
      นอกเหนือจาก label smoothing, เทคนิค regularization อื่น ๆ ที่ควรพิจารณา:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Dropout</strong> บน attention weights (standard dropout หรือ DropHead)</li>
      <li><strong>LayerDrop</strong> — drop entire layers แบบ stochastic</li>
      <li><strong>Early Stopping</strong> โดยใช้ validation BLEU / ROUGE score</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        เทคนิค <strong>Gradient Clipping</strong>, <strong>Mixed Precision</strong>, และ <strong>Curriculum Learning</strong> เป็นส่วนสำคัญของ <strong>training pipelines ของ production-grade Seq2Seq models</strong> เช่น T5, mBART, Meta NLLB. ทีม AI ที่มีประสบการณ์ลึกจะ integrate เทคนิคเหล่านี้อย่างเป็นระบบใน training loop ของตนเสมอ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Popel & Bojar, <em>Training Tips for the Transformer Model</em>, The Prague Bulletin of Mathematical Linguistics, 2018</li>
      <li>Ott et al., <em>Scaling Neural Machine Translation</em>, Facebook AI Research, arXiv 2018</li>
      <li>Raffel et al., <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)</em>, JMLR 2020</li>
      <li>Fan et al., <em>Beyond English-Centric Multilingual Machine Translation</em>, Meta AI, arXiv 2021</li>
    </ul>
  </div>
</section>



<section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Use Cases & Industry Impact</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      โมเดล **Sequence-to-Sequence (Seq2Seq) based on Transformers** ได้กลายเป็น core technology ที่อยู่เบื้องหลังระบบ AI ที่สำคัญในหลากหลายอุตสาหกรรม. ความสามารถในการ map **input sequence → output sequence** อย่างมีประสิทธิภาพ ทำให้สามารถนำไปประยุกต์ใช้ในงานที่มีความซับซ้อนเชิงโครงสร้างสูง. Section นี้นำเสนอ use cases ที่สำคัญพร้อมตัวอย่างการใช้งานจริงในอุตสาหกรรมระดับโลก.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Machine Translation</h3>
    <p>
      งานแปลภาษาแบบ neural machine translation (NMT) เป็นหนึ่งใน use case ที่ผลักดันให้ Seq2Seq Transformer models เติบโต:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Google Translate, Microsoft Translator ใช้ Transformer-based NMT ตั้งแต่ปี 2017</li>
      <li>Meta AI NLLB (No Language Left Behind) ใช้ massive multilingual Transformer models ที่รองรับกว่า 200 ภาษา</li>
      <li>BLEU score improvements 5-10% เมื่อเทียบกับ RNN-based Seq2Seq เดิม</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Text Summarization</h3>
    <p>
      การสรุปข้อความยาวให้เป็นข้อความสั้นด้วย Seq2Seq models ได้รับความนิยมในหลาย use case:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Summarization ของบทความข่าว, รายงานวิจัย, บทวิเคราะห์ทางธุรกิจ</li>
      <li>Document summarization สำหรับ legal documents และ financial reports</li>
      <li>โมเดลที่ใช้ใน production: **BART**, **PEGASUS**, **T5**</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Conversational AI & Chatbots</h3>
    <p>
      Conversational AI ใช้ Seq2Seq models สำหรับ dialog generation:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Meta BlenderBot ใช้ Transformer-based Seq2Seq สำหรับ open-domain conversation</li>
      <li>Customer support chatbots ใช้ Seq2Seq เพื่อ generate dynamic responses</li>
      <li>Voice assistants (เช่น Alexa, Google Assistant) ใช้การ integrate Seq2Seq ใน natural language generation (NLG)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Code Generation</h3>
    <p>
      การ generate source code จาก natural language เป็นอีก use case ที่โดดเด่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>OpenAI Codex ใช้ Transformer Seq2Seq สำหรับ text-to-code generation</li>
      <li>GitHub Copilot ใช้ Seq2Seq Transformer fine-tuned บน code datasets</li>
      <li>ช่วย automate coding tasks ใน software development workflow</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Speech Recognition & Text-to-Speech</h3>
    <p>
      Seq2Seq architecture ยังถูกนำมาใช้ในงาน **ASR** และ **TTS**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Google Speech-to-Text ใช้ Transformer-based models สำหรับ streaming ASR</li>
      <li>Tacotron 2 (Google) ใช้ Seq2Seq สำหรับ text-to-mel-spectrogram generation → TTS pipeline</li>
      <li>ช่วยปรับปรุง naturalness ของเสียงใน virtual assistants</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        โมเดล Seq2Seq based on Transformer ได้สร้างผลกระทบเชิงอุตสาหกรรมอย่างลึกซึ้ง — จาก **Google Translate**, **Meta NLLB**, **OpenAI Codex** จนถึง **Conversational AI** และ **Multimodal models**. องค์กรที่ปรับตัวเร็วและนำเทคโนโลยีเหล่านี้มาใช้สามารถสร้างความแตกต่างในการแข่งขันได้อย่างมีนัยสำคัญ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR 2020</li>
      <li>Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation*, ACL 2020</li>
      <li>Fan et al., *Beyond English-Centric Multilingual Machine Translation*, Meta AI, arXiv 2021</li>
      <li>Chen et al., *Codex: An OpenAI GPT-based Model for Programming*, OpenAI 2021</li>
    </ul>
  </div>
</section>


   <section id="limitations-challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Limitations & Challenges</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      แม้ว่า **Transformer-based Sequence-to-Sequence (Seq2Seq) models** จะได้รับการพิสูจน์แล้วว่ามีประสิทธิภาพสูงในงานระดับโลก แต่ยังคงมีข้อจำกัดและความท้าทายสำคัญที่ต้องพิจารณา. Section นี้จะวิเคราะห์ข้อจำกัดทางวิศวกรรม ข้อจำกัดเชิงโมเดล และปัจจัยด้าน ethical/social ที่เกิดขึ้นจริงในภาคอุตสาหกรรม.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Computational Cost & Scalability</h3>
    <p>
      Seq2Seq models ขนาดใหญ่มีข้อจำกัดด้านทรัพยากรสูง:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Training: ต้องใช้ hardware ระดับ **GPU cluster / TPU pod** ที่มีค่าใช้จ่ายสูงมาก</li>
      <li>Inference: latency สูงเมื่อทำงานใน setting แบบ real-time เช่น voice translation</li>
      <li>Energy consumption: มี carbon footprint สูงหากไม่มีการ optimize อย่างเหมาะสม</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Data Requirements</h3>
    <p>
      ความสำเร็จของ Transformer-based Seq2Seq models มักขึ้นกับ **large-scale datasets**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Domain mismatch: fine-tuning กับ domain ใหม่ (เช่น medical/legal) ต้องใช้ labeled data ปริมาณมาก</li>
      <li>Low-resource languages: performance ต่ำในภาษา/โดเมนที่มีข้อมูลจำกัด</li>
      <li>Data bias: model สามารถเรียนรู้ bias จาก training data ได้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Hallucination & Factual Inaccuracy</h3>
    <p>
      ปัญหานี้สำคัญโดยเฉพาะในงาน **generation**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Model อาจ generate output ที่ **ดูสมจริงแต่ผิดข้อเท็จจริง**</li>
      <li>ยากต่อการควบคุม output ให้ factually consistent ใน Seq2Seq NLG tasks เช่น summarization, translation</li>
      <li>ต้องใช้ **post-processing / retrieval-augmented generation** เพื่อ mitigate</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Interpretability & Debugging</h3>
    <p>
      แม้ว่าจะมีความก้าวหน้าในการ **explainability** ของ Transformer models แต่ Seq2Seq ยังคงยากต่อการ debug:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Hard attention vs soft attention ยังไม่ได้แสดง **clear reasoning path** เสมอไป</li>
      <li>Chain-of-thought generation ใน Seq2Seq ยังอยู่ในขั้นวิจัย</li>
      <li>ความเข้าใจ internal behavior ยังจำกัดเมื่อเกิด model errors</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Long-Range Context Limitation</h3>
    <p>
      แม้จะมีการปรับปรุงเช่น **Longformer**, **BigBird**, **Reformer** แต่ Seq2Seq Transformer รุ่นดั้งเดิมยังมีข้อจำกัด:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Context window จำกัด (~2k tokens - 4k tokens)</li>
      <li>Memory footprint เพิ่มขึ้นแบบ quadratic กับ sequence length</li>
      <li>ต้องพิจารณา architectural trade-offs เสมอเมื่อต้องการ process **long documents**</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        แม้ว่า Transformer-based Seq2Seq จะเป็น state-of-the-art แต่ข้อจำกัดเช่น **cost, bias, hallucination, interpretability**, และ **long-range limitations** ยังคงต้องการการวิจัยและการพัฒนาอย่างต่อเนื่องเพื่อให้โมเดลเหล่านี้สามารถนำไปใช้ในระดับ production ได้อย่างปลอดภัยและมีประสิทธิภาพ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., *Attention Is All You Need*, NeurIPS 2017</li>
      <li>Beltagy et al., *Longformer: The Long-Document Transformer*, arXiv 2020</li>
      <li>Zaheer et al., *BigBird: Transformers for Longer Sequences*, NeurIPS 2020</li>
      <li>Raffel et al., *T5: Exploring the Limits of Transfer Learning*, JMLR 2020</li>
      <li>Meta AI, *No Language Left Behind*, arXiv 2022</li>
    </ul>
  </div>
</section>

<section id="future-directions" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Future Directions</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      แม้ว่า **Transformer-based Seq2Seq models** ได้กลายเป็นแกนกลางของ **modern AI applications** แต่ชุมชนวิจัยยังคงเดินหน้าพัฒนาเพื่อผลักดันขีดจำกัดของโมเดลเหล่านี้. ใน Section นี้ จะกล่าวถึงทิศทางการวิจัยและวิศวกรรมในระดับแนวหน้าที่กำลังเป็นที่สนใจในมหาวิทยาลัยชั้นนำและภาคอุตสาหกรรม.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1. Scaling Models: Beyond Trillion Parameters</h3>
    <p>
      งานวิจัยจาก **OpenAI, Google DeepMind, Meta AI** แสดงให้เห็นว่า Seq2Seq models สามารถ scale ได้อย่างต่อเนื่อง:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Efficient scaling laws ถูกใช้เพื่อประเมิน trade-off ระหว่าง compute, data, model size</li>
      <li>งานเช่น **Chinchilla scaling** แนะนำว่ data-to-parameter ratio เป็นตัวแปรสำคัญ</li>
      <li>Future models อาจมีขนาด มากกว่า 1T parameters และต้องใช้ **adaptive routing** เช่น Mixture-of-Experts (MoE)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2. Efficient Inference & Distillation</h3>
    <p>
      ความต้องการนำ Seq2Seq ไปใช้ใน production ผลักดันการพัฒนา:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Distillation techniques เช่น **TinyBERT**, **DistilBART** กำลังพัฒนาอย่างต่อเนื่อง</li>
      <li>Sparse Transformer architectures กำลังได้รับความนิยม เช่น **Switch Transformer**</li>
      <li>Hardware-aware optimization เช่น tensor-program autotuning ช่วยเพิ่มประสิทธิภาพ deployment</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">3. Memory-Augmented Transformers</h3>
    <p>
      ขีดจำกัดของ Seq2Seq Transformer ใน long-context modeling กำลังถูกท้าทายด้วย:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Memory-augmented models เช่น **Memformer**, **RETRO** ของ DeepMind</li>
      <li>Retrieval-based architectures ช่วยเพิ่ม context window เป็น **10k-100k tokens**</li>
      <li>เพิ่มความสามารถใน tasks เช่น **document-level translation** และ **long dialogue generation**</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4. Multimodal Seq2Seq Models</h3>
    <p>
      แนวโน้มสำคัญคือการขยาย Seq2Seq ไปสู่ multimodal learning:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>งานเช่น **Flamingo** (DeepMind) และ **Kosmos** (Microsoft) ใช้ Seq2Seq backbone ร่วมกับ vision, text, audio</li>
      <li>Multimodal translation (video → caption, speech → text) เป็น use case ที่เติบโตเร็ว</li>
      <li>Alignment techniques เช่น **Contrastive Learning + Seq2Seq decoding** กำลังได้รับความนิยม</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5. Responsible & Ethical Seq2Seq Modeling</h3>
    <p>
      ความกังวลด้าน ethics ใน Seq2Seq models กำลังผลักดันงานวิจัยใหม่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Debiasing methods เช่น **Counterfactual Data Augmentation (CDA)** สำหรับ translation</li>
      <li>Explainability frameworks เพื่อเปิดเผย chain-of-thought ใน generation</li>
      <li>Model cards และ governance frameworks สำหรับ Seq2Seq systems ใน production</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        อนาคตของ **Seq2Seq Transformers** กำลังเปลี่ยนแปลงอย่างรวดเร็ว — จากโมเดลขนาดเล็กสำหรับ translation → สู่ **multimodal, long-context, and memory-augmented models** ที่ขยายขีดความสามารถไปยังระดับ **AI foundation models**. การลงทุนใน research ด้าน scaling, efficiency, alignment และ ethical AI จะมีบทบาทสำคัญในทศวรรษหน้า.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv 2022 (Chinchilla)</li>
      <li>Rae et al., *Scaling Language Models with Retrieval*, DeepMind 2021 (RETRO)</li>
      <li>Liu et al., *Memformer: Memory-Augmented Transformer for Long-Form Generation*, arXiv 2022</li>
      <li>Alayrac et al., *Flamingo: A Visual Language Model for Few-Shot Learning*, DeepMind 2022</li>
      <li>Zhang et al., *Counterfactual Data Augmentation for Mitigating Gender Bias in Language Translation*, arXiv 2021</li>
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
      การพัฒนา **Sequence-to-Sequence (Seq2Seq) models** โดยใช้สถาปัตยกรรม **Transformer** ได้เปลี่ยน landscape ของการประมวลผลข้อมูลลำดับอย่างสิ้นเชิง — ครอบคลุมทั้ง **Natural Language Processing (NLP), Speech Processing, Multimodal Tasks** และ **Scientific Modeling**. Section นี้จะสรุปประเด็นสำคัญเชิงลึกที่เกิดจากบทเรียน Day55.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">การเปลี่ยน Paradigm: จาก RNN → Transformer Seq2Seq</h3>
    <p>
      การเปลี่ยนผ่านจาก **Recurrent-based Seq2Seq** ไปสู่ **Transformer-based Seq2Seq** นั้นมีผลกระทบในหลายมิติ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ความสามารถในการจัดการ long-range dependencies ดีขึ้นมาก</li>
      <li>Training efficiency สูงขึ้นอย่างมากด้วย **parallelization** ที่ native ต่อ Transformer</li>
      <li>เกิด **emergent capabilities** ที่ Seq2Seq RNN-based models ไม่สามารถให้ได้ เช่น conditional reasoning บน long context</li>
      <li>Seq2Seq Transformer กลายเป็น foundation สำหรับ models ขนาดใหญ่ เช่น T5, mT5, BART</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow mt-8">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Sequence-to-Sequence ด้วย Transformer ไม่ใช่แค่การแทนที่ architecture แบบ RNN. มันเป็น **platform shift** ที่ทำให้วิธีคิดเกี่ยวกับ **representation learning, conditioning, และ controllability** ของ generative models เปลี่ยนไปอย่างสิ้นเชิง. Research ใหม่แสดงให้เห็นว่า **Seq2Seq Transformers** สามารถ perform ใน multimodal tasks ได้อย่างน่าทึ่ง (video → text, text → image captioning) และกำลังกลายเป็นพื้นฐานของ **AGI-scale models** ในอนาคต.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">Key Takeaways</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Transformer-based Seq2Seq มีความ flexible, scalable และ generalizable สูงกว่าสถาปัตยกรรมก่อนหน้า</li>
      <li>Emergent properties เช่น long-context reasoning และ multi-hop reasoning เป็นจุดเด่นที่ยังอยู่ระหว่างการสำรวจ</li>
      <li>Multimodal Seq2Seq จะเป็นเสาหลักของ Foundation Models ยุคต่อไป</li>
      <li>ประเด็นด้าน **Responsible AI** และ **Bias in generation** ต้องถูกออกแบบอย่างรอบคอบสำหรับ production use</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">References</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR 2020</li>
      <li>Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*, ACL 2020</li>
      <li>Brown et al., *Language Models are Few-Shot Learners*, NeurIPS 2020</li>
      <li>Radford et al., *Scaling Laws for Neural Language Models*, arXiv 2020</li>
      <li>Gao et al., *Rethinking Sequence-to-Sequence Learning for Speech Recognition*, Interspeech 2020</li>
    </ul>
  </div>
</section>


   <section id="references" className="mb-16 scroll-mt-32 ">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. References</h2>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      เอกสารอ้างอิงต่อไปนี้ได้ถูกนำมาใช้เพื่อประกอบและเสริมความถูกต้องทางวิชาการของเนื้อหาในบทเรียน **Sequence-to-Sequence with Transformers** ใน Day 55.
    </p>

    <ul className="list-disc list-inside space-y-3 mt-6">
      <li>
        Vaswani, A. et al., *Attention is All You Need*, NeurIPS 2017. 
        [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
      </li>
      <li>
        Raffel, C. et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR 2020. 
        [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
      </li>
      <li>
        Lewis, M. et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*, ACL 2020. 
        [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)
      </li>
      <li>
        Dong, L. et al., *Unified Language Model Pre-training for Natural Language Understanding and Generation*, NeurIPS 2019. 
        [arXiv:1905.03197](https://arxiv.org/abs/1905.03197)
      </li>
      <li>
        Zhang, Y. et al., *Transformer-based Sequence-to-Sequence Models for Automatic Speech Recognition*, arXiv 2020. 
        [arXiv:2005.08100](https://arxiv.org/abs/2005.08100)
      </li>
      <li>
        Aharoni, R. et al., *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges*, ACL 2019.
      </li>
      <li>
        Stanford CS224N Course Notes, *Lecture 12: Machine Translation and Advanced Seq2Seq Models*, Stanford NLP Group, 2021.
      </li>
      <li>
        MIT 6.S191, *Introduction to Deep Learning*, Lecture 9: *Sequence Modeling with Transformers*, 2021.
      </li>
      <li>
        Wang, W. et al., *Fine-tuning Pre-trained Language Models with Noisy Data for Text Generation*, ACL Findings 2021.
      </li>
    </ul>

    <p className="mt-8 text-sm text-gray-500">
      หมายเหตุ: การอ้างอิงถูกจัดเตรียมเพื่อให้ผู้อ่านสามารถศึกษาต่อเชิงลึกได้จากต้นฉบับวิจัยที่ได้รับการยอมรับระดับโลก.
    </p>
  </div>
</section>

   <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day55 theme={theme} />
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
        <ScrollSpy_Ai_Day55 />
      </div>
      
      <SupportMeButton />
    </div>
  );
};

export default Day55_Seq2SeqTransformers;

