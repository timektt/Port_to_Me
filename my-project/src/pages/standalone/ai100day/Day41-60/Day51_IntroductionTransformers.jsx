import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day51 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day51";
import MiniQuiz_Day51 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day51";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day51_IntroductionTransformers = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day51_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day51_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day51_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day51_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day51_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day51_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day51_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day51_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day51_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day51_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day51_11").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 51: Introduction to Transformers</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

<section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้องมี Transformer?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในช่วงก่อนปี 2017 โมเดล deep learning ที่ใช้สำหรับประมวลผลข้อมูลลำดับ เช่น ข้อความ เสียง หรือ time series ส่วนใหญ่พึ่งพาสถาปัตยกรรมแบบ recurrent neural network (RNN) หรือ gated recurrent units (GRU) และ long short-term memory (LSTM) networks อย่างไรก็ตาม RNN-based architecture มีข้อจำกัดที่สำคัญหลายประการ ซึ่งกระทบต่อความสามารถในการสเกลขึ้นสู่ข้อมูลขนาดใหญ่และลำดับยาว
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">ข้อจำกัดของ RNN และ LSTM</h3>
    <p>
      แม้ LSTM และ GRU จะสามารถจัดการกับปัญหา vanishing gradient ได้ดีกว่า RNN ธรรมดา แต่ปัญหาการประมวลผลแบบลำดับ (sequential processing) ก็ยังคงจำกัดประสิทธิภาพ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ไม่สามารถ parallelize การประมวลผลลำดับได้เต็มรูปแบบ</li>
      <li>เวลา training ต่อ batch สูงเมื่อ sequence ยาวขึ้น</li>
      <li>dependency ระยะไกล (long-range dependency) ยังยากต่อการเรียนรู้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">ความต้องการโมเดลที่ scalable และ efficient</h3>
    <p>
      ในยุคที่ข้อมูลมีขนาดใหญ่มากขึ้นเรื่อย ๆ และความซับซ้อนของ task เช่น machine translation, language modeling, และ question answering สูงขึ้น ความต้องการสถาปัตยกรรมที่สามารถเรียนรู้ long-range dependencies ได้อย่างมีประสิทธิภาพจึงเพิ่มสูงขึ้น
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Transformer ได้รับการออกแบบมาเพื่อแก้ปัญหาหลักของ RNN-based architecture โดยใช้ self-attention mechanism ซึ่งช่วยให้สามารถ parallelize การประมวลผลได้เต็มที่ และสามารถเรียนรู้ long-range dependency ได้อย่างมีประสิทธิภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">จุดเปลี่ยนของวงการ NLP และ AI</h3>
    <p>
      การเสนอ Transformer ในปี 2017 โดย Vaswani et al. ใน paper "Attention Is All You Need" ถือเป็นจุดเปลี่ยนสำคัญในวงการ AI โดยเฉพาะใน natural language processing (NLP) โดยโมเดล Transformer ได้กลายเป็น backbone ให้กับโมเดลระดับ state-of-the-art เช่น BERT, GPT series, T5 และ Vision Transformer (ViT) ในด้าน computer vision
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">ความแตกต่างเชิงหลัก</h3>
    <p>สถาปัตยกรรม Transformer มีความแตกต่างจาก RNN-based model ในประเด็นหลักต่อไปนี้:</p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN / LSTM</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Transformer</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Processing Mode</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sequential</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Fully Parallel</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Handling Long Dependencies</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Excellent</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Efficiency</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Slow</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Fast</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">สรุป</h3>
    <p>
      ความสามารถในการ parallelize การประมวลผล การเรียนรู้ long-range dependency อย่างมีประสิทธิภาพ และการนำไปสู่ performance ที่เหนือกว่าบนหลาย benchmark ทำให้ Transformer กลายเป็นแกนหลักของ deep learning ในยุคปัจจุบัน ทั้งใน NLP, computer vision, speech, และ multimodal learning
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers." arXiv:1810.04805.</li>
      <li>Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929.</li>
      <li>Stanford CS224N: Natural Language Processing with Deep Learning.</li>
    </ul>
  </div>
</section>


    <section id="origin-paper" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Paper ต้นกำเนิด</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Transformer architecture ถือกำเนิดจากผลงานวิจัยสำคัญเรื่องหนึ่งในประวัติศาสตร์ของ deep learning และ NLP ได้แก่ paper ที่มีชื่อว่า{" "}
      <strong>“Attention Is All You Need”</strong> โดย Vaswani et al. ซึ่งนำเสนอที่งานประชุม NeurIPS 2017
    </p>
    <p>
      Paper นี้ได้เสนอแนวทางใหม่ในการสร้าง model สำหรับ sequence-to-sequence learning โดยไม่ใช้ recurrent neural network (RNN) หรือ convolutional neural network (CNN) เลย แต่ใช้ self-attention เป็นแกนกลางของการประมวลผลข้อมูลลำดับทั้งหมด
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">บริบทของวงการ ณ ขณะนั้น</h3>
    <p>
      ก่อนปี 2017 การพัฒนาระบบแปลภาษาอัตโนมัติ (Neural Machine Translation, NMT) ส่วนใหญ่ใช้ RNN-based encoder-decoder architecture โดยเฉพาะ LSTM หรือ GRU ซึ่งสามารถจัดการ sequence ได้ดี แต่ยังมีข้อจำกัด:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ไม่สามารถ parallelize training ได้ดี</li>
      <li>ค่า time complexity ต่อ sequence length สูง</li>
      <li>การเรียนรู้ dependency ที่ยาวมากเป็นเรื่องยาก</li>
    </ul>
    <p>
      ใน paper "Attention Is All You Need" ผู้เขียนได้เสนอ model ใหม่ที่สามารถเรียนรู้ข้อมูลลำดับอย่างมีประสิทธิภาพด้วย attention mechanism แบบเต็มตัว โดยไม่ต้องพึ่ง recurrent connection ใด ๆ
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">โครงสร้างและแนวคิดหลักของ paper</h3>
    <p>
      Paper นี้นำเสนอ Transformer architecture ซึ่งประกอบด้วย encoder และ decoder stack ที่แต่ละ layer ใช้ multi-head self-attention และ feedforward network
    </p>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Component</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Role</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ representation ของ token ทุกตัวโดยพิจารณา context จากทั้ง sequence</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multi-Head Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สร้าง attention space หลายมุมมอง ทำให้ model เรียนรู้ dependency ได้หลากหลาย</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Positional Encoding</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ถ่ายทอดข้อมูลเกี่ยวกับลำดับ (order) ของ token ให้กับ model</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Feedforward Network</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม non-linearity และ capacity ให้กับ model</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Paper "Attention Is All You Need" เป็นต้นแบบของสถาปัตยกรรม Transformer ที่กลายเป็นรากฐานของ modern NLP และ AI ในปัจจุบัน โดยกระบวนการ self-attention ช่วยให้ model สามารถประมวลผล long-range dependency ได้อย่างมีประสิทธิภาพ และสามารถ training แบบ parallel ได้เต็มรูปแบบ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">ผลกระทบหลังเผยแพร่ paper</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>กลายเป็น backbone ของโมเดล NLP ชั้นนำ เช่น BERT, GPT, T5</li>
      <li>ถูกนำไปขยายใช้ใน vision (ViT), speech, และ multimodal learning</li>
      <li>เป็นหนึ่งใน paper ที่มีการอ้างอิงมากที่สุดในวงการ AI ทั่วโลก</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Stanford CS224N: Lecture notes and Transformer readings.</li>
      <li>Wolf, T. et al. "Transformers: State-of-the-art Natural Language Processing." arXiv:2005.14165.</li>
      <li>Oxford Deep NLP Course, Lecture 10: Transformer models.</li>
    </ul>
  </div>
</section>


      <section id="architecture-overview" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Architecture Overview ของ Transformer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      สถาปัตยกรรมของ Transformer ถือเป็นหนึ่งในนวัตกรรมที่ทรงอิทธิพลที่สุดในวงการ AI และ NLP โดยแกนหลักคือการใช้{" "}
      <strong>self-attention mechanism</strong> เพื่อให้ model เข้าใจ context ของ sequence ได้อย่างลึกซึ้ง และสามารถเรียนรู้ dependency ได้แบบ global โดยไม่ต้องพึ่ง recurrent connection แบบ RNN
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">ภาพรวมโครงสร้าง</h3>
    <p>
      Transformer ประกอบด้วย 2 ส่วนหลักคือ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Encoder stack</strong> — ใช้แปลง input sequence ให้อยู่ในรูป embedding representation ที่เข้าใจ context</li>
      <li><strong>Decoder stack</strong> — ใช้แปลง embedding ที่ได้ออกมาเป็น output sequence (เช่น ประโยคแปลภาษา หรือ label sequence)</li>
    </ul>
    <p>
      ทั้ง encoder และ decoder ประกอบด้วย stack ของ layers แบบซ้ำ ๆ ซึ่งแต่ละ layer มี sub-layer ที่ออกแบบมาอย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Component หลักของแต่ละ Encoder Layer</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Component</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Function</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multi-Head Self-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ representation ของทุก token โดยอิงจาก token อื่น ๆ ใน sequence เดียวกัน</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Feedforward Neural Network</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม non-linearity และ model capacity ต่อ token แต่ละตัว</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Residual Connection + LayerNorm</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ช่วยแก้ vanishing gradient และทำให้ training มีเสถียรภาพ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">Component หลักของแต่ละ Decoder Layer</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Masked Multi-Head Self-Attention — ป้องกันไม่ให้ decoder มองเห็น future tokens</li>
      <li>Encoder-Decoder Multi-Head Attention — ใช้ attend ข้อมูลจาก encoder output</li>
      <li>Feedforward Neural Network — เช่นเดียวกับ encoder</li>
      <li>Residual Connection + LayerNorm — เช่นเดียวกับ encoder</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">Positional Encoding</h3>
    <p>
      เนื่องจาก self-attention ไม่มีข้อมูลเรื่องลำดับของ token โดยตรง จึงมีการเพิ่ม <strong>positional encoding</strong> เข้าไปใน input embedding เพื่อบอกลำดับเวลา/ลำดับ token
    </p>
    <p>
      Positional encoding สามารถใช้ sine/cosine function หรือเรียนรู้เป็น parameter ก็ได้
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        สถาปัตยกรรม Transformer ได้รับการออกแบบมาเพื่อแก้ปัญหาหลักของ RNN ได้แก่ การ training ช้า, parallelization ต่ำ, และ dependency ระยะไกลที่เรียนรู้ได้ยาก
        โดย self-attention ทำให้ model สามารถเรียนรู้ dependency ทั้ง sequence ได้แบบ global และสามารถ parallelize การ training ได้อย่างมีประสิทธิภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">ข้อดีเชิงโครงสร้าง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Training speed สูง เพราะสามารถ parallelize ได้</li>
      <li>Model capacity ดีเยี่ยม เมื่อ scale ขึ้น</li>
      <li>สามารถจับ long-range dependency ได้ดีกว่า RNN แบบเดิม</li>
      <li>โครงสร้าง modular และเข้าใจได้ง่าย</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Stanford CS224N Lecture 13: Transformer Model Architecture.</li>
      <li>Harvard NLP Annotated Transformer Implementation.</li>
      <li>Oxford Deep NLP Course 2023.</li>
    </ul>
  </div>
</section>


     <section id="self-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Self-Attention Mechanism</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      กลไก Self-Attention เป็นหัวใจสำคัญของ Transformer architecture โดยออกแบบมาเพื่อให้ model สามารถประเมินความสัมพันธ์ระหว่าง tokens ทั้งหมดใน sequence เดียวกันแบบ global ในแต่ละ layer ของ network โดยไม่ต้องใช้ recurrent connection แบบเดิม
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Self-Attention ทำงานอย่างไร?</h3>
    <p>
      แนวคิดพื้นฐานของ self-attention คือ การสร้าง representation ของแต่ละ token โดยพิจารณาข้อมูลจาก token อื่น ๆ ใน sequence ด้วยการคำนวณ similarity (attention score) ระหว่าง tokens และใช้ weighted sum ของ representations เหล่านั้นเพื่ออัปเดต embedding ของ token เป้าหมาย
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">ขั้นตอนหลักของ Self-Attention</h3>
    <ol className="list-decimal list-inside space-y-2">
      <li>เริ่มต้นจากการแปลง input embeddings ของ tokens เป็น <strong>Query (Q)</strong>, <strong>Key (K)</strong>, และ <strong>Value (V)</strong> vectors ผ่าน linear projection</li>
      <li>คำนวณ attention scores โดยใช้ dot-product ระหว่าง Q และ K: 
        <pre className="bg-gray-600 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto mt-2">
{`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`}
        </pre>
      </li>
      <li>นำ attention weights ที่ได้มาใช้เป็น weight ของ value vectors (V) เพื่อสร้าง output representation ของ token แต่ละตัว</li>
    </ol>

    <h3 className="text-xl font-semibold mt-6 mb-3">Multi-Head Self-Attention</h3>
    <p>
      ในทางปฏิบัติ Transformer ใช้ <strong>Multi-Head Attention</strong> แทนที่จะใช้ single attention head เพื่อให้ model เรียนรู้ความสัมพันธ์ที่หลากหลายระหว่าง tokens ได้ดีขึ้น โดยการเรียนรู้หลาย subspaces ของ attention
    </p>

    <p>
      จากนั้น outputs ของแต่ละ head จะถูก concatenate และนำผ่าน linear projection เพื่อสร้าง output ของ layer นั้น
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        จุดแข็งของ Self-Attention คือความสามารถในการ model dependency แบบ global ได้ในเวลา O(1) relative to sequence position
        ซึ่งต่างจาก RNN ที่ต้องเรียนรู้ dependency ผ่าน time steps แบบ sequential
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">การเปรียบเทียบกับ Attention แบบอื่น</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Q, K, V มาจาก</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Use Case</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Input เดียวกัน</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer Encoder/Decoder</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Q จาก Decoder, K/V จาก Encoder</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Machine Translation Decoder</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">ข้อดีของ Self-Attention</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถ model long-range dependency ได้ดีมาก</li>
      <li>รองรับ parallelization เต็มที่ — training เร็วกว่าการใช้ RNN แบบเดิม</li>
      <li>มีความ flexibility สูง — ใช้ได้กับข้อมูลหลายประเภท เช่น NLP, Vision, Speech</li>
      <li>ช่วยเพิ่ม interpretability ผ่าน attention maps</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Stanford CS224N Lecture 14: Self-Attention and Transformer.</li>
      <li>Harvard NLP Annotated Transformer Implementation.</li>
      <li>CMU Neural Networks Course 2023.</li>
    </ul>
  </div>
</section>


  <section id="positional-encoding" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Positional Encoding</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      เนื่องจาก architecture ของ Transformer ไม่มี recurrent หรือ convolutional component
      ซึ่งเป็นโครงสร้างที่ inherently จัดการข้อมูลตามลำดับ (sequential), Transformer จำเป็นต้องเพิ่มข้อมูลเกี่ยวกับลำดับ (position) ของ tokens ใน sequence
      โดยใช้ <strong>Positional Encoding</strong> เพื่อรักษาข้อมูล positional dependency ระหว่าง tokens
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Motivation ของ Positional Encoding</h3>
    <p>
      Self-Attention mechanism ภายใน Transformer มอง tokens ทั้งหมดใน sequence พร้อมกัน (fully parallel) และไม่รับรู้ลำดับของ tokens หากไม่มี signal เพิ่มเติม
      ดังนั้นจึงต้อง inject positional information เข้าไปใน input embeddings เพื่อให้ model เรียนรู้โครงสร้างลำดับ
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การออกแบบ Positional Encoding ที่ดีมีผลอย่างมากต่อ performance ของ Transformer บน tasks ที่ sensitive ต่อลำดับ เช่น language modeling และ translation
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">รูปแบบของ Positional Encoding</h3>
    <p>
      ใน paper "Attention Is All You Need" ใช้ <strong>Sinusoidal Positional Encoding</strong> เป็น baseline approach ซึ่งมีลักษณะ deterministic และไม่ต้องเรียนรู้ parameter เพิ่มเติม
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto mt-2">
{`PE(pos, 2i) = sin(pos / 60000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 60000^(2i/d_model))`}
    </pre>

    <p>
      โดยที่:
      <ul className="list-disc list-inside space-y-2 mt-2">
        <li><code>pos</code>: ตำแหน่งของ token ใน sequence</li>
        <li><code>i</code>: dimension index ของ embedding</li>
        <li><code>d_model</code>: ขนาดของ embedding vector</li>
      </ul>
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">ข้อดีของ Sinusoidal Encoding</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ไม่ต้องเรียนรู้ parameter เพิ่มเติม → ประหยัด memory</li>
      <li>สามารถ generalize ไปยัง sequences ที่ยาวกว่า training sequence ได้ดี</li>
      <li>ง่ายต่อการ implement และ optimize</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">ตัวอย่าง Visual ของ Positional Encoding</h3>
    <p>
      ใน practice, Positional Encoding จะถูก <strong>add</strong> เข้ากับ input embedding ของแต่ละ token ก่อนนำเข้า Self-Attention layer:
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto mt-2">
{`InputEmbedding = TokenEmbedding + PositionalEncoding`}
    </pre>

    <h3 className="text-xl font-semibold mt-6 mb-3">เปรียบเทียบรูปแบบ Positional Encoding</h3>
   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ลักษณะ</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อดี</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อจำกัด</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sinusoidal</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Fixed, deterministic</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generalize well, efficient</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ไม่สามารถ adapt ได้ตาม data</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Learned Embedding</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Trainable parameters</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Flexible, adapts to task</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">อาจไม่ generalize ดีบน sequences ยาวกว่า</td>
      </tr>
    </tbody>
  </table>
</div>

    <h3 className="text-xl font-semibold mt-6 mb-3">การใช้งานจริงใน Transformers รุ่นใหม่</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Transformer ดั้งเดิม (Vaswani et al.) ใช้ sinusoidal encoding</li>
      <li>BERT ใช้ learned positional embedding</li>
      <li>Vision Transformer (ViT) ใช้ learned positional embedding แบบ 2D สำหรับ image patches</li>
      <li>Recent models เช่น Performer และ Perceiver ใช้ hybrid approaches</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Stanford CS224N Lecture 15: Transformer Models.</li>
      <li>Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.</li>
      <li>Harvard NLP Annotated Transformer Implementation.</li>
    </ul>
  </div>
</section>


<section id="ffn" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Feedforward Neural Network (FFN)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ภายในแต่ละ layer ของ Transformer architecture นอกจาก Multi-Head Self-Attention แล้ว ยังมี component ที่สำคัญอีกหนึ่งส่วน คือ <strong>Position-wise Feedforward Neural Network (FFN)</strong> ซึ่งทำหน้าที่ประมวลผล feature representation ที่ได้จาก Self-Attention อีกครั้ง ก่อนส่งต่อไปยัง layer ถัดไป
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Motivation ของ FFN ใน Transformer</h3>
    <p>
      Self-Attention layer สามารถ capture relational dependencies ระหว่าง tokens ได้ดี แต่ feature representation หลังจาก Attention อาจจะยัง linear เกินไป การเพิ่ม FFN ที่มี non-linearity (activation function) เข้ามาช่วยให้ model สามารถเรียนรู้ feature space ที่ซับซ้อนยิ่งขึ้น
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การใช้ Position-wise FFN แบบ identical ทุกตำแหน่ง (position-wise identical) ทำให้ Transformer มีความเป็น parallel สูงมาก และสามารถ scale ไปยัง sequence ยาวได้ดี ต่างจาก recurrent-based architecture
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">สถาปัตยกรรมของ Position-wise FFN</h3>
    <p>
      FFN ที่ใช้ใน Transformer มีโครงสร้างแบบ simple two-layer MLP (Multi-Layer Perceptron) ที่ identical สำหรับทุกตำแหน่งใน sequence:
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto mt-2">
{`FFN(x) = max(0, xW1 + b1)W2 + b2`}
    </pre>

    <p>โดยที่:</p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li><code>x</code>: vector ของ token position i จาก output ของ Self-Attention layer</li>
      <li><code>W1, b1</code>: weight และ bias ของ layer แรก</li>
      <li><code>W2, b2</code>: weight และ bias ของ layer ที่สอง</li>
      <li><code>max(0,·)</code>: ReLU activation</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">Hyperparameter ที่มักใช้</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Parameter</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ค่า Typical ที่ใช้</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">d_model</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">512 / 768 / 1024</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">hidden_size ของ FFN</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2048 / 3072 / 4096</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">activation function</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ReLU, GELU (นิยมมาก)</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">การออกแบบ Position-wise FFN</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>เป็น fully connected layer ที่ identical สำหรับทุกตำแหน่ง (position)</li>
      <li>ไม่ใช้ recurrent connection → fully parallel</li>
      <li>ReLU/GELU activation เพิ่ม non-linearity</li>
      <li>ขนาด hidden layer ใหญ่กว่า d_model → เพิ่ม capacity ในการเรียนรู้ feature complex</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">เปรียบเทียบ FFN กับ Self-Attention</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Component</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">บทบาทหลัก</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">การกระจายการคำนวณ</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ dependency ระหว่าง tokens ทั้งหมด</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Across positions (global)</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">FFN</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เพิ่ม non-linearity และ capacity per position</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Position-wise (independent)</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Stanford CS224N Lecture 16: Transformers and BERT.</li>
      <li>Google Research Blog: Transformer Architecture Explained.</li>
      <li>Harvard NLP Annotated Transformer Guide.</li>
    </ul>
  </div>
</section>


  <section id="residual-layernorm" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Residual Connection & Layer Normalization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ใน Transformer architecture ส่วนประกอบสำคัญที่ช่วยให้โมเดลสามารถฝึกได้ลึก (deep) และเสถียร คือ <strong>Residual Connection</strong> และ <strong>Layer Normalization</strong> ซึ่งจะถูกใช้ในทุก sub-layer ทั้ง Self-Attention และ Feedforward Network
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Residual Connection คืออะไร?</h3>
    <p>
      Residual Connection หรือ <em>Skip Connection</em> คือการนำ input ของ layer เดิม (say, $x$) มาบวก (element-wise addition) กับ output ของ sub-layer ที่กำลังคำนวณอยู่ แล้วจึงส่งต่อไปยังขั้นตอนถัดไป:
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto mt-2">
{`output = LayerNorm(x + Sublayer(x))`}
    </pre>

    <p>
      ประโยชน์ของ Residual Connection คือช่วยให้ model สามารถเรียนรู้ identity mapping ได้ง่ายขึ้น ทำให้ gradient สามารถ flow ย้อนกลับได้ดี ลดปัญหา vanishing gradient และช่วยให้ train network ที่ลึกได้เสถียรกว่า
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Residual Connection ได้รับแรงบันดาลใจมาจาก <strong>ResNet (Residual Networks)</strong> ของ He et al. (2016) ซึ่งประสบความสำเร็จอย่างสูงในการฝึก deep convolutional networks และแนวคิดเดียวกันนี้ได้นำมาใช้ใน Transformer ด้วย
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">Layer Normalization คืออะไร?</h3>
    <p>
      หลังจากมีการบวก Residual แล้ว จะมีการนำผลลัพธ์ไปผ่าน Layer Normalization (LayerNorm) ซึ่งเป็นการ normalize activation ของ layer นั้นแบบ per example:
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto mt-2">
{`LayerNorm(x) = (x - mean) / sqrt(variance + epsilon) * gamma + beta`}
    </pre>

    <p>
      โดย <code>gamma</code> และ <code>beta</code> เป็น learnable parameters
    </p>

    <p>
      Layer Normalization แตกต่างจาก Batch Normalization ตรงที่ normalize activation ต่อ sequence element (ต่อ token) ใน dimension feature ทำให้เหมาะกับงาน sequence modeling ที่ batch size อาจเล็กมาก หรือไม่แน่นอน (เช่นใน NLP)
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Residual Connection + LayerNorm ใน Transformer</h3>
    <p>
      ในหนึ่ง layer ของ Transformer จะมีโครงสร้างตามลำดับนี้:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>Input → Self-Attention → Add & Norm (Residual + LayerNorm)</li>
      <li>Output → Feedforward → Add & Norm (Residual + LayerNorm)</li>
    </ul>

    <p>
      กล่าวคือมี Residual + LayerNorm สองครั้งต่อหนึ่ง Transformer layer
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">เปรียบเทียบกับกรณีไม่มี Residual</h3>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Feature</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">With Residual & LayerNorm</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Without Residual & LayerNorm</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training stability</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีมาก</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ยากต่อการฝึก deep model</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Gradient flow</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีเยี่ยม</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">อาจเกิด vanishing gradient ได้</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Normalization</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่อ token</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ไม่มีการ normalize</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">สรุปบทบาทของ Residual + LayerNorm</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ช่วยให้ gradient flow ได้ดี → โมเดลฝึกได้เร็วขึ้น</li>
      <li>ลดปัญหา internal covariate shift</li>
      <li>ทำให้โมเดลสามารถเรียนรู้ layer ลึกๆ ได้เสถียร</li>
      <li>ช่วย regularize การเรียนรู้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.</li>
      <li>Ba, Jimmy Lei et al. "Layer Normalization." arXiv:1607.06450.</li>
      <li>Stanford CS224N: Lecture on Transformer Architectures.</li>
    </ul>
  </div>
</section>


  <section id="benefits-transformer" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ประโยชน์หลักของ Transformer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Transformer architecture ได้เปลี่ยนโฉมหน้าวงการ Deep Learning ไปอย่างมาก โดยเฉพาะในสาขา NLP, Computer Vision และ Multimodal Learning จุดแข็งของสถาปัตยกรรมนี้เกิดจากการออกแบบที่รองรับการประมวลผลแบบ parallel ได้ดี และสามารถ modeling long-range dependencies ได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">การประมวลผลแบบขนาน (Parallelism)</h3>
    <p>
      ต่างจาก RNN ซึ่งประมวลผล sequence แบบ sequential (token ต่อ token) Transformer ใช้ Self-Attention ซึ่งสามารถ process ทุก token พร้อมกันได้ ส่งผลให้สามารถใช้ GPU/TPU ได้เต็มที่ในการ training
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        Transformer ช่วยให้ training time ลดลงมหาศาลเมื่อเทียบกับ RNN-based models โดยเฉพาะเมื่อ batch size และ sequence length ใหญ่ขึ้น
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">Modeling Long-Range Dependencies ได้ดีกว่า RNN</h3>
    <p>
      ด้วย Self-Attention Mechanism ทุก token สามารถ attend ไปยัง token อื่น ๆ ใน sequence ได้โดยตรง ทำให้สามารถ capture ความสัมพันธ์ระยะไกลได้ดีกว่า RNN/LSTM ที่ gradient อาจหายไปเมื่อ sequence ยาว
    </p>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Transformer</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN / LSTM</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Long-Range Dependency</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีเยี่ยม</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด (Vanishing Gradient)</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelism</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">Scalability และ Model Capacity</h3>
    <p>
      Transformer สามารถ scale ได้ดีมาก ไม่ว่าจะเป็นในเชิงจำนวน layer, hidden dimension หรือ training data volume โมเดลขนาดใหญ่เช่น GPT-3 หรือ PaLM สามารถ train บน corpus ขนาดหลาย TB ได้อย่างมีประสิทธิภาพ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ผลการศึกษาโดย OpenAI และ DeepMind พบว่า scaling law ของ Transformer เป็นแบบ power-law กับ model size และ data volume → การเพิ่มขนาดโมเดล + data จะยังคงเพิ่ม performance ได้
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">ความยืดหยุ่นในการใช้งาน (Flexibility)</h3>
    <p>
      Transformer architecture ถูกออกแบบแบบ modular และ generic สามารถนำไปใช้ได้ทั้งใน NLP, Vision (เช่น Vision Transformer - ViT), Audio, Multimodal tasks รวมถึง Graph Neural Networks ด้วย
    </p>
    <ul className="list-disc list-inside space-y-2 mt-4">
      <li>NLP → Translation, Summarization, QA, Language Modeling</li>
      <li>Vision → Image Classification, Object Detection, Segmentation</li>
      <li>Multimodal → CLIP, DALL·E, Flamingo</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">ง่ายต่อการ Fine-tune</h3>
    <p>
      Transformer-based Pretrained Models เช่น BERT, RoBERTa สามารถ fine-tune ได้ง่ายมากกับ downstream task ต่าง ๆ เพียงเติม head และ train บน dataset ขนาดเล็ก
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">บทสรุป</h3>
    <p>
      ประโยชน์ของ Transformer สะท้อนให้เห็นในความสำเร็จของโมเดล state-of-the-art แทบทุกด้านของ Deep Learning ในปัจจุบัน ตั้งแต่ NLP → Vision → Multimodal ซึ่งเป็นเหตุผลที่ Transformer กลายเป็น architecture ที่สำคัญที่สุดตัวหนึ่งในยุค AI สมัยใหม่
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Kaplan, J. et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361.</li>
      <li>Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929.</li>
      <li>OpenAI Blog: GPT-3 Scaling Insights.</li>
      <li>Stanford CS224N Lectures on Transformer.</li>
    </ul>
  </div>
</section>


  <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Use Cases สำคัญ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      นับตั้งแต่การเปิดตัวของ Paper <em>"Attention Is All You Need"</em> สถาปัตยกรรม Transformer ได้รับการนำไปประยุกต์ใช้อย่างแพร่หลายในงานวิจัยและอุตสาหกรรมระดับโลก ข้อดีเชิง structural ของ Transformer ได้แก่ การรองรับ sequence processing แบบ parallel และ modeling long-range dependencies ส่งผลให้เกิด use cases ที่กว้างขวางในหลากหลาย domain ต่อไปนี้
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">Natural Language Processing (NLP)</h3>
    <p>
      Transformer ได้กลายมาเป็นสถาปัตยกรรมพื้นฐานสำหรับเกือบทุก task ด้าน NLP สมัยใหม่ ทั้ง Pretraining และ Fine-tuning เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>Language Modeling → GPT, GPT-2, GPT-3, GPT-4</li>
      <li>Machine Translation → Transformer, MarianMT</li>
      <li>Question Answering → BERT, RoBERTa</li>
      <li>Text Summarization → BART, T5</li>
      <li>Named Entity Recognition → BERT-based models</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในปี 2022–2023 โมเดล LLM ขนาดใหญ่กว่า 600B parameters ที่ใช้งานเชิงพาณิชย์แทบทั้งหมด (OpenAI, Anthropic, Google DeepMind) ล้วนสร้างบน Transformer architecture เป็นหลัก
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">Computer Vision (CV)</h3>
    <p>
      แม้ CNN จะครองพื้นที่ใน Computer Vision มานาน แต่งานวิจัยเช่น Vision Transformer (ViT) ได้พิสูจน์แล้วว่า Transformer สามารถ outperform CNN ในหลาย benchmark:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>Image Classification → ViT, DeiT</li>
      <li>Object Detection → DETR (DEtection TRansformer)</li>
      <li>Semantic Segmentation → Segmenter, MaskFormer</li>
      <li>Video Understanding → TimeSformer, ViViT</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">Speech Processing</h3>
    <p>
      Transformer ถูกนำมาใช้แทน LSTM/GRU ในหลาย task ด้าน Speech Processing เนื่องจาก Self-Attention สามารถ capture temporal dependency ได้ดี:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>Speech Recognition → Conformer, Transformer-ASR</li>
      <li>Speech Synthesis → FastSpeech, Tacotron 2 + Transformer Decoder</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">Multimodal Learning</h3>
    <p>
      Transformer ถูกใช้เป็น backbone สำหรับงานที่ต้อง integrate data หลายรูปแบบ เช่น image + text, video + audio:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>Vision + Text → CLIP (Contrastive Language-Image Pretraining)</li>
      <li>Text-to-Image Generation → DALL·E, Imagen</li>
      <li>Video + Text → Flamingo</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        CLIP ถือเป็นหนึ่งในตัวอย่างการใช้ Cross-Modal Transformer ที่ประสบความสำเร็จมากที่สุด โดยสามารถเชื่อมโยงความเข้าใจของภาษาและภาพใน latent space เดียวกัน
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">Reinforcement Learning (RL)</h3>
    <p>
      ใน RL งานเช่น Decision Transformer (DT) ได้แสดงให้เห็นว่า sequence modeling ด้วย Transformer สามารถนำมาใช้แทน value-based หรือ policy-based method ได้:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>Offline RL → Decision Transformer</li>
      <li>Behavior Cloning → GPT-like autoregressive modeling</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">Summary: Spectrum ของ Use Cases</h3>
    <p>
      ด้านล่างคือตารางสรุปตัวอย่าง Use Case หลัก ๆ ของ Transformer ในสาขาต่าง ๆ:
    </p>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Domain</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Representative Model</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Application</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">NLP</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">BERT, GPT-4</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Language Understanding & Generation</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Vision</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ViT, DETR</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Image Classification, Detection</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Speech</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Conformer, FastSpeech</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Speech Recognition, Synthesis</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multimodal</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP, Flamingo</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-modal Retrieval & Generation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Radford, A. et al. "Language Models are Few-Shot Learners." arXiv:2005.14165.</li>
      <li>Dosovitskiy, A. et al. "An Image is Worth 16x16 Words." arXiv:2010.11929.</li>
      <li>Carion, N. et al. "End-to-End Object Detection with Transformers." ECCV 2020.</li>
      <li>Reed, S. et al. "A Generalist Agent." DeepMind, 2022.</li>
    </ul>
  </div>
</section>


<section id="research-benchmarks" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Research Benchmarks & Paper Timeline</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การพัฒนาของ Transformer architecture ไม่ได้หยุดอยู่เพียงการแนะนำ Self-Attention ในปี 2017 หากแต่ได้รับการต่อยอดอย่างต่อเนื่องในรูปแบบต่าง ๆ ทั้งเชิงสถาปัตยกรรมและเชิง pretraining strategy การวัดความก้าวหน้าของงานวิจัยเหล่านี้จำเป็นต้องพิจารณาผ่าน **Research Benchmarks** ที่ได้รับการยอมรับในระดับนานาชาติ โดย benchmarks เหล่านี้สะท้อนความสามารถของโมเดล Transformer ใน task ที่หลากหลาย
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">NLP Benchmarks</h3>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li><strong>GLUE (General Language Understanding Evaluation):</strong> ชุด benchmark ที่วัดความเข้าใจภาษาธรรมชาติในหลาย task เช่น entailment, sentiment, sentence similarity</li>
      <li><strong>SuperGLUE:</strong> เวอร์ชันยากขึ้นของ GLUE ถูกใช้วัด LLMs ระดับสูง</li>
      <li><strong>SQuAD (Stanford Question Answering Dataset):</strong> การตอบคำถามจาก paragraph</li>
      <li><strong>XGLUE, XTREME:</strong> Benchmarks สำหรับ Cross-Lingual NLP</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ปัจจุบัน LLM รุ่นใหญ่อย่าง GPT-4, Claude, Gemini สามารถทำคะแนนใกล้เคียง Human Upper Bound บน SuperGLUE ได้แล้ว (≳ 90%) — เป็นสัญญาณว่า NLP โมเดลกำลังเข้าใกล้ระดับมนุษย์ใน task มาตรฐาน
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">Vision Benchmarks</h3>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li><strong>ImageNet:</strong> Benchmark คลาสสิคสำหรับ image classification</li>
      <li><strong>COCO (Common Objects in Context):</strong> Object detection และ segmentation</li>
      <li><strong>Cityscapes:</strong> Semantic segmentation ใน environment ของ urban scene</li>
      <li><strong>Video benchmarks:</strong> Kinetics, Something-Something-v2 สำหรับ Video Understanding</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-3">Speech & Multimodal Benchmarks</h3>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li><strong>LibriSpeech:</strong> Benchmark ยอดนิยมสำหรับ Automatic Speech Recognition (ASR)</li>
      <li><strong>VoxCeleb:</strong> Speaker Recognition</li>
      <li><strong>MSR-VTT, Flickr30k:</strong> Cross-modal retrieval (ภาพ ↔ ข้อความ)</li>
      <li><strong>LAION-400M / LAION-5B:</strong> Training benchmark dataset สำหรับ multimodal LLMs เช่น CLIP, Flamingo</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        สถาปัตยกรรม Transformer ถูกพิสูจน์แล้วว่าสามารถ adapt กับ Benchmark หลักในหลาย domain ได้ โดยไม่จำกัดเฉพาะ NLP เท่านั้น — นี่คือหนึ่งในปัจจัยสำคัญที่ทำให้ Transformer เป็น "Universal Backbone" ของ AI ยุคใหม่
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">Paper Timeline ของ Transformer Ecosystem</h3>
    <p>
      ด้านล่างคือลำดับเหตุการณ์สำคัญในวิวัฒนาการของ Transformer ecosystem:
    </p>

    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Year</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Paper / Contribution</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2017</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Attention Is All You Need → Transformer</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2018</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">BERT → Bidirectional Transformer pretraining</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2019</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GPT-2, RoBERTa → Scaling LMs</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2020</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ViT → Transformer for Vision</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2021</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP → Contrastive multimodal Transformer</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-600 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">2023+</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GPT-4, Gemini, Claude → Large Multimodal Models (LMMs)</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers." NAACL 2019.</li>
      <li>Dosovitskiy, A. et al. "An Image is Worth 16x16 Words." arXiv:2010.11929.</li>
      <li>Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021 (CLIP).</li>
      <li>Brown, T. et al. "Language Models are Few-Shot Learners." arXiv:2005.14165.</li>
    </ul>
  </div>
</section>


   <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Box</h2>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      ในช่วงเวลาเพียงไม่ถึงหนึ่งทศวรรษ Transformer ได้กลายเป็นหัวใจหลักของสถาปัตยกรรม AI สมัยใหม่ ข้าม domain อย่างน่าทึ่ง — จาก NLP, Vision, Speech, ไปจนถึง Multimodal Modeling ความสำเร็จดังกล่าวไม่ใช่เพียงผลจากพลังเชิงคำนวณหรือ dataset ขนาดใหญ่ หากแต่เกิดจากคุณสมบัติเชิงสถาปัตยกรรมที่สอดคล้องกับธรรมชาติของข้อมูลและการเรียนรู้ representation ที่ลึกซึ้ง
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-3">หลักการเชิงลึกจากการศึกษา Transformer</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>**Self-Attention** เปิดโอกาสให้โมเดลสามารถเข้าถึง global context ในทุกตำแหน่งของ sequence ได้แบบ dynamic — ต่างจาก RNN/CNN ที่มีข้อจำกัดเชิงโครงสร้าง</li>
      <li>**Positional Encoding** ทำให้โมเดลเรียนรู้ relative position โดยไม่ต้องใช้ recurrent mechanism</li>
      <li>**Residual Connection + Layer Norm** ช่วยให้ gradient flow ดีขึ้นมาก รองรับ training ของโมเดลขนาดใหญ่</li>
      <li>**Architectural Simplicity** ของ Transformer (stacked attention + FFN) ช่วยให้สามารถ scale ได้อย่างตรงไปตรงมา</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Transformer คือการเปลี่ยนมุมมอง fundamental ต่อ neural architecture — แทนที่จะออกแบบ model โดยยึดติดกับ inductive bias แบบเดิม เช่น convolution หรือ recurrence, Transformer เลือก "ให้ model เรียนรู้ bias ที่เหมาะสมด้วยตัวเอง" ผ่าน Attention Mechanism + Position Encoding ซึ่งทำให้เกิดความสามารถ generalization ข้าม domain อย่างมีประสิทธิภาพสูงสุด
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">สิ่งที่เรียนรู้ได้จาก timeline การพัฒนา Transformer</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Transformer → NLP revolution → BERT/GPT (2017-2019)</li>
      <li>Transformer → Vision revolution → ViT/DeiT/Segmenter (2020+)</li>
      <li>Transformer → Multimodal modeling → CLIP/ALIGN/Flamingo/Perceiver (2021+)</li>
      <li>Transformer → Foundation Models → LLaMA, GPT-4, Claude, Gemini (2023+)</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        แม้ว่า Transformer จะถูกออกแบบมาเพื่อแก้ปัญหา sequence modeling แบบ linear ใน NLP เป็นหลัก แต่ในเวลาต่อมากลับถูกพิสูจน์ว่าเป็น "Universal Function Approximator" ที่เรียนรู้ representation ของข้อมูลหลากหลาย modality ได้อย่างมีประสิทธิภาพ — นี่คือจุดเปลี่ยนสำคัญที่สุดของยุค Deep Learning ปัจจุบัน
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3">อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Dosovitskiy, A. et al. "An Image is Worth 16x16 Words." arXiv:2010.11929.</li>
      <li>Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.</li>
      <li>Bommasani, R. et al. "On the Opportunities and Risks of Foundation Models." Stanford CRFM, 2021.</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day51 theme={theme} />
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
          <div className="mb-10" />
        </div>
      </div>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day51 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day51_IntroductionTransformers;
