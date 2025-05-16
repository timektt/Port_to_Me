import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day33 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day33";
import MiniQuiz_Day33 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day33";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day33_SelfAttention = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

 
  const img1 = cld.image("Day33_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day33_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day33_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day33_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day33_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day33_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day33_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day33_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day33_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day33_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day33_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day33_12").format("auto").quality("auto").resize(scale().width(501));
  const img13 = cld.image("Day33_13").format("auto").quality("auto").resize(scale().width(501));
  const img14 = cld.image("Day33_14").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 33: Self-Attention & Multi-Head Attention</h1>
              <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img1} />
            </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>


<section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้อง Attention?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ข้อจำกัดของการประมวลผลลำดับแบบดั้งเดิม</h3>
    <p>
      ก่อนยุคของ Transformer โมเดลลำดับส่วนใหญ่อาศัยโครงสร้างเชิงลำดับ เช่น Recurrent Neural Networks (RNNs) และ Long Short-Term Memory (LSTM) เพื่อประมวลผลข้อมูลตามลำดับเวลา โมเดลเหล่านี้มีจุดแข็งในด้านการเรียนรู้ความสัมพันธ์ระยะสั้นระหว่าง token แต่เผชิญข้อจำกัดหลายประการ:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>ประมวลผลแบบลำดับต่อเนื่อง (sequential processing) ทำให้ไม่สามารถขนานข้อมูลได้</li>
      <li>เกิดปัญหา vanishing/exploding gradients เมื่อลำดับข้อมูลยาว</li>
      <li>มีข้อจำกัดในการเรียนรู้ dependency ระยะไกลอย่างมีประสิทธิภาพ</li>
    </ul>

    <h3 className="text-xl font-semibold">แรงจูงใจหลักของ Attention Mechanism</h3>
    <p>
      Attention ถูกเสนอเพื่อช่วยให้โมเดลสามารถโฟกัสกับส่วนสำคัญของข้อมูลลำดับโดยไม่จำกัดอยู่กับการประมวลผลแบบ sequential อีกต่อไป แนวคิดคือการให้แต่ละ token สามารถ "มองเห็น" token อื่นทั้งหมดภายใน sequence และประเมินความสัมพันธ์ระหว่างกันผ่านคะแนนความสนใจ (attention scores)
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight Box:</strong> Attention เปลี่ยนแนวคิดจาก “จำทั้งหมดตามลำดับ” → เป็น “เลือกโฟกัสเฉพาะจุดที่สำคัญ” ทำให้โมเดลสามารถเข้าใจบริบทเชิงลึกได้แม้ไม่มีความต่อเนื่องแบบ RNN</p>
    </div>

    <h3 className="text-xl font-semibold">Attention กับ Parallelism</h3>
    <p>
      หนึ่งในข้อดีหลักของ Attention คือความสามารถในการคำนวณแบบขนานอย่างเต็มรูปแบบ เนื่องจากไม่มีการพึ่งพาลำดับก่อนหน้า (unlike RNNs) ทำให้สามารถเร่งความเร็วของการฝึกได้อย่างมีนัยสำคัญ โดยเฉพาะใน GPU/TPU
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>ทุก token ประมวลผลพร้อมกันได้ผ่าน Matrix Multiplication</li>
      <li>สามารถเพิ่มจำนวน layer และ dimension ได้โดยไม่กระทบ time step</li>
      <li>เหมาะกับสถาปัตยกรรมแบบ Deep ที่ต้องการ efficiency</li>
    </ul>

    <h3 className="text-xl font-semibold">ภาพรวมของ Attention ใน NLP และ AI สมัยใหม่</h3>
    <p>
      Attention กลายเป็นแกนกลางของโมเดลภาษา (language model) ชั้นนำ เช่น BERT, GPT, T5, และ LLaMA ซึ่งใช้งานทั้งใน text generation, translation, question answering และ multi-modal learning
    </p>
    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl text-black dark:text-blue-100 border-l-4 border-blue-500">
      <p><strong>Highlight Box:</strong> ใน GPT และ LLaMA ทุก token สามารถเลือกคำที่สัมพันธ์กันมากที่สุดในลำดับก่อนหน้าได้อย่างอิสระ ทำให้สามารถสร้างข้อความที่มีบริบทเชิงลึกได้แม้ในลำดับที่ยาว</p>
    </div>

    <h3 className="text-xl font-semibold">การเปรียบเทียบกับโมเดลดั้งเดิม</h3>
    <div className="overflow-x-auto">
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Aspect</th>
            <th className="border px-4 py-2">RNN/LSTM</th>
            <th className="border px-4 py-2">Attention</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">การประมวลผล</td>
            <td className="border px-4 py-2">ลำดับทีละ token</td>
            <td className="border px-4 py-2">พร้อมกันทุก token</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การเรียนรู้ dependency ไกล</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">ดีมาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การใช้พลังประมวลผล</td>
            <td className="border px-4 py-2">น้อยกว่า</td>
            <td className="border px-4 py-2">มากขึ้น แต่มีประสิทธิภาพสูงกว่า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">วิวัฒนาการสู่ Multi-Head Attention</h3>
    <p>
      การใช้งาน Attention ในรุ่นแรกยังมีข้อจำกัดเรื่องมุมมองเดียว (single-head) ต่อความสัมพันธ์ของข้อมูล เพื่อแก้ปัญหานี้ แนวคิด Multi-Head Attention จึงถูกเสนอเพื่อให้โมเดลเรียนรู้หลายรูปแบบของบริบทพร้อมกัน → ซึ่งจะได้อธิบายเชิงลึกใน section ถัดไป
    </p>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>MIT 6.S191 (2024). Deep Learning for Natural Language Processing.</li>
      <li>Stanford CS224n (2023). <em>Lecture 10: Attention Mechanism</em>.</li>
      <li>Oxford NLP Course – Transformers & Attention Overview.</li>
      <li>CMU Neural Sequence Models (2024 Edition).</li>
    </ul>

  </div>
</section>


<section id="scaled-dot-product" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Scaled Dot-Product Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">นิยามพื้นฐานของ Scaled Dot-Product Attention</h3>
    <p>
      กลไก Scaled Dot-Product Attention เป็นแกนกลางของโมเดล Transformer โดยมีหน้าที่ในการคำนวณความสำคัญของ token ต่าง ๆ ในลำดับผ่านการจับคู่ระหว่างเวกเตอร์ Query (Q), Key (K), และ Value (V) ซึ่งเป็นเวกเตอร์ที่ได้จากการแปลง embedding ของแต่ละ token
    </p>

    <h3 className="text-xl font-semibold">สูตรคำนวณหลัก</h3>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`Attention(Q, K, V) = softmax((QKᵀ) / √d_k) V`}</code>
    </pre>
    <ul className="list-disc ml-6 space-y-1">
      <li><code>Q</code>: เวกเตอร์ Query ของ token ปัจจุบัน</li>
      <li><code>K</code>: เวกเตอร์ Key ของ token อื่น ๆ ทั้งหมด</li>
      <li><code>V</code>: เวกเตอร์ Value ที่เกี่ยวข้องกับ Key</li>
      <li><code>d_k</code>: ขนาดของเวกเตอร์ Key ใช้สำหรับการปรับสเกล</li>
    </ul>

    <h3 className="text-xl font-semibold">เหตุผลของการ Scaled (หารด้วย √dₖ)</h3>
    <p>
      หากไม่ทำการหารด้วย √dₖ ค่าที่ได้จาก dot product (QKᵀ) อาจมีค่าสูงเกินไปเมื่อ dₖ มีขนาดใหญ่ ซึ่งจะทำให้ softmax function เข้าสู่ช่วง saturation (คือให้ attention weight ที่ใกล้ 0 หรือ 1 เกินไป) ส่งผลให้ gradients ระหว่างการเรียนรู้มีขนาดเล็กมากจนไม่สามารถอัปเดตโมเดลได้อย่างมีประสิทธิภาพ
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500 text-black dark:text-blue-100">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: จุดเปลี่ยนจาก Traditional Attention</h3>
      <ul className="list-disc list-inside">
        <li>ก่อนหน้านี้ Attention มักใช้ additive function (Bahdanau et al., 2015)</li>
        <li>Dot-product attention คำนวณเร็วและรองรับ parallelization ได้ดีกว่า</li>
        <li>การ scaling ทำให้โมเดลสามารถเรียนรู้ได้อย่างเสถียรยิ่งขึ้นเมื่อขนาดเวกเตอร์สูง</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การประยุกต์ในโมเดล Transformer</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้ในการจับความสัมพันธ์ระหว่างคำในลำดับเดียวกัน (Self-Attention)</li>
      <li>ใช้ใน cross-attention เพื่อให้ Decoder เข้าใจบริบทจาก Encoder</li>
      <li>เป็นรากฐานของ Multi-Head Attention ซึ่งคำนวณหลาย attention พร้อมกัน</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ: Dot-Product vs Additive Attention</h3>
    <div className="overflow-x-auto">
      <table className="min-w-[600px] border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Dot-Product Attention</th>
            <th className="border px-4 py-2">Additive Attention</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Speed</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำกว่า</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">รองรับ GPU parallel</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✖️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความซับซ้อนทางคณิตศาสตร์</td>
            <td className="border px-4 py-2">ต่ำ</td>
            <td className="border px-4 py-2">สูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: Scaled Dot-Product คือหัวใจของ Self-Attention</h3>
      <p>
        การคำนวณ Scaled Dot-Product เป็นกลไกหลักที่ทำให้ Transformer เข้าใจการเชื่อมโยงเชิงบริบทระหว่างคำในลำดับได้อย่างมีประสิทธิภาพ โดยเฉพาะในงานที่มีบริบทยาว เช่น การแปลภาษา การสรุปข้อความ หรือการตอบคำถาม
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture 9: Attention Mechanisms</li>
      <li>MIT 6.S191 (2023) – Self-Attention and Scaling</li>
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>Oxford Deep NLP – Transformer Module</li>
    </ul>
  </div>
</section>


<section id="self-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Self-Attention คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">นิยามของ Self-Attention</h3>
    <p>
      Self-Attention หรือที่เรียกว่า Intra-Attention เป็นกลไกที่ช่วยให้โมเดลสามารถจับความสัมพันธ์ระหว่าง token แต่ละตัวภายใน sequence เดียวกันได้อย่างมีประสิทธิภาพ แนวคิดหลักคือการให้แต่ละตำแหน่งในลำดับมีโอกาส “สนใจ” ตำแหน่งอื่น ๆ ทั้งหมด โดยพิจารณาความเกี่ยวข้องระหว่างกันผ่านการคำนวณ attention score.
    </p>

    <h3 className="text-xl font-semibold">หลักการทำงานเบื้องต้น</h3>
    <p>
      ใน Self-Attention ทุก token จะถูกแปลงเป็นสามเวกเตอร์: Query (Q), Key (K), และ Value (V). จากนั้นจะคำนวณ dot product ระหว่าง Q และ K เพื่อประเมินความสำคัญ ก่อนจะนำไปใช้คูณกับ V เพื่อได้เวกเตอร์แสดงผลลัพธ์ที่คำนึงถึงบริบทของ token อื่น ๆ.
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`Attention(Q, K, V) = softmax(QK^T / sqrt{d_k}) V`}</code>
    </pre>

    <h3 className="text-xl font-semibold">เหตุผลที่ Self-Attention สำคัญ</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>สามารถเรียนรู้บริบทจากตำแหน่งอื่น ๆ ทั้งหมดในลำดับ</li>
      <li>ประมวลผลได้แบบขนาน (parallel) ไม่เหมือน RNN ที่ต้องประมวลผลทีละ token</li>
      <li>เหมาะกับการเรียนรู้ dependency ทั้งใกล้และไกลได้ในเวลาเดียวกัน</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: จุดเด่นของ Self-Attention</h3>
      <ul className="list-disc list-inside space-y-1">
        <li>สามารถมองเห็นทั้งลำดับพร้อมกัน ไม่จำกัดที่ตำแหน่งใกล้เคียงเท่านั้น</li>
        <li>ปรับน้ำหนักความสำคัญระหว่าง token ได้แบบ dynamic ตามบริบท</li>
        <li>ใช้ memory ได้มีประสิทธิภาพมากกว่าระบบที่ใช้ recurrence</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การใช้งาน Self-Attention ในโมเดลจริง</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Transformer (Vaswani et al., 2017):</strong> เป็นโมเดลแรกที่ใช้ self-attention แทน RNN</li>
      <li><strong>BERT:</strong> ใช้ self-attention แบบ bidirectional เพื่อเข้าใจ context แบบเต็มลำดับ</li>
      <li><strong>GPT:</strong> ใช้ self-attention แบบ unidirectional สำหรับ task ด้านการสร้างภาษา</li>
    </ul>

    <h3 className="text-xl font-semibold">ประสิทธิภาพเชิงโครงสร้าง</h3>
    <p>
      กลไก Self-Attention สามารถลดจำนวน layer ที่จำเป็นสำหรับการเข้าใจข้อมูลลำดับยาวได้เมื่อเทียบกับโมเดลแบบ recurrent และยังสามารถแสดงความสามารถในการเรียนรู้โครงสร้างภายในภาษาที่ซับซ้อน เช่น phrase และ syntax tree
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> Self-Attention ไม่เพียงแต่ทำให้การเรียนรู้ sequence มีประสิทธิภาพสูงขึ้น แต่ยังเปิดทางสู่สถาปัตยกรรมใหม่ที่สามารถตีความ, เข้าใจ และสร้างข้อมูลได้ลึกซึ้งยิ่งขึ้น โดยไม่พึ่งพาโครงสร้างแบบลำดับดั้งเดิม</p>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิงวิชาการ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Devlin et al. (2018). <em>BERT: Pre-training of Deep Bidirectional Transformers</em>. NAACL.</li>
      <li>MIT 6.S191 – Deep Learning for Sequences (2023 Edition)</li>
      <li>Stanford CS224n – Lecture 11: Self-Attention & Transformer Architectures</li>
      <li>Harvard NLP Annotated Transformer – Self-Attention Visualization</li>
    </ul>

  </div>
</section>


<section id="heatmap" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Visualization: Self-Attention Heatmap</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">บทบาทของ Visualization ในการวิเคราะห์ Attention</h3>
    <p>
      การทำ Visualization ของ Self-Attention เป็นหนึ่งในเทคนิคสำคัญที่ช่วยให้สามารถวิเคราะห์การเรียนรู้ของโมเดล Transformer ได้ลึกซึ้งยิ่งขึ้น โดยเฉพาะการสังเกตว่าตำแหน่งของ token แต่ละตัวใน sequence สนใจหรือให้ "น้ำหนัก" กับ token ตัวใดบ้างในขณะประมวลผล
    </p>

    <h3 className="text-xl font-semibold">รูปแบบ Visualization ที่ใช้ในงานวิจัย</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Heatmap ของ attention weights (เช่น <code>QK^T</code> หลัง Softmax)</li>
      <li>การรวม attention หลายหัวเข้าด้วยกันเพื่อดู pattern รวม</li>
      <li>การแยก visualization ตามแต่ละ layer และแต่ละ head</li>
      <li>การใช้ PCA หรือ t-SNE เพื่อ project hidden states ลงใน low-dimension</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่าง Attention Heatmap</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attn_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens, cmap='YlGnBu')
    plt.title("Self-Attention Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()`}</code>
    </pre>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
      <p>
        <strong>Highlight:</strong> การสร้าง heatmap จาก attention weights ทำให้สามารถตีความว่าคำใดมีอิทธิพลต่อการเข้าใจ context ได้ชัดเจน และสามารถตรวจจับข้อบกพร่องของโมเดล เช่น attention collapse หรือ dead head
      </p>
    </div>

    <h3 className="text-xl font-semibold">การใช้งาน Visualization ในโมเดลจริง</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>BERTViz:</strong> ใช้สำหรับแสดง multi-head attention แบบ interactive</li>
      <li><strong>TransformerLens:</strong> ใช้เพื่อ trace และ interpret การไหลของ attention ตลอด layer</li>
      <li><strong>TensorBoard Projector:</strong> ช่วยในการ visualize hidden states และ embedding space</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500">
      <p>
        <strong>Insight:</strong> Visualization ของ Self-Attention ไม่เพียงช่วยในการอธิบายโมเดล แต่ยังช่วยตรวจสอบการเรียนรู้ของระบบในงานจริง ทำให้สามารถปรับปรุง architecture หรือ fine-tune ได้มีประสิทธิภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold">ข้อสังเกตจาก Visualization ที่สำคัญ</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Token ที่มีความสัมพันธ์ระยะไกลอาจมี attention สูงแม้อยู่ห่างกันใน sequence</li>
      <li>บางหัว (head) อาจทำหน้าที่เฉพาะ เช่นโฟกัสที่จุดเริ่มหรือจุดจบของประโยค</li>
      <li>Layer ลึกขึ้นมักโฟกัสกับ semantic dependency มากกว่าความใกล้ทางตำแหน่ง</li>
    </ul>

    <h3 className="text-xl font-semibold">การออกแบบ Experiment สำหรับ Visualization</h3>
    <p>
      ในงานวิจัยจาก Harvard NLP และ MIT 6.S191 การทดลองมักรวม Visualization เป็นส่วนหนึ่งของ pipeline เพื่อประเมินผลลัพธ์การฝึก เช่น การเปรียบเทียบ attention pattern ระหว่าง pretrain vs fine-tune หรือระหว่าง task-specific layer
    </p>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>Stanford CS224n – Lecture on Self-Attention Interpretability</li>
      <li>MIT 6.S191 – Module on Visualizing Transformers (2024)</li>
      <li>Vig, J. (2019). <em>BertViz: Visualizing Attention in Transformer Models</em>, EMNLP.</li>
      <li>Rogers et al. (2020). <em>A Primer in BERTology</em>, arXiv:2002.12327</li>
    </ul>
  </div>
</section>


<section id="multihead" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Multi-Head Attention คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวคิดหลักของ Multi-Head Attention</h3>
    <p>
      Multi-Head Attention (MHA) เป็นหนึ่งในองค์ประกอบหลักของสถาปัตยกรรม Transformer ซึ่งช่วยให้โมเดลสามารถเรียนรู้ความสัมพันธ์ภายในลำดับข้อมูลได้จากหลายมุมมองพร้อมกัน โดยแต่ละ "หัว" ของ attention จะเรียนรู้ pattern ที่แตกต่างกัน และเมื่อรวมผลของหลายหัวเข้าด้วยกันจะช่วยเพิ่มพลังการแสดงออก (expressive power) ของโมเดลอย่างมีนัยสำคัญ
    </p>

    <h3 className="text-xl font-semibold">สูตรคำนวณและการทำงาน</h3>
    <p>
      สมมุติว่า d<sub>model</sub> เป็นขนาดของ embedding vector และแบ่งออกเป็น h หัว แต่ละหัวจะมีขนาด d<sub>k</sub> = d<sub>model</sub> / h:
    </p>

    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O

where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)`}</code>
    </pre>

    <ul className="list-disc ml-6 space-y-1">
      <li>Q, K, V: matrices of queries, keys, and values</li>
      <li>W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>: learned weight matrices for each head</li>
      <li>W<sup>O</sup>: output projection matrix</li>
    </ul>

    <h3 className="text-xl font-semibold">ทำไมต้องมีหลายหัว?</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>แต่ละหัวสามารถโฟกัสกับความสัมพันธ์ระหว่าง token ที่ต่างกัน เช่น บางหัวอาจจับ dependency ใกล้ บางหัวอาจเน้น long-distance</li>
      <li>ช่วยให้โมเดลไม่พึ่งพา bias จากหัวใดหัวหนึ่งเพียงอย่างเดียว</li>
      <li>สามารถเรียนรู้ feature ที่ต่างกัน เช่น structure, semantic, syntactic</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: Multi-View Representation</h3>
      <p>
        แนวคิด Multi-Head ทำให้โมเดลสามารถมองลำดับข้อมูลในมุมที่ต่างกันพร้อมกัน โดยไม่ต้องแยกการประมวลผลหลายรอบ — แต่ละหัวเปรียบเหมือนเลนส์กล้องที่เห็น feature ต่างมิติของข้อมูล sequence เดียวกัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">การใช้ Multi-Head ใน Encoder/Decoder</h3>
    <p>
      ใน Transformer encoder แต่ละ layer ประกอบด้วย self-attention แบบ multi-head ในขณะที่ decoder จะมีทั้ง masked multi-head self-attention และ multi-head encoder-decoder attention เพื่อให้โมเดลจับความสัมพันธ์ทั้งจาก input และ output ได้อย่างแม่นยำ
    </p>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบทางวิศวกรรม</h3>
    <ul className="list-disc ml-6 space-y-1">
      <li>สามารถ parallelize การคำนวณในแต่ละหัวได้ → training เร็วขึ้น</li>
      <li>โมเดลมีการกระจายความรู้ผ่านหลาย subspace ของ embedding</li>
      <li>ปรับขนาดความลึก (depth) และกว้าง (width) ได้อิสระ</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
      <p><strong>Insight:</strong> Multi-Head Attention ไม่ใช่เพียงการทำ attention ซ้ำหลายรอบ แต่เป็นการออกแบบที่เพิ่มความสามารถของโมเดลให้เข้าใจ context ได้รอบด้าน — เป็นรากฐานของ GPT, BERT, T5 และโมเดลสมัยใหม่เกือบทั้งหมด</p>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับ Single-Head</h3>
    <div className="overflow-x-auto">
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Single-Head</th>
            <th className="border px-4 py-2">Multi-Head</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">ความสามารถในการจับหลายความสัมพันธ์</td>
            <td className="border px-4 py-2">ต่ำ</td>
            <td className="border px-4 py-2">สูง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การกระจายความรู้</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">กระจายข้าม subspace</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">สามารถ parallelize ได้</td>
            <td className="border px-4 py-2">บางส่วน</td>
            <td className="border px-4 py-2">✔️ เต็มรูปแบบ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al., 2017 – <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture: Multi-Head Attention Deep Dive</li>
      <li>MIT 6.S191 – Self-Attention and Representation Modules</li>
      <li>Oxford Deep NLP – Understanding Transformer Internals</li>
    </ul>
  </div>
</section>


<section id="why-multihead" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ทำไมต้องมีหลายหัว?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวคิดเบื้องหลังการใช้หลายหัว</h3>
    <p>
      ในสถาปัตยกรรมของ Transformer การใช้ <strong>Multi-Head Attention</strong> ไม่ได้เป็นเพียงกลยุทธ์เพื่อขยายขนาดโมเดล แต่เป็นเทคนิคที่สำคัญเพื่อให้โมเดลสามารถเรียนรู้ความสัมพันธ์ในข้อมูลจากหลายมุมมองพร้อมกันได้ การมีหลายหัว (heads) ใน attention mechanism ช่วยให้โมเดลจับ pattern ได้หลากหลาย โดยแต่ละหัวสามารถโฟกัสบริบทที่แตกต่างกัน
    </p>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบของ Multi-Head Attention</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>เรียนรู้ความสัมพันธ์แบบ local และ global พร้อมกัน</li>
      <li>จับบริบทของคำจากตำแหน่งต่าง ๆ ได้ในเวลาเดียวกัน</li>
      <li>เพิ่ม capacity ของโมเดลโดยไม่ต้องเพิ่มขนาด embedding เดียว</li>
      <li>ป้องกันการ overfitting จากการพึ่งพา attention pattern เดียว</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-blue-500">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ความหลากหลายของบริบท</h3>
      <p>
        หัวแต่ละหัวใน Multi-Head Attention ถูกออกแบบให้เรียนรู้ pattern ที่แตกต่างกัน เช่น หัวหนึ่งอาจโฟกัสความสัมพันธ์ระยะใกล้ (short-range dependency) ขณะที่อีกหัวจับความเชื่อมโยงระยะไกล (long-range dependency) สิ่งนี้ช่วยให้โมเดลสามารถเรียนรู้ความเข้าใจที่ซับซ้อนของภาษาธรรมชาติ
      </p>
    </div>

    <h3 className="text-xl font-semibold">การเปรียบเทียบ Attention แบบ Single และ Multi-Head</h3>
    <div className="overflow-x-auto">
      <table className="min-w-[640px] border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Single-Head</th>
            <th className="border px-4 py-2">Multi-Head</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">จำนวนการเรียนรู้บริบท</td>
            <td className="border px-4 py-2">จำกัด (มุมมองเดียว)</td>
            <td className="border px-4 py-2">หลากหลาย (มุมมองหลายชุด)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ประสิทธิภาพการเรียนรู้</td>
            <td className="border px-4 py-2">น้อยกว่า</td>
            <td className="border px-4 py-2">ดีกว่าในข้อมูลซับซ้อน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การกระจาย focus</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">กระจายได้หลายจุดพร้อมกัน</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างจากโมเดลจริง</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>BERT:</strong> ใช้ multi-head attention 12 หัว (base) หรือ 16 หัว (large)</li>
      <li><strong>GPT-3:</strong> ใช้ attention 96 หัว ใน architecture ขนาดใหญ่</li>
      <li><strong>ViT (Vision Transformer):</strong> ใช้ attention หลายหัวสำหรับการประมวลผล patch ของภาพ</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: Multi-Head คือหัวใจของความเข้าใจลึก</h3>
      <p>
        ในงานวิจัยจาก Stanford (CS224n) และ MIT (6.S191) พบว่า การใช้หลายหัวเป็นปัจจัยสำคัญที่ทำให้โมเดลสามารถ generalize ได้ดีขึ้น และเข้าใจ pattern ที่ซับซ้อนได้มากกว่า single head โดยเฉพาะในบริบทที่มีความคลุมเครือหรือลำดับที่ซับซ้อน
      </p>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิงวิชาการ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Clark et al. (2019). <em>What Does BERT Look at? An Analysis of BERT’s Attention</em>. ACL.</li>
      <li>Dosovitskiy et al. (2020). <em>Image is Worth 16x16 Words: Transformers for Image Recognition</em>. ICLR.</li>
      <li>Stanford CS224n – Lecture: Multi-Head Attention</li>
      <li>MIT 6.S191 – Deep Learning for Understanding Contextual Sequences</li>
    </ul>
  </div>
</section>


<section id="complexity" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. ความซับซ้อนด้านคำนวณ (Complexity)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">พื้นฐานของ Time & Space Complexity ใน Attention</h3>
    <p>
      Self-Attention ใน Transformer นั้นมี time complexity และ space complexity เท่ากับ O(n²d) โดยที่ <code>n</code> คือความยาวของ sequence และ <code>d</code> คือ dimension ของ embedding vector ปัจจัยนี้ทำให้เมื่อ sequence ยาวขึ้น ความต้องการในการประมวลผลจะเพิ่มขึ้นแบบยกกำลังสอง ทั้งในด้านเวลาและหน่วยความจำ
    </p>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Complexity</h3>
    <div className="overflow-x-auto">
      <table className="min-w-[640px] border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Model</th>
            <th className="border px-4 py-2">Time Complexity</th>
            <th className="border px-4 py-2">Memory Complexity</th>
            <th className="border px-4 py-2">Remark</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Transformer (Vanilla)</td>
            <td className="border px-4 py-2">O(n²d)</td>
            <td className="border px-4 py-2">O(n²)</td>
            <td className="border px-4 py-2">Standard attention</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Performer</td>
            <td className="border px-4 py-2">O(nd²)</td>
            <td className="border px-4 py-2">O(nd)</td>
            <td className="border px-4 py-2">Linear attention</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Longformer</td>
            <td className="border px-4 py-2">O(n)</td>
            <td className="border px-4 py-2">O(n)</td>
            <td className="border px-4 py-2">Sparse attention</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ทางเลือกที่พัฒนาเพื่อจัดการกับ Complexity</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Performer:</strong> ใช้ kernel approximation เพื่อลดความซับซ้อนของ attention matrix</li>
      <li><strong>Reformer:</strong> ใช้ locality-sensitive hashing (LSH) เพื่อแบ่งกลุ่ม attention แบบมีประสิทธิภาพ</li>
      <li><strong>Linformer:</strong> บีบ matrix ที่ได้จาก dot-product โดยใช้ linear projection</li>
      <li><strong>Longformer:</strong> ใช้ sliding window attention แทน full attention</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
      <p className="font-semibold">
        Highlight Box: ความท้าทายของ Self-Attention แบบดั้งเดิม คือข้อจำกัดด้านทรัพยากรเมื่อความยาวของ sequence มีขนาดใหญ่ เช่นในเอกสารยาว, genome, หรือ video stream → โมเดลใหม่จึงต้องออกแบบให้มี attention แบบ linear หรือ sparse
      </p>
    </div>

    <h3 className="text-xl font-semibold">ประโยชน์ของการลด Complexity</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>สามารถใช้งานกับ sequence ที่ยาวขึ้นใน hardware ขนาดเล็ก</li>
      <li>ลดค่าใช้จ่ายในการฝึก (training cost) ในงานที่มี dataset ขนาดใหญ่</li>
      <li>เปิดทางสู่ application ใหม่ เช่น real-time processing หรือ deployment บนอุปกรณ์ edge</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
      <p className="font-semibold">
        Insight: แนวโน้มของ Self-Attention สมัยใหม่จะมุ่งสู่ efficiency เป็นหลัก ไม่ใช่แค่ accuracy เท่านั้น โดยเฉพาะในยุคที่มี data ขนาดใหญ่และ model ขนาดใหญ่มาก (Large Foundation Models)
      </p>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Katharopoulos et al. (2020). <em>Transformers are RNNs: Fast and Efficient Attention for Sequence Models</em>. ICML.</li>
      <li>Zaheer et al. (2020). <em>Big Bird: Transformers for Longer Sequences</em>. NeurIPS.</li>
      <li>Beltagy et al. (2020). <em>Longformer: The Long-Document Transformer</em>. arXiv.</li>
      <li>Choromanski et al. (2020). <em>Rethinking Attention with Performers</em>. ICLR.</li>
      <li>MIT 6.S191 (2024). Deep Learning for Structured Data – Attention Efficiency</li>
    </ul>
  </div>
</section>


<section id="position" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ตำแหน่งใน Transformer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">บริบทของตำแหน่งในระบบ Attention</h3>
    <p>
      ในโมเดล Transformer ซึ่งอาศัยกลไก Self-Attention การมีข้อมูลเกี่ยวกับตำแหน่งของ token ภายในลำดับถือเป็นสิ่งจำเป็นอย่างยิ่ง เนื่องจาก Attention Mechanism ไม่มีลำดับโดยธรรมชาติ เหมือนกับ RNN ที่ใช้ลำดับเชิงเวลาเป็นตัวจัดลำดับข้อมูล ดังนั้น การฝังตำแหน่ง (Positional Encoding) จึงถูกนำมาใช้เพื่อให้โมเดลสามารถแยกแยะและเข้าใจลำดับที่แท้จริงของข้อมูลได้
    </p>

    <h3 className="text-xl font-semibold">ตำแหน่งใน Encoder กับ Decoder</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Encoder:</strong> ใช้ Positional Encoding เพื่อเพิ่มข้อมูลตำแหน่งเข้าไปในทุก token embedding ก่อนส่งผ่านเข้าสู่ Multi-Head Self-Attention</li>
      <li><strong>Decoder:</strong> ใช้ Positional Encoding เช่นเดียวกัน แต่มีการใช้ Masked Self-Attention เพื่อป้องกันการเห็น token ในอนาคต (causality)</li>
    </ul>

    <h3 className="text-xl font-semibold">โครงสร้างภายในตำแหน่ง</h3>
    <p>
      ข้อมูลตำแหน่งสามารถฝังลงในเวกเตอร์ได้หลายรูปแบบ เช่น:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>การใช้ฟังก์ชันไซน์/โคไซน์ตามแนวทางของ Vaswani et al. (2017)</li>
      <li>Learned Positional Embedding (ฝึกโดยตรงในกระบวนการเรียนรู้)</li>
      <li>Relative Position Bias (เช่น T5, DeBERTa)</li>
      <li>Rotary Position Encoding (RoPE) ซึ่ง encode ความสัมพันธ์เชิงมุม</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: Self-Attention เข้าใจตำแหน่งจากอะไร?</h3>
      <p>
        โมเดล Self-Attention ไม่ได้มีลำดับโดยเนื้อแท้ หากไม่มีการเพิ่ม Positional Encoding ทุก token จะถูกปฏิบัติเสมือนอยู่ในตำแหน่งเดียวกันในเชิงลำดับ การฝังตำแหน่งจึงเปรียบเสมือนการให้ “แผนที่เชิงลำดับ” แก่ระบบ
      </p>
    </div>

    <h3 className="text-xl font-semibold">การประมวลผลแบบขนาน และผลต่อการฝังตำแหน่ง</h3>
    <p>
      จุดแข็งของ Transformer คือความสามารถในการประมวลผลแบบขนาน (Parallelization) ได้เต็มรูปแบบ ซึ่งตรงข้ามกับ RNN ที่ต้องประมวลผลตามลำดับเวลา การใช้ Positional Encoding ทำให้การเรียนรู้ contextual dependency ไม่สูญเสียข้อมูลด้านลำดับแม้ไม่มี recurrence
    </p>

    <h3 className="text-xl font-semibold">การออกแบบที่ส่งผลต่อความแม่นยำของลำดับ</h3>
    <p>
      การเลือกชนิดของ Positional Encoding ส่งผลต่อการเรียนรู้ลำดับ เช่น:
    </p>
    <table className="min-w-full border text-sm text-left border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">วิธีการ</th>
          <th className="border px-4 py-2">ความแม่นยำลำดับ</th>
          <th className="border px-4 py-2">Generalization</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Sinusoidal Encoding</td>
          <td className="border px-4 py-2">ปานกลาง</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Learned Encoding</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">ต่ำ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Relative Bias</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">ปานกลาง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Rotary (RoPE)</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ตัวอย่างในโมเดลที่มีชื่อเสียง</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>BERT:</strong> ใช้ learned absolute position embedding</li>
      <li><strong>GPT-2/GPT-3:</strong> ใช้ learned + causal masking</li>
      <li><strong>Transformer-XL:</strong> ใช้ relative position encoding</li>
      <li><strong>LLaMA:</strong> ใช้ Rotary Encoding สำหรับ long context</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ลำดับกับความเข้าใจของโมเดล</h3>
      <p>
        ตำแหน่งใน Transformer เปรียบเหมือนระบบแกนพิกัดของความเข้าใจ — โดยการเลือก encoding ที่เหมาะสม จะสามารถนำพาโมเดลให้เข้าใจความสัมพันธ์ระยะไกล, แยกบริบท, และเรียนรู้ลำดับซับซ้อนได้ดีขึ้นโดยไม่สูญเสียความเร็วในการประมวลผล
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>Su et al. (2021). <em>RoFormer: Rotary Position Embedding</em>. arXiv.</li>
      <li>He et al. (2021). <em>DeBERTa: Disentangled Attention</em>. ICLR.</li>
      <li>Stanford CS224n – Lecture on Positional Embedding</li>
      <li>MIT 6.S191 – Deep Learning: Attention and Sequences</li>
    </ul>
  </div>
</section>


<section id="research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. งานวิจัยที่เกี่ยวข้อง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">วิวัฒนาการของการวิจัยด้าน Attention</h3>
    <p>
      งานวิจัยด้าน Attention และ Multi-Head Attention เริ่มต้นอย่างโดดเด่นจากการตีพิมพ์ของ Vaswani et al. (2017) ในบทความ “Attention is All You Need” ซึ่งเป็นรากฐานของ Transformer ทั้งหมด ต่อมา งานวิจัยได้แตกแขนงออกไปในหลายทิศทาง เช่น การปรับปรุงประสิทธิภาพของ Self-Attention, การพัฒนา architecture ใหม่, และการประยุกต์ใช้ Attention ใน modal อื่น ๆ เช่น ภาพและเสียง
    </p>

    <h3 className="text-xl font-semibold">หัวข้อหลักของงานวิจัยในรอบ 5 ปีที่ผ่านมา</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Efficient Attention:</strong> เช่น Linformer, Performer, Longformer เพื่อลดเวลาและหน่วยความจำจาก O(n²) → O(n)</li>
      <li><strong>Compositional Attention:</strong> การออกแบบให้ Multi-Head สามารถประมวลผลบริบทแบบ hierarchical ได้ดีขึ้น</li>
      <li><strong>Cross-Modal Attention:</strong> งานวิจัยที่ใช้ Attention เชื่อมภาพ เสียง และข้อความ เช่น Flamingo, CLIP, Gato</li>
      <li><strong>Sparse Attention:</strong> ออกแบบ mask หรือ routing เพื่อลดการคำนวณแบบ dense เช่น Reformer, Routing Transformer</li>
      <li><strong>Position-aware Attention:</strong> เช่น Relative Positional Encoding (Shaw et al.), RoPE, T5 bias</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg border-l-4 border-blue-500">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: จุดเปลี่ยนที่สำคัญจากงานวิจัย</h3>
      <ul className="list-disc list-inside">
        <li><strong>Transformer</strong> (Vaswani et al., 2017) เป็นจุดเริ่มต้นของยุค Modern Deep Learning</li>
        <li><strong>BERT</strong> และ <strong>GPT</strong> นำ Multi-Head Attention ไปใช้แบบ encoder-only และ decoder-only</li>
        <li><strong>Vision Transformer</strong> (ViT, Dosovitskiy et al.) ขยายแนวคิดไปสู่ภาพโดยตรง</li>
        <li><strong>Perceiver</strong> ใช้ latent cross-attention เพื่อประมวลผลข้อมูลขนาดใหญ่</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่าง Model ที่ใช้ Multi-Head Attention</h3>
    <div className="overflow-x-auto">
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Model</th>
            <th className="border px-4 py-2">ปีที่เผยแพร่</th>
            <th className="border px-4 py-2">จุดเด่น</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Transformer</td>
            <td className="border px-4 py-2">2017</td>
            <td className="border px-4 py-2">ต้นกำเนิด Multi-Head Attention</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">BERT</td>
            <td className="border px-4 py-2">2018</td>
            <td className="border px-4 py-2">ใช้ encoder-only สำหรับ pretraining ภาษา</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GPT-3</td>
            <td className="border px-4 py-2">2020</td>
            <td className="border px-4 py-2">สเกลใหญ่ด้วย decoder-only</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ViT</td>
            <td className="border px-4 py-2">2020</td>
            <td className="border px-4 py-2">ใช้ Attention กับภาพโดยตรง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Perceiver</td>
            <td className="border px-4 py-2">2021</td>
            <td className="border px-4 py-2">ใช้ latent attention สำหรับข้อมูล multimodal</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: จาก NLP สู่ Multimodal</h3>
      <p>
        งานวิจัย Multi-Head Attention ได้ขยายจาก NLP ไปสู่ Computer Vision, Audio, และ Robotics โดยเฉพาะอย่างยิ่งในระบบ Multimodal ที่ต้องเข้าใจข้อมูลจากหลายประเภทพร้อมกัน การที่แต่ละ head เรียนรู้ feature จากมุมมองต่างกันทำให้ Attention กลายเป็นเครื่องมือพื้นฐานของการเรียนรู้เชิงบริบท (contextual learning)
      </p>
    </div>

    <h3 className="text-xl font-semibold">อ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>Devlin et al. (2018). <em>BERT: Pre-training of Deep Bidirectional Transformers</em>. NAACL.</li>
      <li>Brown et al. (2020). <em>Language Models are Few-Shot Learners (GPT-3)</em>. NeurIPS.</li>
      <li>Dosovitskiy et al. (2020). <em>An Image is Worth 16x16 Words: ViT</em>. ICLR.</li>
      <li>Jaegle et al. (2021). <em>Perceiver: General Perception with Latent Attention</em>. ICML.</li>
      <li>Stanford CS224n – Lecture 12: Attention Extensions and Applications</li>
      <li>MIT 6.S191 – Lecture 5: Transformer Architectures and Efficient Attention</li>
    </ul>
  </div>
</section>


<section id="multi-query" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. ปรับปรุงจาก Multi-Head: Multi-Query Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert">

    <h3>10.1 ที่มาของ Multi-Query Attention (MQA)</h3>
    <p>
      การพัฒนาโมเดล Transformer แบบดั้งเดิมอาศัยกลไก Multi-Head Attention เพื่อให้สามารถเรียนรู้การแจกแจงความสัมพันธ์จากหลายมุมมองพร้อมกัน โดยใช้ชุดของ Query, Key และ Value หลายชุด อย่างไรก็ตาม โมเดลที่มีจำนวน Head จำนวนมากส่งผลต่อปริมาณการใช้หน่วยความจำที่สูงมาก โดยเฉพาะในงานที่ต้องประมวลผลข้อความยาวหรือมีคำหลายหมื่นคำ เช่น การแปลภาษาแบบ Real-Time และระบบ Large Language Models
    </p>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> แนวคิดของ Multi-Query Attention ถูกพัฒนาขึ้นเพื่อแก้ปัญหาคอขวดด้านหน่วยความจำของ Multi-Head Attention โดยยังคงคุณสมบัติการจับความสัมพันธ์จากหลาย Query ได้อย่างมีประสิทธิภาพ
    </div>

    <h3>10.2 หลักการทำงานของ Multi-Query Attention</h3>
    <p>
      ต่างจาก Multi-Head Attention ที่ใช้ Key และ Value หลายชุดในแต่ละ Head นั้น Multi-Query Attention ใช้ Query หลายชุด แต่ใช้ Key และ Value ชุดเดียวร่วมกันสำหรับทุก Head ซึ่งช่วยลดการใช้หน่วยความจำและเพิ่มประสิทธิภาพของการประมวลผลได้อย่างมีนัยสำคัญ
    </p>

    <pre><code className="language-python">
// โครงสร้างเชิงเปรียบเทียบ
Multi-Head:        Q1,K1,V1 | Q2,K2,V2 | Q3,K3,V3 | ...
Multi-Query:       Q1,K,V   | Q2,K,V   | Q3,K,V   | ...
    </code></pre>

    <ul className="list-disc list-inside my-4">
      <li>Query: ยังคงแยกเป็นหลายชุดเหมือนเดิม</li>
      <li>Key และ Value: ใช้ชุดเดียวร่วมกันในทุก Query</li>
      <li>ลดความซับซ้อนของการ Broadcasting และ Memory Access</li>
    </ul>

    <h3>10.3 ข้อดีหลักของ Multi-Query Attention</h3>
    <ul className="list-decimal list-inside space-y-2">
      <li><strong>ลดภาระหน่วยความจำ:</strong> ลดจำนวน Key/Value projections เหลือเพียง 1 ชุด แทนที่จะมี n ชุดใน Multi-Head</li>
      <li><strong>ความเร็วสูงขึ้น:</strong> การลดจำนวน tensor ทำให้สามารถรันบน GPU ได้รวดเร็วขึ้น</li>
      <li><strong>รองรับงาน Long-Context:</strong> ช่วยให้โมเดลสามารถประมวลผล sequence ยาวขึ้นได้อย่างมีประสิทธิภาพ เช่นการประมวลผลเอกสารหรือคำบรรยาย</li>
    </ul>

    <div className="bg-blue-800 p-4 rounded-xl my-6 border-l-4 border-blue-400">
      <strong>Highlight:</strong> จากการทดสอบใน LLM อย่าง PaLM ของ Google พบว่า Multi-Query Attention ช่วยลด latency ได้ถึง 2 เท่าในบางงาน โดยยังคงคุณภาพของคำตอบไว้ได้ใกล้เคียงกับ Multi-Head ดั้งเดิม
    </div>

    <h3>10.4 ความแตกต่างระหว่าง Multi-Query และ Multi-Head</h3>
    <table className="table-auto w-full text-sm text-left my-6 border border-gray-800">
      <thead className="bg-gray-700">
        <tr>
          <th className="px-4 py-2 border">คุณสมบัติ</th>
          <th className="px-4 py-2 border">Multi-Head Attention</th>
          <th className="px-4 py-2 border">Multi-Query Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border">จำนวน Key/Value per head</td>
          <td className="px-4 py-2 border">หลายชุด</td>
          <td className="px-4 py-2 border">ชุดเดียว</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">ประสิทธิภาพ GPU</td>
          <td className="px-4 py-2 border">ช้า (มากขึ้นตามจำนวน head)</td>
          <td className="px-4 py-2 border">เร็วขึ้นอย่างมีนัยสำคัญ</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">ความเหมาะสมกับ LLM</td>
          <td className="px-4 py-2 border">ไม่เหมาะกับ input ยาวมาก</td>
          <td className="px-4 py-2 border">เหมาะกับ context ยาวมาก</td>
        </tr>
      </tbody>
    </table>

    <h3>10.5 ใช้งานจริงในระบบขนาดใหญ่</h3>
    <p>
      Google Research ได้นำ Multi-Query Attention ไปใช้ในโมเดล PaLM และ Gemini ซึ่งต้องประมวลผล context ขนาดยาวหลายพัน token และต้องตอบกลับแบบ real-time การออกแบบดังกล่าวสามารถลด time-to-first-token ได้มากกว่า 40% โดยที่การเปลี่ยนจาก Multi-Head ไปเป็น Multi-Query มีผลกระทบกับคุณภาพคำตอบเพียงเล็กน้อยเท่านั้น
    </p>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> แนวคิดการใช้ Shared Key/Value ยังเป็นแนวทางสำคัญที่ต่อยอดไปสู่ Hybrid-Query และ Grouped-Query Attention ในอนาคต
    </div>

    <h3>10.6 ข้อจำกัดของ Multi-Query Attention</h3>
    <ul className="list-disc list-inside">
      <li>อาจทำให้ขาดความหลากหลายของ Value projection เมื่อ context มีความหลากหลายสูง</li>
      <li>ไม่เหมาะกับบาง task ที่ต้องการ relational attention หลายมิติ</li>
    </ul>

    <p className="mt-6">แม้จะมีข้อจำกัด แต่ Multi-Query Attention ก็เป็นทางเลือกที่เหมาะสมสำหรับการใช้งานในระบบ real-time, edge device, และการฝึก LLM ที่ต้องการลดต้นทุน</p>

    <h3>10.7 งานวิจัยที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside">
      <li>Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need" – Google AI</li>
      <li>Chowdhery, A. et al. (2022). "PaLM: Scaling Language Models with Pathways" – arXiv:2204.02311</li>
      <li>Rajpurkar, P. et al. (Stanford). "Scaling Laws for LLMs in Low-Resource Contexts"</li>
    </ul>

  </div>
</section>


<section id="tip" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Practical Tip</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert">

    <h3>11.1 การเลือกใช้ Attention Type ที่เหมาะสมกับ Task</h3>
    <p>
      การเลือกกลไก Attention ให้เหมาะสมกับลักษณะของงานเป็นหนึ่งในปัจจัยสำคัญที่ส่งผลต่อประสิทธิภาพของโมเดล ไม่ว่าจะเป็นงานด้านภาษา การวิเคราะห์เอกสารยาว หรือการโต้ตอบแบบเรียลไทม์ โดยนักวิจัยจาก Stanford และ Google แนะนำว่าควรพิจารณาจากปัจจัยต่อไปนี้:
    </p>

    <ul className="list-disc list-inside my-4">
      <li><strong>Sequence Length:</strong> สำหรับข้อความที่ยาวมาก Multi-Query หรือ Grouped Attention เหมาะสมกว่า</li>
      <li><strong>Context Diversity:</strong> หากความสัมพันธ์ระหว่าง token มีความซับซ้อนสูง ควรใช้ Multi-Head</li>
      <li><strong>Inference Speed:</strong> งานที่ต้องการความเร็ว เช่น Chatbot, ควรใช้ Multi-Query หรือ Linear Attention</li>
    </ul>

    <div className="bg-blue-800 p-4 rounded-xl my-6 border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานของ Meta AI ใน LLaMA ใช้ Multi-Query Attention เพื่อลดเวลา latency ในการให้คำตอบแบบ Interactive ได้มากกว่า 30% เมื่อเทียบกับ Multi-Head
    </div>

    <h3>11.2 การใช้ Flash Attention และ Memory-efficient Tricks</h3>
    <p>
      Flash Attention เป็นเทคนิคที่ใช้ CUDA kernel พิเศษในการลดการอ่านข้อมูลซ้ำในหน่วยความจำ ซึ่งช่วยให้การคำนวณ Softmax Attention มีประสิทธิภาพสูงขึ้น เหมาะสำหรับระบบที่มีทรัพยากรจำกัดหรือ LLM ขนาดใหญ่
    </p>

    <pre><code className="language-javascript">
// Pseudocode สำหรับ Flash Attention
for each block Q:
  for each block K, V:
    score = Q * K^T
    prob = softmax(score)
    context = prob * V
    </code></pre>

    <ul className="list-disc list-inside">
      <li>ลด Memory Footprint ได้มากกว่า 50%</li>
      <li>เหมาะสำหรับการเทรนและ Inference LLM บน A100 หรือ H100</li>
    </ul>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> Flash Attention ถูกใช้ในโมเดล GPT-NeoX, Falcon และ Mistral โดยเฉพาะสำหรับ Long Context Attention
    </div>

    <h3>11.3 การปรับ Position Embedding ให้เหมาะสม</h3>
    <p>
      แม้ Positional Encoding แบบ sinusoidal จะเป็นค่ามาตรฐานของ Transformer แต่ในงานที่มีลำดับไม่แน่นอน เช่นการทำ Summarization หรือ Q&A อาจเลือกใช้ Relative Position Encoding หรือ Rotary Position Embedding (RoPE) เพื่อเพิ่มประสิทธิภาพ
    </p>

    <table className="table-auto w-full text-sm text-left my-6 border border-gray-300">
      <thead className="bg-gray-800">
        <tr>
          <th className="px-4 py-2 border">Encoding Type</th>
          <th className="px-4 py-2 border">จุดเด่น</th>
          <th className="px-4 py-2 border">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border">Sinusoidal</td>
          <td className="px-4 py-2 border">ไม่ต้องเรียนรู้ เพิ่มความทั่วไป</td>
          <td className="px-4 py-2 border">ไม่รองรับ context แบบ dynamic</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Relative</td>
          <td className="px-4 py-2 border">เข้าใจความสัมพันธ์ในลำดับได้แม่นยำ</td>
          <td className="px-4 py-2 border">เพิ่มความซับซ้อนของโมเดล</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">RoPE</td>
          <td className="px-4 py-2 border">ใช้ในโมเดล GPT รุ่นใหม่</td>
          <td className="px-4 py-2 border">มีการจำกัดความยาวของ context</td>
        </tr>
      </tbody>
    </table>

    <h3>11.4 เคล็ดลับการใช้งานในงานจริง</h3>
    <ul className="list-decimal list-inside my-4 space-y-2">
      <li><strong>ใช้ Multi-Query + Flash Attention:</strong> สำหรับ Chatbot หรือ LLM ที่ทำงานแบบ Streaming</li>
      <li><strong>ใช้ Relative Encoding:</strong> สำหรับงานที่ต้องการความเข้าใจลำดับ เช่นการสรุปบทความหรือคำอธิบายภาพ</li>
      <li><strong>ใช้ Checkpointing:</strong> แบ่ง Layer เพื่อลด Memory Usage ขณะฝึกโมเดลขนาดใหญ่</li>
      <li><strong>รองรับ Low-Rank Approximation:</strong> เช่นใช้ Linformer หรือ Performer สำหรับระบบที่มีขีดจำกัดด้านทรัพยากร</li>
    </ul>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> จากงานของ Carnegie Mellon พบว่าโมเดลที่ผสาน Multi-Query + Flash Attention + RoPE ให้ความแม่นยำใกล้เคียงกับ LLM ขนาดใหญ่ ขณะที่ใช้ parameter น้อยลงถึง 35%
    </div>

    <h3>11.5 งานวิจัยที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside">
      <li>Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" – arXiv:2205.14135</li>
      <li>Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding" – arXiv:2104.09864</li>
      <li>Huang, L. et al. (2020). "Improving Transformer Models by Reordering their Sublayers" – Stanford NLP</li>
    </ul>

  </div>
</section>


<section id="insight1" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert">

    <h3>12.1 บทบาทของ Multi-Query Attention ต่อทิศทางของโมเดลในอนาคต</h3>
    <p>
      จากการพัฒนาอย่างต่อเนื่องของโมเดลภาษาขนาดใหญ่ (LLM) เช่น GPT, PaLM, LLaMA และ Gemini ได้แสดงให้เห็นว่าการเปลี่ยนแปลงเล็กน้อยในสถาปัตยกรรม Attention สามารถส่งผลต่อ scalability, cost-efficiency และ latency ได้อย่างมีนัยสำคัญ โดยเฉพาะ Multi-Query Attention ซึ่งกลายเป็นหนึ่งในแกนกลางของความก้าวหน้าใน LLM ยุคใหม่
    </p>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> Multi-Query Attention ทำหน้าที่เป็นสะพานเชื่อมระหว่างระบบที่เน้นประสิทธิภาพกับระบบที่เน้นความแม่นยำระดับสูง เป็นแนวทางกลางที่รองรับการ scale และ real-time deployment ได้พร้อมกัน
    </div>

    <h3>12.2 วิวัฒนาการของแนวคิด: จาก Multi-Head → Multi-Query → Hybrid Query</h3>
    <ul className="list-decimal list-inside space-y-2">
      <li><strong>Multi-Head:</strong> เริ่มต้นจากแนวคิดว่าการใช้มุมมองหลากหลาย (หลาย head) ช่วยให้เข้าใจความสัมพันธ์ของข้อมูลได้ลึกขึ้น</li>
      <li><strong>Multi-Query:</strong> ปรับแนวทางโดยลดการใช้ resource ในส่วนของ Key/Value แต่ยังคงใช้ query หลากหลาย</li>
      <li><strong>Hybrid-Query:</strong> แนวคิดล่าสุดที่พยายามเลือกใช้ Multi หรือ Single Query ตามลักษณะ task แบบ dynamic ในแต่ละ layer</li>
    </ul>

    <table className="table-auto w-full text-sm text-left my-6 border border-gray-300">
      <thead className="bg-gray-800">
        <tr>
          <th className="px-4 py-2 border">กลไก Attention</th>
          <th className="px-4 py-2 border">ความสามารถในการ Generalize</th>
          <th className="px-4 py-2 border">การใช้ Memory</th>
          <th className="px-4 py-2 border">ความเร็วขณะ Inference</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border">Multi-Head</td>
          <td className="px-4 py-2 border">สูง</td>
          <td className="px-4 py-2 border">สูง</td>
          <td className="px-4 py-2 border">ช้า</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Multi-Query</td>
          <td className="px-4 py-2 border">กลาง</td>
          <td className="px-4 py-2 border">ต่ำ</td>
          <td className="px-4 py-2 border">เร็ว</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Hybrid</td>
          <td className="px-4 py-2 border">สูง (adaptive)</td>
          <td className="px-4 py-2 border">กลาง</td>
          <td className="px-4 py-2 border">เร็ว</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-800 p-4 rounded-xl my-6 border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานของ Microsoft Research แนะนำว่าการฝึกแบบ Hybrid Query แบบ layer-wise ช่วยให้ Transformer ประหยัดพลังงานในการ inference ได้ถึง 23% โดยไม่ลด accuracy ใน downstream tasks
    </div>

    <h3>12.3 ความสัมพันธ์กับ Retrieval-Augmented Generation (RAG)</h3>
    <p>
      RAG เป็นหนึ่งในเทคนิคสำคัญที่ใช้เชื่อม LLM กับฐานข้อมูลภายนอก โดยในการออกแบบระบบที่ใช้ Retrieval-based Context เช่น Bing AI, Claude หรือ GPT-4 Turbo นั้น Multi-Query Attention ช่วยลดระยะเวลาในการประมวลผล Context ขนาดใหญ่มหาศาลที่ดึงมาจากระบบ vector database
    </p>

    <ul className="list-disc list-inside">
      <li>ลด Latency ในการรวมเอกสารหลายแหล่งเข้าด้วยกัน</li>
      <li>เหมาะกับระบบที่ต้องให้คำตอบแบบ Open-ended เช่น Copilot หรือ Agent</li>
    </ul>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> การออกแบบระบบที่ผสาน RAG + Multi-Query สามารถใช้ memory เพียง 60% ของระบบดั้งเดิม แต่ให้ผลลัพธ์ที่มี contextual grounding สูงกว่า
    </div>

    <h3>12.4 การขยายแนวคิด Multi-Query สู่โมเดล Vision และ Multimodal</h3>
    <p>
      งานของ Google DeepMind และ MIT พบว่า Multi-Query Attention ไม่ได้จำกัดอยู่เพียง NLP เท่านั้น แต่สามารถปรับใช้ในงาน Multimodal เช่น Gemini และ Flamingo ได้เช่นกัน โดยลดความซ้ำซ้อนของการประมวลภาพและข้อความพร้อมกัน
    </p>

    <ul className="list-decimal list-inside">
      <li>ลด redundancy ใน cross-modal attention</li>
      <li>เร่งความเร็วในการ align ข้อมูลภาพและข้อความ</li>
      <li>รองรับ sequence ที่ผสมทั้ง text + image + video ได้ดีขึ้น</li>
    </ul>

    <h3>12.5 งานวิจัยที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside">
      <li>Shazeer, N. (2023). "Multi-Query Attention with Global Key/Value Sharing" – Google Research</li>
      <li>Chen, A. et al. (2022). "On the Efficiency of Attention Mechanisms for Large Context Windows" – arXiv</li>
      <li>Yu, W. et al. (2023). "Cross-Modal Efficiency with Shared Queries in Flamingo" – DeepMind</li>
      <li>Peng, B. et al. (2023). "RAG Fusion with MQA: Scaling Retrieval-based LLMs" – Microsoft Research</li>
    </ul>

  </div>
</section>


<section id="insight2" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">13. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert">

    <h3>13.1 การเปลี่ยนกรอบคิดจาก “ความแม่นยำ” สู่ “ประสิทธิภาพแบบยั่งยืน”</h3>
    <p>
      งานพัฒนาโมเดลภาษาขนาดใหญ่ (LLMs) ไม่ได้เน้นเพียงการเพิ่มพารามิเตอร์หรือการสร้างผลลัพธ์ที่แม่นยำขึ้นเพียงอย่างเดียวอีกต่อไป แต่เริ่มเน้นไปที่การพัฒนาอย่างมีประสิทธิภาพ (Efficient AI) ซึ่ง Multi-Query Attention เป็นหนึ่งในกลไกที่เป็นตัวแทนของปรัชญาดังกล่าว
    </p>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> Multi-Query Attention ช่วยให้ LLMs มีพฤติกรรมใกล้เคียง Real-Time ได้มากขึ้น โดยไม่ต้องแลกกับคุณภาพของคำตอบในระดับงานวิจัย
    </div>

    <h3>13.2 ทิศทางของสถาปัตยกรรม Neural Attention ยุคถัดไป</h3>
    <p>
      นักวิจัยจาก MIT และ Meta AI คาดการณ์ว่าในอีก 1–2 ปีข้างหน้า โมเดลขนาดใหญ่จะหันมาใช้ “Compositional Attention” เป็นหลัก ซึ่งรวมแนวคิดจาก Multi-Query, Cross-Layer Sharing, และ Block-Sparse Mechanism เข้าด้วยกัน
    </p>

    <ul className="list-disc list-inside">
      <li>ใช้ Multi-Query เฉพาะ Layer ต้นและปลาย เพื่อควบคุม Cost</li>
      <li>ใช้ Multi-Head เฉพาะ Layer กลางที่ต้องการ High Interaction</li>
      <li>เปิดทางให้กลไก Attention แบบ Plug-in ได้ในอนาคต (เช่น Reinforcement-Guided Attention)</li>
    </ul>

    <div className="bg-blue-800 p-4 rounded-xl my-6 border-l-4 border-blue-400">
      <strong>Highlight:</strong> สถาปัตยกรรมแบบ Hybrid Attention จะเป็นมาตรฐานใหม่ของ LLMs ที่ต้องรองรับ context ขนาดใหญ่ โดยไม่ลดทอนความเร็วในการ inference
    </div>

    <h3>13.3 ผลกระทบต่อโลกของ AI และระบบจริงในองค์กร</h3>
    <p>
      ความก้าวหน้าของกลไก Attention ที่เน้นประสิทธิภาพ กำลังทำให้โมเดลขนาดใหญ่อย่าง GPT-4, Claude, Gemini สามารถ deploy ได้ในระบบ production จริง ทั้งในองค์กรระดับ Fortune 500 และระบบงานวิจัยชั้นนำ โดยไม่ต้องใช้เซิร์ฟเวอร์ขนาดใหญ่แบบเมื่อก่อน
    </p>

    <ul className="list-decimal list-inside">
      <li>ลดต้นทุน inference ลงได้กว่า 40–60% ในบางระบบ</li>
      <li>ลดการใช้พลังงาน (Carbon-Aware AI) ซึ่งสอดคล้องกับแนวทางของ Google และ Meta</li>
      <li>เปิดโอกาสให้ประเทศกำลังพัฒนาสามารถใช้โมเดล LLM ได้ในต้นทุนที่ต่ำลง</li>
    </ul>

    <h3>13.4 การเปรียบเทียบวิสัยทัศน์ของผู้นำ AI</h3>
    <table className="table-auto w-full text-sm text-left my-6 border border-gray-300">
      <thead className="bg-gray-800">
        <tr>
          <th className="px-4 py-2 border">องค์กร</th>
          <th className="px-4 py-2 border">แนวทางการพัฒนา Attention</th>
          <th className="px-4 py-2 border">เป้าหมาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border">Google DeepMind</td>
          <td className="px-4 py-2 border">Multi-Query + RoPE + Routing</td>
          <td className="px-4 py-2 border">LLM ที่ใช้พลังงานน้อยลง</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">OpenAI</td>
          <td className="px-4 py-2 border">Hybrid Attention (GPT-4 Turbo)</td>
          <td className="px-4 py-2 border">Context 128K+ พร้อม Real-time</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Anthropic</td>
          <td className="px-4 py-2 border">Multi-Query ร่วมกับ Constitutional AI</td>
          <td className="px-4 py-2 border">LLM ที่มีความปลอดภัยเชิงจริยธรรม</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-800 p-4 rounded-xl my-6 border-l-4 border-yellow-400">
      <strong>Insight:</strong> Multi-Query ไม่ได้เป็นเพียงเทคนิคด้านวิศวกรรม แต่คือ "ปรัชญาการออกแบบ" ใหม่ที่ผลักดันแนวคิดของ Efficient AI ไปสู่วงกว้างทั่วโลก
    </div>

    <h3>13.5 งานวิจัยและบทวิเคราะห์ที่ควรอ่านเพิ่มเติม</h3>
    <ul className="list-disc list-inside">
      <li>Kaplan, J. et al. (OpenAI). "Scaling Laws for Neural Language Models" – arXiv:2001.08361</li>
      <li>Chen, A. et al. (MIT). "Efficient Attention Mechanisms for Deployment at Scale"</li>
      <li>Taylor, R. et al. (DeepMind). "Pathways Language Model and Attention Strategies"</li>
      <li>Meta AI. "Towards Carbon-Efficient LLMs for Planet-Scale Deployment"</li>
    </ul>

  </div>
</section>




          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day33 theme={theme} />
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
        <ScrollSpy_Ai_Day33 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day33_SelfAttention;
