import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day30 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day30";
import MiniQuiz_Day30 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day30";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day30_AttentionClassic = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day30_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day30_2").format("auto").quality("auto").resize(scale().width(490));
  const img3 = cld.image("Day30_3").format("auto").quality("auto").resize(scale().width(490));
  const img4 = cld.image("Day30_4").format("auto").quality("auto").resize(scale().width(490));
  const img5 = cld.image("Day30_5").format("auto").quality("auto").resize(scale().width(490));
  const img6 = cld.image("Day30_6").format("auto").quality("auto").resize(scale().width(490));
  const img7 = cld.image("Day30_7").format("auto").quality("auto").resize(scale().width(490));
  const img8 = cld.image("Day30_8").format("auto").quality("auto").resize(scale().width(490));
  const img9 = cld.image("Day30_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day30_10").format("auto").quality("auto").resize(scale().width(400));
  const img11 = cld.image("Day30_11").format("auto").quality("auto").resize(scale().width(499));
  const img12 = cld.image("Day30_12").format("auto").quality("auto").resize(scale().width(490));
  const img13 = cld.image("Day30_13").format("auto").quality("auto").resize(scale().width(500));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 30: Attention Mechanisms (Classic)</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          {/* Placeholder: Section content will be added below */}

<section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. Introduction: ปัญหาของ RNN/Seq2Seq ที่ Attention มาช่วยแก้</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-8">
    <h3 className="text-xl font-semibold">ลักษณะจำกัดของ Encoder–Decoder แบบดั้งเดิม</h3>
    <p>
      โมเดล Sequence-to-Sequence (Seq2Seq) แบบดั้งเดิม ซึ่งมีโครงสร้างหลักประกอบด้วย Encoder และ Decoder ที่เชื่อมกันผ่าน context vector เดียว (fixed-length vector) ได้แสดงให้เห็นข้อจำกัดอย่างมีนัยสำคัญ โดยเฉพาะเมื่อจัดการกับลำดับข้อมูลที่มีความยาวมาก เช่น การแปลประโยคยาว หรือการสรุปเนื้อหาเอกสารหลายย่อหน้า
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>Context vector เดียวไม่สามารถเก็บข้อมูลทั้งหมดของ input sequence ได้อย่างมีประสิทธิภาพ</li>
      <li>เมื่อ input sequence ยาวขึ้น ประสิทธิภาพในการแปลหรือทำนายจะลดลงอย่างรวดเร็ว (information bottleneck)</li>
      <li>Decoder ไม่สามารถเข้าถึงข้อมูลตำแหน่งใด ๆ ใน input ได้โดยตรง ยกเว้นผ่าน vector เดียวที่ Encoder ส่งมา</li>
    </ul>

    <h3 className="text-xl font-semibold">การเกิดขึ้นของ Attention Mechanism</h3>
    <p>
      Attention ถูกเสนอโดย Bahdanau et al. (2014) เพื่อลดข้อจำกัดของการใช้ context vector เดียว โดยเสนอแนวทางใหม่ให้ Decoder สามารถเลือก "โฟกัส" ไปยังตำแหน่งที่สำคัญใน Encoder โดยตรง ทำให้ Decoder ได้รับบริบทที่เฉพาะเจาะจงในแต่ละ timestep แทนที่จะต้องพึ่งพาค่าเฉลี่ยหรือ vector เดียว
    </p>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>โมเดล Seq2Seq แบบดั้งเดิมใช้ context vector คงที่ ส่งผลให้สูญเสียบริบทบางส่วน โดยเฉพาะในลำดับยาว</li>
        <li>Attention Mechanism ช่วยให้ Decoder เข้าถึง Encoder hidden states แต่ละตำแหน่งโดยตรง และอย่างยืดหยุ่น</li>
        <li>แนวคิดนี้เป็นรากฐานสำคัญที่นำไปสู่การพัฒนาโมเดลระดับสูงอย่าง Transformer และ BERT</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ความแตกต่างระหว่าง Context Vector แบบเดิมและแบบ Attention</h3>
    <table className="table-auto w-full border border-gray-300 dark:border-gray-600 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Fixed Context Vector</th>
          <th className="border px-4 py-2">Attention Context Vector</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ลักษณะ</td>
          <td className="border px-4 py-2">เวกเตอร์เดียวที่สรุปข้อมูลทั้งหมด</td>
          <td className="border px-4 py-2">เวกเตอร์ใหม่ในแต่ละ timestep จาก weighted sum</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การประมวลผล</td>
          <td className="border px-4 py-2">ไม่มีการเลือกตำแหน่ง</td>
          <td className="border px-4 py-2">สามารถเลือกโฟกัสตำแหน่งใน input ได้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ประสิทธิภาพเมื่อ input ยาว</td>
          <td className="border px-4 py-2">ลดลง</td>
          <td className="border px-4 py-2">คงที่หรือดีขึ้น</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">แนวคิดเบื้องหลัง Attention ที่กลายเป็นมาตรฐานใหม่</h3>
    <p>
      หลังจากแนวคิดของ Bahdanau ได้รับการยอมรับอย่างแพร่หลาย Luong et al. (2015) ได้เสนอ attention ที่มีประสิทธิภาพมากขึ้น และง่ายต่อการ implement เช่น dot-product attention ซึ่งภายหลังกลายเป็นพื้นฐานของ Transformer
    </p>
    <p>
      ปัจจุบัน ไม่มีโมเดลใดในระดับ state-of-the-art ที่ไม่ใช้ attention ไม่ว่าทางตรงหรือทางอ้อม โดยเฉพาะในงาน NLP, Vision, และ Speech
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: เมื่อ context ไม่เพียงพอ</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Encoder–Decoder แบบเดิมมีข้อจำกัดคล้าย memory หนึ่งบรรทัดที่ต้องจำทุกอย่าง</li>
        <li>Attention เพิ่มความยืดหยุ่นในการดึงข้อมูล → เปรียบเหมือนการอ่านหลายบรรทัดพร้อมกัน</li>
        <li>ในภาษาธรรมชาติ บริบทของคำที่อยู่ไกล เช่นคำแรกกับคำสุดท้าย อาจมีความสัมพันธ์กัน ซึ่ง Attention รองรับได้ดีกว่า</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ข้อค้นพบจากการทดลองของมหาวิทยาลัยชั้นนำ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Bahdanau et al. (2014): การเพิ่ม attention layer ช่วยให้ BLEU score ของ NMT ดีขึ้นมาก</li>
      <li>Luong et al. (2015): เปรียบเทียบ score functions ใน attention ชี้ให้เห็นว่า dot-product เร็วกว่า additive ใน large-scale setting</li>
      <li>Stanford CS224n: Lecture 9 วิเคราะห์การล่มของ Seq2Seq บนประโยคยาว และการฟื้นฟูด้วย attention</li>
      <li>MIT 6.S191: แสดงให้เห็นว่า attention ช่วยลด gradient decay ใน long sequences ได้อย่างชัดเจน</li>
    </ul>
  </div>
</section>


       <section id="concept" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. แนวคิดของ Attention: โฟกัสแบบเลือกตำแหน่ง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">จาก Context Vector เดียวสู่การเลือกตำแหน่งแบบไดนามิก</h3>
    <p>
      แนวคิดของ Attention ในบริบทของ Neural Machine Translation (NMT) หรือ Sequence-to-Sequence (Seq2Seq) คือการเปลี่ยนวิธีการสร้าง context vector จากแบบคงที่ (fixed-length) เป็นแบบปรับเปลี่ยนได้ตาม timestep ของ decoder โดยแต่ละตำแหน่งของ decoder จะคำนวณน้ำหนัก (weights) เพื่อกำหนดความสำคัญของแต่ละ encoder hidden state → นำมาสร้าง context vector ที่เฉพาะเจาะจงกับตำแหน่งนั้น
    </p>

    <h3 className="text-xl font-semibold">การคำนวณ Attention Context Vector</h3>
    <p>กระบวนการหลักประกอบด้วย:</p>
    <ol className="list-decimal ml-6 space-y-2">
      <li>คำนวณ score ระหว่าง decoder hidden state ปัจจุบัน <code>s_t</code> กับ encoder hidden states <code>h_i</code></li>
      <li>นำ score แต่ละตำแหน่งผ่าน softmax → ได้ attention weights <code>a_ti</code></li>
      <li>ใช้ weights คูณ encoder states แล้วรวมกัน → ได้ context vector <code>c_t</code></li>
    </ol>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto text-sm">
<code>{`score(s_t, h_i) → a_ti = softmax(score)
c_t = Σ a_ti * h_i`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ลักษณะเด่นของ Attention เทียบกับโครงสร้างดั้งเดิม</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">มิติ</th>
          <th className="border px-4 py-2">Context Vector แบบเดิม</th>
          <th className="border px-4 py-2">Context Vector แบบ Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การเลือกตำแหน่ง</td>
          <td className="border px-4 py-2">คงที่</td>
          <td className="border px-4 py-2">เปลี่ยนแปลงได้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การจัดการข้อมูลลำดับยาว</td>
          <td className="border px-4 py-2">ประสิทธิภาพลดลง</td>
          <td className="border px-4 py-2">ประสิทธิภาพคงที่/เพิ่มขึ้น</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความสามารถเชิงบริบท</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">เลือกบริบทได้</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Attention เปลี่ยน paradigm จาก "one-size-fits-all" เป็น "ตำแหน่งต่อเฉพาะตำแหน่ง" (position-specific context)</li>
        <li>ช่วยลดปัญหา long-term dependency ได้ดีโดยไม่ต้องเพิ่ม layer ความลึก</li>
        <li>เป็นแนวคิดที่เปิดทางสู่โมเดล non-recurrent เช่น Transformer</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">มุมมองจากงานวิจัยและมหาวิทยาลัยชั้นนำ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Bahdanau et al. (2014) – เสนอแนวคิดการ align ระหว่าง encoder กับ decoder โดยไม่จำกัดบริบท</li>
      <li>Stanford CS224n – แสดงตัวอย่างว่า Attention ช่วยให้ BLEU score ดีขึ้น และเข้าใจภาษาได้แม่นยำกว่าเดิม</li>
      <li>Harvard NLP – วิเคราะห์การเรียนรู้ attention weights ว่าแสดงผลลัพธ์ที่ตีความได้เป็นลักษณะ heatmap</li>
      <li>Oxford Deep NLP – ยกตัวอย่าง contextual embedding ที่ได้จาก attention เทียบกับ non-attentive encoding</li>
      <li>MIT 6.S191 – อธิบายว่าการใช้ attention ช่วยลดการกระจายของ gradient ใน sequence ที่ยาว</li>
    </ul>

  </div>
</section>

   <section id="seq2seq" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Attention in Sequence-to-Sequence Models</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ภาพรวมของ Sequence-to-Sequence Architecture</h3>
    <p>
      Sequence-to-Sequence (Seq2Seq) คือสถาปัตยกรรมที่ใช้กันอย่างแพร่หลายในงานแปลภาษา, การสรุปข้อความ, และคำสั่งเสียง โดยประกอบด้วยสองส่วนหลักคือ Encoder และ Decoder ซึ่งทำงานร่วมกันผ่าน context vector ที่ส่งต่อจาก encoder ไปยัง decoder อย่างไรก็ตามแนวทางนี้มีข้อจำกัดเมื่อประโยคมีความยาวมาก ส่งผลให้ข้อมูลสูญหายบางส่วน
    </p>

    <h3 className="text-xl font-semibold">การแทรก Attention ลงใน Seq2Seq</h3>
    <p>
      กลไก Attention ถูกพัฒนาเพื่อให้ Decoder สามารถเข้าถึง hidden state ของ Encoder ได้ทุกตำแหน่ง แทนที่จะพึ่งเพียง context vector เดียวที่สกัดมาจากขั้นตอนสุดท้ายของ Encoder วิธีนี้ช่วยให้โมเดลมีความสามารถในการพิจารณาบริบทของข้อมูลที่สำคัญในแต่ละ timestep ได้ดีขึ้น
    </p>
    
    <h3 className="text-xl font-semibold">โครงสร้างข้อมูลในระบบ Seq2Seq + Attention</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Input sequence:</strong> <code>[x₁, x₂, ..., x_T]</code> → ผ่าน Encoder → ได้ hidden states <code>[h₁, h₂, ..., h_T]</code></li>
      <li><strong>Decoder timestep t:</strong> สร้าง <code>sᵗ</code> แล้วคำนวณ attention weights สำหรับทุก <code>hᵢ</code></li>
      <li><strong>Context vector:</strong> <code>cᵗ = Σ aᵗᵢ * hᵢ</code> ซึ่งใช้สำหรับสร้าง output ที่ timestep นั้น</li>
    </ul>

    <h3 className="text-xl font-semibold">ฟังก์ชันการให้คะแนน (Score Function)</h3>
    <p>
      หนึ่งในหัวใจของ Attention คือการคำนวณ score ระหว่าง decoder state <code>sᵗ</code> กับ encoder hidden state <code>hⱼ</code> เพื่อนำมาสร้าง attention weights วิธีการมีหลายรูปแบบ เช่น:
    </p>

    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2 text-left">Score Type</th>
          <th className="border px-4 py-2 text-left">Formula</th>
          <th className="border px-4 py-2 text-left">Properties</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Dot-product</td>
          <td className="border px-4 py-2"><code>sᵗ · hⱼ</code></td>
          <td className="border px-4 py-2">Simple, efficient</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Additive</td>
          <td className="border px-4 py-2"><code>vᵀ tanh(W₁sᵗ + W₂hⱼ)</code></td>
          <td className="border px-4 py-2">More expressive, slower</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ความสัมพันธ์ของ Context กับ Output</h3>
    <p>
      Context vector ที่ได้จาก Attention จะถูกนำมารวมกับ decoder state <code>sᵗ</code> เพื่อผลิต output ต่อไป เช่น คำที่ต้องแปลใน NMT โมเดลจึงสามารถกำหนดได้ว่าควรให้ความสำคัญกับ input คำไหนในแต่ละจังหวะ
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ในโมเดล Seq2Seq ดั้งเดิม output ถูกสร้างโดยพิจารณาจาก context vector เดียว</li>
        <li>Attention ทำให้ context เปลี่ยนแปลงได้ตาม timestep ทำให้ Decoder มีความยืดหยุ่นมากขึ้น</li>
        <li>เหมาะกับ task ที่ต้องเข้าใจบริบทยาว เช่น Translation, Summarization, Speech</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">สรุปภาพรวมการทำงานด้วย Code Pseudo</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
{`for each decoder step t:
  for each encoder output h_j:
    score_tj = compute_score(s_t, h_j)
  attention_weights = softmax(score_tj)
  context_vector = sum(attention_weights * h_j)
  output_t = generate(s_t, context_vector)`}
    </pre>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Bahdanau et al. (2014). <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. arXiv:1409.0473</li>
      <li>Luong et al. (2015). <em>Effective Approaches to Attention-based Neural Machine Translation</em>. arXiv:1508.04025</li>
      <li>Stanford CS224n – Lecture 9: Attention and Seq2Seq with Alignment</li>
      <li>Oxford Deep NLP Course – Seq2Seq Architectures with Attention</li>
      <li>Harvard NLP Annotated Transformer – Mechanisms of Context Alignment</li>
    </ul>
  </div>
</section>


        <section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. ประเภทของ Attention แบบ Classic</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">4.1 Additive Attention (Bahdanau et al., 2014)</h3>
    <p>
      Additive Attention ถูกเสนอโดย Bahdanau et al. เพื่อช่วยให้โมเดล Neural Machine Translation (NMT) เรียนรู้การจัดตำแหน่งคำใน input กับ output ได้โดยอัตโนมัติ โดยใช้ feedforward neural network เพื่อคำนวณความคล้ายระหว่าง decoder state ปัจจุบันกับแต่ละ encoder hidden state
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded text-sm overflow-x-auto">
<code>{`score(s_t, h_j) = vᵀ tanh(W₁ s_t + W₂ h_j)`}</code>
    </pre>

    <ul className="list-disc ml-6 space-y-2">
      <li>สามารถ capture ความสัมพันธ์แบบ non-linear</li>
      <li>เหมาะกับ sequence ที่ซับซ้อนหรือภาษาที่มี structure ซับซ้อน</li>
      <li>แต่คำนวณช้ากว่า dot-product ใน training ขนาดใหญ่</li>
    </ul>

    <h3 className="text-xl font-semibold">4.2 Dot-Product Attention (Luong et al., 2015)</h3>
    <p>
      Luong et al. ได้เสนอวิธีที่เร็วกว่าโดยการใช้ dot-product แทน feedforward network ในการวัดความคล้ายระหว่าง states ซึ่งมีความง่ายกว่า และเหมาะกับงานขนาดใหญ่ที่ต้องการความเร็วในการคำนวณสูง
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded text-sm overflow-x-auto">
<code>{`score(s_t, h_j) = s_t · h_j`}</code>
    </pre>

    <ul className="list-disc ml-6 space-y-2">
      <li>คำนวณเร็วมากโดยใช้ matrix multiplication</li>
      <li>เมื่อ vectors ถูก normalized (เช่นใน Transformer) จะมีความแม่นยำใกล้เคียงกับ additive</li>
      <li>เป็นพื้นฐานของ self-attention ที่ปรากฏใน Transformer</li>
    </ul>

    <h3 className="text-xl font-semibold">4.3 Scaled Dot-Product Attention</h3>
    <p>
      ในการใช้งานบน vector ขนาดใหญ่ ค่า dot-product จะมีค่ามาก → softmax กลายเป็น function ที่แบนเกินไป Transformer จึงใช้การ scaling เพื่อลดผลกระทบดังกล่าว โดยหารด้วย √d (ขนาดของ vector)
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded text-sm overflow-x-auto">
<code>{`score(Q, K) = (Q · Kᵀ) / √d`}</code>
    </pre>

    <p>
      Scaled Dot-Product Attention ได้กลายเป็นมาตรฐานในการใช้งานใน Transformer ซึ่งใช้ในทุกระดับของการประมวลผลลำดับ เช่น ใน GPT, BERT และ ViT
    </p>

   <h3 className="text-xl font-semibold">การเปรียบเทียบระหว่าง Attention Types</h3>

<div className="w-full overflow-x-auto">
  <table className="min-w-[640px] border border-gray-300 dark:border-gray-700 text-sm text-left">
    <thead className="bg-gray-100 dark:bg-gray-800">
      <tr>
        <th className="border px-4 py-2">Type</th>
        <th className="border px-4 py-2">Score Function</th>
        <th className="border px-4 py-2">Speed</th>
        <th className="border px-4 py-2">Expressiveness</th>
        <th className="border px-4 py-2">Use Case</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Additive</td>
        <td className="border px-4 py-2 whitespace-nowrap">
          <code>vᵀ tanh(W₁s + W₂h)</code>
        </td>
        <td className="border px-4 py-2">ช้า</td>
        <td className="border px-4 py-2">สูง</td>
        <td className="border px-4 py-2">NMT, language tasks</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Dot-Product</td>
        <td className="border px-4 py-2 whitespace-nowrap">
          <code>s · h</code>
        </td>
        <td className="border px-4 py-2">เร็ว</td>
        <td className="border px-4 py-2">ปานกลาง</td>
        <td className="border px-4 py-2">Large-scale NLP</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Scaled Dot</td>
        <td className="border px-4 py-2 whitespace-nowrap">
          <code>(QKᵀ)/√d</code>
        </td>
        <td className="border px-4 py-2">เร็วมาก</td>
        <td className="border px-4 py-2">สูง</td>
        <td className="border px-4 py-2">Transformer models</td>
      </tr>
    </tbody>
  </table>
</div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: การเลือกประเภท Attention</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ถ้าความเร็วสำคัญ → Dot-product เหมาะที่สุด</li>
        <li>ถ้าข้อมูลมีโครงสร้างซับซ้อน → Additive ให้ performance ดีกว่า</li>
        <li>สำหรับงานโมเดลใหม่ เช่น GPT, BERT → Scaled Dot-Product คือมาตรฐาน</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Bahdanau et al. (2014) – Neural Machine Translation by Jointly Learning to Align and Translate (arXiv:1409.0473)</li>
      <li>Luong et al. (2015) – Effective Approaches to Attention-based Neural Machine Translation (arXiv:1508.04025)</li>
      <li>Vaswani et al. (2017) – Attention Is All You Need (NeurIPS)</li>
      <li>Stanford CS224n – Lecture: Attention Mechanisms</li>
      <li>MIT 6.S191 – Deep Learning Series: Alignment & Attention</li>
    </ul>
  </div>
</section>


         <section id="math" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. คณิตศาสตร์ของ Attention Step-by-Step</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ขั้นตอนหลักในการคำนวณ Attention</h3>
    <p>
      กลไก Attention ทำงานโดยการเปรียบเทียบ “query” จาก decoder กับ “keys” จาก encoder เพื่อสร้าง “weights” ที่แสดงระดับความสำคัญของ encoder state แต่ละตำแหน่ง แล้วนำไปสร้าง context vector ที่ใช้ร่วมกับ output ของ decoder ในแต่ละ timestep
    </p>

    <ol className="list-decimal ml-6 space-y-2">
      <li><strong>Score:</strong> คำนวณความคล้ายระหว่าง query และ key แต่ละคู่</li>
      <li><strong>Weights:</strong> ใช้ softmax กับค่าคะแนนเพื่อแปลงเป็น distribution</li>
      <li><strong>Context:</strong> ใช้ weights เหล่านี้สร้าง vector สรุปข้อมูลจาก encoder</li>
      <li><strong>Combine:</strong> รวม context vector กับ decoder state เพื่อสร้าง output</li>
    </ol>

    <h3 className="text-xl font-semibold">สูตรทางคณิตศาสตร์</h3>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`Score:    eᵢ = score(sᵗ, hᵢ)
Weights:  aᵢ = softmax(eᵢ)
Context:  cᵗ = Σ aᵢ * hᵢ
Output:   yᵗ = f(cᵗ, sᵗ)`}</code>
    </pre>

    <h3 className="text-xl font-semibold">การคำนวณแบบ Matrix Form</h3>
    <p>
      เพื่อเพิ่มประสิทธิภาพในการฝึกโมเดลในระดับ GPU หรือ TPU การคำนวณ attention มักแปลงเป็นรูปแบบ Matrix Multiplication:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`Q = queries  (from decoder)
K = keys     (from encoder)
V = values   (usually same as K)

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ตัวอย่างการคำนวณ Attention Weights</h3>
    <p>
      สมมติว่ามี decoder state <code>sᵗ</code> และ encoder states <code>[h₁, h₂, h₃]</code>:
    </p>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`sᵗ = [0.5, 1.2]
h₁ = [0.1, 0.8]
h₂ = [0.4, 1.0]
h₃ = [0.9, 0.3]

score = dot(sᵗ, hᵢ) = Σ sᵗⱼ * hᵢⱼ
→ [1.06, 1.48, 0.93]
softmax → [0.290, 0.443, 0.267]`}</code>
    </pre>

    <p>
      Context vector = weighted sum ของ h₁, h₂, h₃ ด้วย weights ข้างต้น
    </p>

    <h3 className="text-xl font-semibold">ภาพรวมกราฟิกจากการวิเคราะห์ของ Stanford</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Encoder สร้าง hidden states หลายตำแหน่ง</li>
      <li>Decoder ใช้แต่ละ hidden state พร้อมกับ state ปัจจุบันเพื่อให้ “score”</li>
      <li>Softmax แปลงเป็น probability distribution</li>
      <li>Weighted sum จาก distribution ใช้เป็น context</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ความเข้าใจในเชิงคณิตศาสตร์</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Softmax ทำหน้าที่คัดเลือก “บริบทสำคัญ” จาก encoder</li>
        <li>Dot product เหมาะกับระบบความจำสูงและประสิทธิภาพ GPU</li>
        <li>การ normalize โดย √d ป้องกัน gradient ที่รุนแรงเกินไป</li>
        <li>Attention สามารถเรียนรู้ “ความสัมพันธ์” ที่อยู่ไกลในลำดับได้อย่างยืดหยุ่น</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงเชิงคณิตศาสตร์</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Bahdanau et al. (2014). <em>Neural Machine Translation by Jointly Learning to Align and Translate</em></li>
      <li>Luong et al. (2015). <em>Effective Approaches to Attention-based NMT</em></li>
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em> – NeurIPS</li>
      <li>Stanford CS224n – Lecture 9: Attention Step-by-Step</li>
      <li>MIT 6.S191 – Deep Learning 2024: Sequence Models and Gradients</li>
    </ul>
  </div>
</section>

        <section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Visualization & Diagram</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ความสำคัญของการ Visualize Attention</h3>
    <p>
      Visualization ของ Attention เป็นหนึ่งในเครื่องมือหลักที่ช่วยให้สามารถตีความพฤติกรรมของโมเดลลำดับได้ โดยเฉพาะอย่างยิ่งในงานที่มีการแปลภาษา หรือการแมปข้อมูลข้ามลำดับ เช่นการจับคู่คำในประโยค input กับคำแปลใน output การแสดงผลแบบ heatmap หรือ matrix ทำให้เห็นว่าระบบให้ “น้ำหนัก” กับข้อมูลตำแหน่งใดมากที่สุดในแต่ละ timestep
    </p>

    <h3 className="text-xl font-semibold">การใช้ Attention Heatmap</h3>
    <p>
      Heatmap เป็น visualization ที่แสดงความเข้มของค่าความสนใจในแต่ละตำแหน่ง encoder–decoder ตัวอย่างเช่น:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>แกน x = คำในประโยคต้นฉบับ (input)</li>
      <li>แกน y = คำในประโยคเป้าหมาย (output)</li>
      <li>ค่าสีเข้ม = ให้ attention สูง</li>
    </ul>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`Input:  [I, love, machine, learning]
Output: [J'aime, l'apprentissage, automatique]

Heatmap:
  J'aime           → [0.8, 0.1, 0.05, 0.05]
  l'apprentissage  → [0.05, 0.05, 0.5, 0.4]
  automatique      → [0.1, 0.05, 0.25, 0.6]`}</code>
    </pre>

    <h3 className="text-xl font-semibold">กรณีศึกษาจาก NLP Research</h3>
    <p>
      จากการวิเคราะห์ของ Harvard NLP และงานวิจัยโดย Google Research พบว่า Attention Visualization สามารถใช้ตรวจสอบความผิดพลาดของระบบ NMT ได้ เช่น กรณีที่ attention กระจายตัวมากเกินไป (low confidence) หรือล็อกตำแหน่งผิด (alignment errors)
    </p>

    <h3 className="text-xl font-semibold">ประโยชน์เชิงวิเคราะห์</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ตรวจสอบว่าโมเดล “เข้าใจ” ความสัมพันธ์ในลำดับหรือไม่</li>
      <li>ตรวจจับ bias เช่น ให้ attention กับคำเดิมซ้ำ ๆ</li>
      <li>ช่วยสร้างความน่าเชื่อถือ (model interpretability)</li>
      <li>ใช้ในระบบ real-time สำหรับ debugging หรือ visual analytics</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: การอ่าน Attention อย่างถูกต้อง</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Attention ที่คมและเฉพาะเจาะจง แสดงถึงความเข้าใจเชิงบริบทที่ดี</li>
        <li>Attention ที่แบนหรือกระจาย อาจเกิดจาก overfitting หรือ context vector ไม่ชัดเจน</li>
        <li>กรณี NMT ที่ดี คำใน output ควรสอดคล้องกับคำต้นทางที่สัมพันธ์กันทางไวยากรณ์และความหมาย</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">วิธีการสร้าง Attention Visualization</h3>
    <ol className="list-decimal ml-6 space-y-2">
      <li>ดึง attention weights ขนาด (output_len × input_len)</li>
      <li>แปลงเป็นภาพ 2D matrix</li>
      <li>ใช้ library เช่น matplotlib, seaborn, หรือ plotly</li>
      <li>แสดงผลโดยมีคำในแนวแกน x และ y พร้อมสีสะท้อนค่าความสำคัญ</li>
    </ol>

    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
  <code>
    {`import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(attention_weights, xticklabels=input_tokens, yticklabels=output_tokens)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()`}
  </code>
</pre>


    <h3 className="text-xl font-semibold">แหล่งอ้างอิงด้าน Visualization</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>Google Research – Visualizing Attention in Neural Translation</li>
      <li>Stanford CS224n – Lecture 10: Attention Interpretability</li>
      <li>MIT Deep Learning 6.S191 – Attention Analysis Lab</li>
      <li>Oxford Deep NLP Course – Visualization Techniques</li>
    </ul>
  </div>
</section>


        <section id="comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. การเปรียบเทียบระหว่าง Encoder-Decoder แบบเดิม vs แบบ Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ภาพรวมของโมเดลทั้งสองแนวทาง</h3>
    <p>
      Encoder–Decoder architecture เป็นแกนกลางของหลายโมเดลใน Natural Language Processing (NLP) และ Sequence Learning อย่างไรก็ตาม ในเวอร์ชันดั้งเดิม โมเดลนี้พึ่งพา context vector เพียงเวกเตอร์เดียว ซึ่งกลายเป็นข้อจำกัดเมื่อจัดการกับลำดับข้อมูลที่ยาว ในขณะที่โมเดลที่รวม Attention เข้ามาสามารถหลีกเลี่ยง bottleneck นี้ได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบแบบชัดเจน</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2 text-left">คุณสมบัติ</th>
          <th className="border px-4 py-2 text-left">Basic Seq2Seq</th>
          <th className="border px-4 py-2 text-left">Seq2Seq + Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Context Vector</td>
          <td className="border px-4 py-2">Fixed-length</td>
          <td className="border px-4 py-2">Dynamic (per timestep)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Handling Long Sequences</td>
          <td className="border px-4 py-2">ประสิทธิภาพลดลง</td>
          <td className="border px-4 py-2">ประสิทธิภาพคงที่/ดีขึ้น</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Interpretability</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">สามารถ visualize attention</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Flexibility in Alignment</td>
          <td className="border px-4 py-2">ต่ำ</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Training Complexity</td>
          <td className="border px-4 py-2">น้อย</td>
          <td className="border px-4 py-2">สูงขึ้นเล็กน้อย</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">การแสดงผลเชิงภาพและการตีความ</h3>
    <p>
      ใน Encoder–Decoder แบบดั้งเดิม การอธิบายว่าเหตุใดโมเดลถึงเลือก output คำใดคำหนึ่งอาจทำได้ยาก แต่เมื่อใช้ Attention นักวิจัยสามารถดูได้ว่าโมเดล “โฟกัส” กับ input ตำแหน่งใดบ้างในการสร้างแต่ละ output ทำให้เข้าใจการตัดสินใจของโมเดลได้ดีขึ้น และใช้ตรวจจับความผิดพลาดของระบบได้
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: เหตุผลที่ Attention เหนือกว่า</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ความสามารถในการให้บริบทที่เปลี่ยนแปลงได้ ทำให้ระบบยืดหยุ่นมากขึ้น</li>
        <li>สามารถตีความโมเดลได้ผ่าน attention weights</li>
        <li>ช่วยให้ performance ดีขึ้นโดยเฉพาะในงาน NLP ที่มี sequence ยาว</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การประเมินจากงานวิจัย</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Bahdanau et al. แสดงว่า BLEU score เพิ่มขึ้นเมื่อเพิ่ม attention ใน NMT</li>
      <li>Luong et al. พบว่า dot-product attention ทำงานได้ดีในงาน translation ขนาดใหญ่</li>
      <li>Stanford CS224n ระบุว่า attention ลดปัญหา long-term dependency ได้ดีกว่า LSTM เพียงอย่างเดียว</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Bahdanau et al. (2014) – <em>Neural Machine Translation by Jointly Learning to Align and Translate</em></li>
      <li>Luong et al. (2015) – <em>Effective Approaches to Attention-based Neural Machine Translation</em></li>
      <li>Stanford CS224n – Lecture 10: Encoder–Decoder Models</li>
      <li>MIT 6.S191 – Deep Sequence Models: Context vs Attention</li>
      <li>Oxford Deep NLP – Comparative Analysis of Alignment Methods</li>
    </ul>
  </div>
</section>


         <section id="applications" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Real-World Applications</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">Neural Machine Translation (NMT)</h3>
    <p>
      Attention mechanisms กลายเป็นหัวใจหลักของระบบแปลภาษาอัตโนมัติสมัยใหม่ เช่น Google Translate และ DeepL ซึ่งจำเป็นต้องจับคู่คำหรือวลีในภาษาเป้าหมายกับบริบทที่ถูกต้องในภาษาเดิม กลไก Attention ช่วยให้สามารถเรียนรู้การจัดตำแหน่งแบบยืดหยุ่น (soft alignment) แทน hard-coded alignment แบบเดิม
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>ช่วยให้แปลคำได้ถูกต้องแม้คำจะอยู่ห่างไกลกันในประโยค</li>
      <li>ลดปัญหาการแปลผิดบริบท เช่น คำที่มีหลายความหมาย</li>
      <li>เพิ่ม BLEU score อย่างมีนัยสำคัญจาก baseline ที่ไม่มี attention</li>
    </ul>

    <h3 className="text-xl font-semibold">Text Summarization</h3>
    <p>
      งานสรุปข้อความ (summarization) เช่นการย่อข่าว หรือสรุปรายงานวิชาการ อาศัยการเลือก “ส่วนสำคัญ” จากข้อมูลต้นฉบับ Attention ทำหน้าที่เหมือนเครื่องกรองสาระ ช่วยเน้นสิ่งที่สำคัญและลด noise ที่ไม่เกี่ยวข้อง
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>สามารถใช้แบบ extractive หรือ abstractive summarization</li>
      <li>แสดงความสามารถผ่าน ROUGE score ที่สูงขึ้น</li>
      <li>ใช้ร่วมกับ LSTM, GRU หรือ Transformer-based Encoder</li>
    </ul>

    <h3 className="text-xl font-semibold">Image Captioning</h3>
    <p>
      ในการอธิบายภาพ (image captioning) โมเดลจะใช้ Attention เพื่อโฟกัสไปยัง region หรือ patch ที่เกี่ยวข้องกับคำในประโยค เช่น “A man riding a bike” → attention โฟกัสไปยังคนและจักรยานพร้อมกัน
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้ visual attention map แทน spatial filter ใน CNN</li>
      <li>ช่วยให้คำอธิบายมีความแม่นยำและสมบูรณ์ขึ้น</li>
      <li>ได้รับความนิยมใน dataset เช่น MSCOCO และ Flickr30k</li>
    </ul>

    <h3 className="text-xl font-semibold">Speech Recognition</h3>
    <p>
      ในระบบรู้จำเสียงพูด (ASR) เช่น Google Assistant, Siri หรือระบบ transcription ต่าง ๆ Attention ช่วยแยกแยะลำดับเวลาที่มีข้อมูลสำคัญ เช่น keyword detection หรือการแยกประโยค
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>ช่วยให้เรียนรู้ time frame ที่เกี่ยวข้องกับแต่ละคำ</li>
      <li>ทำให้ระบบ robust ต่อความยาวของเสียงที่ไม่คงที่</li>
      <li>ใช้ใน conjunction กับ CNN+RNN หรือ Transformer-based ASR</li>
    </ul>

    <h3 className="text-xl font-semibold">Biomedical Sequence Analysis</h3>
    <p>
      ในการวิเคราะห์ลำดับพันธุกรรม เช่น DNA หรือโปรตีน Attention ช่วยให้โมเดลจับตำแหน่งที่มีความหมายทางชีววิทยาได้แม่นยำ เช่น motifs หรือ binding sites
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้ร่วมกับ BiLSTM และ Transformer ในงาน genomic analysis</li>
      <li>ช่วยเพิ่ม interpretability ซึ่งสำคัญในงานวิจัยทางการแพทย์</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ทำไม Attention ถึงสำคัญในโลกจริง</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ช่วยให้โมเดลมี “การเลือกบริบท” เหมือนมนุษย์</li>
        <li>สามารถอธิบายได้ว่าทำไมระบบจึงเลือก output บางอย่าง → สร้างความเชื่อถือใน AI</li>
        <li>สามารถประยุกต์ใช้ได้กับข้อมูลหลากหลายประเภท: ข้อความ, ภาพ, เสียง, สารชีวภาพ</li>
        <li>เปิดทางสู่ explainable AI (XAI) ในระดับสูง</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงของการใช้งานจริง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Bahdanau et al. (2014) – <em>Neural Machine Translation with Attention</em></li>
      <li>Xu et al. (2015) – <em>Show, Attend and Tell: Neural Image Captioning with Visual Attention</em></li>
      <li>Chorowski et al. (2015) – <em>Attention-based Models for Speech Recognition</em></li>
      <li>Stanford CS224n – Lecture 11: Applications of Attention in NLP & Vision</li>
      <li>MIT 6.S191 – Applied Attention in Healthcare AI</li>
    </ul>
  </div>
</section>


      <section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ข้อจำกัดของ Attention แบบ Classic</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ขอบเขตการทำงานแบบ Pairwise</h3>
    <p>
      กลไก Attention แบบดั้งเดิม เช่น Additive และ Dot-Product Attention ทำงานในลักษณะ pairwise ระหว่าง decoder กับ encoder → สำหรับแต่ละ timestep ของ decoder จะต้องคำนวณ score กับทุก encoder state ซึ่งมีความซับซ้อนเป็น O(T) ต่อ output token
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>การประมวลผลเป็นเชิงเส้นตามความยาวลำดับ</li>
      <li>ต้องรอ encoder คำนวณเสร็จทั้งหมดก่อน decoder จะเริ่มได้</li>
      <li>ไม่สามารถใช้ parallelism ได้เต็มที่</li>
    </ul>

    <h3 className="text-xl font-semibold">ปัญหาในการสเกลกับลำดับที่ยาว</h3>
    <p>
      ในลำดับข้อมูลที่ยาวมาก เช่น เอกสารหลายหน้า หรือการสตรีมเสียงยาว ๆ กลไก Attention แบบ Classic จะมีข้อจำกัดด้านเวลาและหน่วยความจำ เนื่องจากต้องเก็บ hidden states ทั้งหมดของ encoder และคำนวณ dot-product กับทุกตำแหน่งในลำดับนั้น
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`เวลาในการประมวลผล ~ O(T_encoder × T_decoder)
Memory ~ O(T_encoder)`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ไม่สามารถใช้ Self-Attention</h3>
    <p>
      Classic Attention แบบ encoder-decoder ไม่สามารถเรียนรู้ dependency ภายในลำดับ input หรือ output ได้ด้วยตัวเอง จึงไม่สามารถสร้าง representation ที่อิงจากบริบทรอบข้างภายในลำดับเดียวกัน (เช่น การพิจารณาความสัมพันธ์ระหว่างคำในประโยคเดียวกัน)
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>จำเป็นต้องใช้ RNN หรือ CNN ประกอบเพื่อสร้าง feature ภายในลำดับ</li>
      <li>ไม่เหมาะกับ task ที่ไม่มีการ map ข้ามลำดับ เช่น classification, tagging</li>
      <li>ไม่สามารถใช้งานกับโมเดล encoder-only หรือ decoder-only ได้เต็มที่</li>
    </ul>

    <h3 className="text-xl font-semibold">ความยืดหยุ่นเชิงโครงสร้างต่ำ</h3>
    <p>
      เนื่องจาก Classic Attention ถูกออกแบบมาเพื่อช่วยเสริมในระบบ Seq2Seq ที่มีลำดับชัดเจน การนำไปใช้ในบริบทอื่นที่ซับซ้อน เช่น multimodal หรือ graph-structured data จะต้องมีการปรับแต่งโครงสร้างโดยตรง ทำให้ไม่สามารถ generalize ได้เท่ากับ self-attention
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ขีดจำกัดของโมเดล Classic</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ไม่สามารถประมวลผลแบบขนาน (parallelize) ได้ → ช้ากว่า Transformer มาก</li>
        <li>ขึ้นอยู่กับ hidden states จาก RNN → อาจมีปัญหา vanishing gradient</li>
        <li>ไม่สามารถเรียนรู้โครงสร้างภายใน sequence ได้เอง</li>
        <li>ไม่เหมาะกับข้อมูลที่มี dependency ซับซ้อน เช่น syntax trees หรือ graph</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับ Self-Attention</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">Feature</th>
          <th className="border px-4 py-2">Classic Attention</th>
          <th className="border px-4 py-2">Self-Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Parallelization</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✔️</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Dependency Modeling</td>
          <td className="border px-4 py-2">เฉพาะ encoder-decoder</td>
          <td className="border px-4 py-2">ทุกตำแหน่งในลำดับ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Scalability</td>
          <td className="border px-4 py-2">ต่ำ</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Flexibility</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Bahdanau et al. (2014) – <em>Neural Machine Translation by Jointly Learning to Align and Translate</em></li>
      <li>Luong et al. (2015) – <em>Attention-based NMT: Limitations and Extensions</em></li>
      <li>Vaswani et al. (2017) – <em>Attention Is All You Need</em></li>
      <li>Stanford CS224n – Lecture 11: Limitations of Classical Attention</li>
      <li>MIT 6.S191 – Deep Learning for NLP: Attention and Beyond</li>
    </ul>
  </div>
</section>


        <section id="code-example" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Code Example (PyTorch Pseudo)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">โครงสร้างหลักของ Attention Function</h3>
    <p>
      ด้านล่างนี้เป็นตัวอย่างโค้ดแบบ pseudo code ที่เขียนในลักษณะ PyTorch โดยแสดงกระบวนการทำงานของ Attention แบบ Classic โดยใช้ Dot-product และ Additive Attention ตามลำดับขั้นตอน:
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded p-4 overflow-x-auto">
<code>{`import torch
import torch.nn.functional as F

def attention_score(query, key, method="dot"):
    if method == "dot":
        return torch.sum(query * key, dim=-1)
    elif method == "additive":
        # สมมุติว่า W1, W2, v เป็นพารามิเตอร์ที่เรียนรู้ได้
        x = torch.tanh(torch.matmul(W1, query) + torch.matmul(W2, key))
        return torch.matmul(v, x)

def compute_attention(query, keys, values, method="dot"):
    # query: [batch, 1, d]
    # keys:  [batch, seq_len, d]
    # values: [batch, seq_len, d]
    scores = attention_score(query, keys, method=method)         # [batch, seq_len]
    weights = F.softmax(scores, dim=-1)                          # [batch, seq_len]
    context = torch.sum(weights.unsqueeze(-1) * values, dim=1)  # [batch, d]
    return context, weights
`}</code>
    </pre>

    <h3 className="text-xl font-semibold">การใช้ร่วมกับ Decoder</h3>
    <p>
      ตัวอย่างการนำ context vector ที่ได้จาก Attention ไปผสมกับ decoder hidden state:
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded p-4 overflow-x-auto">
<code>{`decoder_output = torch.cat([context, decoder_hidden], dim=-1)
output_logits = output_layer(decoder_output)`}</code>
    </pre>

    <h3 className="text-xl font-semibold">การใช้งานจริงใน Training Loop</h3>
    <p>
      ใน training loop ของ sequence-to-sequence model ที่ใช้ Attention จะต้องมีการคำนวณ context vector ในแต่ละ timestep:
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded p-4 overflow-x-auto">
<code>{`for t in range(target_seq_len):
    query = decoder_hidden_state[t]
    context, attn_weights = compute_attention(query, encoder_outputs, encoder_outputs)
    decoder_input = torch.cat([context, target_embedding[t]], dim=-1)
    decoder_hidden_state[t+1] = decoder_rnn(decoder_input, decoder_hidden_state[t])`}</code>
    </pre>

    <h3 className="text-xl font-semibold">การตรวจสอบ Attention Weights</h3>
    <p>
      หลังจากโมเดลถูกฝึกเสร็จแล้ว สามารถดึง attention weights ออกมาเพื่อทำ visualization ได้:
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded p-4 overflow-x-auto">
<code>{`# weights: [batch, seq_len]
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(weights.cpu().detach().numpy(), xticklabels=input_tokens, yticklabels=output_tokens)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()`}</code>
    </pre>

    <h3 className="text-xl font-semibold">Insight Box: สรุปองค์ประกอบในโค้ด</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>ฟังก์ชัน <code>attention_score</code> เป็นแกนหลักของการประเมิน similarity</li>
        <li><code>compute_attention</code> แยก logic สำหรับ context vector ออกจาก decoder</li>
        <li>สามารถสลับระหว่าง additive และ dot-product ได้ง่ายด้วย argument</li>
        <li>ใช้ softmax เสมอเพื่อ normalize เป็นความน่าจะเป็น</li>
        <li>โครงสร้างนี้สามารถต่อยอดไปสู่ Multi-Head Attention ได้ในภายหลัง</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงจากงานวิจัยและเอกสารทางเทคนิค</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Bahdanau et al. (2014) – <em>Neural Machine Translation with Attention</em></li>
      <li>Luong et al. (2015) – <em>Effective Approaches to Attention-based Neural MT</em></li>
      <li>PyTorch Tutorials – <em>seq2seq with attention</em>: https://pytorch.org/tutorials</li>
      <li>Stanford CS224n – Lecture 10: Attention Implementation</li>
      <li>MIT 6.S191 – Practical Sequence Modeling</li>
    </ul>

  </div>
</section>


        <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Academic References</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">งานวิจัยต้นฉบับและที่เกี่ยวข้อง</h3>
    <p>
      ต่อไปนี้คือรายชื่อบทความและเอกสารวิชาการที่เป็นรากฐานสำคัญของ Attention Mechanism แบบ Classic รวมถึงการประยุกต์ใช้ในงานต่าง ๆ ทั้งใน NLP, Computer Vision และ Speech Processing โดยอ้างอิงจากแหล่งข้อมูลระดับสากล เช่น arXiv, IEEE, และคอร์สระดับมหาวิทยาลัยชั้นนำ
    </p>

    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>
        Bahdanau, D., Cho, K., & Bengio, Y. (2014). 
        <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. 
        arXiv:1409.0473.
      </li>
      <li>
        Luong, M., Pham, H., & Manning, C. D. (2015). 
        <em>Effective Approaches to Attention-based Neural Machine Translation</em>. 
        arXiv:1508.04025.
      </li>
      <li>
        Vaswani, A., et al. (2017). 
        <em>Attention Is All You Need</em>. 
        NeurIPS. (พื้นฐานของ self-attention ที่ต่อยอดจาก classic attention)
      </li>
      <li>
        Xu, K. et al. (2015). 
        <em>Show, Attend and Tell: Neural Image Captioning with Visual Attention</em>. 
        ICML.
      </li>
      <li>
        Chorowski, J., Bahdanau, D., Serdyuk, D., Cho, K., & Bengio, Y. (2015). 
        <em>Attention-based Models for Speech Recognition</em>. 
        arXiv:1506.07503.
      </li>
    </ul>

    <h3 className="text-xl font-semibold">หลักสูตรจากมหาวิทยาลัยชั้นนำ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Stanford University – CS224n: <em>Natural Language Processing with Deep Learning</em></li>
      <li>MIT – 6.S191: <em>Introduction to Deep Learning</em></li>
      <li>Carnegie Mellon University – <em>Neural Networks for NLP</em> Module</li>
      <li>Oxford University – <em>Deep NLP Course: Sequence to Sequence Models</em></li>
      <li>Harvard NLP Group – <em>Annotated Transformer and Attention Research</em></li>
    </ul>

    <h3 className="text-xl font-semibold">แนวทางคัดเลือกแหล่งอ้างอิงคุณภาพ</h3>
    <p>
      เพื่อให้แน่ใจว่าเนื้อหาในบทเรียนมีความน่าเชื่อถือและอิงจากงานวิชาการที่ได้รับการยอมรับ สามารถพิจารณาแหล่งอ้างอิงตามเกณฑ์ดังนี้:
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: แนวทางการประเมินงานวิจัย</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>เลือกงานจากวารสาร peer-reviewed เช่น NeurIPS, ICML, ACL, IEEE</li>
        <li>ใช้ citation count และ h-index ของผู้เขียนใน Google Scholar</li>
        <li>อ้างอิงรายวิชาจากสถาบันชั้นนำ เช่น Stanford, MIT, CMU, Oxford</li>
        <li>มองหาโครงการที่มี implementation จริงใน Hugging Face หรือ GitHub</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การประยุกต์อ้างอิงในโปรเจกต์จริง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>ใช้บทความ Bahdanau และ Luong เพื่อเข้าใจกลไก attention เบื้องต้น</li>
      <li>อิง Vaswani et al. เพื่อขยายไปสู่ Transformer และ self-attention</li>
      <li>ใช้ Stanford CS224n เป็นโครงสร้างหลักในการออกแบบคอร์ส</li>
      <li>ใช้ annotated notebooks ของ Harvard NLP ในการอธิบายเชิงภาพ</li>
    </ul>

  </div>
</section>


        <section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Summary</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">สรุปภาพรวมของกลไก Attention แบบ Classic</h3>
    <p>
      Attention คือกลไกที่ช่วยให้โมเดลสามารถโฟกัสข้อมูลเฉพาะตำแหน่งที่สำคัญในลำดับ input โดยไม่ต้องพึ่งพา context vector แบบ fixed-length ซึ่งเคยเป็น bottleneck สำคัญในสถาปัตยกรรม Seq2Seq แบบดั้งเดิม กลไกนี้เปิดทางให้โมเดลสามารถจัดการข้อมูลที่ยาวและซับซ้อนขึ้นได้
    </p>

    <h3 className="text-xl font-semibold">ลำดับการประมวลผลของ Attention</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>คำนวณ similarity (score) ระหว่าง decoder state กับ encoder states</li>
      <li>นำ score ไปผ่าน softmax เพื่อให้ได้ attention weights</li>
      <li>ใช้ weights เหล่านี้เพื่อสร้าง context vector ที่สะท้อนข้อมูลสำคัญ</li>
      <li>รวม context กับ decoder เพื่อสร้าง output แต่ละ timestep</li>
    </ul>

    <h3 className="text-xl font-semibold">ประเภทของ Attention ที่ได้รับความนิยม</h3>
    <p>
      กลไกแบบ Classic ได้แก่ Additive (Bahdanau) และ Dot-Product (Luong) ซึ่งเป็นพื้นฐานที่ถูกนำไปใช้และขยายผลในโมเดลขั้นสูง เช่น Transformer:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Additive Attention:</strong> ใช้ feedforward layer เพื่อเรียนรู้ความสัมพันธ์แบบ non-linear</li>
      <li><strong>Dot-Product Attention:</strong> คำนวณเร็ว เหมาะกับงานขนาดใหญ่</li>
      <li><strong>Scaled Dot-Product:</strong> เพิ่มความเสถียรเมื่อลำดับยาวมาก (Transformer ใช้เป็นแกนหลัก)</li>
    </ul>

    <h3 className="text-xl font-semibold">การนำไปใช้งานจริง</h3>
    <p>
      Classic Attention ได้รับการนำไปใช้ในหลากหลายบริบท เช่น การแปลภาษา, การสรุปข้อความ, การรู้จำเสียงพูด และการวิเคราะห์ข้อมูลภาพทางการแพทย์ โดยเฉพาะอย่างยิ่งในระบบ Neural Machine Translation (NMT) ซึ่ง attention ช่วยปรับปรุง alignment และ contextualization ได้ดี
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: เหตุใดต้องเข้าใจ Attention อย่างลึกซึ้ง</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>การเข้าใจ attention เป็นรากฐานของโมเดลเชิงลำดับยุคใหม่ เช่น Transformer และ BERT</li>
        <li>การ visual attention ช่วยในการตรวจสอบการทำงานของโมเดล (interpretability)</li>
        <li>สามารถนำไปประยุกต์ข้าม modality เช่น Text → Image, Audio → Text</li>
        <li>แนวคิด attention เป็นแรงบันดาลใจสู่ self-attention และ multi-head attention</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ข้อจำกัดที่ควรระวัง</h3>
    <p>
      แม้ attention จะมีพลังมาก แต่ version แบบ classic ยังมีข้อจำกัด เช่น ไม่สามารถประมวลผลแบบขนานได้, ไม่รองรับ self-attention, และมี scaling ที่จำกัดเมื่อความยาวลำดับเพิ่มสูงขึ้น จึงเป็นเหตุผลที่ Transformer เข้ามาแทนที่ในหลายกรณี
    </p>

    <h3 className="text-xl font-semibold">สรุปบทเรียนสำคัญ</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>กลไก attention ช่วยให้โมเดลเลือก context ที่เกี่ยวข้องในแต่ละ timestep</li>
      <li>ลดปัญหา long-term dependency ที่เกิดใน RNN แบบดั้งเดิม</li>
      <li>เป็นเครื่องมือสำคัญที่เปิดทางสู่การปฏิวัติโครงข่ายแบบลำดับ</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงเพิ่มเติม</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Stanford CS224n – Lecture 10–11: Attention and Self-Attention</li>
      <li>MIT 6.S191 – <em>Deep Learning for Structured Data</em></li>
      <li>Bahdanau et al. (2014) – <em>Neural MT with Alignment</em></li>
      <li>Luong et al. (2015) – <em>Effective Attention-Based Translation</em></li>
      <li>Vaswani et al. (2017) – <em>Attention Is All You Need</em></li>
    </ul>

    <p>
      ความเข้าใจกลไก Attention ไม่เพียงเป็นก้าวสำคัญในการเรียนรู้ Deep Learning เชิงลำดับเท่านั้น แต่ยังเป็นจุดเริ่มต้นของสถาปัตยกรรมปัญญาประดิษฐ์ยุคใหม่ที่มีความสามารถในการตีความแบบ dynamic และ adaptive อย่างแท้จริง
    </p>

  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day30 theme={theme} />
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
        <ScrollSpy_Ai_Day30 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day30_AttentionClassic;
