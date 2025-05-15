import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day32 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day32";
import MiniQuiz_Day32 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day32";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day32_PositionalEncoding = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day32_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day32_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day32_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day32_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day32_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day32_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day32_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day32_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day32_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day32_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day32_11").format("auto").quality("auto").resize(scale().width(500));
  const img12 = cld.image("Day32_12").format("auto").quality("auto").resize(scale().width(500));
  const img13 = cld.image("Day32_13").format("auto").quality("auto").resize(scale().width(500));
  const img14 = cld.image("Day32_14").format("auto").quality("auto").resize(scale().width(490));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 32: Positional Encoding in Transformers</h1>
              <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img1} />
            </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

        <section id="why-positional" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. ทำไมต้องมี Positional Encoding?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ปัญหาเมื่อไม่มีลำดับ</h3>
    <p>
      โมเดล Transformer ไม่มีโครงสร้างลำดับเช่นเดียวกับ RNN หรือ CNN ซึ่งมีลำดับการประมวลผลที่ชัดเจน ทำให้โมเดลไม่สามารถรับรู้ถึงตำแหน่งของข้อมูลในลำดับได้โดยธรรมชาติ ข้อมูลอย่าง “แมววิ่งไล่หมา” กับ “หมาวิ่งไล่แมว” จะถูกตีความเหมือนกันหากไม่มีตำแหน่งบอกลำดับ
    </p>

    <h3 className="text-xl font-semibold">การตัด Recurrence ส่งผลอย่างไร</h3>
    <p>
      การออกแบบ Transformer โดยตัดกลไกของ Recurrence ออกทั้งหมด ช่วยให้สามารถประมวลผลแบบขนาน (parallel computation) ได้เต็มที่ แต่ส่งผลให้โมเดลไม่มีตัวช่วยในการเข้าใจว่า token ใดมาก่อนหรือหลัง จึงจำเป็นต้องฝังข้อมูลตำแหน่ง (positional information) ลงไปด้วยวิธีเฉพาะ
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: จุดเปลี่ยนจาก Sequential → Parallel</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>โมเดล RNN มีโครงสร้างแบบลำดับที่ชัดเจนในเวลา แต่ประมวลผลช้า</li>
        <li>Transformer เปลี่ยนทุกอย่างให้สามารถประมวลผลพร้อมกันได้</li>
        <li>การขาดลำดับต้องชดเชยด้วย Positional Encoding อย่างสมบูรณ์</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แนวทางที่เป็นไปได้</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้ฟังก์ชันเชิงคณิตศาสตร์ เช่น sine และ cosine (Vaswani et al., 2017)</li>
      <li>เรียนรู้ตำแหน่งแบบ end-to-end (Learned Positional Embeddings)</li>
      <li>เข้ารหัสตำแหน่งสัมพัทธ์ (Relative Position Embedding)</li>
      <li>ใช้การหมุนมุมเวกเตอร์ (Rotary Embeddings หรือ RoPE)</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างปัญหา</h3>
    <p>
      หากไม่มี positional encoding โมเดลจะไม่สามารถแยกความแตกต่างระหว่างประโยค:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`"The cat chased the dog" ≠ "The dog chased the cat"`}</code>
    </pre>
    <p>
      แม้คำเหมือนกันแต่ลำดับเปลี่ยน ความหมายก็เปลี่ยน → หากไม่มีลำดับ โมเดลจะไม่รู้ว่าคำใดเกิดก่อนหรือหลัง
    </p>

    <h3 className="text-xl font-semibold">ประโยชน์ของ Positional Encoding</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ช่วยให้โมเดลเข้าใจลำดับของ token ภายใน sequence</li>
      <li>เปิดทางสู่ความสามารถในการจับ dependency ระยะไกล</li>
      <li>รองรับการประมวลผลแบบขนานขณะยังรักษาบริบทเชิงลำดับไว้</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box: Positional Encoding เป็นแกนหลัก</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> Positional Encoding ไม่ใช่ส่วนเสริม แต่เป็นองค์ประกอบหลักที่ทำให้โมเดลแบบ Attention-only อย่าง Transformer สามารถประมวลผลข้อมูลลำดับได้โดยไม่มีโครงสร้างลำดับในตัว</p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>MIT 6.S191 – Deep Learning Lecture on Sequence Models</li>
      <li>Stanford CS224n – Lecture 11: Positional Representations</li>
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>Oxford NLP – Advanced Topics in Attention Mechanisms</li>
    </ul>
  </div>
</section>


   <section id="abs-vs-rel" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Absolute vs Relative Position Encoding</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">แนวคิดของการเข้ารหัสตำแหน่งแบบ Absolute และ Relative</h3>
    <p>
      Positional Encoding มีบทบาทสำคัญในการทำให้โมเดล Transformer ซึ่งไม่มีโครงสร้างลำดับโดยธรรมชาติ สามารถเข้าใจลำดับของข้อมูลได้ การเข้ารหัสตำแหน่งสามารถแบ่งได้เป็น 2 แนวทางหลัก ได้แก่ <strong>Absolute Position Encoding</strong> และ <strong>Relative Position Encoding</strong>
    </p>

    <h3 className="text-xl font-semibold">Absolute Position Encoding</h3>
    <p>
      ในแนวทางนี้ โมเดลจะได้รับเวกเตอร์ตำแหน่งที่มีค่าคงที่ หรือเรียนรู้ได้ ซึ่งบอกว่า token นั้นอยู่ในลำดับที่เท่าใด เช่นลำดับที่ 0, 1, 2, … โดยมักจะรวมกับ embedding vector ผ่านการบวก:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`input_embedding = token_embedding + absolute_pos_encoding`}</code>
    </pre>
    <ul className="list-disc ml-6 space-y-2">
      <li>เป็นแนวทางที่ใช้ใน Transformer ดั้งเดิม (Vaswani et al., 2017)</li>
      <li>ตำแหน่งเดียวกันจะมีค่ารหัสเหมือนกันทุกครั้งที่ปรากฏ</li>
      <li>ไม่สามารถแยกความแตกต่างของตำแหน่งสัมพันธ์ เช่น "ห่างกันกี่ตำแหน่ง"</li>
    </ul>

    <h3 className="text-xl font-semibold">Relative Position Encoding</h3>
    <p>
      แนวทางนี้เน้นการเข้ารหัสความสัมพันธ์ระหว่าง token เช่น "token นี้อยู่ห่างจากอีก token หนึ่งกี่ตำแหน่ง" แทนที่จะใช้ตำแหน่งแบบสัมบูรณ์ เทคนิคนี้ถูกเสนอครั้งแรกในงานของ <strong>Shaw et al., 2018</strong>
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`score(i, j) = Q_i · (K_j + a_{i-j})`}</code>
    </pre>
    <ul className="list-disc ml-6 space-y-2">
      <li>ค่ารหัสตำแหน่งจะขึ้นอยู่กับระยะห่าง (i − j) ไม่ใช่ค่าคงที่</li>
      <li>มีความยืดหยุ่นในการ generalize กับลำดับใหม่ที่ไม่เคยเห็น</li>
      <li>ช่วยให้โมเดลโฟกัสกับรูปแบบของความสัมพันธ์มากกว่าลำดับจริง</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ข้อแตกต่างสำคัญ</h3>
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Absolute Encoding</th>
            <th className="border px-4 py-2">Relative Encoding</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">บอกตำแหน่งจริง</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✖️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">บอกความสัมพันธ์ตำแหน่ง</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ใช้ใน BERT/GPT</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">บางรุ่น (e.g. Transformer-XL)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ข้อสังเกตเชิงลึก</h3>
    <p>
      การใช้ relative encoding อาจช่วยให้โมเดลเข้าใจโครงสร้างภาษาที่มี pattern ซ้ำซ้อนได้ดีกว่า เช่น dependency ของคำในระยะไกล ที่ไม่ได้ขึ้นกับตำแหน่งแน่นอนแต่ขึ้นกับระยะห่าง
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> การเปลี่ยนจากการเข้ารหัสตำแหน่งแบบสัมบูรณ์ → แบบสัมพันธ์ เป็นหนึ่งในก้าวสำคัญที่ช่วยให้ Transformer พัฒนาไปสู่สถาปัตยกรรมที่ generalize ได้ดีขึ้นในหลายบริบท</p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงวิชาการ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Shaw et al. (2018). <em>Self-Attention with Relative Position Representations</em>. NAACL.</li>
      <li>Stanford CS224n – Lecture: Transformer Positional Encoding</li>
      <li>MIT 6.S191 – 2024 Edition: Positional Encoding Module</li>
      <li>CMU Advanced NLP – Relational Encoding in Transformers</li>
    </ul>
  </div>
</section>


        <section id="sinusoidal" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    3. Sinusoidal Encoding (Vaswani et al., 2017)
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แรงจูงใจในการใช้ Encoding แบบไซน์/โคไซน์</h3>
    <p>
      Vaswani et al. (2017) เสนอ Sinusoidal Positional Encoding เพื่อแก้ปัญหาว่า self-attention ไม่มีความสามารถในการรับรู้ลำดับตำแหน่ง (position order) ด้วยตัวเอง โดยแนวคิดคือการแปลงตำแหน่งเป็นเวกเตอร์ของค่าฟังก์ชันไซน์และโคไซน์ เพื่อฝังบริบทตำแหน่งเข้าไปใน embedding space โดยไม่ต้องเรียนรู้ค่าพารามิเตอร์เพิ่มเติม
    </p>

    <h3 className="text-xl font-semibold">นิยามเชิงคณิตศาสตร์</h3>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`}</code>
    </pre>
    <p>
      โดยที่:
    </p>
    <ul className="list-disc ml-6 space-y-1">
      <li><code>pos</code>: ตำแหน่งของ token ใน sequence</li>
      <li><code>i</code>: ดัชนีใน embedding dimension</li>
      <li><code>d_model</code>: ขนาดของ embedding vector</li>
    </ul>

    <h3 className="text-xl font-semibold">คุณสมบัติเด่นของ Encoding แบบไซน์/โคไซน์</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ไม่ต้องเรียนรู้พารามิเตอร์ → ประหยัดหน่วยความจำและ generalize ได้ดี</li>
      <li>การแจกแจงของฟังก์ชัน sin/cos สร้างลักษณะ pattern ที่เป็นลำดับโดยธรรมชาติ</li>
      <li>สามารถอนุมานตำแหน่งนอกช่วงที่เคยเห็นระหว่างการฝึกได้</li>
      <li>รองรับการคำนวณ dot product ที่เป็นหัวใจของ self-attention ได้อย่างเหมาะสม</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ตัวอย่างการใช้งานจริง</h3>
      <ul className="list-disc list-inside space-y-1">
        <li>ใน BERT base → ใช้ sinusoidal encoding แบบคงที่</li>
        <li>GPT-2 และ GPT-3 ใช้ learnable encoding แทน</li>
        <li>ในงาน machine translation, sinusoidal encoding ช่วยให้โมเดลประมวลผลตำแหน่งได้แม่นยำโดยไม่ต้องฝึกใหม่เมื่อลำดับยาวขึ้น</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับ Encoding แบบเรียนรู้ได้</h3>
    <div className="w-full overflow-x-auto">
      <table className="min-w-[600px] border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Sinusoidal Encoding</th>
            <th className="border px-4 py-2">Learned Encoding</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Parameter count</td>
            <td className="border px-4 py-2">0</td>
            <td className="border px-4 py-2">ขึ้นกับ length</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Generalization (extrapolation)</td>
            <td className="border px-4 py-2">✔️ ดี</td>
            <td className="border px-4 py-2">✖️ จำกัดเฉพาะ training length</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความสามารถในการตีความ</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความยืดหยุ่น</td>
            <td className="border px-4 py-2">ต่ำ</td>
            <td className="border px-4 py-2">สูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> Sinusoidal Encoding เปรียบเหมือน “ภาษาทั่วไป” สำหรับการบอกตำแหน่งของ token — ง่ายต่อการตีความ, ไม่ต้องเรียนรู้ใหม่ และรองรับตำแหน่งที่ไม่เคยเห็นได้</p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture: Transformer Positional Encodings</li>
      <li>Harvard NLP – <em>The Annotated Transformer</em></li>
      <li>Oxford Deep NLP – Module: Representing Sequences</li>
      <li>CMU Advanced NLP – Encoding Strategies in Transformers</li>
    </ul>

  </div>
</section>


 <section id="learned" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Learned Positional Embeddings</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวคิดของการเรียนรู้ตำแหน่ง</h3>
    <p>
      แตกต่างจาก Sinusoidal Encoding ซึ่งค่าตำแหน่งถูกคำนวณแบบคงที่ด้วยฟังก์ชันทางคณิตศาสตร์, <strong>Learned Positional Embeddings</strong> เป็นแนวทางที่ทำให้ตำแหน่งเป็นพารามิเตอร์ที่เรียนรู้ได้ (trainable parameters) ซึ่งจะถูกปรับในระหว่างการฝึกโมเดลผ่านการ backpropagation โดยตรง
    </p>

    <h3 className="text-xl font-semibold">วิธีการทำงาน</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ตำแหน่งแต่ละตำแหน่งใน sequence จะมี vector ที่ถูกฝึกเฉพาะ (เช่น position 0, 1, 2,..., n)</li>
      <li>vector เหล่านี้จะถูกบวก (add) กับ token embedding ก่อนส่งเข้าสู่โมเดล</li>
      <li>ขนาดของ vector เท่ากับ <code>d_model</code> เช่นเดียวกับ embedding ของ token</li>
    </ul>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`x_input = token_embedding + learned_position_embedding`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ข้อดีของ Learned Positional Embedding</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>สามารถเรียนรู้การเข้ารหัสตำแหน่งที่เหมาะสมกับ task โดยเฉพาะ</li>
      <li>ช่วยให้โมเดลปรับตัวกับภาษา/โครงสร้างที่ซับซ้อนได้ดีขึ้น</li>
      <li>สามารถ capture ความสัมพันธ์ที่ไม่เป็นเชิงเส้นระหว่างตำแหน่ง</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อจำกัดของแนวทางนี้</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ไม่สามารถ generalize ไปยังตำแหน่งที่ไม่ได้เห็นระหว่าง training (out-of-distribution)</li>
      <li>จำนวนพารามิเตอร์เพิ่มตาม sequence length</li>
      <li>ไม่สามารถ encode ระยะห่างเชิงสัมพัทธ์ได้ชัดเจน</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: Learned vs Sinusoidal</h3>
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2 text-left">คุณสมบัติ</th>
            <th className="border px-4 py-2 text-left">Learned</th>
            <th className="border px-4 py-2 text-left">Sinusoidal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">การเรียนรู้</td>
            <td className="border px-4 py-2">✓</td>
            <td className="border px-4 py-2">✖️ (fixed)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Generalize ตำแหน่งใหม่</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✓</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ใช้ memory</td>
            <td className="border px-4 py-2">มากขึ้น</td>
            <td className="border px-4 py-2">ประหยัดกว่า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> Learned Positional Embedding อาจให้ performance สูงกว่าใน dataset ที่มีโครงสร้างตำแหน่งเฉพาะ แต่จะเสียความสามารถในการ generalize หาก sequence ยาวกว่าที่โมเดลเคยเห็น</p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Gehring et al. (2017). <em>Convolutional Sequence to Sequence Learning</em>. ICML.</li>
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture 12: Position Embeddings and Architectures</li>
      <li>Harvard NLP – Transformer Codebase and Discussion on Learnable Positions</li>
    </ul>
  </div>
</section>


    <section id="relative" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Relative Positional Encoding (Shaw et al., 2018)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แรงจูงใจเบื้องหลัง Relative Encoding</h3>
    <p>
      ในขณะที่ Absolute Positional Encoding ใช้ค่าตำแหน่งที่แน่นอน (เช่น ตำแหน่ง 1, 2, 3...) เพื่อสื่อถึงลำดับของ token ใน sequence นั้น <strong>Relative Positional Encoding</strong> เสนอแนวคิดใหม่ว่า <em>ความสัมพันธ์ระหว่างตำแหน่ง</em> มีความสำคัญมากกว่า absolute position ซึ่งมีประโยชน์โดยเฉพาะในโมเดลที่ต้องการ generalize ข้ามลำดับความยาวต่างกัน
    </p>

    <h3 className="text-xl font-semibold">กลไกของ Relative Encoding (Shaw et al., 2018)</h3>
    <p>
      งานของ Shaw et al. (2018) เสนอให้โมเดล Transformer เรียนรู้เวกเตอร์ตำแหน่งแบบสัมพัทธ์ โดยมีการปรับ QK Dot Product ใน Self-Attention ด้วย bias term ที่ขึ้นกับระยะห่างของ token (i − j) ดังนี้:
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`e_{ij} = (q_i)^T (k_j + a_{ij})`}</code>
    </pre>

        <p>
        โดยที่ <code>a<sub>ij</sub></code> คือ embedding ของระยะห่าง (relative distance) ระหว่างตำแหน่ง i และ j ซึ่งถูกเรียนรู้แยกจาก embedding ปกติ
        </p>


    <h3 className="text-xl font-semibold">ข้อดีของ Relative Encoding</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>โมเดลเข้าใจความสัมพันธ์เชิงโครงสร้างในลำดับโดยไม่ขึ้นกับความยาว</li>
      <li>เพิ่มประสิทธิภาพในการเรียนรู้ dependency ที่เกิดซ้ำ เช่น pattern ในภาษา</li>
      <li>เหมาะสำหรับ task ที่มีลำดับยาว เช่น document modeling หรือ translation</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบ: Absolute vs Relative</h3>
    <div className="overflow-x-auto">
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Absolute</th>
            <th className="border px-4 py-2">Relative</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">เรียนรู้ position โดยตรง</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✖️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ทั่วไปกับความยาวใหม่</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">เรียนรู้ความสัมพันธ์โดยตรง</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <p>
        Relative Positional Encoding เปลี่ยน paradigm จาก "ตำแหน่งคืออะไร" → สู่ "ตำแหน่งสัมพันธ์กันอย่างไร" ซึ่งสอดคล้องกับธรรมชาติของภาษา เช่น การเรียนรู้คำว่า "เขา" กับ "เธอ" ที่อยู่ในตำแหน่งต่างกันแต่มีความสัมพันธ์ในประโยคเดียวกัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างจาก NLP จริง</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Transformer-XL</strong> ใช้ relative encoding เพื่อเรียนรู้ข้าม segment</li>
      <li><strong>DeBERTa</strong> (Decoding-enhanced BERT) ใช้ relative key/value bias</li>
      <li>งานด้าน syntax-aware translation เช่น parsing trees ใช้แบบ relative มากขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Shaw et al. (2018). <em>Self-Attention with Relative Position Representations</em>. NAACL.</li>
      <li>Dai et al. (2019). <em>Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context</em>. ACL.</li>
      <li>He et al. (2021). <em>DeBERTa: Decoding-enhanced BERT with Disentangled Attention</em>. ICLR.</li>
      <li>Harvard NLP. <em>Relative Positional Encoding: Annotated Transformer Extensions</em>.</li>
    </ul>
  </div>
</section>


 <section id="rope" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Rotary Positional Embedding (RoPE, Su et al., 2021)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวคิดพื้นฐานของ RoPE</h3>
    <p>
      Rotary Positional Embedding (RoPE) ถูกเสนอโดย Su et al. (2021) เพื่อแก้ข้อจำกัดของการฝังตำแหน่งแบบ absolute โดย RoPE ฝังตำแหน่งเข้าไปใน vector space โดยตรงผ่านการหมุนเวกเตอร์ (rotation) ซึ่งทำให้สามารถ encode ความสัมพันธ์เชิงสัมพัทธ์ได้ในเชิงมุม (angular displacement)
    </p>

    <h3 className="text-xl font-semibold">วิธีการฝังตำแหน่งแบบหมุน</h3>
    <p>
      RoPE ใช้การหมุนเวกเตอร์ใน subspace ที่มีมิติคู่ (2D pairs) ของ embedding vector ตามตำแหน่ง ซึ่งสามารถแสดงด้วย matrix rotation หรือ complex multiplication:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`RoPE(pos, x) = x · R(pos)`}</code>
    </pre>
    <p>
      โดยที่ <code>R(pos)</code> คือ rotation matrix ที่ขึ้นกับตำแหน่ง <code>pos</code> ซึ่งเปลี่ยน phase ของเวกเตอร์โดยไม่เปลี่ยนขนาด (norm) → ทำให้รักษาโครงสร้างระยะทางได้ดีขึ้น
    </p>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบของ RoPE</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>สามารถ encode ความสัมพันธ์แบบ relative โดยใช้ embedding ที่เป็น absolute</li>
      <li>รองรับการ generalize ไปยังลำดับที่ยาวกว่าระหว่าง inference</li>
      <li>มีความคงตัวในการ rotate space และคำนวณ attention แบบ dot-product ได้โดยตรง</li>
      <li>ปรับใช้ได้ง่ายในโมเดล self-attention ที่มีอยู่เดิม เช่น GPT-NeoX, ChatGLM</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: Rotary ≠ Sinusoidal</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>RoPE ใช้การ encode แบบ rotation ของเวกเตอร์โดยตรง ต่างจาก Sinusoidal ที่ encode แยกชั้น</li>
        <li>ความสัมพันธ์เชิงมุม (angular encoding) ช่วยให้เรียนรู้ dependency แบบ relative ได้ดีขึ้น</li>
        <li>ไม่ต้องเรียนพารามิเตอร์ใหม่ → มีโครงสร้างที่ง่ายต่อการใช้งานกับ pretrained model</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การใช้งานในโมเดลสมัยใหม่</h3>
    <p>
      RoPE ถูกใช้ในโมเดลอย่าง GLM, ChatGLM, LLaMA และ GPT-J เพื่อเพิ่มความสามารถในการ generalize กับ sequence ที่ยาวและซับซ้อน โดยเฉพาะใน pretraining บนข้อมูล multilinguistic และ cross-modal
    </p>

    <h3 className="text-xl font-semibold">เปรียบเทียบ RoPE กับเทคนิคอื่น</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Sinusoidal</th>
          <th className="border px-4 py-2">Learned</th>
          <th className="border px-4 py-2">RoPE</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Encoding แบบ Relative</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✔️</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">จำเป็นต้องเรียนพารามิเตอร์</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✔️</td>
          <td className="border px-4 py-2">✖️</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การประมวลผลแบบ angular</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✔️</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: RoPE กับ Sequence ที่ยาว</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>RoPE ช่วยให้โมเดลสามารถประมวลผล sequence ที่ยาวกว่าเดิมได้ดี โดยไม่เสีย dependency</li>
        <li>เหมาะกับ task ที่มี long-range attention เช่น code generation, document QA</li>
        <li>ได้รับการพิสูจน์จากการใช้งานจริงในโมเดล state-of-the-art</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Su, J., et al. (2021). <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em>. arXiv:2104.09864</li>
      <li>Liu, Z., et al. (2023). <em>ChatGLM: Open Bilingual Chat Model</em>. Tsinghua NLP</li>
      <li>MIT 6.S191 – Deep Learning for Sequences: Positional Encoding</li>
      <li>Stanford CS224n – Lecture on Efficient Transformers and Positional Embedding</li>
    </ul>

  </div>
</section>


<section id="comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Comparison Table</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบรูปแบบ Positional Encoding</h3>
    <p>
      ตารางด้านล่างแสดงการเปรียบเทียบระหว่างวิธี Positional Encoding แบบต่าง ๆ โดยพิจารณาเกณฑ์สำคัญ เช่น ความสามารถในการ generalize, การเรียนรู้พารามิเตอร์, และการรองรับลำดับยาว
    </p>

    <div className="overflow-x-auto">
      <table className="min-w-[680px] border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">ลักษณะ</th>
            <th className="border px-4 py-2">Sinusoidal</th>
            <th className="border px-4 py-2">Learned</th>
            <th className="border px-4 py-2">Relative</th>
            <th className="border px-4 py-2">RoPE</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">ต้องเรียนรู้พารามิเตอร์</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️ (เฉพาะ bias term)</td>
            <td className="border px-4 py-2">✖️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Generalize ตำแหน่งใหม่</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">รองรับการ encode แบบ relative</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️ (เชิงมุมหมุน)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ใช้กับ Attention แบบเต็ม</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️ (เพิ่ม bias)</td>
            <td className="border px-4 py-2">✔️ (ด้วย dot-product rotation)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การใช้งานจริง</td>
            <td className="border px-4 py-2">Transformer ต้นฉบับ</td>
            <td className="border px-4 py-2">BERT</td>
            <td className="border px-4 py-2">T5, DeBERTa</td>
            <td className="border px-4 py-2">GPT-NeoX, LLaMA</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-blue-500">
      <p><strong>Highlight:</strong> แม้แต่ RoPE ซึ่งไม่ได้มีพารามิเตอร์ให้เรียนรู้ ก็สามารถให้ประสิทธิภาพเทียบเท่าหรือดีกว่าแบบที่เรียนรู้ได้ หากใช้อย่างเหมาะสมในสถาปัตยกรรมที่ออกแบบมารองรับ</p>
    </div>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Devlin et al. (2018). <em>BERT: Pre-training of Deep Bidirectional Transformers</em>. NAACL.</li>
      <li>Shaw et al. (2018). <em>Self-Attention with Relative Position Representations</em>. NAACL.</li>
      <li>Su et al. (2021). <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em>. arXiv.</li>
      <li>Stanford CS224n — Lecture: Positional Embeddings</li>
    </ul>
  </div>
</section>


    <section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Visualization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ความสำคัญของการ Visualization</h3>
    <p>
      ในงานวิจัยและการใช้งาน Transformer จริง การ Visualization กลไก Positional Encoding
      มีบทบาทสำคัญในการวิเคราะห์ความเข้าใจของโมเดลต่อโครงสร้างลำดับ โดยเฉพาะการแสดง
      attention weights หรือ embeddings space เพื่อศึกษาว่าโมเดลเข้าใจลำดับและบริบทใน sequence ได้อย่างไร
    </p>

    <h3 className="text-xl font-semibold">รูปแบบของ Visualization ที่นิยม</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Heatmaps แสดง attention weights ตามตำแหน่ง</li>
      <li>Embedding space projection โดยใช้ PCA หรือ t-SNE</li>
      <li>Graph-based visualization แสดงโครงสร้างการเชื่อมโยงภายใน sequence</li>
      <li>Interactive tools เช่น BertViz, TransformerLens</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ประโยชน์ของ Visualization</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ช่วยอธิบายพฤติกรรมของโมเดลเชิงลึกให้เข้าใจง่ายขึ้น</li>
        <li>ตรวจจับความผิดปกติในการเรียนรู้ของโมเดล เช่น attention collapse หรือ dead heads</li>
        <li>ใช้ในการวิเคราะห์ interpretability และ debugging ของระบบ NLP</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">เครื่องมือที่ใช้บ่อย</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>BertViz:</strong> สำหรับ BERT/GPT ใช้แสดง multi-head attention และ inter-layer interaction</li>
      <li><strong>TransformerLens:</strong> ใช้ interpret ลึกถึง mechanics และ activation path</li>
      <li><strong>TensorBoard Projector:</strong> สำหรับ visualize embedding space</li>
      <li><strong>AttentionViz:</strong> โมดูลใน Hugging Face ที่ช่วยวิเคราะห์ความสำคัญของ token</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ด: การ visualize attention map</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attn_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens, cmap='YlGnBu')
    plt.title("Attention Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()`}</code>
    </pre>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>Alammar, J. – <em>The Illustrated Transformer</em></li>
      <li>Vig, J. – <em>BertViz: Visualizing Attention in Transformer Models</em> (EMNLP 2019)</li>
      <li>Distill.pub – <em>Visualizing and Understanding Self-Attention</em></li>
      <li>MIT 6.S191 – Deep Learning: Interpretable Models</li>
    </ul>

  </div>
</section>


 <section id="research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Academic Research</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">งานวิจัยต้นฉบับและทฤษฎีพื้นฐาน</h3>
    <p>
      การพัฒนา Positional Encoding มีจุดเริ่มต้นสำคัญจากงานของ Vaswani et al. (2017) ซึ่งเสนอ Sinusoidal Encoding สำหรับ Transformer โดยมีเป้าหมายเพื่อให้โมเดลเข้าใจลำดับของข้อมูลโดยไม่ต้องใช้ recurrence. ต่อมาเกิดแนวทางที่แตกต่างออกไป เช่น Learnable Embedding, Relative Position Encoding (Shaw et al., 2018) และ Rotary Positional Embedding (Su et al., 2021) เพื่อรองรับโครงสร้างข้อมูลที่ซับซ้อนมากขึ้น
    </p>

    <h3 className="text-xl font-semibold">การพัฒนาเพิ่มเติมหลังปี 2018</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Shaw et al. (2018):</strong> เสนอ Relative Position Encoding สำหรับจับระยะห่างเชิงสัมพันธ์</li>
      <li><strong>Su et al. (2021):</strong> นำเสนอ RoPE ซึ่งฝังตำแหน่งผ่านการหมุนเชิงเรขาคณิต</li>
      <li><strong>Ke et al. (2020):</strong> แสดงผลลัพธ์ที่ดีกว่าในการ generalize ด้วย Relative Bias</li>
      <li><strong>Press et al. (2021):</strong> เปรียบเทียบวิธีการฝังตำแหน่งต่าง ๆ กับประสิทธิภาพ downstream tasks</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500 text-black dark:text-blue-100">
      <p>
        <strong>Highlight Box:</strong> ข้อมูลจาก MIT และ Stanford ชี้ให้เห็นว่า Positional Encoding ไม่ใช่แค่ส่วนเสริม แต่เป็นองค์ประกอบหลักที่มีอิทธิพลต่อความสามารถในการเรียนรู้ long-range dependency ใน Transformer architecture
      </p>
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มการวิจัยในอนาคต</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>การฝังตำแหน่งแบบ adaptive ที่เปลี่ยนตาม context</li>
      <li>การรวม positional encoding เข้ากับ memory-efficient architectures เช่น Linformer และ Performer</li>
      <li>การประยุกต์ใน Multimodal Learning ที่มีข้อมูลหลากหลายช่องทาง เช่น vision, audio, language</li>
    </ul>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>Shaw et al. (2018). <em>Self-Attention with Relative Position Representations</em>. NAACL.</li>
      <li>Su et al. (2021). <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em>. arXiv:2104.09864.</li>
      <li>Press et al. (2021). <em>Train Short, Test Long: Attention with Linear Biases</em>. arXiv:2108.12409.</li>
      <li>Ke et al. (2020). <em>Rethinking Positional Encoding in Language Pretraining</em>. ICLR.</li>
      <li>Stanford CS224n – Lecture on Transformer and Positional Encoding</li>
      <li>MIT 6.S191 – Introduction to Deep Learning (2024 Edition)</li>
    </ul>
  </div>
</section>


    <section id="practical" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Practical Considerations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">การเลือกใช้ Positional Encoding ให้เหมาะสม</h3>
    <p>
      ในการใช้งานจริง นักพัฒนาโมเดลจำเป็นต้องเลือกวิธีการฝังตำแหน่ง (Positional Encoding) ที่เหมาะกับงาน ทั้งนี้ขึ้นอยู่กับขนาดของข้อมูล, ลักษณะของลำดับข้อมูล และความสามารถในการ generalize ของโมเดลต่อข้อมูลนอก distribution ที่เห็นในการเทรน
    </p>

    <h3 className="text-xl font-semibold">สรุปแนวทางการเลือกใช้งาน</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>งาน NLP ทั่วไป เช่น BERT, GPT: ใช้ learned positional embeddings หรือ sinusoidal</li>
      <li>งานที่มีลำดับยาวมาก เช่น genome, long text: ใช้ relative หรือ rotary positional encoding</li>
      <li>งาน cross-modal หรือ multimodal (เช่น vision-language): ใช้ rotary หรือ encoding แบบ multi-scale</li>
      <li>โมเดลขนาดใหญ่ที่ต้องการ generalization สูง: นิยมใช้ sinusoidal หรือ rotary ที่ไม่ต้องเรียนรู้เพิ่มเติม</li>
    </ul>

    <h3 className="text-xl font-semibold">ผลกระทบต่อ Training และ Performance</h3>
    <p>
      งานวิจัยจาก Google Research และ Meta AI พบว่าการเลือก Positional Encoding ที่ไม่เหมาะสมอาจส่งผลให้ performance ลดลงอย่างมีนัยสำคัญ เช่น:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>Learned embeddings อาจ overfit บนลำดับที่จำกัด → ไม่ generalize</li>
      <li>Sinusoidal มีข้อดีด้าน generalization แต่ไม่ยืดหยุ่นพอใน multimodal</li>
      <li>Relative encoding ช่วยให้โมเดลไม่สนใจแค่ตำแหน่ง absolute แต่ดู “ความสัมพันธ์ของตำแหน่ง”</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างการตั้งค่าค่าพารามิเตอร์</h3>
    <div className="overflow-x-auto">
  <table className="min-w-[600px] border border-gray-300 dark:border-gray-700 text-sm">
    <thead className="bg-gray-100 dark:bg-gray-800">
      <tr>
        <th className="border px-4 py-2">Encoding Type</th>
        <th className="border px-4 py-2">Memory Efficient</th>
        <th className="border px-4 py-2">Generalization</th>
        <th className="border px-4 py-2">Suitable for Long Seq</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Learned</td>
        <td className="border px-4 py-2">✖️</td>
        <td className="border px-4 py-2">ต่ำ</td>
        <td className="border px-4 py-2">✖️</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Sinusoidal</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">สูง</td>
        <td className="border px-4 py-2">✔️</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Relative</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">สูง</td>
        <td className="border px-4 py-2">✔️</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Rotary (RoPE)</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">สูง</td>
        <td className="border px-4 py-2">✔️</td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ไม่ใช่ทุกโมเดลต้องเรียนรู้ตำแหน่ง</h3>
      <ul className="list-disc list-inside text-sm space-y-2">
        <li>งานที่ต้อง generalize ไปยังลำดับใหม่ ควรหลีกเลี่ยง positional ที่เรียนรู้ได้</li>
        <li>ในงาน multi-modal เช่น Vision-Language การใช้ Rotary หรือ Relative ช่วยผสาน modality ได้ดีขึ้น</li>
        <li>สำหรับโมเดลที่ต้องรันบน edge devices ให้เลือกแบบที่ประหยัดพารามิเตอร์ เช่น sinusoidal หรือ RoPE</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">อ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al., 2017 – <em>Attention Is All You Need</em></li>
      <li>Shaw et al., 2018 – <em>Self-Attention with Relative Position Representations</em></li>
      <li>Su et al., 2021 – <em>RoFormer: Rotary Position Embedding for Transformer</em></li>
      <li>Stanford CS224n – Lecture 11: Positional Embedding Practical Usage</li>
      <li>MIT 6.S191 – Deep Learning Best Practices (2024 Edition)</li>
    </ul>
  </div>
</section>




<section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Limitations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ข้อจำกัดของ Positional Encoding</h3>
    <p>
      แม้ Positional Encoding จะเป็นกลไกหลักในการเสริมความสามารถให้ Transformer เข้าใจลำดับของข้อมูล แต่การออกแบบแบบดั้งเดิม เช่น Sinusoidal Encoding และ Learned Embeddings ต่างมีข้อจำกัดในแง่ของการ generalize, efficiency และ alignment กับโครงสร้างภายในของ task บางประเภท
    </p>

    <h3 className="text-xl font-semibold">1. ไม่สามารถรองรับ Sequence ที่ยาวกว่าระหว่างการ Train ได้ดี</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Sinusoidal PE สามารถ generalize บางระดับแต่ไม่มีหลักประกันว่าความถี่ที่โมเดลเรียนรู้จะสอดคล้องกับตำแหน่งใหม่ใน sequence ที่ยาวขึ้น</li>
      <li>Learned Positional Embeddings จะล้มเหลวทันทีหากความยาว sequence ใหม่เกินกว่าที่ฝึกไว้</li>
    </ul>

    <h3 className="text-xl font-semibold">2. ไม่สามารถแยกแยะข้อมูลแบบ Relative ได้ดี</h3>
    <p>
      ในหลาย task เช่น Dependency Parsing, Dialogue หรือ Music Modeling ข้อมูลมักไม่อิงตำแหน่ง absolute ตรง ๆ แต่ต้องเข้าใจ relative position เช่น “คำที่อยู่ถัดไป 2 ตำแหน่ง” → PE แบบเดิมไม่สามารถ encode ความสัมพันธ์เช่นนี้โดยตรงได้
    </p>

    <h3 className="text-xl font-semibold">3. ค่า Encoding คงที่เกินไปสำหรับงานบางประเภท</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Sinusoidal Encoding เป็น deterministic function → ไม่สามารถเรียนรู้ pattern เฉพาะในข้อมูลได้</li>
      <li>แม้ว่า Learned PE จะยืดหยุ่นกว่า แต่ก็ไม่มี inductive bias ด้านลำดับ ซึ่งอาจทำให้ overfit ได้ง่าย</li>
    </ul>

    <h3 className="text-xl font-semibold">4. ค่า Computational Cost เพิ่มขึ้นเมื่อรวมกับกลไกซับซ้อนอื่น</h3>
    <p>
      การใช้ PE ต้องรวมกับ Attention Matrix เสมอ → ในบางงาน เช่น Vision หรือ Long Document การเพิ่ม complexity จาก PE ทำให้ระบบใช้ทรัพยากรเพิ่มขึ้นอย่างชัดเจน โดยเฉพาะเมื่อใช้ RoPE หรือ Relative PE
    </p>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบข้อจำกัดของ Positional Encoding แต่ละแบบ</h3>
    <div className="overflow-x-auto">
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">ประเภท</th>
            <th className="border px-4 py-2">Generalize กับลำดับยาว</th>
            <th className="border px-4 py-2">เข้าใจระยะห่าง (Relative)</th>
            <th className="border px-4 py-2">เรียนรู้ได้เอง</th>
            <th className="border px-4 py-2">ใช้ทรัพยากรสูง</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Sinusoidal</td>
            <td className="border px-4 py-2">ปานกลาง</td>
            <td className="border px-4 py-2">ไม่ดี</td>
            <td className="border px-4 py-2">✖</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Learned</td>
            <td className="border px-4 py-2">ไม่ดี</td>
            <td className="border px-4 py-2">ไม่ดี</td>
            <td className="border px-4 py-2">✔</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Relative</td>
            <td className="border px-4 py-2">ดี</td>
            <td className="border px-4 py-2">ดีมาก</td>
            <td className="border px-4 py-2">✔</td>
            <td className="border px-4 py-2">กลาง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">RoPE</td>
            <td className="border px-4 py-2">ดีมาก</td>
            <td className="border px-4 py-2">ดี</td>
            <td className="border px-4 py-2">✔</td>
            <td className="border px-4 py-2">สูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ทำไมการเข้าใจข้อจำกัดของ PE ถึงสำคัญ?</h3>
      <ul className="list-disc list-inside text-sm space-y-1">
        <li>ช่วยให้สามารถเลือก encoding ให้เหมาะกับงาน เช่น vision vs language</li>
        <li>ช่วยหลีกเลี่ยงจุดอ่อนของโมเดล เช่น การ overfit ตำแหน่งแบบ absolute</li>
        <li>ช่วยผลักดันงานวิจัยใหม่ในด้าน Efficient Attention และ Hybrid PE</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Shaw et al. (2018). <em>Self-Attention with Relative Position Representations</em>. NAACL.</li>
      <li>Su et al. (2021). <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em>. arXiv.</li>
      <li>Press et al. (2021). <em>Train Short, Test Long: Attention with Linear Biases</em>. arXiv.</li>
      <li>MIT 6.S191 (2024 Edition) – Transformer Limitations and Inductive Bias</li>
    </ul>
  </div>
</section>

<section id="insight-2" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">13. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">สรุปมุมมองเชิงกลยุทธ์ของ Positional Encoding</h3>
    <p>
      Positional Encoding ไม่ใช่เพียงกลไกเสริมในสถาปัตยกรรม Transformer แต่เป็นส่วนสำคัญที่หล่อหลอมให้การเรียนรู้ของโมเดลสามารถเข้าใจลำดับได้ในบริบทที่ไม่มี recurrence กลไกนี้จึงกลายเป็นสะพานเชื่อมจากโลกของ RNN สู่โลกของ Attention-based Learning ซึ่งสามารถขนานได้เต็มรูปแบบ
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-6 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p className="font-semibold">
        <strong>Insight:</strong> Positional Encoding เปรียบเสมือน “ระบบพิกัดทางปัญญา” ที่บอกตำแหน่งของข้อมูลในจักรวาลของ attention — หากไม่มีมัน ทุก token จะกลายเป็นอิสระจากกาลเวลา
      </p>
    </div>

    <h3 className="text-xl font-semibold">สิ่งที่เรียนรู้จากวิวัฒนาการของ Position Encoding</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>จาก Fixed → Learned → Relative → Rotary → Sparse — การพัฒนา positional encoding เป็นสัญญาณถึงความซับซ้อนที่เพิ่มขึ้นของโมเดล</li>
      <li>โมเดลใหม่ ๆ ไม่เพียงต้อง encode ตำแหน่งแบบแน่นอน แต่ต้อง encode ความสัมพันธ์ตำแหน่งแบบพลวัต</li>
      <li>จาก NLP → Vision → Graph → Audio — แนวคิด positional encoding กำลังขยายสู่ทุก modality</li>
    </ul>

    <h3 className="text-xl font-semibold">ผลกระทบของ Positional Encoding ต่อโมเดล</h3>
    <p>การเลือกประเภท positional encoding มีผลอย่างยิ่งต่อความสามารถของโมเดลในหลายแง่มุม:</p>
    <div className="overflow-x-auto">
  <table className="min-w-[640px] border border-gray-300 dark:border-gray-700 text-sm text-left">
    <thead className="bg-gray-100 dark:bg-gray-800">
      <tr>
        <th className="border px-4 py-2">Aspect</th>
        <th className="border px-4 py-2">Fixed</th>
        <th className="border px-4 py-2">Learned</th>
        <th className="border px-4 py-2">Relative</th>
        <th className="border px-4 py-2">Rotary</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Generalization</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">✖️</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">✔️</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Parameter Efficiency</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">✖️</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">✔️</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Suitability for Long Sequence</td>
        <td className="border px-4 py-2">✖️</td>
        <td className="border px-4 py-2">✖️</td>
        <td className="border px-4 py-2">✔️</td>
        <td className="border px-4 py-2">✔️</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">แนวโน้มของการพัฒนา</h3>
    <p>
      งานวิจัยหลังปี 2020 เริ่มเน้นไปที่การลด cost ของ positional encoding บนลำดับยาว เช่นใน Longformer, Performer และ GPT-NeoX ซึ่งต้องการ positional design ที่รองรับ context ที่ยาวกว่า 10,000 tokens โดยไม่ใช้หน่วยความจำมากเกินไป
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: Beyond NLP</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>Vision Transformer (ViT) ใช้ learnable 2D positional embedding เพื่อ encode ตำแหน่งในภาพ</li>
        <li>Graph Transformer ใช้ Laplacian eigenvectors แทน sin/cos encoding</li>
        <li>Audio modeling เริ่มประยุกต์ใช้ rotary และ relative positional ในการจับ time-delay ใน waveforms</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แนวทางในอนาคต</h3>
    <p>
      งานวิจัยสมัยใหม่อาจมุ่งไปที่การออกแบบระบบที่เข้าใจ “โครงสร้าง” มากกว่าตำแหน่งแบบดิบ เช่น การ encode hierarchical structure, syntax tree หรือ knowledge graph ด้วยความสัมพันธ์แบบ position-aware
    </p>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al., 2017 — <em>Attention Is All You Need</em></li>
      <li>Shaw et al., 2018 — <em>Self-Attention with Relative Position Representations</em></li>
      <li>Su et al., 2021 — <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em></li>
      <li>MIT 6.S191 — Lecture: Positional Encoding Beyond NLP</li>
      <li>Stanford CS25 — Foundation Models: Scaling Positional Systems</li>
      <li>Hugging Face Transformers Docs — <em>Position Embeddings Overview</em></li>
    </ul>
  </div>
</section>


<section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">14. Summary</h2>
    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">บทสรุปภาพรวมของ Positional Encoding</h3>
    <p>
      Positional Encoding เป็นองค์ประกอบที่ขาดไม่ได้ของสถาปัตยกรรม Transformer โดยทำหน้าที่เป็นตัวแทนลำดับของ token เพื่อช่วยให้โมเดลสามารถเข้าใจลำดับและความสัมพันธ์เชิงตำแหน่งของข้อมูลภายใน sequence แม้ในกรณีที่ไม่มีโครงสร้าง recurrent
    </p>

    <h3 className="text-xl font-semibold">ภาพรวมของแต่ละเทคนิค</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Sinusoidal Encoding (Vaswani et al., 2017):</strong> ไม่มีพารามิเตอร์เพิ่ม มีความสามารถในการ generalize สูง ใช้ฟังก์ชัน sine/cos ในการ encode ตำแหน่ง</li>
      <li><strong>Learned Positional Embeddings:</strong> มีพารามิเตอร์ที่เรียนรู้ได้โดยตรงจากข้อมูล ใช้งานง่ายแต่ generalize ได้น้อยกว่า</li>
      <li><strong>Relative Position (Shaw et al., 2018):</strong> เน้น encode ความสัมพันธ์ระหว่างตำแหน่ง ช่วยให้โมเดลยืดหยุ่นมากขึ้นในงาน sequence ที่เปลี่ยนขนาด</li>
      <li><strong>RoPE (Su et al., 2021):</strong> สร้าง embedding โดยใช้ rotation ใน space ช่วยให้ attention เรียนรู้ dependency แบบสัมพัทธ์และมีคุณสมบัติ invariance บางรูปแบบ</li>
    </ul>

    <h3 className="text-xl font-semibold">เมื่อใดควรเลือกแต่ละชนิด</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm text-left">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">กรณีใช้งาน</th>
          <th className="border px-4 py-2">Encoding ที่แนะนำ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Pretraining ขนาดใหญ่</td>
          <td className="border px-4 py-2">Sinusoidal / Rotary</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Task แบบ Static (e.g. BERT)</td>
          <td className="border px-4 py-2">Learned</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Long sequence / streaming</td>
          <td className="border px-4 py-2">Relative / RoPE</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Vision / Audio Transformers</td>
          <td className="border px-4 py-2">Learned (2D) / Relative</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ประเด็นเชิงกลยุทธ์จากมหาวิทยาลัยระดับโลก</h3>
    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl shadow-sm">
      <ul className="list-disc list-inside text-sm space-y-2">
        <li>Stanford CS224n แนะนำให้เริ่มจาก Sinusoidal หากต้องการ generalization ที่ดีและ memory efficiency</li>
        <li>MIT 6.S191 เสนอว่า RoPE เป็นทางเลือกที่ทันสมัยกว่าในการจับความสัมพันธ์ระยะยาวแบบ compact</li>
        <li>CMU NLP Lab ชี้ว่า Learned Positional Encoding ยืดหยุ่นดีเมื่อ train กับ dataset เดิม ๆ แต่ไม่เหมาะสำหรับ transfer learning</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มของ Positional Encoding ในยุค Multimodal</h3>
    <p>
      Positional Encoding กำลังกลายเป็นแนวคิดแกนกลางในโลกของ Multimodal Transformers โดยมีการขยายจากข้อมูลแบบลำดับ ไปสู่ spatial, hierarchical และ graph-based position representations ที่ออกแบบมาเฉพาะสำหรับการเรียนรู้บริบทที่ซับซ้อนมากขึ้น
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p className="font-semibold">
        <strong>สรุป Insight:</strong> Positional Encoding คือกุญแจสำคัญที่เปลี่ยนโมเดล attention ให้เข้าใจ "บริบทในลำดับ" ได้อย่างลึกซึ้ง และเป็นรากฐานที่ทำให้ Transformer ใช้ได้ในงานจริงตั้งแต่ NLP ไปจนถึง Vision และ Audio
      </p>
    </div>

  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day32 theme={theme} />
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
        <ScrollSpy_Ai_Day32 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day32_PositionalEncoding;
