import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day31 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day31";
import MiniQuiz_Day31 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day31";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day31_TransformerOverview = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day31_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day31_2").format("auto").quality("auto").resize(scale().width(491));
  const img3 = cld.image("Day31_3").format("auto").quality("auto").resize(scale().width(491));
  const img4 = cld.image("Day31_4").format("auto").quality("auto").resize(scale().width(491));
  const img5 = cld.image("Day31_5").format("auto").quality("auto").resize(scale().width(491));
  const img6 = cld.image("Day31_6").format("auto").quality("auto").resize(scale().width(491));
  const img7 = cld.image("Day31_7").format("auto").quality("auto").resize(scale().width(491));
  const img8 = cld.image("Day31_8").format("auto").quality("auto").resize(scale().width(491));
  const img9 = cld.image("Day31_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day31_10").format("auto").quality("auto").resize(scale().width(401));
  const img11 = cld.image("Day31_11").format("auto").quality("auto").resize(scale().width(498));
  const img12 = cld.image("Day31_12").format("auto").quality("auto").resize(scale().width(491));
  const img13 = cld.image("Day31_13").format("auto").quality("auto").resize(scale().width(501));
  const img14 = cld.image("Day31_14").format("auto").quality("auto").resize(scale().width(400));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 31: Transformer Architecture Overview</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

<section id="motivation" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. Motivation: ทำไม Transformer จึงกลายเป็น Game Changer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ข้อจำกัดของ RNN และ LSTM</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ไม่สามารถทำงานแบบขนานได้ (Non-parallelizable)</li>
      <li>เกิดปัญหา vanishing gradients เมื่อ sequence ยาว</li>
      <li>เรียนรู้ long-range dependency ได้ยาก</li>
    </ul>

    <h3 className="text-xl font-semibold">หลักการออกแบบของ Transformer</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ไม่ใช้ recurrence เลย</li>
      <li>ใช้ Self-Attention เพื่อจับ dependency ทั้งลำดับ</li>
      <li>สามารถฝึกแบบขนาน (parallel) ได้เต็มที่บน GPU</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ความต่าง RNN vs Transformer</h3>
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Property</th>
            <th className="border px-4 py-2">RNN / LSTM</th>
            <th className="border px-4 py-2">Transformer</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Parallelization</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Gradient Stability</td>
            <td className="border px-4 py-2">แปรผันตามลำดับ</td>
            <td className="border px-4 py-2">เสถียรกว่า</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Training Time</td>
            <td className="border px-4 py-2">ช้า</td>
            <td className="border px-4 py-2">เร็วกว่า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> Transformer เปลี่ยนแนวคิดจากการประมวลผลลำดับแบบ step-by-step → ไปสู่การเรียนรู้ความสัมพันธ์ทั้งหมดใน sequence พร้อมกันแบบ all-to-all dependency</p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al., 2017 — <em>Attention Is All You Need</em></li>
      <li>Stanford CS224n — Lecture 11: Transformer Architecture</li>
      <li>MIT 6.S191 — Attention Mechanisms</li>
      <li>Harvard NLP Annotated Transformer</li>
    </ul>
  </div>
</section>


<section id="architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Overall Architecture: Encoder-Decoder แบบ Full Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">โครงสร้างหลักของ Transformer</h3>
    <p>
      สถาปัตยกรรมของ Transformer ถูกออกแบบมาเพื่อแก้ปัญหาคอขวดของ RNN และ LSTM โดยใช้แนวคิดของ Attention เต็มรูปแบบในทั้ง Encoder และ Decoder ซึ่งช่วยให้สามารถเรียนรู้ dependency ในลำดับข้อมูลได้อย่างมีประสิทธิภาพและขนานกันได้ทั้งหมด
    </p>

    <h3 className="text-xl font-semibold">ส่วนประกอบของ Encoder และ Decoder</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">Component</th>
          <th className="border px-4 py-2">Encoder</th>
          <th className="border px-4 py-2">Decoder</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Input Embedding</td>
          <td className="border px-4 py-2">Token → Vector</td>
          <td className="border px-4 py-2">Token → Vector</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Positional Encoding</td>
          <td className="border px-4 py-2">เพิ่มตำแหน่ง (sin/cos encoding)</td>
          <td className="border px-4 py-2">เช่นเดียวกัน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Core Layer</td>
          <td className="border px-4 py-2">Multi-Head Self-Attention + FFN × N</td>
          <td className="border px-4 py-2">Masked Multi-Head Self-Attention + FFN × N</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Output</td>
          <td className="border px-4 py-2">Context Representation</td>
          <td className="border px-4 py-2">Token Prediction (step-by-step)</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: แก่นของ Transformer</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Encoder ทำหน้าที่เข้ารหัสความหมายจากลำดับ input โดยใช้ self-attention</li>
        <li>Decoder ใช้ทั้ง self-attention และ encoder-decoder attention เพื่อสร้างคำออกมาทีละคำ</li>
        <li>Residual connections และ Layer Normalization ถูกใช้ในทุก block เพื่อความเสถียรในการเรียนรู้</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การซ้อน Layer และ Residual Pathways</h3>
    <p>
      ทั้ง Encoder และ Decoder ประกอบด้วยหลายชั้น (มักใช้ N=6 ตาม paper ดั้งเดิม) โดยมีการใช้ residual connections และ layer normalization รอบ ๆ ทุก sublayer ซึ่งช่วยให้สามารถฝึกโมเดลลึก ๆ ได้ง่ายขึ้น
    </p>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017) — <em>Attention Is All You Need</em></li>
      <li>Alammar, J. (2018) — <em>The Illustrated Transformer</em></li>
      <li>Stanford CS224n — Lecture: Transformer Architecture</li>
      <li>MIT 6.S191 — Sequence Models in Deep Learning</li>
    </ul>
  </div>
</section>


<section id="positional-encoding" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Positional Encoding: เข้าใจตำแหน่ง โดยไม่ใช้ลำดับ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">พื้นฐานของ Positional Encoding</h3>
    <p>
      ในโมเดล Transformer ซึ่งตัดขาดจากโครงสร้างลำดับของ RNN การประมวลผลทั้งหมดจึงเป็นแบบไม่เรียงลำดับ (order-invariant) โดยธรรมชาติของ self-attention layer ไม่มีข้อมูลเกี่ยวกับลำดับตำแหน่งของ token ใน sequence → จึงต้องมีการเพิ่ม positional encoding เพื่อให้โมเดลเข้าใจลำดับได้อย่างถูกต้อง
    </p>

    <h3 className="text-xl font-semibold">แนวทางการออกแบบ: Sinusoidal Encoding</h3>
    <p>
      Vaswani et al. (2017) เสนอการฝังตำแหน่ง (position) ลงใน input vector โดยใช้ฟังก์ชัน sine และ cosine ที่มีความถี่ต่างกัน เพื่อให้โมเดลสามารถแยกแยะลำดับได้โดยไม่ต้องเรียนรู้ explicit positional vector ดังนี้:
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto text-sm">
<code>{`PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))`}</code>
    </pre>

    <h3 className="text-xl font-semibold">คุณสมบัติของ Sin/Cos Encoding</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>รองรับการคำนวณ dot-product ได้โดยตรง</li>
      <li>สามารถ generalize กับตำแหน่งที่ไม่เคยเห็นในการเทรน</li>
      <li>ให้โมเดลเรียนรู้ระยะห่าง (relative distance) โดยธรรมชาติ</li>
      <li>ไม่ต้องเรียนรู้ parameter เพิ่มเติม → memory efficient</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ทำไมต้องใช้ sin/cos</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>การเลือกใช้ฟังก์ชัน sine/cosine ทำให้ตำแหน่งสามารถแจกแจงแบบเชิงเส้นและไม่ซ้ำ</li>
        <li>สามารถใช้คำนวณความสัมพันธ์ระหว่างตำแหน่ง (e.g. pos₁ − pos₂) ได้จาก encoding</li>
        <li>ไม่ต้องเพิ่มพารามิเตอร์ใหม่ให้กับโมเดล → ทำให้ stable กว่า learnable positional encoding ในบางกรณี</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับ Learnable Positional Embeddings</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2 text-left">คุณสมบัติ</th>
          <th className="border px-4 py-2 text-left">Sinusoidal (Fixed)</th>
          <th className="border px-4 py-2 text-left">Learnable</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">สามารถ generalize</td>
          <td className="border px-4 py-2">✔️</td>
          <td className="border px-4 py-2">✖️</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">จำนวนพารามิเตอร์</td>
          <td className="border px-4 py-2">0</td>
          <td className="border px-4 py-2">เพิ่มขึ้นตาม length</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ตีความได้ทางคณิตศาสตร์</td>
          <td className="border px-4 py-2">✔️</td>
          <td className="border px-4 py-2">✖️</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">รองรับ position ใหม่</td>
          <td className="border px-4 py-2">✔️</td>
          <td className="border px-4 py-2">จำกัด</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">การใช้งานใน Transformer Architectures</h3>
    <p>
      ในการสร้าง input embeddings ให้กับ Encoder และ Decoder จะมีการรวม positional encoding กับ token embeddings ผ่านการบวกแบบ element-wise:
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto text-sm">
<code>{`input = word_embedding + positional_encoding`}</code>
    </pre>

    <p>
      วิธีนี้ช่วยให้โมเดลสามารถรับรู้ตำแหน่งขณะประมวลผลใน self-attention โดยไม่ต้องมี recurrent structure ซึ่งทำให้ Transformer มีความสามารถ parallelize ได้อย่างสมบูรณ์
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: Position Matters</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Transformer ไม่สามารถรับรู้ลำดับได้เอง จึงต้องใช้ positional encoding เป็นช่องทาง</li>
        <li>ความสามารถในการเรียนรู้ long-range dependencies ของโมเดลขึ้นกับการ encoding ตำแหน่งที่แม่นยำ</li>
        <li>ตำแหน่งเป็นปัจจัยสำคัญเทียบเท่ากับความหมายของคำในหลาย task</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture 11: Transformer & Attention</li>
      <li>MIT 6.S191 – Deep Learning 2024: Position Embedding</li>
      <li>Harvard NLP Annotated Transformer – Visualizing Positional Encoding</li>
      <li>Oxford Deep NLP Course – Sequence Embeddings and Structure</li>
    </ul>
  </div>
</section>


<section id="self-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Self-Attention Mechanism</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">หลักการพื้นฐานของ Self-Attention</h3>
    <p>
      Self-Attention เป็นกลไกหลักใน Transformer ที่ทำให้ทุกตำแหน่งของ sequence สามารถ "มองเห็น" ทุกตำแหน่งอื่น ๆ ได้พร้อมกันในเวลาเดียวกัน โดยไม่ต้องพึ่งพาโครงสร้างเชิงลำดับแบบ RNN ทำให้สามารถ parallelize ได้อย่างเต็มที่ และจับความสัมพันธ์ระยะไกลระหว่าง token ได้ดี
    </p>

    <h3 className="text-xl font-semibold">กระบวนการคำนวณ Attention</h3>
    <p>ขั้นตอนของ Self-Attention ประกอบด้วย:</p>
    <ol className="list-decimal ml-6 space-y-2">
      <li>นำ input vector แปลงเป็นสามเวกเตอร์: Query (Q), Key (K), Value (V)</li>
      <li>คำนวณ Attention Score โดยการ dot-product ระหว่าง Q และ K<sup>T</sup></li>
      <li>แบ่งผลลัพธ์ด้วย √d<sub>k</sub> เพื่อ normalization</li>
      <li>ผ่าน softmax เพื่อแปลงเป็น weight distribution</li>
      <li>คูณ weights กับ V → ได้ context vector</li>
    </ol>

    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`Attention(Q, K, V) = softmax((QKᵀ) / √dₖ) · V`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ข้อดีของ Self-Attention</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ประมวลผลได้แบบขนาน (parallelization)</li>
      <li>เข้าใจ dependency แบบ all-to-all ได้ดีกว่า RNN</li>
      <li>รองรับ sequence ที่ยาวได้ดี</li>
      <li>สามารถวิเคราะห์บริบทรอบด้านได้ในแต่ละตำแหน่ง</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับ RNN</h3>
    <div className="overflow-x-auto">
      <table className="min-w-[600px] w-full border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">RNN</th>
            <th className="border px-4 py-2">Self-Attention</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">การประมวลผล</td>
            <td className="border px-4 py-2">ลำดับ</td>
            <td className="border px-4 py-2">ขนาน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความสามารถในการมองเห็นข้อมูล</td>
            <td className="border px-4 py-2">จำกัดใน time-step</td>
            <td className="border px-4 py-2">เห็นทั้ง sequence พร้อมกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การจับ dependency ระยะไกล</td>
            <td className="border px-4 py-2">ยากและช้า</td>
            <td className="border px-4 py-2">ทำได้ดีและเร็ว</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: Self-Attention เปลี่ยนมุมมองการประมวลผลข้อมูลลำดับ</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Self-Attention ไม่จำกัดการเชื่อมโยงแบบลำดับอีกต่อไป แต่ใช้ความสัมพันธ์แบบ all-to-all</li>
        <li>เพิ่มศักยภาพในการเรียนรู้โครงสร้างที่ไม่เป็นเชิงเส้น และสามารถปรับได้ตามบริบท</li>
        <li>เปิดทางสู่การสร้างสถาปัตยกรรมใหม่ เช่น BERT, GPT และ Vision Transformer</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงจากมหาวิทยาลัยและงานวิจัยชั้นนำ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al., 2017 — <em>Attention Is All You Need</em> (NeurIPS)</li>
      <li>Stanford CS224n — Lecture: Self-Attention and Transformer Architectures</li>
      <li>MIT 6.S191 — Introduction to Deep Learning: Sequence Models</li>
      <li>Harvard NLP — Annotated Transformer</li>
      <li>Oxford Deep NLP — Module: Transformer and Attention Mechanism</li>
    </ul>
  </div>
</section>


<section id="multi-head" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Multi-Head Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">หลักการของ Multi-Head Attention</h3>
    <p>
      Multi-Head Attention เป็นการขยายแนวคิดของ Self-Attention โดยการใช้หลายชุดของ Query, Key และ Value (เรียกว่า heads) เพื่อให้โมเดลสามารถเรียนรู้ความสัมพันธ์จากหลายมุมมองในแต่ละตำแหน่งของ sequence พร้อมกัน แต่ละ head สามารถโฟกัสที่รูปแบบเฉพาะต่างกัน เช่น โครงสร้างไวยากรณ์, ความหมายเชิงบริบท, หรือความสัมพันธ์ในระยะไกล
    </p>

    <h3 className="text-xl font-semibold">ขั้นตอนการทำงานของ Multi-Head Attention</h3>
    <ol className="list-decimal ml-6 space-y-2">
      <li>สร้าง Linear Projection ของ Q, K, V สำหรับแต่ละ head</li>
      <li>คำนวณ Attention ของแต่ละ head แบบแยกกัน</li>
      <li>นำผลลัพธ์จากทุก head มาต่อกัน (concatenate)</li>
      <li>ผ่าน Linear Layer อีกครั้งเพื่อผสมผลรวม</li>
    </ol>

    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`}</code>
    </pre>

    <h3 className="text-xl font-semibold">เหตุผลที่ต้องใช้หลายหัว (Multiple Heads)</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>แต่ละ head สามารถเรียนรู้ลักษณะความสัมพันธ์เฉพาะที่ต่างกัน</li>
      <li>ช่วยเพิ่ม expressiveness ของโมเดลโดยไม่เพิ่มจำนวนพารามิเตอร์มากนัก</li>
      <li>ลดปัญหาการ overfit บริบทที่จำกัดเพียงมุมมองเดียว</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight Box:</strong> Multi-Head Attention เปรียบเหมือนการมีหลายผู้เชี่ยวชาญแยกกันพิจารณาข้อมูลแต่ละตำแหน่ง แล้วรวมความเห็นเข้าด้วยกันในภายหลัง</p>
    </div>

    <h3 className="text-xl font-semibold">โครงสร้างของ Multi-Head Attention ใน Transformer</h3>
    <p>ใน Transformer ตามแบบของ Vaswani et al. (2017) Multi-Head Attention ถูกใช้งานทั้งใน Encoder และ Decoder โดยมีรายละเอียดสำคัญดังนี้:</p>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>จำนวนหัว (h):</strong> 8 หรือ 16 หัวขึ้นอยู่กับค่า <code>d_model</code></li>
      <li><strong>ขนาดของแต่ละ head:</strong> <code>d_k = d_model / h</code></li>
      <li><strong>การ Sharing พารามิเตอร์:</strong> ใช้ Linear Layer แยกกันสำหรับแต่ละ head</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ: Single-Head vs Multi-Head</h3>
    <div className="w-full overflow-x-auto">
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
            <td className="border px-4 py-2">มุมมองข้อมูล</td>
            <td className="border px-4 py-2">มุมมองเดียว</td>
            <td className="border px-4 py-2">หลากหลาย</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความสามารถในการเรียนรู้</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">สูงกว่า</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ค่าใช้จ่ายในการคำนวณ</td>
            <td className="border px-4 py-2">ต่ำ</td>
            <td className="border px-4 py-2">สูงขึ้น</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">เหมาะกับลำดับซับซ้อน</td>
            <td className="border px-4 py-2">ไม่มาก</td>
            <td className="border px-4 py-2">มาก</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงระดับโลก</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017) – <em>Attention Is All You Need</em> (NeurIPS)</li>
      <li>Stanford CS224n – Lecture 11: Multi-Head Attention</li>
      <li>MIT 6.S191 – <em>Deep Learning for Sequences</em></li>
      <li>Oxford Deep NLP – <em>Transformer Architectures</em></li>
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
    </ul>

  </div>
</section>


<section id="ffn" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Feed Forward Network (FFN)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">โครงสร้างพื้นฐานของ FFN</h3>
    <p>
      Feed Forward Network (FFN) เป็นองค์ประกอบสำคัญของสถาปัตยกรรม Transformer ที่ทำหน้าที่ประมวลผลเชิงความหมายของ token แต่ละตำแหน่ง โดยมีโครงสร้างเป็น multilayer perceptron (MLP) แบบเรียบง่าย ซึ่งทำงานแบบ position-wise คือ ใช้ฟังก์ชันเดียวกันกับทุก token อย่างเป็นอิสระ
    </p>

    <h3 className="text-xl font-semibold">สูตรคณิตศาสตร์ของ FFN</h3>
    <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`}</code>
    </pre>
    <p>
      โดยที่ <code>W₁</code>, <code>b₁</code>, <code>W₂</code>, และ <code>b₂</code> คือพารามิเตอร์ที่เรียนรู้ได้ (trainable parameters) และ <code>max(0, ...)</code> คือ ReLU activation function
    </p>

    <h3 className="text-xl font-semibold">เหตุผลที่ใช้ FFN ใน Transformer</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>เพิ่ม non-linearity ให้กับการแปลงความหมายของ token</li>
      <li>สามารถทำงานแบบขนานกับทุกตำแหน่งใน sequence ได้</li>
      <li>ช่วยเพิ่มความสามารถในการแสดงความสัมพันธ์ที่ซับซ้อนของข้อมูล</li>
    </ul>

    <h3 className="text-xl font-semibold">ขนาดของ hidden layer</h3>
    <p>
      ใน Transformer ต้นฉบับ (Vaswani et al., 2017) มีการใช้ขนาด <code>d_model = 512</code> และ <code>d_ff = 2048</code> ซึ่งหมายความว่า hidden layer ของ FFN มีขนาดใหญ่กว่าข้อมูล input ถึง 4 เท่า เพื่อให้เกิดการแปลงเชิงลึกที่เพียงพอ
    </p>

    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">Component</th>
          <th className="border px-4 py-2">Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Input</td>
          <td className="border px-4 py-2">เวกเตอร์ของแต่ละ token ขนาด d_model</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Hidden Layer</td>
          <td className="border px-4 py-2">Linear → ReLU → Linear</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Output</td>
          <td className="border px-4 py-2">เวกเตอร์ที่ผ่านการแปลง พร้อมสำหรับ Residual</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">การทำงานร่วมกับ Self-Attention</h3>
    <p>
      FFN จะถูกวางไว้หลัง Multi-Head Attention และก่อน Layer Normalization และ Residual connection เพื่อเสริมความสามารถของโมเดลในการทำ abstraction เพิ่มเติมจากข้อมูลที่ได้รับจาก attention
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>แม้จะดูเรียบง่าย แต่ FFN มีบทบาทสำคัญในการแปลงข้อมูลเชิงลึก</li>
        <li>การทำงานแบบ position-wise ทำให้สามารถประมวลผลแบบขนานได้ 100%</li>
        <li>การใช้ hidden layer ขนาดใหญ่ช่วยให้โมเดลสามารถ encode feature ที่ซับซ้อนได้</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ด FFN ด้วย PyTorch</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))`}</code>
    </pre>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017) – <em>Attention Is All You Need</em></li>
      <li>Stanford CS224n – Lecture 11: Transformer Internal Layers</li>
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>MIT 6.S191 – Deep Learning Series: Transformer Blocks</li>
      <li>CMU Neural Nets Course – FFN in Sequence Models</li>
    </ul>
  </div>
</section>

<section id="residual-norm" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Residual & Layer Normalization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวคิดของ Residual Connection</h3>
    <p>
      ในโครงข่ายประสาทเทียมลึก (Deep Neural Networks) การฝึกโมเดลที่มีหลายเลเยอร์มักเผชิญกับปัญหา gradient vanishing หรือ degradation ซึ่งทำให้เลเยอร์ลึกลงเรียนรู้ได้ยากขึ้น การเพิ่ม <strong>Residual Connection</strong> หรือ skip connection เป็นแนวทางหนึ่งที่ถูกเสนอในงานของ He et al. (2015) เพื่อลดผลกระทบนี้ โดยเฉพาะในสถาปัตยกรรมเช่น ResNet และต่อมาถูกนำมาใช้ใน Transformer อย่างแพร่หลาย
    </p>

    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`output = LayerNorm(x + Sublayer(x))`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ประโยชน์ของ Residual ใน Transformer</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ช่วยให้ข้อมูลไหลผ่านโมเดลได้ลึกขึ้นโดยไม่สูญเสีย signal</li>
      <li>ลดปัญหา gradient ที่อ่อนในระหว่าง backpropagation</li>
      <li>ทำให้สามารถฝึกสถาปัตยกรรมที่มีหลายเลเยอร์ (เช่น N=6) ได้อย่างมีประสิทธิภาพ</li>
    </ul>

    <h3 className="text-xl font-semibold">บทบาทของ Layer Normalization</h3>
    <p>
      Layer Normalization (LayerNorm) ถูกเสนอโดย Ba et al. (2016) เพื่อช่วยให้โมเดลมีการกระจายค่าที่เสถียรขึ้นในแต่ละเลเยอร์ โดยจะคำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานในแนว feature แล้วทำการ normalize ค่าทุกตำแหน่งใน input vector
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>ต่างจาก Batch Normalization ตรงที่ไม่ขึ้นกับขนาด batch</li>
      <li>ทำให้เหมาะกับ task ที่ต้องการ inference แบบ sequence เช่น language modeling</li>
      <li>ช่วยให้ training เสถียรขึ้นในโมเดลที่ลึกมาก</li>
    </ul>

<pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
  <code>
    {"LayerNorm(x) = (x - μ) / (σ + ε) * γ + β"}
  </code>
</pre>

    <h3 className="text-xl font-semibold">โครงสร้างของ Block ใน Transformer</h3>
    <p>
      ในแต่ละเลเยอร์ของ Transformer ทั้งฝั่ง Encoder และ Decoder จะมี 2 sublayer:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Multi-Head Attention</strong> + Residual + LayerNorm</li>
      <li><strong>Feed Forward Network (FFN)</strong> + Residual + LayerNorm</li>
    </ul>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`x1 = LayerNorm(x + MultiHeadAttention(x))
x2 = LayerNorm(x1 + FFN(x1))`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ประเด็นสำคัญในการ implement</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ตำแหน่งของ LayerNorm มีผลต่อประสิทธิภาพ: ใช้ <strong>Post-LN</strong> (ตาม Vaswani et al.) หรือ <strong>Pre-LN</strong> (จากงานของ Xiong et al. 2020)</li>
      <li>Pre-LN ช่วยให้ training เสถียรกว่าในโมเดลที่ลึกมาก</li>
      <li>โมเดลอย่าง BERT ใช้ Post-LN ขณะที่ GPT ใช้ Pre-LN</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: Residual ช่วยให้เรียนลึกได้</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Residual Connection ทำให้ input สามารถผ่านได้โดยไม่ต้องเปลี่ยนรูป</li>
        <li>LayerNorm ช่วยให้ไม่ต้องพึ่ง batch statistics → เหมาะกับ sequence</li>
        <li>แนวคิด Residual + Norm ได้กลายเป็นมาตรฐานของ deep architectures</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>He et al. (2015). <em>Deep Residual Learning for Image Recognition</em> – CVPR</li>
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em> – NeurIPS</li>
      <li>Ba et al. (2016). <em>Layer Normalization</em> – arXiv:1607.06450</li>
      <li>Xiong et al. (2020). <em>On Layer Normalization in the Transformer Architecture</em> – ICML</li>
      <li>MIT 6.S191 – Lecture: Stabilizing Deep Transformers</li>
      <li>Stanford CS224n – Lecture: Transformer Internals and Norm Tricks</li>
    </ul>
  </div>
</section>


<section id="masked-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Masked Self-Attention (ใน Decoder)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">หลักการของ Masked Self-Attention</h3>
    <p>
      ในโมเดล Transformer ส่วนของ Decoder จำเป็นต้องเรียนรู้การสร้างข้อความแบบทีละคำ (auto-regressive) โดยไม่สามารถเห็นคำในอนาคตได้ ดังนั้นระบบจำเป็นต้องใช้เทคนิคที่เรียกว่า Masked Self-Attention เพื่อป้องกันไม่ให้ Decoder “แอบมอง” คำที่ยังไม่ได้สร้าง ซึ่งจะทำให้การเรียนรู้ของโมเดลมีความถูกต้องตามลำดับเวลา
    </p>

    <h3 className="text-xl font-semibold">กลไกการปิดบังใน Attention Matrix</h3>
    <p>
      ในระหว่างการคำนวณ Attention Scores โมเดลจะสร้าง Matrix ที่เปรียบเทียบทุกตำแหน่ง token กับตำแหน่งอื่น ๆ ภายในลำดับของตัวเอง สำหรับ Masked Self-Attention จะมีการตั้งค่าบางตำแหน่งใน Matrix ให้เป็น -∞ เพื่อให้ค่าที่ได้จาก softmax เป็น 0 ซึ่งทำให้โมเดลไม่สามารถให้ความสนใจกับ token ที่อยู่ถัดไปได้
    </p>

    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`Mask[i][j] =
  0 if j <= i
  -∞ if j > i`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ตัวอย่างภาพรวมของการ Mask</h3>
    <p>
      สมมติลำดับมีความยาว 4 ตำแหน่ง การคำนวณ Mask Matrix จะมีลักษณะเป็นสามเหลี่ยมล่างดังนี้:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 text-sm rounded overflow-x-auto">
<code>{`[[0, -∞, -∞, -∞],
 [0,  0, -∞, -∞],
 [0,  0,  0, -∞],
 [0,  0,  0,  0]]`}</code>
    </pre>

    <h3 className="text-xl font-semibold">ประโยชน์ของการ Mask</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>รักษาลำดับเวลาแบบ causal (auto-regressive generation)</li>
      <li>ใช้ใน Language Modeling เช่น GPT ที่ต้องสร้าง token ทีละตัว</li>
      <li>เพิ่มความแม่นยำในการประเมินผลลัพธ์ระยะยาว</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Masked Self-Attention ทำให้ Decoder ของ Transformer เป็นแบบ unidirectional</li>
        <li>ช่วยป้องกัน “data leakage” ระหว่างการเทรนในลักษณะ Language Modeling</li>
        <li>เทคนิคนี้ถูกนำไปใช้ในโมเดลขนาดใหญ่ เช่น GPT-2, GPT-3 อย่างเต็มรูปแบบ</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ความแตกต่างระหว่าง Self-Attention และ Masked Self-Attention</h3>
    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Self-Attention</th>
          <th className="border px-4 py-2">Masked Self-Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การมองข้อมูล</td>
          <td className="border px-4 py-2">ทุกตำแหน่ง</td>
          <td className="border px-4 py-2">เฉพาะตำแหน่งก่อนหน้า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การใช้ใน Encoder</td>
          <td className="border px-4 py-2">✔️</td>
          <td className="border px-4 py-2">✖️</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การใช้ใน Decoder</td>
          <td className="border px-4 py-2">✖️</td>
          <td className="border px-4 py-2">✔️</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017) – <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture 10: Self-Attention and Transformers</li>
      <li>Harvard NLP – <em>Annotated Transformer</em> Repository</li>
      <li>MIT 6.S191 – Deep Learning for NLP (2024 Edition)</li>
    </ul>
  </div>
</section>


<section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Visualization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">บทบาทของ Visualization ใน Transformer</h3>
    <p>
      Visualization ช่วยเปิดเผยการทำงานภายในของโมเดล Transformer ซึ่งยากจะเข้าใจจากค่าพารามิเตอร์เพียงอย่างเดียว การแสดง Attention Maps หรือ Dependency Graphs ช่วยให้นักวิจัยและวิศวกรสามารถวิเคราะห์ความสัมพันธ์ภายใน sequence ได้อย่างแม่นยำ
    </p>

    <h3 className="text-xl font-semibold">Types of Visualization</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Attention Heatmaps:</strong> แสดงความสัมพันธ์ระหว่าง tokens แบบ 2 มิติ</li>
      <li><strong>Graph-Based Views:</strong> แสดงการเชื่อมโยงระหว่าง token nodes</li>
      <li><strong>Interactive Visualizations:</strong> ใช้ใน Tools เช่น BertViz, TransformerLens</li>
    </ul>

    <h3 className="text-xl font-semibold">กรณีศึกษา: การตีความ Attention</h3>
    <p>
      จากงานของ Harvard NLP และ Distill.pub การ visualize attention weights ในโมเดล BERT พบว่า layer ที่ต่างกันโฟกัสกับโครงสร้างภาษาต่างกัน เช่น layer ต้นอาจโฟกัสไปยัง POS (part of speech) ขณะที่ layer ท้ายเน้น semantic dependency
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: ความสำคัญของ Visualization</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ช่วยตรวจจับ bias หรือ error ใน attention</li>
        <li>เปิดโอกาสในการปรับปรุง architecture ตามพฤติกรรมที่พบ</li>
        <li>ช่วยอธิบายการทำงานของโมเดลกับผู้ใช้งานทั่วไป (interpretable AI)</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">เครื่องมือยอดนิยมในการทำ Visualization</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><code>transformer-smith</code>: ใช้ในงาน NLP ที่ต้องการ interpret attention</li>
      <li><code>BertViz</code>: Visualize attention ใน BERT/GPT แบบ inter-layer</li>
      <li><code>TransformerLens</code>: สำหรับ interpret model mechanics โดยเฉพาะ</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดการวาด Attention Heatmap</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
<code>{`import matplotlib.pyplot as plt\nimport seaborn as sns\n\ndef plot_attention(attn_weights, tokens):\n    sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')\n    plt.title("Attention Heatmap")\n    plt.xlabel("Key Tokens")\n    plt.ylabel("Query Tokens")\n    plt.show()`}</code>
    </pre>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Harvard NLP – <em>Annotated Transformer</em></li>
      <li>Alammar, J. – <em>The Illustrated Transformer</em></li>
      <li>Vig, J. – <em>BertViz: Visualizing Attention in Transformer Models</em> (EMNLP 2019)</li>
      <li>Distill.pub – <em>Visualizing and Understanding Self-Attention</em></li>
      <li>MIT 6.S191 – Deep Learning: Interpretable Models</li>
    </ul>
  </div>
</section>


<section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Academic Reference</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-8">

    <h3 className="text-xl font-semibold">แหล่งข้อมูลหลักจากการพัฒนา Transformer</h3>
    <p>
      โมเดล Transformer กลายเป็นรากฐานของสถาปัตยกรรม deep learning ที่ใช้ attention อย่างเต็มรูปแบบ โดยมีจุดเริ่มจากบทความวิจัยระดับตำนานอย่าง “Attention Is All You Need” ที่ได้เสนอ self-attention และ multi-head attention แทนการใช้ recurrence แบบเดิม
    </p>

    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>
        Vaswani, A., et al. (2017). 
        <em>Attention Is All You Need.</em> NeurIPS. 
        [<a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener noreferrer">arXiv:1706.03762</a>]
      </li>
      <li>
        Alammar, J. (2018). 
        <em>The Illustrated Transformer.</em> Blog. 
        [<a href="http://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noopener noreferrer">jalammar.github.io</a>]
      </li>
      <li>
        Stanford CS25: 
        <em>Modern NLP with Deep Learning</em> – Lecture Series on Transformer Architectures.
      </li>
      <li>
        Harvard NLP Group. 
        <em>The Annotated Transformer.</em> GitHub. 
        [<a href="https://github.com/harvardnlp/annotated-transformer" target="_blank" rel="noopener noreferrer">github.com/harvardnlp</a>]
      </li>
      <li>
        MIT 6.S191 (2024 Edition). 
        <em>Introduction to Deep Learning – Attention and Transformer Modules.</em>
      </li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลเพิ่มเติมจากสถาบันชั้นนำ</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>
        Oxford Deep NLP Course – Sequence Models and Attention Mechanisms.
      </li>
      <li>
        CMU Neural Nets for NLP – Module: Transformers and Self-Attention.
      </li>
      <li>
        Google Research – Scaling Transformers and Efficient Attention.
      </li>
      <li>
        Facebook AI Research (FAIR) – Transformer Evolution and Multilingual Pretraining.
      </li>
      <li>
        Hugging Face – Open-Source Transformers, Datasets, and Pretrained Models.
      </li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Highlight Box: การเลือกใช้แหล่งอ้างอิง</h3>
      <ul className="list-disc list-inside text-sm space-y-2">
        <li>เลือกอ้างอิงงานที่ peer-reviewed หรือมีผลกระทบสูง เช่น NeurIPS, ACL, ICML</li>
        <li>ใช้บทความที่มีการ implement จริง พร้อมการอธิบายเข้าใจง่าย เช่น Alammar และ Harvard NLP</li>
        <li>พิจารณา citation count และผลกระทบเชิงวิชาการจาก Google Scholar</li>
        <li>ผสานการเรียนจาก Lecture Notes ของมหาวิทยาลัยกับโค้ดจริงบน GitHub</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">การนำไปใช้งานจริง</h3>
    <p>
      เอกสารอ้างอิงที่กล่าวมา ไม่เพียงช่วยสร้างความเข้าใจเชิงแนวคิด แต่ยังมีบทบาทในการออกแบบโมเดลจริง การสร้างระบบ pretraining ขนาดใหญ่ เช่น BERT, GPT, และ ViT ล้วนใช้โครงสร้าง Transformer ที่วิวัฒนาการมาจากงานเหล่านี้โดยตรง
    </p>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>เริ่มจาก Vaswani et al. (2017) เพื่อเข้าใจแกนหลักของ Transformer</li>
      <li>ศึกษาผ่านภาพและโค้ดจาก Alammar และ Harvard NLP</li>
      <li>ใช้ implementation ของ Hugging Face เพื่อเรียนรู้และ deploy โมเดลจริง</li>
    </ul>
  </div>
</section>


<section id="practical-tips" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Practical Tips</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ค่าพารามิเตอร์มาตรฐานจากงานวิจัยต้นฉบับ</h3>
    <p>
      การออกแบบ Transformer ที่เสถียรและมีประสิทธิภาพ จำเป็นต้องตั้งค่าพารามิเตอร์เริ่มต้นที่เหมาะสม ซึ่งอ้างอิงจากบทความ <em>Attention Is All You Need</em> (Vaswani et al., 2017) โดยค่าที่แนะนำเป็นดังนี้:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>d<sub>model</sub> = 512:</strong> ขนาด embedding vector</li>
      <li><strong>n<sub>head</sub> = 8:</strong> จำนวน attention heads</li>
      <li><strong>N = 6:</strong> จำนวน layers ทั้งใน Encoder และ Decoder</li>
      <li><strong>d<sub>ff</sub> = 2048:</strong> ขนาด hidden layer ของ FFN</li>
    </ul>

    <h3 className="text-xl font-semibold">การใช้ Positional Encoding ร่วมกับโมเดล Pretrained</h3>
    <p>
      แม้จะใช้โมเดลที่ผ่านการฝึกมาแล้ว เช่น BERT หรือ GPT การคงรูปแบบของ Positional Encoding เดิมไว้เป็นสิ่งสำคัญ เพราะช่วยให้ alignment ระหว่าง embedding กับ position ไม่ถูกทำลาย ซึ่งอาจส่งผลให้ performance ลดลงเมื่อ transfer learning
    </p>

    <h3 className="text-xl font-semibold">เทคนิคเพิ่มความเสถียรในการฝึก (Stabilization)</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้ <strong>batch size ขนาดใหญ่</strong> ร่วมกับ gradient accumulation หากหน่วยความจำจำกัด</li>
      <li>เปิดใช้งาน <strong>Layer Normalization หลังทุก residual block</strong></li>
      <li>เริ่มด้วย learning rate ต่ำ แล้วใช้ <strong>learning rate warm-up</strong> 4000 steps (ตาม original paper)</li>
      <li>ใช้ <code>label smoothing</code> เพื่อลด overconfidence ของ logits ในการคาดคะเน</li>
    </ul>

    <h3 className="text-xl font-semibold">การประเมิน performance</h3>
    <p>
      Transformer สามารถประเมินผ่านหลายตัวชี้วัด เช่น BLEU, ROUGE, accuracy หรือ perplexity ขึ้นอยู่กับงานที่นำไปใช้ การวัดค่าเหล่านี้ต้องดำเนินอย่างสม่ำเสมอใน validation set พร้อมทำ early stopping เมื่อ performance เริ่มนิ่ง
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: เคล็ดลับจากงานวิจัยระดับโลก</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>โมเดลขนาดเล็ก (<code>d_model = 128</code>) เหมาะสำหรับ deployment ใน edge device</li>
        <li>การใช้ checkpoint ที่ผ่าน pretraining แล้ว fine-tune บน domain-specific dataset จะให้ผลดีกว่าฝึกใหม่จากศูนย์</li>
        <li>ตำแหน่งของ masking ใน decoder มีผลอย่างมากต่อประสิทธิภาพ → ต้องแน่ใจว่า future tokens ถูก block อย่างถูกต้อง</li>
        <li>ตรวจสอบค่าความแปรปรวนของ attention weights ระหว่าง heads เพื่อหา dead heads ที่ไม่ได้เรียนรู้</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิงเชิงเทคนิค</h3>
    <ul className="list-disc ml-6 space-y-2 text-sm">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need.</em> NeurIPS.</li>
      <li>Stanford CS224n – Lecture 12: Transformer Training & Optimization</li>
      <li>Harvard NLP Annotated Transformer – <em>transformer.py</em> implementation</li>
      <li>Hugging Face – Documentation: Optimizer and Scheduler Strategies</li>
      <li>MIT 6.S191 – 2024 Edition: Deep Learning Practical Tips for Training Large Models</li>
    </ul>

  </div>
</section>


<section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Limitations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ค่าสลับที่ต้องจ่ายจากการใช้ Self-Attention</h3>
    <p>
      แม้ Transformer จะมีความสามารถสูงในการเรียนรู้ dependency ทั่วทั้ง sequence แต่ก็มีข้อจำกัดในหลายด้าน โดยเฉพาะด้าน computational complexity ซึ่งถูกยกเป็นข้อวิจารณ์หลักในการประยุกต์ใช้กับลำดับที่ยาวมาก เช่นในเอกสารหลายหน้า หรือ DNA sequence
    </p>

    <h3 className="text-xl font-semibold">1. Computational Complexity: O(n²)</h3>
    <p>
      หนึ่งในข้อจำกัดสำคัญที่สุดของ Transformer คือ cost ของ self-attention layer ซึ่งต้องคำนวณ dot-product ของทุก token กับทุก token ใน sequence:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded text-sm overflow-x-auto">
<code>{`Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V → O(n²d)`}</code>
    </pre>
    <p>
      สำหรับลำดับที่มีขนาด <code>n</code> tokens การคำนวณนี้ส่งผลให้ใช้หน่วยความจำและเวลาในการประมวลผลเพิ่มขึ้นอย่างรวดเร็ว
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>ไม่เหมาะกับ sequence ยาวเช่น document, audio waveform</li>
      <li>ใช้ memory มากเมื่อ batch size และจำนวน heads เพิ่มขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">2. Inefficiency on Edge Devices</h3>
    <p>
      แม้ Transformer จะทำงานดีบน GPU หรือ TPU ที่มีหน่วยความจำสูง แต่ในกรณีของ real-time applications หรืออุปกรณ์ edge เช่นมือถือหรืออุปกรณ์ IoT, การใช้ self-attention แบบเต็มรูปแบบมักเกินกำลังของอุปกรณ์เหล่านั้น
    </p>

    <h3 className="text-xl font-semibold">3. Lack of Recurrence</h3>
    <p>
      Transformer ตัด RNN ออกโดยสิ้นเชิง ซึ่งส่งผลดีด้านการประมวลผลแบบขนาน แต่ในบางงานที่ข้อมูลมีโครงสร้างแบบ sequential ที่แข็งแรง เช่น time series หรือ music, การไม่มี recurrence อาจทำให้การเรียนรู้ลำดับเวลาลึก ๆ บางรูปแบบเป็นเรื่องยาก
    </p>

    <h3 className="text-xl font-semibold">4. ความต้องการใน Training Data และเวลา</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ต้องใช้ dataset ขนาดใหญ่ในการฝึกให้เกิด generalization</li>
      <li>ต้องการการตั้งค่า learning rate, warm-up, batch size ที่ซับซ้อน</li>
      <li>มีโอกาส overfit หากไม่มี regularization ที่ดี</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <h3 className="text-lg font-semibold mb-2">Insight Box: ทำไมยังต้องพัฒนา Transformer ต่อ</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>แม้ self-attention จะเป็นนวัตกรรมหลัก แต่ยังมีช่องว่างด้าน efficiency และ scalability</li>
        <li>ทำให้เกิดงานวิจัยต่อยอด เช่น <strong>Performer, Linformer, Longformer, BigBird</strong> เพื่อแก้ปัญหา O(n²)</li>
        <li>บางแอปพลิเคชันเช่น streaming หรือ embedded systems ต้องการ hybrid model เช่น LSTransformer หรือ linear attention</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ: RNN vs Transformer</h3>
    <div className="overflow-x-auto">
      <table className="min-w-[680px] border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Aspect</th>
            <th className="border px-4 py-2">RNN / LSTM</th>
            <th className="border px-4 py-2">Transformer</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Parallelization</td>
            <td className="border px-4 py-2">✖️</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Long-Term Dependency</td>
            <td className="border px-4 py-2">Gradient Vanishing</td>
            <td className="border px-4 py-2">Handled via Attention</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Efficiency (Short Seq)</td>
            <td className="border px-4 py-2">ดีกว่า</td>
            <td className="border px-4 py-2">แพ้เมื่อ seq สั้น</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Efficiency (Long Seq)</td>
            <td className="border px-4 py-2">ช้ากว่า</td>
            <td className="border px-4 py-2">เร็วเมื่อมี GPU, แต่ใช้ memory มาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Memory Complexity</td>
            <td className="border px-4 py-2">O(n)</td>
            <td className="border px-4 py-2">O(n²)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Katharopoulos et al. (2020). <em>Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention</em>. ICML.</li>
      <li>Zaheer et al. (2020). <em>BigBird: Transformers for Longer Sequences</em>. NeurIPS.</li>
      <li>Wang et al. (2020). <em>Linformer: Self-Attention with Linear Complexity</em>. arXiv.</li>
      <li>Beltagy et al. (2020). <em>Longformer: The Long-Document Transformer</em>. arXiv.</li>
    </ul>

  </div>
</section>


<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">13. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">ความเปลี่ยนแปลงเชิงวิธีคิดที่นำโดย Transformer</h3>
    <p>
      Transformer ไม่ได้เป็นเพียงการออกแบบโครงสร้างใหม่ในเชิงวิศวกรรมเท่านั้น หากแต่เป็นการเปลี่ยนวิธีคิดพื้นฐานของการเรียนรู้ลำดับ (sequence learning) จากการมองแบบ “ขั้นต่อขั้น” → สู่ “การเรียนรู้ความสัมพันธ์ทั้งหมดพร้อมกัน” (All-to-All Dependencies) นี่คือการสลัดข้อจำกัดที่เคยฝังรากใน RNN/LSTM ทั้งในด้านเวลา, ความสามารถในการจำลำดับยาว, และการเรียนรู้ที่พึ่งพาความต่อเนื่อง
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p className="font-semibold">
        <strong>Insight:</strong> Transformer ไม่ใช่แค่โมเดลใหม่—แต่คือการเปลี่ยนวิธีคิดจาก <em>"ลำดับ"</em> → <em>"ความสัมพันธ์ทั้งหมดในครั้งเดียว"</em> (All-to-All Dependencies)
      </p>
    </div>

    <h3 className="text-xl font-semibold">สิ่งที่ Transformer สร้างขึ้นในวงการวิจัย</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>เปิดทางให้เกิดโมเดลระดับโลก เช่น BERT, GPT, T5, ViT, DALL·E, Stable Diffusion</li>
      <li>ลด dependency ต่อคำสั่งลำดับเวลา (recurrence) และทำให้ประมวลผลได้แบบ parallel เต็มรูปแบบ</li>
      <li>ทำให้ระบบ language models กลายเป็น foundation models ที่ฝึกบนข้อมูลขนาดใหญ่แล้วนำไปปรับใช้ได้ทุกงาน</li>
    </ul>

    <h3 className="text-xl font-semibold">มุมมองจากงานวิจัยระดับสากล</h3>
    <p>
      สถาบันอย่าง Stanford, MIT และ Harvard ต่างอธิบายว่า Transformer เปรียบเสมือนการสร้างสถาปัตยกรรมโมเดลที่ "ไม่จำกัดงาน" อีกต่อไป กล่าวคือ ไม่ได้ออกแบบเฉพาะ NLP แต่สามารถนำไปใช้กับ Vision, Speech, Music, Biology, และ Multimodal Learning ได้ทันทีโดยไม่ต้องเปลี่ยนโครงสร้าง
    </p>

    <h3 className="text-xl font-semibold">เมื่อการเรียนรู้คือ "ความสัมพันธ์" ไม่ใช่ "ลำดับ"</h3>
    <p>
      ก่อนการมาถึงของ Transformer โมเดลลำดับใช้ลำดับเวลาเป็นแกนหลักในการตัดสินใจ (sequential constraint) ซึ่งทำให้เรียนรู้บริบทที่ไกลออกไปได้ยาก โดยเฉพาะใน long sequence หรือข้อมูลที่มีความสัมพันธ์เชิงโครงสร้าง ในขณะที่ Transformer เปิดโอกาสให้โมเดลสามารถมองเห็นทุก token ในเวลาเดียวกัน ทำให้สามารถจับ context และ dependency ได้อย่างลึกซึ้งและยืดหยุ่นกว่ามาก
    </p>

    <h3 className="text-xl font-semibold">ผลกระทบที่กว้างไกล</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>โมเดลอย่าง GPT-4 สามารถเรียนรู้โครงสร้างภาษา การเขียนโค้ด และการวิเคราะห์ตรรกะได้แบบ end-to-end</li>
      <li>ViT (Vision Transformer) ทำให้เกิด paradigm shift จาก CNN → Transformer ในงาน vision</li>
      <li>โมเดลอย่าง AlphaFold ใช้ attention ในการพยากรณ์โครงสร้างโปรตีนระดับโมเลกุล</li>
      <li>Self-attention ถูกใช้ใน Graph Neural Networks, Audio Modeling, Multilingual Translation</li>
    </ul>

    <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Devlin et al. (2018). <em>BERT: Pre-training of Deep Bidirectional Transformers</em>. NAACL.</li>
      <li>Dosovitskiy et al. (2020). <em>An Image is Worth 16x16 Words: Vision Transformer</em>. ICLR.</li>
      <li>Jumper et al. (2021). <em>AlphaFold: Highly Accurate Protein Structure Prediction</em>. Nature.</li>
      <li>Stanford CS25 – Modern NLP with Deep Learning: Transformers in Practice</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day31 theme={theme} />
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
        <ScrollSpy_Ai_Day31 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day31_TransformerOverview;
