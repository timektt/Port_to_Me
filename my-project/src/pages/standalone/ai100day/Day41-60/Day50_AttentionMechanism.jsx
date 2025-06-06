import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day50 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day50";
import MiniQuiz_Day50 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day50";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day50_AttentionMechanism = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day50_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day50_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day50_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day50_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day50_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day50_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day50_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day50_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day50_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day50_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day50_11").format("auto").quality("auto").resize(scale().width(501));

return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 50: Attention Mechanism in Deep Learning</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-700 via-white to-yellow-700 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          {/* Section 1 */}
     <section id="intro" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: แนวคิด Attention คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">พื้นฐานของ Attention Mechanism</h3>
    <p>
      แนวคิด Attention Mechanism ได้รับการพัฒนาขึ้นเพื่อตอบโจทย์ข้อจำกัดของการประมวลผลลำดับข้อมูลในโมเดลแบบ Recurrent Neural Networks (RNNs) และ Long Short-Term Memory (LSTM) โดยเฉพาะอย่างยิ่งในบริบทของ Natural Language Processing (NLP) และ Machine Translation ที่ต้องจัดการกับลำดับข้อมูลที่มีความยาวและความซับซ้อนสูง
    </p>
    <p>
      การใช้ Attention ช่วยให้โมเดลสามารถ "โฟกัส" ไปยังส่วนที่สำคัญของข้อมูลอินพุต ณ เวลาที่ต้องการสร้างผลลัพธ์แต่ละตำแหน่ง โดยไม่จำเป็นต้องบีบอัดข้อมูลทั้งหมดให้อยู่ใน context vector เพียงเวกเตอร์เดียว ซึ่งเป็นข้อจำกัดของสถาปัตยกรรม Encoder-Decoder แบบดั้งเดิม
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> การใช้ Attention ได้เปลี่ยนแปลงแนวทางการออกแบบโมเดล Neural Network จากการประมวลผลแบบลำดับเชิงเส้น (sequential) มาเป็นการเรียนรู้ความสัมพันธ์เชิงบริบท (contextual relationships) อย่างยืดหยุ่นและมีประสิทธิภาพสูง
    </div>

    <h3 className="text-xl font-semibold">กระบวนการทำงานของ Attention</h3>
    <p>
      Attention Mechanism ทำงานโดยการคำนวณค่า **similarity score** ระหว่างตำแหน่งปัจจุบันของผลลัพธ์ที่กำลังสร้าง (query) กับแต่ละตำแหน่งของข้อมูลอินพุต (keys) จากนั้นนำ score เหล่านี้มาใช้เป็นน้ำหนัก (weights) สำหรับการรวมค่าจาก value vectors ของตำแหน่งอินพุตทั้งหมด
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li><strong>Query (Q):</strong> เวกเตอร์ที่แทนข้อมูล ณ ตำแหน่ง output ปัจจุบัน</li>
      <li><strong>Key (K):</strong> เวกเตอร์ที่แทนข้อมูลแต่ละตำแหน่งของอินพุต</li>
      <li><strong>Value (V):</strong> เวกเตอร์ข้อมูลจริงที่ใช้ในการรวมค่า weighted sum</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> Softmax normalization ช่วยให้ค่า similarity score ถูกแปลงเป็น distribution ของ attention weights ที่มีผลรวมเป็น 1 ซึ่งเป็นการควบคุมการโฟกัสของโมเดลอย่างมีประสิทธิภาพ
    </div>

    <h3 className="text-xl font-semibold">การเปรียบเทียบกับสถาปัตยกรรมแบบดั้งเดิม</h3>
    <p>
      ก่อนการถือกำเนิดของ Attention Mechanism โมเดลแบบ Encoder-Decoder จำเป็นต้องพึ่งพา context vector เดียวสำหรับสรุปข้อมูลทั้งหมดจากอินพุต ซึ่งสร้างข้อจำกัดอย่างมีนัยสำคัญเมื่อจัดการกับลำดับข้อมูลที่ยาวหรือมีโครงสร้างซับซ้อน เนื่องจากข้อมูลสำคัญบางส่วนอาจถูกละเลยไป
    </p>
<div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Feature</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Encoder-Decoder แบบดั้งเดิม</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Encoder-Decoder with Attention</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Context Vector</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Single vector</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Dynamic weighted sum of input</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Ability to handle long sequences</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Improved significantly</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Interpretability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Low</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">High (via attention weights)</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Efficiency</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Lower</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Higher (parallelizable)</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">วิวัฒนาการของ Attention ใน Deep Learning</h3>
    <p>
      แนวคิด Attention เริ่มต้นจากการประยุกต์ในงาน Neural Machine Translation (Bahdanau et al., 2015) และต่อมาได้กลายเป็นองค์ประกอบหลักในสถาปัตยกรรมที่ทันสมัย เช่น Transformer (Vaswani et al., 2017) ซึ่งได้ปฏิวัติการออกแบบโมเดล Deep Learning โดยเฉพาะในสาขา NLP, Vision, และ Multimodal Learning
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>**2015:** Bahdanau Attention ใน Neural Machine Translation</li>
      <li>**2017:** Self-Attention ใน Transformer Architecture</li>
      <li>**2018+:** Multi-Head Attention, Cross-Modal Attention</li>
      <li>**ปัจจุบัน:** Universal Attention Mechanisms ใน Foundation Models</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> Self-Attention และ Multi-Head Attention ใน Transformer เป็นตัวอย่างสำคัญที่แสดงถึงพลังของ Attention ในการจับ dependencies ระยะไกล และปรับปรุงทั้งประสิทธิภาพและความสามารถในการตีความของโมเดล
    </div>

    <h3 className="text-xl font-semibold">บทบาทในงานประยุกต์สมัยใหม่</h3>
    <p>
      Attention Mechanism ปัจจุบันเป็นแกนกลางในสถาปัตยกรรม Deep Learning ชั้นนำเกือบทั้งหมด โดยเฉพาะโมเดลภาษาขนาดใหญ่ (Large Language Models), โมเดลด้านการแปลภาษาอัตโนมัติ, ระบบ Question Answering, Text Summarization, และแม้แต่ Vision Transformers (ViT) ในงานด้าน Computer Vision
    </p>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473.</li>
      <li>Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.</li>
      <li>Stanford CS224n Lecture Notes, 2023.</li>
      <li>Harvard NLP Group. (2020). The Annotated Transformer.</li>
      <li>Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Vision Transformer (ViT). arXiv:2010.11929.</li>
    </ul>

  </div>
</section>


          {/* Section 2 */}
     <section id="rnn-problems" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ปัญหาของ RNN แบบดั้งเดิม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">โครงสร้างพื้นฐานของ RNN</h3>
    <p>
      Recurrent Neural Networks (RNNs) ได้รับการออกแบบมาเพื่อจัดการกับข้อมูลลำดับ (sequential data) โดยนำ hidden state จาก timestep ก่อนหน้ามาใช้ในการประมวลผล timestep ถัดไป ส่งผลให้ RNNs สามารถเรียนรู้ dependencies ระหว่างข้อมูลในลำดับได้
    </p>
    <p>
      โมเดล RNN แบบดั้งเดิมมีโครงสร้าง recurrent loop ดังนี้:
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li><strong>h_t:</strong> hidden state ณ timestep t</li>
      <li><strong>x_t:</strong> input vector ณ timestep t</li>
      <li><strong>W_hh, W_xh:</strong> weight matrices</li>
      <li><strong>b_h:</strong> bias vector</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> แม้โครงสร้าง recurrent loop จะทำให้ RNN สามารถ modeling temporal dynamics ได้ แต่ก็สร้างข้อจำกัดเชิงโครงสร้างและประสิทธิภาพในการเรียนรู้ dependencies ระยะไกล
    </div>

    <h3 className="text-xl font-semibold">1. ปัญหา Vanishing Gradient</h3>
    <p>
      ปัญหาสำคัญที่สุดของ RNN แบบดั้งเดิมคือ vanishing gradient ซึ่งเกิดขึ้นเมื่อ gradient มีค่าลดลงอย่างรวดเร็วขณะย้อนกลับ (backpropagation through time, BPTT) ทำให้การอัปเดต weight สำหรับ timestep ที่ห่างจาก timestep ปัจจุบันไม่มีประสิทธิภาพ
    </p>
    <p>
      ผลกระทบของ vanishing gradient ได้แก่:
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>การเรียนรู้ dependencies ระยะสั้นได้ดี</li>
      <li>การเรียนรู้ dependencies ระยะยาวแทบเป็นไปไม่ได้</li>
      <li>ทำให้ contextual modeling มีขอบเขตจำกัด</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> งานวิจัยของ Bengio et al. (1994) ได้พิสูจน์ว่า RNN แบบดั้งเดิมมีความสามารถต่ำมากในการเรียนรู้ dependencies ระยะไกล และแนะนำเทคนิคต่าง ๆ เพื่อแก้ปัญหานี้
    </div>

    <h3 className="text-xl font-semibold">2. ปัญหา Exploding Gradient</h3>
    <p>
      นอกจาก vanishing gradient แล้ว RNN ยังประสบกับ exploding gradient ซึ่งทำให้ gradient มีค่ามากเกินไปในบาง timestep ส่งผลให้เกิด numerical instability และ loss function ไม่ converge
    </p>
    <p>
      วิธีแก้ปัญหานี้มักใช้ gradient clipping เพื่อจำกัดขนาด gradient
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`if ||gradient|| &gt; threshold:
    gradient = gradient * (threshold / ||gradient||)`}
    </pre>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> การใช้ gradient clipping ได้กลายเป็นแนวปฏิบัติมาตรฐานสำหรับการฝึก RNN เพื่อป้องกัน exploding gradient ที่ส่งผลต่อความเสถียรของการเรียนรู้
    </div>

    <h3 className="text-xl font-semibold">3. ปัญหา Sequential Computation</h3>
    <p>
      RNN ต้องประมวลผลข้อมูลแบบลำดับ (sequential) ซึ่งไม่สามารถขนาน (parallelize) ได้อย่างมีประสิทธิภาพ เนื่องจาก hidden state ที่ timestep t ขึ้นกับ hidden state ของ timestep t-1
    </p>

   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Feature</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Transformer</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Computation mode</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sequential</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Fully parallel</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training speed</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Slow</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Fast</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Scalability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">High</td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> ความสามารถในการ parallelize เป็นหนึ่งในข้อได้เปรียบหลักของ Transformer ซึ่งเอาชนะข้อจำกัดเชิงโครงสร้างของ RNN แบบดั้งเดิมได้อย่างมีประสิทธิภาพ
    </div>

    <h3 className="text-xl font-semibold">4. การเก็บบริบทระยะไกล (Long-Term Dependencies)</h3>
    <p>
      RNN แบบดั้งเดิมมีความสามารถจำกัดในการเก็บรักษาบริบทระยะไกล ทำให้ไม่สามารถ modeling ความสัมพันธ์ที่อยู่ห่างกันมากในลำดับได้ดี ซึ่งเป็นข้อจำกัดสำคัญเมื่อจัดการกับงาน NLP ที่ต้องอาศัยความเข้าใจบริบทโดยรวม
    </p>
    <p>
      ตัวอย่างเช่น ในการแปลประโยคยาว การรักษาความสัมพันธ์ระหว่างคำในต้นประโยคและท้ายประโยคเป็นเรื่องท้าทายสำหรับ RNN แบบดั้งเดิม
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> LSTM และ GRU ได้รับการพัฒนาเพื่อลดข้อจำกัดนี้ โดยเพิ่ม gating mechanisms แต่ Transformer ได้พลิกโฉมการ modeling long-term dependencies อย่างแท้จริง
    </div>

    <h3 className="text-xl font-semibold">5. ความซับซ้อนในการเรียนรู้โครงสร้างเชิงลำดับ</h3>
    <p>
      แม้ว่า RNN จะสามารถจับ dependencies แบบลำดับได้ แต่ยังมีข้อจำกัดในการเรียนรู้โครงสร้างเชิงไวยากรณ์ (syntactic structure) หรือ dependencies แบบ hierarchical ซึ่งสำคัญมากในภาษาธรรมชาติ
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ไม่สามารถ modeling dependencies ข้ามระดับได้ดี</li>
      <li>จำกัดที่การประมวลผลแบบ sequential-local</li>
      <li>ไม่รองรับ modeling แบบ tree-based อย่างมีประสิทธิภาพ</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> การใช้ self-attention ใน Transformer ช่วยให้โมเดลสามารถเรียนรู้ความสัมพันธ์เชิงลำดับและ hierarchical structure ได้พร้อมกันในลักษณะ parallel และ global
    </div>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks.</li>
      <li>Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.</li>
      <li>Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.</li>
      <li>Stanford CS224n Lecture Notes, 2023.</li>
      <li>MIT 6.S191 Deep Learning for Self-Driving Cars.</li>
    </ul>

  </div>
</section>


          {/* Section 3 */}
      <section id="attention-design" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. การออกแบบ Attention Mechanism</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">แนวคิดพื้นฐานของ Attention</h3>
    <p>
      Attention Mechanism ถือเป็นหนึ่งในความก้าวหน้าที่สำคัญที่สุดของสถาปัตยกรรม Deep Learning ในช่วงทศวรรษที่ผ่านมา โดยเริ่มได้รับความนิยมจากงาน Neural Machine Translation (Bahdanau et al., 2015) และต่อมาได้กลายเป็นรากฐานของโมเดลอย่าง Transformer (Vaswani et al., 2017) จุดเด่นของ Attention คือการอนุญาตให้โมเดลโฟกัสไปยังส่วนที่เกี่ยวข้องของข้อมูล input แบบ dynamic แทนการใช้ context vector เพียงตัวเดียวแบบ RNN
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> Attention ทำให้โมเดลสามารถ modeling ความสัมพันธ์ระหว่างข้อมูลทุกคู่ใน sequence ได้แบบ adaptive ซึ่งเป็นสิ่งที่โมเดลแบบ sequential อย่าง RNN ทำได้อย่างจำกัด
    </div>

    <h3 className="text-xl font-semibold">องค์ประกอบหลักของ Attention Mechanism</h3>
    <p>
      โครงสร้างพื้นฐานของ Attention Mechanism ประกอบด้วย 3 องค์ประกอบหลัก ได้แก่ Query (Q), Key (K), และ Value (V) โดยมีการคำนวณ Attention Score ระหว่าง Query และ Key เพื่อนำมาใช้ถ่วงน้ำหนัก Value
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) V`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li><strong>Query (Q):</strong> ตัวแทนของข้อมูลที่ต้องการโฟกัสใน timestep ปัจจุบัน</li>
      <li><strong>Key (K):</strong> ตัวแทนของข้อมูลในทุกตำแหน่งของ sequence</li>
      <li><strong>Value (V):</strong> ข้อมูลที่ถูกนำมา weighted sum เพื่อสร้าง output</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> การ scaling ด้วย <code>sqrt(d_k)</code> มีความสำคัญมาก เนื่องจากช่วยให้ softmax ทำงานได้อย่างมีเสถียรภาพมากขึ้นในกรณีที่ dimensionality ของ Key มีค่ามาก
    </div>

    <h3 className="text-xl font-semibold">ขั้นตอนการคำนวณ Attention</h3>
    <ol className="list-decimal list-inside ml-4 space-y-2">
      <li>คำนวณ dot product ระหว่าง Query และ Key</li>
      <li>ปรับขนาดโดยหารด้วย <code>sqrt(d_k)</code> เพื่อลด variance</li>
      <li>นำผลลัพธ์เข้าสู่ softmax เพื่อสร้าง attention weights</li>
      <li>นำ attention weights มาคูณกับ Value เพื่อสร้าง output</li>
    </ol>

    <h3 className="text-xl font-semibold">Self-Attention และ Cross-Attention</h3>
    <p>
      ในสถาปัตยกรรม Deep Learning ที่ทันสมัย มีการใช้ Attention ทั้งแบบ Self-Attention และ Cross-Attention:
    </p>

   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Q, K, V มาจาก</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ตัวอย่างการใช้งาน</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Input เดียวกัน</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Encoder / Decoder ของ Transformer</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Q จาก Decoder, K และ V จาก Encoder</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Decoder ของ Transformer ใน Machine Translation</td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> Self-Attention มีความสามารถ modeling dependencies แบบ global ได้ดีกว่า RNN และ CNN ซึ่งช่วยให้โมเดลอย่าง Transformer สามารถเข้าใจโครงสร้างข้อมูลแบบ long-range ได้อย่างมีประสิทธิภาพ
    </div>

    <h3 className="text-xl font-semibold">Multi-Head Attention</h3>
    <p>
      เพื่อเพิ่มความสามารถในการจับ pattern ที่หลากหลาย Multi-Head Attention ถูกออกแบบมาให้มีหลายชุดของ Q, K, V (เรียกว่า heads) ทำให้สามารถเรียนรู้ feature space ที่หลากหลายพร้อมกัน
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o`}
    </pre>

    <p>แต่ละ head คำนวณ Attention ตามสูตรเดียวกับ Attention ปกติ:</p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)`}
    </pre>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> Multi-Head Attention ช่วยให้โมเดลสามารถจับ patterns ที่หลากหลาย เช่น syntax, semantics, และ dependencies ข้ามลำดับ ในการเรียนรู้แบบ parallel
    </div>

    <h3 className="text-xl font-semibold">การประยุกต์ใช้งานจริง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Neural Machine Translation (NMT): Transformer-based NMT ใช้ Self-Attention + Cross-Attention เพื่อจับคู่คำข้ามภาษาได้อย่างยืดหยุ่น</li>
      <li>Image Recognition: Vision Transformer (ViT) ใช้ Self-Attention แทน CNN ในการจับ spatial dependencies</li>
      <li>Document Summarization: ใช้ Self-Attention เพื่อรวบรวมบริบทจากทั้ง document สำหรับการสรุปแบบ global</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานวิจัยล่าสุดแสดงให้เห็นว่า Attention Mechanism ไม่เพียงแต่สำคัญใน NLP เท่านั้น แต่ยังมีบทบาทสำคัญใน Vision, Speech, และ Multimodal Learning อย่างกว้างขวาง
    </div>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473</li>
      <li>Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS</li>
      <li>Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Vision Transformer. arXiv:2010.11929</li>
      <li>Stanford CS224n Lecture Notes, 2023</li>
      <li>MIT 6.S191 Deep Learning for Self-Driving Cars</li>
    </ul>

  </div>
</section>


          {/* Section 4 */}
      <section id="attention-types" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. ประเภทของ Attention Mechanisms</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      ตั้งแต่การนำเสนอแนวคิด Attention Mechanism ครั้งแรกใน Neural Machine Translation โดย Bahdanau et al. (2015) โลกของ Deep Learning ได้พัฒนา variant ของ Attention มากมาย เพื่อให้เหมาะสมกับลักษณะข้อมูลและงานที่แตกต่างกัน Section นี้จะนำเสนอประเภทที่สำคัญที่สุด พร้อมตัวอย่างและเปรียบเทียบระหว่างกัน
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> ในแต่ละโดเมน เช่น NLP, Computer Vision หรือ Multimodal Learning มีประเภทของ Attention ที่เหมาะสมต่างกัน การเลือกประเภทที่เหมาะสมมีผลอย่างมากต่อประสิทธิภาพของโมเดล
    </div>

    <h3 className="text-xl font-semibold">Self-Attention</h3>
    <p>
      Self-Attention เป็นประเภทที่สำคัญที่สุดในปัจจุบัน โดยใช้ Q, K, V ที่ได้จาก sequence เดียวกัน จุดเด่นคือสามารถเรียนรู้ dependencies ใน sequence แบบ global ซึ่ง RNN ทำได้จำกัด
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>ใช้ใน Transformer Encoder / Decoder</li>
      <li>ใช้ใน Vision Transformer (ViT) แทน Convolution</li>
      <li>ช่วย model ความสัมพันธ์ระยะไกลใน sequence</li>
    </ul>

    <h3 className="text-xl font-semibold">Cross-Attention</h3>
    <p>
      Cross-Attention ใช้ Q มาจากชุดข้อมูลหนึ่ง (เช่น output ของ Decoder layer ก่อนหน้า) และ K, V มาจากอีกชุด (เช่น Encoder output) ซึ่งช่วยให้การแลกเปลี่ยนข้อมูลระหว่างโมดูลทำได้ยืดหยุ่น
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>ใช้ใน Transformer Decoder</li>
      <li>ใช้ใน Multimodal models เช่น CLIP, Flamingo, LLaVA</li>
      <li>ช่วย Align ข้อมูลจาก domain ต่าง ๆ เช่น Text ↔ Image</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> Cross-Attention เป็นหัวใจสำคัญที่ทำให้ Multimodal AI สามารถประมวลผลข้อมูลต่างชนิดร่วมกันได้อย่างยืดหยุ่น
    </div>

    <h3 className="text-xl font-semibold">Additive Attention (Bahdanau Attention)</h3>
    <p>
      Additive Attention หรือ Bahdanau Attention เป็นรูปแบบดั้งเดิมที่ไม่ได้ใช้ dot-product แต่ใช้ feedforward network เพื่อคำนวณ score ซึ่งทำให้สามารถเรียนรู้ attention weights ได้แม่นยำแม้ในกรณีที่ dimension ของ Q, K แตกต่างกัน
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`score(Q, K) = v^T tanh( W_1 Q + W_2 K )`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li>ต้นกำเนิดในงาน NMT ของ Bahdanau et al. (2015)</li>
      <li>ยังคงใช้ในบาง NMT models ที่ต้องการ flexibility สูง</li>
    </ul>

    <h3 className="text-xl font-semibold">Dot-Product Attention (Luong Attention)</h3>
    <p>
      Dot-Product Attention หรือ Luong Attention (Luong et al., 2015) ใช้ dot product ตรงระหว่าง Q และ K ทำให้คำนวณได้รวดเร็วและขนานได้ดี โดยมีการ scaling เพื่อเสถียรภาพ
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`score(Q, K) = Q K^T / sqrt(d_k)`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li>ใช้ใน Transformer และ model NLP ส่วนใหญ่</li>
      <li>เป็นมาตรฐานในงาน modern NLP และ Vision</li>
    </ul>

    <h3 className="text-xl font-semibold">Global Attention vs Local Attention</h3>
    <p>
      ขอบเขตของ Attention สามารถแบ่งได้เป็น Global และ Local:
    </p>

   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อดี</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อจำกัด</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Global Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ dependency แบบ long-range ได้เต็มรูปแบบ</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Complexity O(n²), ใช้หน่วยความจำสูง</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Local Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Complexity ต่ำ, เหมาะกับ sequence ยาวมาก</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">มองเห็น context ใน window ขนาดจำกัด</td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานวิจัยเช่น Longformer (Beltagy et al., 2020) และ BigBird (Zaheer et al., 2020) แสดงให้เห็นว่า Local + Sparse Attention เป็นทางเลือกที่มีศักยภาพสำหรับการประมวลผลเอกสารยาว
    </div>

    <h3 className="text-xl font-semibold">Sparse Attention</h3>
    <p>
      Sparse Attention เป็นเทคนิคที่พยายามลด complexity ของ Attention จาก O(n²) ลง โดย mask การคำนวณบางส่วนของ Attention map:
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>ใช้ใน Longformer, BigBird, Reformer</li>
      <li>ลด memory footprint สำหรับ long-sequence processing</li>
      <li>สามารถ pretrain บน document ยาวได้ดีกว่า Transformer ปกติ</li>
    </ul>

    <h3 className="text-xl font-semibold">Multi-Scale Attention</h3>
    <p>
      Multi-Scale Attention ใช้ Attention ในหลาย scale ของข้อมูล เช่น spatial scale ใน Vision หรือ temporal scale ใน Speech ซึ่งช่วยให้โมเดลสามารถจับ pattern ได้หลายระดับ
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>ใช้ใน Vision Transformer variants เช่น Swin Transformer</li>
      <li>ช่วยเรียนรู้ hierarchical representation</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> Multi-Scale Attention เป็นแนวโน้มสำคัญใน Computer Vision โดย Swin Transformer และ MetaFormer แสดงให้เห็นว่าการรวม Attention หลาย scale ให้ผลที่ดีกว่าแบบ global อย่างเดียว
    </div>

    <h3 className="text-xl font-semibold">สรุปภาพรวมประเภทของ Attention</h3>
   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">โดเมนที่นิยมใช้</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อเด่น</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">NLP, Vision, Speech</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Global dependencies, Parallelism</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">NLP, Multimodal</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-domain alignment</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Additive Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">NMT (Legacy)</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Flexibility, interpretability</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Dot-Product Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer, ViT</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Speed, scalability</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sparse / Local Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Long-sequence processing</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Memory efficiency</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multi-Scale Attention</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Vision</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Hierarchical feature learning</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Bahdanau, D., et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473</li>
      <li>Luong, M.-T., et al. (2015). Effective Approaches to Attention-based NMT. arXiv:1508.04025</li>
      <li>Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS</li>
      <li>Beltagy, I., et al. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150</li>
      <li>Zaheer, M., et al. (2020). Big Bird: Transformers for Longer Sequences. arXiv:2007.14062</li>
      <li>Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer. arXiv:2103.14030</li>
      <li>Stanford CS224n, MIT 6.S191, Oxford Deep Learning Lecture Series</li>
    </ul>

  </div>
</section>


          {/* Section 5 */}
   <section id="transformer-revolution" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Transformer Architecture → Revolution ด้วย Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      การนำเสนอสถาปัตยกรรม Transformer ในงาน "Attention Is All You Need" โดย Vaswani et al. (2017) ถือเป็นจุดเปลี่ยนครั้งสำคัญของวงการ Deep Learning โดยเฉพาะในด้าน Natural Language Processing (NLP) ซึ่งเดิมอาศัย RNN และ LSTM เป็นหลัก Transformer ได้แสดงให้เห็นว่าสามารถจัดการกับ long-range dependencies ได้ดีกว่าเดิมมาก ทั้งยังมีประสิทธิภาพในการฝึกแบบขนานสูง
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> การตัด Recurrent connections ออกจากสถาปัตยกรรม ทำให้ Transformer สามารถ leverage GPU/TPU ได้เต็มประสิทธิภาพ ซึ่งส่งผลให้สามารถฝึกโมเดลขนาดใหญ่ขึ้นกว่าที่เคยเป็นมา
    </div>

    <h3 className="text-xl font-semibold">องค์ประกอบหลักของ Transformer</h3>
    <p>
      Transformer แบ่งออกเป็นสองส่วนหลัก: Encoder และ Decoder โดยทั้งสองส่วนประกอบด้วย stack ของ layers ที่มีโครงสร้างคล้ายกัน แต่มีรายละเอียดที่แตกต่างในบางจุด
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Multi-Head Self-Attention</li>
      <li>Position-wise Feedforward Networks</li>
      <li>Residual Connections และ Layer Normalization</li>
      <li>Positional Encoding เพื่อรักษาลำดับข้อมูล</li>
    </ul>

    <h3 className="text-xl font-semibold">สถาปัตยกรรมโดยรวม</h3>
    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`Input Embedding + Positional Encoding
      ↓
[Multi-Head Attention + Add & Norm]
      ↓
[Feed Forward + Add & Norm]
      ↓
Output Representation (Encoder)`}
    </pre>

    <h3 className="text-xl font-semibold">Multi-Head Attention</h3>
    <p>
      Multi-Head Attention เป็นหัวใจสำคัญของ Transformer โดยมีหลาย "head" ที่เรียนรู้ representation ที่แตกต่างกันจาก input sequence แบบขนาน ซึ่งทำให้ model สามารถจับ pattern ได้หลากหลายและลึกซึ้ง
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li>แต่ละ head เรียนรู้ feature space ที่ต่างกัน</li>
      <li>ช่วยให้ model มีความสามารถในการ generalize ได้ดีขึ้น</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> Multi-Head Attention ไม่เพียงแต่ช่วยให้ model จับ dependencies ได้หลายระดับ แต่ยังช่วยให้เกิด interpretability ที่ดีขึ้นในงานวิจัยด้าน Explainable AI (XAI)
    </div>

    <h3 className="text-xl font-semibold">Position-wise Feedforward Networks</h3>
    <p>
      หลังจาก Attention Layer ข้อมูลจะถูกส่งต่อไปยัง Feedforward Network (FFN) แบบ position-wise ซึ่งทำให้ model สามารถเพิ่ม non-linearity และความสามารถในการประมวลผล complex pattern ได้
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`FFN(x) = max(0, x W1 + b1) W2 + b2`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li>ใช้ activation function เช่น ReLU หรือ GELU</li>
      <li>ช่วยให้ model สามารถเรียนรู้ representation ที่ซับซ้อนขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">Positional Encoding</h3>
    <p>
      เนื่องจาก Transformer ไม่มี Recurrence จึงต้องใช้ Positional Encoding เพื่อ encode ข้อมูลเกี่ยวกับลำดับของ token ต่าง ๆ ใน sequence
    </p>

    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`PE(pos, 2i) = sin(pos / 70000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 70000^(2i/d_model))`}
    </pre>

    <ul className="list-disc list-inside ml-4">
      <li>ช่วย model เข้าใจลำดับเชิงเวลา/เชิงตำแหน่ง</li>
      <li>ถูกนำไปใช้ในงานต่าง ๆ เช่น Time Series Forecasting</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับสถาปัตยกรรมแบบเดิม</h3>
   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คุณสมบัติ</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN / LSTM</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Transformer</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelism</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Long-range Dependency</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีเยี่ยม</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Time</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ช้า</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เร็ว</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Scalability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีเยี่ยม</td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> ข้อมูลจาก Google Brain และ Stanford CS224n ยืนยันว่า Transformer สามารถแทนที่ RNN/LSTM ได้เกือบทั้งหมดในงาน NLP ปัจจุบัน และกำลังขยายไปยัง Vision, Speech และ Multimodal AI
    </div>

    <h3 className="text-xl font-semibold">ผลกระทบเชิงอุตสาหกรรม</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Transformer เป็นพื้นฐานของ BERT, GPT, T5, PaLM และโมเดล Multimodal รุ่นใหม่</li>
      <li>เป็น architecture หลักที่ใช้ใน production systems ของ Google Translate, Facebook AI, OpenAI และ Microsoft</li>
      <li>ช่วยให้เกิด AI ที่สามารถประมวลผลข้อมูลขนาดใหญ่มากในเวลาจำกัด</li>
    </ul>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.</li>
      <li>Al-Rfou, R. et al. (2019). Character-Level Language Modeling with Transformer. arXiv:1909.03427</li>
      <li>Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.</li>
      <li>Brown, T. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165 (GPT-3).</li>
      <li>Stanford CS224n Lecture Notes, 2023.</li>
      <li>MIT 6.S191 Deep Learning Lecture Series.</li>
      <li>IEEE Transactions on Neural Networks and Learning Systems.</li>
    </ul>

  </div>
</section>


          {/* Section 6 */}
     <section id="attention-use-cases" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Use Cases ของ Attention ใน Deep Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      Attention Mechanism ได้กลายเป็นส่วนประกอบหลักในหลายงานประยุกต์ของ Deep Learning ซึ่งมีความสามารถในการจัดการข้อมูลที่มีลักษณะลำดับ และสร้างการเชื่อมโยงระหว่างส่วนต่าง ๆ ของข้อมูลได้อย่างยืดหยุ่น ในช่วงไม่กี่ปีที่ผ่านมา โมเดลที่ใช้ Attention เป็นแกนหลัก เช่น Transformer ได้แสดงให้เห็นถึงประสิทธิภาพที่สูงกว่าวิธีการแบบดั้งเดิมในหลากหลายสาขา ทั้งใน Natural Language Processing (NLP), Computer Vision (CV), Speech Processing, และ Multimodal Learning
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานของ Google Brain, OpenAI และ Meta AI แสดงให้เห็นว่า Attention Mechanism ไม่ใช่เพียงนวัตกรรมเฉพาะด้าน NLP อีกต่อไป แต่ได้กลายเป็น core building block ของ Deep Learning ในระดับข้ามสาขา
    </div>

    <h3 className="text-xl font-semibold">Natural Language Processing (NLP)</h3>
    <p>
      ในงานด้าน NLP, Attention ได้ปฏิวัติการแปลภาษาอัตโนมัติ, การสรุปข้อความ, และการทำ Question Answering อย่างสิ้นเชิง โดยช่วยให้โมเดลสามารถจับ dependencies ที่มีระยะห่างกันในลำดับข้อความได้อย่างแม่นยำ
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Machine Translation: Transformer-based NMT เช่น Google Translate</li>
      <li>Summarization: BART, PEGASUS ใช้ Attention ในการจัดเรียงบริบทที่สำคัญ</li>
      <li>Question Answering: BERT, T5 ใช้ Self-Attention เพื่อ encode บริบทอย่างลึกซึ้ง</li>
    </ul>

    <h3 className="text-xl font-semibold">Computer Vision (CV)</h3>
    <p>
      Attention Mechanism ได้เริ่มถูกนำมาใช้ในงาน Computer Vision อย่างแพร่หลาย โดยเฉพาะในสถาปัตยกรรม Vision Transformer (ViT) ที่สามารถแทน CNN แบบดั้งเดิมในหลายงานได้อย่างมีประสิทธิภาพ
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Image Classification: ViT, DeiT ใช้ Self-Attention แทน convolution layers</li>
      <li>Object Detection: DETR ใช้ Encoder-Decoder Attention สำหรับ object localization</li>
      <li>Image Generation: GANs + Attention เพิ่มความแม่นยำในการสร้างภาพ</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> งานวิจัยจาก Google Research (2020) แสดงให้เห็นว่า Vision Transformer (ViT) ที่ฝึกบน dataset ขนาดใหญ่ เช่น JFT-300M สามารถ outperform CNN ที่ state-of-the-art ในงาน ImageNet ได้สำเร็จ
    </div>

    <h3 className="text-xl font-semibold">Speech Processing</h3>
    <p>
      ในด้าน Speech Recognition และ Speech Synthesis, Attention Mechanism ได้ช่วยปรับปรุงประสิทธิภาพของโมเดลอย่างมาก โดยเฉพาะในงาน Sequence-to-Sequence ที่ต้องประมวลผลสัญญาณเสียงที่มีความซับซ้อนสูง
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Speech Recognition: Transformer-based ASR เช่น Speech-Transformer, Conformer</li>
      <li>Speech Synthesis: Tacotron 2 ใช้ Attention เพื่อต่อเชื่อมระหว่าง phoneme กับ acoustic features</li>
    </ul>

    <h3 className="text-xl font-semibold">Multimodal Learning</h3>
    <p>
      Attention เป็นเครื่องมือสำคัญในการเชื่อมโยงข้อมูลจากหลาย modality เช่น ข้อความ + ภาพ หรือ ข้อความ + วิดีโอ โดยสามารถใช้ Cross-Attention ในการ align ข้อมูลจากแหล่งต่าง ๆ อย่างมีประสิทธิภาพ
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Vision-Language Models: CLIP ใช้ Cross-Attention ระหว่างภาพและคำบรรยาย</li>
      <li>Video-Language Models: Flamingo จาก DeepMind ใช้ Attention ข้าม video frames และ captions</li>
      <li>Multimodal QA: Unified-IO, OFA ใช้ Attention ในการ integrate modality ต่าง ๆ เข้าด้วยกัน</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานจาก Stanford CS224N และ MIT-IBM Watson AI Lab ยืนยันว่า Cross-Attention เป็นกุญแจสำคัญในการสร้าง multimodal models ที่สามารถเข้าใจความสัมพันธ์ระหว่าง modality ต่าง ๆ ได้ลึกซึ้ง
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบ Use Cases หลัก</h3>
   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Use Case</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">โมเดลหลัก</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">รูปแบบ Attention</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Machine Translation</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention, Encoder-Decoder Attention</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Image Classification</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ViT, DeiT</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Self-Attention</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Speech Synthesis</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Tacotron 2</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Attention Mechanism (Alignment Layer)</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multimodal Retrieval</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-Attention</td>
      </tr>
    </tbody>
  </table>
</div>



    <h3 className="text-xl font-semibold">แนวโน้มในอนาคต</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Self-Supervised Pretraining ใช้ Attention เป็นแกนหลักสำหรับ learning จาก unlabeled data</li>
      <li>Large Multimodal Models เช่น GPT-4o, Gemini ใช้ Cross-Attention แบบลึกเพื่อ integrate modality</li>
      <li>Efficient Attention Mechanisms เช่น Linformer, Performer กำลังถูกพัฒนาเพื่อลด quadratic complexity</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> การขยายขนาดของ Attention-based models ในงาน Multimodal Learning กำลังเป็น frontier ใหม่ที่ผลักดันความก้าวหน้าของ AI แบบ Generalist models
    </div>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention is All You Need. NeurIPS.</li>
      <li>Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv.</li>
      <li>Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML (CLIP).</li>
      <li>Popel, M. et al. (2020). Transformer-based Speech Recognition. arXiv.</li>
      <li>Stanford CS224n: Natural Language Processing with Deep Learning.</li>
      <li>MIT 6.S191 Deep Learning Lecture Series.</li>
      <li>DeepMind. Flamingo: A Visual Language Model for Few-Shot Learning. arXiv.</li>
    </ul>

  </div>
</section>


          {/* Section 7 */}
     <section id="attention-advantages" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. ข้อดีของ Attention Mechanism</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      Attention Mechanism ได้กลายเป็นเครื่องมือหลักในสถาปัตยกรรม Deep Learning สมัยใหม่ โดยมีข้อได้เปรียบหลายประการเหนือกว่าโมเดลลำดับแบบดั้งเดิม เช่น RNN หรือ CNN ข้อดีเหล่านี้เป็นปัจจัยสำคัญที่ทำให้ Attention ถูกนำไปใช้อย่างแพร่หลายทั้งใน Natural Language Processing (NLP), Computer Vision (CV), Speech Processing, และ Multimodal AI
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> ผลการวิจัยจาก Stanford CS224n และ MIT-IBM Watson Lab ชี้ว่า Attention Mechanism ได้เปลี่ยน paradigm ของ sequence modeling จาก sequential processing → ไปสู่ parallel processing อย่างมีประสิทธิภาพ
    </div>

    <h3 className="text-xl font-semibold">ความสามารถในการโฟกัสแบบ Dynamic</h3>
    <p>
      หนึ่งในข้อได้เปรียบที่สำคัญที่สุดของ Attention คือความสามารถในการโฟกัสแบบ dynamic กล่าวคือ Attention Layer สามารถเรียนรู้ที่จะจัดสรร "ความสำคัญ" ให้กับส่วนต่าง ๆ ของ input sequence ได้แบบ adaptive ตามบริบท
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>ไม่จำเป็นต้องใช้ fixed-size context vector เช่นใน RNN</li>
      <li>สามารถ focus บริเวณที่สำคัญได้แม้ลำดับจะยาว</li>
      <li>ช่วยให้ model generalize ได้ดีขึ้นใน task ที่ซับซ้อน</li>
    </ul>

    <h3 className="text-xl font-semibold">Parallelization และ Scalability</h3>
    <p>
      ในขณะที่ RNN ต้องประมวลผลข้อมูลแบบ sequential ซึ่งทำให้ training time เพิ่มขึ้นแบบ linear ตาม sequence length, Attention สามารถประมวลผลทุกตำแหน่งใน sequence พร้อมกันได้ (parallelization)
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Self-Attention สามารถ implement ได้ด้วย matrix multiplication → fully parallel</li>
      <li>สามารถใช้ hardware accelerator (GPU/TPU) ได้อย่างมีประสิทธิภาพ</li>
      <li>Training time ต่อ epoch ต่ำกว่ามากเมื่อเทียบกับ RNN / LSTM</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> Paper "Attention Is All You Need" ของ Vaswani et al. (2017) แสดงให้เห็นว่า Transformer สามารถ train ได้เร็วกว่า RNN-based Seq2Seq มากกว่า 10 เท่า บน hardware เดียวกัน
    </div>

    <h3 className="text-xl font-semibold">การจัดการกับ Long-Range Dependencies</h3>
    <p>
      หนึ่งในปัญหาหลักของ RNN-based models คือ vanishing gradients เมื่อพยายามเรียนรู้ long-range dependencies ใน sequence ที่ยาวมาก Attention แก้ปัญหานี้ได้อย่างมีประสิทธิภาพ
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Attention สามารถสร้าง direct path ระหว่างทุกตำแหน่งใน sequence</li>
      <li>ไม่เกิด vanishing gradient เพราะไม่ต้อง propagate ผ่าน hidden states</li>
      <li>ช่วยให้ model สามารถจับ dependencies ที่ห่างกันมากได้ดีขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">Flexibility และ Interpretability</h3>
    <p>
      อีกหนึ่งข้อได้เปรียบสำคัญคือความยืดหยุ่นของ Attention Mechanism ที่สามารถนำไปใช้ใน task ที่หลากหลาย ทั้ง sequence-to-sequence, sequence-to-one, sequence-to-multi-modality รวมถึงความสามารถในการตีความ (interpretability)
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>สามารถ visualize Attention weights เพื่อเข้าใจ decision process ของ model</li>
      <li>ง่ายต่อการ debug model behavior ใน task ที่ซับซ้อน</li>
      <li>สนับสนุนงาน research ด้าน model interpretability</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบข้อได้เปรียบหลักของ Attention กับ RNN และ CNN</h3>
    <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คุณสมบัติ</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Attention Mechanism</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN / LSTM</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">CNN</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelization</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง (Fully Parallel)</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ (Sequential)</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Long-Range Dependency</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีเยี่ยม</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด (Vanishing Gradient)</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">จำกัด (Kernel Size)</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Interpretability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดี (ผ่าน Attention weights)</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ปานกลาง</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Scalability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดีเยี่ยม</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ดี</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">การประยุกต์ใช้งานจริง</h3>
    <p>ข้อได้เปรียบของ Attention ทำให้ถูกนำไปใช้ใน:</p>

    <ul className="list-disc list-inside ml-4">
      <li>Natural Language Processing: BERT, GPT, T5</li>
      <li>Computer Vision: Vision Transformer (ViT), DETR</li>
      <li>Speech Processing: Transformer-based ASR, Tacotron 2</li>
      <li>Multimodal Learning: CLIP, Flamingo, PaLM-E</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> Meta AI และ Google Research รายงานว่าทั้ง Multimodal models และ Foundation models ส่วนใหญ่ในยุค 2024-2025 ใช้ Attention Mechanism เป็นแกนกลางแทบทุกตัว
    </div>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบที่สำคัญที่สุด (สรุป)</h3>
    <ul className="list-disc list-inside ml-4">
      <li>เรียนรู้ dependencies ได้ทั่วทั้ง sequence</li>
      <li>ประมวลผลแบบ parallel → training เร็วขึ้นมาก</li>
      <li>ตีความการทำงานของ model ได้ (interpretable)</li>
      <li>ใช้งานได้ในทุก domain (NLP, CV, Speech, Multimodal)</li>
      <li>รองรับ scaling สู่ Foundation models ขนาดใหญ่</li>
    </ul>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention is All You Need. NeurIPS.</li>
      <li>Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv.</li>
      <li>Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML (CLIP).</li>
      <li>DeepMind. Flamingo: A Visual Language Model for Few-Shot Learning. arXiv.</li>
      <li>Stanford CS224n: Natural Language Processing with Deep Learning.</li>
      <li>MIT 6.S191: Deep Learning for Self-Driving Cars.</li>
      <li>Meta AI Research. (2024). Scaling Multimodal Foundation Models. arXiv.</li>
    </ul>

  </div>
</section>

          {/* Section 8 */}
      <section id="attention-limitations" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Limitations และ Challenge</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      แม้ Attention Mechanism จะมีข้อได้เปรียบอย่างมากในด้านการประมวลผลข้อมูลลำดับและ multimodal data แต่การนำไปใช้ในระบบจริงยังมีข้อจำกัดและความท้าทายที่ต้องพิจารณาอย่างรอบคอบ งานวิจัยล่าสุดจาก Stanford, MIT, CMU และ Oxford ได้ระบุประเด็นเชิงลึกที่สำคัญเกี่ยวกับประสิทธิภาพ, scaling, และความเข้าใจเชิงโครงสร้างของ Attention-based models
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานของ Tay et al. (2023, Google Research) ชี้ให้เห็นว่าการ scale Attention Mechanism ไปสู่ billion-scale models ต้องใช้ optimization ที่ซับซ้อนเพื่อหลีกเลี่ยงปัญหา computational bottleneck และ memory overhead
    </div>

    <h3 className="text-xl font-semibold">1. Computational Cost</h3>
    <p>
      ข้อจำกัดที่สำคัญของ Attention คือค่าใช้จ่ายในการประมวลผลที่สูง โดยเฉพาะใน Self-Attention ซึ่งมีความซับซ้อนเชิงเวลาเป็น O(n²) ต่อ sequence length n
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Matrix multiplication Q×K<sup>T</sup> ต้องใช้ O(n²) space → memory usage สูง</li>
      <li>การ scale ไปยัง long sequence (เช่น full document, video) ต้องใช้ optimization เช่น Sparse Attention หรือ Linearized Attention</li>
      <li>Training cost สูงมากสำหรับ large-scale Transformer</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> งานของ Child et al. (2019) ใน Sparse Transformer ลดความซับซ้อนเป็น O(n log n) แต่ยังมี trade-off ด้าน accuracy ในบาง task
    </div>

    <h3 className="text-xl font-semibold">2. Data Efficiency และ Pretraining Cost</h3>
    <p>
      การ train Attention-based models ให้ได้ประสิทธิภาพสูงต้องใช้ dataset ขนาดใหญ่มาก ซึ่งอาจไม่เหมาะกับทุกองค์กรหรือ domain ที่มีข้อมูลจำกัด
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Pretraining Transformer เช่น BERT ต้องใช้ corpus ขนาดหลาย TB</li>
      <li>Training time บน TPU cluster อาจกินเวลาหลายสัปดาห์</li>
      <li>ใน domain เฉพาะ (เช่น biomedical, legal) ข้อมูลไม่เพียงพอ อาจทำให้ model overfit หรือ generalize ได้ไม่ดี</li>
    </ul>

    <h3 className="text-xl font-semibold">3. Lack of Inductive Bias</h3>
    <p>
      Unlike CNNs ที่มี built-in bias ด้าน spatial locality หรือ RNNs ที่มี sequential bias, Attention Mechanism ไม่มี inductive bias โดยธรรมชาติ ซึ่งอาจทำให้ต้องการข้อมูลฝึกมากขึ้นเพื่อเรียนรู้โครงสร้างพื้นฐาน
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Vision Transformer (ViT) ต้อง pretrain บน dataset ขนาดใหญ่เพื่อ outperform CNN</li>
      <li>ใน domain ที่มีโครงสร้างชัดเจน (e.g. time series, graph) ต้องออกแบบ hybrid architecture เพื่อเสริม inductive bias</li>
      <li>Random initialization → slower convergence compared to inductively biased models</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานจาก Google Brain (Steiner et al., 2022) พบว่า ViT ต้องใช้ JFT-300M+ dataset เพื่อ outperform ResNet บน ImageNet, แสดงถึง cost ของการขาด inductive bias
    </div>

    <h3 className="text-xl font-semibold">4. Interpretability ที่ยังจำกัด</h3>
    <p>
      แม้ว่าจะสามารถ visualize Attention weights ได้ แต่ recent studies (Wiegreffe & Pinter, 2019) แสดงให้เห็นว่า Attention weights อาจไม่สื่อถึง causal reasoning ที่แท้จริงเสมอไป
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Attention weight ≠ model explanation → อาจ misleading</li>
      <li>ต้องใช้การวิเคราะห์เพิ่มเติม เช่น Integrated Gradients, LRP</li>
      <li>ความเข้าใจเชิง causal dependency ภายใน Transformer ยังเป็น active research area</li>
    </ul>

    <h3 className="text-xl font-semibold">5. Memory Constraints ใน Inference</h3>
    <p>
      ใน deployment จริง เช่น Edge device หรือ Mobile, Memory constraint เป็นข้อจำกัดใหญ่ในการใช้ full attention
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Full Transformer → memory footprint สูงเกินไปสำหรับ real-time inference</li>
      <li>ต้องใช้ optimization เช่น Distillation, Pruning, Quantization</li>
      <li>Research เช่น MobileBERT, TinyBERT มุ่งเน้นลด memory footprint</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> งานจาก Google Research พบว่า MobileBERT ใช้ memory เพียง 10% ของ BERT-base ในขณะที่ maintain ~97% performance บน GLUE benchmark
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบข้อจำกัดหลักของ Attention Mechanism</h3>
   <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Limitation</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Impact</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Computational Cost</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">O(n²) complexity</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Slow training on long sequences</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Data Efficiency</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Large-scale data required</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">High pretraining cost</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Inductive Bias</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">None by default</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Requires large data to learn structure</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Interpretability</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Attention weight ≠ true explanation</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited transparency</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Memory in Inference</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">High memory requirement</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Difficult deployment on edge devices</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.</li>
      <li>Child, R. et al. (2019). Generating Long Sequences with Sparse Transformers. arXiv:1904.10509.</li>
      <li>Steiner, A. et al. (2022). How Well Do Vision Transformers Generalize? arXiv:2201.03529.</li>
      <li>Tay, Y. et al. (2023). Efficient Transformers: A Survey. arXiv:2009.06732.</li>
      <li>Wiegreffe, S., & Pinter, Y. (2019). Attention is not Explanation. EMNLP-IJCNLP 2019.</li>
      <li>Google AI. MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. arXiv:2004.02984.</li>
      <li>MIT 6.S191 Deep Learning Lecture Notes 2023.</li>
      <li>Stanford CS224n Lecture Notes 2023.</li>
    </ul>

  </div>
</section>


          {/* Section 9 */}
        <section id="attention-research" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Research Benchmarks & State-of-the-art Papers</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      ในช่วงไม่กี่ปีที่ผ่านมา Attention Mechanism ได้กลายเป็นแกนหลักของงานวิจัยด้าน Deep Learning โดยเฉพาะในงาน Natural Language Processing (NLP), Computer Vision (CV), และ Multimodal AI การพัฒนา benchmark ที่มีมาตรฐานและ open dataset เป็นตัวผลักดันสำคัญที่ทำให้เกิดความก้าวหน้าอย่างรวดเร็วในโมเดลที่ใช้ Attention-based architecture
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> งานจาก Stanford NLP และ Google Research ระบุว่าการออกแบบ benchmark ที่สะท้อน real-world task complexity เป็นปัจจัยสำคัญที่ทำให้ Transformer-based models เช่น BERT และ T5 ก้าวสู่ state-of-the-art ในหลาย task
    </div>

    <h3 className="text-xl font-semibold">1. Benchmarks หลักสำหรับ NLP</h3>
    <p>
      Benchmarks สำหรับ NLP เป็นจุดเริ่มต้นที่สำคัญที่สุดของความก้าวหน้าในงานวิจัย Attention Mechanism โมเดลเช่น Transformer, BERT, GPT, และ T5 ได้รับการประเมินและขับเคลื่อนโดยชุด benchmark เหล่านี้:
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>
        <strong>GLUE (General Language Understanding Evaluation):</strong> รวม task หลายด้าน เช่น textual entailment, sentiment analysis, sentence similarity
      </li>
      <li>
        <strong>SQuAD (Stanford Question Answering Dataset):</strong> ประเมินความสามารถในการทำ machine reading comprehension
      </li>
      <li>
        <strong>SuperGLUE:</strong> เวอร์ชันยากกว่า GLUE ใช้ push ความสามารถ reasoning ของ models
      </li>
      <li>
        <strong>WMT (Workshop on Machine Translation):</strong> Benchmark มาตรฐานสำหรับการแปลภาษา
      </li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> T5 ของ Google เป็นโมเดลแรกที่ได้ score เกิน human baseline บน SuperGLUE ซึ่งสะท้อนพลังของ full Transformer encoder-decoder architecture ที่ใช้ Attention เป็นหลัก
    </div>

    <h3 className="text-xl font-semibold">2. Benchmarks สำหรับ Computer Vision</h3>
    <p>
      แม้ Attention จะเริ่มใน NLP แต่ได้ขยายเข้าสู่งาน Vision อย่างรวดเร็ว โดย ViT (Vision Transformer) และ hybrid architecture ต่าง ๆ ได้ถูกประเมินผ่าน benchmarks ชั้นนำของวงการ:
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>
        <strong>ImageNet:</strong> การจำแนกภาพ 1,000 classes เป็น benchmark มาตรฐาน
      </li>
      <li>
        <strong>COCO (Common Objects in Context):</strong> สำหรับ object detection, instance segmentation, captioning
      </li>
      <li>
        <strong>OpenImages:</strong> Dataset ขนาดใหญ่สำหรับ multi-label classification และ object detection
      </li>
    </ul>

    <h3 className="text-xl font-semibold">3. Benchmarks สำหรับ Multimodal & Cross-domain AI</h3>
    <p>
      งานวิจัยล่าสุดได้รวม Attention เข้ากับ multimodal learning เพื่อจัดการข้อมูลที่มีหลาย modality เช่น ภาพ + ข้อความ หรือ วิดีโอ + เสียง
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>
        <strong>VQA (Visual Question Answering):</strong> โมเดลต้องตอบคำถามเกี่ยวกับภาพ
      </li>
      <li>
        <strong>Flickr30K & MS COCO Captioning:</strong> การจับคู่ภาพกับ caption แบบ natural language
      </li>
      <li>
        <strong>HowTo700M, YouCook2:</strong> วิดีโอ + ข้อความ → ใช้ Self-attention และ cross-attention เป็นแกนหลัก
      </li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> DALL·E และ Flamingo ของ DeepMind เป็นตัวอย่างของ state-of-the-art multimodal models ที่ใช้ Attention mechanism เต็มรูปแบบในการ align modality ต่าง ๆ
    </div>

    <h3 className="text-xl font-semibold">4. ตัวอย่างงานวิจัยสำคัญที่ผลักดัน Attention</h3>
    <p>รายการ paper ต่อไปนี้ถือเป็น landmark ของวงการ ที่นิยมนำมาอ้างอิง และเป็นแกนของ SOTA research:</p>

    <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Paper</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Contribution</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Attention Is All You Need</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เสนอ Transformer architecture ที่ใช้ Self-Attention แทน RNN/CNN</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">BERT: Pre-training of Deep Bidirectional Transformers</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ใช้ masked language modeling + bidirectional attention เพื่อ representation learning</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ViT: An Image is Worth 16x16 Words</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">แสดงให้เห็นว่า pure Transformer สามารถ outperform CNN บน ImageNet</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">T5: Exploring the Limits of Transfer Learning</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Unified text-to-text framework สำหรับ NLP tasks โดยใช้ Transformer encoder-decoder</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Perceiver</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Architecture ที่ใช้ Attention กับ arbitrary modality input เช่น image, video, audio</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">5. แนวโน้มการพัฒนา Benchmark และ Evaluation</h3>
    <p>
      การประเมิน Attention-based models กำลังก้าวไปสู่ benchmark ที่ซับซ้อนมากขึ้น:
    </p>

    <ul className="list-disc list-inside ml-4">
      <li>Meta-Evaluation: เช่น HELM ของ Stanford ที่ประเมิน cross-domain performance</li>
      <li>Fairness & Robustness Benchmarks: e.g., Datasets for testing bias and adversarial robustness</li>
      <li>Efficiency Benchmarks: e.g., Energy efficiency per token or per image</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> HELM (Holistic Evaluation of Language Models) ของ Stanford กำลังเป็น benchmark มาตรฐานใหม่ที่ประเมิน model ทั่วไปแบบ multi-metric
    </div>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.</li>
      <li>Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.</li>
      <li>Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: ViT. ICLR.</li>
      <li>Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with T5. JMLR.</li>
      <li>Jaegle, A. et al. (2021). Perceiver: General Perception with Iterative Attention. ICML.</li>
      <li>Stanford HELM Project: https://crfm.stanford.edu/helm/latest/</li>
      <li>MIT 6.S191 Deep Learning Lecture Notes 2023.</li>
      <li>Stanford CS224n Lecture Notes 2023.</li>
    </ul>

  </div>
</section>


          {/* Section 10 */}
     <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[700px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">บทนำ</h3>
    <p>
      ในงานวิจัยและบทเรียนด้าน Deep Learning การสื่อสารแนวคิดเชิงลึกที่ไม่สามารถแสดงผ่านตัวชี้วัดเชิงปริมาณอย่าง accuracy หรือ loss curve ได้อย่างเพียงพอ ถือเป็นสิ่งจำเป็นเพื่อเสริมสร้างความเข้าใจในพฤติกรรมภายในของโมเดล Attention-based architectures แนวทางหนึ่งที่ได้รับการยอมรับจากสถาบันชั้นนำ เช่น Stanford และ MIT คือการใช้ <strong>Insight Box</strong> หรือกล่องสรุปข้อสังเกตเชิงลึก ซึ่งช่วยยกระดับการสื่อสารทางวิชาการและช่วยให้นักวิจัยและนักพัฒนาตีความผลลัพธ์ของโมเดลได้อย่างมีบริบท
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> การใช้ Insight Box อย่างเป็นระบบในงานวิจัย Deep Learning ช่วยเปิดเผยข้อสังเกตเชิงกลยุทธ์เกี่ยวกับ inductive bias ของโมเดล, การกระจาย attention weight, และ failure modes ที่ไม่สามารถเข้าใจได้จากค่า loss เพียงอย่างเดียว
    </div>

    <h3 className="text-xl font-semibold">องค์ประกอบสำคัญของ Insight Box</h3>
    <p>Insight Box ที่มีประสิทธิภาพในงานวิชาการประกอบด้วยองค์ประกอบดังนี้:</p>
    <ul className="list-disc list-inside ml-4">
      <li><strong>Contextualization:</strong> ให้บริบทว่ากล่องนี้มาจากผลลัพธ์หรือการสังเกตจากส่วนใดของ pipeline</li>
      <li><strong>Core Insight:</strong> ข้อสังเกตเชิงกลยุทธ์ที่สื่อให้เห็นพฤติกรรม, ข้อจำกัด, หรือ pattern เชิงลึก</li>
      <li><strong>Implications:</strong> ผลกระทบของ insight ต่อการออกแบบหรือการปรับปรุง future iterations ของโมเดล</li>
      <li><strong>Actionable Suggestions (optional):</strong> ข้อเสนอแนะที่สามารถนำไปใช้ได้จริง</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่าง Insight Box ที่ดี</h3>
    <div className="overflow-x-auto w-full">
  <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
    <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
      <tr>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Component</th>
        <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Example Content</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Contextualization</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Analysis of attention maps during fine-tuning of ViT on ImageNet</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Core Insight</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Attention heads in layer 4-6 consistently focus on object borders across various object classes</td>
      </tr>
      <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Implications</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">This suggests that intermediate layers specialize in learning shape boundaries, informing pruning strategies</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ease-in-out">
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Actionable Suggestions</td>
        <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Consider visualizing layer-wise attention to inform architecture search for specific vision tasks</td>
      </tr>
    </tbody>
  </table>
</div>

    <h3 className="text-xl font-semibold">การนำ Insight Box ไปใช้ในงานสอนและงานวิจัย</h3>
    <p>
      หลักสูตรระดับบัณฑิตศึกษาจาก Stanford (CS224n), MIT (6.S191), และ Harvard NLP แนะนำให้นักศึกษาเพิ่ม Insight Box ในรายงานโปรเจกต์เพื่อสะท้อนความเข้าใจเชิงลึกเหนือกว่าตัวเลข metric พื้นฐาน โดย Insight Box มีบทบาทสำคัญในการ:
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ชี้ให้เห็น behavior ที่ unexpected ของโมเดล (เช่น attention drift ใน long sequences)</li>
      <li>สื่อสาร reasoning เชิงกลยุทธ์สำหรับ architectural decisions</li>
      <li>ช่วย debugging failure cases</li>
      <li>เสนอแนวทาง future work ที่มีพื้นฐานจาก insight จริง</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-700">
      <strong>Highlight:</strong> งาน BERTology (Clark et al., 2019) ใช้ Insight Box อย่างเป็นระบบในการสรุปพฤติกรรมของ attention head ในแต่ละ layer ซึ่งนำไปสู่การพัฒนา lightweight BERT variants
    </div>

    <h3 className="text-xl font-semibold">โครงสร้าง JSX มาตรฐานของ Insight Box</h3>
    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
      <code>
{`<div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
  <strong>Insight:</strong> Attention head 3 in layer 5 specializes in capturing entity boundaries in long-form text, as evidenced by consistent weight concentration on noun phrase tokens across multiple examples.
</div>`}
      </code>
    </pre>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ Insight Box</h3>
    <p>
      แม้ Insight Box จะมีประโยชน์อย่างมาก แต่การใช้โดยขาดการ validate อย่างเป็นระบบอาจนำไปสู่ bias หรือ overinterpretation ได้ ข้อแนะนำจากงานวิจัย (Lipton, 2016; Doshi-Velez & Kim, 2017) คือ:
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ควรใช้ร่วมกับ quantitative validation เสมอ</li>
      <li>ควรแสดงตัวอย่าง concrete พร้อม insight</li>
      <li>ควรหลีกเลี่ยงการ generalize จากตัวอย่างที่มี sample size เล็ก</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-700">
      <strong>Insight:</strong> การใช้ Insight Box อย่างมีวินัยในงานวิจัย Deep Learning ช่วยเพิ่ม interpretability และ transparency ของโมเดล ซึ่งเป็นเป้าหมายสำคัญของ explainable AI (XAI)
    </div>

    <h3 className="text-xl font-semibold">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Clark, K. et al. (2019). What Does BERT Look at? An Analysis of BERT's Attention. ACL.</li>
      <li>Lipton, Z. C. (2016). The Mythos of Model Interpretability. arXiv:1606.03490.</li>
      <li>Doshi-Velez, F., & Kim, B. (2017). Towards A Rigorous Science of Interpretable Machine Learning. arXiv:1702.08608.</li>
      <li>Stanford CS224n Lecture Notes 2023.</li>
      <li>MIT 6.S191 Deep Learning Course Materials 2023.</li>
    </ul>

  </div>
</section>


          {/* Quiz */}
          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day50 theme={theme} />
          </section>

          {/* Tags */}
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

          {/* Comments */}
          <Comments theme={theme} />
          <div className="mb-20" />
        </div>
      </div>

      {/* ScrollSpy */}
      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day50 />
      </div>

      {/* SupportMeButton */}
      <SupportMeButton />
    </div>
  );
};

export default Day50_AttentionMechanism;
