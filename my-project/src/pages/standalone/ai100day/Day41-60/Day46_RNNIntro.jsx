import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day46 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day46";
import MiniQuiz_Day46 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day46";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day46_RNNIntro = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day46_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day46_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day46_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day46_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day46_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day46_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day46_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day46_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day46_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day46_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day46_11").format("auto").quality("auto").resize(scale().width(500));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 46: Introduction to RNNs & Sequence Models</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

       <section id="why-sequence" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. ทำไมต้องใช้ Sequence Models?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className=" prose-base dark:prose-invert max-w-none space-y-6">
    <p>
      Sequence Models เช่น RNN, LSTM, GRU และ Transformer ถูกออกแบบมาเพื่อจัดการกับข้อมูลที่มีลำดับ เช่น ข้อความ เสียง วิดีโอ หรือชุดข้อมูลเวลา (time series) ซึ่งมี dependency ระหว่างตำแหน่งหรือช่วงเวลาในข้อมูลต่างจากการประมวลผลข้อมูลที่เป็นอิสระต่อกันเช่นใน CNN
    </p>

    <h3 className="text-xl font-semibold">ความแตกต่างจากโมเดลแบบทั่วไป</h3>
    <ul className="list-disc pl-6">
      <li>โมเดลทั่วไป (เช่น MLP) จะรับ input แบบ fixed-length vector เท่านั้น</li>
      <li>Sequence Models รับ input ที่มีความยาวเปลี่ยนแปลงได้ เช่น ประโยคที่มีความยาวต่างกัน</li>
      <li>สามารถจดจำบริบทในลำดับได้ ซึ่งสำคัญสำหรับความเข้าใจเชิงเวลา</li>
    </ul>

    <div className="bg-yellow-600 dark:bg-yellow-800 text-black dark:text-white rounded-lg p-4">
      <strong>Insight:</strong> ในปี 2017 Google Brain ได้เผยแพร่ Transformer ซึ่งพลิกโฉมวงการ Sequence Modeling โดยกำจัดการพึ่งพา RNN และใช้ Attention แทนทั้งหมด (Vaswani et al., 2017)
    </div>

    <h3 className="text-xl font-semibold">ลักษณะงานที่ต้องการลำดับ</h3>
    <ul className="list-disc pl-6">
      <li>การแปลภาษา (Neural Machine Translation)</li>
      <li>การรู้จำเสียงพูด (Speech Recognition)</li>
      <li>การวิเคราะห์หุ้นแบบ Time-Series</li>
      <li>การตอบคำถามจากเอกสาร (QA Systems)</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ</h3>
    <div className="overflow-auto rounded-lg border border-gray-300 dark:border-gray-700">
      <table className="table-auto w-full text-left text-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="px-4 py-2">โมเดล</th>
            <th className="px-4 py-2">เหมาะกับ</th>
            <th className="px-4 py-2">จุดเด่น</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border-t px-4 py-2">RNN</td>
            <td className="border-t px-4 py-2">ข้อมูลเรียงลำดับ</td>
            <td className="border-t px-4 py-2">มี memory ในตัว</td>
          </tr>
          <tr>
            <td className="border-t px-4 py-2">LSTM</td>
            <td className="border-t px-4 py-2">ลำดับยาว</td>
            <td className="border-t px-4 py-2">แก้ vanishing gradient</td>
          </tr>
          <tr>
            <td className="border-t px-4 py-2">GRU</td>
            <td className="border-t px-4 py-2">ประสิทธิภาพดี</td>
            <td className="border-t px-4 py-2">ง่ายกว่า LSTM</td>
          </tr>
          <tr>
            <td className="border-t px-4 py-2">Transformer</td>
            <td className="border-t px-4 py-2">ทุกชนิดลำดับ</td>
            <td className="border-t px-4 py-2">ขนานได้, เข้าใจบริบทดี</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบหลักของ Sequence Models</h3>
    <div className="bg-blue-500 dark:bg-blue-900 text-black dark:text-white rounded-lg p-4">
      <p>
        การเข้าใจความสัมพันธ์ระหว่างข้อมูลตามลำดับเวลา เช่น คำว่า “กิน” ตามด้วย “ข้าว” มีความหมายมากกว่าการสลับลำดับคำกัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มในปัจจุบัน</h3>
    <p>
      งานวิจัยล่าสุดจาก Stanford (Jurafsky & Manning) และ Google Research ชี้ให้เห็นว่า Transformer และ Self-Attention เป็นแกนหลักของสถาปัตยกรรมสมัยใหม่ เช่น BERT, GPT, T5 ซึ่งสามารถประมวลผลข้อมูลลำดับได้ดีกว่าทุกวิธีที่เคยมีมา
    </p>

    <h3 className="text-xl font-semibold">อ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Vaswani et al., “Attention is All You Need”, NeurIPS 2017</li>
      <li>Jurafsky & Martin, “Speech and Language Processing”, Stanford</li>
      <li>Hochreiter & Schmidhuber, “LSTM”, Neural Computation 1997</li>
      <li>Oxford Deep Learning Course: Sequence Models Module</li>
    </ul>
  </div>
</section>


      <section id="rnn-basics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    2. แนวคิดพื้นฐานของ RNN (Recurrent Neural Network)
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">ลักษณะเด่นของ RNN</h3>
  <p className="text-base leading-relaxed mb-4">
    RNN ถูกออกแบบมาเพื่อจัดการกับข้อมูลลำดับ (sequence data) เช่น ข้อความ เสียง หรือข้อมูลเชิงเวลา โดยใช้กลไกของ recurrent connection ซึ่งหมายถึงการนำ output ของ timestep ก่อนหน้ากลับมาเป็น input ของ timestep ถัดไป ทำให้สามารถจดจำข้อมูลในอดีตเพื่อประมวลผลปัจจุบันได้
  </p>

  <div className="bg-blue-500 rounded-lg p-4 mb-6">
    <p className="text-sm">
      Insight: การทำงานของ RNN คล้ายกับหน่วยความจำระยะสั้นของสมองมนุษย์ ที่สามารถจำข้อมูลก่อนหน้าเพื่อวิเคราะห์ข้อมูลถัดไปได้อย่างต่อเนื่อง
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">โครงสร้างพื้นฐานของ RNN</h3>
<p className="text-base leading-relaxed mb-4">
  ในแต่ละ timestep ของลำดับข้อมูล RNN จะรับ input ปัจจุบัน <code>x_t</code> และ hidden state จาก timestep ก่อนหน้า <code>h_t-1</code> แล้วคำนวณ hidden state ใหม่ <code>h_t</code> ตามสูตร:
</p>
<pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4 mb-6">
  <code>h_t = tanh(W_h · h_t-1 + W_x · x_t + b)</code>
</pre>


  <h3 className="text-xl font-semibold mt-8 mb-4">จุดแข็งของ RNN</h3>
  <ul className="list-disc ml-6 text-base mb-6">
    <li>สามารถรับข้อมูลที่มีลำดับเวลาได้โดยตรง</li>
    <li>เหมาะสมกับข้อมูลเชิงลำดับ เช่น เสียง, คำพูด, ข้อความ, หรือ stock price</li>
    <li>สามารถแชร์พารามิเตอร์ข้าม timestep ได้ ทำให้โมเดลขนาดเล็กลง</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-4">ข้อจำกัดของ RNN ดั้งเดิม</h3>
  <p className="text-base leading-relaxed mb-4">
    ถึงแม้ว่า RNN จะสามารถจัดการกับข้อมูลลำดับได้ดีในระยะสั้น แต่เมื่อ sequence ยาวขึ้น มักประสบปัญหา vanishing gradient หรือ exploding gradient ซึ่งทำให้โมเดลไม่สามารถเรียนรู้ long-term dependency ได้อย่างมีประสิทธิภาพ
  </p>

  <table className="table-auto w-full border border-gray-500 mb-8">
    <thead>
      <tr className="bg-gray-500 text-white">
        <th className="px-4 py-2 border">คุณสมบัติ</th>
        <th className="px-4 py-2 border">รายละเอียด</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Memory</td>
        <td className="border px-4 py-2">Short-term (จำข้อมูลล่าสุดได้ดี)</td>
      </tr>
      <tr className="bg-gray-500 dark:bg-gray-800">
        <td className="border px-4 py-2">Parameter Sharing</td>
        <td className="border px-4 py-2">ใช้พารามิเตอร์ชุดเดียวในแต่ละ timestep</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Vanishing Gradient</td>
        <td className="border px-4 py-2">พบบ่อยเมื่อ sequence ยาวมาก</td>
      </tr>
    </tbody>
  </table>

  <div className="bg-yellow-500 rounded-lg p-4 mb-6">
    <p className="text-sm">
      Highlight: งานวิจัยจาก Stanford และ Oxford ชี้ว่า RNN เหมาะสำหรับ task ที่มีความต่อเนื่องทางเวลาสูง เช่น การจำแนกเสียงพูด หรือการแปลภาษาในลำดับคำแบบ real-time
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc ml-6 text-base mb-2">
    <li>Stanford CS224N: Natural Language Processing with Deep Learning</li>
    <li>arXiv: "Learning long-term dependencies with gradient descent is difficult" (Bengio et al., 1994)</li>
    <li>Oxford Deep Learning Lecture Notes: Recurrent Networks</li>
  </ul>
</section>

       <section id="seq2seq-types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    3. ประเภทของ Sequence-to-Sequence Tasks
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">Sequence-to-Sequence คืออะไร?</h3>
  <p className="mb-4">
    Sequence-to-Sequence (Seq2Seq) เป็นรูปแบบของโมเดลที่รับข้อมูลลำดับหนึ่ง (input sequence) แล้วสร้างผลลัพธ์เป็นอีกลำดับหนึ่ง (output sequence) โดยมีการใช้งานอย่างแพร่หลายในงานประมวลผลภาษาธรรมชาติ เช่น การแปลภาษา สรุปข้อความ และการสร้างคำบรรยายภาพ
  </p>

  <div className="bg-yellow-500 rounded-md p-4 border mb-6">
    <strong className="block mb-2">Insight:</strong>
    โมเดล Seq2Seq มักประกอบด้วย Encoder และ Decoder ซึ่งสามารถพัฒนาเพิ่มเติมได้ด้วย Attention Mechanism เพื่อเพิ่มประสิทธิภาพในการจำบริบทของข้อมูลยาว ๆ
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">ประเภทของ Seq2Seq Tasks</h3>
  <ul className="list-disc list-inside mb-6">
    <li><strong>One-to-One:</strong> ใช้กับข้อมูลที่มีความยาวคงที่ เช่น การจำแนกประเภทข้อความสั้น</li>
    <li><strong>One-to-Many:</strong> เช่น การสร้างคำอธิบายภาพจากรูปเดียว</li>
    <li><strong>Many-to-One:</strong> เช่น การวิเคราะห์ความรู้สึกจากประโยค</li>
    <li><strong>Many-to-Many:</strong> เช่น การแปลภาษา ที่ input และ output เป็นลำดับ</li>
  </ul>

  <h3 className="text-xl font-semibold mt-10 mb-4">ตัวอย่างการใช้งาน Seq2Seq</h3>
  <table className="table-auto w-full border mt-4 text-sm">
    <thead>
      <tr className="bg-gray-300 dark:bg-gray-600">
        <th className="border px-4 py-2">ประเภท</th>
        <th className="border px-4 py-2">ตัวอย่าง</th>
        <th className="border px-4 py-2">การใช้งานจริง</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">One-to-One</td>
        <td className="border px-4 py-2">ข้อความสั้น → label</td>
        <td className="border px-4 py-2">Spam Detection</td>
      </tr>
      <tr className="bg-gray-500 dark:bg-gray-700">
        <td className="border px-4 py-2">One-to-Many</td>
        <td className="border px-4 py-2">รูปภาพ → คำบรรยาย</td>
        <td className="border px-4 py-2">Image Captioning</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Many-to-One</td>
        <td className="border px-4 py-2">ประโยค → ความรู้สึก</td>
        <td className="border px-4 py-2">Sentiment Analysis</td>
      </tr>
      <tr className="bg-gray-500 dark:bg-gray-700">
        <td className="border px-4 py-2">Many-to-Many</td>
        <td className="border px-4 py-2">ประโยคภาษาอังกฤษ → ประโยคภาษาไทย</td>
        <td className="border px-4 py-2">Machine Translation</td>
      </tr>
    </tbody>
  </table>

  <div className="bg-blue-500 rounded-md p-4 border mt-10">
    <strong className="block mb-2">Highlight:</strong>
    การเลือกประเภทของ Seq2Seq ขึ้นกับลักษณะข้อมูลและเป้าหมายของงาน เช่น ระบบแชทบอทอาจใช้ Many-to-Many เพื่อโต้ตอบข้อความแบบต่อเนื่อง
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">ความท้าทายในการออกแบบ</h3>
  <p className="mb-4">
    การเลือกโมเดลที่เหมาะสมกับแต่ละประเภทของงาน Seq2Seq ต้องคำนึงถึงหลายปัจจัย เช่น ความยาวลำดับ ความไม่แน่นอนของโครงสร้างข้อมูล และความต้องการด้าน latency
  </p>
  <ul className="list-disc list-inside mb-10">
    <li>ระบบที่ต้องการการตอบสนองเร็วอาจเลือกใช้ GRU แทน LSTM</li>
    <li>งานแปลภาษาที่ยาวมากจะได้ประสิทธิภาพดีกว่าเมื่อใช้ Attention</li>
    <li>การจัดการ Sequence ที่มีความแปรปรวนสูงควรใช้โมเดลที่รองรับ Dynamic Length</li>
  </ul>

  <h3 className="text-xl font-semibold mt-10 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside">
    <li>Stanford CS224n: Natural Language Processing with Deep Learning</li>
    <li>Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation", arXiv 2014</li>
    <li>Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2015</li>
    <li>Oxford NLP Group, Sequence Modeling Notes</li>
  </ul>
</section>


     <section id="training-challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การฝึก RNN และปัญหาที่พบ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <h3 className="text-xl font-medium mt-10 mb-4">ปัญหา Gradient Vanishing และ Exploding</h3>
  <p className="text-base leading-relaxed">
    หนึ่งในความท้าทายหลักของการฝึก RNN คือปัญหา Gradient Vanishing และ Exploding ซึ่งส่งผลต่อการเรียนรู้ระยะยาวของเครือข่าย โดยเฉพาะเมื่อจำนวน time steps มีความยาวมาก ตัว gradient ที่ถูกถ่ายทอดย้อนกลับจะลดลงหรือเพิ่มขึ้นแบบทวีคูณ
  </p>

  <div className="bg-yellow-500 dark:bg-yellow-800 rounded-lg p-4 my-6">
    <p className="text-sm font-medium">
      Insight:
    </p>
    <p className="text-sm">
      งานวิจัยจาก Bengio et al. (1994) แสดงให้เห็นว่าความยาวของ sequence ส่งผลโดยตรงต่ออัตราการลืมหรือสะสมข้อมูลใน RNN ซึ่งนำไปสู่การเปลี่ยนแปลงของ gradient อย่างมีนัยสำคัญ
    </p>
  </div>

  <h3 className="text-xl font-medium mt-10 mb-4">เทคนิคการแก้ปัญหาเบื้องต้น</h3>
  <ul className="list-disc list-inside space-y-2">
    <li>ใช้การ normalize ค่า gradient ผ่านการทำ Gradient Clipping</li>
    <li>เปลี่ยนจาก RNN ปกติเป็น LSTM หรือ GRU ซึ่งมีโครงสร้างที่ออกแบบมาเพื่อแก้ปัญหา gradient vanishing</li>
    <li>ใช้ Initialization ที่มีความเหมาะสม เช่น Xavier หรือ He Initialization</li>
  </ul>

  <div className="bg-blue-500 dark:bg-blue-800 rounded-lg p-4 my-6">
    <p className="text-sm font-medium">
      Highlight:
    </p>
    <p className="text-sm">
      เทคนิค Gradient Clipping เป็นหนึ่งในวิธีที่ง่ายและได้ผลในการควบคุมการระเบิดของ gradient โดยจำกัด norm ของ gradient ไม่ให้เกินค่าที่กำหนด
    </p>
  </div>

  <h3 className="text-xl font-medium mt-10 mb-4">การใช้ Optimizer แบบ Adaptive</h3>
  <p className="text-base leading-relaxed">
    การใช้งาน Optimizer ที่สามารถปรับ learning rate ได้เอง เช่น Adam, RMSProp หรือ Adagrad ช่วยให้การฝึก RNN มีเสถียรภาพมากขึ้นในลักษณะที่ gradient มีความไม่แน่นอนสูง
  </p>

  <table className="w-full text-sm mt-6 border border-gray-500">
    <thead>
      <tr className="bg-gray-500 text-white">
        <th className="p-2">Optimizer</th>
        <th className="p-2">ลักษณะเด่น</th>
        <th className="p-2">เหมาะกับ</th>
      </tr>
    </thead>
    <tbody>
      <tr className="border-t border-gray-500">
        <td className="p-2">Adam</td>
        <td className="p-2">มี momentum และ adaptive learning rate</td>
        <td className="p-2">โมเดลที่มี non-stationary gradient</td>
      </tr>
      <tr className="border-t border-gray-500">
        <td className="p-2">RMSProp</td>
        <td className="p-2">ควบคุม learning rate ด้วย moving average</td>
        <td className="p-2">RNN ที่ไม่สม่ำเสมอใน gradient</td>
      </tr>
      <tr className="border-t border-gray-500">
        <td className="p-2">Adagrad</td>
        <td className="p-2">ลด learning rate อัตโนมัติตาม parameter</td>
        <td className="p-2">โมเดลที่ต้องการ regularization</td>
      </tr>
    </tbody>
  </table>

  <h3 className="text-xl font-medium mt-10 mb-4">ผลกระทบต่อการวิเคราะห์ข้อมูลตามลำดับ</h3>
  <p className="text-base leading-relaxed">
    หากไม่จัดการกับปัญหา gradient ที่เกิดขึ้นอย่างเหมาะสม จะส่งผลให้โมเดลสูญเสียความสามารถในการเรียนรู้ความสัมพันธ์เชิงลึกระหว่าง token ใน sequence เช่น ในงานแปลภาษา RNN อาจแปลผิดในส่วนที่เกี่ยวข้องกับคำแรก ๆ
  </p>

  <h3 className="text-xl font-medium mt-10 mb-4">แนวทางการพัฒนาในอนาคต</h3>
  <ul className="list-disc list-inside space-y-2">
    <li>ใช้เครือข่ายที่มีหน่วยควบคุมเวลา เช่น Clockwork RNN หรือ Skip-RNN</li>
    <li>รวม RNN เข้ากับ Attention Mechanism เพื่อเสริมพลังในการจดจำ</li>
    <li>เปลี่ยนมาใช้สถาปัตยกรรมแบบ Transformer แทนในกรณี sequence ยาว</li>
  </ul>

  <div className="bg-yellow-500 dark:bg-yellow-800 rounded-lg p-4 my-6">
    <p className="text-sm font-medium">
      Insight:
    </p>
    <p className="text-sm">
      การประยุกต์ใช้ Attention Mechanism มีผลอย่างยิ่งต่อการลดการสูญเสีย gradient และเพิ่มประสิทธิภาพของการเรียนรู้แบบลำดับ ซึ่งเป็นที่นิยมในการพัฒนา NLP Model ปัจจุบัน
    </p>
  </div>

  <h3 className="text-xl font-medium mt-10 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside text-sm space-y-2">
    <li>Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks.</li>
    <li>Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078</li>
    <li>Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.</li>
    <li>Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980</li>
  </ul>
</section>


   <section id="visual-flow" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Visual Insight: การไหลของข้อมูลใน RNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="text-base leading-relaxed space-y-6">
    <p>
      การไหลของข้อมูลในโครงข่าย RNN มีลักษณะเฉพาะที่แตกต่างจากโครงข่ายแบบ feedforward โดยโครงสร้างของ RNN อนุญาตให้แต่ละโหนดสามารถส่งข้อมูลย้อนกลับมายังตัวเองหรือไปยังโหนดก่อนหน้าในลำดับเวลาได้ ซึ่งทำให้ RNN มีความสามารถในการจดจำลำดับและบริบททางเวลา
    </p>

    <h3 className="text-xl font-semibold mt-8">ลักษณะของการไหลข้อมูลใน RNN</h3>
    <p>
      การคำนวณของ RNN ในแต่ละ timestep จะใช้ข้อมูลจาก input ปัจจุบันและ hidden state จาก timestep ก่อนหน้าเป็นหลัก โดยสามารถสรุป flow ได้ในขั้นตอนดังนี้:
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>รับ input ที่ timestep t คือ <code>x<sub>t</sub></code></li>
      <li>คำนวณ hidden state ปัจจุบันจาก <code>h<sub>t</sub> = f(Wx<sub>t</sub> + Uh<sub>t−1</sub> + b)</code></li>
      <li>คำนวณ output ตาม task เช่น <code>y<sub>t</sub> = g(Vh<sub>t</sub> + c)</code></li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">สถาปัตยกรรมเชิงเวลา (Unfolded Architecture)</h3>
    <p>
      หาก "คลี่" โครงสร้างของ RNN ออกตามลำดับเวลา จะพบว่าโครงข่ายหนึ่งเดียวใน RNN จะถูกใช้งานซ้ำหลายครั้งในลำดับ input แต่มี state ที่เปลี่ยนไปตามเวลา ลักษณะนี้แสดงให้เห็นการ reuse weights และ temporal context
    </p>

    <div className="bg-yellow-500 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold mb-2">Insight</p>
      <p>
        ความสามารถในการสร้างบริบทของลำดับแบบต่อเนื่องใน RNN เป็นพื้นฐานสำคัญของการประมวลผลภาษาธรรมชาติ (NLP) เช่น machine translation และ text generation
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8">รูปแบบ Output: One-to-One, One-to-Many, Many-to-Many</h3>
    <table className="table-auto border border-gray-500 w-full mt-4 text-sm">
      <thead>
        <tr className="bg-gray-500 text-white">
          <th className="border border-gray-600 px-4 py-2">Type</th>
          <th className="border border-gray-600 px-4 py-2">Description</th>
          <th className="border border-gray-600 px-4 py-2">Use Case</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border border-gray-500 px-4 py-2">One-to-One</td>
          <td className="border border-gray-500 px-4 py-2">Input และ Output เดี่ยว</td>
          <td className="border border-gray-500 px-4 py-2">Image Classification</td>
        </tr>
        <tr>
          <td className="border border-gray-500 px-4 py-2">One-to-Many</td>
          <td className="border border-gray-500 px-4 py-2">Input เดียว ส่งออกลำดับ</td>
          <td className="border border-gray-500 px-4 py-2">Image Captioning</td>
        </tr>
        <tr>
          <td className="border border-gray-500 px-4 py-2">Many-to-One</td>
          <td className="border border-gray-500 px-4 py-2">ลำดับ Input สู่ Output เดียว</td>
          <td className="border border-gray-500 px-4 py-2">Sentiment Analysis</td>
        </tr>
        <tr>
          <td className="border border-gray-500 px-4 py-2">Many-to-Many</td>
          <td className="border border-gray-500 px-4 py-2">Input และ Output เป็นลำดับ</td>
          <td className="border border-gray-500 px-4 py-2">Machine Translation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8">ข้อจำกัดที่ปรากฏผ่านการไหลของข้อมูล</h3>
    <p>
      แม้ว่า RNN จะสามารถเรียนรู้บริบทในลำดับได้ แต่ก็มีข้อจำกัด เช่น การสูญเสีย gradient เมื่อ sequence ยาวเกินไป (vanishing gradients) และการใช้เวลาในการฝึกมากขึ้นหากต้องจัดการกับ long-term dependencies
    </p>

    <div className="bg-blue-500 border-l-4 border-blue-500 p-4 rounded mt-6">
      <p className="font-semibold mb-2">Highlight</p>
      <p>
        งานวิจัยจาก Harvard และ DeepMind ชี้ให้เห็นว่า long-sequence modeling ด้วย RNN อาจไม่สามารถจับ long-term dependencies ได้อย่างมีประสิทธิภาพเทียบเท่ากับ Gated Models เช่น LSTM และ GRU
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 mt-2">
      <li>Graves, A. (2013). Generating Sequences With Recurrent Neural Networks. arXiv:1308.0850</li>
      <li>Le, Q.V., Jaitly, N., & Hinton, G.E. (2015). A Simple Way to Initialize Recurrent Networks. arXiv:1504.00941</li>
      <li>Karpathy, A., & Fei-Fei, L. (2015). Deep Visual-Semantic Alignments for Generating Image Descriptions. CVPR.</li>
      <li>Sutskever, I., Martens, J., & Hinton, G. (2011). Generating Text with Recurrent Neural Networks. ICML.</li>
    </ul>
  </div>
</section>


     <section id="rnn-nlp" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. RNN กับ NLP: การประมวลผลประโยค</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">ลำดับคำในภาษาและการพึ่งพาข้อมูลลำดับ</h3>
    <p>
      การประมวลผลภาษาธรรมชาติ (NLP) ต้องการการวิเคราะห์ข้อมูลในเชิงลำดับ โดยเฉพาะในกรณีที่คำแต่ละคำมีความหมายขึ้นอยู่กับคำก่อนหน้า เช่น โครงสร้างประโยคในภาษาอังกฤษ เช่น “He went to the…” ต้องการให้โมเดลเข้าใจบริบทของคำก่อนหน้าเพื่อคาดเดาคำถัดไป
    </p>
    <h3 className="text-xl font-semibold">บทบาทของ RNN ในการประมวลผลลำดับ</h3>
    <p>
      Recurrent Neural Networks (RNNs) ถูกออกแบบมาเพื่อจัดการกับข้อมูลลำดับ โดยนำค่าผลลัพธ์จากขั้นก่อนหน้ากลับมาใช้งานกับขั้นตอนถัดไป (feedback connection) โมเดลสามารถรักษาสถานะภายใน (hidden state) เพื่อจำข้อมูลที่เกิดขึ้นก่อนหน้า ทำให้ RNNs เหมาะสำหรับการเรียนรู้โครงสร้างของภาษา เช่น คำสรรพนาม, ลำดับประโยค, หรือความสัมพันธ์ระหว่างคำในข้อความยาว ๆ
    </p>
    <div className="bg-yellow-500 p-4 rounded-md border-l-4 border-yellow-500">
      <p className="font-semibold">Insight Box:</p>
      <p>
        การใช้ RNN ในงานด้าน NLP ได้กลายเป็นจุดเริ่มต้นของความก้าวหน้าในเทคโนโลยีการแปลภาษา การรู้จำเสียงพูด และการสรุปความ โดยเฉพาะอย่างยิ่งในยุคก่อนที่จะมี Transformer โมเดลอย่าง Seq2Seq และ LSTM ถูกใช้อย่างแพร่หลาย
      </p>
    </div>
    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งาน RNN ในภาษา</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Machine Translation: แปลข้อความจากภาษาหนึ่งไปอีกภาษา</li>
      <li>Speech Recognition: แปลงเสียงพูดเป็นข้อความ</li>
      <li>Text Summarization: สรุปใจความสำคัญของบทความ</li>
      <li>Sentiment Analysis: วิเคราะห์ความคิดเห็นของผู้ใช้จากข้อความ</li>
    </ul>
    <h3 className="text-xl font-semibold">ปัญหาและข้อจำกัดของ RNN ใน NLP</h3>
    <p>
      แม้ว่า RNN จะสามารถเข้าใจลำดับของภาษาได้ แต่มีข้อจำกัดหลายประการ เช่น:
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Vanishing Gradient: ค่าน้ำหนักย้อนกลับอาจเล็กลงจนโมเดลไม่สามารถเรียนรู้บริบทระยะไกลได้</li>
      <li>Memory Constraint: โมเดลมีความสามารถจำกัดในการเก็บบริบทยาว ๆ</li>
      <li>Computation Time: ต้องคำนวณตามลำดับเวลาซึ่งทำให้การฝึกใช้เวลานาน</li>
    </ul>
    <div className="bg-blue-500 p-4 rounded-md border-l-4 border-blue-500">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยจาก Harvard NLP และ Google Brain ได้เสนอแนวทางในการปรับปรุง RNN ผ่าน LSTM และ GRU เพื่อเพิ่มความสามารถในการจัดการข้อมูลลำดับที่ยาวขึ้น ซึ่งกลายเป็นพื้นฐานสำคัญก่อนการมาถึงของ Transformer
      </p>
    </div>
    <h3 className="text-xl font-semibold">การเปรียบเทียบ RNN กับโมเดลสมัยใหม่</h3>
    <table className="table-auto w-full border mt-6">
      <thead>
        <tr className="bg-gray-200">
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">RNN</th>
          <th className="border px-4 py-2">Transformer</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การประมวลผลลำดับ</td>
          <td className="border px-4 py-2">เชิงลำดับ (Sequential)</td>
          <td className="border px-4 py-2">ขนาน (Parallelizable)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความยาวบริบท</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">กว้างและยืดหยุ่น</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การฝึกอบรม</td>
          <td className="border px-4 py-2">ช้า</td>
          <td className="border px-4 py-2">เร็ว</td>
        </tr>
      </tbody>
    </table>
    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Cho et al., 2014, "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"</li>
      <li>Sutskever et al., 2014, "Sequence to Sequence Learning with Neural Networks"</li>
      <li>Harvard NLP Group, "The Annotated RNN"</li>
      <li>Google Brain, "Neural Machine Translation by Jointly Learning to Align and Translate"</li>
    </ul>
  </div>
</section>


       <section id="rnn-vs-cnn" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. การเปรียบเทียบ RNN vs CNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="text-base leading-relaxed space-y-6">
    <h3 className="text-xl font-semibold">ลักษณะโครงสร้างของ RNN และ CNN</h3>
    <p>
      Recurrent Neural Networks (RNNs) และ Convolutional Neural Networks (CNNs) เป็นสถาปัตยกรรมหลักที่ใช้กันอย่างแพร่หลายใน Deep Learning แต่ละแบบมีลักษณะโครงสร้างและวิธีการประมวลผลที่แตกต่างกัน โดย RNN ได้เปรียบในการจัดการกับข้อมูลลำดับ เช่น ข้อความหรือเสียง ขณะที่ CNN มีความโดดเด่นในการวิเคราะห์ภาพและข้อมูลแบบ 2 มิติ
    </p>

    <h3 className="text-xl font-semibold">การทำงานภายในของแต่ละโครงข่าย</h3>
    <p>
      CNN ใช้ kernel หรือ filter ขนาดเล็กในการสแกนพื้นที่ของภาพเพื่อดึง feature ที่สำคัญ ในขณะที่ RNN ทำการวนซ้ำผ่าน sequence โดยคงสถานะของหน่วยความจำไว้ระหว่างขั้นตอนเวลา (time steps)
    </p>

    <table className="table-auto w-full border border-gray-500">
      <thead>
        <tr className="bg-gray-500 text-white">
          <th className="border px-4 py-2 text-left">คุณสมบัติ</th>
          <th className="border px-4 py-2 text-left">CNN</th>
          <th className="border px-4 py-2 text-left">RNN</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ข้อมูลที่เหมาะสม</td>
          <td className="border px-4 py-2">ภาพ, วิดีโอ</td>
          <td className="border px-4 py-2">ข้อความ, เสียง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การจัดการลำดับ</td>
          <td className="border px-4 py-2">ไม่เหมาะ</td>
          <td className="border px-4 py-2">เหมาะสม</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">พารามิเตอร์ที่ใช้ร่วมกัน</td>
          <td className="border px-4 py-2">ผ่าน Convolution Filters</td>
          <td className="border px-4 py-2">ผ่าน Time Steps</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การฝึกโมเดล</td>
          <td className="border px-4 py-2">ง่ายและเสถียรกว่า</td>
          <td className="border px-4 py-2">มีปัญหา vanishing gradients</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-500 border border-blue-300 p-4 rounded-md">
      <h3 className="font-bold mb-2">Insight Box: ความเหมาะสมตามงานประยุกต์</h3>
      <p>
        ในทางปฏิบัติ CNN มักใช้กับการรู้จำภาพทางการแพทย์หรือกล้องวงจรปิด ขณะที่ RNN นิยมใช้ในงานแปลภาษา, การสร้างคำอัตโนมัติ และการพยากรณ์ลำดับเวลา
      </p>
    </div>

    <h3 className="text-xl font-semibold">ข้อจำกัดของแต่ละแบบ</h3>
    <p>
      แม้ว่า RNN จะมีข้อได้เปรียบในด้านข้อมูลลำดับ แต่ก็มีข้อจำกัดในการเรียนรู้ความสัมพันธ์ระยะยาว ซึ่งปัญหานี้ได้รับการแก้ไขบางส่วนผ่าน LSTM และ GRU ขณะที่ CNN แม้จะมีประสิทธิภาพในการประมวลผลภาพ แต่ไม่สามารถเรียนรู้ความสัมพันธ์ตามลำดับได้ดีเท่า RNN
    </p>

    <div className="bg-yellow-500 border border-yellow-300 p-4 rounded-md">
      <h3 className="font-bold mb-2">Highlight Box: เมื่อไหร่ควรใช้ CNN หรือ RNN</h3>
      <ul className="list-disc list-inside space-y-1">
        <li>เลือกใช้ CNN สำหรับข้อมูลที่มีโครงสร้างแบบภาพ</li>
        <li>เลือกใช้ RNN หรือ LSTM เมื่อข้อมูลเป็นลำดับเวลา เช่น ภาษา หรือสัญญาณเสียง</li>
        <li>การรวม CNN + RNN เป็นเทคนิคยอดนิยมในงาน Video Captioning</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>CMU 11-785: Deep Learning for NLP</li>
      <li>LeCun et al., "Gradient-based learning applied to document recognition", Proceedings of the IEEE</li>
      <li>arXiv:1409.2329 - A Critical Review of Recurrent Neural Networks for Sequence Learning</li>
    </ul>
  </div>
</section>


      <section id="rnn-limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ข้อจำกัดของ RNN แบบดั้งเดิม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">ลักษณะของข้อจำกัดในเชิงโครงสร้าง</h3>
  <p className="text-base leading-relaxed mb-4">
    แม้ว่า Recurrent Neural Networks (RNNs) จะถูกออกแบบมาเพื่อจัดการกับข้อมูลตามลำดับ เช่น ข้อความหรือสัญญาณเวลา แต่โครงสร้างของมันยังมีข้อจำกัดหลายด้าน โดยเฉพาะการเรียนรู้ระยะยาว (long-term dependencies) ที่จำเป็นต่อการเข้าใจบริบทลึกของข้อมูลภาษา
  </p>

  <div className="bg-yellow-500 text-sm p-4 rounded-xl border border-yellow-400 mb-6">
    <strong>Insight:</strong> RNN ทั่วไปมีปัญหาเรื่อง vanishing gradient ซึ่งทำให้โมเดลไม่สามารถเรียนรู้ข้อมูลในอดีตที่อยู่ไกลจากลำดับปัจจุบันได้ดี
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">Vanishing และ Exploding Gradients</h3>
  <p className="text-base leading-relaxed mb-4">
    ปัญหาหลักที่ส่งผลต่อประสิทธิภาพของ RNN คือการที่ gradient มีแนวโน้มลดลงหรือเพิ่มขึ้นอย่างรวดเร็วเมื่อทำ backpropagation ผ่านลำดับเวลาที่ยาวมาก ทำให้โมเดลไม่สามารถเรียนรู้ข้อมูลในอดีตได้อย่างมีประสิทธิภาพ
  </p>

  <table className="w-full table-auto border border-gray-300 mb-6">
    <thead>
      <tr className="bg-gray-500 text-white">
        <th className="px-4 py-2">ประเภทของ Gradient</th>
        <th className="px-4 py-2">ผลกระทบต่อการเรียนรู้</th>
      </tr>
    </thead>
    <tbody>
      <tr className="border-b">
        <td className="px-4 py-2">Vanishing Gradient</td>
        <td className="px-4 py-2">การเรียนรู้ข้อมูลในอดีตล้มเหลว, น้ำหนักไม่ถูกอัปเดต</td>
      </tr>
      <tr>
        <td className="px-4 py-2">Exploding Gradient</td>
        <td className="px-4 py-2">ทำให้การฝึกโมเดลไม่เสถียร, ค่าพารามิเตอร์กลายเป็น infinity</td>
      </tr>
    </tbody>
  </table>

  <div className="bg-blue-500 text-sm p-4 rounded-xl border border-blue-400 mb-6">
    <strong>Highlight:</strong> การใช้ LSTM หรือ GRU เป็นวิธีหลักที่ใช้เพื่อแก้ไขปัญหา vanishing gradient โดยเพิ่มโครงสร้าง gate เพื่อควบคุมการไหลของข้อมูลและ gradient
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">ข้อจำกัดด้าน Parallelism</h3>
  <p className="text-base leading-relaxed mb-4">
    โครงสร้างของ RNN ทำให้ไม่สามารถขนานการประมวลผลได้ เนื่องจากแต่ละขั้นตอนต้องรอผลลัพธ์จากขั้นก่อนหน้า ซึ่งแตกต่างจาก CNN หรือ Transformer ที่สามารถประมวลผลพร้อมกันได้หลายตำแหน่งในเวลาเดียวกัน
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-4">ข้อจำกัดเชิงข้อมูลและบริบท</h3>
  <p className="text-base leading-relaxed mb-4">
    แม้ว่า RNN จะสามารถเข้าใจลำดับข้อมูล แต่ในบริบทที่ซับซ้อน เช่น ประโยคที่มีคำขยายหรือโครงสร้างแบบ nested ความสามารถของ RNN ทั่วไปยังมีข้อจำกัดในการจับบริบทได้ครบถ้วน
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-4">แนวทางแก้ไข</h3>
  <ul className="list-disc list-inside space-y-2 mb-6">
    <li>การใช้ LSTM (Long Short-Term Memory) และ GRU (Gated Recurrent Unit)</li>
    <li>การประยุกต์ใช้ Attention Mechanisms เพื่อจับบริบทระยะไกล</li>
    <li>การเปลี่ยนไปใช้ Transformer ซึ่งไม่มีข้อจำกัดของลำดับ</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside space-y-2">
    <li>Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory". Neural computation.</li>
    <li>Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". arXiv.</li>
    <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
  </ul>
</section>


     <section id="research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Research & References</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-10 text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <h3 className="text-xl font-semibold">แนวทางการศึกษาจากมหาวิทยาลัยชั้นนำ</h3>
    <p>
      การศึกษาด้าน Deep Learning ได้รับความสนใจอย่างกว้างขวางจากสถาบันวิจัยและมหาวิทยาลัยระดับโลก 
      โดยเฉพาะอย่างยิ่ง MIT, Stanford และ Carnegie Mellon ที่ได้ตีพิมพ์งานวิจัยมากมายครอบคลุมทั้งสถาปัตยกรรมโมเดล 
      การประยุกต์ใช้งาน และการวิเคราะห์เชิงทฤษฎี
    </p>

    <div className="bg-blue-500 dark:bg-blue-900/40 p-4 rounded-xl text-sm">
      <strong>Highlight:</strong> โครงการ Deep Learning จาก Stanford (CS231n) และ MIT (6.S191) ได้รับความนิยมในระดับโลก 
      เนื่องจากให้พื้นฐานทางทฤษฎีที่แข็งแกร่งและกรณีศึกษาเชิงลึกที่ครอบคลุมหลากหลายโดเมน
    </div>

    <h3 className="text-xl font-semibold">สรุปแหล่งข้อมูลอ้างอิงที่สำคัญ</h3>
    <p>
      แหล่งอ้างอิงที่เลือกใช้ในการสร้างเนื้อหาทั้งหมดของบทเรียนนี้มีความน่าเชื่อถือระดับสูง ครอบคลุมทั้งงานวิจัยและเนื้อหาการสอนในระดับบัณฑิตศึกษา
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>Stanford University - CS231n:</strong> Convolutional Neural Networks for Visual Recognition
      </li>
      <li>
        <strong>MIT - 6.S191:</strong> Introduction to Deep Learning, Spring 2020
      </li>
      <li>
        <strong>Oxford University - Deep Learning Lecture Series:</strong> โดย Prof. Nando de Freitas
      </li>
      <li>
        <strong>arXiv:</strong> Preprint repository ที่รวบรวมงานวิจัย cutting-edge ทั่วโลก เช่น Transformers, BERT, GANs
      </li>
      <li>
        <strong>IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)</strong>
      </li>
      <li>
        <strong>Nature Machine Intelligence</strong> และ <strong>Science Robotics</strong>
      </li>
    </ul>

    <h3 className="text-xl font-semibold">ประโยชน์ของการเรียนรู้จากแหล่งอ้างอิงระดับโลก</h3>
    <p>
      การอ้างอิงแหล่งข้อมูลที่น่าเชื่อถือและผ่านการ peer-reviewed ช่วยยืนยันความถูกต้องของเนื้อหา 
      และยังเปิดโอกาสให้ผู้เรียนสามารถขยายความเข้าใจเพิ่มเติมจากแหล่งต้นฉบับได้อย่างมั่นใจ
    </p>

    <table className="w-full table-auto border border-gray-500 text-sm mt-6">
      <thead>
        <tr className="bg-gray-500 text-white">
          <th className="px-4 py-2 text-left">แหล่งอ้างอิง</th>
          <th className="px-4 py-2 text-left">ประเภท</th>
          <th className="px-4 py-2 text-left">คุณสมบัติเด่น</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t border-gray-500">
          <td className="px-4 py-2">CS231n - Stanford</td>
          <td className="px-4 py-2">Course</td>
          <td className="px-4 py-2">เน้น CNN และ Vision</td>
        </tr>
        <tr className="border-t border-gray-500">
          <td className="px-4 py-2">6.S191 - MIT</td>
          <td className="px-4 py-2">Course</td>
          <td className="px-4 py-2">ครอบคลุม Deep Learning หลายประเภท</td>
        </tr>
        <tr className="border-t border-gray-500">
          <td className="px-4 py-2">arXiv</td>
          <td className="px-4 py-2">Preprints</td>
          <td className="px-4 py-2">เข้าถึงงานวิจัยล่าสุดได้ฟรี</td>
        </tr>
        <tr className="border-t border-gray-500">
          <td className="px-4 py-2">IEEE TPAMI</td>
          <td className="px-4 py-2">Journal</td>
          <td className="px-4 py-2">เน้นวิชาการด้าน Pattern Recognition</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-500 dark:bg-yellow-900/40 p-4 rounded-xl text-sm mt-6">
      <strong>Insight:</strong> การศึกษาเชิงลึกจากแหล่งข้อมูลต้นทาง 
      ส่งผลโดยตรงต่อคุณภาพการเรียนรู้ โดยเฉพาะการนำไปประยุกต์ใช้งานจริง 
      เช่นการพัฒนาโมเดล หรือการตีความผลลัพธ์ในบริบทใหม่ ๆ
    </div>

    <h3 className="text-xl font-semibold">แนวทางการต่อยอดและค้นคว้าเพิ่มเติม</h3>
    <p>
      สำหรับผู้สนใจสามารถใช้แหล่งอ้างอิงเหล่านี้เพื่อต่อยอดความรู้ เช่น การวิเคราะห์สถาปัตยกรรมใหม่อย่าง Vision Transformers (ViT), Diffusion Models 
      หรือ Neural Radiance Fields (NeRF) ที่กำลังได้รับความสนใจในช่วงหลัง
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>ViT:</strong> Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”
      </li>
      <li>
        <strong>Diffusion Models:</strong> Ho et al., “Denoising Diffusion Probabilistic Models”
      </li>
      <li>
        <strong>NeRF:</strong> Mildenhall et al., “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis”
      </li>
    </ul>
  </div>
</section>



<section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. กล่องความเข้าใจ (Insight Box)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="max-w-3xl mx-auto space-y-10 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">บทบาทของ Insight ในระบบ Deep Learning</h3>
    <p>
      ในระดับการศึกษาขั้นสูงด้าน Deep Learning "กล่องความเข้าใจ" ไม่ได้เป็นเพียงแค่เครื่องมือช่วยมองเห็น แต่ถูกใช้เป็นเครื่องมือเชิงกลยุทธ์เพื่อเสริมสร้างความเข้าใจทางปัญญา ปัจจุบันสถาบันอย่าง MIT และ Stanford ได้นำองค์ประกอบนี้มาใช้ในแพลตฟอร์มการเรียนรู้ เพื่อแปลงความซับซ้อนให้กลายเป็นความกระจ่าง
    </p>

    <div className="bg-yellow-500 rounded-xl p-6 border border-yellow-300 shadow-md">
      <h4 className="text-lg font-semibold mb-2">กล่องความเข้าใจ: คุณลักษณะที่ตีความได้ใน CNN</h4>
      <p>
        ข้อค้นพบสำคัญจากการแสดงภาพฟิลเตอร์ของ CNN คือเลเยอร์ต้น ๆ มักตรวจจับลวดลายระดับต่ำ เช่น ขอบหรือสี ในขณะที่เลเยอร์ลึกจะดึงคุณลักษณะเชิงนามธรรมและเฉพาะกลุ่มออกมา ลำดับชั้นเช่นนี้ทำให้ CNN สามารถทำงานด้านการมองเห็นได้อย่างแม่นยำ
      </p>
    </div>

    <h3 className="text-xl font-semibold">ประโยชน์ของกล่องความเข้าใจในการเรียนการสอน</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>กระตุ้นให้ผู้เรียนสนใจจุดสำคัญทางความรู้</li>
      <li>เชื่อมโยงทฤษฎีที่เป็นนามธรรมกับการประยุกต์ใช้งานจริง</li>
      <li>ปลุกความอยากรู้อยากเห็นและช่วยให้จดจำได้ดีขึ้นผ่านการออกแบบที่เน้นย้ำ</li>
      <li>ปรับปรุงประสบการณ์การเรียนรู้ให้เหมาะสมกับอุปกรณ์พกพา</li>
    </ul>

    <div className="bg-blue-500 rounded-xl p-6 border border-blue-300 shadow-md">
      <h4 className="text-lg font-semibold mb-2">กรณีศึกษา: ตัวอย่างจากมหาวิทยาลัย Carnegie Mellon</h4>
      <p>
        มหาวิทยาลัย Carnegie Mellon ใช้กล่องความเข้าใจในรายวิชา 11-785: Introduction to Deep Learning เพื่อแสดงข้อผิดพลาดในกรณีเฉพาะ ช่วยให้นักเรียนเข้าใจจุดอ่อนของโมเดลได้ชัดเจนยิ่งขึ้น
      </p>
    </div>

    <h3 className="text-xl font-semibold">หลักการออกแบบกล่องความเข้าใจให้รองรับ Responsive</h3>
    <p>
      โครงร่างของกล่องความเข้าใจควรอ่านง่ายและมีความสมดุลด้านภาพบนทุกขนาดหน้าจอ รวมถึงการปรับขนาดฟอนต์ ระยะห่าง (padding) และอัตราส่วนความต่างของสีให้เป็นไปตามมาตรฐานด้านการเข้าถึง
    </p>

    <table className="table-auto w-full border border-gray-500 mt-6">
      <thead>
        <tr className="bg-gray-500 text-white">
          <th className="px-4 py-2">แพลตฟอร์ม</th>
          <th className="px-4 py-2">ลักษณะการใช้งาน</th>
          <th className="px-4 py-2">จุดประสงค์</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800">
          <td className="border px-4 py-2">Stanford CS231n</td>
          <td className="border px-4 py-2">แสดงข้อผิดพลาดของโมเดลแทรกในเนื้อหา</td>
          <td className="border px-4 py-2">ช่วยให้เข้าใจจุดที่โมเดลล้มเหลว</td>
        </tr>
        <tr className="bg-gray-500 dark:bg-gray-700">
          <td className="border px-4 py-2">MIT 6.S191</td>
          <td className="border px-4 py-2">เปรียบเทียบทฤษฎีกับการใช้งานจริง</td>
          <td className="border px-4 py-2">ตอกย้ำวัตถุประสงค์ในการเรียนรู้</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800">
          <td className="border px-4 py-2">Harvard AI4ALL</td>
          <td className="border px-4 py-2">เปิดเผยอคติของโมเดล</td>
          <td className="border px-4 py-2">ส่งเสริมการคิดเชิงจริยธรรม</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-10">แหล่งข้อมูลอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside space-y-1">
      <li>
        Fei-Fei Li, Justin Johnson, Serena Yeung. Stanford CS231n: Convolutional Neural Networks for Visual Recognition
      </li>
      <li>
        MIT 6.S191: Introduction to Deep Learning. MIT OpenCourseWare
      </li>
      <li>
        IEEE Access Journal: Visualization of Deep Neural Networks: A Survey
      </li>
      <li>
        arXiv: Feature Visualization Techniques for CNN Interpretability (2022)
      </li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day46 theme={theme} />
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
        <ScrollSpy_Ai_Day46 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day46_RNNIntro;
