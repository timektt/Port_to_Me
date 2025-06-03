import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day47 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day47";
import MiniQuiz_Day47 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day47";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day47_RNNVariants = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day47_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day47_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day47_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day47_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day47_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day47_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day47_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day47_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day47_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day47_10").format("auto").quality("auto").resize(scale().width(501));

return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 47: RNN Variants – LSTM & GRU</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

      <section id="vanishing-gradient" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    1. จุดอ่อนของ RNN ดั้งเดิม: Vanishing Gradient
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <h3 className="text-xl font-semibold mb-4 mt-8">ความเข้าใจพื้นฐานของ RNN</h3>
  <p className="text-base leading-relaxed mb-4">
    Recurrent Neural Networks (RNNs) ได้รับการออกแบบมาเพื่อประมวลผลข้อมูลตามลำดับ เช่น ข้อความหรือสัญญาณเวลา โดยโครงสร้างหลักคือการส่งต่อสถานะ (hidden state) จากเวลาหนึ่งไปยังอีกเวลาหนึ่ง อย่างไรก็ตาม ความสามารถในการเรียนรู้ความสัมพันธ์ระยะยาวถูกจำกัดอย่างรุนแรงเนื่องจากปัญหา Vanishing Gradient
  </p>

  <h3 className="text-xl font-semibold mb-4 mt-8">นิยามของ Vanishing Gradient</h3>
  <p className="text-base leading-relaxed mb-4">
    ปัญหานี้เกิดขึ้นเมื่อทำการ backpropagation ในลำดับยาว ๆ โดย gradients ที่ไหลย้อนกลับจะค่อย ๆ เล็กลงจนเข้าใกล้ศูนย์ ทำให้การอัปเดตพารามิเตอร์ในเลเยอร์ต้น ๆ ไม่เกิดขึ้น ส่งผลให้โมเดลล้มเหลวในการจดจำข้อมูลในอดีตที่อยู่ไกล
  </p>

  <div className="bg-yellow-500 dark:bg-yellow-200 p-4 rounded-lg border-l-4 border-yellow-500 shadow-md mb-6">
    <p className="text-sm text-gray-800">
      Insight: งานวิจัยจาก Bengio et al. (1994) เป็นผู้ชี้ชัดปัญหา Vanishing Gradient และแนะนำว่าการใช้ sigmoid หรือ tanh activation บนลำดับที่ยาวเกินไปจะทำให้ gradients หายไปโดยอัตโนมัติ
    </p>
  </div>

  <h3 className="text-xl font-semibold mb-4 mt-8">การวิเคราะห์เชิงคณิตศาสตร์</h3>
  <p className="text-base leading-relaxed mb-4">
    สมมุติว่า loss function คือ L และ h<sub>t</sub> คือ hidden state ที่เวลา t, gradient ของ L ต่อพารามิเตอร์ W จะเป็นผลคูณของหลายค่า Jacobian:
  </p>

<div className="bg-gray-800 text-white p-4 rounded-lg text-sm overflow-x-auto mb-6">
  <code className="whitespace-pre">
    {`∇₍W₎ L = Σₜ (∂L/∂hₜ · ∏ₖ₌ₜ¹ ∂hₖ/∂hₖ₋₁ · ∂hₖ₋₁/∂W)`}
  </code>
</div>

  <p className="text-base leading-relaxed mb-4">
    ค่าผลคูณนี้จะมีแนวโน้มเข้าใกล้ศูนย์อย่างรวดเร็วหาก eigenvalues ของ Jacobian มีค่าน้อยกว่า 1 ซึ่งพบได้บ่อยเมื่อใช้ activation functions แบบ sigmoid หรือ tanh
  </p>

  <h3 className="text-xl font-semibold mb-4 mt-8">ผลกระทบในงานจริง</h3>
  <ul className="list-disc list-inside mb-6">
    <li>ทำให้โมเดลจดจำ context ระยะไกลไม่ได้</li>
    <li>ส่งผลต่อความแม่นยำในงานเช่นการแปลภาษาและการสร้างข้อความ</li>
    <li>ไม่สามารถเรียนรู้ dependencies ที่กินช่วงเวลานาน</li>
  </ul>

  <div className="bg-blue-500 dark:bg-blue-200 p-4 rounded-lg border-l-4 border-blue-500 shadow-md mb-6">
    <p className="text-sm text-gray-800">
      Highlight: แม้ว่า RNN ดั้งเดิมจะมีโครงสร้างที่เรียบง่าย แต่ปัญหา Vanishing Gradient ทำให้มันไม่สามารถขยายขอบเขตการประยุกต์ได้อย่างมีประสิทธิภาพ จึงนำไปสู่การคิดค้น LSTM และ GRU
    </p>
  </div>

  <h3 className="text-xl font-semibold mb-4 mt-8">ตารางเปรียบเทียบ Activation Functions</h3>
  <div className="overflow-auto mb-6">
    <table className="table-auto w-full text-sm text-left border border-gray-300 dark:border-gray-700">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-800">
          <th className="px-4 py-2 border">Function</th>
          <th className="px-4 py-2 border">Vanishing Gradient</th>
          <th className="px-4 py-2 border">Range</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border">Sigmoid</td>
          <td className="px-4 py-2 border">High</td>
          <td className="px-4 py-2 border">(0, 1)</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Tanh</td>
          <td className="px-4 py-2 border">Moderate</td>
          <td className="px-4 py-2 border">(-1, 1)</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">ReLU</td>
          <td className="px-4 py-2 border">Low</td>
          <td className="px-4 py-2 border">[0, ∞)</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mb-4 mt-8">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300">
    <li>Bengio et al., "Learning Long-Term Dependencies with Gradient Descent is Difficult", 1994 (arXiv:1211.5063)</li>
    <li>Stanford CS231n Lecture Notes: Recurrent Neural Networks</li>
    <li>Goodfellow et al., "Deep Learning", MIT Press, Chapter 10</li>
  </ul>
</section>


<section id="lstm" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    2. LSTM (Long Short-Term Memory) คืออะไร?
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="text-base leading-relaxed space-y-6">
    <h3 className="text-xl font-semibold">พื้นฐานของ LSTM</h3>
    <p>
      LSTM หรือ Long Short-Term Memory เป็นโครงสร้างของ Recurrent Neural Network (RNN) ที่ออกแบบมาเพื่อแก้ปัญหา vanishing gradient ที่มักพบใน RNN ทั่วไป โดย LSTM ถูกพัฒนาขึ้นในปี 1997 โดย Sepp Hochreiter และ Jürgen Schmidhuber เพื่อให้สามารถเรียนรู้ลำดับข้อมูลระยะยาวได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">กลไกภายในของ LSTM</h3>
    <p>
      LSTM ประกอบด้วยองค์ประกอบหลัก 3 ส่วน ได้แก่ Forget Gate, Input Gate และ Output Gate ซึ่งทำหน้าที่ควบคุมกระแสข้อมูลที่ไหลเข้าสู่ cell state หรือหน่วยความจำหลักของ LSTM
    </p>

    <div className="bg-blue-500 text-blue-900 p-4 rounded-lg">
      <strong>Insight:</strong> LSTM ใช้ cell state เป็นเส้นทางหลักในการเก็บข้อมูลระยะยาว โดยมี gating mechanisms คอยควบคุมการเพิ่มหรือลบข้อมูลแบบละเอียด
    </div>

    <h3 className="text-xl font-semibold">สูตรคณิตศาสตร์ของ LSTM</h3>
<div className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4 mb-6">
  <code className="whitespace-pre">
    {`fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
C̃ₜ = tanh(WC · [hₜ₋₁, xₜ] + bC)
Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ × tanh(Cₜ)`}
  </code>
</div>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบหลักของ LSTM</h3>
    <ul className="list-disc pl-6 space-y-1">
      <li>สามารถจัดการข้อมูลลำดับยาวได้โดยไม่สูญเสียบริบท</li>
      <li>ลดปัญหา gradient หายหรือ gradient ระเบิด</li>
      <li>รองรับการนำไปใช้กับงานหลากหลาย เช่น การแปลภาษา, การรู้จำเสียง</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ RNN กับ LSTM</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-left text-sm border border-gray-300">
        <thead>
          <tr className="bg-gray-500 dark:bg-gray-700">
            <th className="px-4 py-2 font-semibold">คุณสมบัติ</th>
            <th className="px-4 py-2 font-semibold">RNN ดั้งเดิม</th>
            <th className="px-4 py-2 font-semibold">LSTM</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t">
            <td className="px-4 py-2">รองรับลำดับยาว</td>
            <td className="px-4 py-2">ไม่ดี</td>
            <td className="px-4 py-2">ดีมาก</td>
          </tr>
          <tr className="border-t">
            <td className="px-4 py-2">Vanishing Gradient</td>
            <td className="px-4 py-2">เกิดขึ้นบ่อย</td>
            <td className="px-4 py-2">ลดลงอย่างมีนัยสำคัญ</td>
          </tr>
          <tr className="border-t">
            <td className="px-4 py-2">สถาปัตยกรรม</td>
            <td className="px-4 py-2">เรียบง่าย</td>
            <td className="px-4 py-2">ซับซ้อน มี gating</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">การใช้งานจริงในงานวิจัย</h3>
    <p>
      LSTM ถูกใช้อย่างแพร่หลายในงานวิจัยที่ต้องการบริบทระยะยาว เช่น การรู้จำเสียงพูด (speech recognition) โดย Google DeepMind และระบบแปลภาษาเชิงลึกโดย Facebook AI Research รวมถึงการสร้างระบบ chatbot และ music composition
    </p>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.</li>
      <li>Stanford CS231n Lecture Notes – Recurrent Neural Networks</li>
      <li>MIT Deep Learning Book – Chapter 10: Sequence Modeling</li>
      <li>Google AI Blog: Advances in Speech Recognition with LSTM</li>
    </ul>
  </div>
</section>


   <section id="lstm-architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Visual Flow: LSTM Architecture</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">โครงสร้างพื้นฐานของ LSTM</h3>
  <p className="text-base leading-relaxed mb-4">
    LSTM (Long Short-Term Memory) เป็นสถาปัตยกรรมประเภทหนึ่งของ RNN ที่ออกแบบมาเพื่อแก้ปัญหา vanishing gradient ในการเรียนรู้ระยะยาว โดยโครงสร้างของ LSTM จะเพิ่มกลไกควบคุมข้อมูลผ่านทาง gate ต่าง ๆ ซึ่งมีหน้าที่เฉพาะตัวในการจัดการกับข้อมูลที่ต้องจำไว้หรือข้อมูลที่ควรลืม
  </p>
  <ul className="list-disc list-inside text-base mb-6">
    <li>Forget Gate: ตัดสินใจว่าจะลบข้อมูลใดจากสถานะหน่วยความจำ</li>
    <li>Input Gate: ตัดสินใจว่าจะเพิ่มข้อมูลใหม่เข้าไปในหน่วยความจำหรือไม่</li>
    <li>Output Gate: ตัดสินใจว่าข้อมูลใดจะถูกส่งออกเป็นผลลัพธ์ในแต่ละช่วงเวลา</li>
  </ul>

  <div className="bg-blue-500 dark:bg-blue-900 rounded-lg p-4 mb-6">
    <p className="text-base leading-relaxed text-gray-900 dark:text-gray-500">
      <strong>Insight:</strong> กลไกของ LSTM ทำให้สามารถจดจำข้อมูลระยะยาวได้ดีขึ้น เหมาะสำหรับงานที่มีลำดับข้อมูล เช่น การแปลภาษา หรือการพยากรณ์แบบ sequence.
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">สมการภายในของ LSTM</h3>
  <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4 mb-6">
<code>{`f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t * C_{t-1} + i_t * C̃_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)`}</code>
  </pre>

  <h3 className="text-xl font-semibold mt-8 mb-4">เปรียบเทียบหน้าที่ของแต่ละ Gate</h3>
  <div className="overflow-x-auto mb-6">
    <table className="min-w-full table-auto text-sm border border-gray-300 dark:border-gray-600">
      <thead>
        <tr className="bg-gray-500 dark:bg-gray-700 text-white">
          <th className="px-4 py-2 text-left">Gate</th>
          <th className="px-4 py-2 text-left">Function</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Forget Gate</td>
          <td className="px-4 py-2">ลบข้อมูลเก่าออกจาก cell state</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Input Gate</td>
          <td className="px-4 py-2">เพิ่มข้อมูลใหม่เข้าไปใน cell state</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Output Gate</td>
          <td className="px-4 py-2">สร้าง hidden state ใหม่จาก cell state</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">การประยุกต์ใช้งานของ LSTM</h3>
  <ul className="list-disc list-inside text-base mb-6">
    <li>Speech Recognition: เช่น Google Voice</li>
    <li>Machine Translation: เช่น Google Translate</li>
    <li>Stock Prediction: การพยากรณ์ราคาหุ้น</li>
    <li>Music Generation: เช่นการสร้างโน้ตเพลง</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside text-base mb-6">
    <li>Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory". Neural Computation.</li>
    <li>Stanford University - CS224n: Natural Language Processing with Deep Learning</li>
    <li>MIT 6.S191: Introduction to Deep Learning</li>
    <li>arXiv preprint: A Critical Review of RNN Architectures</li>
  </ul>
</section>


 <section id="gru" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. GRU (Gated Recurrent Unit)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">แนวคิดพื้นฐานของ GRU</h3>
  <p className="text-base leading-relaxed mb-4">
    GRU (Gated Recurrent Unit) เป็นสถาปัตยกรรมที่พัฒนาขึ้นเพื่อปรับปรุงข้อจำกัดของ RNN แบบดั้งเดิม โดยมีโครงสร้างที่เรียบง่ายกว่าของ LSTM แต่ยังสามารถจดจำลำดับข้อมูลระยะยาวได้อย่างมีประสิทธิภาพ กลไก gating ของ GRU ทำให้สามารถเลือกได้ว่าข้อมูลจากอดีตควรถูกเก็บไว้หรือไม่ในแต่ละ timestep
  </p>
  <p className="text-base leading-relaxed mb-6">
    GRU มี gate หลักอยู่ 2 ประเภท ได้แก่ <strong>Update Gate</strong> และ <strong>Reset Gate</strong> ซึ่งช่วยควบคุมการไหลของข้อมูลระหว่างสถานะก่อนหน้าและปัจจุบันโดยไม่ต้องใช้ cell state แยกเหมือน LSTM
  </p>

  <div className="bg-yellow-500 dark:bg-yellow-900 rounded-lg p-4 mb-6">
    <p className="text-base leading-relaxed text-gray-900 dark:text-gray-500">
      <strong>Insight:</strong> จากงานวิจัยของ Cho et al. (2014) GRU มักแสดงประสิทธิภาพใกล้เคียงหรือดีกว่า LSTM โดยใช้พารามิเตอร์น้อยลง ทำให้เหมาะสำหรับโมเดลที่ต้องการความเร็วในการเรียนรู้หรือทรัพยากรจำกัด
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">สมการของ GRU</h3>
  <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4 mb-6">
<code>{`z_t = σ(W_z · [h_{t-1}, x_t])
r_t = σ(W_r · [h_{t-1}, x_t])
h̃_t = tanh(W_h · [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t`}</code>
  </pre>

  <h3 className="text-xl font-semibold mt-8 mb-4">ความแตกต่างระหว่าง GRU กับ LSTM</h3>
  <div className="overflow-x-auto mb-6">
    <table className="min-w-full table-auto text-sm border border-gray-300 dark:border-gray-600">
      <thead>
        <tr className="bg-gray-500 dark:bg-gray-700 text-white">
          <th className="px-4 py-2 text-left">คุณสมบัติ</th>
          <th className="px-4 py-2 text-left">GRU</th>
          <th className="px-4 py-2 text-left">LSTM</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">จำนวน Gate</td>
          <td className="px-4 py-2">2 (Update, Reset)</td>
          <td className="px-4 py-2">3 (Input, Forget, Output)</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">มี Cell State แยกหรือไม่</td>
          <td className="px-4 py-2">ไม่มี (ใช้ h_t แทน)</td>
          <td className="px-4 py-2">มี Cell State แยกจาก h_t</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">จำนวนพารามิเตอร์</td>
          <td className="px-4 py-2">น้อยกว่า</td>
          <td className="px-4 py-2">มากกว่า</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">ความเร็วในการเทรน</td>
          <td className="px-4 py-2">เร็วกว่า</td>
          <td className="px-4 py-2">ช้ากว่า</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">การประยุกต์ใช้ GRU ในโลกจริง</h3>
  <ul className="list-disc list-inside text-base mb-6">
    <li>Speech Recognition: เช่น Baidu Deep Speech ใช้ GRU แทน LSTM เพื่อความเร็ว</li>
    <li>Chatbot และ Dialogue Systems: สำหรับระบบที่ต้องการตอบสนองเร็ว</li>
    <li>Time Series Forecasting: เช่น พยากรณ์พลังงานหรือการจราจร</li>
    <li>Video Captioning: การแปลงวิดีโอเป็นข้อความโดยใช้ GRU เพื่อวิเคราะห์ลำดับภาพ</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-4">ข้อดี-ข้อจำกัดของ GRU</h3>
  <ul className="list-disc list-inside text-base mb-6">
    <li><strong>ข้อดี:</strong> โครงสร้างเรียบง่าย เทรนเร็ว และใช้หน่วยความจำน้อย</li>
    <li><strong>ข้อดี:</strong> ประสิทธิภาพใกล้เคียงหรือดีกว่า LSTM ในหลายงาน</li>
    <li><strong>ข้อจำกัด:</strong> อาจไม่เหมาะกับข้อมูลที่ต้องจดจำระยะยาวมาก ๆ เช่น บทความยาว</li>
    <li><strong>ข้อจำกัด:</strong> ความสามารถในการควบคุมสถานะภายในจำกัดกว่า LSTM</li>
  </ul>

  <div className="bg-blue-500 dark:bg-blue-900 rounded-lg p-4 mb-6">
    <p className="text-base leading-relaxed text-gray-900 dark:text-gray-500">
      <strong>Highlight:</strong> แม้ GRU จะมีโครงสร้างที่เบากว่า LSTM แต่ในบางกรณีเช่นใน low-resource environments หรือ real-time processing GRU เป็นตัวเลือกที่เหมาะสมกว่า
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside text-base mb-6">
    <li>Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. arXiv:1406.1078</li>
    <li>Stanford CS224n: NLP with Deep Learning (Lecture 6)</li>
    <li>MIT 6.S191: Deep Learning Lecture Notes</li>
    <li>IEEE Transactions on Neural Networks and Learning Systems</li>
  </ul>
</section>


 <section id="lstm-vs-gru" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. เปรียบเทียบ LSTM vs GRU</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="text-base leading-relaxed space-y-6">
    <p>
      ในโลกของโครงข่ายประสาทเทียมแบบลำดับ (Recurrent Neural Networks: RNNs) ทั้ง LSTM และ GRU ถือเป็นสถาปัตยกรรมที่ได้รับความนิยมสูงสุดเนื่องจากสามารถจัดการกับปัญหา vanishing gradients ได้อย่างมีประสิทธิภาพ โดยเฉพาะในงานที่เกี่ยวข้องกับการเรียนรู้ลำดับข้อมูลระยะยาว เช่น การแปลภาษา การรู้จำเสียงพูด และการพยากรณ์ลำดับข้อมูลเชิงเวลา
    </p>

    <h3 className="text-xl font-semibold mt-10">5.1 โครงสร้างภายในของ LSTM และ GRU</h3>
    <p>
      LSTM ใช้กลไกที่ซับซ้อนกว่าด้วยการแยกหน่วยความจำ (cell state) และการควบคุมด้วย gate ทั้งสาม ได้แก่ input gate, forget gate และ output gate ในขณะที่ GRU รวมกลไกเหล่านี้ให้เรียบง่ายขึ้นโดยใช้เพียง update gate และ reset gate
    </p>
    <div className="bg-blue-500 rounded-lg p-4">
      <p className="font-semibold">Highlight:</p>
      <p>
        GRU มีโครงสร้างเบากว่า LSTM ทำให้เหมาะกับระบบที่ต้องการความเร็วในการประมวลผลหรือมีข้อจำกัดด้านทรัพยากร
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">5.2 การเรียนรู้ระยะยาวและประสิทธิภาพเชิงเวลา</h3>
    <ul className="list-disc list-inside">
      <li>LSTM เหมาะสมกับลำดับข้อมูลที่ต้องจดจำในระยะยาว เช่น ประโยคที่มีโครงสร้างซับซ้อน</li>
      <li>GRU ทำงานได้ดีในลำดับข้อมูลที่สั้นหรือมีโครงสร้างที่ไม่ซับซ้อนมาก</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">5.3 ตารางเปรียบเทียบเชิงเทคนิค</h3>
    <div className="overflow-x-auto">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300">
        <thead className="bg-gray-500">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">LSTM</th>
            <th className="border px-4 py-2">GRU</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">จำนวน Gates</td>
            <td className="border px-4 py-2">3 (input, forget, output)</td>
            <td className="border px-4 py-2">2 (update, reset)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">หน่วยความจำแยกต่างหาก (cell state)</td>
            <td className="border px-4 py-2">มี</td>
            <td className="border px-4 py-2">ไม่มี</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความเร็วในการฝึก</td>
            <td className="border px-4 py-2">ช้ากว่า</td>
            <td className="border px-4 py-2">เร็วกว่า</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ประสิทธิภาพเชิงพยากรณ์</td>
            <td className="border px-4 py-2">ดีกว่าในลำดับยาว</td>
            <td className="border px-4 py-2">ดีกว่าในลำดับสั้น</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">5.4 การใช้งานในโลกจริง</h3>
    <p>
      LSTM มักถูกใช้ในงานที่ต้องการความสามารถในการจดจำระยะยาว เช่น การแปลภาษาอัตโนมัติและระบบแนะนำ (recommendation systems) ส่วน GRU นิยมในแอปพลิเคชันที่มีข้อจำกัดด้านทรัพยากร เช่น ระบบ edge computing และ embedded systems
    </p>

    <div className="bg-yellow-500 rounded-lg p-4">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ในงานที่ต้องเลือกใช้งานจริง นักวิจัยมักแนะนำให้เริ่มด้วย GRU และเปรียบเทียบกับ LSTM ด้วย validation performance เนื่องจาก GRU มีต้นทุนการฝึกที่ต่ำกว่า
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">5.5 บทสรุปทางวิชาการ</h3>
    <p>
      แม้ LSTM จะสามารถเรียนรู้ dependencies ระยะยาวได้ลึกกว่า GRU แต่ GRU ก็สามารถให้ผลลัพธ์ที่ใกล้เคียงกันในหลายสถานการณ์ ทั้งนี้การเลือกใช้งานควรขึ้นอยู่กับลักษณะของชุดข้อมูล และข้อจำกัดด้านเวลาและทรัพยากรในการฝึก
    </p>

    <h3 className="text-xl font-semibold mt-10">5.6 แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside">
      <li>Chung, Junyoung, et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." arXiv preprint arXiv:1412.3555 (2014).</li>
      <li>Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-Term Memory." Neural computation 9.8 (1997): 1735-1780.</li>
      <li>Greff, Klaus, et al. "LSTM: A Search Space Odyssey." IEEE Transactions on Neural Networks and Learning Systems (2017).</li>
    </ul>
  </div>
</section>


        <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Use Cases จากงานวิจัยจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  
  <div className="space-y-10 text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <h3 className="text-xl font-semibold">การประมวลผลภาษาธรรมชาติ (Natural Language Processing)</h3>
    <p>
      GRU และ LSTM ถูกนำมาใช้กันอย่างแพร่หลายในงานด้านภาษาธรรมชาติ เช่น การแปลภาษาอัตโนมัติ การสรุปความ และการตอบคำถามอัตโนมัติ
      ตัวอย่างหนึ่งคือการใช้ LSTM ใน Google Neural Machine Translation (GNMT) ซึ่งสามารถลดอัตราความผิดพลาดในการแปลได้อย่างมีนัยสำคัญ
    </p>
    <div className="bg-yellow-500 p-4 rounded-md">
      <p className="font-semibold">Insight:</p>
      <p>
        LSTM ช่วยให้ระบบแปลภาษาเข้าใจบริบทที่ครอบคลุมระยะไกลได้ดีขึ้น ซึ่งเป็นข้อได้เปรียบเหนือ RNN แบบดั้งเดิมอย่างมาก
      </p>
    </div>

    <h3 className="text-xl font-semibold">การรู้จำเสียงพูด (Speech Recognition)</h3>
    <p>
      งานวิจัยของ Baidu Deep Speech และ Google Voice Search ใช้ GRU และ LSTM สำหรับการแปลงคลื่นเสียงเป็นข้อความ
      โดย LSTM มีความสามารถในการเก็บข้อมูลเสียงที่มีลักษณะต่อเนื่องและผันแปรได้ดี
    </p>
    <div className="bg-blue-500 p-4 rounded-md">
      <p className="font-semibold">Highlight:</p>
      <p>
        Baidu Deep Speech ใช้ RNN แบบ LSTM ร่วมกับ Connectionist Temporal Classification (CTC) เพื่อให้สามารถจัดการกับลำดับเสียงที่ไม่มีการจัดแนวได้อย่างแม่นยำ
      </p>
    </div>

    <h3 className="text-xl font-semibold">การวิเคราะห์ทางการเงินและเศรษฐกิจ</h3>
    <p>
      ในการคาดการณ์แนวโน้มราคาหุ้นและสภาพเศรษฐกิจที่มีความไม่แน่นอนสูง งานวิจัยจาก Oxford และ Harvard แสดงให้เห็นว่า GRU สามารถให้ผลลัพธ์ที่มีประสิทธิภาพดีกว่า LSTM ในบางกรณี โดยเฉพาะเมื่อต้องฝึกในชุดข้อมูลขนาดเล็กหรือมี noise สูง
    </p>

    <h3 className="text-xl font-semibold">การประเมินความเสี่ยงด้านสุขภาพ</h3>
    <p>
      งานวิจัยจาก Stanford และ MIT ใช้ GRU ในการวิเคราะห์ประวัติการแพทย์เพื่อพยากรณ์ความเสี่ยงของผู้ป่วย เช่น ความเสี่ยงต่อภาวะหัวใจล้มเหลวหรือเบาหวานจากข้อมูล EHR (Electronic Health Records)
    </p>
    <ul className="list-disc list-inside mt-4">
      <li>ข้อมูลขนาดใหญ่ที่มีลักษณะลำดับ เช่น ความดัน, อัตราการเต้นหัวใจ</li>
      <li>ข้อมูลไม่สมบูรณ์หรือขาดหายบางช่วง — GRU ช่วยลดผลกระทบจากข้อมูลขาด</li>
    </ul>

    <h3 className="text-xl font-semibold">การทำนายลำดับเหตุการณ์ในระบบ IoT</h3>
    <p>
      ในบริบทของ Smart Cities และระบบอุตสาหกรรมอัจฉริยะ งานจาก IEEE Internet of Things Journal ได้แสดงให้เห็นว่า LSTM สามารถเรียนรู้ลำดับข้อมูลจากอุปกรณ์เซ็นเซอร์ที่หลากหลายและผันผวนได้ดี โดยใช้ในการตรวจจับเหตุการณ์ผิดปกติ เช่น ไฟฟ้าขัดข้อง, อุบัติเหตุ หรือการโจมตีไซเบอร์
    </p>

    <h3 className="text-xl font-semibold">บทสรุปและการประยุกต์ใช้ในอนาคต</h3>
    <p>
      ทั้ง GRU และ LSTM มีจุดแข็งเฉพาะตัว การเลือกใช้งานต้องพิจารณาความซับซ้อนของข้อมูล, ปริมาณข้อมูล, และทรัพยากรการประมวลผลที่มีอยู่
    </p>
    <table className="table-auto w-full mt-6 text-left border border-gray-300 dark:border-gray-700">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-800">
          <th className="px-4 py-2">Use Case</th>
          <th className="px-4 py-2">Best Choice</th>
          <th className="px-4 py-2">เหตุผล</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2">NLP (Translation)</td>
          <td className="px-4 py-2">LSTM</td>
          <td className="px-4 py-2">เข้าใจบริบทระยะยาวได้ดี</td>
        </tr>
        <tr>
          <td className="px-4 py-2">เสียงพูด</td>
          <td className="px-4 py-2">GRU</td>
          <td className="px-4 py-2">ประหยัดพารามิเตอร์, ฝึกได้เร็ว</td>
        </tr>
        <tr>
          <td className="px-4 py-2">สุขภาพ</td>
          <td className="px-4 py-2">GRU</td>
          <td className="px-4 py-2">รองรับข้อมูลไม่สมบูรณ์</td>
        </tr>
        <tr>
          <td className="px-4 py-2">IoT & Event Detection</td>
          <td className="px-4 py-2">LSTM</td>
          <td className="px-4 py-2">จับลำดับสัญญาณที่ซับซ้อนได้ดี</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-10">อ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Cho, K., et al. "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation." arXiv:1406.1078</li>
      <li>Hannun, A., et al. "Deep Speech: Scaling up end-to-end speech recognition." arXiv:1412.5567</li>
      <li>Lipton, Z. C., et al. "Learning to diagnose with LSTM recurrent neural networks." arXiv:1511.03677</li>
      <li>IEEE Internet of Things Journal, 2021. "Anomaly Detection in Smart City Infrastructure using Deep Sequence Models."</li>
    </ul>
  </div>
</section>


        <section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Visualization: GRU vs LSTM</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-10 text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <h3 className="text-xl font-semibold">พื้นฐานของ Visualization ในโมเดล RNN</h3>
    <p>
      การทำความเข้าใจกลไกการทำงานของ GRU และ LSTM ผ่าน Visualization เป็นวิธีที่ช่วยให้ตีความพฤติกรรมของโมเดลได้ชัดเจนมากขึ้น โดยเฉพาะการวิเคราะห์ hidden state dynamics, gating mechanisms และ temporal attention ซึ่งมักถูกใช้ในงานวิจัยด้าน Explainable AI (XAI) จาก Stanford และ MIT
    </p>

    <div className="bg-yellow-500 dark:bg-yellow-900 p-4 rounded-lg border-l-4 border-yellow-500 dark:border-yellow-400">
      <p className="font-medium">Insight:</p>
      <p>
        งานวิจัยของ Karpathy et al. แสดงให้เห็นว่า neuron บางตัวใน LSTM สามารถเรียนรู้การตรวจจับ pattern ที่มีความหมายทางภาษาศาสตร์ เช่น การจับคำสั่งหรือจุดสิ้นสุดของวลี
      </p>
    </div>

    <h3 className="text-xl font-semibold">เทคนิค Visualization ที่นิยมใช้</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Heatmaps:</strong> ใช้แสดงความเข้มข้นของการเปิด/ปิด gate</li>
      <li><strong>Hidden State Projection:</strong> ใช้ PCA หรือ t-SNE แสดง trajectory ของ hidden states ใน vector space</li>
      <li><strong>Attention Maps:</strong> สำหรับกรณี hybrid models เช่น GRU+Attention</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Visualization Features</h3>
    <div className="overflow-x-auto">
      <table className="table-auto border border-gray-300 dark:border-gray-600 w-full text-left text-sm">
        <thead>
          <tr className="bg-gray-500 text-white">
            <th className="p-2 border">Aspect</th>
            <th className="p-2 border">GRU</th>
            <th className="p-2 border">LSTM</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t">
            <td className="p-2 border">จำนวน Gate</td>
            <td className="p-2 border">2 (Update, Reset)</td>
            <td className="p-2 border">3 (Input, Forget, Output)</td>
          </tr>
          <tr>
            <td className="p-2 border">ความซับซ้อนของ Hidden State</td>
            <td className="p-2 border">ต่ำกว่า</td>
            <td className="p-2 border">สูงกว่า</td>
          </tr>
          <tr>
            <td className="p-2 border">Interpretability</td>
            <td className="p-2 border">ชัดเจนกว่าในการ Visualize</td>
            <td className="p-2 border">ต้องใช้เทคนิคขั้นสูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการ Visualize Hidden States</h3>
    <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
<code>{`import matplotlib.pyplot as plt
import numpy as np

# Simulate hidden state trajectories
hidden_states = np.cumsum(np.random.randn(50), axis=0)
plt.plot(hidden_states)
plt.title("Hidden State Trajectory")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.grid(True)
plt.show()`}</code>
    </pre>

    <div className="bg-blue-500 dark:bg-blue-900 p-4 rounded-lg border-l-4 border-blue-500 dark:border-blue-400">
      <p className="font-medium">Highlight:</p>
      <p>
        การ visual hidden activation ในแต่ละ timestep ช่วยวิเคราะห์ว่า neuron ใดกำลังจดจำหรือมองข้ามข้อมูลใดในลำดับเวลา
      </p>
    </div>

    <h3 className="text-xl font-semibold">ข้อสังเกตจากงานวิจัย</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>
        LSTM มักแสดงพฤติกรรม memory retention ได้ดีกว่าในงานที่มี sequence ยาว เช่น machine translation
      </li>
      <li>
        GRU มักให้ผลลัพธ์ใกล้เคียงกันในงาน classification แต่ใช้พลังคำนวณน้อยกว่า
      </li>
      <li>
        Visualization ของ GRU ง่ายกว่าสำหรับ interpretability ในแง่ educational usage
      </li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>J. Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", arXiv 2014</li>
      <li>K. Greff et al., "LSTM: A Search Space Odyssey", IEEE TPAMI 2017</li>
      <li>Chris Olah, "Understanding LSTM Networks", colah.github.io</li>
    </ul>
  </div>
</section>


       <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Research & Global References</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-10 text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <h3 className="text-xl font-semibold">แหล่งงานวิจัยระดับโลกเกี่ยวกับ GRU และ LSTM</h3>
    <p>
      ตลอดสองทศวรรษที่ผ่านมา โครงข่ายลำดับเช่น GRU และ LSTM ได้กลายเป็นรากฐานของการประมวลผลภาษาธรรมชาติและลำดับข้อมูลอื่น ๆ โดยมีมหาวิทยาลัยชั้นนำและองค์กรวิจัยระดับโลกทำการศึกษาและปรับปรุงโครงสร้างเหล่านี้อย่างต่อเนื่อง โดยเฉพาะอย่างยิ่งในด้านการแปลภาษา, การรู้จำเสียงพูด, และการพยากรณ์ลำดับเวลา
    </p>

    <h3 className="text-xl font-semibold">สถาบันที่มีบทบาทสูง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Stanford NLP Group — วิจัยโครงข่ายลำดับและการวิเคราะห์ข้อความในหลายภาษา</li>
      <li>MIT CSAIL — วิจัยการพัฒนา GRU สำหรับการใช้งานจริงด้านหุ่นยนต์</li>
      <li>Oxford Machine Learning Group — สำรวจการเปรียบเทียบ GRU และ LSTM สำหรับลำดับภาพและวิดีโอ</li>
      <li>Google Brain — พัฒนาโครงข่ายลำดับในงาน Google Translate และ Speech-to-Text</li>
      <li>Facebook AI Research (FAIR) — วิจัย GRU แบบ Multilingual และ Multitask</li>
    </ul>

    <h3 className="text-xl font-semibold">แนวโน้มการตีพิมพ์ในวารสารและการประชุม</h3>
    <p>
      ผลงานจำนวนมากเกี่ยวกับ GRU และ LSTM ได้รับการตีพิมพ์ในวารสารวิชาการและงานประชุมระดับสูง เช่น NeurIPS, ICML, ACL, และ IEEE Transactions on Neural Networks and Learning Systems ซึ่งแสดงให้เห็นถึงความต่อเนื่องและความลึกของการพัฒนาโครงสร้างลำดับเหล่านี้
    </p>

    <div className="bg-yellow-500 p-4 rounded-xl border-l-4 border-yellow-400">
      <h4 className="font-semibold mb-2">Insight Box: คำแนะนำในการศึกษาเพิ่มเติม</h4>
      <p>
        เพื่อเข้าใจเบื้องลึกของสถาปัตยกรรม GRU และ LSTM ควรอ่านบทความจาก arXiv และ IEEE Access ที่เปรียบเทียบเชิงโครงสร้างและประสิทธิภาพในบริบทต่าง ๆ
      </p>
    </div>

    <h3 className="text-xl font-semibold">บทความอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Cho et al. (2014). "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation." arXiv preprint arXiv:1406.1078</li>
      <li>Greff et al. (2017). "LSTM: A Search Space Odyssey." IEEE Transactions on Neural Networks and Learning Systems</li>
      <li>Sherstinsky, A. (2020). "Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network." Physica A</li>
      <li>Olah, C. (2015). "Understanding LSTM Networks." Blog post, colah.github.io</li>
      <li>Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." Neural Computation, MIT Press</li>
      <li>Bahdanau et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv:1409.0473</li>
    </ul>

    <div className="bg-blue-500 p-4 rounded-xl border-l-4 border-blue-500">
      <h4 className="font-semibold mb-2">Highlight Box: ความสำคัญของการอ้างอิงงานวิจัย</h4>
      <p>
        การสร้างความเข้าใจที่ถูกต้องและลึกซึ้งใน Deep Learning ต้องอาศัยการเรียนรู้จากต้นฉบับงานวิจัยที่ได้รับการยอมรับจากสถาบันและนักวิจัยชั้นนำทั่วโลก ซึ่งช่วยลดการเข้าใจผิด และทำให้สามารถนำไปประยุกต์ใช้งานได้จริงอย่างมีประสิทธิภาพ
      </p>
    </div>
  </div>
</section>


 <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. กล่องข้อมูลเชิงลึก (Insight Box)</h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-10 text-base leading-relaxed text-gray-800 dark:text-gray-200">
    <h3 className="text-xl font-semibold">การทำความเข้าใจบทบาทของข้อมูลเชิงลึกด้านสถาปัตยกรรม</h3>
    <p>
      ในงานวิจัยด้าน Deep Learning การดึงข้อมูลเชิงลึกจากรูปแบบของสถาปัตยกรรมเป็นองค์ประกอบสำคัญต่อการพัฒนาโมเดลให้มีประสิทธิภาพ ข้อมูลเชิงลึกเหล่านี้ก้าวข้ามตัวชี้วัดเชิงประจักษ์ โดยให้ความเข้าใจเชิงคุณภาพเกี่ยวกับพฤติกรรมของโมเดล ความสามารถในการทั่วไป และความทนทานต่อบริบทที่หลากหลาย
    </p>

    <div className="bg-yellow-500 dark:bg-yellow-200 rounded-xl p-5 border border-yellow-400">
      <p className="font-medium text-yellow-900 dark:text-yellow-950">
        จุดเน้น: ห้องปฏิบัติการ Stanford AI ชี้ว่า "สัญชาตญาณทางสถาปัตยกรรม" ช่วยเร่งการออกแบบและดีบักโมเดลในงาน NLP, Computer Vision และ Multi-modal ได้อย่างมีประสิทธิภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold">บทเรียนสถาปัตยกรรมข้ามโดเมน</h3>
    <p>
      โมเดลที่มีผลกระทบสูงหลายตัวได้ถ่ายทอดนวัตกรรมด้านสถาปัตยกรรมร่วมกัน เช่น:
    </p>
    <ul className="list-disc list-inside pl-4">
      <li>Residual Connections จาก ResNet จุดประกายการพัฒนาใน Transformer (เช่น LayerNorm skip)</li>
      <li>Recurrent Dropout จาก RNN ส่งผลต่อเทคนิค Regularization ของ Attention</li>
      <li>การเปรียบเทียบระหว่าง Batch Normalization กับ Layer Normalization ส่งผลต่อเสถียรภาพการฝึกโมเดลในหลายสถาปัตยกรรม</li>
    </ul>

    <h3 className="text-xl font-semibold">การตีความโครงสร้างภายในโมเดล</h3>
    <p>
      แม้โมเดลอย่าง LSTM และ GRU จะมีโครงสร้างภายในซับซ้อน แต่กลับให้กลไก Gating ที่สามารถตีความได้ ซึ่งการเข้าใจโครงสร้างเหล่านี้ช่วยให้วิเคราะห์ความสัมพันธ์ของข้อมูลตามลำดับเวลา และความไวต่ออินพุตได้อย่างมีระบบ
    </p>

    <div className="bg-blue-500 dark:bg-blue-200 rounded-xl p-5 border border-blue-400">
      <p className="font-medium text-blue-900 dark:text-blue-950">
        ข้อมูลเชิงลึก: งานวิจัยจาก MIT (NeurIPS 2022) พบว่า การ Visualize ค่า Hidden State ของ GRU ให้ผลวิเคราะห์ที่ดีกว่าโมเดล Attention แบบ Black-box
      </p>
    </div>

    <h3 className="text-xl font-semibold">การชั่งน้ำหนักระหว่างสถาปัตยกรรมและการตัดสินใจ</h3>
    <p>
      วิศวกร AI มักต้องตัดสินใจระหว่างความเรียบง่ายของโมเดลกับประสิทธิภาพ GRU เป็นตัวอย่างที่ดี เพราะสามารถให้ประสิทธิภาพใกล้เคียงกับ LSTM แต่ใช้พารามิเตอร์น้อยกว่า จึงเหมาะกับแอปพลิเคชันที่ต้องการความเร็วในการตอบสนอง
    </p>

    <div className="overflow-x-auto mt-6 rounded-lg border border-gray-300 dark:border-gray-600">
      <table className="min-w-full text-sm text-left table-auto">
        <thead className="bg-gray-500 text-white dark:bg-gray-700">
          <tr>
            <th className="border px-4 py-2 dark:border-gray-600">สถาปัตยกรรม</th>
            <th className="border px-4 py-2 dark:border-gray-600">จำนวนพารามิเตอร์</th>
            <th className="border px-4 py-2 dark:border-gray-600">ความเร็วในการฝึก</th>
            <th className="border px-4 py-2 dark:border-gray-600">ความสามารถในการตีความ</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-black/30">
          <tr className="dark:hover:bg-black/20 hover:bg-gray-100 transition">
            <td className="border px-4 py-2 dark:border-gray-600">LSTM</td>
            <td className="border px-4 py-2 dark:border-gray-600">~1.3M</td>
            <td className="border px-4 py-2 dark:border-gray-600">ปานกลาง</td>
            <td className="border px-4 py-2 dark:border-gray-600">มี Gating ที่ชัดเจน</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-black/10 dark:hover:bg-black/20 hover:bg-gray-100 transition">
            <td className="border px-4 py-2 dark:border-gray-600">GRU</td>
            <td className="border px-4 py-2 dark:border-gray-600">~0.9M</td>
            <td className="border px-4 py-2 dark:border-gray-600">เร็วกว่า</td>
            <td className="border px-4 py-2 dark:border-gray-600">กลไก Gating ที่ง่ายกว่า</td>
          </tr>
          <tr className="dark:hover:bg-black/20 hover:bg-gray-100 transition">
            <td className="border px-4 py-2 dark:border-gray-600">Transformer</td>
            <td className="border px-4 py-2 dark:border-gray-600">~10M+</td>
            <td className="border px-4 py-2 dark:border-gray-600">ช้ากว่า</td>
            <td className="border px-4 py-2 dark:border-gray-600">ตีความยาก (หากไม่ใช้ probing)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิงที่เลือกใช้</h3>
    <ul className="list-disc list-inside pl-4">
      <li>Stanford CS231n Lecture Notes, ฉบับปี 2023</li>
      <li>MIT Deep Learning Book โดย Ian Goodfellow et al., ฉบับที่ 2</li>
      <li>"Understanding LSTM Networks" – โดย Olah (Google Brain)</li>
      <li>IEEE Transactions on Neural Networks, 2021</li>
      <li>arXiv:2303.15991 - การตีความ Attention Pathway ใน GRU และ LSTM</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day47 theme={theme} />
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
        <ScrollSpy_Ai_Day47 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day47_RNNVariants;
