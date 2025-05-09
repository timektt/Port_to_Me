import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day27 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day27";
import MiniQuiz_Day27 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day27";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day27_LSTM = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("LSTM1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("LSTM2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("LSTM3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("LSTM4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("LSTM5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("LSTM6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("LSTM7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("LSTM8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("LSTM9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("LSTM10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("LSTM11").format("auto").quality("auto").resize(scale().width(600));
  const img12 = cld.image("LSTM12").format("auto").quality("auto").resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 27: Long Short-Term Memory (LSTM)</h1>
          <div className="flex justify-center my-6">
                <AdvancedImage cldImg={img1} />
            </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          {/** Section 1 */}
<section id="introduction" className="mb-16 scroll-mt-25 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    1. Introduction: ทำไมต้องมี LSTM?
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Long Short-Term Memory (LSTM) คือโครงข่ายประสาทเทียมประเภทหนึ่งที่พัฒนาขึ้นเพื่อแก้ปัญหาสำคัญของ Recurrent Neural Networks (RNNs) ซึ่งมักประสบกับปัญหา vanishing gradients ทำให้ไม่สามารถจดจำบริบทระยะยาวได้อย่างมีประสิทธิภาพ แม้ว่า RNN จะสามารถจัดการกับข้อมูลที่มีลำดับได้ แต่เมื่อ sequence มีความยาวมาก gradients ที่ส่งย้อนกลับจะค่อย ๆ หายไป ส่งผลให้โมเดลลืมข้อมูลในอดีต
    </p>

    <p>
      งานวิจัยของ Sepp Hochreiter และ Jürgen Schmidhuber ในปี 1997 ได้เสนอ LSTM เป็นวิธีแก้ปัญหานี้ โดยแนะนำแนวคิด cell state ซึ่งสามารถเก็บข้อมูลผ่านช่วงเวลาที่ยาวได้ ด้วยกลไกของ gates ที่ควบคุมการลืม การจดจำ และการส่งข้อมูลออก ทำให้โมเดลสามารถเรียนรู้ long-term dependencies ได้ดีขึ้น
    </p>

    <p>
      รายวิชา CS224N ของ Stanford และ Deep Learning Specialization ของ Andrew Ng (Coursera) ยืนยันว่า LSTM ได้กลายเป็นพื้นฐานสำคัญในงานด้าน Natural Language Processing, Time Series Forecasting และ Speech Recognition
    </p>

    <h3 className="text-xl font-semibold">ปัญหา Vanishing Gradient ใน RNN</h3>
    <p>
      ใน RNN ปกติ การคำนวณย้อนกลับในลำดับยาวจะทำให้ gradients ลดลงแบบเอ็กซ์โปเนนเชียล เมื่อ gradients มีค่าต่ำเกินไป โมเดลจะไม่สามารถปรับน้ำหนักได้อย่างมีประสิทธิภาพ ส่งผลให้ข้อมูลเก่าไม่สามารถนำมาพิจารณาได้เมื่อเวลาผ่านไปนาน
    </p>

    <p>
      ปัญหานี้ถูกเน้นย้ำในงานวิจัยของ Bengio et al. (1994) ซึ่งแสดงให้เห็นว่าเมื่อใช้ activation function เช่น sigmoid หรือ tanh การแพร่ของ gradient ผ่านเวลาจะยิ่งยากขึ้น เมื่อค่าต่ำมากจะไม่สามารถอัปเดตพารามิเตอร์ในช่วงเวลาเริ่มต้นได้เลย
    </p>

    <h3 className="text-xl font-semibold">LSTM เข้ามาแก้ปัญหาอย่างไร?</h3>
    <p>
      LSTM ได้รับการออกแบบด้วย cell state และ 3 gates ได้แก่ Forget Gate, Input Gate, และ Output Gate ซึ่งช่วยให้โมเดลสามารถควบคุมได้อย่างเป็นระบบว่าอะไรควรถูกจดจำ ลืม หรือส่งออกไปในแต่ละ timestep การออกแบบนี้ทำให้สามารถเรียนรู้ความสัมพันธ์ระยะยาวโดยไม่สูญเสีย gradients
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Forget Gate:</strong> กำหนดว่าข้อมูลจาก timestep ก่อนหน้าส่วนใดควรถูกลืม
      </li>
      <li>
        <strong>Input Gate:</strong> เลือกว่าข้อมูลใหม่จาก timestep ปัจจุบันส่วนใดควรเข้าสู่หน่วยความจำ
      </li>
      <li>
        <strong>Cell State:</strong> ทำหน้าที่เป็นเส้นทางหลักของการไหลของข้อมูลแบบต่อเนื่อง
      </li>
      <li>
        <strong>Output Gate:</strong> ตัดสินใจว่าจะส่งข้อมูลใดออกมาเป็น hidden state
      </li>
    </ul>

    <p>
      ด้วยความสามารถในการจัดการบริบทที่ห่างไกล LSTM จึงกลายเป็นตัวเลือกหลักในหลายงาน เช่น
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Language Modeling (e.g., GPT ยุคแรกก่อน Transformer)</li>
      <li>Machine Translation (e.g., Google Translate เวอร์ชันก่อน Transformer)</li>
      <li>Speech Recognition (e.g., DeepSpeech โดย Baidu)</li>
      <li>Time Series Forecasting (เช่น การคาดการณ์ราคาหุ้น พลังงาน หรืออุณหภูมิ)</li>
    </ul>

    <h3 className="text-xl font-semibold">ความแตกต่างหลักระหว่าง RNN กับ LSTM</h3>

    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">RNN ปกติ</th>
          <th className="border px-4 py-2">LSTM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การจัดการระยะยาว</td>
          <td className="border px-4 py-2">จำได้ยาก</td>
          <td className="border px-4 py-2">จำได้นาน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ปัญหา vanishing gradients</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">ลดลงมาก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความซับซ้อนโครงสร้าง</td>
          <td className="border px-4 py-2">ง่าย</td>
          <td className="border px-4 py-2">ซับซ้อนขึ้น</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การฝึก</td>
          <td className="border px-4 py-2">เร็วกว่า</td>
          <td className="border px-4 py-2">เสถียรกว่า</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>LSTM คือทางออกที่แท้จริงต่อ long-term dependencies</li>
        <li>Cell state ทำให้ข้อมูลสามารถไหลต่อเนื่องโดยไม่สูญเสียความสำคัญ</li>
        <li>Gating mechanism ช่วยให้โมเดลตัดสินใจอย่างมีเหตุผลว่าควรเก็บ ลืม หรือส่งอะไรออกไป</li>
        <li>สามารถขยายเป็น BiLSTM, Stacked LSTM และ CuDNN LSTM สำหรับงานขนาดใหญ่</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      LSTM คือหนึ่งในโครงสร้างสำคัญของยุค Deep Learning ที่สามารถแก้ปัญหา vanishing gradients และความจำระยะยาวในลำดับข้อมูลได้อย่างมีประสิทธิภาพ
      โดยเฉพาะในงานที่ต้องการเข้าใจลำดับเวลาและบริบทอย่างลึกซึ้ง โครงสร้างที่ยืดหยุ่นและความสามารถในการควบคุมความจำทำให้ LSTM ยังคงเป็นเครื่องมือสำคัญทั้งในงานวิจัยและการใช้งานจริง
    </p>

  </div>
</section>


        <section id="architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Architecture Overview: ภาพรวมกลไกภายใน LSTM</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Long Short-Term Memory (LSTM) คือหน่วยความจำภายในโครงข่ายประสาทเทียมแบบลำดับ ที่ออกแบบมาเพื่อแก้ปัญหาการลืมบริบทที่ยาวนานใน Recurrent Neural Networks (RNN)
      โครงสร้างของ LSTM ได้รับแรงบันดาลใจจากการวิจัยของ Hochreiter และ Schmidhuber (1997) และต่อมาได้รับการปรับปรุงจากหลายสถาบันชั้นนำ เช่น Google Brain และ Stanford NLP Group
    </p>

    <h3 className="text-xl font-semibold">องค์ประกอบหลักของ LSTM</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Cell State (C<sub>t</sub>):</strong> เส้นทางหลักของการไหลของข้อมูล ที่สามารถถ่ายทอดข้อมูลระยะยาวผ่านเวลา</li>
      <li><strong>Input Gate (i<sub>t</sub>):</strong> กำหนดว่า "ข้อมูลใหม่" ใดควรถูกเพิ่มเข้าไปใน cell state</li>
      <li><strong>Forget Gate (f<sub>t</sub>):</strong> กำหนดว่า "ข้อมูลเก่า" ใดควรถูกลืมจาก cell state</li>
      <li><strong>Output Gate (o<sub>t</sub>):</strong> กำหนดว่า cell state ควรส่งผลอย่างไรต่อ hidden state</li>
    </ul>

    <p>
      การควบคุมการไหลของข้อมูลทั้งหมดดำเนินการผ่าน activation function ได้แก่ sigmoid และ tanh เพื่อควบคุมน้ำหนักของข้อมูลที่ผ่านแต่ละ gate
    </p>

    <h3 className="text-xl font-semibold">สถาปัตยกรรมโดยรวม</h3>
    <p>
      ข้อมูลที่เข้าสู่ LSTM ในแต่ละ timestep คืออินพุต x<sub>t</sub> และ hidden state จาก timestep ก่อนหน้า h<sub>t-1</sub>
      ข้อมูลทั้งสองนี้จะถูกเชื่อมต่อและผ่านเครือข่ายย่อยของแต่ละ gate เพื่อคำนวณการเปลี่ยนแปลงของสถานะ
    </p>

    <h3 className="text-xl font-semibold">ลำดับการประมวลผลภายใน LSTM</h3>
    <ol className="list-decimal list-inside ml-6 space-y-2">
      <li>คำนวณ forget gate เพื่อกำหนดว่าข้อมูลเก่าใน cell state ควรถูกเก็บไว้เท่าไร</li>
      <li>คำนวณ input gate และข้อมูลใหม่ที่จะเพิ่มเข้าไปใน cell state</li>
      <li>อัปเดต cell state โดยรวมผลจาก forget และ input gate</li>
      <li>คำนวณ output gate เพื่อสร้าง hidden state ใหม่ h<sub>t</sub></li>
    </ol>

    <h3 className="text-xl font-semibold">แผนภาพประกอบ</h3>
    <p>
      ในแผนภาพมาตรฐานของ LSTM จะเห็นว่า cell state ไหลอยู่ในแนวนอน โดยถูกเปลี่ยนแปลงบางส่วนด้วย gate แต่ยังคงโครงสร้างพื้นฐานไว้
      เส้นทางนี้ช่วยให้ gradient ไม่หายระหว่าง backpropagation through time
    </p>

    <h3 className="text-xl font-semibold">จุดเด่นของ LSTM ตามการวิเคราะห์จาก Stanford</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>รักษาข้อมูลระยะยาวผ่าน cell state ได้ดี</li>
      <li>การใช้ sigmoid + tanh ทำให้สามารถควบคุมการไหลของข้อมูลในแต่ละจุดได้อย่างละเอียด</li>
      <li>ลดปัญหา vanishing gradient ได้ดีกว่า RNN ปกติ</li>
      <li>สามารถเรียนรู้ pattern แบบ sequential ที่มีความซับซ้อนได้ เช่น ภาษาธรรมชาติและ series ทางเวลา</li>
    </ul>

    <h3 className="text-xl font-semibold">มุมมองจาก MIT 6.S191 และ Oxford</h3>
    <p>
      หลักสูตร MIT 6.S191 เน้นว่า LSTM ใช้โครงสร้างที่เรียกว่า "constant error carousel" เพื่อให้ gradient ไหลต่อเนื่องใน cell state
      ในขณะที่ Oxford กล่าวถึง LSTM ว่าเป็นรูปแบบที่สมดุลระหว่าง expressiveness และ controllability
    </p>

    <h3 className="text-xl font-semibold">สูตรคณิตศาสตร์พื้นฐานของแต่ละ Gate</h3>
    <pre>
{`f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
~C_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t * C_{t-1} + i_t * ~C_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)`}
    </pre>

    <h3 className="text-xl font-semibold">การฝึกฝน LSTM อย่างมีประสิทธิภาพ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ใช้ Adam optimizer เนื่องจากสามารถปรับ learning rate ได้อัตโนมัติ</li>
      <li>Dropout ช่วยลด overfitting โดยเฉพาะเมื่อมีจำนวนพารามิเตอร์สูง</li>
      <li>Gradient clipping เป็นเทคนิคสำคัญในการป้องกัน exploding gradients</li>
      <li>Batch normalization อาจใช้ได้จำกัด แต่ LayerNorm ใช้ได้ใน LSTM ที่ปรับแต่งเอง</li>
    </ul>

    <h3 className="text-xl font-semibold">โครงสร้างที่ปรับแต่งได้</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Bidirectional LSTM ใช้ hidden state สองชุด — forward และ backward</li>
      <li>Stacked LSTM เพิ่มความสามารถในการจับ pattern ระดับลึก</li>
      <li>CuDNN LSTM ใช้ GPU acceleration เพื่อเร่งการฝึกโมเดลขนาดใหญ่</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box: จุดเชื่อมโยงเชิงชีวภาพ</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>LSTM จำลองการทำงานของระบบความจำในสมองมนุษย์ โดยแยกหน้าที่จำ ลืม และส่งออก</li>
        <li>แนวทางการออกแบบ gate สามารถพบในโครงสร้างของ synaptic gating ในระบบประสาทจริง</li>
        <li>การควบคุมข้อมูลโดยใช้ sigmoid เปรียบได้กับกลไกการเปิด-ปิดของเซลล์ประสาท</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-6 text-sm">
      <li>Hochreiter & Schmidhuber, “Long Short-Term Memory”, Neural Computation, 1997</li>
      <li>Stanford CS224n: Natural Language Processing with Deep Learning</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
      <li>Oxford Deep NLP Course, University of Oxford</li>
      <li>Google AI Blog on LSTM Applications</li>
    </ul>
  </div>
</section>


      <section id="flow" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Flow of Data Inside LSTM</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      LSTM (Long Short-Term Memory) ถูกพัฒนาขึ้นเพื่อจัดการกับปัญหา long-term dependency ในข้อมูลลำดับ เช่น ข้อความ เสียง และสัญญาณเวลา
      กลไกภายในของ LSTM ทำงานโดยควบคุมข้อมูลที่ไหลเข้าออก cell state ผ่านชุดของ gate ได้แก่ Forget Gate, Input Gate และ Output Gate
    </p>

    <h3 className="text-xl font-semibold">ลำดับการไหลของข้อมูลในแต่ละ Timestep</h3>
    <ol className="list-decimal list-inside ml-6 space-y-2">
      <li><strong>Step 1:</strong> ประเมินว่า “ข้อมูลเก่า” ควรถูกลืมผ่าน Forget Gate</li>
      <li><strong>Step 2:</strong> คัดเลือก “ข้อมูลใหม่” ที่ควรจดจำผ่าน Input Gate</li>
      <li><strong>Step 3:</strong> อัปเดต Cell State ด้วยข้อมูลใหม่ที่ผ่านการคัดเลือก</li>
      <li><strong>Step 4:</strong> คำนวณ Output Gate เพื่อส่งต่อ hidden state ไป timestep ถัดไป</li>
    </ol>

    <h3 className="text-xl font-semibold">Forget Gate: กรองข้อมูลเก่า</h3>
    <p>
      Forget Gate ใช้ sigmoid function ในการตัดสินใจว่าแต่ละส่วนของ cell state ก่อนหน้าควรถูกลืมหรือคงไว้:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`}
    </pre>
    <p>
      ค่า f_t ที่ได้จะอยู่ระหว่าง 0 ถึง 1 ซึ่งจะถูกนำไปคูณกับ cell state เก่า (C_t-1) เพื่อเลือกว่าจะลบข้อมูลเก่าบางส่วนหรือไม่
    </p>

    <h3 className="text-xl font-semibold">Input Gate: เพิ่มข้อมูลใหม่</h3>
    <p>
      จากนั้นจะเข้าสู่ขั้นตอนของการอัปเดต cell state โดยใช้ input gate และ candidate value:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
~C_t = tanh(W_C · [h_{t-1}, x_t] + b_C)`}
    </pre>
    <p>
      ค่า i_t ใช้ควบคุมว่าข้อมูลใหม่ควรถูกเพิ่มเข้าไปใน memory มากแค่ไหน ส่วน ~C_t เป็นค่าที่เสนอให้บันทึกเข้าไปใน cell state
    </p>

    <h3 className="text-xl font-semibold">Cell State Update</h3>
    <p>
      Cell State ใหม่จะถูกอัปเดตโดยการผสมผสานข้อมูลเก่าที่ถูกคัดกรอง และข้อมูลใหม่ที่ถูกเลือกไว้:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`C_t = f_t * C_{t-1} + i_t * ~C_t`}
    </pre>

    <h3 className="text-xl font-semibold">Output Gate: ส่งผลลัพธ์ต่อไป</h3>
    <p>
      สุดท้าย Output Gate จะเป็นตัวกำหนดว่า hidden state ที่ควรส่งต่อไป timestep ถัดไปคืออะไร:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)`}
    </pre>

    <h3 className="text-xl font-semibold">ภาพรวม Flow ทั้งหมด</h3>
    <p>
      แผนภาพการไหลของข้อมูลใน LSTM แสดงลำดับการทำงานต่อเนื่อง: 
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>รับ input x_t และ hidden state จาก timestep ก่อนหน้า</li>
      <li>ผ่านการคำนวณสาม gate → อัปเดต cell state</li>
      <li>ผลลัพธ์ถูกนำไปใช้ทั้งใน prediction และการคำนวณ timestep ถัดไป</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>LSTM เลียนแบบการควบคุมความทรงจำของสมองมนุษย์ผ่านกลไก gate</li>
        <li>การควบคุม cell state อย่างมีเหตุผลช่วยป้องกันปัญหา vanishing gradient</li>
        <li>การอัปเดตแบบค่อยเป็นค่อยไปใน cell state ทำให้ข้อมูลสำคัญคงอยู่ได้นานหลาย timestep</li>
        <li>แนวคิดนี้ได้รับการยืนยันและอธิบายอย่างละเอียดในวิชา CS224n จาก Stanford</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงและการเรียนรู้ต่อยอด</h3>
    <ul className="list-disc list-inside ml-6 space-y-2 text-sm">
      <li>Stanford CS224n: Natural Language Processing with Deep Learning</li>
      <li>MIT Deep Learning for Self-Driving Cars (6.S191)</li>
      <li>Oxford Deep Learning Lectures</li>
      <li>Andrej Karpathy’s Blog on RNN and LSTM</li>
      <li>Christopher Olah’s Blog: “Understanding LSTM Networks”</li>
    </ul>
  </div>
</section>

   
        <section id="lstm-equations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. LSTM Equations (แบบเข้าใจง่าย)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      สมการของ LSTM อธิบายการทำงานของเซลล์ในแต่ละช่วงเวลา โดยใช้แนวคิด "gate" ในการควบคุมข้อมูลที่ควรจดจำ ลืม หรือส่งต่อไปยังเวลาถัดไป
      แนวคิดนี้ช่วยให้โมเดลมีความสามารถในการเรียนรู้ลำดับข้อมูลที่มีบริบทห่างกันอย่างมีประสิทธิภาพ
    </p>
    <h3 className="text-xl font-semibold">1. Forget Gate</h3>
    <p>
      หน้าที่ของ forget gate คือการตัดสินใจว่าจะลืมข้อมูลจากสถานะก่อนหน้า (previous cell state) เท่าใด โดยใช้ sigmoid activation function
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto"><code>f_t = σ(W_f · [hₜ₋₁, xₜ] + b_f)</code></pre>
    <h3 className="text-xl font-semibold">2. Input Gate</h3>
    <p>
      Input gate ควบคุมว่าข้อมูลใหม่ใดควรถูกเพิ่มเข้าสู่สถานะเซลล์ โดยประกอบด้วยสองส่วน: gate และ candidate values
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto"><code>i_t = σ(W_i · [hₜ₋₁, xₜ] + b_i)
Ĉ_t = tanh(W_C · [hₜ₋₁, xₜ] + b_C)</code></pre>
    <h3 className="text-xl font-semibold">3. Cell State Update</h3>
    <p>
      สถานะของเซลล์ (cell state) ถูกอัปเดตโดยรวมข้อมูลที่ผ่าน forget gate และ input gate
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto"><code>C_t = f_t * Cₜ₋₁ + i_t * Ĉ_t</code></pre>
    <h3 className="text-xl font-semibold">4. Output Gate</h3>
    <p>
      Output gate ตัดสินใจว่าส่วนใดของสถานะเซลล์ควรถูกส่งออกเป็น hidden state
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto"><code>o_t = σ(W_o · [hₜ₋₁, xₜ] + b_o)
h_t = o_t * tanh(C_t)</code></pre>
    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Forget gate ทำให้ LSTM สามารถควบคุมข้อมูลที่ควรถูกลืมจากอดีต</li>
        <li>Input gate ช่วยกำหนดว่าควรเรียนรู้อะไรใหม่เข้าสู่สถานะเซลล์</li>
        <li>Cell state คือหน่วยความจำหลักที่ไหลผ่านเวลาโดยมีการอัปเดตอย่างต่อเนื่อง</li>
        <li>Output gate ส่งต่อเฉพาะข้อมูลที่สำคัญไปยัง timestep ถัดไป</li>
        <li>การรวม tanh และ sigmoid ทำให้เกิดการควบคุมที่ยืดหยุ่นต่อข้อมูล</li>
      </ul>
    </div>
    <p>
      LSTM ได้รับการออกแบบมาเพื่อแก้ปัญหา vanishing gradient ใน RNN แบบดั้งเดิม
      โดยใช้โครงสร้าง gate เหล่านี้เพื่อรักษาข้อมูลสำคัญตลอดลำดับเวลา
      รายวิชาจากมหาวิทยาลัย Stanford และงานวิจัยของ Hochreiter & Schmidhuber
      ยืนยันความสามารถนี้ทั้งทางทฤษฎีและเชิงปฏิบัติ
    </p>
  </div>
</section>


{/** Section 5 */}
<section id="comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Differences: LSTM vs Vanilla RNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      ความแตกต่างระหว่าง Long Short-Term Memory (LSTM) และ Recurrent Neural Network (RNN) แบบดั้งเดิม เป็นหัวใจสำคัญในการทำความเข้าใจวิวัฒนาการของการเรียนรู้ลำดับข้อมูลตามเวลา โดยเฉพาะในงานที่ต้องการเก็บข้อมูลระยะยาว LSTM ได้รับการออกแบบมาเพื่อแก้ปัญหาหลักของ Vanilla RNN คือปัญหา vanishing gradient ซึ่งทำให้โมเดลไม่สามารถเรียนรู้บริบทระยะยาวได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">ข้อเปรียบเทียบด้านโครงสร้างภายใน</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Vanilla RNN:</strong> ใช้เพียง tanh หรือ ReLU ในการเปลี่ยนสถานะของ hidden state โดยไม่มีโครงสร้างสำหรับการควบคุมการไหลของข้อมูล
      </li>
      <li>
        <strong>LSTM:</strong> ประกอบด้วย 3 gates คือ input, forget และ output gate รวมถึง cell state ที่ช่วยเก็บข้อมูลในระยะยาวอย่างมีประสิทธิภาพ
      </li>
    </ul>

    <h3 className="text-xl font-semibold">ปัญหา Vanishing Gradient ใน Vanilla RNN</h3>
    <p>
      ตามรายงานของ Stanford CS224n และงานวิจัยของ Bengio (1994) โมเดล RNN ทั่วไปจะประสบปัญหา gradient หายไปเมื่อทำการ backpropagation ผ่านเวลาหลาย step ซึ่งทำให้โมเดลไม่สามารถปรับพารามิเตอร์ได้สำหรับข้อมูลที่อยู่ไกลออกไปใน sequence
    </p>

    <p>
      ในทางตรงกันข้าม LSTM ถูกเสนอโดย Hochreiter & Schmidhuber (1997) เพื่อแก้ปัญหาดังกล่าว โดยการออกแบบให้ cell state สามารถส่งผ่านข้อมูลโดยมีการควบคุมอย่างแม่นยำผ่าน gating mechanism
    </p>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบเชิงลึก</h3>
    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Vanilla RNN</th>
          <th className="border px-4 py-2">LSTM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Memory Capacity</td>
          <td className="border px-4 py-2">จำข้อมูลระยะสั้นได้เท่านั้น</td>
          <td className="border px-4 py-2">เก็บบริบทระยะยาวได้ดี</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Gradient Stability</td>
          <td className="border px-4 py-2">เกิด vanishing gradient</td>
          <td className="border px-4 py-2">แก้ปัญหานี้ด้วย gating</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Architecture</td>
          <td className="border px-4 py-2">ไม่มี gating mechanism</td>
          <td className="border px-4 py-2">มี 3 gates + cell state</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Training Time</td>
          <td className="border px-4 py-2">เร็วกว่าเล็กน้อย</td>
          <td className="border px-4 py-2">ช้ากว่าแต่เสถียรกว่า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Use Cases</td>
          <td className="border px-4 py-2">ข้อมูล sequence สั้น เช่น POS tagging</td>
          <td className="border px-4 py-2">long sequence เช่น machine translation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ผลกระทบจากการเลือกโมเดลที่เหมาะสม</h3>
    <p>
      LSTM ได้พิสูจน์แล้วว่าสามารถเรียนรู้จากลำดับข้อมูลที่มี dependency ระยะยาวได้ดีกว่า Vanilla RNN อย่างมาก ซึ่งเป็นเหตุผลที่ทำให้ LSTM ถูกใช้อย่างแพร่หลายในงานที่เกี่ยวข้องกับ natural language processing, time-series forecasting และ signal modeling ในอุตสาหกรรม
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>RNN แบบดั้งเดิมไม่สามารถจำบริบทใน sequence ที่ยาวได้ดีเท่า LSTM</li>
        <li>LSTM ใช้กลไก gate เพื่อควบคุมการไหลของข้อมูลระหว่าง timestep</li>
        <li>งานวิจัยของ Stanford, MIT และ DeepMind สนับสนุนการใช้ LSTM แทน RNN ในระบบที่มี dependency สูง</li>
        <li>ความสามารถของ LSTM มีบทบาทสำคัญต่อระบบ voice recognition, text generation และ health signal modeling</li>
      </ul>
    </div>
  </div>
</section>


     {/** Section 6 */}
<section id="variants" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Variants of LSTM</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Long Short-Term Memory (LSTM) ได้ถูกพัฒนาต่อยอดในรูปแบบต่าง ๆ โดยมีจุดมุ่งหมายเพื่อเพิ่มประสิทธิภาพด้านความสามารถในการประมวลผลข้อมูลลำดับ (sequences) 
      โดยเฉพาะอย่างยิ่งในบริบทที่ต้องใช้การตีความระยะยาว งานวิจัยจาก Stanford, MIT, Google Brain, และมหาวิทยาลัย Carnegie Mellon ล้วนมีส่วนสำคัญในการผลักดันแนวคิด LSTM variants เหล่านี้
    </p>

    <h3 className="text-xl font-semibold">1. Peephole LSTM</h3>
    <p>
      Peephole LSTM เชื่อม cell state กับ gate ต่าง ๆ โดยตรง (input, forget, output) เพื่อให้แต่ละ gate สามารถเห็นค่าความจำปัจจุบันและตัดสินใจแม่นยำขึ้น
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เหมาะกับงานควบคุมจังหวะเวลา (timing-sensitive tasks)</li>
      <li>ช่วยเพิ่มความละเอียดของการตัดสินใจภายใน cell</li>
      <li>มีการใช้งานในงานวิเคราะห์เสียงระดับโฟนีม</li>
    </ul>

    <h3 className="text-xl font-semibold">2. Bidirectional LSTM (BiLSTM)</h3>
    <p>
      BiLSTM ใช้ LSTM สองชุดประมวลผลในทิศทางตรงและย้อนกลับ ทำให้สามารถเรียนรู้ทั้งอดีตและอนาคตได้พร้อมกัน
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>นิยมใช้ใน NLP เช่น Named Entity Recognition (NER), Machine Translation</li>
      <li>เพิ่มความเข้าใจคำและบริบททั้งประโยค</li>
      <li>ถูกใช้อย่างแพร่หลายในระบบถามตอบ, Text-to-Speech</li>
    </ul>

    <h3 className="text-xl font-semibold">3. Stacked LSTM</h3>
    <p>
      Stacked หรือ Deep LSTM ใช้หลายชั้นของ LSTM เพื่อเรียนรู้ลำดับที่มีโครงสร้างซับซ้อน
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>แต่ละเลเยอร์สามารถเรียนรู้ลำดับความสัมพันธ์ในระดับนามธรรมที่สูงขึ้น</li>
      <li>ใช้ใน DeepSpeech, translation systems, และ sequence generation</li>
      <li>ความลึกช่วยให้ประสิทธิภาพการจำแนกเพิ่มขึ้น แต่ต้องควบคุม overfitting</li>
    </ul>

    <h3 className="text-xl font-semibold">4. CuDNN LSTM</h3>
    <p>
      CuDNN LSTM พัฒนาโดย NVIDIA ให้สามารถคำนวณ LSTM ได้เร็วขึ้นบน GPU โดยรวมฟังก์ชันต่าง ๆ ให้กลายเป็น operation เดียว
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เหมาะกับระบบฝึกโมเดลขนาดใหญ่เชิงพาณิชย์</li>
      <li>มีการใช้งานใน PyTorch, TensorFlow ผ่าน API โดยตรง</li>
      <li>ลดเวลา training บน dataset ใหญ่ เช่น Image Captioning หรือ Speech Synthesis</li>
    </ul>

    <h3 className="text-xl font-semibold">5. ConvLSTM</h3>
    <p>
      ConvLSTM ใช้ convolution ภายใน LSTM cell แทนที่ dense layers เพื่อรักษาโครงสร้าง spatial ของข้อมูล เช่น วิดีโอหรือ radar map
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เหมาะกับงาน forecasting แบบ spatiotemporal เช่น weather prediction</li>
      <li>งานต้นแบบโดย Shi et al. (NIPS 2015)</li>
      <li>ใช้ใน traffic flow prediction, medical imaging, dynamic scene modeling</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ</h3>
    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">รูปแบบ</th>
          <th className="border px-4 py-2">จุดเด่น</th>
          <th className="border px-4 py-2">การใช้งาน</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Peephole</td>
          <td className="border px-4 py-2">เห็น cell state โดยตรง</td>
          <td className="border px-4 py-2">วิเคราะห์จังหวะเวลา</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">BiLSTM</td>
          <td className="border px-4 py-2">อ่านจากทั้งสองทิศทาง</td>
          <td className="border px-4 py-2">NLP, คำพูด</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stacked</td>
          <td className="border px-4 py-2">เลเยอร์ลึกหลายชั้น</td>
          <td className="border px-4 py-2">ลำดับซับซ้อน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">CuDNN</td>
          <td className="border px-4 py-2">เร็วมากบน GPU</td>
          <td className="border px-4 py-2">Production-scale AI</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ConvLSTM</td>
          <td className="border px-4 py-2">รองรับ spatial data</td>
          <td className="border px-4 py-2">Video, ภาพ 3 มิติ</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>แต่ละ LSTM variant แก้ปัญหาเฉพาะจุดที่ LSTM แบบดั้งเดิมยังขาด</li>
        <li>BiLSTM คือพื้นฐานของโมเดล NLP สมัยใหม่ เช่น BERT</li>
        <li>ConvLSTM กำลังถูกใช้มากขึ้นในระบบ multi-modal AI</li>
        <li>CuDNN LSTM คือหัวใจของการฝึกโมเดลความเร็วสูงบนคลาวด์</li>
      </ul>
    </div>
  </div>
</section>


       {/** Section 7 */}
<section id="training" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Training LSTM: ข้อควรระวัง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การฝึก LSTM จำเป็นต้องระมัดระวังเป็นพิเศษ เนื่องจากโครงสร้างของโมเดลมีความซับซ้อนและไวต่อค่าพารามิเตอร์ การออกแบบที่ไม่เหมาะสมสามารถทำให้โมเดลเกิดปัญหาเช่น exploding gradients, overfitting หรือ convergence ที่ช้าลง
    </p>

    <h3 className="text-xl font-semibold">1. Gradient Clipping</h3>
    <p>
      การไหลของ gradient อาจทำให้ค่าอัปเดตของน้ำหนักมีขนาดใหญ่มากในบางรอบการฝึก ซึ่งส่งผลเสียต่อการเรียนรู้ เทคนิค gradient clipping จึงถูกนำมาใช้เพื่อตัด gradient ที่เกินค่ากำหนด
    </p>
    <pre><code>{`from tensorflow.keras.optimizers import Adam
optimizer = Adam(clipvalue=1.0)`}</code></pre>
    <p>
      แนวทางนี้ถูกแนะนำในรายวิชา CS224n ของ Stanford University ว่าเป็นเทคนิคพื้นฐานที่ช่วยให้การฝึก LSTM มีเสถียรภาพมากขึ้น
    </p>

    <h3 className="text-xl font-semibold">2. Layer Normalization แทน Batch Norm</h3>
    <p>
      LSTM ตอบสนองได้ดีกับ Layer Normalization เนื่องจาก batch normalization ไม่เหมาะกับลำดับข้อมูลแบบ time-series ที่ความยาวไม่เท่ากัน
    </p>
    <pre><code>{`from tensorflow.keras.layers import LSTM, LayerNormalization
x = LSTM(128, return_sequences=True)(x)
x = LayerNormalization()(x)`}</code></pre>

    <h3 className="text-xl font-semibold">3. การใช้ Dropout อย่างเหมาะสม</h3>
    <p>
      LSTM รองรับ dropout สำหรับทั้ง input และ recurrent connection โดย recurrent dropout เป็นเทคนิคสำคัญที่ช่วยป้องกัน overfitting ในงานที่มีข้อมูลน้อย
    </p>
    <pre><code>{`model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2))`}</code></pre>

    <h3 className="text-xl font-semibold">4. Sequence Padding และ Masking</h3>
    <p>
      ลำดับที่มีความยาวต่างกันจำเป็นต้องใช้ padding และ masking เพื่อไม่ให้โมเดลเข้าใจข้อมูล padding ว่าเป็นข้อมูลจริง
    </p>
    <pre><code>{`Embedding(input_dim=10000, output_dim=128, mask_zero=True)`}</code></pre>

    <h3 className="text-xl font-semibold">5. Learning Rate Scheduler</h3>
    <p>
      การเริ่มต้นด้วย learning rate ที่ไม่เหมาะสมอาจทำให้โมเดลไม่ converge หรือเรียนรู้ได้ช้า
    </p>
    <pre><code>{`from tensorflow.keras.callbacks import ReduceLROnPlateau
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)`}</code></pre>

    <h3 className="text-xl font-semibold">6. การเลือก Optimizer</h3>
    <p>
      Adam และ RMSprop เป็นตัวเลือกยอดนิยมสำหรับการฝึก LSTM เนื่องจากสามารถจัดการกับ gradient ที่เปลี่ยนแปลงเร็วได้ดีกว่า SGD
    </p>
    <pre><code>{`model.compile(optimizer='adam', loss='mse')`}</code></pre>

    <h3 className="text-xl font-semibold">7. Early Stopping</h3>
    <p>
      เทคนิค early stopping ช่วยหยุดการฝึกเมื่อ validation loss ไม่ลดลงอีก โดยรักษาน้ำหนักที่ดีที่สุด
    </p>
    <pre><code>{`from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`}</code></pre>

    <h3 className="text-xl font-semibold">8. Data Augmentation สำหรับ Time Series</h3>
    <p>
      ในบางกรณีสามารถใช้เทคนิค time warping, jittering หรือ window slicing เพื่อเพิ่มความหลากหลายของลำดับ
    </p>

    <h3 className="text-xl font-semibold">9. Evaluation Metrics ที่เหมาะสม</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Regression: Mean Squared Error (MSE), Mean Absolute Error (MAE)</li>
      <li>Classification: Accuracy, Precision, Recall, F1 Score</li>
      <li>Sequence: BLEU, ROUGE, Perplexity</li>
    </ul>

    <h3 className="text-xl font-semibold">10. Cross-Validation สำหรับ Time Series</h3>
    <p>
      ควรใช้ TimeSeriesSplit แทน k-fold ปกติ เพื่อไม่สลับลำดับเวลาในการประเมิน
    </p>
    <pre><code>{`from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)`}</code></pre>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>การฝึก LSTM ต้องระวังปัญหาการไหลของ gradient โดยเฉพาะ exploding gradients</li>
        <li>Layer Normalization และ recurrent dropout ช่วยเพิ่มเสถียรภาพของโมเดล</li>
        <li>การวัดผลควรเลือก metric ที่ตรงกับลักษณะปัญหา เช่น Perplexity สำหรับ sequence</li>
        <li>แนะนำให้ใช้ masking กับข้อมูลที่ pad แล้ว เพื่อให้โมเดลไม่เรียนรู้จาก padding</li>
      </ul>
    </div>

  </div>
</section>


       {/** Section 8 */}
<section id="use-cases" className="mb-16 scroll-mt-32 min-h-[300px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. กรณีใช้งานจริงของ LSTM</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      เครือข่าย LSTM ถูกนำไปใช้ในโลกจริงอย่างแพร่หลายในงานที่ต้องจัดการกับข้อมูลที่มีลำดับเวลา ซึ่งต้องการการจดจำข้อมูลระยะยาว ด้านล่างคือตัวอย่างการใช้งานจากหลายสาขา ที่แสดงให้เห็นถึงจุดแข็งของ LSTM ที่เหนือกว่า RNN แบบดั้งเดิมหรือโมเดลชนิดอื่น
    </p>

    <h3 className="text-xl font-semibold">1. การประมวลผลภาษาธรรมชาติ (NLP)</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การสร้างข้อความ:</strong> สร้างประโยคต่อเนื่องที่มีโครงสร้างภาษาถูกต้อง</li>
      <li><strong>การระบุชื่อเฉพาะ:</strong> ตรวจหาชื่อบุคคล สถานที่ องค์กร</li>
      <li><strong>การวิเคราะห์ไวยากรณ์:</strong> จัดประเภทคำในประโยคโดยรักษาความเข้าใจบริบท</li>
      <li><strong>การแปลภาษา:</strong> ใช้สถาปัตยกรรม encoder-decoder แปลระหว่างภาษา</li>
      <li><strong>สรุปเนื้อหาอัตโนมัติ:</strong> สร้างบทสรุปแบบย่อโดยใช้ LSTM หลายชั้นร่วมกับ attention</li>
    </ul>

    <h3 className="text-xl font-semibold">2. การรู้จำเสียงและสัญญาณ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>ผู้ช่วยเสียง:</strong> Siri และ Alexa ใช้ LSTM แปลงเสียงเป็นข้อความ</li>
      <li><strong>การรู้จำหน่วยเสียง:</strong> จัดประเภทเสียงย่อยเพื่อสร้างคำพูด</li>
      <li><strong>การระบุตัวผู้พูด:</strong> แยกผู้พูดออกจากกันแม้มีสภาพเสียงแตกต่าง</li>
    </ul>

    <h3 className="text-xl font-semibold">3. การคาดการณ์ทางการเงิน</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การทำนายราคาหุ้น:</strong> คาดการณ์จากแนวโน้มของข้อมูลในอดีต</li>
      <li><strong>การตรวจจับเหตุการณ์ผิดปกติ:</strong> ตรวจหาความผิดปกติในพฤติกรรมตลาด</li>
      <li><strong>การวิเคราะห์ความเสี่ยง:</strong> ประเมินความน่าเชื่อถือจากพฤติกรรมเครดิต</li>
    </ul>

    <h3 className="text-xl font-semibold">4. ระบบการแพทย์</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การวิเคราะห์ ECG:</strong> ทำนายภาวะหัวใจผิดจังหวะ</li>
      <li><strong>การตรวจจับชักจาก EEG:</strong> วิเคราะห์คลื่นสมองอย่างต่อเนื่อง</li>
      <li><strong>การพยากรณ์เวชระเบียน:</strong> คาดการณ์ลำดับโรคหรือยาที่จะได้รับ</li>
    </ul>

    <h3 className="text-xl font-semibold">5. การประพันธ์ดนตรีและเสียง</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การสร้างเพลง:</strong> MuseNet ของ OpenAI ใช้ LSTM สร้างเพลงหลายเครื่องดนตรี</li>
      <li><strong>การพยากรณ์คอร์ด:</strong> คาดเดาคอร์ดเพลงต่อไปอย่างสอดคล้องทางดนตรี</li>
    </ul>

    <h3 className="text-xl font-semibold">6. หุ่นยนต์และระบบควบคุม</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การวางแผนเส้นทาง:</strong> พยากรณ์เส้นทางเดินหลีกสิ่งกีดขวาง</li>
      <li><strong>การรวมสัญญาณหลายแหล่ง:</strong> ใช้กับยานไร้คนขับเพื่อตัดสินใจ</li>
      <li><strong>การทำนายการเคลื่อนไหวของมนุษย์:</strong> คาดการณ์การเคลื่อนไหวในการร่วมมือกับหุ่นยนต์</li>
    </ul>

    <h3 className="text-xl font-semibold">7. พลังงานและสิ่งแวดล้อม</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การคาดการณ์การใช้พลังงาน:</strong> พยากรณ์รายวันและตามฤดูกาล</li>
      <li><strong>การวิเคราะห์ภูมิอากาศ:</strong> คาดการณ์อุณหภูมิและฝนในระยะยาว</li>
      <li><strong>สมดุลโหลดในระบบไฟฟ้า:</strong> ป้องกันการใช้พลังงานเกินพิกัด</li>
    </ul>

    <h3 className="text-xl font-semibold">8. การศึกษาและวิเคราะห์ผู้เรียน</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การพยากรณ์ผลการเรียน:</strong> วิเคราะห์จากพฤติกรรมในอดีต</li>
      <li><strong>การตรวจจับความเสี่ยงในการเลิกเรียน:</strong> ตรวจจากลำดับกิจกรรมออนไลน์</li>
      <li><strong>คำแนะนำการเรียนถัดไป:</strong> แนะนำบทเรียนตามเส้นทางเดิม</li>
    </ul>

    <h3 className="text-xl font-semibold">9. ระบบขนส่งและโลจิสติกส์</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การพยากรณ์จราจร:</strong> คาดการณ์การจราจรเพื่อหลีกเลี่ยงรถติด</li>
      <li><strong>การคาดเวลาเดินทาง:</strong> ใช้ข้อมูล GPS ทำนายเวลาถึงปลายทาง</li>
      <li><strong>การบริหารขนส่งสาธารณะ:</strong> คาดการณ์จำนวนผู้โดยสาร</li>
    </ul>

    <h3 className="text-xl font-semibold">10. ความปลอดภัยไซเบอร์</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การตรวจจับการบุกรุกระบบ:</strong> ตรวจลักษณะแปลกปลอมในเครือข่าย</li>
      <li><strong>การจำแนกมัลแวร์:</strong> วิเคราะห์ลำดับคำสั่งหรือ API call</li>
    </ul>

    <h3 className="text-xl font-semibold">11. IoT อุตสาหกรรมและโรงงาน</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การควบคุมกระบวนการ:</strong> วิเคราะห์ข้อมูลจากเซ็นเซอร์แบบต่อเนื่อง</li>
      <li><strong>การวินิจฉัยความผิดปกติ:</strong> คาดการณ์ความล้มเหลวจากข้อมูลลำดับ</li>
    </ul>

    <h3 className="text-xl font-semibold">12. การเฝ้าระวังวิดีโอและคำบรรยายภาพ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>การจำแนกพฤติกรรม:</strong> วิเคราะห์การกระทำจากภาพวิดีโอ</li>
      <li><strong>การคาดการณ์กิจกรรม:</strong> คาดท่าทางหรือพฤติกรรมจากลำดับเฟรม</li>
      <li><strong>การเขียนคำอธิบายวิดีโอ:</strong> สร้างข้อความจากข้อมูลภาพแบบลำดับ</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงงานวิจัย</h3>
    <ul className="list-disc list-inside ml-6 space-y-2 text-sm">
      <li>Stanford CS224n: NLP with LSTM</li>
      <li>MIT CSAIL: Time-series modeling in healthcare</li>
      <li>Google DeepMind: Audio sequence modeling</li>
      <li>OpenAI: Music generation with LSTM</li>
      <li>Carnegie Mellon: Grid forecasting using RNNs</li>
      <li>Harvard Medical School: ECG/EEG temporal analysis</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>LSTM ยังคงมีบทบาทสำคัญในงาน edge computing เนื่องจากใช้งานได้รวดเร็วและเบากว่า Transformer</li>
        <li>LSTM แบบหลายชั้นหรือสองทิศทางยิ่งช่วยเพิ่มความแม่นยำในการเรียนรู้ลำดับ</li>
        <li>สามารถนำไปใช้ในหลายสาขา เช่น การแพทย์ ความมั่นคง อุตสาหกรรม และศิลปะ</li>
      </ul>
    </div>

  </div>
</section>


        {/* Section 9 */}
<section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Visualization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การทำ Visualization ใน LSTM มีบทบาทสำคัญในการทำความเข้าใจกลไกการทำงานภายในของ cell โดยเฉพาะ cell state และการเปิด/ปิด gate ต่าง ๆ
      ซึ่งช่วยให้สามารถวิเคราะห์ว่าโมเดลเลือก “จำ” หรือ “ลืม” อะไรในแต่ละ timestep ได้อย่างแม่นยำมากขึ้น
    </p>

    <h3 className="text-xl font-semibold">การแสดงค่า Cell State ตลอดเวลา</h3>
    <p>
      จากบทเรียนของ Stanford CS224n และ Google Research วิธีการหนึ่งที่นิยมใช้คือ การ plot ค่าของ cell state C_t ในแต่ละ timestep
      เพื่อติดตามว่า LSTM กำลังจำหรือปล่อยข้อมูลใดออกไปอย่างไรในระยะยาว
    </p>
    <pre>{`import matplotlib.pyplot as plt
import numpy as np

timesteps = 50
cell_states = np.random.randn(timesteps) * 0.2 + np.linspace(0, 1, timesteps)

plt.figure(figsize=(10,4))
plt.plot(range(timesteps), cell_states, label='Cell State C_t')
plt.title("LSTM Cell State Over Time")
plt.xlabel("Timestep")
plt.ylabel("Cell State Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()`}</pre>

    <h3 className="text-xl font-semibold">เปรียบเทียบ Hidden State ระหว่าง RNN และ LSTM</h3>
    <p>
      การเปรียบเทียบ h_t ของ Vanilla RNN และ LSTM ภายใต้ sequence เดียวกัน จะเผยให้เห็นว่า LSTM สามารถรักษาความทรงจำได้ยาวนานกว่ามาก
    </p>
    <pre>{`import numpy as np
import matplotlib.pyplot as plt

timesteps = 100
rnn_h = np.exp(-np.linspace(0, 5, timesteps))  # RNN จะลืมเร็วมาก
lstm_h = np.ones(timesteps) * 0.8              # LSTM รักษาค่าคงที่ไว้ได้

plt.figure(figsize=(10,4))
plt.plot(rnn_h, label="Vanilla RNN h_t", linestyle='--')
plt.plot(lstm_h, label="LSTM h_t", linestyle='-')
plt.title("Comparison of Hidden State Decay")
plt.xlabel("Timestep")
plt.ylabel("Hidden State Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()`}</pre>

    <h3 className="text-xl font-semibold">Heatmap การทำงานของ Gates</h3>
    <p>
      การแสดงค่าของ forget, input, และ output gate ผ่าน heatmap เป็นอีกวิธีหนึ่งที่ช่วยให้เห็นว่าข้อมูลถูกเปิด/ปิดที่ตำแหน่งใดของ sequence
      โดยทั่วไปจะนำค่าจาก sigmoid activation มาแสดงในรูปแบบนี้
    </p>
    <pre>{`import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

timesteps = 30
gates = np.random.rand(timesteps, 3)  # Fake gate activations: forget, input, output

plt.figure(figsize=(8, 6))
sns.heatmap(gates.T, cmap="YlGnBu", xticklabels=False, yticklabels=["Forget", "Input", "Output"])
plt.title("Gate Activation Heatmap")
plt.xlabel("Timestep")
plt.ylabel("Gate")
plt.tight_layout()
plt.show()`}</pre>

    <h3 className="text-xl font-semibold">ตัวอย่างการตีความจากงานวิจัย</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>MIT 6.S191 ใช้ heatmap เพื่อแสดงว่า LSTM สนใจ timestep ใดในการประมวลผลภาษา</li>
      <li>Google Brain วิเคราะห์ input gate เพื่อศึกษาผลกระทบจากคำในอดีตกับการตอบสนองของโมเดล</li>
      <li>Stanford NLP ใช้การแสดงค่า cell state เพื่อออกแบบ regularization ใหม่ (LayerNorm LSTM)</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>LSTM visualization ช่วยวิเคราะห์พฤติกรรมของโมเดลลึกใน task ที่มี long-term dependency</li>
        <li>การแสดง hidden state decay และ cell memory trend เป็นเทคนิคที่พบในงานวิจัยของ Stanford, DeepMind และ FAIR</li>
        <li>Heatmap gate activation ทำให้เข้าใจได้ว่าโมเดลเลือกจำ/ลืมแบบมีเหตุผลหรือไม่ในแต่ละ timestep</li>
        <li>การวิเคราะห์ค่า gradient และ cell value ช่วยตรวจจับ bias และเรียนรู้การ fine-tune โครงสร้างของ LSTM ได้ดีขึ้น</li>
      </ul>
    </div>
  </div>
</section>

      {/** Section 10 */}
<section id="code" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Coding Walkthrough (สรุปโค้ด)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      ตัวอย่างการใช้งาน Long Short-Term Memory (LSTM) ด้านล่างนี้เป็นโค้ดที่ออกแบบมาให้เข้าใจได้ง่าย โดยอ้างอิงจากแนวปฏิบัติที่พบในรายวิชา CS231n จาก Stanford University, หลักสูตร Deep Learning Specialization ของ Andrew Ng และตัวอย่างจาก TensorFlow Official Docs ซึ่งครอบคลุมการเตรียมข้อมูล การสร้างโมเดล การฝึก และการประเมินผลในงาน time series regression
    </p>

    <h3 className="text-xl font-semibold">การติดตั้งไลบรารี</h3>
    <pre>
{`pip install numpy pandas matplotlib tensorflow scikit-learn`}
    </pre>

    <h3 className="text-xl font-semibold">1. การเตรียมข้อมูล Time Series</h3>
    <pre>
{`import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# จำลองข้อมูล Time Series
np.random.seed(42)
time = np.arange(0, 1000)
data = np.sin(0.02 * time) + np.random.normal(scale=0.5, size=len(time))
df = pd.DataFrame({"value": data})

# การปรับ scale ข้อมูล
scaler = MinMaxScaler()
df["scaled"] = scaler.fit_transform(df[["value"]])

# สร้างชุดข้อมูลลำดับ
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 30
X, y = create_sequences(df["scaled"].values, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))
print("Input Shape:", X.shape)
print("Target Shape:", y.shape)`}
    </pre>

    <h3 className="text-xl font-semibold">2. สร้างโมเดล LSTM ด้วย Keras</h3>
    <pre>
{`from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.summary()`}
    </pre>

    <h3 className="text-xl font-semibold">3. การฝึกโมเดล</h3>
    <pre>
{`early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)`}
    </pre>

    <h3 className="text-xl font-semibold">4. การประเมินและการวิเคราะห์</h3>
    <pre>
{`predictions = model.predict(X)

# แปลงกลับไปยังสเกลเดิม
pred_scaled = scaler.inverse_transform(predictions)
y_scaled = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(y_scaled, label='True')
plt.plot(pred_scaled, label='Predicted')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()`}
    </pre>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>โครงสร้าง LSTM ที่มี Dropout และ EarlyStopping สามารถลด overfitting ได้ดีในข้อมูล sequence</li>
        <li>การปรับขนาดข้อมูลด้วย MinMaxScaler เป็นสิ่งจำเป็นสำหรับ LSTM เพราะการเรียนรู้มีความไวต่อสเกล</li>
        <li>การสร้าง sequence อย่างถูกต้อง (sliding window) มีผลต่อการเรียนรู้ของโมเดลอย่างมาก</li>
        <li>LSTM สามารถนำไปใช้กับข้อมูล ECG, ราคา, demand หรือข้อมูลเซ็นเซอร์ที่มีลักษณะตามลำดับเวลา</li>
        <li>การเลือก batch size และจำนวน epoch ต้องพิจารณาจากขนาดข้อมูลและความซับซ้อนของสัญญาณ</li>
      </ul>
    </div>
  </div>
</section>


        <section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. สรุป (Summary)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Long Short-Term Memory (LSTM) ได้กลายเป็นหนึ่งในโครงข่ายประสาทเทียมที่สำคัญที่สุดในประวัติศาสตร์ของ Deep Learning 
      โดยเฉพาะในงานที่ต้องจัดการกับข้อมูลที่เป็นลำดับ เช่น ภาษา พยากรณ์เวลา และสัญญาณทางชีวการแพทย์
    </p>

    <p>
      LSTM สามารถแก้ปัญหาหลักของ RNN แบบดั้งเดิม ซึ่งได้แก่การสูญเสียข้อมูลระยะยาว (long-term dependency) และปัญหา vanishing gradient 
      ผ่านกลไกของ “Cell State” และ “Gate” ที่ควบคุมการจำ ลืม และส่งต่อข้อมูลอย่างมีระบบ
    </p>

    <h3 className="text-xl font-semibold"> สาระสำคัญที่ควรจดจำ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>LSTM มีความสามารถสูงในการจัดการกับบริบทระยะยาว โดยไม่สูญเสียความแม่นยำตามระยะเวลา</li>
      <li>กลไกของ 3 ประตูหลัก (Input, Forget, Output) ช่วยให้โครงสร้างมีความยืดหยุ่นและสามารถเลือกจำหรือมองข้ามข้อมูลได้</li>
      <li>เทคโนโลยีนี้เป็นพื้นฐานของระบบอย่าง Siri, Google Assistant, GPT และระบบพยากรณ์ที่ใช้ในอุตสาหกรรม</li>
      <li>รูปแบบหลากหลายของ LSTM เช่น BiLSTM, Stacked LSTM และ CuDNN LSTM ช่วยให้สามารถปรับใช้กับงานหลากหลายได้อย่างมีประสิทธิภาพ</li>
      <li>การฝึก LSTM อย่างเหมาะสมต้องใช้เทคนิคป้องกัน overfitting และ exploding gradient</li>
    </ul>

    <h3 className="text-xl font-semibold"> เปรียบเทียบภาพรวมระหว่าง RNN และ LSTM</h3>
    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">หัวข้อ</th>
          <th className="border px-4 py-2">RNN ปกติ</th>
          <th className="border px-4 py-2">LSTM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ความจำระยะยาว</td>
          <td className="border px-4 py-2">ต่ำ</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ปัญหา Gradient</td>
          <td className="border px-4 py-2">Vanishing</td>
          <td className="border px-4 py-2">ป้องกันได้ดี</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความยืดหยุ่น</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความซับซ้อนของโครงสร้าง</td>
          <td className="border px-4 py-2">ง่าย</td>
          <td className="border px-4 py-2">ซับซ้อนขึ้นแต่มีประโยชน์</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความสามารถเชิงบริบท</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">ดีมาก</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold"> ประยุกต์ใช้งานจริง</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ภาษา: Language modeling, Translation, Chatbot</li>
      <li>เสียง: Speech-to-Text, Emotion detection</li>
      <li>เวลา: Stock prediction, Sensor monitoring</li>
      <li>สัญญาณชีวภาพ: ECG, EEG analysis</li>
    </ul>

    <h3 className="text-xl font-semibold"> แหล่งอ้างอิงที่เกี่ยวข้อง</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Stanford CS224n: Natural Language Processing with Deep Learning</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
      <li>“Understanding LSTM Networks” – Christopher Olah</li>
      <li>“Sequence Modeling” – DeepLearning.AI, Andrew Ng</li>
      <li>“Learning Long-Term Dependencies with Gradient Descent is Difficult” – Bengio et al., 1994</li>
    </ul>

  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day27 theme={theme} />
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
        <ScrollSpy_Ai_Day27 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day27_LSTM;
