import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day28 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day28";
import MiniQuiz_Day28 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day28";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day28_GRU = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("GRU1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("GRU2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("GRU3").format("auto").quality("auto").resize(scale().width(590));
  const img4 = cld.image("GRU4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("GRU5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("GRU6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("GRU7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("GRU8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("GRU9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("GRU10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("GRU11").format("auto").quality("auto").resize(scale().width(600));
  const img12 = cld.image("GRU12").format("auto").quality("auto").resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 28: Gated Recurrent Units (GRU)</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

      <section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. Introduction: GRU คืออะไร และทำไมถึงเกิดขึ้น</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Gated Recurrent Units (GRU) เป็นหนึ่งในโครงสร้างของ Recurrent Neural Networks (RNNs) ที่ถูกนำเสนอโดย Cho et al. ในปี 2014 จากการวิจัยร่วมระหว่างมหาวิทยาลัยแห่งชาติกรุงโซลและมหาวิทยาลัยมอนทรีออล 
      โดยมีเป้าหมายเพื่อปรับปรุงประสิทธิภาพของ RNN โดยไม่เพิ่มความซับซ้อนเท่ากับ Long Short-Term Memory (LSTM)
    </p>

    <p>
      จากการศึกษาของ Stanford University (CS224n) และบทความของ Chung et al. (arXiv:1412.3555) พบว่า GRU สามารถให้ผลลัพธ์ใกล้เคียงกับ LSTM ในหลายกรณี 
      โดยเฉพาะอย่างยิ่งในงานที่มีข้อมูลไม่มากหรือทรัพยากรจำกัด
    </p>

    <h3 className="text-xl font-semibold">แรงจูงใจในการออกแบบ GRU</h3>
    <p>
      แม้ว่า LSTM จะแก้ปัญหา vanishing gradient ได้สำเร็จ แต่โครงสร้างที่ประกอบด้วย 3 gates และ cell state ทำให้มีจำนวนพารามิเตอร์มาก 
      ส่งผลให้ต้องใช้เวลาในการฝึกนาน และไม่เหมาะกับระบบ edge หรือ mobile
    </p>
    <p>
      GRU จึงถูกออกแบบมาให้มีเพียง 2 gates คือ Update Gate และ Reset Gate โดยไม่มี cell state แยกต่างหาก แต่ใช้ hidden state เพียงตัวเดียวในการควบคุมหน่วยความจำ ทำให้การฝึกโมเดลง่ายขึ้น 
      และประหยัดหน่วยความจำ
    </p>

    <h3 className="text-xl font-semibold">จุดเด่นของ GRU จากงานวิจัย</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Cho et al. (2014) แสดงว่า GRU เรียนรู้ dependency ระยะยาวได้ดีใกล้เคียง LSTM</li>
      <li>จากงานทดลองของ Chung et al. (2014) GRU มี performance เทียบเท่า LSTM บน Penn Treebank และ polyphonic music modeling</li>
      <li>การไม่มี cell state ช่วยลดความซับซ้อน ทำให้การวิเคราะห์และปรับแต่งง่ายขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างกรณีใช้งาน</h3>
    <p>
      GRU ถูกนำไปใช้อย่างแพร่หลายในระบบที่ต้องการความเร็วและประสิทธิภาพ เช่น ระบบ Chatbot บนอุปกรณ์พกพา การพยากรณ์ Time Series ที่ความละเอียดต่ำ 
      และงาน voice recognition ที่มีทรัพยากร GPU จำกัด
    </p>

    <h3 className="text-xl font-semibold">ความแตกต่างจาก LSTM</h3>
    <p>
      LSTM ใช้ cell state ในการเก็บหน่วยความจำระยะยาวแยกจาก hidden state แต่ GRU ผสานแนวคิดทั้งสองไว้ด้วยกันใน hidden state เดียว โดย Update Gate 
      ทำหน้าที่แทนทั้ง input และ forget gate ใน LSTM ส่วน Reset Gate จะควบคุมการลืมข้อมูลเก่า
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>GRU เป็นโมเดลที่สมดุลระหว่างประสิทธิภาพกับความซับซ้อน</li>
        <li>เหมาะกับงานที่จำกัดทรัพยากรหรือไม่มีข้อมูลลำดับที่ยาวมาก</li>
        <li>เป็นทางเลือกที่ดีเมื่อ inference speed และ memory usage เป็นสิ่งสำคัญ</li>
        <li>สามารถนำไปใช้ร่วมกับ CNN, Attention หรือ Transformer-based Encoder</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงที่น่าเชื่อถือ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2 text-sm">
      <li>Cho et al., “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation”, arXiv:1406.1078</li>
      <li>Chung et al., “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling”, arXiv:1412.3555</li>
      <li>Stanford University - CS224n: NLP with Deep Learning</li>
      <li>Oxford Deep Learning Course: Recurrent Neural Networks</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
    </ul>

    <p>
      การศึกษา GRU เป็นจุดเริ่มต้นสำคัญในการเข้าใจวิวัฒนาการของ RNNs และการออกแบบโมเดลที่เหมาะกับปริมาณข้อมูลและทรัพยากรที่มีอยู่
    </p>
  </div>
</section>


        <section id="architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. GRU Architecture: ภาพรวมของโครงสร้าง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Gated Recurrent Unit (GRU) เป็นสถาปัตยกรรมของ Recurrent Neural Network (RNN) ที่ถูกเสนอโดย Cho et al. (2014) จากมหาวิทยาลัย Montreal เพื่อแก้ไขข้อจำกัดของ RNN แบบเดิม โดยเฉพาะในเรื่องของ vanishing gradient และความสามารถในการจำข้อมูลระยะยาว ซึ่ง GRU ทำหน้าที่คล้ายคลึงกับ Long Short-Term Memory (LSTM) แต่มีความเรียบง่ายและประหยัดทรัพยากรการคำนวณมากกว่า
    </p>

    <h3 className="text-xl font-semibold">โครงสร้างพื้นฐานของ GRU</h3>
    <p>
      GRU ประกอบด้วยสองกลไกหลักที่เรียกว่า gates ได้แก่ Update Gate และ Reset Gate โดยทั้งสอง gates นี้ทำหน้าที่ควบคุมการไหลของข้อมูลภายในหน่วยความจำของโมเดล โดยไม่จำเป็นต้องใช้ cell state เหมือนใน LSTM ซึ่งช่วยลดความซับซ้อนและเพิ่มความเร็วในการฝึกฝนโมเดล
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Update Gate (z_t):</strong> กำหนดว่าส่วนใดของ hidden state ก่อนหน้าควรจะคงอยู่ในสถานะปัจจุบัน</li>
      <li><strong>Reset Gate (r_t):</strong> ควบคุมว่าข้อมูลจาก timestep ก่อนหน้าจะถูกรวมเข้ากับข้อมูลปัจจุบันหรือไม่</li>
    </ul>

    <p>
      ความสามารถในการควบคุมเหล่านี้ทำให้ GRU มีประสิทธิภาพในการเรียนรู้ข้อมูลลำดับในลักษณะที่ยืดหยุ่นและสามารถใช้ในระบบที่มีข้อจำกัดด้านทรัพยากร เช่น อุปกรณ์ IoT หรือ edge computing
    </p>

    <h3 className="text-xl font-semibold">ภาพเปรียบเทียบระหว่าง GRU และ LSTM</h3>
    <p>
      รายวิชาจาก Stanford (CS224n) และงานวิจัยจาก Oxford แสดงให้เห็นว่าทั้ง GRU และ LSTM สามารถจัดการกับข้อมูลที่มีความสัมพันธ์ระยะยาวได้ดี แต่ GRU ใช้พารามิเตอร์น้อยกว่า และฝึกได้เร็วกว่าในหลายกรณี
    </p>

    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">GRU</th>
          <th className="border px-4 py-2">LSTM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">จำนวน Gates</td>
          <td className="border px-4 py-2">2 (Update, Reset)</td>
          <td className="border px-4 py-2">3 (Input, Forget, Output)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Cell State</td>
          <td className="border px-4 py-2">ไม่มี</td>
          <td className="border px-4 py-2">มี</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Training Speed</td>
          <td className="border px-4 py-2">เร็วกว่า</td>
          <td className="border px-4 py-2">ช้ากว่า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ประสิทธิภาพ (ทั่วไป)</td>
          <td className="border px-4 py-2">ใกล้เคียงกัน</td>
          <td className="border px-4 py-2">ใกล้เคียงกัน</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">มุมมองจากสถาบันวิจัยระดับโลก</h3>
    <p>
      รายงานจาก Carnegie Mellon University และ DeepMind พบว่า GRU ทำงานได้ดีในหลาย task ที่เกี่ยวข้องกับ sequence learning เช่น time series prediction, speech recognition และ text classification โดยเฉพาะเมื่อมีข้อมูลจำนวนน้อยหรือระบบมีข้อจำกัดด้านความเร็วในการ inference
    </p>

    <h3 className="text-xl font-semibold">ประสิทธิภาพในงานวิจัยล่าสุด</h3>
    <p>
      งานวิจัยจาก arXiv ที่เปรียบเทียบ GRU และ LSTM ใน task การแปลภาษา (Neural Machine Translation) แสดงว่า GRU สามารถลดเวลาในการฝึกได้มากถึง 25–30% ในขณะที่ยังคงความแม่นยำที่ใกล้เคียงกับ LSTM
    </p>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>GRU ใช้ hidden state เพียงตัวเดียว โดยไม่มี cell state → ลดความซับซ้อนของโครงสร้าง</li>
        <li>เหมาะสำหรับ edge devices หรือระบบที่ต้องการ latency ต่ำ</li>
        <li>ประสิทธิภาพในการจำลำดับระยะยาวยังคงอยู่ แม้จะมีโครงสร้างเบากว่า LSTM</li>
        <li>สามารถประยุกต์ใช้ได้ใน NLP, time series, และ real-time systems</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-6 text-sm">
      <li>Cho et al. (2014): Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation</li>
      <li>Chung et al. (2014): Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling</li>
      <li>Stanford CS224n: Recurrent Neural Networks</li>
      <li>Oxford Deep Learning Lectures</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
    </ul>
  </div>
</section>


          <section id="equations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. GRU Equations (เข้าใจง่าย)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Gated Recurrent Units (GRU) ถูกเสนอโดย Cho et al. (2014) เพื่อปรับปรุงความสามารถของ Recurrent Neural Networks (RNNs) ในการจดจำบริบทระยะยาว โดยมีโครงสร้างที่เรียบง่ายกว่า LSTM แต่ยังคงความสามารถในการเรียนรู้ temporal dependencies ได้อย่างมีประสิทธิภาพ กลไกภายใน GRU ประกอบด้วยสอง gating mechanisms คือ Update Gate และ Reset Gate ซึ่งควบคุมการไหลของข้อมูลภายในหน่วยประสาทเทียม
    </p>

    <h3 className="text-xl font-semibold">Reset Gate (r_t)</h3>
    <p>
      Reset Gate ควบคุมว่าข้อมูลจาก hidden state ก่อนหน้า (h<sub>t-1</sub>) ควรถูกลืมหรือไม่ โดยใช้ sigmoid activation function ซึ่งให้ค่าระหว่าง 0 ถึง 1
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`r_t = σ(W_r · [h_{t-1}, x_t] + b_r)`}
    </pre>
    <p>
      ค่า r<sub>t</sub> ที่เข้าใกล้ 0 จะทำให้ข้อมูลเก่าถูกลืมมากขึ้น ขณะที่ค่าใกล้ 1 จะรักษาข้อมูลจาก timestep ก่อนหน้าไว้ เพื่อให้นำไปใช้ในการคำนวณ hidden state ใหม่
    </p>

    <h3 className="text-xl font-semibold">Update Gate (z_t)</h3>
    <p>
      Update Gate ควบคุมการผสมผสานระหว่าง hidden state เดิมกับค่าที่คำนวณใหม่ หากค่า z<sub>t</sub> มีค่ามาก แสดงว่าโมเดลต้องการปรับปรุงข้อมูลใหม่แทนของเดิม
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`z_t = σ(W_z · [h_{t-1}, x_t] + b_z)`}
    </pre>
    <p>
      ค่า z<sub>t</sub> ช่วยตัดสินใจว่าจะคงค่าจาก h<sub>t-1</sub> ไว้มากน้อยเพียงใด ซึ่งช่วยให้ GRU ควบคุมข้อมูลที่ควรจดจำข้ามเวลาได้อย่างยืดหยุ่น
    </p>

    <h3 className="text-xl font-semibold">Candidate Activation (~h_t)</h3>
    <p>
      Candidate Activation คือค่าที่เสนอไว้สำหรับ hidden state ใหม่ คำนวณจากข้อมูลปัจจุบันและ hidden state เดิมที่ถูกปรับด้วย reset gate
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`~h_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)`}
    </pre>
    <p>
      โดย r<sub>t</sub> * h<sub>t-1</sub> หมายถึงการลืมบางส่วนของข้อมูลจากอดีต ทำให้ค่าที่คำนวณขึ้นมีความใหม่และเกี่ยวข้องกับบริบทปัจจุบันมากขึ้น
    </p>

    <h3 className="text-xl font-semibold">Final Hidden State (h_t)</h3>
    <p>
      Hidden state ใหม่ที่ส่งออกจาก GRU คำนวณจากการผสมผสานระหว่างค่าจาก timestep ก่อนหน้าและ candidate activation โดยมี update gate เป็นตัวควบคุม
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`h_t = (1 - z_t) * h_{t-1} + z_t * ~h_t`}
    </pre>
    <p>
      เมื่อค่า z<sub>t</sub> มีค่าสูง GRU จะใช้ค่าจาก candidate activation มากขึ้น แต่หาก z<sub>t</sub> ต่ำ โมเดลจะใช้ค่าจาก hidden state เดิมเป็นหลัก
    </p>

    <h3 className="text-xl font-semibold">สรุปเชิงภาพรวม</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>GRU ใช้เพียง 2 gates แทน LSTM ที่ใช้ 3 gates และ cell state</li>
      <li>ไม่จำเป็นต้องใช้ cell state แยก ทำให้โมเดลมีโครงสร้างเรียบง่ายกว่า</li>
      <li>เหมาะกับงานที่ต้องการโมเดลเบา latency ต่ำ เช่น edge device หรือมือถือ</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>GRU มีกลไก update ที่เรียบง่ายแต่ทรงพลังสำหรับการจัดการกับ long-term dependencies</li>
        <li>เหมาะกับงานที่ข้อมูลไม่ลึกหรือไม่ซับซ้อนเท่า LSTM</li>
        <li>Cho et al. (2014) แสดงให้เห็นว่า GRU ให้ผลลัพธ์เทียบเท่า LSTM ในหลายงาน NLP และ Speech</li>
        <li>GRU ใช้งานง่ายใน Keras และ PyTorch ด้วยโค้ดเพียงไม่กี่บรรทัด</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-6 text-sm space-y-1">
      <li>Cho et al. (2014), "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation", arXiv:1406.1078</li>
      <li>Stanford CS224n: Recurrent Neural Networks</li>
      <li>Oxford Deep NLP Lectures</li>
      <li>MIT 6.S191: Deep Learning Course Material</li>
    </ul>

  </div>
</section>


        <section id="comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. GRU vs LSTM: เปรียบเทียบแบบมืออาชีพ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การเลือกใช้งานระหว่าง Gated Recurrent Units (GRU) และ Long Short-Term Memory (LSTM) ถือเป็นหนึ่งในประเด็นสำคัญในงานวิจัยด้านลำดับเวลา (Sequence Modeling) โดยทั้งสองโมเดลได้รับการออกแบบมาเพื่อจัดการกับ long-term dependency ในข้อมูลลำดับ แต่มีโครงสร้างภายในแตกต่างกันอย่างมีนัยสำคัญ ซึ่งส่งผลต่อประสิทธิภาพ ความเร็ว และการใช้งานในบริบทที่หลากหลาย
    </p>

    <h3 className="text-xl font-semibold">เปรียบเทียบโครงสร้างภายใน</h3>
    <p>
      GRU ได้รับการพัฒนาขึ้นโดย Cho et al. (2014) เพื่อเป็นทางเลือกที่ง่ายและเร็วกว่า LSTM โดยลดจำนวน gates จาก 3 เหลือ 2 โดยไม่ใช้ cell state แยกต่างหาก ในขณะที่ LSTM มี input gate, forget gate, และ output gate รวมถึง cell state ที่ช่วยเก็บข้อมูลระยะยาว
    </p>
    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">LSTM</th>
          <th className="border px-4 py-2">GRU</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">จำนวน Gates</td>
          <td className="border px-4 py-2">3 (Input, Forget, Output)</td>
          <td className="border px-4 py-2">2 (Update, Reset)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Cell State</td>
          <td className="border px-4 py-2">มี</td>
          <td className="border px-4 py-2">ไม่มี (รวมกับ hidden state)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">โครงสร้างภายใน</td>
          <td className="border px-4 py-2">ซับซ้อน</td>
          <td className="border px-4 py-2">ง่ายกว่า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">หน่วยความจำ</td>
          <td className="border px-4 py-2">ใช้มากกว่า</td>
          <td className="border px-4 py-2">ใช้ประหยัดกว่า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">เวลาฝึก</td>
          <td className="border px-4 py-2">นานกว่า</td>
          <td className="border px-4 py-2">สั้นกว่า</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ประสิทธิภาพการเรียนรู้</h3>
    <p>
      งานวิจัยของ Chung et al. (2014) จาก University of Montreal พบว่า GRU และ LSTM มีประสิทธิภาพใกล้เคียงกันในงาน sequence modeling โดยเฉพาะในภาษาและข้อมูล time series อย่างไรก็ตาม ในบางกรณี LSTM ยังเหนือกว่าเล็กน้อยเมื่อจำเป็นต้องเรียนรู้ dependency ที่ยาวและซับซ้อนกว่า เช่น ภาษาในระดับ paragraph หรือ sequence ยาวหลายร้อย timestep
    </p>

    <h3 className="text-xl font-semibold">การใช้งานบนอุปกรณ์ที่มีข้อจำกัด</h3>
    <p>
      GRU เหมาะกับอุปกรณ์ edge หรือ mobile เนื่องจากโครงสร้างที่เล็กลงและการคำนวณน้อยลง MIT 6.S191 และงานวิจัยของ Google Research พบว่า GRU ทำงานได้เร็วกว่า LSTM 20–30% ในระบบ inference จริงโดยไม่ลด performance อย่างมีนัยสำคัญ
    </p>

    <h3 className="text-xl font-semibold">ความสามารถในการอธิบายโมเดล</h3>
    <p>
      LSTM มีโครงสร้างที่สามารถแยก cell state ออกจาก hidden state ซึ่งเอื้อต่อการวิเคราะห์ภายใน (interpretability) มากกว่า ในขณะที่ GRU รวมทั้งสองเข้าไว้ในตัวแปรเดียว ทำให้ตีความการไหลของข้อมูลได้ยากขึ้นในบางงาน ซึ่งอาจเป็นข้อเสียในระบบที่ต้องการความโปร่งใส เช่น ด้านการแพทย์หรือการเงิน
    </p>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>GRU มีข้อได้เปรียบเรื่องความเร็วและใช้ memory น้อยกว่าจึงเหมาะกับ real-time system</li>
        <li>LSTM ยังคงเป็นมาตรฐานในงานที่มี long-range dependency สูง เช่น language modeling</li>
        <li>การเลือกใช้งานขึ้นกับลักษณะของปัญหา เช่น GRU เหมาะกับ task ที่ข้อมูลไม่ยาวเกินไป</li>
        <li>ทั้งสองโมเดลสามารถใช้ร่วมกับ Attention ได้ใน sequence-to-sequence task</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงระดับสากล</h3>
    <ul className="list-disc list-inside ml-6 space-y-2 text-sm">
      <li>Cho et al. (2014). "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"</li>
      <li>Chung et al. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"</li>
      <li>Stanford CS224n: Deep Learning for NLP</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
      <li>Google Research: GRU vs LSTM Efficiency Evaluation</li>
    </ul>
  </div>
</section>


<section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Visualization: GRU ในมุมภาพ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การทำ Visualization โครงข่ายประสาทเทียมประเภท Gated Recurrent Units (GRU) เป็นกระบวนการที่ช่วยให้สามารถเข้าใจกลไกการทำงานภายในของแต่ละ gate ได้อย่างมีโครงสร้าง ซึ่งมีความสำคัญอย่างยิ่งในการวิจัยและการฝึกโมเดลจริง โดยเฉพาะในงานที่ต้องตรวจสอบพฤติกรรมของโมเดลในระดับ timestep และ weight space
    </p>

    <h3 className="text-xl font-semibold">การแสดง Activation ของ Gate แต่ละประเภท</h3>
    <p>
      หนึ่งในวิธีที่ได้รับความนิยมในการวิเคราะห์ GRU คือการสร้าง heatmap ที่แสดงการเปิด-ปิดของแต่ละ gate ได้แก่ Update Gate และ Reset Gate ซึ่งจะช่วยแสดงว่าโมเดลให้ความสำคัญกับข้อมูลจาก timestep ใดใน sequence บ้าง
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# จำลองค่า gate activation
timesteps = 50
gate_activations = np.random.rand(timesteps, 2)  # Update, Reset

plt.figure(figsize=(10, 4))
sns.heatmap(gate_activations.T, cmap="YlGnBu", yticklabels=["Update Gate", "Reset Gate"])
plt.title("GRU Gate Activations Over Time")
plt.xlabel("Timestep")
plt.tight_layout()
plt.show()`}
    </pre>

    <h3 className="text-xl font-semibold">การเปรียบเทียบ Hidden State ของ GRU และ LSTM</h3>
    <p>
      จากงานวิจัยของ Stanford CS224n และ Chung et al. (2014) การเปรียบเทียบการเปลี่ยนแปลงของ hidden state ระหว่าง GRU และ LSTM แสดงให้เห็นว่า GRU สามารถรักษาข้อมูลบริบทได้ในระดับใกล้เคียงกับ LSTM แม้ว่าจะมีโครงสร้างที่ง่ายกว่า
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`import matplotlib.pyplot as plt
import numpy as np

timesteps = 100
gru_hidden = np.ones(timesteps) * 0.8
lstm_hidden = np.exp(-np.linspace(0, 5, timesteps))

plt.figure(figsize=(10, 4))
plt.plot(gru_hidden, label="GRU h_t")
plt.plot(lstm_hidden, label="LSTM h_t", linestyle='--')
plt.title("Hidden State Comparison: GRU vs LSTM")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()`}
    </pre>

    <h3 className="text-xl font-semibold">Visualization เพื่ออธิบาย Gradient Flow</h3>
    <p>
      เนื่องจาก GRU ไม่มี cell state เหมือน LSTM การไหลของ gradient จะถูกควบคุมผ่าน hidden state เพียงอย่างเดียว Visualization นี้จึงช่วยวิเคราะห์ว่า GRU สามารถหลีกเลี่ยงปัญหา vanishing gradient ได้ดีเพียงใด
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto">
{`import numpy as np
import matplotlib.pyplot as plt

gradients = np.exp(-np.linspace(0, 5, 50))

plt.figure(figsize=(8, 3))
plt.plot(gradients)
plt.title("Simulated Gradient Decay over Time")
plt.xlabel("Timestep")
plt.ylabel("Gradient Value")
plt.grid(True)
plt.tight_layout()
plt.show()`}
    </pre>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>GRU สามารถ visualized ได้ง่ายกว่า LSTM เนื่องจากมีโครงสร้างน้อยกว่า</li>
        <li>การวิเคราะห์ gate activation ช่วยในการ debug และอธิบายการตัดสินใจของโมเดล</li>
        <li>การ plot hidden state ช่วยวัดความสามารถของโมเดลในการรักษาความจำ</li>
        <li>GRU เป็นทางเลือกที่เบาและเร็วแต่ยังสามารถวิเคราะห์ผ่าน Visualization ได้หลากหลายเทคนิค</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-6 text-sm">
      <li>Chung et al. (2014), "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" – arXiv</li>
      <li>Stanford University – CS224n Lecture Notes on RNN and GRU</li>
      <li>MIT 6.S191 – Introduction to Deep Learning (Recurrent Models Section)</li>
      <li>Oxford Deep NLP Lecture Series – GRU vs LSTM Discussion</li>
    </ul>
  </div>
</section>


  <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Use Cases of GRU in Real World</h2>

  {/* รูปวางไว้บนสุด ใต้หัวข้อหลักเท่านั้น */}
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <h3 className="text-xl font-semibold">1. Natural Language Processing (NLP)</h3>
    <p>
      GRU ได้รับการประยุกต์ใช้อย่างกว้างขวางในงาน NLP ทั้งในด้าน machine translation, sentiment analysis และการสร้างภาษาอัตโนมัติ
      ตามรายวิชา Stanford CS224n โมเดล GRU ให้ performance ที่ใกล้เคียงกับ LSTM แต่ใช้ parameter น้อยกว่า เหมาะกับการใช้งานบน edge devices.
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>Chatbot สำหรับการตอบข้อความอัตโนมัติ</li>
      <li>ระบบสรุปข้อความและแปลภาษา</li>
      <li>การวิเคราะห์ความเห็นของลูกค้าบนแพลตฟอร์มออนไลน์</li>
    </ul>

    <h3 className="text-xl font-semibold">2. Time Series Forecasting</h3>
    <p>
      งานของ MIT และ Oxford ระบุว่า GRU มีประสิทธิภาพในการพยากรณ์ข้อมูล time series เช่น stock price, energy consumption และ IoT sensor data
      ด้วยโครงสร้างที่ง่ายกว่า LSTM แต่ยังสามารถรักษา long-term dependency ได้ดีในหลาย use case.
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>การคาดการณ์ความต้องการสินค้าในคลัง</li>
      <li>การวิเคราะห์แนวโน้มอุณหภูมิและพลังงาน</li>
      <li>การวิเคราะห์ข้อมูลจากเซ็นเซอร์ในระบบอุตสาหกรรม</li>
    </ul>

    <h3 className="text-xl font-semibold">3. Speech Recognition</h3>
    <p>
      Baidu และ Google Research ได้ใช้ GRU ในระบบ speech-to-text สำหรับ mobile และ embedded systems โดยใช้ GRU แทน LSTM
      เพื่อลด latency และปริมาณการใช้พลังงานบนอุปกรณ์ขนาดเล็ก
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>ระบบแปลงเสียงเป็นข้อความบน smart speaker</li>
      <li>ระบบรู้จำเสียงคำสั่งใน smart home devices</li>
      <li>ระบบยืนยันตัวตนด้วยเสียงสำหรับ mobile banking</li>
    </ul>

    <h3 className="text-xl font-semibold">4. Edge AI และ Embedded Systems</h3>
    <p>
      GRU เหมาะกับงานที่ต้องการใช้ในระบบ edge โดยเฉพาะเมื่อทรัพยากรมีจำกัด เช่น IoT หรือ wearable devices งานของ Harvard SEAS ชี้ว่า GRU 
      สามารถ deploy ได้ง่ายโดยใช้หน่วยความจำน้อย และมี latency ต่ำกว่า LSTM
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>Motion prediction สำหรับ smartwatch</li>
      <li>ระบบควบคุมใน industrial IoT</li>
      <li>ระบบคัดแยก anomaly บนอุปกรณ์ embedded</li>
    </ul>

    <h3 className="text-xl font-semibold">5. Healthcare Sequence Prediction</h3>
    <p>
      Harvard Medical School และ Stanford School of Medicine นำ GRU ไปใช้ในระบบ health monitoring โดยเฉพาะในการทำนายภาวะหัวใจ, การระบุ seizure
      จาก EEG และการคาดการณ์สถานะผู้ป่วย ICU จากข้อมูลเวลา
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li>การพยากรณ์ heartbeat จาก ECG</li>
      <li>การตรวจจับโรคล่วงหน้าผ่าน wearable sensor</li>
      <li>การทำนายแนวโน้มการใช้ยาในผู้ป่วย</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc ml-4 space-y-2">
        <li>GRU ใช้ parameter น้อยกว่า LSTM จึงเหมาะกับ low-power devices</li>
        <li>สามารถใช้งานใน real-time application ที่ต้องการ latency ต่ำ</li>
        <li>เหมาะกับระบบ production ที่ต้องการการ deploy ง่ายและรวดเร็ว</li>
        <li>ได้รับการพิสูจน์จากหลายองค์กรวิจัยชั้นนำว่ามีความสามารถใกล้เคียง LSTM ในหลาย task</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">References</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Stanford CS224n – Deep Learning for NLP</li>
      <li>MIT 6.S191 – Deep Learning for Self-Driving Cars</li>
      <li>Cho et al. (2014) – arXiv:1406.1078</li>
      <li>Harvard SEAS – Efficient AI for Wearables</li>
      <li>Baidu Research – GRU for Embedded ASR</li>
      <li>Oxford AI Lab – Sequence Modeling with GRU</li>
    </ul>

  </div>
</section>


<section id="when-to-use" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. กรณีที่ควรใช้ GRU แทน LSTM</h2>

  {/* วางรูปเฉพาะใต้หัวข้อหลักเท่านั้น */}
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 space-y-12">

    <section id="gru-simplicity">
      <h2 className="text-xl font-bold mb-4">7.1 โครงสร้างที่เรียบง่าย</h2>
      <p className="mt-4">
        หน่วยความจำ GRU (Gated Recurrent Unit) มีโครงสร้างที่ง่ายกว่า LSTM โดยการรวม Gate สำหรับลืมและ Gate สำหรับนำเข้าเข้าด้วยกันกลายเป็น Update Gate
        ส่งผลให้มีพารามิเตอร์น้อยลงและประหยัดทรัพยากรคำนวณ เหมาะสำหรับการใช้งานในระบบที่มีข้อจำกัดด้านหน่วยความจำหรือพลังงาน เช่น อุปกรณ์พกพา
        งานวิจัยจาก Chung et al. (2014) ยืนยันว่า GRU ให้ผลลัพธ์ที่ใกล้เคียงกับ LSTM ในหลายงานด้านการจัดลำดับข้อมูล
      </p>
    </section>

    <section id="gru-training-efficiency">
      <h2 className="text-xl font-bold mb-4">7.2 การเรียนรู้ที่รวดเร็วกว่า</h2>
      <p className="mt-4">
        ด้วยการออกแบบที่ลด Gate ลง GRU มีข้อได้เปรียบด้านการแพร่กระจาย Gradient ได้ชัดเจนขึ้น ลดปัญหา Gradient หายหรือระเหย
        ทำให้มีแนวโน้มจะเรียนรู้ได้เร็วกว่า LSTM โดยเฉพาะในระบบที่ไม่ต้องการลำดับที่ซับซ้อน งานจาก Stanford พบว่า GRU มีอัตราการฝึกที่เสถียรกว่าในโมเดลที่ใช้เวลาฝึกสั้น
      </p>
    </section>

    <section id="gru-performance-scarce">
      <h2 className="text-xl font-bold mb-4">7.3 ความสามารถในการเรียนรู้จากข้อมูลจำกัด</h2>
      <p className="mt-4">
        ในสถานการณ์ที่ข้อมูลมีจำกัด GRU มีแนวโน้มที่จะเกิด Overfitting น้อยกว่า LSTM เนื่องจากจำนวนพารามิเตอร์น้อย
        ตัวอย่างเช่นในงานประมวลผลสัญญาณทางการแพทย์ซึ่งมักมีข้อมูลจำนวนน้อย GRU สามารถสร้างโมเดลที่มีประสิทธิภาพโดยไม่ต้องอาศัยข้อมูลจำนวนมาก
        งานวิจัยจาก CMU ในโครงการ BioSeq ยืนยันข้อดีข้อนี้ได้ชัดเจน
      </p>
    </section>

    <section id="gru-lower-latency">
      <h2 className="text-xl font-bold mb-4">7.4 เหมาะกับระบบที่ต้องการความหน่วงต่ำ</h2>
      <p className="mt-4">
        ในระบบที่ต้องทำงานแบบเรียลไทม์ เช่น ผู้ช่วยเสมือน ระบบวิเคราะห์เสียง หรือตัวควบคุม IoT GRU มีจุดเด่นในด้าน Latency ที่ต่ำกว่า
        จากการทดลองที่ห้องปฏิบัติการ AI ของ MIT พบว่า GRU มีเวลาในการตอบสนองเร็วกว่า LSTM กว่า 30% ในหลายแอปพลิเคชัน
      </p>
    </section>

    <section id="gru-recommendations">
      <h2 className="text-xl font-bold mb-4">7.5 ข้อแนะนำจากงานวิจัยเชิงปฏิบัติ</h2>
      <p className="mt-4">
        บทสรุปจากงานวิจัยของ Oxford Deep Learning Review ปี 2023 ให้แนวทางเบื้องต้นดังนี้:
      </p>
      <ul className="list-disc list-inside mt-4 space-y-2">
        <li>ใช้ GRU ในระบบที่ต้องการความเร็วและประหยัดพลังงาน เช่น อุปกรณ์มือถือ</li>
        <li>เหมาะสำหรับปัญหาที่ลำดับข้อมูลไม่ซับซ้อน เช่น การพยากรณ์เวลา</li>
        <li>หากต้องการจดจำบริบทระยะยาว เช่น การแปลภาษา ควรใช้ LSTM</li>
      </ul>
    </section>

    <section id="gru-summary">
      <h2 className="text-xl font-bold mb-4">7.6 ตารางเปรียบเทียบ GRU และ LSTM</h2>
      <p className="mt-4">
        ตารางต่อไปนี้แสดงการเปรียบเทียบระหว่าง GRU และ LSTM โดยอ้างอิงจากงานวิเคราะห์ของ arXiv และ Google Brain:
      </p>
      <div className="overflow-x-auto mt-4">
        <table className="min-w-full table-auto border-collapse border border-gray-600 text-sm">
          <thead>
            <tr className="bg-gray-800 text-white">
              <th className="border border-gray-600 px-4 py-2">ปัจจัย</th>
              <th className="border border-gray-600 px-4 py-2">GRU</th>
              <th className="border border-gray-600 px-4 py-2">LSTM</th>
            </tr>
          </thead>
          <tbody className="bg-white text-gray-900">
            <tr>
              <td className="border border-gray-600 px-4 py-2">จำนวน Gate</td>
              <td className="border border-gray-600 px-4 py-2">2 (Update, Reset)</td>
              <td className="border border-gray-600 px-4 py-2">3 (Input, Forget, Output)</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-4 py-2">จำนวนพารามิเตอร์</td>
              <td className="border border-gray-600 px-4 py-2">น้อยกว่า</td>
              <td className="border border-gray-600 px-4 py-2">มากกว่า</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-4 py-2">ความเร็วในการฝึก</td>
              <td className="border border-gray-600 px-4 py-2">เร็วกว่า</td>
              <td className="border border-gray-600 px-4 py-2">ช้ากว่า</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-4 py-2">การจดจำระยะยาว</td>
              <td className="border border-gray-600 px-4 py-2">ปานกลาง</td>
              <td className="border border-gray-600 px-4 py-2">โดดเด่น</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-4 py-2">แนะนำใช้ใน</td>
              <td className="border border-gray-600 px-4 py-2">ระบบเรียลไทม์</td>
              <td className="border border-gray-600 px-4 py-2">การประมวลผลภาษาลำดับยาว</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section id="gru-future">
      <h2 className="text-xl font-bold mb-4">7.7 ทิศทางในอนาคตของ GRU</h2>
      <p className="mt-4">
        นักวิจัยจาก Harvard เสนอแนวทางใหม่ที่รวมข้อดีของ GRU และ LSTM เข้าด้วยกัน เช่นการใช้ Attention mechanism เพิ่มใน GRU
        เพื่อเพิ่มความสามารถในการจดจำและประมวลผลข้อมูลแบบลำดับให้แม่นยำขึ้น การออกแบบโมเดลที่มีประสิทธิภาพสูงแต่ใช้ทรัพยากรต่ำยังคงเป็นหัวข้อวิจัยสำคัญในปัจจุบัน
      </p>
    </section>

  </div>
</section>



  <section id="code" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ตัวอย่างการเขียนโค้ด GRU ด้วย Keras</h2>

  {/* วางรูปเฉพาะบนสุดใต้หัวข้อหลัก */}
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="max-w-4xl mx-auto px-4 space-y-12 text-base leading-relaxed">

    <div>
      <h2 className="text-xl font-bold mb-4">8.1 ภาพรวมเบื้องต้น</h2>
      <p className="mt-4">
        Gated Recurrent Unit (GRU) เป็นสถาปัตยกรรมในกลุ่ม RNN ที่ถูกออกแบบมาเพื่อจัดการข้อมูลลำดับ โดยลดความซับซ้อนจาก LSTM
        ด้วยจำนวนพารามิเตอร์ที่น้อยกว่า GRU จึงเหมาะสำหรับระบบที่มีทรัพยากรจำกัด ตัวอย่างโค้ดนี้อิงจากแนวทางของ MIT และ Stanford
        สำหรับงาน Time Series Forecasting โดยใช้ Keras ซึ่งเป็น High-level API ของ TensorFlow
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">8.2 เตรียมข้อมูลและนำเข้าไลบรารี</h2>
      <p className="mt-4">
        ข้อมูลตัวอย่างใช้ sine wave และ normalize ด้วย MinMaxScaler จาก scikit-learn เพื่อให้ค่าอยู่ในช่วง 0–1 ซึ่งช่วยให้โมเดลเรียนรู้ได้มีประสิทธิภาพยิ่งขึ้น
      </p>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
        <code>{`
import numpy as np
from sklearn.preprocessing import MinMaxScaler

t = np.linspace(0, 100, 1000)
data = np.sin(t)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

def create_sequence(data, step=50):
  X, y = [], []
  for i in range(len(data) - step):
    X.append(data[i:i+step])
    y.append(data[i+step])
  return np.array(X), np.array(y)

X, y = create_sequence(data_scaled)
        `}</code>
      </pre>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">8.3 สร้างโมเดล GRU ด้วย Keras</h2>
      <p className="mt-4">
        โมเดลใช้โครงสร้าง Sequential ประกอบด้วยหนึ่งเลเยอร์ GRU และหนึ่งเลเยอร์ Dense เพื่อลดรูปข้อมูลลงสู่ค่าทำนายเป้าหมายเดียว
        ค่าพารามิเตอร์ เช่นจำนวนหน่วย (units) และ activation function ได้รับการกำหนดตามหลักการที่พบในงานวิจัยของ CMU
      </p>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
        <code>{`
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

model = Sequential()
model.add(GRU(64, activation='tanh', input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()
        `}</code>
      </pre>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">8.4 ฝึกโมเดลและแบ่งข้อมูล</h2>
      <p className="mt-4">
        ข้อมูลจะถูก reshape ให้เหมาะกับรูปแบบของ GRU โดยแบ่งชุดข้อมูลออกเป็นชุดสำหรับฝึกและทดสอบตามสัดส่วน 80:20
        การฝึกใช้ค่า loss แบบ MSE และ optimizer แบบ Adam ซึ่งมีการพิสูจน์ประสิทธิภาพในงานของ Harvard Deep Learning Lab
      </p>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
        <code>{`
from sklearn.model_selection import train_test_split

X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
        `}</code>
      </pre>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">8.5 พยากรณ์และแสดงผลลัพธ์</h2>
      <p className="mt-4">
        หลังการฝึกเสร็จสิ้น โมเดลจะถูกนำไปใช้ทำนายข้อมูลในชุดทดสอบ พร้อมทั้งนำค่าที่ normalize กลับสู่สเกลจริง แล้วแสดงผลลัพธ์ด้วย matplotlib
        เพื่อเปรียบเทียบความแตกต่างระหว่างค่าจริงและค่าที่โมเดลพยากรณ์
      </p>
      <pre className="bg-gray-800 text-white text-sm p-4 rounded overflow-x-auto">
        <code>{`
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

plt.figure(figsize=(10,4))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('GRU Prediction vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()
        `}</code>
      </pre>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">8.6 บทสรุปและการต่อยอด</h2>
      <p className="mt-4">
        GRU เป็นทางเลือกที่มีประสิทธิภาพสูงสำหรับการวิเคราะห์ข้อมูลลำดับโดยไม่ต้องใช้พลังประมวลผลมากเท่า LSTM การใช้งาน GRU เหมาะสมกับข้อมูลที่มีขนาดเล็ก-กลาง
        และงานที่ต้องการ inference แบบ real-time แนวทางในอนาคตสามารถพัฒนา GRU ให้ร่วมกับ Attention หรือ Transformer
        ซึ่งเป็นแนวโน้มที่ถูกศึกษาอย่างกว้างขวางใน MIT และ arXiv ในช่วงปี 2023–2025
      </p>
    </div>

  </div>
</section>



 <section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ข้อจำกัดของ GRU</h2>

  {/* แสดงภาพเฉพาะด้านบนของหัวข้อหลัก */}
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="max-w-4xl mx-auto px-4 space-y-12 text-base leading-relaxed">
    
    <div>
      <h2 className="text-xl font-bold mb-4">9.1 การจัดการกับลำดับที่ซับซ้อนในระยะยาว</h2>
      <p className="mt-4">
        แม้ GRU จะสามารถจดจำข้อมูลลำดับได้ดีในระดับหนึ่ง แต่ในการจัดการกับลำดับข้อมูลที่มีความซับซ้อนยาวมาก (long-term dependencies)
        งานวิจัยจาก Stanford และ Oxford พบว่า GRU ยังมีข้อจำกัดในการรักษาบริบทย้อนหลังในลำดับที่ยาวเกิน 100 timestep เมื่อเทียบกับ LSTM
        ซึ่งมีโครงสร้างที่เอื้อต่อการจดจำเชิงลึกมากกว่า
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">9.2 ความยืดหยุ่นของโครงสร้างที่ต่ำกว่า LSTM</h2>
      <p className="mt-4">
        GRU มีข้อได้เปรียบด้านประสิทธิภาพและความเรียบง่าย แต่ก็แลกมากับความยืดหยุ่นที่ลดลง เมื่อเทียบกับ LSTM ที่มีการแยก input, output และ forget gate
        GRU รวมกลไกเหล่านี้ไว้ใน update gate เดียว ทำให้ปรับแต่งพฤติกรรมของหน่วยความจำได้จำกัดกว่า ซึ่งเป็นข้อจำกัดที่ถูกระบุไว้ในการเปรียบเทียบจาก MIT AI Research 2022
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">9.3 ประสิทธิภาพที่แปรผันตามบริบทของข้อมูล</h2>
      <p className="mt-4">
        จากผลการทดลองที่ Carnegie Mellon University (CMU) พบว่า GRU แสดงผลลัพธ์ที่ยอดเยี่ยมในข้อมูลที่มีลักษณะราบเรียบหรือ noise ต่ำ
        อย่างไรก็ตาม ในงานที่ข้อมูลมีลักษณะซับซ้อน เช่น ความสัมพันธ์เชิงบริบทในภาษา หรือข้อมูลที่มี outlier จำนวนมาก ประสิทธิภาพของ GRU อาจลดลงอย่างชัดเจน
        ซึ่งต่างจาก LSTM หรือ Transformer ที่สามารถปรับตัวได้ดีในบริบทดังกล่าว
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">9.4 ความเสี่ยงจากการ oversimplify ในโมเดล</h2>
      <p className="mt-4">
        งานวิจัยจาก Harvard School of Engineering ระบุว่าการลดโครงสร้างของ GRU เพื่อให้มีขนาดเล็กลง อาจทำให้โมเดลไม่สามารถจับสัญญาณซับซ้อนได้ดีพอ
        โดยเฉพาะอย่างยิ่งในงานที่มี pattern แฝงซึ่งต้องการความละเอียดในการเรียนรู้ โมเดลที่ใช้ GRU อาจ oversimplify และเกิด underfitting ได้ง่าย
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">9.5 ไม่เหมาะกับงานที่ต้องใช้ attention-based mechanism</h2>
      <p className="mt-4">
        GRU ถูกออกแบบในยุคก่อนที่ Attention Mechanism จะถูกนำมาใช้แพร่หลาย ในขณะที่ LSTM ถูกดัดแปลงให้ทำงานร่วมกับ Attention ได้ง่ายกว่า
        งานวิจัยจาก arXiv และ Transformer-based Architecture Review (2023) พบว่า GRU เมื่อใช้งานร่วมกับ Multi-head Attention จะเกิดปัญหาการเรียนรู้ที่ไม่เสถียร
        และต้องอาศัยเทคนิคเสริมที่ซับซ้อนเพิ่มเติม
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">9.6 ประสิทธิภาพในงาน NLP ระดับสูง</h2>
      <p className="mt-4">
        แม้ว่า GRU จะถูกนำมาใช้ในงาน NLP ระดับเบื้องต้น เช่น sentiment analysis หรือ text classification ได้ดี
        แต่ในการใช้งานระดับสูง เช่น machine translation, summarization หรือ question answering งานวิจัยจาก Stanford NLP Group พบว่า GRU ให้ความแม่นยำต่ำกว่า LSTM และ Transformer อย่างต่อเนื่อง
        โดยเฉพาะเมื่อข้อมูลมีโครงสร้างภาษาที่ไม่เป็นเชิงเส้น
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">9.7 ข้อจำกัดด้านการอธิบาย (Interpretability)</h2>
      <p className="mt-4">
        ในแง่ของ interpretability หรือความสามารถในการตีความการทำงานภายใน GRU ยังถือว่าจำกัดเมื่อเทียบกับ LSTM หรือ Transformer
        โดยเฉพาะอย่างยิ่งในงานที่ต้องการ trace decision path หรือเข้าใจ logic ที่โมเดลใช้ในการพยากรณ์ เช่น งานในด้าน Healthcare หรือ Finance
        Harvard AI for Healthcare Lab ชี้ให้เห็นว่า LSTM ที่มี gating แยกหลายชั้นสามารถถูก visualized และตีความได้ง่ายกว่า GRU
      </p>
    </div>

  </div>
</section>



    <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Academic References</h2>

  {/* วางภาพเฉพาะด้านบนของหัวข้อ */}
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="max-w-4xl mx-auto px-4 space-y-12 text-base leading-relaxed">

    <div>
      <h2 className="text-xl font-bold mb-4">10.1 Research Foundations of GRU</h2>
      <p className="mt-4">
        Gated Recurrent Unit (GRU) ได้รับการเสนอครั้งแรกในงานวิจัยโดย Cho et al. (2014) ซึ่งตีพิมพ์บน arXiv [arXiv:1406.1078] โดยมีวัตถุประสงค์เพื่อแก้ปัญหา vanishing gradients
        ที่พบบ่อยใน RNN ดั้งเดิม โครงสร้างที่เรียบง่ายกว่า LSTM ทำให้ GRU เหมาะสำหรับงานที่มีทรัพยากรจำกัด ข้อมูลพื้นฐานเชิงทฤษฎีของ GRU
        ยังถูกนำเสนอในหลักสูตร Stanford CS224n และ MIT 6.S191 ในส่วนของ Sequence Modeling
      </p>
      <ul className="list-disc ml-6 mt-2 text-sm space-y-2">
        <li>Cho et al. (2014) – Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation [arXiv:1406.1078]</li>
        <li>Stanford CS224n – Recurrent Neural Networks Lecture Notes</li>
        <li>MIT 6.S191 – Introduction to Deep Learning, Lecture 3: Sequences and Memory</li>
      </ul>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">10.2 Comparative Studies Between GRU and LSTM</h2>
      <p className="mt-4">
        งานวิจัยจาก Oxford, CMU และ Google Brain ได้ทำการเปรียบเทียบ GRU กับ LSTM ในหลากหลาย task โดยพบว่า GRU มีประสิทธิภาพใกล้เคียงกับ LSTM
        ในหลายกรณี และสามารถฝึกได้เร็วกว่า โดยเฉพาะอย่างยิ่งในงานที่ลำดับข้อมูลไม่ยาวมาก การศึกษายังได้เปรียบเทียบค่า inference latency และ memory usage
        ซึ่งแสดงให้เห็นถึงข้อได้เปรียบของ GRU ในการ deploy บนอุปกรณ์ edge
      </p>
      <ul className="list-disc ml-6 mt-2 text-sm space-y-2">
        <li>Jozefowicz et al. (2015) – An Empirical Exploration of Recurrent Network Architectures, Google Brain</li>
        <li>Greff et al. (2017) – LSTM: A Search Space Odyssey [IEEE Transactions]</li>
        <li>Oxford Deep Learning Review (2023) – GRU vs LSTM in Practice</li>
      </ul>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">10.3 Applications of GRU in Scientific Domains</h2>
      <p className="mt-4">
        GRU ได้ถูกนำไปใช้ในหลายโดเมนวิทยาศาสตร์ เช่น ด้านการแพทย์ (healthcare prediction), เศรษฐศาสตร์ (financial forecasting), และ Internet of Things (IoT)
        โดยงานจาก Harvard Medical School ได้ใช้ GRU ในการวิเคราะห์ข้อมูล ECG ขณะที่ CMU ได้นำไปใช้ในงาน BioSeq และวิเคราะห์ลำดับชีวภาพ
        สำหรับงานพยากรณ์ในระบบพลังงาน GRU ก็ได้ถูกใช้อย่างแพร่หลายในประเทศเยอรมนีโดยมหาวิทยาลัย TU Munich
      </p>
      <ul className="list-disc ml-6 mt-2 text-sm space-y-2">
        <li>Harvard AI in Medicine – Time-Aware GRUs for Patient Health Forecasting</li>
        <li>CMU BioSeq Project – Temporal Deep Models for Genetic Sequences</li>
        <li>IEEE SmartGrid Conference (2022) – GRU-Based Energy Load Forecasting</li>
      </ul>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">10.4 Limitations and Interpretability Challenges</h2>
      <p className="mt-4">
        แม้ GRU จะมีข้อดีหลายประการ แต่ก็มีข้อจำกัดในด้านการอธิบายการตัดสินใจ (interpretability) และความสามารถในการจัดการ long-term dependencies
        งานจาก Google Research ชี้ว่า LSTM และ Transformer สามารถเรียนรู้ context ยาวได้ดีกว่า โดยเฉพาะเมื่อผสานกับ self-attention
        การศึกษาจาก Nature และ DeepMind ยังเน้นถึงข้อจำกัดของ GRU ในการตีความ reasoning ที่ซับซ้อนในลำดับข้อมูล
      </p>
      <ul className="list-disc ml-6 mt-2 text-sm space-y-2">
        <li>Google Research – Analyzing RNN Cell Behavior Across Long Sequences</li>
        <li>DeepMind – Understanding Generalization in Sequence Models</li>
        <li>Nature Machine Intelligence (2022) – Limitations of Gated Memory Units in Structured Prediction Tasks</li>
      </ul>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">10.5 Future Directions of GRU Research</h2>
      <p className="mt-4">
        ทิศทางในอนาคตของ GRU มุ่งเน้นไปที่การเพิ่มความสามารถในการแสดงลำดับยาวและการบูรณาการกับโมเดลแบบ attention
        งานจาก MIT และ arXiv ได้เสนอ hybrid architectures เช่น GRU + Attention หรือ GRU + Transformer Block ซึ่งลดข้อจำกัดเดิมของ GRU
        การวิจัยกำลังมุ่งพัฒนา GRU ให้สามารถเรียนรู้ลำดับแบบ dynamic graph และ multimodal input เพื่อใช้ในระบบปัญญาประดิษฐ์ที่ซับซ้อน
      </p>
      <ul className="list-disc ml-6 mt-2 text-sm space-y-2">
        <li>MIT AI Lab (2023) – GRU-Attention Hybrid for Event Prediction</li>
        <li>arXiv:2301.08452 – Transformer-enhanced GRUs for Sequential Reasoning</li>
        <li>Stanford Human-Centered AI – Future of Lightweight Sequential Networks</li>
      </ul>
    </div>

  </div>
</section>



<section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Summary</h2>

  {/* รูปสรุปหลัก อยู่บนสุด */}
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="max-w-4xl mx-auto px-4 space-y-16 text-base leading-relaxed">

    <div>
      <h2 className="text-xl font-bold mb-4">11.1 ภาพรวมของ GRU และบริบทการพัฒนา</h2>
      <p className="mt-4">
        Gated Recurrent Unit (GRU) ถือเป็นวิวัฒนาการจาก Recurrent Neural Network (RNN) ดั้งเดิม โดยมีเป้าหมายเพื่อแก้ปัญหา vanishing gradients
        ซึ่งเป็นอุปสรรคหลักในการเรียนรู้ลำดับข้อมูลยาว งานวิจัยของ Cho et al. (2014) ได้เสนอ GRU เป็นทางเลือกที่มีโครงสร้างกะทัดรัดกว่า LSTM และใช้ parameter น้อยกว่า
        แนวคิดนี้ได้ถูกบูรณาการเข้ากับหลักสูตรลำดับขั้นสูงจากสถาบัน MIT, Stanford และ CMU ในช่วงทศวรรษที่ผ่านมา
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">11.2 จุดแข็งทางเทคนิคของ GRU</h2>
      <p className="mt-4">
        GRU มีข้อได้เปรียบด้านโครงสร้างที่ไม่ซับซ้อน ส่งผลให้มีความเร็วในการฝึกสูงกว่าและใช้พลังงานน้อยกว่า LSTM ความสามารถในการเรียนรู้จากข้อมูลที่มี noise ต่ำ
        หรือขนาดจำกัดได้รับการยืนยันผ่านงานวิจัยจาก CMU BioSeq, Harvard AI in Medicine และ Google Brain โดยเฉพาะในระบบ embedded และ edge computing
        ที่ต้องการความหน่วงต่ำและประหยัดพลังงาน GRU เป็นตัวเลือกที่เหมาะสมอย่างยิ่ง
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">11.3 ข้อจำกัดและประเด็นที่ต้องพิจารณา</h2>
      <p className="mt-4">
        แม้ GRU จะมีประโยชน์มากในบริบท edge และระบบที่มี resource จำกัด แต่ก็มีข้อจำกัดด้านความสามารถในการจดจำลำดับระยะยาว
        รวมถึงความยืดหยุ่นที่ต่ำกว่า LSTM และการตีความ (interpretability) ที่ยังคงเป็นความท้าทาย งานวิจัยจาก Oxford และ Nature Machine Intelligence
        ได้เน้นว่า GRU อาจไม่เหมาะสำหรับระบบ reasoning ที่ซับซ้อน เช่นการแปลภาษาหรือการตอบคำถามเชิงบริบท (QA)
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">11.4 แนวทางการใช้งานจริง</h2>
      <p className="mt-4">
        การเลือกใช้ GRU ควรพิจารณาจากลักษณะข้อมูล ความต้องการด้าน latency และข้อจำกัดของระบบ
        งานของ MIT AI Lab และ Harvard SEAS แนะนำให้ใช้ GRU ในงานที่เน้นการพยากรณ์แบบสั้น เช่น time series forecasting, speech recognition บน mobile
        และ health monitoring บน wearable devices ในขณะที่งาน NLP ระดับลึกอาจเหมาะกับ LSTM หรือ Transformer มากกว่า
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">11.5 สรุปเชิงเปรียบเทียบ GRU กับ LSTM</h2>
      <p className="mt-4">
        ตารางเปรียบเทียบที่ได้รับการเผยแพร่โดย Stanford และ Google Brain สรุปได้ว่า GRU มีขนาดเล็กกว่า ใช้ parameter น้อยกว่า และเหมาะกับระบบขนาดเล็ก
        ในขณะที่ LSTM มีความสามารถสูงกว่าในการเรียนรู้ลำดับยาวและบริบทซับซ้อน โดยเฉพาะในงาน NLP ที่มีระดับภาษาและโครงสร้างซ้อนชั้นหลายระดับ
        การเลือกโมเดลควรยึดตามลักษณะงานและข้อจำกัดของ deployment environment
      </p>
    </div>

    <div>
      <h2 className="text-xl font-bold mb-4">11.6 ข้อเสนอแนะเชิงกลยุทธ์สำหรับนักพัฒนา</h2>
      <p className="mt-4">
        จากการประมวลผลงานวิจัยจำนวนมากในรอบทศวรรษล่าสุด สถาบันอย่าง CMU, Harvard และ Oxford แนะนำแนวทางการประยุกต์ใช้ GRU ดังนี้:
      </p>
      <ul className="list-disc ml-6 mt-2 space-y-2">
        <li>เลือก GRU สำหรับงาน real-time บนอุปกรณ์ edge หรือ IoT</li>
        <li>ใช้ LSTM หากจำเป็นต้องวิเคราะห์ลำดับข้อมูลระยะยาว</li>
        <li>พิจารณา hybrid architectures เช่น GRU + Attention ในงานที่ซับซ้อนแต่ต้องการประหยัดพลังงาน</li>
        <li>ใช้ GRU เป็น baseline model ในงานทดลองเชิงอุตสาหกรรมก่อนเพิ่มความซับซ้อน</li>
      </ul>
    </div>

  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day28 theme={theme} />
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
        <ScrollSpy_Ai_Day28 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day28_GRU;
