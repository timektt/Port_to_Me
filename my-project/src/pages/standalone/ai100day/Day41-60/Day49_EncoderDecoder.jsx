import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day49 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day49";
import MiniQuiz_Day49 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day49";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day49_EncoderDecoder = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day49_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day49_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day49_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day49_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day49_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day49_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day49_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day49_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day49_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day49_10").format("auto").quality("auto").resize(scale().width(501));

 return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 49: Encoder-Decoder Architecture</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

        <section id="basic-concept" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. แนวคิดพื้นฐานของ Encoder-Decoder</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">Overview</h3>
    <p>
      โครงสร้าง Encoder-Decoder ถือเป็นรากฐานสำคัญของสถาปัตยกรรม Deep Learning ในการประมวลผลข้อมูลแบบลำดับ เช่น ข้อความ เสียง และวิดีโอ โดยเฉพาะอย่างยิ่งในงานแปลภาษา (Machine Translation) และการสังเคราะห์ข้อความ (Text Generation) โครงสร้างนี้ถูกเสนอขึ้นครั้งแรกในบริบทของ Neural Machine Translation โดยงานของ Sutskever et al. จาก Google Brain (2014)
    </p>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> Encoder-Decoder model กลายเป็นรากฐานของโมเดลอย่าง Seq2Seq และ Transformer ที่ใช้ในระบบ Google Translate และ ChatGPT
    </div>

    <h3 className="text-xl font-semibold">โครงสร้างหลักของ Encoder-Decoder</h3>
    <ul className="list-disc list-inside ml-4">
      <li><strong>Encoder:</strong> แปลง input sequence ให้อยู่ในรูปแบบ vector ที่สื่อความหมาย (context vector)</li>
      <li><strong>Decoder:</strong> ใช้ context vector เพื่อสร้างลำดับ output ทีละตำแหน่ง</li>
    </ul>

    <h3 className="text-xl font-semibold">ความท้าทายที่สำคัญ</h3>
    <p>
      แม้ Encoder-Decoder แบบพื้นฐานจะมีประสิทธิภาพในงานทั่วไป แต่ยังมีข้อจำกัดสำคัญ เช่น ข้อมูลที่ยาวมากอาจทำให้ context vector เก็บข้อมูลได้ไม่ครบถ้วน ส่งผลให้ output ไม่สมบูรณ์ จึงนำไปสู่แนวคิดของ Attention Mechanism
    </p>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> การใช้ context vector เพียงเวกเตอร์เดียวอาจไม่เพียงพอในลำดับที่ยาว จึงเป็นที่มาของ Attention
    </div>

    <h3 className="text-xl font-semibold">การประยุกต์ใช้ในงานจริง</h3>
    <table className="table-auto w-full text-sm text-left border">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="px-4 py-2 border">Use Case</th>
          <th className="px-4 py-2 border">Input</th>
          <th className="px-4 py-2 border">Output</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Machine Translation</td>
          <td className="border px-4 py-2">ประโยคภาษาอังกฤษ</td>
          <td className="border px-4 py-2">ประโยคภาษาไทย</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Image Captioning</td>
          <td className="border px-4 py-2">ภาพ</td>
          <td className="border px-4 py-2">ข้อความคำอธิบาย</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Speech Recognition</td>
          <td className="border px-4 py-2">เสียงพูด</td>
          <td className="border px-4 py-2">ข้อความ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">วิวัฒนาการจาก Encoder-Decoder สู่ Transformer</h3>
    <p>
      สถาปัตยกรรม Transformer ที่ถูกเสนอโดย Vaswani et al. (2017) จาก Google ได้ปฏิวัติแนวคิด Encoder-Decoder โดยตัด RNN และแทนที่ด้วย Attention ล้วน ๆ ซึ่งช่วยลดปัญหา vanishing gradient และเร่งความเร็วในการประมวลผล sequence ยาว
    </p>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> Transformer ได้กลายเป็นมาตรฐานของ NLP และนำไปสู่การพัฒนาโมเดล GPT, BERT, T5, และอื่น ๆ
    </div>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบและข้อจำกัด</h3>
    <ul className="list-disc list-inside ml-4">
      <li>สามารถจัดการลำดับที่มีความยาวไม่คงที่</li>
      <li>มีความยืดหยุ่นสูงในการประยุกต์กับหลายโดเมน</li>
      <li>จำเป็นต้องมีข้อมูลฝึกจำนวนมากเพื่อให้มีประสิทธิภาพสูง</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. NeurIPS.</li>
      <li>Vaswani, A., et al. (2017). Attention is All You Need. arXiv:1706.03762.</li>
      <li>Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder. arXiv:1406.1078.</li>
      <li>Stanford CS224n Lecture Notes, 2023.</li>
    </ul>
  </div>
</section>


     <section id="encoder-components" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ส่วนประกอบของ Encoder</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">2.1 Input Embedding Layer</h3>
    <p>
      ในโมเดล Encoder ของระบบประมวลผลภาษาธรรมชาติ เช่น Transformer หรือ BERT นั้น ชั้นแรกสุดคือการแปลง token หรือคำให้เป็นเวกเตอร์ที่สามารถประมวลผลทางคณิตศาสตร์ได้ โดยทั่วไปจะใช้ embedding matrix ที่เรียนรู้ได้เพื่อแปลง token index ให้กลายเป็นเวกเตอร์ความยาวคงที่ เช่น 512 หรือ 768 มิติ
    </p>
    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> Embedding layer เป็นพื้นฐานสำคัญที่ช่วยให้โมเดลเข้าใจ semantics ของคำในเชิงคณิตศาสตร์
    </div>

    <h3 className="text-xl font-semibold">2.2 Positional Encoding</h3>
    <p>
      เนื่องจาก Transformer ไม่มีความสามารถในการเข้าใจลำดับคำโดยธรรมชาติ จึงต้องมีการเติม positional information เข้าไป โดยใช้ฟังก์ชันไซน์และโคไซน์ที่มีความถี่ต่างกันในแต่ละตำแหน่ง เพื่อให้โมเดลสามารถแยกแยะคำว่าอยู่ตำแหน่งใดในลำดับ
    </p>
    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`PE(pos, 2i)   = sin(pos / 50000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 50000^(2i/d_model))`}
    </pre>
    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> Positional encoding ทำให้โมเดลเรียนรู้ลำดับของข้อมูล โดยไม่ต้องใช้ recurrent structure
    </div>

    <h3 className="text-xl font-semibold">2.3 Multi-Head Self-Attention</h3>
    <p>
      กลไกหลักของ Encoder คือ Self-Attention ซึ่งช่วยให้โมเดลสามารถโฟกัสข้อมูลในลำดับที่สัมพันธ์กันได้ในแต่ละตำแหน่ง โดยการคำนวณ dot-product ระหว่าง Query, Key และ Value ซึ่งได้จากการแปลง input เดิมผ่าน weight matrix
    </p>
    <p>
      ใน Multi-Head Self-Attention จะมีหลายชุดของ Q, K, V ทำให้สามารถจับ pattern ได้หลากหลาย
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          <th className="px-4 py-2">หัวข้อ</th>
          <th className="px-4 py-2">รายละเอียด</th>
        </tr>
      </thead>
      <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Query</td>
          <td className="px-4 py-2">ใช้แทนข้อมูลปัจจุบันที่กำลังมองหา context</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Key</td>
          <td className="px-4 py-2">ใช้เปรียบเทียบกับ Query เพื่อหาความสัมพันธ์</td>
        </tr>
        <tr className="border-t border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Value</td>
          <td className="px-4 py-2">ให้ข้อมูลที่ถูกดึงมาโดยอิงจาก score ของ Q และ K</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">2.4 Feed Forward Neural Network</h3>
    <p>
      หลังจาก Attention Layer ข้อมูลจะถูกส่งผ่าน Dense Layer แบบ Position-wise FFN ซึ่งเป็นโครงข่ายแบบสองชั้นที่มี activation function เช่น ReLU หรือ GELU ใช้สำหรับเพิ่มความไม่เชิงเส้นให้กับระบบ
    </p>

    <h3 className="text-xl font-semibold">2.5 Residual Connection & Layer Normalization</h3>
    <p>
      Encoder แต่ละชั้นจะมีการใช้ residual connection เพื่อช่วยลดปัญหา gradient vanishing และการฝึกที่ลึกขึ้น ในขณะเดียวกันก็มีการใช้ layer normalization หลังจาก residual เพื่อรักษา distribution ของค่าพารามิเตอร์ให้เสถียร
    </p>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> ส่วนประกอบเหล่านี้ทำงานร่วมกันเป็นองค์รวมเพื่อสร้างการเข้ารหัสที่ทรงพลังจากข้อมูลลำดับ
    </div>

    <h3 className="text-xl font-semibold mt-6">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani et al. "Attention Is All You Need." NeurIPS 2017.</li>
      <li>Stanford CS224N Lecture Notes: Natural Language Processing with Deep Learning.</li>
      <li>MIT 6.S191 Deep Learning Course Materials.</li>
      <li>arXiv:1706.03762</li>
    </ul>
  </div>
</section>


 <section id="decoder-components" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. ส่วนประกอบของ Decoder</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">โครงสร้างทั่วไปของ Decoder</h3>
    <p>
      ในสถาปัตยกรรม Encoder-Decoder ที่ใช้กันอย่างแพร่หลายในงานประมวลผลภาษาธรรมชาติ (NLP) และงานด้าน Computer Vision นั้น Decoder มีหน้าที่หลักในการแปลง representation ที่เข้ารหัสจาก Encoder ให้กลับมาเป็นข้อมูลในรูปแบบที่ตีความได้ เช่น ประโยคในภาษาธรรมชาติ หรือภาพที่มีความละเอียดสูง โดยเฉพาะในโมเดลอย่าง Transformer และ Seq2Seq Decoder ประกอบด้วยชั้นต่าง ๆ ที่มีการเรียนรู้แบบลำดับขั้นเพื่อประมวลผลข้อมูลย้อนกลับจาก representation ที่ได้จาก Encoder
    </p>
    <h3 className="text-xl font-semibold">1. Input Embedding และ Positional Encoding</h3>
    <p>
      Decoder เริ่มต้นด้วยการรับข้อมูลเป้าหมาย (target sequence) ซึ่งโดยทั่วไปเป็นข้อความที่เลื่อนมาหนึ่งตำแหน่ง (shifted right) เพื่อหลีกเลี่ยงการเห็นข้อมูลอนาคต จากนั้นทำการแปลง token เหล่านี้เป็น embedding และเพิ่ม Positional Encoding เพื่อรักษาลำดับเชิงเวลา ซึ่งเป็นส่วนสำคัญในโมเดลที่ไม่มีโครงสร้างลำดับโดยธรรมชาติ เช่น Transformer
    </p>
    <h3 className="text-xl font-semibold">2. Masked Multi-Head Self-Attention</h3>
    <p>
      กลไกนี้ช่วยให้ Decoder สามารถเรียนรู้บริบทก่อนหน้าได้โดยไม่สามารถเข้าถึง token ที่ยังไม่ถูกสร้าง ซึ่งแตกต่างจาก Encoder ที่ใช้ attention แบบเต็ม โดยใช้ masking เพื่อซ่อนตำแหน่งในอนาคตระหว่างการฝึก
    </p>
    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <p>
        Insight: Masked Self-Attention ช่วยป้องกันการรั่วไหลของข้อมูล (information leakage) จากอนาคต ซึ่งเป็นกุญแจสำคัญในการสร้างข้อความที่ถูกต้องตามลำดับเวลา
      </p>
    </div>
    <h3 className="text-xl font-semibold">3. Encoder-Decoder Attention</h3>
    <p>
      ส่วนนี้จะคำนวณ attention ระหว่าง output ของ Encoder และ output ของ layer ก่อนหน้าใน Decoder เพื่อให้ Decoder เข้าถึงข้อมูลจาก input sequence อย่างเหมาะสม ใช้แนวคิด Query จาก Decoder และ Key/Value จาก Encoder
    </p>
    <h3 className="text-xl font-semibold">4. Feedforward Neural Network</h3>
    <p>
      คล้ายกับ Encoder Decoder ก็ใช้ fully-connected layer สองชั้นพร้อม activation function เช่น ReLU เพื่อแปลงข้อมูลเชิงลึกในลักษณะไม่เชิงเส้น (non-linear transformation)
    </p>
    <h3 className="text-xl font-semibold">5. Normalization และ Residual Connections</h3>
    <p>
      Layer normalization ถูกใช้เพื่อรักษาเสถียรภาพของการเรียนรู้ พร้อมกับ residual connection ที่ช่วยให้ gradient ไม่หายไประหว่างการ backpropagation ซึ่งเป็นแนวคิดที่นำมาจาก ResNet
    </p>
    <h3 className="text-xl font-semibold">6. Output Linear Layer และ Softmax</h3>
    <p>
      หลังจากผ่านทุก layer แล้ว output จะถูกส่งผ่าน linear layer ที่แมปไปยังขนาดของ vocabulary ก่อนจะใช้ softmax เพื่อคำนวณความน่าจะเป็นของ token ถัดไป
    </p>
    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <p>
        Highlight: Decoder สามารถฝึกได้แบบ teacher forcing โดยใช้ target จริงเป็น input เพื่อให้การเรียนรู้มีเสถียรภาพมากขึ้นในช่วงต้นของการฝึก
      </p>
    </div>
    <h3 className="text-xl font-semibold">การเปรียบเทียบระหว่าง Encoder และ Decoder</h3>
    <div className="overflow-x-auto w-full">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <th className="px-4 py-2">องค์ประกอบ</th>
            <th className="px-4 py-2">Encoder</th>
            <th className="px-4 py-2">Decoder</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Self-Attention</td>
            <td className="px-4 py-2">Multi-Head</td>
            <td className="px-4 py-2">Masked Multi-Head</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Cross Attention</td>
            <td className="px-4 py-2">ไม่มี</td>
            <td className="px-4 py-2">มี (กับ Encoder Output)</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Output</td>
            <td className="px-4 py-2">Representation</td>
            <td className="px-4 py-2">Sequence Prediction</td>
          </tr>
        </tbody>
      </table>
    </div>
    <h3 className="text-xl font-semibold mt-6">อ้างอิงทางวิชาการ</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani et al., "Attention Is All You Need," NeurIPS 2017.</li>
      <li>Stanford CS224N: Natural Language Processing with Deep Learning.</li>
      <li>MIT 6.S191: Deep Learning for Self-Driving Cars.</li>
      <li>arXiv:1706.03762v5 [cs.CL]</li>
    </ul>
  </div>
</section>


     <section id="architecture-problems" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. ปัญหาใน Architecture แบบเดิม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">ข้อจำกัดของโมเดลลำดับ</h3>
    <p>โมเดลอย่าง RNN มีข้อจำกัดในด้านการจดจำบริบทที่อยู่ห่างไกลกันในลำดับข้อมูลยาว ทำให้เกิดปัญหา vanishing gradient เมื่อเพิ่มจำนวน layer</p>
    <ul className="list-disc list-inside ml-4">
      <li>จำบริบทไกลได้น้อย</li>
      <li>ฝึกช้า และไม่สามารถขนานได้</li>
      <li>เกิดปัญหา gradient หาย (vanishing gradients)</li>
    </ul>
    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> งานจาก Stanford (Karpathy, 2015) แสดงว่า RNN แม้จะทรงพลัง แต่ไม่เหมาะกับ long-term context โดยไม่ใช้เทคนิคเสริม
    </div>

    <h3 className="text-xl font-semibold">CNN กับงานที่ไม่ใช่ภาพ</h3>
    <p>CNN ได้ผลดีมากในงานด้านภาพ แต่ข้อจำกัดอยู่ที่ kernel ไม่สามารถรับบริบทเชิงลำดับได้ ทำให้มีปัญหาใน NLP</p>
    <ul className="list-disc list-inside ml-4">
      <li>ต้องใช้หลายชั้นเพื่อเพิ่ม receptive field</li>
      <li>ไม่สามารถเข้าใจโครงสร้างไวยากรณ์ในข้อความ</li>
      <li>ความเข้าใจบริบทจำกัดตาม kernel size</li>
    </ul>
    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> จาก MIT-IBM Watson Lab, CNN เหมาะกับการจัดประเภทข้อความ แต่ไม่เหมาะกับการเข้าใจบริบทเชิงโครงสร้าง
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบความสามารถของโมเดลต่าง ๆ</h3>
    <div className="overflow-x-auto w-full">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <th className="px-4 py-2">โมเดล</th>
            <th className="px-4 py-2">รองรับข้อมูลลำดับ</th>
            <th className="px-4 py-2">บริบทระยะไกล</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">MLP</td>
            <td className="px-4 py-2">ไม่</td>
            <td className="px-4 py-2">ไม่ได้</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">CNN</td>
            <td className="px-4 py-2">บางกรณี</td>
            <td className="px-4 py-2">จำกัด</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">RNN</td>
            <td className="px-4 py-2">ได้ดี</td>
            <td className="px-4 py-2">ปานกลาง</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Transformer</td>
            <td className="px-4 py-2">สูง</td>
            <td className="px-4 py-2">ดีเยี่ยม</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ปัญหาที่ตามมา</h3>
    <ul className="list-disc list-inside ml-4">
      <li>เวลาในการเทรนสูงขึ้นตามลำดับความลึกของเครือข่าย</li>
      <li>ต้องใช้ข้อมูลมากขึ้นในการฝึกโมเดลที่ใหญ่ขึ้น</li>
      <li>เกิด overfitting ได้ง่ายหากไม่ได้ regularization ที่ดี</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> บทเรียนจาก Oxford ระบุว่า โมเดลต้องถูกออกแบบให้สมดุลระหว่างความลึก ความเร็ว และความสามารถในการตีความข้อมูลระยะไกล
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Karpathy, A. (2015). The Unreasonable Effectiveness of RNNs. Stanford CS231n.</li>
      <li>MIT-IBM Watson AI Lab (2018). Sequence Modeling Challenges.</li>
      <li>Vaswani et al. (2017). Attention is All You Need. arXiv:1706.03762</li>
      <li>Oxford Deep Learning Lectures (2021)</li>
    </ul>
  </div>
</section>


      <section id="attention-extension" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. การพัฒนา: Encoder-Decoder with Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">แนวคิดพื้นฐานของ Encoder-Decoder</h3>
    <p>
      สถาปัตยกรรม Encoder-Decoder ถูกออกแบบมาเพื่อจัดการกับปัญหาการประมวลผลข้อมูลลำดับที่มี input และ output คนละความยาว เช่น การแปลภาษา หรือการสร้างคำอธิบายภาพ
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Encoder ทำหน้าที่แปลงลำดับข้อมูลอินพุตเป็นเวกเตอร์บริบท</li>
      <li>Decoder สร้างลำดับเอาต์พุตโดยอิงจากบริบทที่เข้ารหัสไว้</li>
    </ul>
    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> แม้ Encoder-Decoder จะใช้งานได้ดีในหลายกรณี แต่เมื่อลำดับข้อมูลยาวมาก บริบททั้งหมดที่บีบอัดไว้ในเวกเตอร์เดียวอาจไม่เพียงพอ ซึ่งนำไปสู่การพัฒนา Attention
    </div>

    <h3 className="text-xl font-semibold">การเพิ่ม Attention: แก้ปัญหาบริบทจำกัด</h3>
    <p>
      กลไก Attention ถูกเสนอเพื่อให้ Decoder ไม่ต้องพึ่งพาเวกเตอร์บริบทเพียงตัวเดียว แต่สามารถ "โฟกัส" ไปยังส่วนต่าง ๆ ของอินพุตได้อย่างยืดหยุ่นตามความจำเป็นในแต่ละ timestep
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ใช้ค่า similarity (เช่น dot-product) เพื่อวัดความสัมพันธ์ระหว่าง hidden state กับ input</li>
      <li>ถ่วงน้ำหนักข้อมูลอินพุตให้เหมาะกับการสร้าง output แต่ละคำ</li>
      <li>ลดภาระของ bottleneck ที่เกิดจาก encoding เดียว</li>
    </ul>
    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานของ Bahdanau et al. (2015) คือจุดเปลี่ยนสำคัญที่นำ Attention เข้ามาใช้ใน Neural Machine Translation และสร้างผลลัพธ์ที่ดีกว่า RNN แบบดั้งเดิม
    </div>

    <h3 className="text-xl font-semibold">ภาพรวมโครงสร้าง Encoder-Decoder with Attention</h3>
    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
      <code>
{`Input Sequence  -->  Encoder (Bi-RNN/CNN/Transformer)
                          |
                          V
                 Context Vectors with Attention
                          |
                          V
                 Decoder generates output -> Y1, Y2, ..., Yn`}
      </code>
    </pre>

    <h3 className="text-xl font-semibold">เปรียบเทียบก่อนและหลังมี Attention</h3>
    <div className="overflow-x-auto w-full">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <th className="px-4 py-2">คุณสมบัติ</th>
            <th className="px-4 py-2">Encoder-Decoder เดิม</th>
            <th className="px-4 py-2">Encoder-Decoder + Attention</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">ความแม่นยำการแปล</td>
            <td className="px-4 py-2">ปานกลาง</td>
            <td className="px-4 py-2">สูงขึ้นมาก</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">ความยาวข้อความที่รองรับ</td>
            <td className="px-4 py-2">จำกัด</td>
            <td className="px-4 py-2">รองรับได้ดีกว่า</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">การตีความบริบท</td>
            <td className="px-4 py-2">อิงจาก vector เดียว</td>
            <td className="px-4 py-2">ใช้ attention ในการโฟกัสแบบ adaptive</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ผลลัพธ์และการใช้งานในโลกจริง</h3>
    <p>
      โมเดล Encoder-Decoder with Attention ถูกนำไปใช้ในหลายระบบจริง เช่น Google Translate, Summarization System, Chatbot และระบบ OCR
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Google Neural Machine Translation (GNMT)</li>
      <li>Automatic Image Captioning</li>
      <li>Question Answering with BERT-based Encoders</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> Encoder-Decoder with Attention ถือเป็นรากฐานสำคัญของ Transformer และโมเดลสมัยใหม่อย่าง BERT, GPT, และ T5 ที่ใช้งานในระบบ AI ขนาดใหญ่ทั่วโลก
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473</li>
      <li>Luong, M. et al. (2015). Effective Approaches to Attention-based Neural Machine Translation. arXiv:1508.04025</li>
      <li>Vaswani, A. et al. (2017). Attention is All You Need. arXiv:1706.03762</li>
      <li>Harvard NLP Group. (2020). The Annotated Transformer.</li>
      <li>Stanford CS224N (2022). Lecture Notes on Attention and Sequence Models.</li>
    </ul>
  </div>
</section>


   <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ตัวอย่างการใช้งาน Encoder-Decoder</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">การแปลภาษาด้วย Neural Machine Translation (NMT)</h3>
    <p>
      หนึ่งในตัวอย่างที่ประสบความสำเร็จที่สุดของ Encoder-Decoder คือการประยุกต์ใช้ในงานแปลภาษาด้วย Neural Machine Translation (NMT) ซึ่งระบบอย่าง Google Translate ในปัจจุบันได้เปลี่ยนจากระบบ phrase-based มาเป็น encoder-decoder with attention อย่างเต็มรูปแบบ
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Encoder เข้ารหัสข้อความต้นทาง (source sentence) เป็น sequence representation</li>
      <li>Attention ช่วยให้ decoder โฟกัสคำที่เกี่ยวข้องในตำแหน่งที่แตกต่างกันของต้นฉบับ</li>
      <li>Decoder สร้างประโยคปลายทางทีละคำ โดยอิงจาก context vector แบบ dynamic</li>
    </ul>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานวิจัยของ Google (Wu et al., 2016) ระบุว่า NMT แบบ Encoder-Decoder ให้ผลลัพธ์แม่นยำกว่า phrase-based models มาก โดยเฉพาะในภาษาที่มีโครงสร้างซับซ้อน
    </div>

    <h3 className="text-xl font-semibold">การสรุปข้อความ (Text Summarization)</h3>
    <p>
      Encoder-Decoder ถูกนำไปใช้งานในการสรุปข้อความ (Summarization) ทั้งในรูปแบบ extractive และ abstractive โดยเฉพาะ abstractive summarization ที่โมเดลต้องสร้างประโยคใหม่ที่ไม่จำเป็นต้องมีอยู่ในต้นฉบับ
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Encoder วิเคราะห์โครงสร้างต้นฉบับ</li>
      <li>Decoder สร้างข้อความสรุปที่มีความกระชับแต่สื่อความหมายเดิม</li>
      <li>นิยมใช้ร่วมกับ attention และ coverage mechanism เพื่อหลีกเลี่ยงการซ้ำคำ</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> งานจาก Harvard NLP (See et al., 2017) เสนอว่า coverage vector ช่วยลดปัญหา repetitive generation และทำให้ระบบ summarization มีความเป็นธรรมชาติยิ่งขึ้น
    </div>

    <h3 className="text-xl font-semibold">การสร้างคำอธิบายภาพ (Image Captioning)</h3>
    <p>
      การจับคู่ระหว่างข้อมูลภาพกับข้อความคือโจทย์ที่ Encoder-Decoder สามารถจัดการได้ดี โดยใช้ CNN เป็น encoder และ RNN/Transformer เป็น decoder เพื่อแปลงภาพเป็นคำอธิบาย
    </p>
    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
<code>{`[Image] --(CNN Encoder)--> Feature Vector --(Attention)--> Text Decoder --> "A dog running on the grass"`}</code>
    </pre>
    <ul className="list-disc list-inside ml-4">
      <li>ใช้ CNN เช่น InceptionV3 หรือ ResNet ในการดึง features จากภาพ</li>
      <li>ใช้ attention mechanism เพื่อโฟกัสไปยังส่วนที่สำคัญของภาพ</li>
      <li>Decoder ทำหน้าที่สร้างลำดับคำโดยอิงจาก feature ที่เข้ารหัสไว้</li>
    </ul>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานของ Xu et al. (2015) ได้รับการอ้างอิงอย่างกว้างขวางจากการใช้ visual attention ในการสร้างคำอธิบายภาพแบบอัตโนมัติ
    </div>

    <h3 className="text-xl font-semibold">Question Answering (QA)</h3>
    <p>
      Encoder-Decoder ถูกประยุกต์ในงาน QA แบบ generative ที่คำตอบไม่ได้อยู่ตรง ๆ ในบริบท แต่ต้องสังเคราะห์ขึ้นจากความเข้าใจ เช่นในการสนทนา (conversational QA) หรือการตอบแบบสรุปจากเอกสารหลายชุด
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Encoder รับคำถามและบริบทเข้ามาในรูปแบบ joint representation</li>
      <li>Decoder สร้างคำตอบใหม่โดยอิงจากความเข้าใจรวมของโมเดล</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> T5 จาก Google ใช้แนวทาง text-to-text สำหรับทุก task รวมถึง QA โดยถือว่า QA เป็นการ "แปล" จากคำถาม + บริบท ไปยังคำตอบ
    </div>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบการใช้งาน</h3>
    <div className="overflow-x-auto w-full">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <th className="px-4 py-2">Use Case</th>
            <th className="px-4 py-2">Encoder</th>
            <th className="px-4 py-2">Decoder</th>
            <th className="px-4 py-2">Model</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Translation</td>
            <td className="px-4 py-2">Bi-RNN / Transformer</td>
            <td className="px-4 py-2">RNN / Transformer</td>
            <td className="px-4 py-2">GNMT, Transformer</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Summarization</td>
            <td className="px-4 py-2">Transformer</td>
            <td className="px-4 py-2">Transformer</td>
            <td className="px-4 py-2">PEGASUS, BART</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Image Captioning</td>
            <td className="px-4 py-2">CNN (e.g., ResNet)</td>
            <td className="px-4 py-2">LSTM / Transformer</td>
            <td className="px-4 py-2">Show & Tell, Visual Transformer</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Question Answering</td>
            <td className="px-4 py-2">Transformer (Joint Input)</td>
            <td className="px-4 py-2">Transformer</td>
            <td className="px-4 py-2">T5, UnifiedQA</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Wu, Y. et al. (2016). Google's Neural Machine Translation System. arXiv:1609.08144</li>
      <li>See, A. et al. (2017). Get To The Point: Summarization with Coverage. arXiv:1704.04368</li>
      <li>Xu, K. et al. (2015). Show, Attend and Tell. ICML</li>
      <li>Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with T5. JMLR</li>
      <li>Harvard NLP Group, Stanford CS224N Lecture Notes</li>
    </ul>

  </div>
</section>

  <section id="comparison-transformer" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Encoder-Decoder vs Transformer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">โครงสร้างพื้นฐานของ Encoder-Decoder แบบดั้งเดิม</h3>
    <p>
      สถาปัตยกรรม Encoder-Decoder แบบดั้งเดิมใช้ RNN หรือ LSTM ในการประมวลผลข้อมูลลำดับ โดย encoder จะอ่านอินพุตทีละ timestep และสร้าง context vector เพื่อส่งให้ decoder สร้างผลลัพธ์ในลำดับต่อเนื่อง
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Encoder ทำงานแบบ sequential บน input sequence</li>
      <li>Decoder ใช้ hidden state ก่อนหน้าในการสร้างคำถัดไป</li>
      <li>อิงกับโครงสร้าง recurrent ทำให้ parallelization ยาก</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> แม้ LSTM และ GRU จะพัฒนาให้ RNN มีความสามารถเรียนรู้บริบทได้ลึกขึ้น แต่ยังคงมีข้อจำกัดด้านการประมวลผลแบบขนานและบริบทระยะไกลที่ไม่สมบูรณ์
    </div>

    <h3 className="text-xl font-semibold">แนวคิดพื้นฐานของ Transformer</h3>
    <p>
      Transformer ถูกเสนอโดย Vaswani et al. (2017) เป็นสถาปัตยกรรมที่ไม่ใช้ recurrence แต่ใช้ attention mechanism แบบ self-attention ทั้งหมด ช่วยให้สามารถประมวลผลลำดับข้อมูลได้แบบขนานทั้งหมด
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ใช้ Multi-Head Attention ในการจับบริบททุกตำแหน่ง</li>
      <li>มี positional encoding เพื่อรักษาลำดับของคำ</li>
      <li>สามารถเรียนรู้ long-range dependencies ได้ดีกว่า</li>
    </ul>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> การเลิกใช้ RNN ทำให้ Transformer สามารถ train เร็วขึ้นหลายเท่า และสามารถขยายขนาดได้สู่ระดับโมเดลที่มีพารามิเตอร์นับแสนล้าน เช่น GPT-3 หรือ PaLM
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบความแตกต่างระหว่างสองสถาปัตยกรรม</h3>
    <div className="overflow-x-auto w-full">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <th className="px-4 py-2">คุณสมบัติ</th>
            <th className="px-4 py-2">Encoder-Decoder (RNN-based)</th>
            <th className="px-4 py-2">Transformer</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">การเรียนรู้ลำดับ</td>
            <td className="px-4 py-2">ผ่าน hidden state ทีละ timestep</td>
            <td className="px-4 py-2">ผ่าน self-attention ทั้งลำดับ</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">ความสามารถในการประมวลผลขนาน</td>
            <td className="px-4 py-2">ต่ำ (sequential)</td>
            <td className="px-4 py-2">สูง (parallelizable)</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">บริบทระยะไกล</td>
            <td className="px-4 py-2">จำกัด</td>
            <td className="px-4 py-2">ดีเยี่ยม</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">ความซับซ้อนของโครงสร้าง</td>
            <td className="px-4 py-2">น้อยกว่า</td>
            <td className="px-4 py-2">สูงกว่า แต่ scale ได้ดี</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">ตัวอย่างโมเดล</td>
            <td className="px-4 py-2">GNMT, Seq2Seq, Pointer Networks</td>
            <td className="px-4 py-2">BERT, GPT, T5, PaLM, LLaMA</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มการใช้งานในอุตสาหกรรม</h3>
    <p>
      ในปัจจุบัน Transformer ได้กลายเป็นสถาปัตยกรรมมาตรฐานในงาน NLP, Computer Vision, และ Multimodal AI แทบทุกระบบการเรียนรู้ลึกขนาดใหญ่ที่ใช้ในอุตสาหกรรม เช่น ระบบสรุปเอกสารของ Google หรือระบบแปลภาษาแบบเรียลไทม์ของ DeepL ล้วนใช้ Transformer
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>การฝึกแบบ distributed training บน GPU/TPU ช่วยเร่งความเร็ว</li>
      <li>ความสามารถในการ generalize ข้าม domain ได้ดี</li>
      <li>เหมาะกับ pretraining แล้ว fine-tune ตาม task</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> ข้อมูลจาก OpenAI และ Google Brain ชี้ว่า Transformer ไม่เพียงแต่แทนที่ Encoder-Decoder แบบเดิมใน NLP เท่านั้น แต่ยังกำลังเข้ามาแทน CNN ในงาน Vision ด้วย เช่นในโมเดล Vision Transformer (ViT)
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention is All You Need. arXiv:1706.03762</li>
      <li>Bahdanau, D. et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR</li>
      <li>Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL</li>
      <li>Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Vision Transformer. arXiv:2010.11929</li>
      <li>Stanford CS224N (2023). Lecture: Transformer Models</li>
    </ul>

  </div>
</section>


      <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Research Benchmarks & Citations</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">แนวคิดของ Benchmark ในการประเมินสถาปัตยกรรม</h3>
    <p>
      ในงานวิจัยด้าน Deep Learning การวัดผลสถาปัตยกรรมไม่ได้จำกัดเพียงค่า accuracy หรือ loss เท่านั้น แต่ต้องประเมินจาก Benchmark ที่ได้รับการยอมรับในระดับอุตสาหกรรมและงานวิจัย เพื่อสะท้อนถึงการ generalize, scalability และความเหมาะสมของโมเดลในสภาพแวดล้อมที่ต่างกัน
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ใช้ชุดข้อมูลที่ได้รับการยอมรับ เช่น ImageNet, GLUE, SQuAD, WMT</li>
      <li>ใช้เกณฑ์วัดผลเฉพาะ domain เช่น BLEU, ROUGE, F1, Exact Match</li>
      <li>การทดสอบโมเดลในหลายภาษา หลายโดเมน และหลายรูปแบบข้อมูล</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> การที่โมเดลทำคะแนนสูงใน benchmark หนึ่ง ไม่ได้แปลว่าจะทำงานได้ดีในทุก context การพิจารณาหลาย benchmark พร้อมกันจึงจำเป็นในงานวิจัยยุคใหม่
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่าง Benchmark หลักใน NLP และ Vision</h3>
    <div className="overflow-x-auto w-full">
      <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
            <th className="px-4 py-2">Benchmark</th>
            <th className="px-4 py-2">ประเภท</th>
            <th className="px-4 py-2">ใช้วัด</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">GLUE</td>
            <td className="px-4 py-2">NLP</td>
            <td className="px-4 py-2">ความเข้าใจภาษาทั่วไป (textual entailment, sentiment)</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">SQuAD</td>
            <td className="px-4 py-2">QA</td>
            <td className="px-4 py-2">ความสามารถในการหาคำตอบจากบริบท</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">WMT</td>
            <td className="px-4 py-2">Translation</td>
            <td className="px-4 py-2">ความแม่นยำของการแปลภาษา (BLEU)</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">ImageNet</td>
            <td className="px-4 py-2">Vision</td>
            <td className="px-4 py-2">ความสามารถในการจำแนกภาพ (top-1/top-5 accuracy)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">มาตรฐานการอ้างอิงงานวิจัยใน Deep Learning</h3>
    <p>
      การอ้างอิงที่ถูกต้องไม่เพียงแต่เป็นการให้เครดิตกับต้นทาง แต่ยังช่วยให้สามารถตรวจสอบผลลัพธ์ และต่อยอดงานวิจัยเดิมได้อย่างแม่นยำ มหาวิทยาลัยและสถาบันวิจัยชั้นนำ เช่น Stanford, MIT, และ Oxford มีมาตรฐานการเผยแพร่ผ่าน arXiv, NeurIPS, ICML, ACL และ Nature AI อย่างเข้มงวด
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>อ้างอิงชื่อผู้เขียน, ปี, หัวข้อ และแหล่งตีพิมพ์</li>
      <li>ใช้ DOI หรือ arXiv ID เพื่ออ้างอิงที่ตรวจสอบได้</li>
      <li>นิยมอ้างอิงผ่าน BibTeX หรือ CSL JSON ในงานตีพิมพ์</li>
    </ul>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานวิจัยระดับสูงเช่น “Attention is All You Need” ของ Vaswani et al. มีผลกระทบในระดับโลก ถูกอ้างอิงมากกว่า 80,000 ครั้ง และกลายเป็นพื้นฐานของสถาปัตยกรรม GPT, BERT, และอีกหลายโมเดล
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มการประเมินโมเดลในอนาคต</h3>
    <p>
      ในอนาคต การประเมินโมเดลจะไม่จำกัดเพียง performance แต่จะพิจารณาความสามารถด้าน interpretability, fairness, robustness และการใช้พลังงาน (energy efficiency) เช่น Carbon Emissions per Training Cycle
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>Energy Efficiency Benchmarks เช่น MLCO2 Tracker</li>
      <li>Fairness Metric เช่น Demographic Parity, Equalized Odds</li>
      <li>Robustness Evaluation เช่น Adversarial Testing</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> สถาบันวิจัย AI ชั้นนำทั่วโลกเริ่มเผยแพร่โมเดลพร้อม checklist ด้าน ethical impact เพื่อความโปร่งใส เช่น BigScience, OpenAI, และ Hugging Face
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Vaswani, A. et al. (2017). Attention is All You Need. arXiv:1706.03762</li>
      <li>Rajpurkar, P. et al. (2016). SQuAD: Stanford Question Answering Dataset. EMNLP</li>
      <li>Wang, A. et al. (2019). GLUE: A Multi-Task Benchmark and Analysis Platform. ICLR</li>
      <li>Deng, J. et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. CVPR</li>
      <li>BigScience Workshop (2022). Data Governance and Responsible Benchmarking in AI</li>
    </ul>
    
  </div>
</section>


      <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">

    <h3 className="text-xl font-semibold">แนวคิดของ Insight Box ในงานวิจัย Deep Learning</h3>
    <p>
      ในการออกแบบบทเรียนหรือบทความวิจัยที่มีเนื้อหาเชิงลึกทางด้าน Deep Learning โดยเฉพาะในระดับมหาวิทยาลัย เช่น Stanford หรือ MIT การใช้ <strong>Insight Box</strong> ช่วยให้ผู้อ่านสามารถเข้าถึงแนวคิดหลักหรือสิ่งที่ “มองไม่เห็นจากค่าชี้วัดเชิงปริมาณ” ได้อย่างเป็นระบบและลึกซึ้งมากขึ้น
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>เป็นกล่องข้อความที่เน้นข้อมูลเชิงกลยุทธ์ หรือข้อสังเกตเชิงวิจัย</li>
      <li>ช่วยย่อสาระสำคัญที่ได้จากพฤติกรรมของโมเดลหรือผลการทดลอง</li>
      <li>สะท้อนจุดที่ค่าชี้วัดแบบ traditional (accuracy, loss) ไม่สามารถอธิบายได้</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> ในหลายกรณี ความเข้าใจเชิงลึกเกี่ยวกับ failure cases หรือ generalization ไม่สามารถได้จากกราฟ loss แต่ได้จาก insight ที่เชื่อมโยงกับโครงสร้างของข้อมูลหรือ bias ของโมเดล
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งาน Insight Box</h3>
    <p>
      ต่อไปนี้คือตัวอย่างสถานการณ์ที่ Insight Box ช่วยขยายความเข้าใจในเชิงระบบ:
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ใน LSTM ที่ดูเหมือนจะ overfit ข้อมูล — อาจพบว่า sequence ที่ overfit นั้นมีลักษณะเหมือนกันซ้ำ ๆ จึงทำให้ดูแม่นยำแต่ generalize ไม่ได้</li>
      <li>ใน Attention Score ที่สูงผิดปกติในบาง timestep — อาจแสดงว่า model กำลัง rely กับ keyword เดียวมากเกินไป</li>
      <li>ใน CNN ที่มี feature map ต่ำชั้น — อาจมี activation คล้ายกันในหลายคลาส แสดงถึงการแชร์ feature ที่ยังไม่เพียงพอ</li>
    </ul>

    <div className="bg-blue-500 p-4 rounded-lg border-l-4 border-blue-400">
      <strong>Highlight:</strong> งานวิจัยของ Olah et al. (OpenAI, 2020) ได้ใช้กล่อง insight ในการ visualize feature level ของ GPT-2 และพบว่าโมเดลมีการจัดเก็บข้อมูลอย่างเป็นระบบระดับ concept
    </div>

    <h3 className="text-xl font-semibold">โครงสร้างมาตรฐานของ Insight Box</h3>
    <pre className="bg-gray-800 text-white rounded-lg overflow-x-auto text-sm p-4">
{`<div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
  <strong>Insight:</strong> โมเดล LSTM มีแนวโน้มเก็บบริบทคำล่าสุดมากกว่า context ที่อยู่ห่างไกล ซึ่งต้องใช้ bidirectional หรือ attention เข้ามาเสริม
</div>`}
    </pre>
    <p>
      ในโปรเจกต์ AI/500Day และงานบทเรียนระดับสูงทั่วไป โครงสร้างนี้ถูกใช้เป็นมาตรฐาน เพื่อให้ผู้อ่านสามารถจดจำรูปแบบของสาระสำคัญได้ชัดเจนและเป็นระบบ
    </p>

    <h3 className="text-xl font-semibold">Insight Box กับการวิเคราะห์ความล้มเหลวของโมเดล</h3>
    <p>
      การวิเคราะห์ความล้มเหลว (Failure Analysis) ถือเป็นกระบวนการสำคัญในการออกแบบโมเดลล้ำสมัย เช่น GPT, PaLM หรือ LLaMA การแสดง insight ที่มีคุณภาพสามารถช่วยนักวิจัยและวิศวกรเข้าใจว่าโมเดล “เข้าใจผิด” ตรงไหน หรือ “เชื่อมโยงแบบผิดบริบท” ได้อย่างไร
    </p>
    <ul className="list-disc list-inside ml-4">
      <li>ช่วยแยกความแตกต่างระหว่าง performance ที่ดีจริงกับ performance ที่หลอกตา</li>
      <li>ใช้ในการปรับ tuning strategy, loss function หรือ data sampling</li>
      <li>ช่วยค้นพบ inductive bias ที่ไม่ได้ตั้งใจ</li>
    </ul>

    <div className="bg-yellow-500 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Insight:</strong> การใช้ Insight Box เป็นเครื่องมือวิจัยทำให้หลายโครงการ เช่น BERTology หรือ interpretability research จาก Anthropic สามารถเปิดเผยพฤติกรรมภายในโมเดลขนาดใหญ่ได้มากกว่าการพล็อตกราฟทั่วไป
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside ml-4">
      <li>Olah, C. et al. (2020). A Visual Guide to GPT-2. Distill.pub</li>
      <li>Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? ACL</li>
      <li>OpenAI (2022). GPT-3 Model Interpretability Experiments</li>
      <li>Stanford CS25 Lecture Series (2022). Interpretability in Neural Networks</li>
      <li>Harvard NLP Group. (2023). Best Practices in Deep Model Debugging</li>
    </ul>

  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day49 theme={theme} />
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
        <ScrollSpy_Ai_Day49 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day49_EncoderDecoder;
