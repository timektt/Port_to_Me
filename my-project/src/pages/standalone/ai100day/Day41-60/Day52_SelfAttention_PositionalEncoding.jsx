import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day52 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day52";
import MiniQuiz_Day52 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day52";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day52_SelfAttention_PositionalEncoding = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day52_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day52_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day52_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day52_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day52_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day52_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day52_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day52_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day52_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day52_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day52_11").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 52: Self-Attention & Positional Encoding</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

         <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้องมี Self-Attention &amp; Positional Encoding?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การพัฒนาโมเดล Deep Learning สำหรับข้อมูลลำดับ (sequence data) เช่น ข้อความ, เสียง, และสัญญาณเวลา จำเป็นต้องเข้าใจ **โครงสร้างเชิงลำดับ (sequential structure)** ของข้อมูลอย่างลึกซึ้ง ในอดีต Recurrent Neural Networks (RNNs) และ Long Short-Term Memory (LSTM) เป็นทางเลือกหลักสำหรับการจัดการข้อมูลประเภทนี้ อย่างไรก็ตาม โมเดลเหล่านี้มีข้อจำกัดเชิงโครงสร้างที่ทำให้การเรียนรู้ dependencies ระยะไกล (long-range dependencies) เป็นเรื่องยากและไม่มีประสิทธิภาพสูงสุด
    </p>

    <p>
      การเปิดตัว **Transformer architecture** (Vaswani et al., 2017) ได้เปลี่ยนแปลงแนวทางของ Deep Learning สำหรับข้อมูลลำดับอย่างสิ้นเชิง โดยแทนที่การพึ่งพาโครงสร้าง RNN ด้วยกลไก **Self-Attention** ซึ่งช่วยให้โมเดลสามารถ "โฟกัส" ไปยังส่วนที่สำคัญของ input ได้อย่างยืดหยุ่นโดยไม่ขึ้นกับระยะทางเชิงลำดับ นี่คือจุดเปลี่ยนสำคัญที่ทำให้ Transformer กลายเป็นโครงสร้างหลักในงาน Natural Language Processing (NLP), Computer Vision (CV), Speech Processing และสาขาอื่น ๆ
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Self-Attention ช่วยให้โมเดลเข้าใจ **contextual relationships** ภายใน sequence ได้อย่างมีประสิทธิภาพ ในขณะที่ Positional Encoding เป็นกลไกที่ช่วยเติมข้อมูลเกี่ยวกับ "ลำดับเวลา" หรือ "ตำแหน่ง" ให้กับ Transformer ซึ่งโดยธรรมชาติไม่สามารถเข้าใจลำดับได้หากไม่มี encoding ประเภทนี้
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.1 บทบาทของ Self-Attention ในการแทนที่ RNN</h3>
    <p>
      Self-Attention ช่วยให้โมเดลสามารถ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เรียนรู้ **ความสัมพันธ์แบบ long-range** ได้ดีขึ้น โดยไม่มีข้อจำกัดเชิงโครงสร้างแบบ RNN</li>
      <li>รองรับ **การคำนวณแบบขนาน (parallelization)** ได้เต็มรูปแบบบน GPU</li>
      <li>ปรับขนาดได้ง่าย (**scalable**) สำหรับ dataset ขนาดใหญ่และงานที่มีความซับซ้อนสูง</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.2 ทำไมต้องมี Positional Encoding?</h3>
    <p>
      แม้ว่า Self-Attention จะมีข้อได้เปรียบเชิงโครงสร้างสูง แต่ด้วยความที่ Transformer ไม่มีโครงสร้างวนรอบ (recurrent structure) หรือ convolutional hierarchy จึงไม่มีความรู้เกี่ยวกับ **ลำดับ (order)** ของ token หรือ feature ใน sequence โดยตรง ซึ่งจะส่งผลเสียต่อความเข้าใจ semantic ของข้อมูล
    </p>
    <p>
      การใช้ **Positional Encoding** ช่วยเติมข้อมูลเกี่ยวกับตำแหน่งหรือลำดับของแต่ละ token เข้าไปใน embedding vector ซึ่งทำให้โมเดลสามารถ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>รับรู้ลำดับของข้อมูล</li>
      <li>เข้าใจความสัมพันธ์เชิงเวลา หรือ spatial dependency</li>
      <li>ถอดรหัส pattern ที่ขึ้นอยู่กับตำแหน่ง เช่น word order ใน NLP หรือ spatial layout ใน Vision</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.3 ข้อจำกัดของ Positional Encoding แบบเดิม และแนวโน้มวิจัยใหม่</h3>
    <p>
      ในระยะแรก Positional Encoding แบบ **Sinusoidal (Vaswani et al., 2017)** ได้รับความนิยม เนื่องจากมีความสามารถในการ **generalize** ไปยัง sequence ที่มีความยาวต่างกันได้ดี อย่างไรก็ตาม ในงานวิจัยใหม่ ๆ มีการพัฒนา Positional Encoding ที่เรียนรู้ได้ (learned positional embedding), Relative Positional Encoding และวิธี encoding แบบ adaptive ซึ่งช่วยเพิ่มประสิทธิภาพในงานประเภทต่าง ๆ
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การผสมผสาน **Self-Attention** กับ **Positional Encoding** คือหัวใจสำคัญที่ทำให้ Transformer สามารถเป็น **universal architecture** สำหรับ sequence modeling ได้อย่างแท้จริง และนี่คือแกนหลักของการปฏิวัติวงการ AI ในช่วง 5 ปีที่ผ่านมา
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.4 บทสรุปของบทนำ</h3>
    <p>
      การเข้าใจและใช้งาน Self-Attention ร่วมกับ Positional Encoding อย่างเหมาะสม ถือเป็นปัจจัยสำคัญในการออกแบบระบบ Deep Learning ยุคใหม่ สำหรับทุกสาขาที่เกี่ยวข้องกับข้อมูลลำดับ ความเข้าใจเชิงลึกในประเด็นเหล่านี้จะช่วยให้สามารถสร้างโมเดลที่มีประสิทธิภาพสูงขึ้น และสามารถประยุกต์ใช้ได้หลากหลายมากยิ่งขึ้น
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">1.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. (2017), "Attention Is All You Need", arXiv:1706.03762</li>
      <li>Gehring, J. et al. (2017), "Convolutional Sequence to Sequence Learning", arXiv:1705.03122</li>
      <li>Shaw, P. et al. (2018), "Self-Attention with Relative Position Representations", arXiv:1803.02155</li>
      <li>Liu, L. et al. (2020), "Learning Relative Positional Representations for Transformers", arXiv:2009.13658</li>
      <li>Stanford CS224N: Natural Language Processing with Deep Learning (2023), Lecture 9: Transformers and Self-Attention</li>
    </ul>
  </div>
</section>


  <section id="self-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Self-Attention คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      **Self-Attention** เป็นกลไกพื้นฐานที่อยู่เบื้องหลังความสำเร็จของ Transformer architecture ซึ่งทำให้สามารถ **เรียนรู้ความสัมพันธ์เชิงบริบท (contextual relationships)** ภายใน sequence ได้อย่างยืดหยุ่นและมีประสิทธิภาพ แตกต่างจากโครงสร้าง RNN หรือ CNN ที่พึ่งพาโครงสร้างเชิงลำดับหรือ receptive field แบบจำกัด
    </p>

    <p>
      แนวคิดหลักคือ การอนุญาตให้ทุก token ภายใน sequence สามารถ "มอง" ไปยัง token อื่น ๆ ได้ทั้งหมด โดยมี **attention weights** เป็นตัวกำหนดว่า token ใดควรได้รับความสำคัญมากน้อยเพียงใดเมื่อสร้าง representation ใหม่ของ token นั้น
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Self-Attention เปิดโอกาสให้โมเดลสามารถจับ **long-range dependency** ได้อย่างมีประสิทธิภาพ โดยไม่ขึ้นกับระยะทางของ token ภายใน sequence และทำให้สามารถคำนวณแบบขนาน (parallelization) ได้อย่างเต็มที่ — ข้อได้เปรียบเหนือกว่า RNN อย่างชัดเจน
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.1 แนวคิดพื้นฐานของ Self-Attention</h3>
    <p>
      ในแต่ละ layer ของ Self-Attention โมเดลจะเรียนรู้การ **re-weight** ข้อมูลภายใน sequence ผ่านขั้นตอนต่อไปนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>แปลง token แต่ละตัวเป็น **query (Q)**, **key (K)** และ **value (V)** vector</li>
      <li>คำนวณ **similarity score** ระหว่าง query ของ token ปัจจุบัน กับ key ของทุก token</li>
      <li>แปลง similarity score ด้วย Softmax เพื่อให้ได้ attention weights</li>
      <li>ใช้ attention weights ในการ **weighted sum** ของ value vectors เพื่อสร้าง representation ใหม่ของ token</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.2 สูตรคำนวณ Scaled Dot-Product Attention</h3>
    <pre className="bg-gray-800 text-green-300 text-sm p-4 rounded-lg overflow-x-auto">
      <code>
        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
      </code>
    </pre>
    <p>
      โดยที่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Q, K, V: Matrices of query, key, value vectors</li>
      <li>dₖ: Dimension ของ key vector เพื่อใช้ในการ scaling</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การใช้ scaling factor √dₖ มีความสำคัญอย่างมาก เพราะช่วยป้องกัน **numerical instability** ที่จะเกิดขึ้นเมื่อ dot product มีค่าใหญ่เกินไป ซึ่งจะทำให้ softmax saturated และ gradients มีขนาดเล็ก
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.3 ประโยชน์หลักของ Self-Attention</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">คุณสมบัติ</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อดี</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelization</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">คำนวณได้แบบขนาน ทำให้ training เร็วขึ้นมากเมื่อเทียบกับ RNN</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Long-range Dependency</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สามารถจับ dependency ระยะไกลได้โดยไม่ถูกจำกัดโดยโครงสร้าง sequential</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Interpretability</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สามารถวิเคราะห์ attention weights เพื่อดูว่าโมเดลโฟกัสไปที่ token ใด</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.4 ตัวอย่าง intuitive ของ Self-Attention</h3>
    <p>
      ตัวอย่างเช่น ในประโยคภาษาอังกฤษ "The animal didn't cross the street because it was too tired",  
      การแทนค่า "it" สามารถขึ้นกับ context ก่อนหน้าได้ เช่น "animal" หรือ "street" ซึ่ง Self-Attention จะช่วยโมเดลจัดการและ resolve ได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">2.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. (2017), "Attention Is All You Need", arXiv:1706.03762</li>
      <li>Stanford CS224N, Lecture 9-10: Transformers and Attention</li>
      <li>Shaw, P. et al. (2018), "Self-Attention with Relative Position Representations", arXiv:1803.02155</li>
      <li>Harvard NLP: "The Annotated Transformer" (Online Resource)</li>
    </ul>
  </div>
</section>


      <section id="multi-head-self-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Multi-Head Self-Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      **Multi-Head Self-Attention** เป็นการขยายแนวคิดของ Self-Attention โดยการเพิ่ม "หัว" (head) หลายชุด ที่สามารถเรียนรู้ **pattern ที่หลากหลาย** ภายใน sequence พร้อมกัน ส่งผลให้ model สามารถ capture ความสัมพันธ์เชิงซับซ้อนระหว่าง token ได้อย่างลึกซึ้งและมีมิติ
    </p>

    <p>
      แนวคิดหลักคือการคำนวณ attention หลายชุดแบบขนาน (parallel heads) จากนั้นนำผลลัพธ์ของแต่ละ head มารวมกัน (concatenate) แล้วส่งผ่าน linear projection เพื่อสร้าง representation ใหม่
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Multi-Head Self-Attention ทำให้ model มีความสามารถในการ "มองเห็น" ความสัมพันธ์จากมุมมองที่แตกต่างกัน เช่น บาง head อาจโฟกัสที่ dependency ใกล้เคียง (local dependency) ขณะที่อีก head อาจโฟกัสที่ dependency ระยะไกล (global dependency)
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.1 โครงสร้างพื้นฐานของ Multi-Head Self-Attention</h3>
    <p>
      ขั้นตอนการคำนวณ Multi-Head Self-Attention มีดังนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>นำ input sequence มาคำนวณ Q, K, V vectors แยกสำหรับแต่ละ head</li>
      <li>สำหรับแต่ละ head: คำนวณ Scaled Dot-Product Attention</li>
      <li>Concatenate ผลลัพธ์จากทุก head</li>
      <li>ส่งผ่าน Linear layer เพื่อสร้าง output representation</li>
    </ul>

    <pre className="bg-gray-800 text-green-300 text-sm p-4 rounded-lg overflow-x-auto">
      <code>
        MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wₒ {"\n"}
        where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
      </code>
    </pre>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.2 ข้อดีของการใช้หลาย Head</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Benefit</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Representation Diversity</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เรียนรู้ pattern ได้หลากหลายประเภทพร้อมกัน</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelization</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">คำนวณแบบขนานได้ดีบน hardware สมัยใหม่ (เช่น GPU)</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Efficiency</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ประสิทธิภาพสูงกว่า single-head attention เมื่อใช้ dimensionality เท่ากัน</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ในงานวิจัยเช่น "Attention Is All You Need" ได้แสดงให้เห็นว่า Multi-Head Self-Attention ทำให้ model สามารถเรียนรู้ **compositionality** และ **syntax awareness** ได้อย่างลึกซึ้งกว่าการใช้ Single-Head Self-Attention
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.3 การตั้งค่าหัว (Number of Heads)</h3>
    <p>
      จำนวนของ heads (h) ที่เลือกใช้ใน practice ขึ้นอยู่กับ architecture และขนาดของ model โดยทั่วไปค่าที่นิยมใช้คือ **8** หรือ **12** heads ใน model ขนาดกลาง และอาจสูงถึง **16-32** heads ใน model ขนาดใหญ่
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.4 ความซับซ้อนเชิงคำนวณ</h3>
    <p>
      แม้ว่า Multi-Head Self-Attention จะเพิ่มการใช้หน่วยความจำ (memory footprint) แต่สามารถใช้การ **parallelization** ของ hardware สมัยใหม่อย่างมีประสิทธิภาพ ซึ่งทำให้ training speed ได้รับผลกระทบน้อยมากเมื่อเทียบกับการใช้ single-head แบบ naive
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">3.5 อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani, A. et al. (2017), "Attention Is All You Need", arXiv:1706.03762</li>
      <li>Stanford CS224N, Lecture 10: Transformers and Attention</li>
      <li>Shaw, P. et al. (2018), "Self-Attention with Relative Position Representations", arXiv:1803.02155</li>
      <li>Harvard NLP: "The Annotated Transformer"</li>
      <li>Google AI Blog, Transformer Interpretability Visualization, 2021</li>
    </ul>
  </div>
</section>


 <section id="self-attention-benefits" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. ประโยชน์ของ Self-Attention</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Self-Attention mechanism ได้กลายมาเป็นแกนกลางของสถาปัตยกรรมสมัยใหม่ เช่น <strong>Transformer</strong> และ <strong>Vision Transformer (ViT)</strong> ด้วยเหตุผลสำคัญหลายประการ ซึ่งในหัวข้อนี้จะกล่าวถึง <strong>ประโยชน์หลัก</strong> ของ Self-Attention ทั้งในด้านเชิงสถาปัตยกรรม และเชิงการประมวลผลข้อมูล
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.1 ความสามารถในการ Modeling Dependency แบบ Global</h3>
    <p>
      Self-Attention สามารถเรียนรู้ <strong>ความสัมพันธ์ระหว่าง token ทั้งหมด</strong> ได้แบบ global โดยไม่ถูกจำกัดด้วย window ขนาดคงที่หรือลำดับเวลาเหมือน RNN หรือ CNN:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถจับ long-range dependency ได้ดีกว่า RNN/LSTM</li>
      <li>สามารถ model ความสัมพันธ์ non-sequential ได้อย่างมีประสิทธิภาพ</li>
      <li>ใช้ attention weights ที่ interpretable ทำให้ model สามารถ focus ไปยัง token ที่สำคัญได้</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ใน paper <strong>"Attention Is All You Need"</strong> ได้แสดงให้เห็นว่า Self-Attention สามารถจัดการกับ <strong>dependency ระยะไกล</strong> ได้อย่างมีประสิทธิภาพมากกว่า LSTM หรือ GRU และยังช่วยให้ model converges ได้เร็วขึ้น
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.2 ความสามารถในการคำนวณแบบขนาน (Parallelization)</h3>
    <p>
      Self-Attention ไม่มีโครงสร้างแบบ recurrent เหมือน RNN ส่งผลให้สามารถใช้ GPU/TPU ได้เต็มที่ในกระบวนการ training:
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">RNN/LSTM</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Self-Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Parallelization</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ต่ำ (sequential)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">สูง (fully parallel)</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Training Speed</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ช้า</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">เร็วมาก</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.3 ความยืดหยุ่นในการใช้งานข้ามโดเมน</h3>
    <p>
      Self-Attention ไม่ได้ถูกจำกัดอยู่เพียงงาน <strong>NLP</strong> เท่านั้น แต่ยังขยายไปสู่งานด้าน <strong>Vision</strong>, <strong>Speech</strong>, และ <strong>Multimodal</strong> ได้อย่างกว้างขวาง เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Vision Transformer (ViT) ใช้ Self-Attention แทน Convolution ในการเรียนรู้ features ของภาพ</li>
      <li>Speech Transformer ใช้ Self-Attention กับ sequence ของ spectral features</li>
      <li>Multimodal Transformer ใช้ Self-Attention สำหรับ align ข้อมูล text-image (เช่น CLIP)</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        งานวิจัยจาก <strong>Google Research</strong> (Dosovitskiy et al., 2020) พบว่า ViT ที่ใช้ Self-Attention สามารถ outperform CNN architectures ที่ออกแบบด้วยมือ บน task การจำแนกภาพขนาดใหญ่ (ImageNet) ได้อย่างมีนัยสำคัญ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.4 การลด inductive bias</h3>
    <p>
      ในขณะที่ CNN มี inductive bias สูง เช่น local receptive field, Self-Attention มี inductive bias ต่ำ ทำให้ model สามารถเรียนรู้ pattern ที่หลากหลายได้เอง:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>Self-Attention ไม่บังคับให้ focus แค่ local feature → สามารถเรียนรู้ pattern global ได้โดยอิสระ</li>
      <li>ทำให้โมเดลมีความยืดหยุ่นในการ generalize → เมื่อ train กับ domain ใหม่ ๆ</li>
      <li>ช่วยให้ Transformer-based models ถูกนำไปใช้ใน task ที่ CNN อาจไม่เหมาะสม เช่น modeling sequence ที่ไม่มี locality</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">4.5 การ scale ขึ้นแบบ linear</h3>
    <p>
      Self-Attention architecture ทำให้สามารถ scale models ได้อย่างมีประสิทธิภาพเมื่อเพิ่มขนาดของ data และ parameter:
    </p>
    <ul className="list-disc list-inside space-y-2 mt-2">
      <li>เพิ่มจำนวน layer หรือ hidden dimension → performance ของ Transformer เพิ่มขึ้นต่อเนื่อง</li>
      <li>งาน research หลายงาน (Kaplan et al., 2020) แสดงให้เห็น scaling law ที่ชัดเจนใน Self-Attention models</li>
      <li>เหมาะกับ era ของ Big Data และ High-capacity compute</li>
    </ul>
  </div>
</section>



<section id="positional-encoding-importance" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Positional Encoding: ทำไมจำเป็น?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      แม้ว่า **Self-Attention** จะมีความสามารถในการจับ **dependency แบบ global** ได้อย่างยอดเยี่ยม แต่กลไกพื้นฐานของมัน **ไม่มีความรู้เกี่ยวกับลำดับของข้อมูล (order)** ใน sequence โดยธรรมชาติ.  
      ซึ่งถือเป็นข้อจำกัดสำคัญเมื่อทำงานกับข้อมูลประเภท sequence เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ข้อความในภาษา (Natural Language) → ลำดับของคำส่งผลต่อความหมาย</li>
      <li>สัญญาณเวลา (Time Series) → ลำดับของจุดข้อมูลมีความหมายเชิง causal</li>
      <li>DNA/RNA Sequence → ลำดับของ nucleotide มีผลต่อโครงสร้างทางชีวภาพ</li>
    </ul>

    <p>
      เนื่องจาก **Self-Attention** คำนวณ similarity matrix โดยไม่คำนึงถึงตำแหน่ง token โดยตรง → ต้องมีวิธี inject ข้อมูลตำแหน่งเข้าไปใน embedding.
      เทคนิคนี้เรียกว่า **Positional Encoding**.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.1 ปัญหาถ้าไม่มี Positional Encoding</h3>
    <p>
      หากไม่ใส่ Positional Encoding → Self-Attention จะ treat input เป็น **bag of tokens**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ประโยค "The cat chased the mouse." กับ "The mouse chased the cat." → จะให้ vector representation ที่เกือบเหมือนกัน</li>
      <li>ในงาน Time Series → ไม่สามารถ model temporal trend ได้</li>
      <li>ในงาน Speech → ลำดับ phoneme จะถูกทำลาย → ส่งผลให้การสังเคราะห์เสียงล้มเหลว</li>
    </ul>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ใน paper "Attention Is All You Need" (Vaswani et al., 2017) การเพิ่ม **Sinusoidal Positional Encoding** ลงใน input embedding ของ Transformer เป็น key factor ที่ทำให้ model สามารถเรียนรู้ structure ของลำดับข้อมูลได้อย่างถูกต้อง.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.2 วิธีการเพิ่ม Positional Encoding</h3>
    <p>
      Positional Encoding ถูก inject เข้าไปใน model โดยการ **บวกเข้ากับ input embedding** แต่ละตำแหน่ง:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`InputEmbedding + PositionalEncoding → Input to Self-Attention`}
      </code>
    </pre>

    <p>โดยทั่วไป จะมี 2 วิธีหลัก:</p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Fixed Encoding</strong> เช่น Sinusoidal Function → สามารถ generalize ข้าม sequence length ที่ไม่เคยเห็น</li>
      <li><strong>Learned Encoding</strong> → เรียนรู้ vector encoding แต่ละตำแหน่งระหว่าง training</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.3 ตัวอย่าง Sinusoidal Positional Encoding</h3>
    <p>
      Vaswani et al. ใช้ sinusoidal encoding ดังนี้:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`}
      </code>
    </pre>
    <p>
      โดยที่ <code>pos</code> คือตำแหน่ง token, <code>i</code> คือ dimension index → ทำให้ model สามารถ encode **relative distance** ได้อย่างต่อเนื่อง.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.4 คุณสมบัติที่ desirable ของ Positional Encoding</h3>
    <p>Positional Encoding ที่ดีควรมี property ต่อไปนี้:</p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Property</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Benefit</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Smoothness</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Model สามารถเรียนรู้ความสัมพันธ์ต่อเนื่อง</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Extrapolation</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generalize ข้าม sequence length ได้</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Relative distance preservation</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Encoding ควร encode relative order อย่างเหมาะสม</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">5.5 บทสรุป</h3>
    <p>
      **Positional Encoding** เป็นกลไกที่ขาดไม่ได้ในการทำให้ Transformer สามารถเรียนรู้และจัดการข้อมูลลำดับได้อย่างถูกต้อง.
      ความเรียบง่ายของการ implement (เพียงการบวกเข้ากับ embedding) ทำให้เป็น technique ที่มีประสิทธิภาพสูง และยังสามารถขยายต่อยอดได้หลากหลาย เช่น **Relative Positional Encoding** และ **Rotary Positional Embedding** (RoPE) ใน LLM รุ่นใหม่.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Shaw et al., "Self-Attention with Relative Position Representations", NAACL 2018</li>
      <li>Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2021</li>
    </ul>
  </div>
</section>


  <section id="positional-encoding-types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Types of Positional Encoding</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การออกแบบ **Positional Encoding** ถือเป็นหัวใจสำคัญในการเสริมความสามารถของ Transformer ในการประมวลผลข้อมูลลำดับ (sequence data).  
      ตลอดช่วงหลายปีที่ผ่านมา มีการพัฒนา **หลายแนวทาง** ของ Positional Encoding โดยแต่ละแบบมีข้อดีข้อด้อยและ **ความเหมาะสม** กับประเภท task ที่ต่างกัน.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        งานวิจัยในช่วงหลังแสดงให้เห็นว่า **รูปแบบ Positional Encoding** ที่เลือกใช้งาน ส่งผลต่อ performance และ generalization ของ Transformer อย่างมีนัยสำคัญ, โดยเฉพาะอย่างยิ่งเมื่อทำงานกับ **long sequence**, **streaming** หรือ **extrapolation** task.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.1 Fixed Positional Encoding (Sinusoidal)</h3>
    <p>
      นำเสนอครั้งแรกใน paper <strong>Attention Is All You Need</strong> (Vaswani et al., 2017).  
      วิธีนี้ใช้ฟังก์ชัน <code>sin</code> และ <code>cos</code> ที่มีความถี่ต่างกัน เพื่อ encode ตำแหน่งของ token:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`}
      </code>
    </pre>
    <p>
      ข้อดีคือมีความสามารถ **generalize** ได้ดีแม้กับ sequence ที่ยาวกว่าที่เคยเห็นระหว่าง training.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.2 Learned Positional Embedding</h3>
    <p>
      อีกทางเลือกคือการเรียนรู้ vector positional embedding สำหรับแต่ละตำแหน่ง:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`PositionEmbedding[position] = trainable vector ∈ ℝ^d_model`}
      </code>
    </pre>
    <p>
      แบบนี้สามารถปรับให้เหมาะสมกับ task ได้มากกว่า **Fixed encoding** แต่มีข้อเสียคืออาจ **generalize ได้ไม่ดี** กับ sequence length ที่ไม่เคยเจอ.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.3 Relative Positional Encoding</h3>
    <p>
      การ encode **relative distance** ระหว่าง tokens แทนที่จะ encode absolute position:
    </p>
    <p>
      ตัวอย่างงานสำคัญ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Shaw et al. (2018) → <strong>Self-Attention with Relative Position Representations</strong></li>
      <li>T5 → ใช้ Relative Position Bias (learnable bias term)</li>
      <li>DeBERTa → ใช้ Disentangled attention + relative position</li>
    </ul>
    <p>
      ข้อดีคือช่วยให้ model สามารถ **จับ pattern ที่ขึ้นกับระยะห่างระหว่าง token** ได้ดี และทำงานได้ robust กว่าบน **long sequence**.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในงาน NLP ขนาดใหญ่เช่น **T5** และ **DeBERTa**, Relative Positional Encoding กลายเป็น **มาตรฐานใหม่** แทน Fixed หรือ Learned absolute encoding → ช่วยปรับปรุง **generalization** บน long-sequence task อย่างเห็นได้ชัด.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.4 Rotary Positional Encoding (RoPE)</h3>
    <p>
      เทคนิคใหม่ที่นำไปใช้ในหลาย LLM เช่น LLaMA, GPT-NeoX.  
      วิธีนี้แทนที่การบวก Positional Encoding ด้วยการ **rotate** vector Q และ K ตามมุมที่ขึ้นกับตำแหน่ง:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`[x1, x2] → [x1 * cos(θ) - x2 * sin(θ),  
              x1 * sin(θ) + x2 * cos(θ)]`}
      </code>
    </pre>
    <p>
      จุดเด่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Memory-efficient</li>
      <li>สามารถ **extrapolate** ไปยัง sequence length ที่ไม่เคยเห็นได้ดี</li>
      <li>ง่ายต่อการ integrate เข้ากับ Self-Attention</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">6.5 เปรียบเทียบภาพรวม</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ประเภท</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อดี</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sinusoidal (Fixed)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generalize ดี, Parameter-free</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited modeling capacity</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Learned Absolute</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Flexible, Powerful</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generalize ข้าม length ไม่ดี</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Relative</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Handle long sequence, capture relative distance</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Implementation complex กว่า</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Rotary (RoPE)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Efficient, powerful, great extrapolation</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Complex to analyze theoretically</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Shaw et al., "Self-Attention with Relative Position Representations", NAACL 2018</li>
      <li>Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2021</li>
      <li>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)", JMLR 2020</li>
      <li>He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", ICLR 2021</li>
    </ul>
  </div>
</section>

   <section id="positional-vs-relative" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Positional Encoding vs Relative Positional Encoding</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      เมื่อ Transformer ถูกเสนอขึ้นครั้งแรก (Vaswani et al., 2017), โมเดลนี้ใช้ **Fixed Absolute Positional Encoding** เพื่อแทรกข้อมูลเกี่ยวกับตำแหน่ง token เข้าไปใน input representation.  
      อย่างไรก็ตาม งานวิจัยในช่วงหลังได้แสดงให้เห็นว่า **Relative Positional Encoding** มีข้อได้เปรียบในหลาย task โดยเฉพาะกับ **long sequence modeling** และ **generalization**.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การเปลี่ยนจาก Absolute → Relative Positional Encoding เป็นหนึ่งในปัจจัยหลักที่ทำให้โมเดล NLP รุ่นใหม่ เช่น **T5, DeBERTa** สามารถแซง BERT รุ่นเดิมใน benchmark ต่าง ๆ ได้อย่างมีนัยสำคัญ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.1 Absolute Positional Encoding</h3>
    <p>
      Absolute Positional Encoding (PE) ใส่ตำแหน่ง **absolute** ของแต่ละ token เข้าไปใน embedding layer เช่น:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`InputEmbedding[token_position] + PositionalEncoding[position]`}
      </code>
    </pre>
    <p>
      ลักษณะการทำงาน:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Sinusoidal** encoding → Fixed deterministic function</li>
      <li>**Learned absolute embedding** → Parameterized, trainable</li>
    </ul>
    <p>
      ข้อจำกัด:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ไม่สะท้อนความสัมพันธ์ **relative** ระหว่าง tokens (เช่น "token นี้อยู่ถัดจาก token ไหน")</li>
      <li>ยากต่อการ generalize ไปยัง sequence ที่ยาวกว่าช่วง training</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.2 Relative Positional Encoding</h3>
    <p>
      Relative Positional Encoding แทนที่จะ encode **position ของ token**, จะ encode **relative distance** ระหว่างคู่ token ที่เข้าสู่ attention mechanism:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`Attention(Q_i, K_j) += RelativeBias[i - j]`}
      </code>
    </pre>
    <p>
      ข้อดี:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>สามารถจับ pattern ที่ขึ้นกับ **ความห่าง** ระหว่าง token ได้ดี (เช่น n-gram patterns, dependency distance)</li>
      <li>Generalize ได้ดีแม้ sequence length เปลี่ยน</li>
      <li>เหมาะกับ task ที่ context มีลักษณะ **local** + **long-range** ปะปนกัน (เช่น language modeling, document modeling, audio)</li>
    </ul>
    <p>
      ตัวอย่างโมเดลที่ใช้ Relative PE:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>T5</strong>: Relative Bias term</li>
      <li><strong>DeBERTa</strong>: Disentangled Relative Attention</li>
      <li><strong>Transformer-XL</strong>: Relative PE + Recurrence mechanism</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">7.3 เปรียบเทียบภาพรวม</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Aspect</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Absolute PE</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Relative PE</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Generalization to Long Sequences</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Excellent</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Capturing Relative Patterns</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Poor</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Good</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Implementation Complexity</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Simple</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">More Complex</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Performance on NLP Benchmarks</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Competitive (baseline)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Often superior</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow mt-6">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Relative Positional Encoding ช่วยให้ Transformer เข้าใจความสัมพันธ์ระหว่าง token ได้ลึกซึ้งกว่า Absolute PE และเป็นแนวทางที่นิยมใน **state-of-the-art NLP models** รุ่นใหม่มากขึ้นเรื่อย ๆ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Shaw et al., "Self-Attention with Relative Position Representations", NAACL 2018</li>
      <li>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)", JMLR 2020</li>
      <li>He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", ICLR 2021</li>
      <li>Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", ACL 2019</li>
    </ul>
  </div>
</section>


   <section id="contextualized-representation" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Insight: Self-Attention + Positional Encoding = "Contextualized Representation"</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      หนึ่งในเหตุผลหลักที่ **Transformer** ได้รับความนิยมและประสบความสำเร็จอย่างสูงในหลากหลาย task คือความสามารถในการสร้าง **Contextualized Representation** — การนำเสนอ vector ของ token ที่มีการปรับเปลี่ยนแบบ dynamic ตาม context โดยรอบ.  
      กุญแจสำคัญของความสามารถนี้เกิดจากการผสาน **Self-Attention** + **Positional Encoding** เข้าด้วยกัน.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ในขณะที่ Positional Encoding ให้ความรู้เรื่องตำแหน่ง **absolute / relative** ของ token, Self-Attention สร้างการเชื่อมโยงระหว่าง token แต่ละตัว → ส่งผลให้ representation ของ token แต่ละตัวมี "ความหมาย" ที่ขึ้นกับ context รอบข้าง → เรียกว่า **Contextualized Representation**.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.1 Self-Attention ทำอะไร?</h3>
    <p>
      Self-Attention เป็นกลไกที่ช่วยให้แต่ละ token **เรียนรู้การเชื่อมโยง (dependencies)** กับ token อื่น ๆ ใน sequence:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>แต่ละ token จะถูกแมปเป็น Q, K, V</li>
      <li>Attention scores คำนวณ pairwise interaction ระหว่าง token ทุกคู่</li>
      <li>ผลลัพธ์ → linear combination ของ value vectors ที่ถูก weight ด้วย attention score</li>
    </ul>
    <p>
      กล่าวคือ Representation สุดท้ายของ token ใด ๆ จะขึ้นกับ **ทุก token ใน sequence**, แต่ในอัตราส่วนที่ปรับตาม **Attention Weights**.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.2 ทำไมต้องมี Positional Encoding?</h3>
    <p>
      อย่างไรก็ตาม **Self-Attention ไม่ได้ preserve positional order** ของ token — มัน invariant ต่อ permutation ของ input.  
      → เพื่อให้โมเดลเข้าใจลำดับ token → จำเป็นต้องเพิ่ม Positional Encoding เข้าไปใน embedding layer:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`InputEmbedding[token_position] + PositionalEncoding[position]`}
      </code>
    </pre>
    <p>
      เมื่อผ่าน attention → token representation จึง encode ทั้ง:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Content** ของ token</li>
      <li>**Position** ของ token ใน sequence</li>
      <li>**Dependency** กับ tokens อื่น ๆ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.3 Contextualization ในเชิงคณิตศาสตร์</h3>
    <p>
      Representation สุดท้ายของ token i ใน layer l สามารถมองเป็น:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`H_i^{(l)} = LayerNorm(H_i^{(l-1)} + MultiHeadSelfAttention(H^{(l-1)}))`}
      </code>
    </pre>
    <p>
      โดย H_i^(l) → contextualized vector ของ token i → ซึ่ง encode **context ของ tokens อื่น ๆ ทั้งหมด**.
    </p>
    <p>
      และเนื่องจากมี Positional Encoding:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto text-sm">
      <code>
{`H_i^{(0)} = TokenEmbedding + PositionalEncoding`}
      </code>
    </pre>
    <p>
      ทำให้ H_i^(l) ใน layer บน ๆ → ไม่เพียงแต่ encode **semantic meaning** แต่ยัง encode **relative / absolute position awareness** ด้วย.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.4 Visualization</h3>
    <p>
      Visualization งานวิจัยหลายฉบับแสดงให้เห็นว่า Self-Attention Heads ใน layer ต่าง ๆ จะ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>บาง head → focus ที่ **local context** (เช่น bigram, trigram)</li>
      <li>บาง head → focus ที่ **global context** (เช่น subject → verb linkage, document structure)</li>
      <li>บาง head → encode **positional patterns** → ทำงาน complement กับ Positional Encoding</li>
    </ul>
    <p>
      ตัวอย่าง visualization จาก *"Visualizing Transformer Representations" (Stanford CS224N)* → แสดงให้เห็นว่า attention heads สามารถเรียนรู้ **hierarchical syntactic structure** ได้แม้ไม่มี explicit parsing tree.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">8.5 บทสรุปเชิง Insight</h3>
    <p>
      เมื่อ **Self-Attention** + **Positional Encoding** ผสานกัน:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Token Representation → ไม่ใช่แค่ static embedding → แต่เป็น **Contextualized Representation** → dynamic ตาม context รอบตัว</li>
      <li>ทำให้ Transformer สามารถ model **complex dependencies** ได้อย่างมีประสิทธิภาพสูงกว่าระบบ RNN / CNN</li>
      <li>พื้นฐานของ power ของ LLM (เช่น GPT, T5, BERT) → อยู่ตรงนี้ → การ encode sequence ทั้งหมดใน vector ของแต่ละ token</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        แนวคิด "Contextualized Representation" เป็นเสาหลักที่ทำให้โมเดล NLP รุ่นใหม่ → เข้าใจ meaning ของ token อย่างลึกซึ้ง → ส่งผลต่อความสำเร็จใน task เช่น QA, summarization, translation, code generation → ทิ้งโมเดล static embedding รุ่นก่อน (เช่น word2vec, GloVe) ไว้ขาดลอย.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)", JMLR 2020</li>
      <li>Clark et al., "What Does BERT Look at? An Analysis of BERT’s Attention", ACL 2019</li>
      <li>Stanford CS224N, "Visualizing Transformer Representations", Lecture Notes</li>
      <li>Tenney et al., "BERT Rediscovers the Classical NLP Pipeline", ACL 2019</li>
    </ul>
  </div>
</section>


 <section id="research-evolution" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Research Evolution</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การพัฒนาของ **Self-Attention และ Positional Encoding** ในวงการวิจัย AI และ Deep Learning เป็นหนึ่งในตัวอย่างของวิวัฒนาการอย่างรวดเร็วที่นำไปสู่การเปลี่ยนแปลงโครงสร้างพื้นฐานของสถาปัตยกรรมโมเดล.  
      ใน Section นี้จะสำรวจการเปลี่ยนแปลงจาก Transformer รุ่นแรก จนถึงแนวทางใหม่ ๆ ที่ผลักดันขอบเขตความสามารถของ Self-Attention และ Positional Encoding ให้ลึกซึ้งยิ่งขึ้น.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        Self-Attention และ Positional Encoding ไม่ได้หยุดพัฒนาอยู่แค่ใน Transformer รุ่นแรก. งานวิจัยในช่วง 5 ปีหลังได้ขยายแนวคิดนี้ให้สามารถรองรับ sequence ที่ยาวมากขึ้น, เพิ่ม efficiency, และประยุกต์สู่ multimodal learning อย่างมีประสิทธิภาพ.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.1 Transformer รุ่นแรก (2017)</h3>
    <p>
      งาน **Attention is All You Need** (Vaswani et al., 2017) ได้เสนอ Self-Attention + Sinusoidal Positional Encoding:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Self-Attention → Global interaction ระหว่าง tokens</li>
      <li>Positional Encoding → ใช้ sine และ cosine เพื่อ encode ตำแหน่งแบบ deterministic</li>
    </ul>
    <p>
      ผลลัพธ์: โมเดล NLP รุ่นแรกที่เอาชนะ RNN/CNN แบบเดิมได้อย่างมีนัยสำคัญ.
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.2 จาก NLP → Multimodal Learning</h3>
    <p>
      หลังปี 2018-2019 โมเดล Transformer ได้ขยายไปสู่ **Vision** และ **Multimodal**:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**ViT (Vision Transformer)** (Dosovitskiy et al., 2020) → ใช้ Positional Embedding กับ image patches</li>
      <li>**CLIP (Radford et al., 2021)** → cross-modal attention → alignment text-image representation</li>
      <li>**Perceiver (Jaegle et al., 2021)** → ใช้ learned positional encoding กับ arbitrary modality (video, audio, sensor data)</li>
    </ul>
    <p>
      ในแต่ละกรณี Positional Encoding ได้มีการปรับเปลี่ยนให้เหมาะกับ structure ของ data ที่ไม่ใช่แค่ sequence text.
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        การนำ Self-Attention + Positional Encoding จาก text → image → video → audio → sensor data → เป็นตัวอย่างของ generalization power ของกลไกนี้ → พิสูจน์ว่าไม่จำกัดอยู่แค่ NLP.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.3 Long-Sequence Models</h3>
    <p>
      Limit ดั้งเดิมของ Self-Attention → complexity O(n²).  
      จึงเกิดงานวิจัยพยายามแก้ไข:
    </p>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Model</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Technique</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Impact</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer-XL</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Segment-level recurrence</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Handle long-range dependency</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Reformer</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Locality-sensitive hashing</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Reduce attention complexity to O(n log n)</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Longformer</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sliding window attention</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Efficient document-level NLP</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.4 Relative Positional Encoding</h3>
    <p>
      งานใหม่ ๆ เช่น Transformer-XL และ T5 ได้ปรับจาก absolute → relative positional encoding:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Relative Encoding → model สามารถ generalize ไปยัง sequences ที่ยาวกว่าที่เห็นตอน training ได้ดีกว่า</li>
      <li>ช่วยให้โมเดลสามารถ reuse attention head ในระดับที่ดีขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">9.5 โมเดลล่าสุด</h3>
    <p>
      ตัวอย่างของงานวิจัยล่าสุดที่ยังผลักดัน Self-Attention + Positional Encoding:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**BigBird** → sparse global-local attention</li>
      <li>**Perceiver IO** → handling arbitrary input-output spaces</li>
      <li>**FlashAttention** → efficient attention computation on GPU</li>
      <li>**xPos** → continuous and infinite-length positional encoding</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Transformer-XL: Dai et al., ACL 2019</li>
      <li>Reformer: Kitaev et al., ICLR 2020</li>
      <li>Longformer: Beltagy et al., ACL 2020</li>
      <li>Perceiver: Jaegle et al., ICML 2021</li>
      <li>FlashAttention: Dao et al., ICML 2023</li>
    </ul>
  </div>
</section>


    <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Use Cases ที่ Self-Attention สำคัญมาก</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      Self-Attention ได้กลายเป็นแกนกลางของสถาปัตยกรรม Deep Learning สมัยใหม่ และถูกนำไปประยุกต์ใช้ใน Use Cases ที่หลากหลายอย่างลึกซึ้ง.  
      จาก NLP, Computer Vision, ไปจนถึง Multimodal learning และ Scientific modeling — ประสิทธิภาพของ Self-Attention ได้เปลี่ยน landscape ของงานวิจัยและการใช้งานเชิงพาณิชย์ไปอย่างสิ้นเชิง.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        ความสามารถในการ "โฟกัสแบบ dynamic" บนทุกตำแหน่งของ input → ทำให้ Self-Attention เหมาะกับงานที่ต้องเข้าใจ "contextual relationship" ทั้งแบบ global และ local — ซึ่งเป็นหัวใจสำคัญของการเรียนรู้ representation ที่ generalizable.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.1 NLP: Language Understanding & Generation</h3>
    <p>
      Self-Attention เริ่มต้นจาก NLP และยังคงเป็นแกนสำคัญที่สุดในงานกลุ่มนี้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Machine Translation** → Transformer model แทน RNN-based models ได้อย่างสมบูรณ์</li>
      <li>**Text Summarization** → attention ช่วยให้เข้าใจความสำคัญแบบ global ของแต่ละประโยค</li>
      <li>**Question Answering** → contextual embedding ที่เรียนรู้ผ่าน attention</li>
      <li>**Large Language Models (LLM)** → เช่น GPT series, BERT, T5 ใช้ stacked Self-Attention เป็นแกนหลัก</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.2 Computer Vision: Beyond CNN</h3>
    <p>
      Vision Transformer (ViT) ได้เปลี่ยนแปลงแนวทางของ Computer Vision:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Image Classification** → ViT outperform CNN บน ImageNet เมื่อมี data เพียงพอ</li>
      <li>**Object Detection** → DETR ใช้ Self-Attention แทนการใช้ anchor-based CNN detection</li>
      <li>**Image Segmentation** → Self-Attention ช่วย capture long-range spatial dependency ได้ดีกว่า CNN</li>
    </ul>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        ใน Vision, Self-Attention ได้เปลี่ยน paradigm: จาก local receptive field ของ CNN → สู่ global receptive field ในทุก layer — ส่งผลต่อความสามารถในการเข้าใจ scene ที่ซับซ้อน.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.3 Multimodal Learning</h3>
    <p>
      Self-Attention มีบทบาทสำคัญใน Multimodal Models ที่ต้อง align representations จาก data ที่มี modality แตกต่างกัน:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**CLIP** → cross-attention ระหว่าง text ↔ image</li>
      <li>**ALIGN** → scalable multimodal representation learning</li>
      <li>**Flamingo (DeepMind)** → vision-language model ที่ใช้ cross-attention stack</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.4 Speech Processing</h3>
    <p>
      Self-Attention ยังมีบทบาทสำคัญในงาน Speech:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Speech Recognition** → conformer model (convolution + self-attention hybrid)</li>
      <li>**Speech Synthesis (TTS)** → attention alignment layer ใน Tacotron2 และ FastSpeech</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">10.5 Scientific Modeling</h3>
    <p>
      ตัวอย่าง emerging ใช้งานในด้าน Scientific ML:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>**Protein Structure Prediction** → AlphaFold2 ใช้ stacked attention layers เพื่อ model residue-residue contact map</li>
      <li>**Molecular modeling** → SE(3)-equivariant transformers สำหรับเรียนรู้ structure of molecules</li>
      <li>**Weather Forecasting** → graph-based attention models สำหรับ multi-scale temporal modeling</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">ตัวอย่างสรุป Use Cases แบบเปรียบเทียบ</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Domain</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Example Model</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Key Role of Self-Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">NLP</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GPT-4, T5</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Contextual token representation</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Vision</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">ViT, DETR</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Global spatial understanding</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Multimodal</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CLIP, Flamingo</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Cross-modal alignment</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Speech</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Conformer</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Temporal global context modeling</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021</li>
      <li>Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021</li>
      <li>Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", NeurIPS 2020</li>
      <li>Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature 2021</li>
    </ul>
  </div>
</section>


        <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Box</h2>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การผสานกันของ **Self-Attention** และ **Positional Encoding** ได้สร้างความก้าวหน้าที่สำคัญให้กับโลกของ Deep Learning.  
      ความสามารถของโมเดลในการสร้าง **Contextualized Representation** — ที่ทั้งเข้าใจลำดับและบริบทเชิงลึก — เป็นหัวใจที่ทำให้ Transformer และ LLMs ประสบความสำเร็จอย่างกว้างขวาง.
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight Box</p>
      <p>
        การเพิ่ม Positional Encoding ช่วยให้ Self-Attention "เข้าใจ" ตำแหน่งในลำดับ — เมื่อรวมกับ Self-Attention ที่มี global receptive field → โมเดลสามารถ **จับ dependency ที่หลากหลายข้ามตำแหน่งได้ดีเยี่ยม**.  
        นี่คือจุดแตกต่างสำคัญที่ทำให้ Transformer outperform RNN และ CNN ในหลากหลายงาน.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">Contextualization จากการรวม Self-Attention + Positional Encoding</h3>
    <p>
      Transformer ได้แสดงให้เห็นว่า **ไม่จำเป็นต้องมีโครงสร้างลูปหรือ convolution แบบเดิม** — การใช้ Self-Attention ที่มี Positional Encoding ช่วยให้เกิด embedding ที่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>เข้าใจความสัมพันธ์ global → dependency ข้าม span ยาว ๆ</li>
      <li>เข้าใจลำดับ → ผ่าน encoding ของตำแหน่งที่ชัดเจน</li>
      <li>สร้าง embedding ที่ generalizable และ reusable → pretrain + finetune ได้ดี</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">Highlight: ความแตกต่างจาก CNN และ RNN</h3>
    <table className="min-w-full table-auto text-sm text-left border border-gray-300 dark:border-gray-600 rounded-xl overflow-hidden shadow-sm mt-4">
      <thead className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
        <tr>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Architecture</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Sequence Modeling</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Position Awareness</th>
          <th className="px-4 py-3 border border-gray-300 dark:border-gray-600 font-semibold">Scalability</th>
        </tr>
      </thead>
      <tbody>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">RNN / LSTM</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Sequential dependency</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Inherent in recursion</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Limited by sequential computation</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">CNN</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Local receptive field</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Positional via structure</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Highly parallel</td>
        </tr>
        <tr className="bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200">
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Transformer (Self-Attention + PE)</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Global contextualized</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Explicit via PE</td>
          <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Highly scalable & parallel</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 shadow mt-6">
      <p className="font-semibold mb-2">Highlight Box</p>
      <p>
        **Contextualized representation** จาก Self-Attention + Positional Encoding คือรากฐานที่ทำให้ Transformer-based models เหมาะกับงาน sequence modeling ในยุคปัจจุบันมากกว่า architecture แบบเดิม.  
        หลักการนี้ถูกนำไปใช้อย่างกว้างขวางในทุก domain → NLP, Vision, Speech, Multimodal, Science.
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-4">Implications สำหรับอนาคต</h3>
    <p>
      แนวโน้มล่าสุดใน research เช่น "Attention-Free Transformer" หรือ "Linear Transformers" แสดงให้เห็นว่า community กำลังพยายาม **ขยายขีดจำกัดของ Self-Attention** ให้ใช้กับ sequence ที่ยาวมากขึ้น → โดยยังคงประโยชน์ของ contextualized representation ไว้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ใช้ linearized attention → ลด O(n²) → O(n) complexity</li>
      <li>พัฒนา PE รูปแบบใหม่ เช่น Relative Position Bias, Rotary Positional Embedding</li>
      <li>สร้าง efficient Transformer architecture สำหรับ vision, video, speech</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-4">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Vaswani et al., "Attention Is All You Need", NeurIPS 2017</li>
      <li>Press et al., "Train Short, Test Long: Attention with Linear Biases", arXiv 2021</li>
      <li>Su et al., "RoFormer: Enhanced Transformer with Rotary Positional Embedding", arXiv 2021</li>
      <li>Peng et al., "RWKV: Reinventing RNNs with Transformer Quality", arXiv 2023</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day52 theme={theme} />
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
        <ScrollSpy_Ai_Day52 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day52_SelfAttention_PositionalEncoding;
