import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day34 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day34";
import MiniQuiz_Day34 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day34";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day34_EncoderDecoder = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });


  const img1 = cld.image("Day34_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day34_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day34_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day34_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day34_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day34_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day34_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day34_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day34_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day34_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day34_11").format("auto").quality("auto").resize(scale().width(500));
  const img12 = cld.image("Day34_12").format("auto").quality("auto").resize(scale().width(500));
  const img13 = cld.image("Day34_13").format("auto").quality("auto").resize(scale().width(500));
  const img14 = cld.image("Day34_14").format("auto").quality("auto").resize(scale().width(500));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 34: Encoder–Decoder Structure</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

<section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้องมี Encoder–Decoder?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">บริบทของปัญหาในลำดับข้อมูล</h3>
    <p>
      ในงานด้านการประมวลผลภาษาธรรมชาติ (NLP), การแปลภาษา, และการสรุปข้อความ ระบบต้องสามารถเข้าใจข้อมูลต้นทาง (input) และสร้างผลลัพธ์ใหม่ (output) อย่างมีโครงสร้าง ซึ่งเป็นปัญหาที่มีความซับซ้อนสูง เนื่องจากทั้ง input และ output มีลำดับ (sequence) ที่ยาว และอาจมีความยาวแตกต่างกัน โมเดลแบบ Encoder–Decoder จึงถูกพัฒนาเพื่อจัดการกับปัญหาเหล่านี้อย่างเป็นระบบ
    </p>

    <h3 className="text-xl font-semibold">นิยามของ Encoder–Decoder</h3>
    <p>
      Encoder–Decoder เป็นโครงสร้างของโมเดลที่แบ่งการประมวลผลออกเป็นสองขั้นตอนหลัก คือ:
    </p>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Encoder:</strong> รับข้อมูลขาเข้าเป็นลำดับ และแปลงเป็น representation หรือเวกเตอร์บริบท (context vector)</li>
      <li><strong>Decoder:</strong> สร้างลำดับข้อมูลใหม่โดยอิงจาก representation ที่ได้จาก Encoder</li>
    </ul>

    <h3 className="text-xl font-semibold">ความสำคัญทางแนวคิด</h3>
    <p>
      Encoder–Decoder เปรียบได้กับกระบวนการแปลภาษาของมนุษย์: ผู้ฟังต้องเข้าใจความหมายของประโยคทั้งหมดก่อน จึงสามารถตอบสนองหรือแปลเป็นอีกภาษาหนึ่งได้อย่างถูกต้อง แนวคิดนี้นำไปสู่การพัฒนาโมเดลที่สามารถสร้างข้อความ, คำอธิบายภาพ, และการตอบคำถามเชิงตรรกะได้ดีขึ้น
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
      <strong>Insight Box:</strong> โครงสร้าง Encoder–Decoder เป็นหนึ่งในกรอบสถาปัตยกรรมหลักที่ทำให้เกิดความก้าวหน้าในโมเดล deep learning ด้านลำดับ (sequence-to-sequence) โดยเฉพาะในยุคก่อน Transformer
    </div>

    <h3 className="text-xl font-semibold">การใช้งานจริงในระบบสมัยใหม่</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Google Translate (ก่อนยุค Transformer) ใช้ RNN-based Encoder–Decoder</li>
      <li>BERT-to-GPT (Encoder–Decoder แบบ pretrain + generate)</li>
      <li>Image captioning: แปลงภาพเป็นข้อความโดยใช้ CNN เป็น Encoder และ LSTM เป็น Decoder</li>
    </ul>

    <h3 className="text-xl font-semibold">ความแตกต่างจากการประมวลผลแบบโมโนลิทิก</h3>
    <p>
      โมเดลทั่วไปที่ไม่มีการแยกส่วน Encoder–Decoder จะต้องเรียนรู้บริบท input และการ generate output พร้อมกันในเฟรมเวิร์กเดียว ซึ่งอาจทำให้โมเดลมีความสามารถจำกัดในการเข้าใจหรือถ่ายโอนบริบทเมื่อเปลี่ยน domain
    </p>

    <h3 className="text-xl font-semibold">ประโยชน์ของการแยกส่วน (Modularization)</h3>
    <ul className="list-disc ml-6 space-y-1">
      <li>สามารถปรับเปลี่ยน encoder หรือ decoder โดยไม่กระทบทั้งระบบ</li>
      <li>สนับสนุน transfer learning ระหว่าง task</li>
      <li>เพิ่มความยืดหยุ่นในการนำไปใช้งานในหลาย domain เช่น เสียง → ข้อความ, ข้อความ → ภาพ เป็นต้น</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500 text-black dark:text-blue-100">
      <strong>Highlight Box:</strong> จาก Encoder–Decoder ในรูปแบบ RNN และ LSTM → การประยุกต์ใช้ Attention → จนมาถึงสถาปัตยกรรมแบบ Transformer ที่แยก encoder และ decoder อย่างชัดเจน เช่นใน BART, T5 และ MT5
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างเปรียบเทียบ Input/Output ที่ต่างกัน</h3>
    <div className="overflow-x-auto">
      <table className="w-full border border-gray-300 dark:border-gray-700 text-sm text-left">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Task</th>
            <th className="border px-4 py-2">Input</th>
            <th className="border px-4 py-2">Output</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Translation</td>
            <td className="border px-4 py-2">English Sentence</td>
            <td className="border px-4 py-2">French Sentence</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Summarization</td>
            <td className="border px-4 py-2">Long Article</td>
            <td className="border px-4 py-2">Short Summary</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Image Captioning</td>
            <td className="border px-4 py-2">Image Features</td>
            <td className="border px-4 py-2">Sentence Description</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Sutskever et al. (2014). <em>Sequence to Sequence Learning with Neural Networks</em>. NeurIPS.</li>
      <li>Bahdanau et al. (2015). <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. ICLR.</li>
      <li>Stanford CS224n (2023). <em>Lecture 8: Sequence-to-Sequence Models</em>.</li>
      <li>MIT 6.S191 (2024). Deep Learning for Structured Data.</li>
      <li>Google Research Blog – The evolution of Google Translate architecture (2017).</li>
    </ul>
  </div>
</section>

                  {/* Section 2 */}
          <section id="architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
            <h2 className="text-2xl font-semibold mb-6 text-center">2. สถาปัตยกรรม Encoder–Decoder เบื้องต้น</h2>
            <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img3} />
            </div>
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
              <h3 className="text-xl font-semibold">ภาพรวมของโครงสร้าง Encoder–Decoder</h3>
              <p>
                โครงสร้าง Encoder–Decoder เป็นสถาปัตยกรรมพื้นฐานที่ใช้ในงาน sequence-to-sequence โดยเฉพาะในด้านการแปลภาษา การสรุปข้อความ และการสร้างคำตอบแบบอัตโนมัติ โครงสร้างนี้แยกโมดูลการเข้ารหัส (encoder) และการถอดรหัส (decoder) ออกจากกัน เพื่อให้สามารถแปลงข้อมูลจาก domain หนึ่งไปยังอีก domain ได้อย่างมีประสิทธิภาพ
              </p>
              <h3 className="text-xl font-semibold">การแบ่งหน้าที่ระหว่าง Encoder และ Decoder</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li><strong>Encoder:</strong> รับลำดับอินพุตและสร้าง representation หรือ context vector</li>
                <li><strong>Decoder:</strong> ใช้ context vector ดังกล่าวร่วมกับกลไก attention เพื่อสร้างลำดับคำตอบแบบต่อเนื่อง</li>
              </ul>
              <h3 className="text-xl font-semibold">โมเดลประเภทใดใช้โครงสร้างนี้</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li>Transformer</li>
                <li>Seq2Seq RNN with Attention</li>
                <li>Pointer-generator network</li>
                <li>T5, mT5, BART (Encoder–Decoder pretrained models)</li>
              </ul>
              <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500 text-black dark:text-blue-100">
                <p><strong>Highlight Box:</strong> โครงสร้าง Encoder–Decoder เป็นฐานของโมเดลภาษา (Language Models) ที่สามารถเข้าใจบริบทและสร้างข้อความได้ในหลากหลายภาษาหรือรูปแบบสื่อ เช่น ข้อความ ภาพ หรือโค้ด</p>
              </div>
              <h3 className="text-xl font-semibold">การไหลของข้อมูลในระบบ</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`Input Sequence → [Encoder] → Context Vector → [Decoder] → Output Sequence`}</code>
              </pre>
              <p>
                ในโครงสร้างพื้นฐานแบบนี้ ข้อมูลจาก Encoder จะไม่ถูกใช้โดยตรงทั้งหมด แต่จะถูกสรุปเป็นเวกเตอร์เดียว (หรือ matrix) เพื่อส่งให้ Decoder เรียนรู้ต่อไป ซึ่งในโมเดลสมัยใหม่มักใช้กลไก Self-Attention และ Cross-Attention เพื่อให้การส่งต่อข้อมูลนี้มีความยืดหยุ่นและละเอียดมากยิ่งขึ้น
              </p>
              <h3 className="text-xl font-semibold">ข้อดีของสถาปัตยกรรมแยกส่วน</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li>สามารถฝึก Encoder และ Decoder แยกกันหรือร่วมกันได้</li>
                <li>รองรับการเรียนรู้ที่ generalize ได้ดีขึ้นเมื่อมี pretraining</li>
                <li>สามารถใช้ encoder เดิมกับ decoder ใหม่ (หรือกลับกัน) ได้ในหลาย task</li>
              </ul>
              <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
                <p><strong>Insight Box:</strong> โมเดลประเภท T5 ใช้สถาปัตยกรรม Encoder–Decoder แบบสมบูรณ์ โดยแปลงทุก task ให้กลายเป็น text-to-text ทำให้สามารถใช้ชุดโมเดลเดียวกันกับหลายงานได้อย่างยืดหยุ่น</p>
              </div>
              <h3 className="text-xl font-semibold">เปรียบเทียบการใช้งาน: Classification vs Generation</h3>
              <table className="min-w-full border border-gray-300 dark:border-gray-700 text-sm">
                <thead className="bg-gray-100 dark:bg-gray-800">
                  <tr>
                    <th className="border px-4 py-2">ประเภท</th>
                    <th className="border px-4 py-2">Input</th>
                    <th className="border px-4 py-2">Output</th>
                    <th className="border px-4 py-2">โมเดลที่เหมาะสม</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border px-4 py-2">Classification</td>
                    <td className="border px-4 py-2">Text / Image</td>
                    <td className="border px-4 py-2">Label</td>
                    <td className="border px-4 py-2">Encoder-only</td>
                  </tr>
                  <tr>
                    <td className="border px-4 py-2">Text Generation</td>
                    <td className="border px-4 py-2">Text</td>
                    <td className="border px-4 py-2">Text</td>
                    <td className="border px-4 py-2">Encoder–Decoder</td>
                  </tr>
                </tbody>
              </table>
              <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
              <ul className="list-disc ml-6 text-sm space-y-2">
                <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
                <li>Google AI (2020). <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)</em>.</li>
                <li>Stanford CS224n – Lecture 12: Encoder–Decoder and Applications.</li>
                <li>MIT 6.S191 – Sequence Learning Architectures (2023).</li>
                <li>Oxford NLP – Sequence-to-Sequence Models and Attention</li>
              </ul>
            </div>
          </section>


           {/* Section 3 */}
          <section id="encoder" className="mb-16 scroll-mt-32 min-h-[400px]">
            <h2 className="text-2xl font-semibold mb-6 text-center">3. Encoder ทำหน้าที่อะไร?</h2>
            <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img4} />
            </div>

            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

              <h3 className="text-xl font-semibold">3.1 นิยามของ Encoder ในสถาปัตยกรรม Neural Network</h3>
              <p>
                ในโครงสร้างของ Encoder–Decoder ระบบ Encoder ทำหน้าที่สำคัญในการประมวลผลอินพุตดิบ (เช่น ข้อความ รูปภาพ หรือข้อมูลลำดับ) ให้กลายเป็นเวกเตอร์แสดงบริบท (context vector) ที่สรุปความหมายหลักของข้อมูลทั้งหมดเพื่อส่งต่อให้ Decoder ใช้ในการสร้างผลลัพธ์. แนวทางนี้เริ่มต้นในงาน Neural Machine Translation (NMT) และขยายสู่ทุกประเภทของ sequence-to-sequence modeling
              </p>

              <h3 className="text-xl font-semibold">3.2 โครงสร้างภายในของ Encoder</h3>
              <p>โดยทั่วไป Encoder จะประกอบด้วยหลายชั้น (layers) ของโมดูลต่อไปนี้:</p>
              <ul className="list-disc ml-6 space-y-2">
                <li><strong>Embedding Layer:</strong> แปลง token หรือ input ดิบให้เป็นเวกเตอร์</li>
                <li><strong>Positional Encoding:</strong> ใส่ข้อมูลตำแหน่งเพื่อให้เข้าใจลำดับ</li>
                <li><strong>Self-Attention Layers:</strong> เรียนรู้บริบทภายในลำดับของ input</li>
                <li><strong>Feedforward Layers:</strong> แปลงข้อมูลเพื่อเพิ่มความซับซ้อนในการเรียนรู้</li>
                <li><strong>Residual Connections + Layer Normalization:</strong> เสถียรภาพในการฝึก</li>
              </ul>

              <h3 className="text-xl font-semibold">3.3 การทำงานของ Self-Attention ใน Encoder</h3>
              <p>
                Self-Attention ช่วยให้ Encoder เข้าใจว่าคำในลำดับ input ใดมีความเกี่ยวข้องกับคำอื่น ๆ ในบริบทเดียวกัน โดยไม่จำเป็นต้องพึ่งลำดับเวลา ทำให้สามารถจับ long-range dependencies ได้ดี
              </p>
              <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500 text-black dark:text-blue-100">
                <h3 className="text-lg font-semibold mb-2">Highlight Box: Encoder ทำอะไร?</h3>
                <ul className="list-disc list-inside">
                  <li>สร้าง representation ที่สรุปเนื้อหาของข้อมูล input</li>
                  <li>แยกคำสำคัญออกมาให้ Decoder ใช้งานได้</li>
                  <li>เรียนรู้บริบทภายในอินพุตทั้งหมดพร้อมกัน</li>
                </ul>
              </div>

              <h3 className="text-xl font-semibold">3.4 ตารางเปรียบเทียบหน้าที่ของ Encoder</h3>
              <div className="overflow-x-auto">
                <table className="w-full border text-sm border-gray-300 dark:border-gray-700">
                  <thead className="bg-gray-100 dark:bg-gray-800">
                    <tr>
                      <th className="border px-4 py-2">องค์ประกอบ</th>
                      <th className="border px-4 py-2">หน้าที่หลัก</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2">Embedding</td>
                      <td className="border px-4 py-2">แปลง input ให้เป็นเวกเตอร์ที่เข้าใจได้</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Positional Encoding</td>
                      <td className="border px-4 py-2">ใส่ตำแหน่งเชิงลำดับให้กับ input</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Self-Attention</td>
                      <td className="border px-4 py-2">จับบริบทในลำดับเดียวกัน</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Feedforward</td>
                      <td className="border px-4 py-2">เพิ่มความสามารถเชิงนามธรรม</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-semibold">3.5 Encoder ในโมเดลต่าง ๆ</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li><strong>BERT:</strong> ใช้ Encoder อย่างเดียวแบบ bidirectional</li>
                <li><strong>T5:</strong> ใช้ Encoder แบบ full transformer สำหรับ pretraining</li>
                <li><strong>Encoder-Only Vision Models:</strong> เช่น MAE (Masked Autoencoder)</li>
              </ul>

              <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
                <h3 className="text-lg font-semibold mb-2">Insight Box: ทำไม Encoder ถึงสำคัญ?</h3>
                <p>
                  Encoder ไม่เพียงเป็นส่วนเริ่มต้นของสถาปัตยกรรม แต่ยังทำหน้าที่เป็น “ผู้สังเคราะห์ความหมาย” ของข้อมูล input ทั้งหมด หาก Encoder ไม่สามารถเข้าใจอินพุตได้อย่างถูกต้อง Decoder ก็จะไม่สามารถผลิต output ที่ถูกต้องได้เช่นกัน
                </p>
              </div>

              <h3 className="text-xl font-semibold">3.6 อ้างอิงทางวิชาการ</h3>
              <ul className="list-disc ml-6 space-y-2 text-sm">
                <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
                <li>Devlin et al. (2018). <em>BERT: Pre-training of Deep Bidirectional Transformers</em>. NAACL.</li>
                <li>Google Research (2021). <em>T5: Exploring the Limits of Transfer Learning</em>.</li>
                <li>MIT 6.S191 – Lecture: Deep Learning for NLP (2024)</li>
                <li>Stanford CS224n – Lecture 9: Transformer Encoder</li>
              </ul>

            </div>
          </section>


              {/* Section 4 */}
          <section id="decoder" className="mb-16 scroll-mt-32 min-h-[400px]">
            <h2 className="text-2xl font-semibold mb-6 text-center">4. Decoder ทำหน้าที่อะไร?</h2>
            <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img5} />
            </div>

            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
              <h3 className="text-xl font-semibold">หน้าที่หลักของ Decoder</h3>
              <p>Decoder เป็นส่วนสำคัญของสถาปัตยกรรม Encoder–Decoder ซึ่งทำหน้าที่แปลง representation ที่ได้จาก Encoder ไปเป็นผลลัพธ์สุดท้าย เช่น ลำดับคำที่แปลแล้ว หรือข้อความที่สร้างขึ้นใหม่ ในโมเดลภาษา เช่น GPT หรือ T5 ส่วน Decoder จะทำหน้าที่สร้าง token ทีละตัวในกระบวนการ auto-regressive โดยพิจารณาข้อมูลจาก Encoder และ token ก่อนหน้า</p>

              <h3 className="text-xl font-semibold">Masked Self-Attention ภายใน Decoder</h3>
              <p>Decoder ใช้กลไก Masked Self-Attention เพื่อป้องกันไม่ให้ตำแหน่งปัจจุบัน “เห็น” token ในอนาคต ซึ่งสำคัญมากในงานด้านการสร้างลำดับ เช่น text generation หรือการแปลภาษา</p>
              <ul className="list-disc ml-6 space-y-2">
                <li>Mask จะกำหนดให้ตำแหน่งในลำดับสามารถเข้าถึงได้เฉพาะ token ก่อนหน้าและปัจจุบันเท่านั้น</li>
                <li>ช่วยให้โมเดลสามารถเรียนรู้การพยากรณ์ลำดับแบบเป็นขั้นตอน</li>
              </ul>
              <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
                <p><strong>Insight Box:</strong> กลไก Masked Self-Attention ใน Decoder ช่วยสร้างความเป็นลำดับของภาษาที่สอดคล้องกับรูปแบบการใช้จริง เช่น ในงานแปลหรือสรุปข้อความ</p>
              </div>

              <h3 className="text-xl font-semibold">Cross-Attention กับ Encoder</h3>
              <p>Decoder ไม่ได้ใช้เพียงข้อมูลจากตัวเองเท่านั้น แต่ยังมีการใช้ Cross-Attention เพื่อเชื่อมต่อกับผลลัพธ์ของ Encoder โดยให้ Decoder สามารถ “โฟกัส” กับตำแหน่งสำคัญในอินพุตต้นฉบับได้อย่างแม่นยำ</p>
              <ul className="list-disc ml-6 space-y-2">
                <li>Query มาจากตัว Decoder</li>
                <li>Key และ Value มาจาก Encoder</li>
                <li>กลไกนี้ช่วยให้ Decoder เลือกข้อมูลที่เกี่ยวข้องจาก Encoder ได้โดยตรง</li>
              </ul>

              <h3 className="text-xl font-semibold">การใช้ Linear Layer และ Softmax</h3>
              <p>หลังจากที่ Decoder ประมวลผลบริบทผ่าน Self-Attention และ Cross-Attention แล้ว ข้อมูลจะถูกส่งผ่าน linear layer และ softmax เพื่อคำนวณความน่าจะเป็นของ token ถัดไป ซึ่งเป็นขั้นตอนสุดท้ายในการพยากรณ์ผลลัพธ์</p>
              <ul className="list-disc ml-6 space-y-2">
                <li>Linear projection แปลงเวกเตอร์ hidden state ไปยังขนาดเท่ากับ vocabulary</li>
                <li>Softmax แปลงค่าดังกล่าวให้เป็น distribution สำหรับการสุ่มหรือเลือก token ถัดไป</li>
              </ul>

              <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500 text-black dark:text-blue-100">
                <p><strong>Highlight:</strong> ในโมเดล GPT การใช้ Decoder เพียงอย่างเดียว (decoder-only) โดยไม่มี encoder ก็สามารถเรียนรู้บริบทได้ลึกผ่าน masking และ self-supervised learning</p>
              </div>

              <h3 className="text-xl font-semibold">สถาปัตยกรรมของ Decoder (แบบละเอียด)</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full border border-gray-300 dark:border-gray-700 text-sm">
                  <thead className="bg-gray-100 dark:bg-gray-800">
                    <tr>
                      <th className="border px-4 py-2">ลำดับชั้น</th>
                      <th className="border px-4 py-2">หน้าที่</th>
                      <th className="border px-4 py-2">หมายเหตุ</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2">Masked Self-Attention</td>
                      <td className="border px-4 py-2">จับบริบทภายในลำดับ</td>
                      <td className="border px-4 py-2">ไม่เห็น token ในอนาคต</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Cross-Attention</td>
                      <td className="border px-4 py-2">เชื่อมโยงกับ output ของ Encoder</td>
                      <td className="border px-4 py-2">ใช้ Q จาก Decoder, K/V จาก Encoder</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Feedforward Network</td>
                      <td className="border px-4 py-2">แปลง feature ให้ซับซ้อนขึ้น</td>
                      <td className="border px-4 py-2">ใช้ ReLU หรือ GELU</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2">Linear + Softmax</td>
                      <td className="border px-4 py-2">แปลงเป็น token output</td>
                      <td className="border px-4 py-2">ใช้สำหรับ prediction</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-semibold">การประยุกต์ใช้ Decoder</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li>Text Generation: สร้างคำตอบที่มีบริบทต่อเนื่อง</li>
                <li>Machine Translation: ถอดรหัสจาก representation ของ Encoder</li>
                <li>Summarization: เรียบเรียงเนื้อหาจากข้อมูลต้นฉบับใหม่</li>
              </ul>

              <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
              <ul className="list-disc ml-6 text-sm space-y-2">
                <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
                <li>Stanford CS224n (2023) – Lecture on Decoder Architecture</li>
                <li>MIT 6.S191 (2024) – Deep Learning for NLP</li>
                <li>Oxford Transformers Course – Sequence Decoding in Practice</li>
              </ul>
            </div>
          </section>


         {/* Section 5 */}
      <section id="cross-attention" className="mb-16 scroll-mt-32 min-h-[400px]">
        <h2 className="text-2xl font-semibold mb-6 text-center">5. Cross-Attention คืออะไร?</h2>
        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img6} />
        </div>

        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
          <h3 className="text-xl font-semibold">นิยามของ Cross-Attention</h3>
          <p>
            Cross-Attention เป็นกลไกที่สำคัญภายในโครงสร้าง Encoder-Decoder โดยเฉพาะในโมเดล Transformer ซึ่งใช้เพื่อจับคู่บริบทระหว่างข้อมูลจาก Encoder กับ Decoder กล่าวคือ ตัว Decoder จะโฟกัสกับข้อมูลที่ได้จาก Encoder เพื่อนำมาสร้างลำดับผลลัพธ์ที่มีความสอดคล้องกับข้อมูลอินพุต
          </p>

          <h3 className="text-xl font-semibold">หลักการทำงานของ Cross-Attention</h3>
          <p>
            ใน Cross-Attention เวกเตอร์ Query (Q) มาจาก Decoder ส่วนเวกเตอร์ Key (K) และ Value (V) มาจาก Encoder โดยที่ Decoder จะใช้ Q เหล่านี้เพื่อค้นหาว่าในข้อมูลจาก Encoder มีบริบทใดบ้างที่ควรให้ความสำคัญเพื่อช่วยในการคาดการณ์คำถัดไป
          </p>

          <pre className="bg-gray-100 dark:bg-gray-800 text-sm p-4 rounded overflow-x-auto">
<code>{`CrossAttention(Q_decoder, K_encoder, V_encoder) = softmax(QKᵀ / √d_k) V`}</code>
          </pre>

          <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
            <p><strong>Insight Box:</strong> Cross-Attention ทำหน้าที่เชื่อมโยงการเรียนรู้ระหว่างข้อมูลต้นทางกับข้อมูลที่กำลังสร้าง ทำให้ระบบสามารถอ้างอิงความหมายเดิมได้แม้ในการสร้างประโยคใหม่</p>
          </div>

          <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Self-Attention กับ Cross-Attention</h3>
          <div className="overflow-x-auto">
            <table className="w-full border border-gray-300 dark:border-gray-700 text-sm text-left">
              <thead className="bg-gray-100 dark:bg-gray-800">
                <tr>
                  <th className="border px-4 py-2">ลักษณะ</th>
                  <th className="border px-4 py-2">Self-Attention</th>
                  <th className="border px-4 py-2">Cross-Attention</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border px-4 py-2">Input</td>
                  <td className="border px-4 py-2">Q, K, V มาจาก sequence เดียวกัน</td>
                  <td className="border px-4 py-2">Q มาจาก Decoder, K และ V จาก Encoder</td>
                </tr>
                <tr>
                  <td className="border px-4 py-2">การเชื่อมโยงบริบท</td>
                  <td className="border px-4 py-2">ภายในลำดับเดียวกัน</td>
                  <td className="border px-4 py-2">ระหว่างลำดับต่างกัน (input/output)</td>
                </tr>
                <tr>
                  <td className="border px-4 py-2">การใช้งานหลัก</td>
                  <td className="border px-4 py-2">เข้าใจลำดับปัจจุบัน</td>
                  <td className="border px-4 py-2">อ้างอิงข้อมูลจากต้นฉบับเพื่อสร้าง output</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold">การประยุกต์ใช้ Cross-Attention ในโมเดล</h3>
          <ul className="list-disc ml-6 space-y-2">
            <li>ใช้ใน Decoder ของ Transformer สำหรับ Machine Translation เช่น EN → DE</li>
            <li>ใช้ในระบบ Multimodal ที่ต้องจับความสัมพันธ์ระหว่างภาพและข้อความ เช่น Flamingo</li>
            <li>ใช้ใน Retrieval-Augmented Generation (RAG) เพื่อโฟกัสกับ context จาก retrieval</li>
          </ul>

          <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
            <p><strong>Highlight Box:</strong> Cross-Attention ช่วยให้ Decoder ไม่ได้ "เดา" จากลำดับก่อนหน้าเพียงอย่างเดียว แต่มีข้อมูลจริงจาก Encoder ที่ใช้ร่วมในการพิจารณาทุกคำ</p>
          </div>

          <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
          <ul className="list-disc ml-6 text-sm space-y-2">
            <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
            <li>Stanford CS224n – Lecture 12: Cross-Attention and Applications</li>
            <li>MIT 6.S191 (2024). Deep Learning for Structured Sequences</li>
            <li>Google Research – Applications of Cross-Attention in PaLM and Gemini</li>
            <li>CMU – Sequence-to-Sequence Learning: Encoder-Decoder with Attention</li>
          </ul>
        </div>
      </section>


{/* Section 6 */}
<section id="diagram" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    6. Visual Diagram: Encoder–Decoder (Transformer Style)
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">ภาพรวมของโครงสร้าง Encoder–Decoder</h3>
    <p>
      โครงสร้างของโมเดล Encoder–Decoder ที่ใช้ Attention เป็นกลไกหลักได้รับความนิยมสูงสุดในช่วงทศวรรษที่ผ่านมา โดยเฉพาะโมเดลสมัยใหม่ เช่น Transformer และ T5 ที่นำเสนอวิธีใหม่ในการเข้ารหัสและถอดรหัสข้อมูลลำดับ ซึ่งถูกนำไปใช้ทั้งใน NLP และ Vision
    </p>

    <h3 className="text-xl font-semibold">ส่วนประกอบที่สำคัญในภาพ</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li><strong>Encoder Block:</strong> รับลำดับ input และสร้าง contextual embedding</li>
      <li><strong>Decoder Block:</strong> รับ target sequence และนำ encoder output มาร่วมพิจารณาผ่าน Cross-Attention</li>
      <li><strong>Multi-Head Attention:</strong> ใช้จับความสัมพันธ์แบบหลายมุมมองระหว่าง tokens</li>
      <li><strong>Feedforward Layers:</strong> ใช้แปลงข้อมูลเชิงลึกหลังจาก Attention</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
      <p className="font-semibold">
        Highlight Box: ภาพ diagram แบบ Transformer ได้รับการใช้อ้างอิงในงานวิจัยจำนวนมาก เนื่องจากอธิบายกลไกการเข้ารหัสและถอดรหัสแบบ Attention-centric ได้อย่างชัดเจน
      </p>
    </div>

    <h3 className="text-xl font-semibold">ทิศทางของการไหลข้อมูล</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>Input → Positional Encoding → Encoder → Encoder Output</li>
      <li>Target → Positional Encoding → Decoder</li>
      <li>Decoder มี Masked Self-Attention ตามด้วย Cross-Attention จาก Encoder</li>
      <li>ผลลัพธ์สุดท้ายถูกแปลงผ่าน Linear + Softmax</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p>
        <strong>Insight:</strong> โครงสร้าง Encoder–Decoder แบบ Transformer ทำให้เกิด paradigm shift ด้านการประมวลผลลำดับข้อมูลจากแบบ sequential → เป็นแบบ parallel ซึ่งช่วยเพิ่มทั้งประสิทธิภาพและคุณภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>Stanford CS224n – Lecture: Transformers Architecture</li>
      <li>MIT 6.S191 – Deep Learning for Sequences</li>
      <li>Harvard NLP Annotated Transformer</li>
    </ul>
  </div>
</section>


 {/* Section 7 */}
<section id="rnn-vs-transformer" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    7. ความแตกต่างระหว่าง RNN-Based vs Transformer-Based Encoder–Decoder
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
    <h3 className="text-xl font-semibold">ภาพรวมของสองสถาปัตยกรรม</h3>
    <p>
      ทั้ง RNN-based และ Transformer-based Encoder–Decoder ถูกใช้ในงานแปลภาษา สรุปข้อความ และการสร้างภาษา แต่มีพื้นฐานการประมวลผลที่แตกต่างกันอย่างสิ้นเชิง ซึ่งส่งผลต่อความเร็ว ความสามารถในการเรียนรู้บริบท และการสเกลระบบ
    </p>

    <h3 className="text-xl font-semibold">RNN-Based Encoder–Decoder</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้การประมวลผลข้อมูลแบบลำดับ (sequential)</li>
      <li>มี hidden state ที่ถ่ายทอดจาก token หนึ่งไปยังอีก token</li>
      <li>ขึ้นอยู่กับการเรียนรู้ long-term dependencies ผ่านโครงสร้างเวลา</li>
      <li>ต้องใช้กลไก attention แบบ Bahdanau หรือ Luong เพื่อแก้ปัญหาการลืมข้อมูลต้นลำดับ</li>
    </ul>

    <h3 className="text-xl font-semibold">Transformer-Based Encoder–Decoder</h3>
    <ul className="list-disc ml-6 space-y-2">
      <li>ใช้ attention แทน recurrence จึงสามารถประมวลผลข้อมูลแบบขนาน</li>
      <li>สามารถเข้าถึงทุก token ใน sequence พร้อมกันผ่าน self-attention</li>
      <li>เรียนรู้ความสัมพันธ์ระยะไกลได้อย่างมั่นคงโดยไม่ต้องใช้ gating</li>
      <li>เหมาะกับ training บน GPU/TPU และสเกลขึ้นเป็น LLM ได้ง่าย</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
      <p>
        <strong>Highlight:</strong> RNN-based โมเดลเหมาะกับงานที่มีลำดับแน่นอนและ resource จำกัด ส่วน Transformer-based เหมาะกับงานที่ต้องการ parallelism และ context การเรียนรู้ที่ยาว เช่น GPT และ T5
      </p>
    </div>

    <h3 className="text-xl font-semibold">การเปรียบเทียบเชิงโครงสร้าง</h3>
    <div className="overflow-x-auto">
      <table className="min-w-[640px] border text-sm text-left border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">มิติเปรียบเทียบ</th>
            <th className="border px-4 py-2">RNN-Based</th>
            <th className="border px-4 py-2">Transformer-Based</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">การประมวลผล</td>
            <td className="border px-4 py-2">แบบลำดับ (Sequential)</td>
            <td className="border px-4 py-2">แบบขนาน (Parallel)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความสามารถในการเรียนรู้ระยะไกล</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">ดีมาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">สเกลได้ง่ายหรือไม่</td>
            <td className="border px-4 py-2">ยาก</td>
            <td className="border px-4 py-2">ง่าย (เหมาะกับ LLM)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ความเร็วการฝึก</td>
            <td className="border px-4 py-2">ช้า</td>
            <td className="border px-4 py-2">เร็ว</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p>
        <strong>Insight:</strong> จากงานวิจัยของ Stanford และ MIT พบว่า Transformer สามารถแทนที่ RNN ได้เกือบทุกด้านใน NLP และนำไปขยายผลใน Vision, Audio และ Multimodal ได้อย่างมีประสิทธิภาพมากกว่า
      </p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
      <li>Cho et al. (2014). <em>Learning Phrase Representations using RNN Encoder–Decoder</em>. arXiv:1406.1078</li>
      <li>MIT 6.S191 (2023) – Lecture on Sequence Modeling</li>
      <li>Stanford CS224n – Lecture: Encoder–Decoder Architectures</li>
    </ul>
  </div>
</section>

         {/* Section 8 */}
  <section id="applications" className="mb-16 scroll-mt-32 min-h-[400px]">
    <h2 className="text-2xl font-semibold mb-6 text-center">8. Applications จริงของ Encoder–Decoder</h2>
    <div className="flex justify-center my-6">
      <AdvancedImage cldImg={img9} />
    </div>

    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
      <h3 className="text-xl font-semibold">การประยุกต์ใช้งานในโลกจริง</h3>
      <p>
        Encoder–Decoder architectures ได้กลายเป็นแกนหลักในระบบ AI ที่ใช้ประมวลผลข้อมูลลำดับ เช่น ภาษา ภาพ และเสียง โดยโมเดลเหล่านี้สามารถเรียนรู้การ mapping จาก sequence หนึ่งไปยังอีก sequence หนึ่งที่มีความสัมพันธ์เชิงโครงสร้าง ซึ่งส่งผลให้เกิดการนำไปใช้ในหลายอุตสาหกรรมอย่างแพร่หลาย
      </p> 

      <h3 className="text-xl font-semibold">งานด้านภาษาธรรมชาติ (NLP)</h3>
      <ul className="list-disc ml-6 space-y-2">
        <li><strong>Neural Machine Translation:</strong> เช่น Google Translate ที่ใช้ Transformer-based encoder–decoder ในการแปลภาษาแบบคู่ขนาน</li>
        <li><strong>Text Summarization:</strong> เช่น BART และ T5 ใช้ encoder อ่านเอกสาร และ decoder สร้างบทสรุปที่กระชับ</li>
        <li><strong>Question Answering (QA):</strong> ใช้ encoder อ่าน context และ decoder สร้างคำตอบตามบริบท</li>
      </ul>

      <h3 className="text-xl font-semibold">งานด้านเสียงและภาพ</h3>
      <ul className="list-disc ml-6 space-y-2">
        <li><strong>Speech Recognition:</strong> เช่น Whisper และ DeepSpeech ใช้ encoder–decoder เพื่อแปลงเสียงเป็นข้อความ</li>
        <li><strong>Image Captioning:</strong> ใช้ encoder จาก CNN หรือ Vision Transformer เข้ารหัสภาพ และ decoder สร้างคำอธิบาย</li>
        <li><strong>Video-to-Text:</strong> ใช้ encoder วิเคราะห์เฟรมภาพและ decoder แปลงเป็นคำอธิบายเชิงลำดับ</li>
      </ul>

      <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
        <p>
          <strong>Highlight:</strong> ในระบบ multimodal เช่น Flamingo หรือ GPT-4-Vision มีการใช้ encoder–decoder เพื่อจัดการกับข้อมูลภาพและข้อความพร้อมกัน โดย decoder สามารถตอบคำถามที่เกี่ยวข้องกับภาพได้อย่างแม่นยำ
        </p>
      </div>

      <h3 className="text-xl font-semibold">ระบบสนับสนุนการตัดสินใจและ Agent</h3>
      <ul className="list-disc ml-6 space-y-2">
        <li>ระบบ Copilot และ AI Agent ต่าง ๆ ใช้ encoder–decoder เป็นแกนหลักในการอ่านคำสั่งและสร้างคำตอบ</li>
        <li>ในระบบอัตโนมัติ เช่น n8n หรือ LangChain decoder สามารถสร้าง action chain จากคำอธิบายธรรมชาติ</li>
        <li>ใช้ใน RAG (Retrieval-Augmented Generation) เพื่อรวมข้อมูลจากแหล่งต่าง ๆ แล้วสร้าง output แบบมีบริบท</li>
      </ul>

      <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
        <p>
          <strong>Insight:</strong> โมเดล encoder–decoder ไม่ได้ถูกจำกัดอยู่ในงานแปลภาษาอีกต่อไป แต่ได้กลายเป็น core architecture ของ AI ที่มีความสามารถสูงสุดในหลายสายงาน ตั้งแต่ Chatbot, Content Generation, ไปจนถึง Bioinformatics
        </p>
      </div>

      <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
      <ul className="list-disc ml-6 text-sm space-y-2">
        <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
        <li>Lewis et al. (2020). <em>BART: Denoising Sequence-to-Sequence Pre-training</em>. ACL.</li>
        <li>Radford et al. (2023). <em>GPT-4 Technical Report</em>. OpenAI.</li>
        <li>Brown et al. (2020). <em>Language Models are Few-Shot Learners (GPT-3)</em>. NeurIPS.</li>
        <li>MIT 6.S191 – Sequence Learning Applications (2024)</li>
      </ul>
    </div>
  </section>


          {/* Section 9 */}
          <section id="models" className="mb-16 scroll-mt-32 min-h-[400px]">
            <h2 className="text-2xl font-semibold mb-6 text-center">9. ตัวอย่างโมเดลที่ใช้ Encoder–Decoder</h2>
            <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img10} />
            </div>

            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
              <h3 className="text-xl font-semibold">ภาพรวมของโมเดลในโลกจริง</h3>
              <p>
                สถาปัตยกรรม Encoder–Decoder เป็นโครงสร้างพื้นฐานของโมเดลปัญญาประดิษฐ์ที่ประสบความสำเร็จในหลากหลายโดเมน ตั้งแต่งานประมวลผลภาษาธรรมชาติไปจนถึงการแปลภาพหรือเสียง ตัวอย่างโมเดลที่มีชื่อเสียง ได้แก่ Transformer, BERT2BERT, T5, และระบบแปลภาษาของ Google Neural Machine Translation (GNMT)
              </p>

              <h3 className="text-xl font-semibold">Transformer (Vaswani et al., 2017)</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li>เป็นโมเดลแรกที่ใช้ Self-Attention และ Encoder–Decoder เต็มรูปแบบโดยไม่มี RNN</li>
                <li>ใช้ในงานแปลภาษาเดิม และภายหลังกลายเป็นรากฐานของ LLM ทั้งหมด</li>
                <li>Encoder ทำหน้าที่เข้าใจประโยคต้นฉบับ → Decoder ใช้เพื่อสร้างประโยคปลายทาง</li>
              </ul>

              <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl text-black dark:text-blue-100 border-l-4 border-blue-500">
                <strong>Highlight:</strong> Transformer เปิดทางให้กับการเรียนรู้แบบขนานเต็มรูปแบบ โดยลดข้อจำกัดของ RNN ในการเรียนรู้ลำดับยาว
              </div>

              <h3 className="text-xl font-semibold">T5 (Text-to-Text Transfer Transformer)</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li>พัฒนาโดย Google Research โดยมองทุกงาน NLP เป็นปัญหา “text → text”</li>
                <li>ใช้ Encoder–Decoder เพื่อเรียนรู้การแปลงจาก input text ไปยัง output text โดยตรง</li>
                <li>รองรับ task ได้หลากหลาย เช่น summarization, translation, classification</li>
              </ul>

              <h3 className="text-xl font-semibold">BART และ BERT2BERT</h3>
              <p>
                BART ผสาน BERT เป็น encoder และ GPT เป็น decoder โดยมีการ pretrain แบบ denoising autoencoder ส่วน BERT2BERT เป็นการนำ BERT มาใช้งานทั้งฝั่ง encoder และ decoder เหมาะสำหรับงานที่ต้องการ contextual encoding ลึกทั้งสองฝั่ง
              </p>

           <div className="overflow-x-auto">
  <table className="min-w-[640px] text-sm border border-gray-300 dark:border-gray-700">
    <thead className="bg-gray-100 dark:bg-gray-800">
      <tr>
        <th className="border px-4 py-2">Model</th>
        <th className="border px-4 py-2">Encoder</th>
        <th className="border px-4 py-2">Decoder</th>
        <th className="border px-4 py-2">Use Case</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Transformer</td>
        <td className="border px-4 py-2">Self-Attention</td>
        <td className="border px-4 py-2">Masked Attention + Cross</td>
        <td className="border px-4 py-2">Translation</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">T5</td>
        <td className="border px-4 py-2">Transformer</td>
        <td className="border px-4 py-2">Transformer</td>
        <td className="border px-4 py-2">Text-to-Text Tasks</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">BART</td>
        <td className="border px-4 py-2">BERT-style</td>
        <td className="border px-4 py-2">GPT-style</td>
        <td className="border px-4 py-2">Summarization, Generation</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">BERT2BERT</td>
        <td className="border px-4 py-2">BERT</td>
        <td className="border px-4 py-2">BERT</td>
        <td className="border px-4 py-2">QA, Sentence Fusion</td>
      </tr>
    </tbody>
  </table>
</div>


              <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
                <p><strong>Insight Box:</strong> สถาปัตยกรรม Encoder–Decoder ช่วยให้สามารถแยก “การเข้าใจ” และ “การสร้าง” ข้อมูลออกจากกัน ทำให้ระบบสามารถปรับปรุงได้อย่างเป็นอิสระในแต่ละส่วน และเพิ่ม flexibility ในการออกแบบระบบ AI ขนาดใหญ่</p>
              </div>

              <h3 className="text-xl font-semibold">ระบบแปลภาษาของ Google (GNMT)</h3>
              <p>
                Google Neural Machine Translation (GNMT) เป็นหนึ่งในระบบแรกที่ใช้ RNN-based Encoder–Decoder อย่างเต็มรูปแบบ ก่อนการเกิดของ Transformer โดยใช้ LSTM encoder และ decoder ร่วมกับ attention mechanism ช่วยให้ Google Translate มีความแม่นยำสูงขึ้นอย่างมีนัยสำคัญ
              </p>

              <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
              <ul className="list-disc ml-6 text-sm space-y-2">
                <li>Vaswani et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
                <li>Raffel et al. (2019). <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)</em>. arXiv.</li>
                <li>Lewis et al. (2020). <em>BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation</em>. ACL.</li>
                <li>Wu et al. (2016). <em>Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation</em>. arXiv.</li>
                <li>Stanford CS224n (2023). <em>Lecture 10–12: Transformer Architectures & Applications</em>.</li>
              </ul>
            </div>
          </section>

          {/* Section 10 */}
          <section id="challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
            <h2 className="text-2xl font-semibold mb-6 text-center">10. ความท้าทายและจุดสำคัญ</h2>
            <div className="flex justify-center my-6">
              <AdvancedImage cldImg={img11} />
            </div>

            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
              <h3 className="text-xl font-semibold">ข้อจำกัดด้านโครงสร้างและความสามารถ</h3>
              <p>
                แม้ว่า Encoder–Decoder จะเป็นสถาปัตยกรรมหลักของหลายโมเดลในยุคใหม่ แต่ก็มีข้อจำกัดหลายด้านที่นักวิจัยจาก Stanford, MIT และ CMU วิเคราะห์ไว้ เช่น ความซับซ้อนเชิงคำนวณ (computational complexity), ข้อจำกัดในการเรียนรู้ระยะไกล (long-range dependencies) ใน input sequence, และความไม่แน่นอนระหว่างขั้นตอน decoding
              </p>

              <h3 className="text-xl font-semibold">การจัดการกับ Sequence ที่ยาวมาก</h3>
              <p>
                หนึ่งในความท้าทายสำคัญคือการจัดการกับลำดับข้อมูลที่ยาว เช่น เอกสารขนาดใหญ่ หรือการสนทนาต่อเนื่อง ซึ่งการใช้ Self-Attention ใน Encoder และ Cross-Attention ใน Decoder จะต้องใช้ memory และเวลาในการประมวลผลแบบ O(n²) ส่งผลต่อ scalability โดยตรง
              </p>
              <ul className="list-disc ml-6 space-y-2">
                <li>โมเดลเช่น Longformer และ Reformer ถูกพัฒนาขึ้นเพื่อรองรับลำดับที่ยาว</li>
                <li>การใช้ sparse attention หรือ linear attention ช่วยลด resource ลงได้อย่างมีนัยสำคัญ</li>
              </ul>

              <h3 className="text-xl font-semibold">ความไม่แน่นอนของการทำ Decoding</h3>
              <p>
                Decoding ใน Encoder–Decoder model มีลักษณะเป็น autoregressive ซึ่งแต่ละ token ที่ถูกสร้างขึ้นจะมีผลกับ token ถัดไป → นำไปสู่การสะสมของความผิดพลาด (error accumulation)
              </p>
              <ul className="list-disc ml-6 space-y-2">
                <li>ในงานด้าน machine translation หาก token แรกผิด การแปลคำต่อ ๆ ไปมักเบี่ยงเบน</li>
                <li>เทคนิค Beam Search, Top-k sampling และ Temperature control ถูกพัฒนาเพื่อลดความไม่แน่นอนนี้</li>
              </ul>

              <h3 className="text-xl font-semibold">การเรียนรู้ข้ามโดเมน (Cross-Domain Adaptation)</h3>
              <p>
                การนำโมเดล Encoder–Decoder ที่ฝึกด้วย domain หนึ่งไปใช้ในอีก domain หนึ่ง (เช่น ฝึกจากข่าว → ใช้ในบทสนทนา) มักเจอปัญหา domain shift ซึ่งส่งผลต่อความแม่นยำของ prediction
              </p>
              <p>
                เทคนิค เช่น transfer learning, domain adaptation layer, หรือการ fine-tune เฉพาะ decoder ถูกใช้เพื่อแก้ไขปัญหานี้
              </p>

              <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
                <p className="font-semibold">
                  Highlight Box: งานวิจัยจาก Harvard NLP และ DeepMind แสดงว่า ความไม่สอดคล้องระหว่าง distribution ของ training input กับ inference input คือสาเหตุหลักของความล้มเหลวใน downstream task ของ Encoder–Decoder
                </p>
              </div>

              <h3 className="text-xl font-semibold">ความยากในการ Interpret</h3>
              <p>
                แม้ Attention mechanism จะช่วยให้เข้าใจว่าโมเดล “โฟกัส” กับอะไรระหว่างการประมวลผล แต่โดยรวมแล้ว Encoder–Decoder ยังคงยากต่อการอธิบายการตัดสินใจเชิง causal
              </p>
              <p>
                จึงมีงานพัฒนาเทคนิค interpretability เช่น Attention Rollout, Layer-wise Relevance Propagation (LRP), และ probing hidden states เพื่อแยกแยะว่าขั้นตอนใดใน Encoder/Decoder มีผลต่อผลลัพธ์มากที่สุด
              </p>

              <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
                <p><strong>Insight:</strong> Encoder–Decoder ยังเป็น “black box” ที่ซับซ้อน แม้จะมี interpretability tools จำนวนมาก แต่ก็ยังไม่สามารถอธิบายพฤติกรรมบางอย่างได้ทั้งหมดในระดับที่ human-comprehensible
                </p>
              </div>

              <h3 className="text-xl font-semibold">ข้อเสนอเพื่อพัฒนาในอนาคต</h3>
              <ul className="list-disc ml-6 space-y-2">
                <li>ปรับ attention ให้มีความเป็น modular และ compositional มากขึ้น</li>
                <li>ผสาน hybrid encoder–decoder กับ external memory หรือ retrieval augmentation</li>
                <li>ฝึกด้วย objective function ที่สะท้อน context dependency จริง ๆ ของงาน</li>
              </ul>

              <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
              <ul className="list-disc ml-6 text-sm space-y-2">
                <li>Bahdanau et al. (2015). <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. ICLR.</li>
                <li>Raffel et al. (2020). <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</em>. JMLR.</li>
                <li>Lewis et al. (2020). <em>BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation</em>. ACL.</li>
                <li>Vig et al. (2019). <em>Visualizing Attention in Transformers</em>. arXiv:1904.02668</li>
                <li>Stanford CS224n – Lecture 12: Challenges in Neural Architectures</li>
              </ul>

            </div>
          </section>


         {/* Section 11 */}
      <section id="research" className="mb-16 scroll-mt-32 min-h-[400px]">
        <h2 className="text-2xl font-semibold mb-6 text-center">11. Research Highlights</h2>
        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img12} />
        </div>

        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">
          <h3 className="text-xl font-semibold">วิวัฒนาการของงานวิจัย Encoder–Decoder</h3>
          <p>
            โครงสร้าง Encoder–Decoder ถือเป็นหนึ่งในนวัตกรรมที่เปลี่ยนแปลงวงการ Deep Learning โดยเฉพาะใน Natural Language Processing (NLP) อย่างลึกซึ้ง งานวิจัยดั้งเดิมอย่าง Seq2Seq จาก Google (Sutskever et al., 2014) ได้วางรากฐานให้โมเดลสามารถประมวลผลลำดับข้อมูลในรูปแบบ input → output ที่มีโครงสร้างต่างกัน เช่น แปลภาษา สรุปความ หรือแปลงคำพูดเป็นข้อความ
          </p>

          <h3 className="text-xl font-semibold">จุดเปลี่ยนสำคัญในการออกแบบ Encoder–Decoder</h3>
          <ul className="list-disc ml-6 space-y-2">
            <li><strong>2014 – Sequence-to-Sequence (Seq2Seq):</strong> งานของ Sutskever et al. แสดงให้เห็นว่า RNN encoder-decoder สามารถแปลภาษาด้วยการ encode ความหมายก่อน decode ออกมา</li>
            <li><strong>2015 – Attention Mechanism:</strong> Bahdanau et al. เพิ่ม attention ให้ decoder สามารถโฟกัสส่วนต่าง ๆ ของ input sequence ได้ ทำให้ translation แม่นยำขึ้น</li>
            <li><strong>2017 – Transformer:</strong> Vaswani et al. เปิดตัวโมเดลที่ตัด RNN ออกโดยสิ้นเชิง ใช้ Self-Attention และ Cross-Attention เป็นแกนหลัก ทำให้สามารถ parallelize ได้อย่างมีประสิทธิภาพ</li>
            <li><strong>2018–2020 – Pretraining:</strong> โมเดลเช่น BERT, GPT, และ T5 นำ Encoder–Decoder ไปประยุกต์ใช้ในสเกลขนาดใหญ่ที่ไม่ต้องการ labeled data โดยตรง</li>
            <li><strong>2020+ – Multimodal Encoder–Decoder:</strong> งานวิจัยเช่น Flamingo, Gato, และ PaLM-E รวมข้อมูลจากหลาย modality เช่น ข้อความ+ภาพ+เสียง</li>
          </ul>

          <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-500">
            <p>
              <strong>Highlight Box:</strong> จุดเปลี่ยนที่สำคัญที่สุดในวิวัฒนาการของ Encoder–Decoder คือการรวม Attention เข้ากับระบบลำดับ → ทำให้โมเดลสามารถเข้าใจความเชื่อมโยงระยะไกลได้ดีกว่าเดิมมาก ทั้งในข้อมูลภาษาและภาพ
            </p>
          </div>

          <h3 className="text-xl font-semibold">แนวโน้มของงานวิจัยในปี 2020 ขึ้นไป</h3>
          <ul className="list-disc ml-6 space-y-2">
            <li><strong>Efficient Attention:</strong> งานวิจัยจาก Google และ Stanford เน้นลด memory footprint ใน cross-attention layer เพื่อรองรับ long-sequence</li>
            <li><strong>Cross-Modal Transfer:</strong> MIT และ DeepMind พัฒนาโมเดลที่ใช้ Encoder เป็น text และ Decoder เป็น image (หรือกลับกัน) เพื่อการเรียนรู้ที่ไม่จำกัด modal</li>
            <li><strong>Memory-Augmented Decoding:</strong> Harvard และ CMU ศึกษาการใช้ external memory ร่วมกับ decoder ในการสรุปเนื้อหายาว ๆ หรือจัดการ reasoning</li>
            <li><strong>Unified Models:</strong> เช่น Gato หรือ Perceiver IO ที่ออกแบบ encoder–decoder ให้ทำได้ทุก task ทั้ง language, vision, action</li>
          </ul>

          <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-500 text-black dark:text-yellow-100">
            <p>
              <strong>Insight Box:</strong> Encoder–Decoder ไม่ใช่แค่สถาปัตยกรรมแปลภาษาอีกต่อไป แต่กลายเป็นรากฐานของโมเดลที่สามารถเข้าใจข้อมูลหลากหลายและสื่อสารผ่านหลาย modality ในระบบเดียว
            </p>
          </div>

          <h3 className="text-xl font-semibold">รายการอ้างอิง</h3>
          <ul className="list-disc ml-6 space-y-2 text-sm">
            <li>Sutskever, I., Vinyals, O., Le, Q.V. (2014). <em>Sequence to Sequence Learning with Neural Networks</em>. NeurIPS.</li>
            <li>Bahdanau, D. et al. (2015). <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. ICLR.</li>
            <li>Vaswani, A. et al. (2017). <em>Attention is All You Need</em>. NeurIPS.</li>
            <li>Raffel, C. et al. (2020). <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)</em>. JMLR.</li>
            <li>Jaegle, A. et al. (2021). <em>Perceiver IO: A General Architecture for Structured Inputs & Outputs</em>. ICML.</li>
            <li>Reed, S. et al. (2022). <em>A Generalist Agent</em>. DeepMind (Gato). arXiv:2205.06175</li>
            <li>MIT 6.S191, Stanford CS224n, CMU Neural Sequence Models (2024 editions)</li>
          </ul>
        </div>
      </section>


       {/* Section 12 */}
<section id="tip" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Practical Tip</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">กลยุทธ์การเลือกใช้สถาปัตยกรรม Encoder–Decoder</h3>
    <p>
      การเลือกใช้ Encoder–Decoder architecture ควรขึ้นอยู่กับลักษณะของปัญหา เช่น translation, summarization, หรือ multi-modal generation โดยต้องพิจารณาว่างานนั้นต้องการการเข้าใจบริบทเชิงลึก, การตอบสนองแบบ real-time หรือการสร้างลำดับที่มีคุณภาพสูงเพียงใด
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li><strong>ใช้ Encoder-only:</strong> สำหรับ task ที่ต้องการความเข้าใจ เช่น classification หรือ sentence embedding</li>
      <li><strong>ใช้ Decoder-only:</strong> สำหรับ generation เช่น GPT-based chatbot</li>
      <li><strong>ใช้ Encoder–Decoder เต็มรูปแบบ:</strong> สำหรับ translation, summarization, และ multi-input-output tasks</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p><strong>Insight:</strong> โมเดล Encoder–Decoder มีความยืดหยุ่นสูงและเป็นพื้นฐานของหลายระบบ NLP และ Multi-modal AI ที่ใช้จริงในโลกอุตสาหกรรม เช่น Google Translate, ChatGPT และ DALL·E</p>
    </div>

    <h3 className="text-xl font-semibold">แนวทางสำหรับ Training ที่มีประสิทธิภาพ</h3>
    <p>
      การฝึก Encoder–Decoder จำเป็นต้องจัดการทรัพยากรให้เหมาะสม โดยเฉพาะในงานที่ใช้ sequence ยาวหรือต้องการเรียนรู้เชิงลึกจากข้อมูลหลาย modality เช่น text + image
    </p>

    <ul className="list-decimal ml-6 space-y-2">
      <li>ใช้ Optimizer เช่น AdamW ร่วมกับ learning rate scheduler</li>
      <li>ฝึกโดยใช้ mixed precision เพื่อประหยัด GPU memory</li>
      <li>ใช้ technique อย่าง label smoothing เพื่อหลีกเลี่ยง overconfidence</li>
      <li>ใส่ regularization เช่น dropout ใน encoder/decoder layers</li>
    </ul>

    <h3 className="text-xl font-semibold">แนวทางการ Deploy ให้พร้อม Production</h3>
    <p>
      เมื่อเตรียม Encoder–Decoder สำหรับ deployment ต้องคำนึงถึงความเร็ว ความเสถียร และ scalability:
    </p>

    <ul className="list-disc ml-6 space-y-1">
      <li>ใช้ quantization หรือ pruning ลดขนาดโมเดล</li>
      <li>รองรับ batching สำหรับความเร็วสูงขึ้นใน inference</li>
      <li>ใช้ beam search หรือ nucleus sampling ใน decoding อย่างมีประสิทธิภาพ</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl text-black dark:text-blue-100 border-l-4 border-blue-500">
      <p><strong>Highlight Box:</strong> Multi-head attention ใน encoder และ cross-attention ใน decoder เป็นหัวใจของการเรียนรู้แบบ context-aware ในหลาย application</p>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Vaswani et al. (2017). <em>Attention Is All You Need</em>. NeurIPS.</li>
      <li>Raffel et al. (2020). <em>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</em>.</li>
      <li>Harvard NLP Group. <em>The Annotated Transformer</em>. https://nlp.seas.harvard.edu</li>
      <li>Stanford CS224n (2023). <em>Lecture 11: Transformers in Practice</em>.</li>
      <li>Oxford NLP Lab. <em>Practical Guide to Transformer-based Architectures</em>.</li>
    </ul>

  </div>
</section>

  {/* Section 13 */}
<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">13. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">วิวัฒนาการของ Encoder–Decoder ต่อ AI ยุคใหม่</h3>
    <p>
      สถาปัตยกรรม Encoder–Decoder ไม่เพียงเป็นหัวใจของงานแปลภาษา (Neural Machine Translation) แต่ยังกลายเป็นต้นแบบของโมเดล AI สมัยใหม่อย่าง T5, BART, และ PaLM ที่พัฒนาต่อจากแนวคิดเดิม โดยการแยกหน้าที่การเข้ารหัสข้อมูลและการถอดรหัสให้ชัดเจนเปิดทางให้สามารถออกแบบโมดูลที่ยืดหยุ่น, ปรับขนาดได้ง่าย และสามารถนำไปใช้ในหลายสาขา เช่น computer vision, reinforcement learning และ audio processing
    </p>

    <h3 className="text-xl font-semibold">การปรับใช้ในยุค Multi-modal และ Generative AI</h3>
    <p>
      หนึ่งในก้าวกระโดดสำคัญคือการใช้ Encoder สำหรับประมวลผลภาพ เสียง หรือแม้กระทั่งสัญญาณ sensor ขณะที่ Decoder รับผิดชอบการ generate ภาษา การจัดการกับข้อมูลข้าม modality ได้กลายเป็นกลยุทธ์สำคัญของโมเดลยุคใหม่อย่าง Flamingo, Gato, และ Gemini
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl text-black dark:text-yellow-100 border-l-4 border-yellow-500">
      <p>
        <strong>Insight Box:</strong> การแยกหน้าที่ของ Encoder และ Decoder อย่างชัดเจน ทำให้สามารถใช้ backbone เดิมร่วมกับ task ใหม่ได้อย่างรวดเร็ว ตัวอย่างเช่นการ fine-tune encoder เดิมที่ฝึกจาก vision-language pretraining แล้วเปลี่ยน decoder เพื่อใช้งานเฉพาะทางในระบบ healthcare หรือ robotics
      </p>
    </div>

    <h3 className="text-xl font-semibold">ความสำคัญต่อ Explainability และ Interpretability</h3>
    <p>
      Encoder–Decoder ช่วยให้สามารถแยกและวิเคราะห์ได้ว่า "ข้อมูลต้นทางถูกเข้ารหัสอย่างไร" และ "กระบวนการ decoding ตัดสินใจจาก context อะไรบ้าง" ซึ่งเหมาะสำหรับงานที่ต้องมีความโปร่งใส เช่น งานทางการแพทย์ การเงิน และระบบแนะนำ (recommendation systems)
    </p>

    <ul className="list-disc ml-6 space-y-2">
      <li>Encoder สามารถ visualized ได้ว่า token ใดมีอิทธิพลต่อ representation ที่เรียนรู้</li>
      <li>Decoder สามารถตรวจสอบ attention weights เพื่อวิเคราะห์การ generate output</li>
      <li>ช่วยสร้างความเชื่อมั่นในระบบ AI ที่ใช้งานในระบบวิกฤติ</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-5 rounded-xl text-black dark:text-blue-100 border-l-4 border-blue-500">
      <p>
        <strong>Highlight Box:</strong> ในยุค Explainable AI (XAI) การออกแบบสถาปัตยกรรมให้แยกหน้าที่ encoder–decoder อย่างชัดเจนถือเป็นแนวทางที่สนับสนุนการตรวจสอบและตีความได้อย่างมีประสิทธิภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold">ทิศทางในอนาคตของ Encoder–Decoder</h3>
    <p>
      หลายงานวิจัยเริ่มมองหาวิธีทำให้ encoder–decoder ทำงานได้ดีกว่าเดิม เช่น การปรับโครงสร้างให้ efficient มากขึ้น, การใช้ routing หรือ sparse attention ใน encoder, หรือการสร้าง decoder แบบ reusable สำหรับ task หลายรูปแบบ เช่น multi-task learning และ zero-shot generation
    </p>

    <table className="w-full border border-gray-300 dark:border-gray-700 text-sm">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">แนวโน้ม</th>
          <th className="border px-4 py-2">ผลกระทบ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Modular Decoder</td>
          <td className="border px-4 py-2">ลด cost ในการ fine-tune ลงแบบ exponential</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Universal Encoder</td>
          <td className="border px-4 py-2">ใช้งานได้กับหลาย domain โดยไม่ต้อง retrain</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Dual-path Decoder</td>
          <td className="border px-4 py-2">เพิ่ม performance ใน long-form generation</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc ml-6 text-sm space-y-2">
      <li>Google Research (2022). <em>Scaling Autoregressive Models Beyond 1 Trillion Parameters</em>.</li>
      <li>CMU Multimodal NLP Group. <em>Lecture: Multimodal Encoder-Decoder Systems</em>.</li>
      <li>MIT Deep Learning Series. <em>Transformer Variants & Interpretability</em>.</li>
      <li>Meta AI. <em>Modular Decoding in Large Language Models</em>.</li>
      <li>Stanford CRFM. <em>Universal Transfer with Text-to-Text Models</em>.</li>
    </ul>

  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day34 theme={theme} />
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
        <ScrollSpy_Ai_Day34 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day34_EncoderDecoder;
