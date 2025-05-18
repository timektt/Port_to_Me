import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day35 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day35";
import MiniQuiz_Day35 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day35";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day35_TransferLearning = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day35_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day35_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day35_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day35_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day35_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day35_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day35_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day35_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day35_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day35_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day35_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day35_12").format("auto").quality("auto").resize(scale().width(501));
  const img13 = cld.image("Day35_13").format("auto").quality("auto").resize(scale().width(501));
  const img14 = cld.image("Day35_14").format("auto").quality("auto").resize(scale().width(501));
  const img15 = cld.image("Day35_15").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 35: Transfer Learning & Pretraining</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

    <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Transfer Learning ถึงเปลี่ยนเกม?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-6 text-base leading-relaxed">
    <p>
      Transfer Learning ได้กลายเป็นหนึ่งในกลยุทธ์หลักที่ขับเคลื่อนความก้าวหน้าอย่างรวดเร็วของ Deep Learning ในช่วงทศวรรษที่ผ่านมา โดยเฉพาะในยุคของโมเดลขนาดใหญ่ เช่น BERT, GPT และ CLIP ซึ่งสามารถเรียนรู้ความรู้ทั่วไปจากการ pretrain แล้วนำไปประยุกต์กับงาน downstream ได้อย่างมีประสิทธิภาพ
    </p>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded-md">
      <p className="font-semibold">Highlight:</p>
      <p>
        แนวคิดของการถ่ายทอดความรู้จากโมเดลที่เรียนรู้มาก่อนหน้า ช่วยลดความต้องการข้อมูล labeled จำนวนมาก และสามารถทำให้ระบบเรียนรู้ได้เร็วขึ้น โดยเฉพาะใน domain ที่มีข้อมูลจำนวนน้อยหรือหายาก
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6">วิวัฒนาการจาก Zero Start สู่การ Reuse</h3>
    <p>
      ก่อนการมาถึงของ Transfer Learning การฝึกโมเดลแต่ละตัวจำเป็นต้องเริ่มเรียนรู้จากศูนย์ (training from scratch) ซึ่งทำให้ต้องใช้พลังประมวลผลสูงและข้อมูล labeled จำนวนมาก ในขณะที่ Transfer Learning เปิดโอกาสให้ reuse representation จาก task หนึ่งสู่อีก task หนึ่งได้อย่างยืดหยุ่น
    </p>

    <h3 className="text-xl font-semibold mt-6">รูปแบบการ Transfer ที่ใช้จริงในอุตสาหกรรม</h3>
    <ul className="list-disc pl-6">
      <li>การใช้โมเดล pretrain เช่น ResNet, BERT เป็น feature extractor</li>
      <li>การ fine-tune เฉพาะบาง layer ใน task ใหม่</li>
      <li>การฝึกแบบ continual หรือ lifelong learning</li>
    </ul>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded-md">
      <p className="font-semibold">Insight:</p>
      <p>
        ในงานวิจัยของ Google Research (JFT-300M) พบว่า การ pretrain บน dataset ขนาดใหญ่ช่วยเพิ่ม accuracy ใน downstream task อย่างมีนัยสำคัญ แม้จะ fine-tune ด้วยข้อมูลน้อยมากในภายหลัง
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6">เปรียบเทียบ: Training from Scratch vs Transfer Learning</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">มิติ</th>
            <th className="border px-4 py-2">Training from Scratch</th>
            <th className="border px-4 py-2">Transfer Learning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Data Requirement</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Training Time</td>
            <td className="border px-4 py-2">นาน</td>
            <td className="border px-4 py-2">สั้นลงมาก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Performance</td>
            <td className="border px-4 py-2">ขึ้นกับข้อมูล</td>
            <td className="border px-4 py-2">มักจะดีกว่าใน task เดียวกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Generalization</td>
            <td className="border px-4 py-2">จำกัด</td>
            <td className="border px-4 py-2">ดีขึ้นชัดเจน</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-6">บทสรุป</h3>
    <p>
      การเปลี่ยนจากการ train from scratch ไปสู่การใช้ transfer learning ไม่ใช่เพียงแค่การประหยัดเวลาและทรัพยากร แต่คือการเปลี่ยนวิธีคิดเกี่ยวกับการเรียนรู้ของระบบ AI อย่างสิ้นเชิง ส่งผลให้เกิดโมเดลที่เข้าใจภาษา ภาพ และมัลติโมดัลได้ลึกซึ้งขึ้น และสามารถนำไปประยุกต์ใช้ในบริบทใหม่ ๆ ได้หลากหลายอย่างน่าเชื่อถือ
    </p>

    <h3 className="text-xl font-semibold mt-6">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300">
      <li>Howard, J., & Gugger, S. (2020). <i>Fastai: A layered API for deep learning</i>. arXiv:2002.04688</li>
      <li>Sun, C., et al. (2017). <i>Revisiting the unreasonable effectiveness of data</i>. arXiv:1707.02968</li>
      <li>Raghu, M., et al. (2019). <i>Transfusion: Understanding transfer learning for medical imaging</i>. NeurIPS</li>
      <li>Stanford CS231n Lecture Notes: Transfer Learning</li>
    </ul>
  </div>
</section>


   <section id="definition" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Transfer Learning คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-6 text-base leading-relaxed">
    <p>
      Transfer Learning คือกระบวนการที่นำโมเดลที่ได้ผ่านการเรียนรู้ (pretrained) มาแล้วบน task หนึ่ง มาประยุกต์ใช้กับ task อื่น โดยไม่จำเป็นต้องเริ่มการฝึกใหม่จากศูนย์ วิธีการนี้ใช้ได้ดีในกรณีที่ task เป้าหมายมีข้อมูลจำกัด ในขณะที่ task ต้นทางมีข้อมูลขนาดใหญ่และความรู้ทั่วไปที่สามารถถ่ายทอดได้
    </p>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded-md">
      <p className="font-semibold">Highlight:</p>
      <p>
        Transfer Learning ช่วยให้การฝึกโมเดลเร็วขึ้น ประหยัดพลังงาน และยังได้ผลลัพธ์ที่ดีกว่าหากโครงสร้างของ task ต้นทางและปลายทางมีความสัมพันธ์กันอย่างเหมาะสม
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6">นิยามเชิงคณิตศาสตร์</h3>
    <p>
      จากงานของ Pan & Yang (2010), Transfer Learning ถูกนิยามในกรอบ formal ดังนี้:
    </p>
    <div className="overflow-x-auto bg-gray-800 text-white text-sm p-4 rounded-lg my-4 font-mono">
  <code>
    {`ให้ D_S = {X_S, P(X_S)} เป็น domain ต้นทาง (source domain)
ให้ T_S = {Y_S, f_S(X_S)} เป็น task ต้นทาง
ให้ D_T = {X_T, P(X_T)} เป็น domain ปลายทาง (target domain)
ให้ T_T = {Y_T, f_T(X_T)} เป็น task ปลายทาง

Transfer Learning คือการเรียนรู้ f_T โดยอาศัยความรู้จาก D_S, T_S แม้ว่า D_S ≠ D_T หรือ T_S ≠ T_T`}
  </code>
</div>


    <h3 className="text-xl font-semibold mt-6">กลไกพื้นฐานของการ Transfer</h3>
    <ul className="list-disc pl-6">
      <li>เรียนรู้ representation จาก task ต้นทาง</li>
      <li>นำ representation นั้นมาใช้กับ task ปลายทาง</li>
      <li>อาจ fine-tune บางส่วนของโมเดลให้เข้ากับ task ใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6">ประเภทของ Knowledge ที่ Transfer ได้</h3>
    <ul className="list-disc pl-6">
      <li>Instance Transfer (ยืมข้อมูลมาใช้โดยตรง)</li>
      <li>Feature Representation Transfer (ใช้ latent space ที่เรียนรู้มาแล้ว)</li>
      <li>Parameter Transfer (ถ่ายน้ำหนักของโมเดล)</li>
      <li>Relational Knowledge Transfer (ถ่ายโอนโครงสร้างหรือความสัมพันธ์)</li>
    </ul>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded-md">
      <p className="font-semibold">Insight:</p>
      <p>
        Transfer Learning เป็นแกนหลักของ paradigm ใหม่ เช่น Foundation Models และ Self-Supervised Learning ที่เปลี่ยนแปลงวิธีการฝึก AI ทั่วโลกในปัจจุบัน
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6">ตัวอย่างการใช้งานจริง</h3>
    <ul className="list-disc pl-6">
      <li>BERT ที่ pretrain บน Wikipedia แล้วนำไป finetune สำหรับ sentiment analysis</li>
      <li>ResNet ที่ pretrain บน ImageNet แล้วใช้ใน medical image classification</li>
      <li>CLIP ที่เรียนรู้จากภาพ-ข้อความ นำไปใช้กับ retrieval, multimodal tasks</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300">
      <li>Pan, S. J., & Yang, Q. (2010). <i>A survey on transfer learning</i>. IEEE Transactions on Knowledge and Data Engineering</li>
      <li>Ruder, S. (2019). <i>Neural Transfer Learning for NLP</i>. PhD Thesis, NUI Galway</li>
      <li>Tan, C. et al. (2018). <i>A Survey on Deep Transfer Learning</i>. arXiv:1808.01974</li>
      <li>Oxford Deep Learning Lectures: Transfer Learning</li>
    </ul>
  </div>
</section>


<section id="pretrain-vs-finetune" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Pretraining vs Finetuning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <h3 className="text-xl font-semibold mb-4">แนวคิดพื้นฐาน</h3>
  <p className="prose prose-lg dark:prose-invert max-w-none">
    ในระบบการเรียนรู้ของโมเดลขนาดใหญ่ การแยกขั้นตอนระหว่าง <strong>Pretraining</strong> และ <strong>Finetuning</strong> ถือเป็นแนวปฏิบัติหลักที่ช่วยเพิ่มประสิทธิภาพในการเรียนรู้และการประยุกต์ใช้งานจริง โดยเฉพาะในโมเดลขนาดใหญ่ที่ใช้ข้อมูลจำนวนมาก เช่น GPT, BERT หรือ ViT
  </p>

  <div className="bg-yellow-700 text-yellow-900 p-4 rounded-lg my-6">
    <strong>Insight:</strong> Pretraining คือการสร้างความเข้าใจพื้นฐานของโมเดลจากข้อมูลขนาดใหญ่ โดยไม่มีเป้าหมายเฉพาะเจาะจง ส่วน Finetuning คือการปรับโมเดลให้เหมาะสมกับงานเฉพาะ เช่น การวิเคราะห์ความรู้สึกหรือการแปลภาษา
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">เปรียบเทียบแบบตาราง</h3>
  <div className="overflow-auto">
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">Aspect</th>
          <th className="border px-4 py-2">Pretraining</th>
          <th className="border px-4 py-2">Finetuning</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Objective</td>
          <td className="border px-4 py-2">เรียนรู้ pattern ทั่วไปจากข้อมูลขนาดใหญ่</td>
          <td className="border px-4 py-2">ปรับให้เหมาะสมกับ task เฉพาะ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Data</td>
          <td className="border px-4 py-2">Unlabeled (เช่น Wikipedia, Common Crawl)</td>
          <td className="border px-4 py-2">Labeled (เช่น GLUE, SQuAD)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Training Time</td>
          <td className="border px-4 py-2">ยาวนานและต้องใช้ compute สูง</td>
          <td className="border px-4 py-2">สั้นกว่าและต้องใช้ข้อมูลจำเพาะ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ผลลัพธ์</td>
          <td className="border px-4 py-2">Representation ที่ general</td>
          <td className="border px-4 py-2">Model ที่ optimize กับ downstream task</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">การใช้งานในโลกจริง</h3>
  <ul className="list-disc pl-6 space-y-2">
    <li>BERT ถูก pretrain ด้วย masked language modeling แล้ว finetune สำหรับ named entity recognition (NER)</li>
    <li>CLIP ของ OpenAI ใช้ pretraining ข้าม modal (ภาพ+ข้อความ) แล้ว finetune สำหรับ classification หรือ retrieval</li>
    <li>ใน computer vision มีการ pretrain บน ImageNet แล้ว finetune บน datasets เล็ก เช่น CIFAR-10 หรือ ChestX-ray</li>
  </ul>

  <div className="bg-blue-700 text-blue-900 p-4 rounded-lg my-6">
    <strong>Highlight:</strong> การแบ่งขั้นตอนช่วยให้สามารถใช้ทรัพยากรอย่างมีประสิทธิภาพสูงสุด และยังส่งเสริมการเรียนรู้แบบ few-shot หรือ zero-shot ได้ในบางกรณี
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">แหล่งอ้างอิงระดับโลก</h3>
  <ul className="list-decimal pl-6 space-y-1 text-sm">
    <li>Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", arXiv:1810.04805</li>
    <li>Radford et al., "Language Models are Few-Shot Learners", OpenAI, 2020</li>
    <li>Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021</li>
    <li>Stanford CS224N Lecture Notes: https://web.stanford.edu/class/cs224n/</li>
  </ul>
</section>


<section id="transfer-strategies" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. กลยุทธ์การ Transfer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">การเลือกกลยุทธ์การถ่ายโอน</h3>
    <p>
      การเลือกกลยุทธ์การ Transfer Learning อย่างเหมาะสมถือเป็นปัจจัยสำคัญที่ส่งผลต่อประสิทธิภาพของโมเดลในการเรียนรู้จากข้อมูลใหม่ โดยกลยุทธ์แต่ละแบบมีความเหมาะสมกับบริบท ปริมาณข้อมูล และลักษณะของ task ที่แตกต่างกัน
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500">
      <p className="font-semibold">Highlight:</p>
      <p>
        หากใช้กลยุทธ์ไม่เหมาะสมกับ distribution ของ task ใหม่ อาจนำไปสู่ negative transfer ซึ่งเป็นภาวะที่การถ่ายโอนความรู้ทำให้โมเดลมีประสิทธิภาพแย่ลง
      </p>
    </div>

    <h3 className="text-xl font-semibold">กลยุทธ์ที่พบบ่อย</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Feature Extraction:</strong> ใช้ pretrained model เป็น feature extractor โดยไม่อัปเดตพารามิเตอร์</li>
      <li><strong>Fine-tuning:</strong> ปรับพารามิเตอร์บางส่วนหรือทั้งหมดของโมเดลที่ผ่าน pretraining ให้เหมาะกับ task ใหม่</li>
      <li><strong>Frozen Backbone:</strong> แช่พารามิเตอร์ของ base model และเรียนรู้เฉพาะ layer ที่เพิ่มเติม</li>
      <li><strong>Layer-wise Transfer:</strong> ถ่ายโอนเฉพาะบาง layer ที่เกี่ยวข้องกับ task ปลายทางเท่านั้น</li>
      <li><strong>Multi-task Pretraining:</strong> ทำ pretraining หลาย task พร้อมกันเพื่อสร้าง representation ที่ general มากขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">เมื่อไรควรใช้กลยุทธ์ใด</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">กลยุทธ์</th>
            <th className="border px-4 py-2">ข้อมูลใหม่</th>
            <th className="border px-4 py-2">ปรับพารามิเตอร์</th>
            <th className="border px-4 py-2">ความยืดหยุ่น</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Feature Extraction</td>
            <td className="border px-4 py-2">น้อย</td>
            <td className="border px-4 py-2">ไม่</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Fine-tuning</td>
            <td className="border px-4 py-2">มาก</td>
            <td className="border px-4 py-2">ใช่</td>
            <td className="border px-4 py-2">สูง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Frozen Backbone</td>
            <td className="border px-4 py-2">ปานกลาง</td>
            <td className="border px-4 py-2">บางส่วน</td>
            <td className="border px-4 py-2">ปานกลาง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานวิจัยของ <em>Howard and Ruder (2018)</em> แสดงให้เห็นว่า Universal Language Model Fine-tuning (ULMFiT) สามารถเพิ่มประสิทธิภาพใน low-resource task ได้มาก โดยไม่จำเป็นต้องฝึกใหม่ทั้งโมเดล
      </p>
    </div>

    <h3 className="text-xl font-semibold">ข้อควรระวัง</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ไม่ควร fine-tune ทั้งโมเดลหากมีข้อมูลจำกัด อาจเกิด overfitting ได้ง่าย</li>
      <li>ต้องระวัง feature shift และควรพิจารณาการทำ normalization ใหม่</li>
      <li>การ freeze layer มากเกินไปอาจจำกัดศักยภาพของการปรับตัวใน task ใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300">
      <li>Ruder, S., Peters, M., Swayamdipta, S., & Wolf, T. (2019). <em>Transfer Learning in NLP</em>. arXiv:1903.11260</li>
      <li>Howard, J., & Ruder, S. (2018). <em>Universal Language Model Fine-tuning for Text Classification</em>. ACL</li>
      <li>Yosinski, J. et al. (2014). <em>How transferable are features in deep neural networks?</em> NeurIPS</li>
    </ul>
  </div>
</section>


<section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. ประเภทของ Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert space-y-6">
    <h3>การจำแนกประเภทโดยอิงจากลักษณะการเรียนรู้</h3>
    <p>Transfer Learning สามารถจำแนกออกได้หลายประเภทตามธรรมชาติของปัญหาและความสัมพันธ์ระหว่างชุดข้อมูลต้นทางและปลายทาง ซึ่งการจำแนกประเภทเหล่านี้ได้รับการศึกษาและใช้งานจริงในงานวิจัยชั้นนำของมหาวิทยาลัยอย่าง Stanford และ MIT</p>

    <h3>1. Inductive Transfer Learning</h3>
    <p>ในกรณีนี้ งานต้นทาง (source task) และงานปลายทาง (target task) แตกต่างกันอย่างชัดเจน โดยมี label ให้ทั้งสองงาน ตัวอย่างเช่น การเรียนรู้จากข้อมูลภาษาอังกฤษเพื่อนำไปประยุกต์ใช้กับการวิเคราะห์ข้อความในภาษาฝรั่งเศส</p>

    <h3>2. Transductive Transfer Learning</h3>
    <p>งานต้นทางและปลายทางเหมือนกัน แต่ข้อมูลที่มีอยู่ในงานปลายทางไม่มี label ตัวอย่างเช่น การเทรนโมเดลจากข้อมูลของผู้ใช้ในสหรัฐฯ แล้วนำไปใช้กับผู้ใช้ในญี่ปุ่นโดยไม่มี label</p>

    <h3>3. Unsupervised Transfer Learning</h3>
    <p>ทั้งงานต้นทางและงานปลายทางไม่มี label เช่น การใช้ embedding ที่เรียนรู้จาก Word2Vec บน Wikipedia เพื่อนำมาใช้กับ Clustering เอกสารข่าว</p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-400">
      <strong>Highlight:</strong> การจำแนกประเภทเหล่านี้มีผลต่อการเลือกโมเดล เทคนิคการฝึกสอน และการประเมินผล ซึ่งหากเลือกไม่เหมาะสมอาจส่งผลให้ Transfer Learning ล้มเหลวหรือเกิด negative transfer ได้
    </div>

    <h3>ตารางเปรียบเทียบประเภทหลัก</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">ประเภท</th>
            <th className="border px-4 py-2">Label ใน Source</th>
            <th className="border px-4 py-2">Label ใน Target</th>
            <th className="border px-4 py-2">ความเหมือนของ Task</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Inductive</td>
            <td className="border px-4 py-2">มี</td>
            <td className="border px-4 py-2">มี</td>
            <td className="border px-4 py-2">ต่างกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Transductive</td>
            <td className="border px-4 py-2">มี</td>
            <td className="border px-4 py-2">ไม่มี</td>
            <td className="border px-4 py-2">เหมือนกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Unsupervised</td>
            <td className="border px-4 py-2">ไม่มี</td>
            <td className="border px-4 py-2">ไม่มี</td>
            <td className="border px-4 py-2">ต่างกัน</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-500">
      <strong>Insight Box:</strong> ในโลกของ LLMs และ CV วันนี้ การแยกประเภทที่ชัดเจนไม่เพียงพออีกต่อไป ต้องอาศัย meta-learning หรือ continual learning ร่วมด้วยเพื่อจัดการบริบทและข้อมูลใหม่
    </div>

    <h3>แหล่งอ้างอิง:</h3>
    <ul className="list-disc list-inside">
      <li>Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering.</li>
      <li>Ruder, S. (2019). Neural Transfer Learning for NLP. arXiv:1903.11260.</li>
      <li>Yosinski, J. et al. (2014). How transferable are features in deep neural networks?. NeurIPS.</li>
    </ul>
  </div>
</section>


       <section id="pretraining-research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ตัวอย่างงานวิจัย Pretraining สำคัญ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <h3 className="text-xl font-bold mb-4">แนวคิดเบื้องต้นจากงานวิจัยสำคัญ</h3>
  <p className="mb-4 leading-relaxed">
    Pretraining ได้กลายเป็นแนวทางหลักใน Deep Learning สมัยใหม่ โดยเฉพาะในงาน NLP และ Vision ซึ่งงานวิจัยสำคัญตั้งแต่ BERT (Devlin et al., 2018) ไปจนถึง GPT-4 (OpenAI, 2023) ได้ชี้ให้เห็นถึงประสิทธิภาพของโมเดลขนาดใหญ่ที่ผ่านการ Pretrain บนข้อมูลขนาดใหญ่หลากหลายประเภท ก่อนนำไป Finetune บนงานเฉพาะ
  </p>

  <h3 className="text-lg font-semibold mb-2">BERT: Bidirectional Encoder Representations from Transformers</h3>
  <ul className="list-disc pl-6 mb-4 space-y-1">
    <li>ใช้ Masked Language Model (MLM) และ Next Sentence Prediction (NSP)</li>
    <li>เรียนรู้ context แบบ bidirectional</li>
    <li>กลายเป็นโมเดลพื้นฐานให้กับหลากหลาย downstream tasks</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">GPT: Generative Pretraining Transformer</h3>
  <ul className="list-disc pl-6 mb-4 space-y-1">
    <li>ใช้ causal (autoregressive) language modeling</li>
    <li>เรียนรู้จากข้อมูลจำนวนมหาศาลโดยไม่มี supervision</li>
    <li>สเกลโมเดลไปถึง GPT-4 ที่ใช้ RLHF และ few-shot prompting</li>
  </ul>

  <h3 className="text-lg font-semibold mb-2">CLIP (Radford et al., OpenAI, 2021)</h3>
  <p className="mb-4">
    เป็นการ pretrain ด้วยภาพและข้อความพร้อมกัน โดยใช้ contrastive learning ระหว่าง text-image pair ทำให้สามารถ zero-shot classification ได้อย่างมีประสิทธิภาพ
  </p>

  <div className="bg-blue-700 dark:bg-blue-900 p-4 rounded-md shadow-inner mb-6">
    <p className="text-sm font-medium">
      ✅ <strong>Insight:</strong> งานวิจัยเหล่านี้ยืนยันว่า Pretraining ไม่เพียงแต่ลดความต้องการ label แต่ยังช่วยให้โมเดลสามารถ generalize ได้ดีกว่าการ train จาก scratch อย่างชัดเจน
    </p>
  </div>

  <h3 className="text-lg font-semibold mb-2">ตารางสรุป: จุดเด่นของงานวิจัย Pretraining สำคัญ</h3>
  <div className="overflow-x-auto mb-6">
    <table className="w-full text-sm border border-gray-300 dark:border-gray-600">
      <thead className="bg-gray-700 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2 text-left">Model</th>
          <th className="border px-4 py-2 text-left">Year</th>
          <th className="border px-4 py-2 text-left">Objective</th>
          <th className="border px-4 py-2 text-left">Domain</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">BERT</td>
          <td className="border px-4 py-2">2018</td>
          <td className="border px-4 py-2">Masked LM, NSP</td>
          <td className="border px-4 py-2">Text (NLP)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">GPT</td>
          <td className="border px-4 py-2">2018–2023</td>
          <td className="border px-4 py-2">Autoregressive LM</td>
          <td className="border px-4 py-2">Text (NLP)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">CLIP</td>
          <td className="border px-4 py-2">2021</td>
          <td className="border px-4 py-2">Contrastive</td>
          <td className="border px-4 py-2">Multimodal</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-lg font-semibold mb-2">แหล่งอ้างอิง</h3>
  <ul className="list-disc pl-6 space-y-1 text-sm">
    <li>Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, NAACL 2019</li>
    <li>Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”, ICML 2021 (CLIP)</li>
    <li>Brown et al., “Language Models are Few-Shot Learners”, NeurIPS 2020 (GPT-3)</li>
    <li>OpenAI, “GPT-4 Technical Report”, 2023</li>
  </ul>
</section>


<section id="pretraining-cv" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Pretraining ในคอมพิวเตอร์วิทัศน์ (CV)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <h3 className="text-xl font-bold mb-4">แนวทางของ Pretraining ในงาน CV</h3>
  <p className="mb-4 leading-relaxed">
    ในงาน Computer Vision การ Pretraining บน dataset ขนาดใหญ่ เช่น ImageNet, OpenImages, หรือ LAION-400M ถือเป็นแนวทางมาตรฐานที่ช่วยให้โมเดลสามารถเรียนรู้ feature พื้นฐานของภาพก่อนจะนำไป finetune บนงานจำเพาะ เช่น object detection หรือ medical imaging ซึ่งช่วยให้ได้ผลลัพธ์ที่แม่นยำแม้มีข้อมูล labeled น้อย
  </p>

  <h3 className="text-lg font-semibold mb-2">โมเดล Pretrain สำคัญในงาน CV</h3>
  <ul className="list-disc pl-6 mb-4 space-y-1">
    <li><strong>ResNet (He et al., 2015)</strong>: backbone ยอดนิยมที่ใช้ Pretrain บน ImageNet</li>
    <li><strong>ViT (Vision Transformer)</strong>: ใช้ patch-based self-attention และ Pretrain บน JFT-300M</li>
    <li><strong>MoCo / SimCLR</strong>: contrastive learning ที่ไม่ต้องใช้ label</li>
    <li><strong>DINO</strong>: self-supervised vision transformer ที่โดดเด่นเรื่อง semantic emergence</li>
  </ul>

  <div className="bg-yellow-700 dark:bg-yellow-900 p-4 rounded-md shadow-inner mb-6">
    <p className="text-sm font-medium">
      💡 <strong>Highlight:</strong> งาน vision สมัยใหม่แทบทั้งหมดตั้งต้นจาก Pretraining — ไม่ว่าจะใช้ label (supervised) หรือไม่ใช้ label (self-supervised) ก็ตาม
    </p>
  </div>

  <h3 className="text-lg font-semibold mb-2">เปรียบเทียบแนวทาง Pretraining</h3>
  <div className="overflow-x-auto mb-6">
    <table className="w-full text-sm border border-gray-700 dark:border-gray-600">
      <thead className="bg-gray-700 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2 text-left">Method</th>
          <th className="border px-4 py-2 text-left">Supervision</th>
          <th className="border px-4 py-2 text-left">Backbone</th>
          <th className="border px-4 py-2 text-left">Dataset</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ResNet</td>
          <td className="border px-4 py-2">Supervised</td>
          <td className="border px-4 py-2">CNN</td>
          <td className="border px-4 py-2">ImageNet</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ViT</td>
          <td className="border px-4 py-2">Supervised / Self</td>
          <td className="border px-4 py-2">Transformer</td>
          <td className="border px-4 py-2">JFT-300M</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">MoCo v2</td>
          <td className="border px-4 py-2">Self-supervised</td>
          <td className="border px-4 py-2">CNN</td>
          <td className="border px-4 py-2">ImageNet</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">DINO</td>
          <td className="border px-4 py-2">Self-supervised</td>
          <td className="border px-4 py-2">ViT</td>
          <td className="border px-4 py-2">ImageNet</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-lg font-semibold mb-2">แหล่งอ้างอิง</h3>
  <ul className="list-disc pl-6 space-y-1 text-sm">
    <li>He et al., "Deep Residual Learning for Image Recognition", CVPR 2016</li>
    <li>Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021</li>
    <li>Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020 (SimCLR)</li>
    <li>Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021 (DINO)</li>
  </ul>
</section>


<section id="pretraining-nlp" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Pretraining ใน NLP และ Multimodal</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">8.1 ความสำคัญของ Pretraining ใน NLP</h3>
  <p className="mb-4 text-base leading-relaxed">
    ในงานประมวลผลภาษาธรรมชาติ (NLP) การ Pretraining กลายเป็นแนวทางหลักที่ช่วยยกระดับประสิทธิภาพโมเดลอย่างมีนัยสำคัญ โดยเฉพาะเมื่อมีข้อมูล labeled จำกัด เช่นเดียวกับโมเดล BERT, GPT, RoBERTa, T5 และ DeBERTa ที่ถูก pretrain บน corpora ขนาดใหญ่นับพันล้าน token แล้วจึงนำไป finetune บน task-specific เช่น QA, summarization หรือ translation
  </p>

  <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 mb-6">
    <p className="font-semibold">Insight:</p>
    <p>
      BERT ถูก pretrain บน BookCorpus และ Wikipedia โดยใช้ Masked Language Modeling (MLM) และ Next Sentence Prediction ซึ่งช่วยให้สามารถเรียนรู้ semantic และ syntactic pattern ได้ลึกซึ้ง
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">8.2 Multimodal Pretraining</h3>
  <p className="mb-4 text-base leading-relaxed">
    แนวโน้มล่าสุดในงาน NLP คือการรวมข้อมูลหลาย modality เช่น ข้อความ-ภาพ-เสียง โดยใช้แนวคิด pretrain ร่วมกัน เช่น CLIP (Contrastive Language-Image Pretraining) จาก OpenAI ที่จับคู่ระหว่างข้อความและภาพ หรือ Flamingo และ GIT ที่เรียนรู้จาก caption, visual context และลำดับเหตุการณ์แบบต่อเนื่อง
  </p>

  <div className="bg-blue-700 border-l-4 border-blue-500 p-4 mb-6">
    <p className="font-semibold">Highlight:</p>
    <p>
      CLIP ใช้เทคนิค contrastive loss ในการ map ข้อความและภาพให้อยู่ใน latent space เดียวกัน ส่งผลให้สามารถค้นหาหรืออธิบายภาพได้อย่างแม่นยำโดยไม่ต้อง labeled dataset
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">8.3 สถาปัตยกรรมหลักของ NLP Pretrained Models</h3>
  <div className="overflow-x-auto my-4">
  <table className="min-w-[600px] w-full text-sm border border-gray-300 dark:border-gray-700">
    <thead className="bg-gray-700 dark:bg-gray-800">
      <tr>
        <th className="border px-4 py-2">Model</th>
        <th className="border px-4 py-2">Architecture</th>
        <th className="border px-4 py-2">Objective</th>
        <th className="border px-4 py-2">Pretraining Corpus</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">BERT</td>
        <td className="border px-4 py-2">Transformer Encoder</td>
        <td className="border px-4 py-2">MLM, NSP</td>
        <td className="border px-4 py-2">BookCorpus, Wikipedia</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">GPT</td>
        <td className="border px-4 py-2">Transformer Decoder</td>
        <td className="border px-4 py-2">Causal LM</td>
        <td className="border px-4 py-2">WebText</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">T5</td>
        <td className="border px-4 py-2">Encoder-Decoder</td>
        <td className="border px-4 py-2">Span Corruption</td>
        <td className="border px-4 py-2">C4</td>
      </tr>
    </tbody>
  </table>
</div>


  <h3 className="text-xl font-semibold mt-8 mb-4">8.4 อ้างอิงจากงานวิจัยชั้นนำ</h3>
  <ul className="list-disc pl-6 space-y-2 text-base">
    <li>Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL 2019.</li>
    <li>Radford et al., "Language Models are Few-Shot Learners," OpenAI, 2020.</li>
    <li>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," JMLR 2020.</li>
    <li>Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," ICML 2021.</li>
  </ul>
</section>


<section id="scaling-laws" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Scaling Laws และผลของ Pretraining</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose lg:prose-lg dark:prose-invert max-w-none space-y-6 text-base leading-relaxed">
    <h3 className="text-xl font-semibold">แนวคิดเบื้องหลัง Scaling Laws</h3>
    <p>
      Scaling Laws คือแนวทางในการเข้าใจความสัมพันธ์ระหว่างขนาดของโมเดล ปริมาณข้อมูล และทรัพยากรที่ใช้ในการฝึก กับประสิทธิภาพที่ได้จากการ Pretraining โดยเฉพาะอย่างยิ่งในงานของ OpenAI (Kaplan et al., 2020) ได้แสดงให้เห็นว่า loss จะลดลงอย่างเป็นระบบเมื่อมีการเพิ่มขนาดของ parameter, dataset และ compute ในลักษณะ log-log scale
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 border-l-4 border-yellow-500 text-yellow-900 dark:text-yellow-100 p-4 rounded-md">
      <p className="font-semibold">Insight Box:</p>
      <p>
        โมเดลอย่าง GPT-3, PaLM และ LLaMA ได้รับประโยชน์โดยตรงจากการขยายขนาดแบบไม่เปลี่ยนโครงสร้าง โดยใช้ Scaling Laws เป็นพื้นฐานในการตัดสินใจด้านงบประมาณและการฝึกฝน
      </p>
    </div>

    <h3 className="text-xl font-semibold">การทดลองจริงจาก OpenAI และ DeepMind</h3>
    <ul className="list-disc pl-6">
      <li>Kaplan et al. พบว่า loss ลดลงสม่ำเสมอเมื่อเพิ่มจำนวนพารามิเตอร์และข้อมูลบนชุด WikiText-103 และ LAMBADA</li>
      <li>DeepMind เสนอ Chinchilla scaling rule ซึ่งปรับจุด compute-optimal โดยลดพารามิเตอร์และเพิ่มจำนวน token แทน</li>
    </ul>

    <h3 className="text-xl font-semibold">การปรับใช้ Scaling Laws ใน Pretraining</h3>
    <p>
      Scaling Laws ไม่เพียงช่วยให้ทำนาย performance ได้ล่วงหน้า แต่ยังช่วยในการออกแบบโมเดลที่ compute-efficient และวางแผน resource allocation อย่างมีประสิทธิภาพสูงสุด
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 border-l-4 border-blue-500 text-blue-900 dark:text-blue-100 p-4 rounded-md">
      <p className="font-semibold">Highlight Box:</p>
      <p>
        ปริมาณข้อมูลและจำนวน training steps มีผลต่อผลลัพธ์เทียบเท่าหรือมากกว่าการเพิ่มขนาดพารามิเตอร์เพียงอย่างเดียว จึงควรมีการ balance ให้เหมาะสมกับ compute budget
      </p>
    </div>

    <h3 className="text-xl font-semibold">ข้อควรระวังและข้อถกเถียง</h3>
    <ul className="list-disc pl-6">
      <li>การเพิ่มขนาดโมเดลอาจทำให้เกิด overfitting หากไม่มี regularization หรือ early stopping</li>
      <li>การฝึกโมเดลขนาดใหญ่ส่งผลต่อสิ่งแวดล้อมและการใช้พลังงาน</li>
      <li>Scaling Laws ไม่สามารถทดแทนการออกแบบ architecture ที่ดีได้เสมอไป และไม่รองรับ task-specific variation</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Kaplan et al., "Scaling Laws for Neural Language Models", arXiv:2001.08361</li>
      <li>Hoffmann et al., "Training Compute-Optimal Large Language Models", arXiv:2203.15556</li>
      <li>Stanford CS25 Lecture Notes on Scaling and Pretraining</li>
      <li>DeepMind Blog on Chinchilla</li>
    </ul>
  </div>
</section>



<section id="pitfalls" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. ปัญหาและข้อควรระวังของ Transfer Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose dark:prose-invert text-base leading-relaxed space-y-6 max-w-none">
    <h3 className="text-xl font-semibold">ปัญหาหลักที่พบใน Transfer Learning</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Negative Transfer:</strong> เมื่อข้อมูลจาก task ต้นทางมีความต่างจาก task ปลายทางมากเกินไป การถ่ายโอนอาจลดประสิทธิภาพลงแทนที่จะเพิ่มขึ้น</li>
      <li><strong>Catastrophic Forgetting:</strong> การปรับ fine-tune อาจทำให้โมเดลลืมความรู้เดิมจาก pretraining</li>
      <li><strong>Data Mismatch:</strong> หาก distribution ของข้อมูลต่างกัน (domain shift) อาจส่งผลให้ performance ลดลง</li>
      <li><strong>Bias Propagation:</strong> ถ้า pretraining dataset มีอคติ จะถ่ายโอน bias เหล่านั้นไปยัง task ใหม่ด้วย</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 rounded">
      <p className="text-sm font-medium text-yellow-900">Insight Box</p>
      <p className="text-sm text-gray-800">
        การใช้ transfer learning อย่างไม่ระมัดระวังอาจนำไปสู่ความเข้าใจผิดหรือการนำโมเดลไปใช้ในสถานการณ์ที่ไม่เหมาะสม เช่น การวิเคราะห์ทางการแพทย์ที่ต้องการ high reliability
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8">ข้อควรระวังด้านการออกแบบระบบ</h3>
    <p>
      ในการนำ Transfer Learning มาใช้ใน production environment ควรคำนึงถึงปัจจัยดังต่อไปนี้:
    </p>
    <ul className="list-decimal list-inside space-y-2">
      <li>ควรมีการ validate performance ใหม่ทุกครั้งเมื่อเปลี่ยน domain</li>
      <li>ต้องตรวจสอบการแพร่กระจายของ bias โดยเฉพาะใน use case ที่มีผลต่อผู้ใช้ปลายทาง</li>
      <li>หลีกเลี่ยงการใช้โมเดล pretrained ที่ไม่มี transparency หรือไม่มีการเปิดเผย dataset ที่ใช้เทรน</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">แนวทางแก้ไขปัญหาทั่วไป</h3>
    <table className="w-full border border-gray-300 text-sm mt-4">
      <thead className="bg-gray-700">
        <tr>
          <th className="border px-4 py-2">ปัญหา</th>
          <th className="border px-4 py-2">แนวทางป้องกัน</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Negative Transfer</td>
          <td className="border px-4 py-2">เลือก task ที่ใกล้เคียงกันหรือใช้ meta-learning เพื่อเรียนรู้วิธีเลือก task ที่เหมาะสม</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Catastrophic Forgetting</td>
          <td className="border px-4 py-2">ใช้ technique เช่น Elastic Weight Consolidation (EWC)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Bias Propagation</td>
          <td className="border px-4 py-2">เลือก dataset ที่หลากหลายและตรวจสอบ bias อย่างเป็นระบบ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-8">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on knowledge and data engineering.</li>
      <li>Ruder, S. (2019). Neural Transfer Learning for NLP. arXiv:1903.11260</li>
      <li>Goodfellow, I. et al. (2016). Deep Learning. MIT Press.</li>
    </ul>
  </div>
</section>


{/* ───────────────────── 11. เทคนิคช่วยให้ Transfer ดีขึ้น ───────────────────── */}
<section
  id="transfer-techniques"
  className="mb-16 scroll-mt-32 min-h-[400px]"
>
  <h2 className="text-2xl font-semibold mb-6 text-center">
    11. เทคนิคช่วยให้ Transfer ดีขึ้น
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-10">
    {/* Learning-rate scheduling */}
    <section>
      <h3 className="text-xl font-semibold">
        การปรับ Learning Rate และ Scheduling
      </h3>
      <p>
        การปรับ learning rate อย่างเหมาะสมเป็นหนึ่งในเทคนิคสำคัญที่ช่วยเพิ่ม
        ประสิทธิภาพของ transfer learning โดยเฉพาะเมื่อใช้ pre-trained model
        จาก domain ที่แตกต่างกับ target domain…
      </p>

      <div className="bg-yellow-100 dark:bg-yellow-900 rounded-lg p-4 border-l-4 border-yellow-500/80">
        <p className="font-medium">Insight Box</p>
        <p>
          การใช้ learning rate สูงเกินไปในช่วงเริ่มต้นอาจทำให้โมเดลลืม
          ความรู้เดิม (catastrophic forgetting)…
        </p>
      </div>
    </section>

    {/* Layer freezing */}
    <section>
      <h3 className="text-xl font-semibold">Layer Freezing</h3>
      <p>
        การ freezing layers หมายถึงการล็อกค่า&nbsp;weights&nbsp;ของบาง layer
        ไม่ให้เรียนรู้ใหม่ โดยเฉพาะ layer ต้นทางซึ่งมักเก็บ representation
        ระดับ edge หรือลวดลายพื้นฐาน…
      </p>
    </section>

    {/* Data augmentation */}
    <section>
      <h3 className="text-xl font-semibold">Data Augmentation</h3>
      <p>
        ในงานที่มีข้อมูลจำกัด การใช้ data augmentation
        อย่างชาญฉลาดสามารถเพิ่มความหลากหลายของข้อมูลและลด overfitting…
      </p>
    </section>

    {/* Intermediate supervision */}
    <section>
      <h3 className="text-xl font-semibold">Intermediate Layer Supervision</h3>
      <p>
        การฝึกให้ intermediate layer มี auxiliary loss
        ช่วยให้โมเดลเรียนรู้ representation ที่เป็นประโยชน์ระหว่างทาง…
      </p>

      <div className="bg-blue-100 dark:bg-blue-900 rounded-lg p-4 border-l-4 border-blue-500/80">
        <p className="font-medium">Highlight Box</p>
        <p>
          ในโมเดล NLP เช่น GPT หรือ T5 การสอนให้โมเดลทำ&nbsp;multitask&nbsp;หรือใช้
          unsupervised objective เช่น masked language modeling
          คือตัวเร่งความสำเร็จบนหลาย benchmark
        </p>
      </div>
    </section>

    {/* Fine-tuning strategy */}
    <section>
      <h3 className="text-xl font-semibold">
        การเลือก Fine-tuning Strategy ให้เหมาะกับ Task
      </h3>
      <p>
        ไม่ใช่ทุกงานจะได้ผลดีที่สุดจากการ fine-tune โมเดลทั้งหมด…
        นักวิจัยจาก Oxford เสนอให้เริ่มจาก training เฉพาะหัวโมเดล
        แล้วค่อย unfreeze layers ทีละกลุ่ม
      </p>
    </section>

    {/* References */}
    <section>
      <h3 className="text-xl font-semibold">อ้างอิง</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Howard & Gugger (2018) ACL</li>
        <li>Raghu et al. (2019) NeurIPS</li>
        <li>Sun et al. (2019) arXiv:1905.05583</li>
        <li>Ruder (2019) Blog post</li>
      </ul>
    </section>
  </div>
</section>

{/* ───────────────── 12. ตัวอย่างงานจริงที่ใช้ Transfer Learning ─────────────── */}
<section
  id="realworld-examples"
  className="mb-16 scroll-mt-32 min-h-[400px]"
>
  <h2 className="text-2xl font-semibold mb-6 text-center">
    12. ตัวอย่างงานจริงที่ใช้ Transfer Learning
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-10">
    {/* Medical */}
    <section>
      <h3 className="text-xl font-semibold">การใช้งานในอุตสาหกรรมการแพทย์</h3>
      <p>
        Transfer Learning มีบทบาทสำคัญในระบบวินิจฉัยภาพถ่ายทางการแพทย์ เช่น
        การจำแนกมะเร็งเต้านมจาก X-ray หรือ MRI…
      </p>

      <div className="bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-500/80 p-4 rounded-lg">
        <strong>Highlight</strong> โมเดล ResNet ที่ pretrain บน ImageNet
        สามารถ fine-tune ด้วยข้อมูลเพียง 10 % แล้วให้ผลลัพธ์ดีเทียบเท่าฝึกจากศูนย์
      </div>
    </section>

    {/* Automotive */}
    <section>
      <h3 className="text-xl font-semibold">อุตสาหกรรมยานยนต์</h3>
      <p>
        บริษัทอย่าง Tesla, Waymo และ NVIDIA ใช้ Transfer Learning
        เพื่อปรับโมเดล perception ระหว่างภูมิภาคโดยไม่ต้องฝึกใหม่ทั้งชุด…
      </p>

      <ul className="list-disc list-inside space-y-1">
        <li>Pretrain จาก dataset การขับขี่ขนาดใหญ่</li>
        <li>Fine-tune ด้วยข้อมูลเฉพาะประเทศ</li>
        <li>ลดต้นทุน annotation</li>
      </ul>

      <div className="bg-yellow-100 dark:bg-yellow-900 border-l-4 border-yellow-500/80 p-4 rounded-lg">
        <strong>Insight</strong> งานยานยนต์เน้น generalize
        ระหว่างสภาพถนนต่างกัน เช่น LA → โตเกียว
      </div>
    </section>

    {/* Mobile apps */}
    <section>
      <h3 className="text-xl font-semibold">แอปพลิเคชันบนมือถือ</h3>
      <p>
        Google Lens, Snapchat และ LINE Camera
        นำโมเดลที่เบาอย่าง MobileNet/EfficientNet-Lite
        มาปรับใช้เพื่อทำ AR บนอุปกรณ์ edge
      </p>
    </section>

    {/* Finance & security */}
    <section>
      <h3 className="text-xl font-semibold">การเงินและความปลอดภัย</h3>
      <p>
        โมเดลพฤติกรรมผู้ใช้ที่ pretrain ไว้ช่วยตรวจจับ fraud หรือ
        วิเคราะห์ sentiment บนแชตลูกค้าได้แม้ข้อมูลมีจำกัด
      </p>
    </section>

    {/* MT / Chatbot */}
    <section>
      <h3 className="text-xl font-semibold">ระบบแปลภาษาและ Chatbot</h3>
      <p>
        ระบบ M2M-100 ของ Meta แสดงให้เห็นว่าโมเดลหลายภาษา
        สามารถถ่ายทอดความรู้จากภาษาใหญ่ไปสู่ภาษา low-resource
        โดยไม่ต้องใช้ English เป็น pivot
      </p>

      <div className="bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-500/80 p-4 rounded-lg">
        <strong>Highlight</strong> M2M-100 ถ่ายทอดความรู้ได้โดยตรง (en ↔ th, vi ฯลฯ)
      </div>
    </section>

    {/* References */}
    <section>
      <h3 className="text-xl font-semibold">อ้างอิง</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>CheXNet – Rajpurkar et al., 2017</li>
        <li>SimCLR – Chen et al., 2020</li>
        <li>M2M-100 – Fan et al., 2021</li>
        <li>NVIDIA Research Blog (Autonomous Vehicles)</li>
        <li>Google AI Blog (AR Apps)</li>
      </ul>
    </section>
  </div>
</section>



<section id="decision-guide" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-xl font-semibold mb-6 text-center">13. การเลือกว่าจะ Transfer แบบไหน</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">แนวทางการเลือกเชิงกลยุทธ์</h3>
  <p className="text-base leading-relaxed mb-4">
    การเลือกกลยุทธ์การ Transfer Learning ไม่สามารถกำหนดได้แบบตายตัว จำเป็นต้องพิจารณาจากหลายปัจจัย เช่น ความคล้ายกันของ domain, จำนวนข้อมูลของ task เป้าหมาย และระดับของความรู้ที่ model เดิมเรียนรู้มา โดยทั่วไปมีแนวทางหลักๆ ดังนี้:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>ถ้า domain เหมือนกันมาก เช่น ภาพถ่ายแมวจาก dataset ต่างกัน — ใช้ Finetuning ทั้ง model ได้เลย</li>
    <li>ถ้า domain ต่างกันพอสมควร เช่น จากภาษาพูด → ภาพ — ควรใช้แค่ส่วน encoder และ retrain decoder ใหม่</li>
    <li>ถ้าข้อมูลเป้าหมายน้อยมาก — ใช้ few-shot หรือ prompt-based transfer จะเหมาะสมกว่า</li>
  </ul>

  <div className="bg-yellow-700 text-black p-4 rounded-lg border-l-4 border-yellow-400 shadow-md mb-6">
    <h4 className="font-semibold mb-2">Highlight: การพิจารณาจาก Task Similarity</h4>
    <p className="text-sm leading-relaxed">
      งานวิจัยจาก Stanford (Pan et al., 2010) ระบุว่าการใช้ model เดิมใน task ใหม่จะได้ผลดีที่สุดเมื่อ task นั้นมี structure ใกล้เคียงกัน เช่น document classification → sentiment classification เพราะ embedding เดิมยังใช้ได้ดี
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-4">ตารางเปรียบเทียบการเลือกแบบต่าง ๆ</h3>
  <div className="overflow-x-auto">
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">สถานการณ์</th>
          <th className="border px-4 py-2">กลยุทธ์ที่แนะนำ</th>
          <th className="border px-4 py-2">ตัวอย่าง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ข้อมูลใหม่มีลักษณะคล้ายของเดิม</td>
          <td className="border px-4 py-2">Full Model Finetuning</td>
          <td className="border px-4 py-2">BERT → Medical BERT</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ข้อมูลใหม่มี domain ต่างไป</td>
          <td className="border px-4 py-2">Freeze encoder, train decoder</td>
          <td className="border px-4 py-2">Vision → Text-to-Image</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">มีข้อมูลใหม่น้อย</td>
          <td className="border px-4 py-2">Prompt Tuning / Few-shot</td>
          <td className="border px-4 py-2">GPT-4 → Legal advice</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div className="bg-blue-700 text-black p-4 rounded-lg border-l-4 border-blue-400 shadow-md my-6">
    <h4 className="font-semibold mb-2">Insight Box: ไม่ใช่ทุก Task จะได้ผลจาก Transfer</h4>
    <p className="text-sm leading-relaxed">
      บาง task เช่น symbolic reasoning หรืองานที่ใช้ตรรกะเชิง symbolic มากๆ อาจไม่เห็นประโยชน์ชัดเจนจาก transfer learning โดยเฉพาะเมื่อ model เดิมถูก pretrain ด้วยข้อมูลแบบ end-to-end ที่ไม่สามารถถ่ายทอด reasoning ได้โดยตรง
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc pl-6 space-y-2 text-sm">
    <li>Pan, S.J. and Yang, Q., 2010. A survey on transfer learning. IEEE Transactions on knowledge and data engineering.</li>
    <li>Ruder, S., 2019. Neural Transfer Learning for Natural Language Processing. Thesis, National University of Ireland.</li>
    <li>Howard, J. and Gugger, S., 2018. Universal Language Model Fine-tuning for Text Classification. ACL.</li>
  </ul>
</section>


<section id="research-spotlight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">14. Research Spotlight</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img15} />
  </div>

  <div className="prose lg:prose-lg max-w-none dark:prose-invert">
    <h3 className="text-xl font-semibold">งานวิจัยที่เปลี่ยนแนวทางของ Transfer Learning</h3>
    <p>
      Transfer Learning ได้กลายเป็นรากฐานสำคัญของงานวิจัย AI สมัยใหม่ โดยเฉพาะเมื่อโมเดลมีขนาดใหญ่และสามารถเรียนรู้ representation ที่หลากหลายเพียงพอ ทำให้เกิดความสามารถแบบ generalization ที่เหนือกว่าวิธีเดิมที่ต้องฝึกใหม่ในแต่ละ task
    </p>

    <h3 className="text-xl font-semibold">1. BERT: Pretraining for Language Understanding</h3>
    <p>
      โมเดล <strong>BERT</strong> (Bidirectional Encoder Representations from Transformers) โดย Google (2018) ได้เปลี่ยนแนวทางของการประมวลผลภาษาธรรมชาติอย่างสิ้นเชิง ด้วยกลไก Masked Language Modeling และ Next Sentence Prediction ทำให้โมเดลสามารถเรียนรู้ context ได้ในลักษณะ bidirectional อย่างแท้จริง
    </p>

    <div className="bg-yellow-100 text-black p-4 rounded-lg border-l-4 border-yellow-500 my-6">
      <p className="font-semibold">Highlight:</p>
      <p>
        BERT ทำให้เกิด paradigm ใหม่ของ "pretrain แล้ว finetune" ที่สามารถนำโมเดลเดียวไปใช้กับงาน NLP ได้หลากหลาย เช่น classification, NER, QA และ sentiment analysis
      </p>
    </div>

    <h3 className="text-xl font-semibold">2. SimCLR & MoCo: Pretraining สำหรับ Vision</h3>
    <p>
      ในสายงาน Computer Vision แนวคิด contrastive learning จากงาน <strong>SimCLR</strong> (Google Brain, 2020) และ <strong>MoCo</strong> (Facebook AI Research) ได้รับความสนใจอย่างมาก เพราะสามารถ pretrain encoder โดยไม่ต้องใช้ label ใด ๆ และให้ผลลัพธ์ที่ใกล้เคียงหรือดีกว่าการใช้ supervised learning
    </p>

    <h3 className="text-xl font-semibold">3. GPT Series: จาก Pretraining สู่ Zero/Few-Shot Learning</h3>
    <p>
      โมเดล GPT-2 และ GPT-3 จาก OpenAI นำเสนอแนวคิดใหม่คือการใช้ pretraining อย่างเดียว แล้วนำมาใช้งานกับ task ใหม่ได้โดยตรงผ่านการ prompt โดยไม่ต้อง finetune เลย นี่เป็นจุดเริ่มต้นของยุค "Foundation Models" อย่างแท้จริง
    </p>

    <div className="bg-blue-100 text-black p-4 rounded-lg border-l-4 border-blue-500 my-6">
      <p className="font-semibold">Insight:</p>
      <p>
        GPT-3 ที่มีพารามิเตอร์ 175B ได้แสดงให้เห็นว่าโมเดลที่ pretrain ดีพอสามารถทำ reasoning, summarization, translation และอีกมากมาย เพียงแค่เปลี่ยน prompt
      </p>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบความก้าวหน้าตามช่วงเวลา</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-left border border-gray-300 dark:border-gray-700 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-3 py-2">Year</th>
            <th className="border px-3 py-2">Model</th>
            <th className="border px-3 py-2">Domain</th>
            <th className="border px-3 py-2">Impact</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-3 py-2">2018</td>
            <td className="border px-3 py-2">BERT</td>
            <td className="border px-3 py-2">NLP</td>
            <td className="border px-3 py-2">Benchmark disruption, finetune paradigm</td>
          </tr>
          <tr>
            <td className="border px-3 py-2">2020</td>
            <td className="border px-3 py-2">SimCLR</td>
            <td className="border px-3 py-2">Vision</td>
            <td className="border px-3 py-2">Self-supervised revolution</td>
          </tr>
          <tr>
            <td className="border px-3 py-2">2020</td>
            <td className="border px-3 py-2">GPT-3</td>
            <td className="border px-3 py-2">NLP</td>
            <td className="border px-3 py-2">Few-shot & prompt learning</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-2 text-sm">
      <li>Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805</li>
      <li>Chen, T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. arXiv:2002.05709</li>
      <li>Brown, T. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165</li>
      <li>He, K. et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. arXiv:1911.05722</li>
    </ul>
  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day35 theme={theme} />
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
        <ScrollSpy_Ai_Day35 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day35_TransferLearning;
