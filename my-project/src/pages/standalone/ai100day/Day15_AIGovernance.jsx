import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day15 from "./scrollspy/ScrollSpy_Ai_Day15";
import MiniQuiz_Day15 from "./miniquiz/MiniQuiz_Day15";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day15_AIGovernance = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("AIGovernance1").format("auto").quality("auto").resize(scale().width(600));
  const img2 = cld.image("AIGovernance2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("AIGovernance3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("AIGovernance4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("AIGovernance5").format("auto").quality("auto").resize(scale().width(550));
  const img6 = cld.image("AIGovernance6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("AIGovernance7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("AIGovernance8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("AIGovernance9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("AIGovernance10").format("auto").quality("auto").resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 15: AI Governance & Risk Management</h1>

        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    บทนำ: ทำไม Governance ถึงจำเป็นในยุค AI-first
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-4">
    <p>
      เมื่อปัญญาประดิษฐ์กลายเป็นส่วนหนึ่งของทุกฟังก์ชันในองค์กร ไม่ว่าจะเป็นการแนะนำสินค้า การอนุมัติสินเชื่อ หรือการกลั่นกรองเนื้อหา บทบาทของ AI กำลังเปลี่ยนจากเพียงเครื่องมือไปสู่การเป็นผู้มีอิทธิพลต่อการตัดสินใจเชิงกลยุทธ์โดยตรง ในบริบทนี้ คำว่า "AI Governance" ไม่ใช่เพียงการกำกับการใช้งานเท่านั้น แต่หมายถึงการกำหนดกรอบคิดเชิงระบบ ที่ให้ทั้งความโปร่งใส ความรับผิดชอบ และการควบคุมความเสี่ยงได้แบบองค์รวม
    </p>

    <p>
      การออกแบบระบบ AI โดยไม่มี Governance ที่แข็งแรงเปรียบได้กับการสร้างอัลกอริธึมในสุญญากาศ—แม้จะมีความแม่นยำสูง แต่ขาดบริบท ขาดการตอบสนองต่อผลกระทบในโลกจริง และเสี่ยงต่อการสร้างอคติหรือความเสียหายโดยไม่ตั้งใจ การทำให้ระบบ AI ปลอดภัย โปร่งใส และเป็นธรรม จึงไม่ใช่ความหรูหรา แต่คือ Core Requirement สำหรับองค์กรที่ต้องการรักษาความน่าเชื่อถือในระยะยาว
    </p>

    <p>
      ผู้บริหารระดับสูงในองค์กรเทคโนโลยีขนาดใหญ่ เช่น Google, Microsoft และ Meta ต่างกำหนดนโยบาย AI Governance ไว้อย่างเข้มงวด โดยเริ่มตั้งแต่หลักการออกแบบโมเดลไปจนถึงกระบวนการ monitoring หลัง deploy จริง นโยบายเหล่านี้ไม่เพียงเพื่อป้องกันผลกระทบต่อผู้ใช้เท่านั้น แต่ยังช่วยสร้าง competitive advantage ผ่านการสร้างความไว้วางใจจากผู้ใช้และหน่วยงานกำกับดูแล
    </p>

    <div className="grid md:grid-cols-2 gap-6 mt-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-3">ปัญหาที่เกิดขึ้นเมื่อขาด Governance</h3>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>AI ตัดสินใจผิดพลาดในประเด็นที่มีผลกระทบสูง เช่น การแพทย์หรือการเงิน</li>
          <li>การละเมิดข้อมูลส่วนบุคคลโดยไม่ได้ตั้งใจ</li>
          <li>โมเดลมี bias แฝงในระดับที่มองไม่เห็นด้วย accuracy เพียงอย่างเดียว</li>
          <li>เกิดผลกระทบต่อชื่อเสียงองค์กรเมื่อเกิดเหตุการณ์ที่ไม่คาดคิด</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-3">สัญญาณว่าองค์กรต้องการ Governance Framework</h3>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>AI ถูกใช้ในกระบวนการตัดสินใจเชิงธุรกิจที่มีผลกระทบสูง</li>
          <li>มีโมเดลหลายชุดอยู่ในการดูแลของทีมต่าง ๆ โดยขาดมาตรฐานร่วม</li>
          <li>ขาดระบบ logging และ audit trail ที่สามารถตรวจสอบย้อนหลังได้</li>
          <li>ไม่สามารถตอบคำถามจากฝ่ายกฎหมายหรือหน่วยงานกำกับได้อย่างโปร่งใส</li>
        </ul>
      </div>
    </div>

    <p>
      การมี AI Governance ที่ดีไม่ใช่การควบคุมเพื่อจำกัดการพัฒนา แต่คือการสร้างรากฐานให้สามารถ scale ได้อย่างมั่นคง มีความพร้อมในการรับมือกับความเสี่ยง และสามารถปรับตัวต่อกฎระเบียบที่เปลี่ยนแปลงไปในอนาคต
    </p>

    <p>
      ในระดับองค์กร การวาง AI Governance ที่ดีควรตั้งอยู่บน 3 แกนหลัก: Policy, Ownership และ Monitoring ซึ่งทุกฝ่ายตั้งแต่ Data Scientist, Legal, Product Manager จนถึงผู้บริหารระดับ C ต้องมีบทบาทร่วมกันอย่างเป็นระบบ ไม่ใช่แค่ทำตาม checklist แต่ต้องสร้างวัฒนธรรมที่ AI ไม่ใช่เพียงระบบอัตโนมัติ แต่คือระบบที่ต้อง "รับผิดชอบได้"
    </p>

    <p>
      ความสามารถในการบริหารความเสี่ยงด้าน AI จึงเป็นหนึ่งในตัวชี้วัดที่สะท้อน maturity ขององค์กรด้านเทคโนโลยีในยุคใหม่ ความเชื่อมั่นของลูกค้า ความพร้อมต่อกฎหมาย เช่น EU AI Act และความสามารถในการสร้างโมเดลที่ไม่เพียงฉลาด แต่ยังไว้วางใจได้ คือเป้าหมายสูงสุดขององค์กรที่เข้าสู่ยุค AI-first อย่างแท้จริง
    </p>
  </div>
</section>


<section id="definition" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">AI Governance คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  
  <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700 max-w-3xl mx-auto">
    <p className="mb-4 leading-relaxed">
      AI Governance คือกระบวนการวางกรอบการตัดสินใจและมาตรฐานในการออกแบบ พัฒนา และใช้งานระบบปัญญาประดิษฐ์อย่างเป็นระบบ โดยมีเป้าหมายเพื่อควบคุมให้ระบบ AI ทำงานอย่างมีประสิทธิภาพ โปร่งใส ตรวจสอบได้ ปลอดภัย ยุติธรรม และสอดคล้องกับค่านิยมขององค์กรและสังคม
    </p>
    
    <p className="mb-4 leading-relaxed">
      แนวคิดนี้ไม่ได้จำกัดอยู่แค่การควบคุมทางเทคนิค แต่ยังรวมถึงโครงสร้างองค์กร การกำกับดูแลภายใน ความรับผิดชอบในระดับนโยบาย และกระบวนการจัดการความเสี่ยงในตลอดวงจรชีวิตของ AI ตั้งแต่การออกแบบ (Design) ไปจนถึงการเลิกใช้งาน (Retirement)
    </p>

    <h3 className="text-xl font-semibold mt-6 mb-2">องค์ประกอบหลักของ AI Governance</h3>
    <ul className="list-disc pl-6 space-y-2 mb-6 text-sm">
      <li><strong>Policy & Strategy:</strong> นโยบายการใช้ AI ที่สะท้อนจริยธรรม เป้าหมายองค์กร และข้อกำหนดของภาครัฐ</li>
      <li><strong>Accountability:</strong> ระบุความรับผิดชอบในแต่ละระดับ ตั้งแต่ Developer, Data Owner, ไปจนถึง Executive</li>
      <li><strong>Oversight:</strong> การตรวจสอบภายในและภายนอก เช่น AI Review Board หรือการ Audit อิสระ</li>
      <li><strong>Transparency:</strong> การให้ข้อมูลผู้ใช้และภายในองค์กรว่าระบบทำงานอย่างไร (Model Card, Decision Log)</li>
      <li><strong>Compliance:</strong> การปฏิบัติตามข้อกำหนด เช่น GDPR, EU AI Act, ISO/IEC 23894</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 mb-2">การจัดการ AI Lifecycle</h3>
    <p className="mb-4 leading-relaxed">
      AI Governance ต้องครอบคลุมการจัดการความเสี่ยงและการตัดสินใจในทุกช่วงของ AI Lifecycle:
    </p>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-xl border dark:border-gray-700 text-sm">
        <h4 className="font-medium mb-2">1. Design & Development</h4>
        <ul className="list-disc pl-5 space-y-1">
          <li>กำหนด Target Outcomes และ Metrics อย่างชัดเจน</li>
          <li>ทำ Data Risk Assessment และจัดกลุ่มความเสี่ยงล่วงหน้า</li>
          <li>เลือก Algorithm/Architecture ให้เหมาะสมกับบริบท</li>
        </ul>
      </div>
      <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-xl border dark:border-gray-700 text-sm">
        <h4 className="font-medium mb-2">2. Testing & Deployment</h4>
        <ul className="list-disc pl-5 space-y-1">
          <li>วิเคราะห์ fairness, bias, drift, และ robustness</li>
          <li>ทำ Approval Workflow ก่อน Deployment</li>
          <li>จัดทำ Pre-launch Audit และ Stress Testing</li>
        </ul>
      </div>
      <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-xl border dark:border-gray-700 text-sm">
        <h4 className="font-medium mb-2">3. Monitoring & Control</h4>
        <ul className="list-disc pl-5 space-y-1">
          <li>ตั้งค่าระบบ Drift Detection และ Alert Automation</li>
          <li>สร้าง Dashboard ติดตาม Performance & Fairness</li>
          <li>เก็บ Decision Trace Logs และ Feedback Loops</li>
        </ul>
      </div>
      <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-xl border dark:border-gray-700 text-sm">
        <h4 className="font-medium mb-2">4. Decommissioning</h4>
        <ul className="list-disc pl-5 space-y-1">
          <li>วางแผน Exit Strategy เมื่อ AI ไม่เหมาะสมแล้ว</li>
          <li>ลบ/ล้างข้อมูลส่วนบุคคลตามนโยบายความเป็นส่วนตัว</li>
          <li>เก็บ Audit Trail เพื่อใช้ในการตรวจสอบภายหลัง</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-2">ความสำคัญระดับองค์กร</h3>
    <p className="mb-4 leading-relaxed">
      ในองค์กรระดับโลก เช่น Google, Microsoft, Meta และ Apple มีการตั้งทีม Responsible AI โดยเฉพาะ เพื่อสร้างมาตรฐานด้านความยุติธรรม ความปลอดภัย และความโปร่งใสให้กับระบบ AI
    </p>
    <p className="mb-4 leading-relaxed">
      CTO และ CISO จะต้องวางโครงสร้างให้ระบบ AI สามารถควบคุมและตรวจสอบได้จากมุมมองทั้ง Technology, Compliance และ Business Continuity โดยเฉพาะในบริบทที่มีความเสี่ยงสูง เช่น การแพทย์ การเงิน การทหาร และการขนส่ง
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600 mt-8">
      <p className="font-medium mb-2">Insight จากระดับ CTO:</p>
      <p className="text-sm">
        AI Governance ไม่ใช่แค่การป้องกันความเสี่ยง แต่คือการสร้างระบบที่เติบโตได้อย่างยั่งยืนภายใต้กรอบที่ตรวจสอบได้ เชื่อถือได้ และพร้อมเข้าสู่ Production ด้วยความมั่นใจในระดับองค์กร
      </p>
    </div>
  </div>
</section>

<section id="risks" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    ประเภทของความเสี่ยงจากระบบ AI
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <p className="mb-4 leading-relaxed">
    การพัฒนาระบบ AI ในระดับองค์กร โดยเฉพาะระดับ enterprise หรือเทคโนโลยีขนาดใหญ่ที่มีการนำไปใช้งานจริง จำเป็นต้องเข้าใจความเสี่ยงเชิงลึกจากหลายมิติ โดยแบ่งออกเป็นกลุ่มความเสี่ยงหลักที่มีผลต่อ Operational, Legal, Ethical, และ Technical Stability ของระบบทั้งหมด
  </p>

  <div className="grid md:grid-cols-2 gap-6 mt-6">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow">
      <h3 className="text-lg font-bold mb-3">1. ความเสี่ยงด้านข้อมูล (Data Risk)</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ข้อมูลที่มี Bias เช่นข้อมูลทางเชื้อชาติ เพศ หรือภูมิหลังทางเศรษฐกิจ</li>
        <li>ข้อมูลไม่สมดุล ส่งผลต่อ fairness ของโมเดล</li>
        <li>ข้อมูลฝึกที่ไม่สอดคล้องกับสถานการณ์ปัจจุบัน (Data Drift)</li>
        <li>การละเมิด Privacy เช่น ไม่ได้ขอ Consent จากผู้ให้ข้อมูล</li>
        <li>ไม่สามารถ Audit ข้อมูลย้อนหลังได้ (Lack of Traceability)</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow">
      <h3 className="text-lg font-bold mb-3">2. ความเสี่ยงด้านอัลกอริทึม (Algorithmic Risk)</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>โมเดล Black-box ไม่สามารถอธิบายเหตุผลของผลลัพธ์</li>
        <li>เกิดการเรียนรู้ผิดแบบ systemic เช่นเรียนจาก pattern ที่ไม่ควรเรียน</li>
        <li>Overfitting จาก training data ที่เฉพาะเจาะจง</li>
        <li>Under-generalization ทำให้โมเดลใช้ไม่ได้กับข้อมูลจริง</li>
        <li>เกิด emergent behavior เมื่อใช้โมเดลขนาดใหญ่ เช่น LLMs</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow">
      <h3 className="text-lg font-bold mb-3">3. ความเสี่ยงด้าน Operational และ Deployment</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ไม่มีระบบ monitoring ความเสถียรของโมเดลหลัง deploy</li>
        <li>ความล้มเหลวในการ detect concept drift แบบ real-time</li>
        <li>ไม่มี fallback mechanism หรือ manual override เมื่อ AI พลาด</li>
        <li>Deploy model โดยไม่มีระบบ A/B Testing หรือ Canary Deployment</li>
        <li>การใช้ model pipeline ที่ไม่ reproducible</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow">
      <h3 className="text-lg font-bold mb-3">4. ความเสี่ยงด้านกฎหมายและจริยธรรม (Regulatory & Ethical Risk)</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ไม่สอดคล้องกับ GDPR, EU AI Act หรือมาตรฐานท้องถิ่น</li>
        <li>การตัดสินใจของ AI ไม่มี explainability รองรับ</li>
        <li>ระบบไม่สามารถตอบสนองต่อคำขอ Right to Explanation</li>
        <li>ขาด framework การประเมิน fairness และ non-discrimination</li>
        <li>มีผลกระทบเชิงลบต่อประชาชนหรือความเหลื่อมล้ำ</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-3">ความสัมพันธ์ระหว่างความเสี่ยง</h3>
  <p className="mb-4 leading-relaxed">
    ความเสี่ยงเหล่านี้ไม่ได้แยกขาดจากกัน แต่เชื่อมโยงถึงกันอย่างซับซ้อน เช่น ข้อมูลที่มี Bias อาจนำไปสู่โมเดลที่ไม่เป็นธรรม ซึ่งเมื่อถูก Deploy ก็จะส่งผลต่อภาพลักษณ์และความน่าเชื่อถือขององค์กร ขณะเดียวกันหากไม่มีระบบ Audit ที่เหมาะสมก็จะไม่สามารถย้อนตรวจสอบหรือแก้ไขความเสียหายที่เกิดขึ้นได้ทันเวลา
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="font-semibold mb-2">Insight จากประสบการณ์ในระดับองค์กรขนาดใหญ่:</p>
    <p className="text-sm">
      แทนที่จะพยายามลดทุกความเสี่ยงให้เหลือศูนย์ ควรจัดกลุ่มและวางเกณฑ์การยอมรับความเสี่ยงตามบริบทขององค์กร แล้วทำ mitigation เฉพาะจุดที่ส่งผลกระทบสูง เช่น โปรเจกต์ในสาธารณสุขต้องตรวจสอบ fairness อย่างละเอียด แต่ระบบ recommendation ภายในอาจใช้ monitoring-based approach แทนการ enforce rule-based fairness
    </p>
  </div>
</section>


<section id="assessment" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">การประเมินความเสี่ยงและการจัดระดับความเสี่ยง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในระบบ AI ที่เริ่มมีบทบาทในการตัดสินใจระดับองค์กร การประเมินและจัดการความเสี่ยง (Risk Assessment & Categorization) ไม่ใช่เพียงการคาดการณ์ผลกระทบเชิงเทคนิค แต่ต้องเป็นกระบวนการที่ครอบคลุมผลกระทบต่อสิทธิมนุษยชน ความเป็นส่วนตัว การเงิน ความมั่นคง และการยอมรับของสาธารณชน
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">1. กรอบแนวคิดการประเมินความเสี่ยง (AI Risk Frameworks)</h3>
  <p className="mb-4 leading-relaxed">
    ปัจจุบันมีกรอบแนวคิดที่องค์กรขนาดใหญ่ใช้เป็นมาตรฐาน เช่น:
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>NIST AI Risk Management Framework (RMF)</strong> – มุ่งเน้นการพัฒนา AI อย่างเชื่อถือได้ โดยจัดการความเสี่ยงอย่างเป็นระบบในทุกขั้นตอนของ AI Lifecycle</li>
    <li><strong>ISO/IEC 23894:2023</strong> – มาตรฐานสากลที่กำหนดแนวทางจัดการความเสี่ยง AI ทั้งระดับองค์กรและระบบ</li>
    <li><strong>EU AI Act Classification</strong> – แบ่งความเสี่ยงออกเป็น 4 ระดับ: Unacceptable, High, Limited, Minimal</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-3">2. เทคนิคสำคัญในการประเมินความเสี่ยง</h3>
  <div className="grid md:grid-cols-2 gap-6 mb-8">
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border shadow">
      <h4 className="font-semibold mb-2">Impact-Likelihood Matrix</h4>
      <p className="text-sm leading-relaxed">
        ประเมินระดับความเสี่ยงจาก 2 มิติหลัก คือ ความรุนแรงของผลกระทบ (Impact) และโอกาสเกิด (Likelihood)
        เพื่อจัดลำดับความสำคัญของระบบ/ฟีเจอร์ที่จะ deploy
      </p>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border shadow">
      <h4 className="font-semibold mb-2">Red Flag List</h4>
      <p className="text-sm leading-relaxed">
        รายการปัจจัยที่อาจส่งผลให้โมเดลเสี่ยงสูง เช่น การตัดสินใจโดยไม่มี human-in-the-loop, ใช้ข้อมูลที่มีอคติ, หรือระบบที่เรียนรู้แบบต่อเนื่องโดยไม่มี oversight
      </p>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border shadow">
      <h4 className="font-semibold mb-2">Human Oversight Matrix</h4>
      <p className="text-sm leading-relaxed">
        กำหนดระดับการควบคุมมนุษย์ที่เหมาะสมกับความเสี่ยง เช่น Pre-approval, On-demand Override หรือ Monitoring-only
      </p>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border shadow">
      <h4 className="font-semibold mb-2">Use-case Filtering</h4>
      <p className="text-sm leading-relaxed">
        คัดกรอง use case ที่ไม่ควรใช้ AI โดยอัตโนมัติ เช่น การแปลความหมายทางจิตวิทยา การจำแนกเพศจากเสียง หรือการให้เครดิตทางการเงินจากภาพถ่าย
      </p>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-6 mb-3">3. การจัดระดับความเสี่ยง (Risk Tiering)</h3>
  <p className="mb-4 leading-relaxed">
    หลังจากประเมินความเสี่ยง ระบบจะถูกจัดอยู่ในระดับต่าง ๆ เพื่อกำหนดมาตรการกำกับดูแลให้เหมาะสม เช่น:
  </p>
  <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700 mb-6">
    <thead className="bg-gray-100 dark:bg-gray-700">
      <tr>
        <th className="border px-4 py-2 text-left">ระดับความเสี่ยง</th>
        <th className="border px-4 py-2 text-left">ลักษณะ</th>
        <th className="border px-4 py-2 text-left">ตัวอย่างมาตรการ</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Unacceptable</td>
        <td className="border px-4 py-2">ละเมิดสิทธิขั้นพื้นฐาน หรือไม่สามารถตรวจสอบได้</td>
        <td className="border px-4 py-2">ห้ามพัฒนา / ใช้งาน</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">High-risk</td>
        <td className="border px-4 py-2">มีผลกระทบต่อชีวิต ทรัพย์สิน หรือเสรีภาพ</td>
        <td className="border px-4 py-2">ต้องมี documentation, audit log, human review</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Limited-risk</td>
        <td className="border px-4 py-2">กระทบบริการผู้บริโภค แต่ไม่รุนแรง</td>
        <td className="border px-4 py-2">แจ้งเตือนความเสี่ยง, UX Transparency</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Minimal-risk</td>
        <td className="border px-4 py-2">ไม่เกี่ยวข้องกับสิทธิเสรีภาพ</td>
        <td className="border px-4 py-2">ไม่ต้องกำกับพิเศษ</td>
      </tr>
    </tbody>
  </table>

  <h3 className="text-xl font-semibold mt-6 mb-3">4. การวางระบบประเมินความเสี่ยงในองค์กร</h3>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>ใช้ Risk Register รวมข้อมูลความเสี่ยงจากทุก use case</li>
    <li>วาง Governance Workflow: ใครอนุมัติ, ใครปฏิเสธ, ใครรับผิดชอบ</li>
    <li>ฝัง Risk Checklist ลงใน CI/CD pipeline ของ AI model</li>
    <li>ใช้ Data & Model cards ในการ review ความเสี่ยงก่อน deploy</li>
    <li>สร้างระบบ alert & monitoring สำหรับ use case ที่ผ่านระดับ "High-risk" เท่านั้น</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      การประเมินความเสี่ยงที่ดีควรเป็นระบบที่ฝังอยู่ในทุกขั้นของการพัฒนา AI — ไม่ใช่ตรวจทีหลังเมื่อเกิดปัญหาแล้ว
      การเตรียมข้อมูลความเสี่ยงและการออกแบบ workflow ตั้งแต่แรก คือหัวใจของ AI ที่ปลอดภัย มีความรับผิดชอบ และผ่านเกณฑ์ด้านกฎหมายและจริยธรรมในอนาคต
    </p>
  </div>
</section>

<section id="internal-governance" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    การกำกับดูแลภายในองค์กร & Cross-functional Governance
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในองค์กรระดับโลก การบริหารระบบ AI ไม่สามารถแยกจากบริบทขององค์กรได้อีกต่อไป ต้องสร้างโครงสร้างการกำกับดูแลที่เป็นระบบตั้งแต่ระดับปฏิบัติการไปจนถึงระดับบอร์ด โดยมีทั้งฝ่ายเทคนิค นโยบาย และกฎหมายเข้าร่วมตัดสินใจร่วมกัน เพื่อควบคุมทั้ง Lifecycle ของ AI และความเสี่ยงที่อาจเกิดขึ้นทั้งทางตรงและทางอ้อม
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-4">AI Governance Committee ภายในองค์กร</h3>
  <ul className="list-disc pl-6 space-y-2 mb-6 text-sm">
    <li>จัดตั้งคณะกรรมการเฉพาะกิจด้าน AI โดยมีตัวแทนจาก Engineering, Data, Legal, Compliance, Product</li>
    <li>วางนโยบายอนุมัติ/ระงับการ deploy ระบบ AI ใน production</li>
    <li>ตรวจสอบความเสี่ยงของระบบก่อนออกสู่ตลาดตาม checklist</li>
    <li>กำหนดหลักเกณฑ์ในการเลือกใช้ third-party model หรือ API</li>
    <li>วางเกณฑ์การตรวจสอบเชิงจริยธรรม (Ethical Impact Assessment)</li>
  </ul>

  <h3 className="text-xl font-semibold mt-10 mb-4">บทบาทหลักของหน่วยงานในองค์กร</h3>
  <div className="grid md:grid-cols-2 gap-6">
    <div className="bg-white dark:bg-gray-800 border rounded-xl p-5 shadow text-sm">
      <h4 className="font-semibold mb-2">Data Science / Engineering</h4>
      <ul className="list-disc pl-5 space-y-1">
        <li>พัฒนาและประเมิน model ตามเกณฑ์ fairness, bias, safety</li>
        <li>เตรียม documentation เช่น model cards, datasheets</li>
        <li>ติดตั้ง model monitoring สำหรับ production</li>
        <li>วิเคราะห์การ drift ของข้อมูล</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 border rounded-xl p-5 shadow text-sm">
      <h4 className="font-semibold mb-2">Legal / Compliance</h4>
      <ul className="list-disc pl-5 space-y-1">
        <li>ตรวจสอบการใช้ข้อมูลตาม GDPR / PDPA</li>
        <li>วิเคราะห์ความเสี่ยงด้านความโปร่งใสและ traceability</li>
        <li>ให้คำปรึกษาเรื่องการใช้ AI ในงานที่เสี่ยงสูง (High-risk AI)</li>
        <li>วาง policy สำหรับการ consent และ explainability</li>
      </ul>
    </div>
  </div>

  <div className="grid md:grid-cols-2 gap-6 mt-6">
    <div className="bg-white dark:bg-gray-800 border rounded-xl p-5 shadow text-sm">
      <h4 className="font-semibold mb-2">Product / UX</h4>
      <ul className="list-disc pl-5 space-y-1">
        <li>กำหนดบริบทของผู้ใช้และ feedback loop</li>
        <li>ออกแบบ UI ให้รองรับ explainability & user override</li>
        <li>ระบุกรณีใช้ที่ควรมี human-in-the-loop</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 border rounded-xl p-5 shadow text-sm">
      <h4 className="font-semibold mb-2">Executive / Risk Office</h4>
      <ul className="list-disc pl-5 space-y-1">
        <li>กำหนดระดับความเสี่ยงที่ยอมรับได้ของแต่ละ use case</li>
        <li>ตรวจสอบการใช้ AI สอดคล้องกับ vision และ ESG</li>
        <li>อนุมัติ/ปฏิเสธการนำระบบไปใช้งานจริง</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">Cross-functional Workflow</h3>
  <p className="mb-4 leading-relaxed text-sm">
    การพัฒนาระบบ AI ที่ปลอดภัยและยั่งยืนจำเป็นต้องมี workflow ที่เปิดให้ทุกฝ่ายเข้ามาร่วมตัดสินใจในแต่ละขั้นตอน โดยเฉพาะในงานที่มีความเสี่ยงสูง เช่น ระบบคัดกรอง, ระบบให้คะแนนเครดิต, ระบบแนะนำอัตโนมัติ ซึ่งอาจส่งผลต่อชีวิตผู้ใช้งานหรือภาพลักษณ์องค์กร
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-4">
    <h4 className="font-semibold mb-2">Insight:</h4>
    <p className="text-sm">
      AI Governance ที่ดีไม่ได้เริ่มต้นหลังจากโมเดลพร้อมใช้งานแล้ว แต่เริ่มตั้งแต่ก่อนเริ่มโครงการ ผ่านการวางบทบาท ขอบเขตความรับผิดชอบ และช่องทางตรวจสอบร่วมกัน โดยเฉพาะเมื่อระบบ AI เริ่มเข้าสู่การใช้งานในระดับ real-world
    </p>
  </div>
</section>


<section id="policy-response" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    AI Policy และการตอบสนองเหตุการณ์ผิดปกติ
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="max-w-3xl mx-auto text-base leading-relaxed">
    <p className="mb-4">
      นโยบาย AI ไม่ควรเป็นเพียงเอกสารที่เขียนไว้เพื่อการตรวจสอบเท่านั้น แต่ต้องเป็นแนวทางเชิงระบบที่ฝังอยู่ในวัฒนธรรมการพัฒนา การทดสอบ และการ deploy โมเดล โดยครอบคลุมตั้งแต่ Design Principles, Guardrails, ไปจนถึง Incident Response Protocol
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-3">1. นโยบายพื้นฐานของ AI (AI Policy Foundation)</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>กำหนดว่าโมเดลใดสามารถนำไปใช้ใน Production ได้ ต้องผ่าน Fairness, Safety, Robustness Test</li>
      <li>กำหนดว่าโมเดลใดต้องผ่านการตรวจสอบโดย Human-in-the-loop (เช่นระบบแนะนำยา หรือการตัดสินทางกฎหมาย)</li>
      <li>ระบุหลักจริยธรรมที่ไม่อนุญาตให้ละเมิด เช่น การวิเคราะห์ลักษณะทางพันธุกรรมโดยไม่ได้รับอนุญาต</li>
      <li>ระบุ Critical Use Cases ที่ต้องการการ Review ข้ามแผนก เช่น Legal + Data + Security</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10 mb-3">2. Policy Lifecycle: Policy ต้องปรับตามเทคโนโลยี</h3>
    <p className="mb-4">
      นโยบายต้องถูกรีวิวตามรอบเวลา หรือเมื่อเกิดเหตุการณ์สำคัญ เช่น มีการอัปเดต LLM รุ่นใหม่ หรือเกิดกรณี bias ที่กระทบชื่อเสียง นโยบายต้องไม่แข็งตัว แต่ปรับได้อย่างรวดเร็วแบบ Agile Governance
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-3">3. Incident Response สำหรับระบบ AI</h3>
    <p className="mb-4">
      การตอบสนองเหตุการณ์ผิดปกติไม่ใช่เรื่องของ Security เพียงอย่างเดียว แต่รวมถึง Drift Detection, Unexpected Output, และ Fairness Violation ที่เกิดขึ้นหลังจากโมเดลถูก deploy
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
        <h4 className="font-semibold mb-2 text-lg"> ตัวอย่างเหตุการณ์ที่ควรแจ้งเตือน</h4>
        <ul className="list-disc pl-6 space-y-1 text-sm">
          <li>โมเดลเปลี่ยนพฤติกรรมหลัง deploy (Model Drift)</li>
          <li>ค่าความแม่นยำลดลงอย่างรวดเร็วในกลุ่มผู้ใช้บางกลุ่ม</li>
          <li>โมเดลให้ผลลัพธ์ที่มีลักษณะเลือกปฏิบัติ (Discriminatory Outcome)</li>
          <li>เกิด Bias ใหม่หลังมีการ retrain</li>
          <li>ผู้ใช้รายงานผลลัพธ์ผิดพลาดใน use case ที่ critical</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl p-5 shadow">
        <h4 className="font-semibold mb-2 text-lg"> ขั้นตอน Incident Response</h4>
        <ol className="list-decimal pl-6 space-y-1 text-sm">
          <li>Trigger Alert → ส่งต่อทีม Responsible AI</li>
          <li>Analyze Log และเช็ค Training Data + Input</li>
          <li>ตรวจสอบว่าผิดพลาดมาจาก Model / Infra / Pipeline</li>
          <li>หากมีผลกระทบเชิงจริยธรรมหรือกฎหมาย → แจ้ง Legal & Compliance</li>
          <li>สร้าง Post-mortem และ Update Fairness Constraint ถาวร</li>
        </ol>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10 mb-3">4. ตัวอย่างนโยบายจากองค์กรชั้นนำ</h3>
    <div className="space-y-3 text-sm">
      <p>
        <strong>Google:</strong> มี AI Principles 7 ข้อ และใช้ AI Incident Database ในการเรียนรู้จากกรณีศึกษาเก่า
      </p>
      <p>
        <strong>Microsoft:</strong> ใช้ระบบ Responsible AI Standard Framework ที่บังคับใช้ในทุกทีม product
      </p>
      <p>
        <strong>Facebook (Meta):</strong> ใช้ระบบ "Embedding Fairness into Product Flow" ที่ฝังขั้นตอน fairness ตรวจสอบไว้ใน pipeline product ทุกชั้น
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10 mb-3">5. เทคโนโลยีสนับสนุน Policy & Incident Management</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Feature Flag:</strong> ปิดการทำงานของโมเดลบางส่วนได้ทันทีเมื่อตรวจพบความเสี่ยง</li>
      <li><strong>Audit Trails:</strong> บันทึกเหตุการณ์แบบ Immutable และ Timestamped</li>
      <li><strong>Automated Policy Checkers:</strong> ตรวจสอบ code / model / data ก่อน deploy เทียบกับ Fairness Checklist</li>
      <li><strong>Explainability Dashboards:</strong> แสดงเหตุผลและผลลัพธ์ย้อนหลัง เพื่อใช้ยืนยันเหตุผลของโมเดลได้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10 mb-3">Insight</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow">
      <p className="mb-2">
        นโยบาย AI ไม่ใช่ข้อจำกัด แต่คือเครื่องมือบริหารความเสี่ยงในโลกที่ระบบเรียนรู้ด้วยตนเองและเปลี่ยนแปลงตลอดเวลา
      </p>
      <p>
        ทีมที่เข้าใจว่า "Policy = Safety Architecture" จะสามารถ deploy AI ได้อย่างมั่นใจและไม่ถูก disrupt เมื่อเกิดเหตุการณ์ผิดปกติ
      </p>
    </div>
  </div>
</section>


<section id="audit" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">AI Audit, Logging & Monitoring</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <p className="mb-4 leading-relaxed">
    การทำ AI Audit, Logging และ Monitoring คือการวางระบบเฝ้าระวัง และบันทึกพฤติกรรมของระบบ AI ทั้งในช่วง Training, Deployment และ Post-deployment เพื่อให้สามารถตรวจสอบย้อนหลัง วิเคราะห์ความเสถียร และระบุความผิดปกติที่อาจเกิดขึ้นได้แบบโปร่งใสและตรวจสอบได้ในทุกมิติ
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-4">AI Audit คืออะไร?</h3>
  <p className="mb-4 leading-relaxed">
    AI Audit ไม่ใช่แค่การตรวจสอบผลลัพธ์ แต่เป็นการวิเคราะห์แบบ end-to-end ของกระบวนการออกแบบ พัฒนา และใช้งานระบบ AI ให้เป็นไปตามนโยบายที่ตั้งไว้ เช่น ความยุติธรรม ความโปร่งใส ความเป็นส่วนตัว และความปลอดภัยในการตัดสินใจอัตโนมัติ
  </p>

  <div className="grid md:grid-cols-2 gap-6">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold mb-2">องค์ประกอบหลักของ AI Audit</h4>
      <ul className="list-disc pl-5 space-y-2 text-sm">
        <li>ตรวจสอบข้อมูลต้นทาง (Data Lineage & Quality)</li>
        <li>ทวนกระบวนการ Training และ Hyperparameters</li>
        <li>ตรวจสอบการปรับเปลี่ยนหรือ retrain</li>
        <li>พิจารณาการตัดสินใจของโมเดลต่อกลุ่มเป้าหมายเฉพาะ</li>
        <li>วัดผลด้วย fairness, privacy และ robustness metrics</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold mb-2">เครื่องมือที่ใช้ใน AI Audit</h4>
      <ul className="list-disc pl-5 space-y-2 text-sm">
        <li><strong>Model Cards (Google)</strong> – เอกสาร metadata ที่สื่อสารจุดแข็ง จุดอ่อน และข้อจำกัดของโมเดล</li>
        <li><strong>Datasheets for Datasets (MIT)</strong> – สรุปแหล่งที่มา กระบวนการจัดเก็บ และข้อพิจารณาด้านจริยธรรมของข้อมูล</li>
        <li><strong>AI Incident Tracker</strong> – ระบบบันทึกเหตุการณ์ไม่พึงประสงค์ที่เกิดขึ้นในระบบอัตโนมัติ</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">Logging & Monitoring</h3>
  <p className="mb-4 leading-relaxed">
    ระบบ Logging และ Monitoring ช่วยให้ทีมสามารถสังเกตความเปลี่ยนแปลงของโมเดลที่อาจเกิดจาก drift, input shift หรือ performance degradation ได้แบบ real-time พร้อมแจ้งเตือนเมื่อเกิด anomaly เพื่อป้องกันผลกระทบในเชิงธุรกิจหรือผู้ใช้งาน
  </p>

  <div className="grid md:grid-cols-2 gap-6">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold mb-2">ควร Log อะไรบ้าง</h4>
      <ul className="list-disc pl-5 space-y-2 text-sm">
        <li>Input features ที่เข้าสู่โมเดล</li>
        <li>ผลลัพธ์ของโมเดล (Prediction)</li>
        <li>Confidence score และ uncertainty</li>
        <li>Time stamp และ session metadata</li>
        <li>Contextual factors (เช่น User Segment, Device)</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-semibold mb-2">Monitoring Metrics ที่สำคัญ</h4>
      <ul className="list-disc pl-5 space-y-2 text-sm">
        <li>Prediction Drift: ความเปลี่ยนแปลงของ output distribution</li>
        <li>Data Drift: ความเปลี่ยนแปลงใน input feature distribution</li>
        <li>Latency & Throughput</li>
        <li>Error rate ต่อกลุ่มผู้ใช้</li>
        <li>Feature Importance Shift</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-4">Post-Deployment Testing & Response</h3>
  <p className="mb-4 leading-relaxed">
    การทดสอบภายหลังการ deploy มีบทบาทสำคัญในการระบุปัญหาทางเทคนิคหรือจริยธรรมที่ไม่สามารถตรวจจับได้ในช่วง training เช่น พฤติกรรมเชิง bias เมื่อรับ input จากกลุ่มผู้ใช้จริง หรือโมเดลเปลี่ยนพฤติกรรมเมื่อมี real-time feedback
  </p>

  <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto mb-4">
{`
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

report.run(reference_data=train_df, current_data=live_df)
report.save_html("monitoring-report.html")
`}
  </pre>

  <p className="mb-4 leading-relaxed">
    ตัวอย่างข้างต้นใช้ EvidentlyAI ในการสร้างรายงานวิเคราะห์ความเปลี่ยนแปลงของข้อมูลและ performance หลัง deployment โดยช่วยให้ทีมตรวจสอบแบบ proactive ได้อย่างมีประสิทธิภาพ
  </p>

  <h3 className="text-xl font-semibold mt-10 mb-4 text-center">Best Practice: ทำอย่างไรให้ตรวจสอบย้อนหลังได้จริง</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <ul className="list-disc pl-6 mb-6 space-y-2 text-sm">
    <li>ใช้ระบบ version control สำหรับทั้ง model weights และ datasets</li>
    <li>บันทึก Experiment ID, Git Commit, Environment Tags ร่วมกับ Log</li>
    <li>มี timestamp ที่ชัดเจนในทุก Log และสามารถย้อน trace ได้ถึง Data Source</li>
    <li>กำหนด Threshold และ Alert Rule ที่ผูกกับ Business KPI</li>
    <li>สรุป Log Report รายสัปดาห์ และ Visualize ผ่าน Dashboard</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      การตรวจสอบ AI แบบโปร่งใสไม่ใช่แค่เทคโนโลยี แต่คือ Infrastructure ด้านความไว้วางใจที่จะกลายเป็นมาตรฐานในองค์กรระดับโลก AI ที่ดีไม่ใช่แค่แม่นยำ แต่ต้องตรวจสอบย้อนหลังได้ทุกขั้นตอน ตั้งแต่ Training ไปจนถึง Production
    </p>
  </div>
</section>

<section id="realworld" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    ตัวอย่างจากองค์กรระดับโลก
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <p className="mb-4 leading-relaxed">
    หลายองค์กรระดับโลกได้ปรับตัวรับมือกับความเสี่ยงของ AI ด้วยแนวทางการกำกับดูแลที่เป็นระบบ มีการประยุกต์ใช้ทั้งเครื่องมือเชิงเทคนิคและกระบวนการนโยบายเพื่อสร้างความโปร่งใส ความยุติธรรม และความรับผิดชอบในระบบ AI อย่างแท้จริง โดยเฉพาะองค์กรในกลุ่ม Big Tech, FinTech และหน่วยงานภาครัฐในระดับสากล
  </p>

  <div className="grid md:grid-cols-2 gap-6 mt-8">
    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
      <h3 className="text-xl font-semibold mb-3">Google: Model Cards & Responsible AI</h3>
      <ul className="list-disc pl-5 text-sm space-y-2">
        <li>ออกแบบ <strong>Model Cards</strong> สำหรับทุกโมเดลที่พัฒนา โดยแสดงข้อมูล metadata, ขอบเขตการใช้งาน, performance บนกลุ่มผู้ใช้งานที่แตกต่าง</li>
        <li>สร้าง Responsible AI Practice Guide ที่รวมแนวทางทั้งด้าน fairness, explainability, และ privacy-by-design</li>
        <li>มี AI Principles ที่ถูกฝังอยู่ใน Lifecycle ของผลิตภัณฑ์ทุกตัว และมีทีม AI Ethics ทำงานร่วมกับทีม product และ legal</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
      <h3 className="text-xl font-semibold mb-3">Microsoft: Responsible AI Standard</h3>
      <ul className="list-disc pl-5 text-sm space-y-2">
        <li>ออก Responsible AI Standard 2.0 ที่ครอบคลุมการใช้งาน AI ตั้งแต่ design, training, deployment, monitoring</li>
        <li>ตั้ง <strong>Office of Responsible AI (ORA)</strong> ทำหน้าที่ approve/reject โมเดลที่มีความเสี่ยงสูง</li>
        <li>เน้นการทำ <strong>Impact Assessment</strong> และการบันทึก Audit Trail สำหรับ AI ที่ใช้ในระบบจริง</li>
        <li>เผยแพร่ AI Impact Templates เพื่อให้ทุกทีมสามารถประเมินตนเองได้ก่อนขึ้น production</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
      <h3 className="text-xl font-semibold mb-3">Meta: Fairness Flow & Model Monitoring</h3>
      <ul className="list-disc pl-5 text-sm space-y-2">
        <li>พัฒนา <strong>Fairness Flow</strong> เป็นเครื่องมือภายในเพื่อประเมินผลกระทบของโมเดลต่อกลุ่มประชากรต่าง ๆ</li>
        <li>ทำ Group-based Confusion Matrix และเชื่อมต่อกับ live monitoring system</li>
        <li>ระบบแบบ real-time drift detection บน infrastructure เดียวกับ production ML</li>
        <li>เน้นการสร้าง culture ภายในที่ให้ทีมวิศวกรต้องพิจารณา fairness เหมือนกับ performance</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
      <h3 className="text-xl font-semibold mb-3">OECD, EU AI Act & International Guidelines</h3>
      <ul className="list-disc pl-5 text-sm space-y-2">
        <li>OECD แนะนำ <strong>AI Policy Observatory</strong> เพื่อให้รัฐบาลและองค์กรต่าง ๆ สามารถศึกษาแนวปฏิบัติที่เป็นมาตรฐาน</li>
        <li>EU AI Act จัดระดับความเสี่ยงของระบบ AI และกำหนดให้ High-risk system ต้องมี audit log, documentation และ human-in-the-loop</li>
        <li>เน้นแนวคิด <strong>Trustworthy AI</strong> ซึ่งรวมถึง fairness, accountability, robustness, privacy</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-12 mb-4">บทเรียนจากการนำไปใช้จริง</h3>
  <p className="mb-4 leading-relaxed">
    องค์กรที่สามารถขับเคลื่อนการกำกับดูแล AI ได้อย่างมีประสิทธิภาพ มักมีคุณลักษณะร่วมกัน คือ ความร่วมมือข้ามสายงาน (Legal, Data, Engineering, Policy), การวาง framework ให้ตรวจสอบได้, และการติดตาม monitor แบบ continuous ไม่ใช่เพียง one-time check ก่อน deploy เท่านั้น
  </p>

  <ul className="list-disc pl-6 space-y-2 mb-4 text-sm">
    <li>ตั้ง governance body ที่มีอำนาจตัดสินใจในระดับ product lifecycle</li>
    <li>ออกแบบระบบ Audit Trail ตั้งแต่ design phase โดยไม่ต้องรอจนเกิดปัญหา</li>
    <li>ทำ A/B Test เพื่อดูว่า fairness metric เปลี่ยนไปอย่างไรเมื่อ deploy</li>
    <li>ใช้ Feedback Loop จาก real user เพื่อตรวจสอบ unintended consequences</li>
    <li>มีการกำกับร่วมกันระหว่าง compliance, ML engineer, product owner และผู้เชี่ยวชาญด้าน ethics</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <h4 className="font-semibold mb-2">Insight:</h4>
    <p className="leading-relaxed">
      การทำ AI Governance ที่ยั่งยืนไม่ได้อยู่แค่ที่เครื่องมือหรือกฎระเบียบ แต่ต้องเกิดจากวัฒนธรรมภายในองค์กรที่เคารพผลกระทบต่อผู้ใช้งานจริง และออกแบบระบบให้ตรวจสอบและอธิบายได้ตั้งแต่ต้นทางถึงปลายทาง
    </p>
  </div>
</section>


<section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">สรุป: จาก Fairness สู่ Governance</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="space-y-6 text-base leading-relaxed">
    <p>
      การเดินทางจากแนวคิด Fairness ใน AI ไปสู่ระบบ AI Governance ไม่ใช่แค่การเปลี่ยนมุมมองทางเทคนิค แต่คือการยกระดับแนวคิดไปสู่ระดับองค์กรและระบบนิเวศ โดยเฉพาะเมื่อระบบ AI มีบทบาทในการตัดสินใจที่มีผลกระทบต่อชีวิตผู้คนและธุรกิจขนาดใหญ่ การมี Fairness เพียงอย่างเดียวอาจไม่เพียงพอ หากปราศจากระบบควบคุมที่ยั่งยืน ครอบคลุม และตรวจสอบได้
    </p>

    <p>
      แนวคิดของ Governance ในระดับ CTO หรือ Tech Leadership ไม่ได้หมายถึงการควบคุมแบบรวมศูนย์ แต่คือการสร้าง "สัญญาร่วม" ระหว่างทีมวิศวกรรม ข้อมูล กฎหมาย และผู้มีส่วนเกี่ยวข้องทั้งหมด เพื่อให้มั่นใจว่าระบบ AI ทำงานตามเป้าหมายที่ชัดเจน โปร่งใส และปลอดภัยทั้งในระยะสั้นและระยะยาว
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-gray-200 dark:border-gray-700 shadow">
        <h3 className="font-semibold mb-2 text-lg">การเปลี่ยนมุมมอง: Fairness → Governance</h3>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>Fairness คือจุดเริ่มของ Ethical AI</li>
          <li>Governance คือระบบควบคุมที่ทำให้ Fairness ดำรงอยู่ได้</li>
          <li>เป้าหมายคือ "AI ที่เชื่อถือได้ในระดับองค์กร"</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-gray-200 dark:border-gray-700 shadow">
        <h3 className="font-semibold mb-2 text-lg">บทบาทของ CTO ในการสร้าง Governance</h3>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>สร้าง Vision ที่เชื่อมโยงระหว่างความเร็วทางเทคนิคและความเสถียรขององค์กร</li>
          <li>กำหนด Framework สำหรับ Lifecycle ของ AI: Design → Deploy → Monitor → Improve</li>
          <li>สนับสนุนให้ทีมสร้าง AI ที่อธิบายได้ โปร่งใส และตรวจสอบได้</li>
        </ul>
      </div>
    </div>

    <p>
      การวาง Governance ที่ดีจะต้องมองลึกไปถึงกระบวนการตั้งแต่ต้นน้ำ เช่น การเลือก Data Source การ Preprocess การวาง Objective Function ของโมเดล ไปจนถึงปลายน้ำ เช่น การวางระบบ Logging, Monitoring และ Audit Trail ที่ทำงานเชิงรุกไม่ใช่เชิงรับเพียงอย่างเดียว
    </p>

    <h3 className="text-xl font-semibold mt-10 mb-3">ประเด็นสำคัญจากบทนี้</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Fairness เป็นเพียงองค์ประกอบหนึ่งของ Governance แต่ไม่ใช่ทั้งหมด</li>
      <li>Governance รวมถึงการจัดการความเสี่ยง ความปลอดภัย ความเป็นส่วนตัว และความโปร่งใส</li>
      <li>ต้องมี Framework ที่ชัดเจน เช่น AI Risk Matrix, Oversight Responsibility และ Incident Handling Plan</li>
      <li>การจัดระเบียบในองค์กรให้ทำงานร่วมกันข้ามแผนก เป็นหัวใจของความสำเร็จ</li>
      <li>แนวทางเชิงปฏิบัติ เช่น Model Card, Fairness Report, Audit Log ควรฝังเข้าไปใน workflow ตั้งแต่ต้น</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10 mb-3">กรอบความคิดที่ควรยึดถือในระดับองค์กร</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500">
      <p className="mb-2 font-semibold">"AI ที่ปลอดภัยและยั่งยืน ไม่ได้เริ่มจากเทคโนโลยี แต่เริ่มจากระบบความรับผิดชอบ"</p>
      <p>
        องค์กรที่สามารถ scale AI ได้อย่างยั่งยืนคือองค์กรที่ไม่เพียงเก่งด้านการพัฒนาโมเดล แต่สามารถสื่อสารแนวคิด Governance ไปยังทุกฝ่ายได้อย่างชัดเจน ไม่ว่าจะเป็น Developer, Stakeholder หรือ User
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10 mb-3">แนวทางปฏิบัติเมื่อระบบ AI ขยายขนาด</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ฝัง checkpoint การประเมินความเสี่ยงในทุก stage ของการพัฒนา</li>
      <li>ทำ Incident Simulation (เหมือน Cybersecurity Tabletop Exercise)</li>
      <li>จัดทำเอกสาร Governance Register สำหรับทุก AI Component</li>
      <li>ระบุ owner ที่รับผิดชอบต่อ AI ทุกรุ่นอย่างชัดเจน</li>
      <li>ทำ Retrospective กับทีมเมื่อมีความล้มเหลว เพื่อปรับ Governance</li>
    </ul>

    <div className="bg-green-100 dark:bg-green-800 text-black dark:text-green-100 p-5 rounded-xl border-l-4 border-green-500 shadow">
      <h4 className="font-semibold mb-2">มุมมองจาก Big Tech</h4>
      <p>
        องค์กรระดับ Google, Microsoft, Meta และ IBM ต่างมีทีมเฉพาะด้าน AI Governance โดยใช้เครื่องมืออย่าง Model Cards, Responsible AI Dashboards และมี Policy ที่บังคับใช้ภายใน องค์กรเหล่านี้ไม่เพียงพูดถึง AI ที่ยุติธรรม แต่ลงทุนเพื่อให้สามารถพิสูจน์ความยุติธรรมนั้นได้จริงในระดับ production
      </p>
    </div>

    <p className="mt-10">
      หาก Fairness เปรียบเสมือนเข็มทิศ Governance คือแผนที่และมาตรวัดความปลอดภัยที่จะนำพาองค์กรไปข้างหน้าอย่างมั่นคง ในโลกที่ AI เป็นศูนย์กลางการตัดสินใจ การขาด Governance คือความเสี่ยงเชิงกลยุทธ์ ไม่ใช่แค่ความเสี่ยงเชิงเทคนิค
    </p>
  </div>
</section>
        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day15 theme={theme} />
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
      </main>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day15 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day15_AIGovernance;
