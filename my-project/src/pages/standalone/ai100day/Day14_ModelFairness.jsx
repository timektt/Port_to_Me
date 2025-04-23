// Day14_ModelFairness.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day14 from "./scrollspy/ScrollSpy_Ai_Day14";
import MiniQuiz_Day14 from "./miniquiz/MiniQuiz_Day14";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day14_ModelFairness = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  const img1 = cld.image('Fairness1').format('auto').quality('auto').resize(scale().width(500));
  const img2 = cld.image('Fairness2').format('auto').quality('auto').resize(scale().width(500));
  const img3 = cld.image('Fairness3').format('auto').quality('auto').resize(scale().width(500));
  const img4 = cld.image('Fairness4').format('auto').quality('auto').resize(scale().width(500));
  const img5 = cld.image('Fairness5').format('auto').quality('auto').resize(scale().width(500));
  const img6 = cld.image('Fairness6').format('auto').quality('auto').resize(scale().width(500));
  const img7 = cld.image('Fairness7').format('auto').quality('auto').resize(scale().width(500));
  const img8 = cld.image('Fairness8').format('auto').quality('auto').resize(scale().width(500));
  const img9 = cld.image('Fairness9').format('auto').quality('auto').resize(scale().width(500));
  const img10 = cld.image('Fairness10').format('auto').quality('auto').resize(scale().width(500));
  const img11 = cld.image('Fairness11').format('auto').quality('auto').resize(scale().width(500));
  const img12 = cld.image('Fairness12').format('auto').quality('auto').resize(scale().width(500));
  const img13 = cld.image('Fairness13').format('auto').quality('auto').resize(scale().width(500));
  const img14 = cld.image('Fairness14').format('auto').quality('auto').resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 14: Fairness, Bias & Ethics in AI</h1>

        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">บทนำ: ทำไม Fairness และ Ethics ถึงสำคัญใน AI</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <p className="mb-4 leading-relaxed">
    ปัญญาประดิษฐ์ (AI) ได้กลายเป็นกลไกหลักที่อยู่เบื้องหลังการตัดสินใจในหลายบริบทของชีวิต ตั้งแต่การคัดกรองใบสมัครงาน การให้กู้ยืมเงิน การตรวจสุขภาพ ไปจนถึงการคัดเลือกข่าวสารบนโซเชียลมีเดีย หากระบบเหล่านี้ไม่ยึดหลักความยุติธรรม (Fairness) และจริยธรรม (Ethics) ผลลัพธ์ที่เกิดขึ้นอาจส่งผลกระทบร้ายแรงต่อผู้ใช้งาน สิ่งแวดล้อม และสังคมในวงกว้าง
  </p>

  <p className="mb-4 leading-relaxed">
    ตัวอย่างที่ชัดเจนคือ กรณีที่ระบบให้กู้เงินถูกฝึกด้วยข้อมูลประวัติที่มีอคติทางเพศหรือเชื้อชาติ ทำให้การอนุมัติสินเชื่หอไม่เท่าเทียม หรือกรณีที่โมเดลคัดกรองใบสมัครงานให้คะแนนผู้สมัครชายสูงกว่าหญิงแม้มีคุณสมบัติเหมือนกัน สิ่งเหล่านี้ชี้ให้เห็นว่า หากขาดการออกแบบที่ตระหนักถึง Fairness ระบบ AI อาจเป็นเครื่องมือในการตอกย้ำความไม่เท่าเทียมแทนที่จะช่วยลดช่องว่างเหล่านั้น
  </p>

  <p className="mb-4 leading-relaxed">
    นอกจากนี้ ความกังวลด้าน Ethics ยังครอบคลุมไปถึงประเด็นความโปร่งใส ความรับผิดชอบ และสิทธิของผู้ใช้งาน เช่น AI มีสิทธิ์ใช้ข้อมูลส่วนตัวโดยไม่ได้รับอนุญาตหรือไม่? ใครเป็นผู้รับผิดชอบหาก AI ตัดสินใจผิดพลาด? จะมั่นใจได้อย่างไรว่าระบบเหล่านี้ปลอดภัยและไม่ละเมิดสิทธิมนุษยชน?
  </p>

  <p className="mb-4 leading-relaxed">
    ในโลกความเป็นจริง โมเดล AI ไม่สามารถเรียนรู้ได้โดยปราศจากบริบทของสังคม ข้อมูลที่ใช้ฝึกมักสะท้อนความเหลื่อมล้ำ อคติ หรือความไม่สมดุลในอดีต ดังนั้นการพัฒนา AI ที่ไม่คำนึงถึง Fairness และ Ethics เท่ากับการปล่อยให้ระบบเรียนรู้อคติเหล่านั้น และตอกย้ำมันซ้ำอีกในการใช้งานจริง
  </p>

  <p className="mb-4 leading-relaxed">
    การออกแบบระบบ AI ที่เป็นธรรมจึงไม่ใช่แค่เป้าหมายทางเทคนิค แต่เป็นภารกิจด้านความรับผิดชอบต่อมนุษยชาติ ต้องคิดตั้งแต่ต้นทางของการเก็บข้อมูล การเลือก feature การตั้งค่า loss function ไปจนถึงการประเมินผลลัพธ์และการนำไปใช้งานจริง ทุกขั้นตอนควรมีความระมัดระวังต่อผลกระทบเชิงสังคมและจริยธรรม
  </p>

  <p className="mb-4 leading-relaxed">
    Fairness ไม่สามารถวัดได้จาก Accuracy เพียงอย่างเดียว จำเป็นต้องมีการวิเคราะห์อย่างละเอียดว่าผลลัพธ์ของโมเดลส่งผลกระทบอย่างไรต่อกลุ่มผู้ใช้ที่หลากหลาย มีความแตกต่างเชิงระบบเกิดขึ้นหรือไม่? โมเดลแสดงพฤติกรรมแบบ double standard หรือ bias เชิงโครงสร้างหรือเปล่า? การตอบคำถามเหล่านี้ต้องอาศัยทั้งเทคนิคทางสถิติ เครื่องมือวิเคราะห์ fairness และความเข้าใจในบริบทสังคมวัฒนธรรม
  </p>

  <p className="mb-4 leading-relaxed">
    ด้าน Ethics จำเป็นต้องอาศัยแนวทางที่ครอบคลุม เช่น การกำหนดหลักการของ AI ที่โปร่งใส ตรวจสอบได้ มีความรับผิดชอบ และให้ความเคารพต่อสิทธิและศักดิ์ศรีของมนุษย์ หลายองค์กรเริ่มออกแบบนโยบายที่ชัดเจน เช่น Google AI Principles หรือ EU AI Act ซึ่งเน้นการป้องกันความเสี่ยงเชิงจริยธรรมตั้งแต่ระยะเริ่มต้น
  </p>

  <p className="mb-4 leading-relaxed">
    หน่วยงานกำกับดูแล เช่น สหภาพยุโรป หรือองค์กรวิจัยอย่าง OECD และ IEEE ก็เริ่มออกกฎเกณฑ์ที่สนับสนุนการพัฒนา AI อย่างมีความรับผิดชอบ โดยเฉพาะในงานที่มีผลกระทบสูง เช่น ระบบการแพทย์ ระบบยุติธรรม หรือการให้บริการทางการเงิน ซึ่งไม่สามารถยอมให้เกิด bias หรือ error ได้ง่าย ๆ
  </p>

  <p className="mb-4 leading-relaxed">
    อีกหนึ่งหัวใจของการพัฒนา AI อย่างมีจริยธรรมคือ การสร้าง AI Alignment ซึ่งหมายถึงการทำให้ AI มีเป้าหมายการตัดสินใจที่สอดคล้องกับคุณค่ามนุษย์ เช่น การปกป้องเสรีภาพ การไม่เลือกปฏิบัติ และการเคารพข้อมูลส่วนบุคคล การทำให้ AI ตอบสนองต่อความต้องการมนุษย์ในทางที่ไม่ละเมิดจริยธรรมจึงเป็นโจทย์ที่ต้องออกแบบอย่างรอบคอบ
  </p>

  <p className="mb-4 leading-relaxed">
    ท้ายที่สุด Fairness และ Ethics ไม่ใช่เรื่องของฝ่ายใดฝ่ายหนึ่ง เช่น วิศวกร นักจริยธรรม หรือนักนโยบาย เท่านั้น แต่ต้องเป็นความร่วมมือข้ามสาขา ทั้งจากนักพัฒนา ผู้ใช้งาน ผู้เชี่ยวชาญ และผู้มีส่วนได้เสีย เพื่อให้ AI ที่ออกแบบมาเป็นระบบที่ยุติธรรม น่าเชื่อถือ และมีผลดีต่อสังคมอย่างแท้จริง
  </p>
</section>


<section id="definition" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">ความหมายของ Fairness และ Ethical AI</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

 

  <p className="mb-4 leading-relaxed">
    Fairness ในบริบทของ AI หมายถึงการออกแบบและใช้งานระบบที่ไม่เลือกปฏิบัติต่อกลุ่มใดกลุ่มหนึ่งโดยไม่เป็นธรรม ซึ่งอาจรวมถึงเชื้อชาติ เพศ อายุ ศาสนา สถานะทางเศรษฐกิจ หรือปัจจัยอื่น ๆ ที่อาจก่อให้เกิดความเหลื่อมล้ำหรืออคติในการตัดสินใจของโมเดล AI
  </p>

  <p className="mb-4 leading-relaxed">
    ในขณะที่ AI สามารถเพิ่มประสิทธิภาพในการตัดสินใจและการวิเคราะห์ข้อมูลขนาดใหญ่ได้ดี แต่หากไม่ได้ออกแบบให้คำนึงถึงความเป็นธรรม ระบบอาจกลายเป็นเครื่องมือในการทำให้ความไม่เท่าเทียมที่มีอยู่แล้วรุนแรงขึ้น ยิ่งระบบถูกนำไปใช้งานในบริบทที่มีผลกระทบต่อชีวิตคน เช่น การเงิน การแพทย์ หรือการคัดเลือกบุคลากร การรักษา fairness จึงเป็นปัจจัยสำคัญ
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-8 text-center">Fairness ในทางเทคนิค</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <p className="mb-4 leading-relaxed">
    Fairness ใน AI อาจมีนิยามที่แตกต่างกันตามมุมมองทางสถิติ เช่น:
  </p>
  <ul className="list-disc pl-6 mb-6 space-y-2">
    <li><strong>Demographic Parity:</strong> ระบบควรให้ผลลัพธ์ในอัตราเท่ากันในแต่ละกลุ่ม (เช่น อัตราการให้ผ่านการสมัครงานของเพศหญิงและชายควรใกล้เคียงกัน)</li>
    <li><strong>Equal Opportunity:</strong> อัตราการคาดการณ์ถูก (true positive rate) ควรเท่าเทียมกันในแต่ละกลุ่ม</li>
    <li><strong>Equalized Odds:</strong> ทั้ง true positive และ false positive rate ควรใกล้เคียงกันระหว่างกลุ่ม</li>
    <li><strong>Individual Fairness:</strong> คนที่คล้ายกันควรได้รับผลลัพธ์ที่คล้ายกัน</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">Ethical AI คืออะไร</h3>
  <p className="mb-4 leading-relaxed">
    Ethical AI หมายถึงระบบ AI ที่ออกแบบและพัฒนาโดยยึดหลักจริยธรรม ซึ่งรวมถึงการให้ความสำคัญกับสิทธิมนุษยชน ความโปร่งใส ความเป็นส่วนตัว ความรับผิดชอบ และการหลีกเลี่ยงอคติที่ไม่ยุติธรรม
  </p>

  <p className="mb-4 leading-relaxed">
    จริยธรรมใน AI ไม่ใช่เพียงการเลือกใช้โมเดลที่อธิบายได้ แต่ครอบคลุมถึงกระบวนการทั้งหมด ตั้งแต่การเก็บข้อมูล การฝึกโมเดล การเลือก feature การตัดสินใจของระบบ และผลกระทบที่เกิดขึ้นในชีวิตจริง
  </p>

  <div className="grid md:grid-cols-2 gap-6 my-6">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-medium mb-3">หลักการของ Ethical AI</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li><strong>Transparency:</strong> ระบบควรอธิบายได้ว่าทำไมจึงตัดสินใจเช่นนั้น</li>
        <li><strong>Fairness:</strong> หลีกเลี่ยงการเลือกปฏิบัติอย่างไม่เป็นธรรม</li>
        <li><strong>Accountability:</strong> มีคนรับผิดชอบหากระบบทำงานผิดพลาด</li>
        <li><strong>Privacy:</strong> เคารพความเป็นส่วนตัวของข้อมูลผู้ใช้</li>
        <li><strong>Human-Centric:</strong> ระบบต้องคำนึงถึงผลกระทบต่อมนุษย์เป็นหลัก</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-medium mb-3">ความแตกต่างระหว่าง Fairness กับ Ethics</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>Fairness เป็นเพียงส่วนหนึ่งของ Ethics</li>
        <li>Ethics มีขอบเขตกว้างกว่า ครอบคลุมทั้งวิธีการออกแบบ ผลกระทบ และความรับผิดชอบ</li>
        <li>โมเดลที่มี fairness สูง อาจยังไม่ถือว่าจริยธรรมดี หากละเมิดสิทธิของผู้ใช้</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3 mt-8">ความท้าทายในการนิยาม Fairness</h3>
  <p className="mb-4 leading-relaxed">
    ไม่มีนิยามเดียวที่ใช้ได้ในทุกสถานการณ์ บางครั้งการพยายามให้ demographic parity อาจส่งผลให้เกิด unfair ต่อบางกลุ่มที่ควรได้รับโอกาสจริง ๆ ดังนั้นการกำหนด fairness ต้องอิงกับบริบทของแอปพลิเคชัน และปรึกษาผู้มีส่วนได้ส่วนเสียทุกกลุ่ม
  </p>

  <p className="mb-4 leading-relaxed">
    ตัวอย่างเช่น ในระบบให้กู้เงิน การให้โอกาสเท่ากันสำหรับทุกกลุ่มโดยไม่คำนึงถึงประวัติการเงิน อาจสร้างความเสี่ยงทางการเงิน ในขณะที่หากพิจารณาเฉพาะตัวเลขโดยไม่สนใจบริบท ก็อาจตัดโอกาสผู้ที่มีความสามารถจริง
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>Fairness และ Ethical AI ไม่ใช่แค่เรื่องเทคนิค แต่คือหลักการที่ต้องฝังไว้ในทุกขั้นตอนของการสร้าง AI เพื่อให้เทคโนโลยีรับใช้มนุษย์อย่างยุติธรรมและรับผิดชอบ</p>
  </div>
</section>

<section id="real-case" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">ตัวอย่างกรณีศึกษาในชีวิตจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>


  <p className="mb-4 leading-relaxed">
    การเข้าใจผลกระทบของ Bias และความไม่เป็นธรรมในระบบ AI ต้องอาศัยตัวอย่างที่เกิดขึ้นจริงในโลก เพื่อแสดงให้เห็นว่าแม้จะมีการออกแบบอัลกอริทึมอย่างดีเพียงใด หากละเลยบริบททางสังคมหรือโครงสร้างของข้อมูลที่ป้อนเข้าไป โมเดลก็ยังอาจทำงานในลักษณะที่ไม่ยุติธรรมได้
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">1. ระบบให้กู้เงินอัตโนมัติที่เลือกปฏิบัติตามเพศ</h3>
  <p className="mb-4 leading-relaxed">
    บริษัทเทคโนโลยีแห่งหนึ่งพัฒนาโมเดลให้คะแนนเครดิตผู้กู้โดยอัตโนมัติ โดยอิงจากข้อมูลย้อนหลังหลายปี ซึ่งชุดข้อมูลนั้นสะท้อนพฤติกรรมในอดีตที่ผู้ชายมักได้รับวงเงินสูงกว่า เมื่อโมเดลนำข้อมูลนี้ไปเรียนรู้ จึงกลายเป็นว่าผู้หญิงแม้จะมีคุณสมบัติเหมือนกัน กลับได้รับวงเงินน้อยกว่าโดยอัตโนมัติ
  </p>
  <p className="mb-4 leading-relaxed">
    เมื่อมีการร้องเรียน พบว่าแม้โมเดลจะไม่มีฟีเจอร์ "เพศ" โดยตรง แต่มีฟีเจอร์ทางอ้อมที่สามารถระบุเพศได้ เช่น อาชีพ ประวัติการชำระเงิน หรือแม้แต่ที่อยู่ ซึ่งกลายเป็น proxy ที่สะท้อนอคติโดยไม่รู้ตัว
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">2. ระบบตรวจสุขภาพจากภาพถ่ายที่เรียนรู้จากข้อมูลผิดพลาด</h3>
  <p className="mb-4 leading-relaxed">
    โรงพยาบาลแห่งหนึ่งใช้โมเดล Deep Learning วิเคราะห์ภาพถ่าย X-ray เพื่อตรวจหาปอดบวม (Pneumonia) โดยเริ่มจากชุดข้อมูลที่ถ่ายจากหลายโรงพยาบาลในระบบสุขภาพเดียวกัน แต่เมื่อทดลองใช้กับภาพจากโรงพยาบาลภายนอก ผลลัพธ์กลับแย่ลงอย่างชัดเจน
  </p>
  <p className="mb-4 leading-relaxed">
    หลังตรวจสอบพบว่าโมเดลไม่ได้เรียนรู้ความผิดปกติจากปอดจริง แต่เรียนรู้ว่าโรงพยาบาลใดมักจะมีผู้ป่วยปอดบวมมาก จาก text mark หรือเครื่องมือที่อยู่ในภาพ ส่งผลให้โมเดลตัดสินใจจาก "ที่มา" ของภาพ มากกว่าเนื้อหาภายในภาพ
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">3. ระบบคัดกรองใบสมัครงานที่ลดคะแนนผู้หญิง</h3>
  <p className="mb-4 leading-relaxed">
    บริษัทด้านเทคโนโลยีรายใหญ่ใช้โมเดลเพื่อวิเคราะห์ประวัติย่อของผู้สมัครงาน และให้คะแนนว่าใบสมัครใดมีแนวโน้มจะประสบความสำเร็จในการสัมภาษณ์ โดยอิงจากประวัติพนักงานเดิม
  </p>
  <p className="mb-4 leading-relaxed">
    โมเดลเรียนรู้จากชุดข้อมูลที่มีพนักงานชายจำนวนมากในตำแหน่งเทคนิค ส่งผลให้ใบสมัครที่มีคำว่า “women’s college” หรือกิจกรรมที่เกี่ยวข้องกับเพศหญิงได้รับคะแนนต่ำลงโดยอัตโนมัติ แม้จะไม่มีฟีเจอร์ “เพศ” อยู่ในโมเดลก็ตาม
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">4. ระบบแนะนำการรักษาที่ไม่แม่นยำสำหรับกลุ่มชาติพันธุ์</h3>
  <p className="mb-4 leading-relaxed">
    ในสหรัฐอเมริกา มีการพบว่าโมเดล AI ที่ใช้จัดลำดับความสำคัญในการเข้าถึงบริการสุขภาพมักประเมินว่าผู้ป่วยผิวดำมีความจำเป็นเร่งด่วนน้อยกว่าผู้ป่วยผิวขาว แม้จะมีอาการใกล้เคียงกัน
  </p>
  <p className="mb-4 leading-relaxed">
    เหตุผลคือโมเดลใช้ข้อมูลย้อนหลังเกี่ยวกับ “ค่าใช้จ่ายด้านการรักษา” เป็นตัวแปรในการพยากรณ์ ซึ่งในอดีต กลุ่มผู้ป่วยผิวดำมักเข้าถึงบริการน้อยกว่า ทำให้ค่าใช้จ่ายต่ำ โมเดลจึงเรียนรู้ผิด ๆ ว่าความรุนแรงของโรคนั้นต่ำตามไปด้วย
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">5. ระบบวิเคราะห์ความเสี่ยงทางอาญาที่ทำให้คนผิวดำถูกจัดอยู่ในความเสี่ยงสูง</h3>
  <p className="mb-4 leading-relaxed">
    ระบบ COMPAS ซึ่งใช้ในบางรัฐของสหรัฐ ถูกใช้เพื่อประเมินว่าผู้ต้องสงสัยจะมีแนวโน้มก่อเหตุซ้ำหรือไม่ในอนาคต โดยใช้เป็นส่วนหนึ่งของการตัดสินใจประกันตัวหรือกำหนดโทษ
  </p>
  <p className="mb-4 leading-relaxed">
    การศึกษาจาก ProPublica พบว่า โมเดลดังกล่าวมีความเอนเอียง โดยให้คะแนนความเสี่ยงสูงกับผู้ต้องสงสัยผิวดำ แม้จะไม่ได้ก่อเหตุซ้ำ และให้คะแนนความเสี่ยงต่ำกับผู้ต้องสงสัยผิวขาวที่มีพฤติกรรมรุนแรงมากกว่า
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">6. ระบบตรวจสอบใบหน้าไม่แม่นยำในกลุ่มผิวเข้ม</h3>
  <p className="mb-4 leading-relaxed">
    หลายบริษัทพัฒนาเทคโนโลยีจดจำใบหน้า (Facial Recognition) และอ้างว่ามี accuracy สูงกว่า 95% แต่เมื่อทดสอบกับคนผิวดำหรือผู้หญิงที่มีผิวเข้ม ค่า precision ลดลงอย่างชัดเจน บางกรณีต่ำกว่า 80%
  </p>
  <p className="mb-4 leading-relaxed">
    เหตุผลหนึ่งคือ training data ของโมเดลมักประกอบด้วยใบหน้าของผู้ชายผิวขาวเป็นหลัก ซึ่งทำให้โมเดล generalize ไม่ได้เมื่อใช้ในกลุ่มอื่น ๆ และอาจนำไปสู่การจับผิดตัว หรือการระบุผิดพลาดในระบบรักษาความปลอดภัย
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <strong className="block mb-2">Insight:</strong>
    <p className="mb-2">Bias ที่ฝังอยู่ในข้อมูลหรือในระบบ AI แม้เพียงเล็กน้อย สามารถขยายผลลัพธ์อย่างผิดพลาดในระดับที่ส่งผลกระทบต่อชีวิต ความยุติธรรม และโอกาสของผู้คนได้</p>
    <p className="mb-2">การตรวจสอบ วิเคราะห์ และออกแบบเพื่อป้องกัน bias ไม่ใช่เพียงงานวิศวกรรม แต่เป็นความรับผิดชอบทางสังคมของทุกทีมที่พัฒนา AI</p>
  </div>
</section>
<section id="impact" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">ผลกระทบเมื่อระบบ AI มี Bias หรือไม่เป็นธรรม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <p className="mb-4 leading-relaxed">
    เมื่อระบบ AI มีอคติ (Bias) หรือไม่สามารถตัดสินใจอย่างเป็นธรรม ผลกระทบที่เกิดขึ้นอาจขยายวงกว้างกว่าที่คาดคิด โดยเฉพาะอย่างยิ่งเมื่อระบบเหล่านั้นถูกใช้งานในพื้นที่ที่มีผลต่อชีวิตมนุษย์โดยตรง เช่น การเงิน การแพทย์ การยุติธรรม หรือแม้แต่การจ้างงาน
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">1. การเลือกปฏิบัติ (Discrimination)</h3>
  <p className="mb-4 leading-relaxed">
    ระบบ AI ที่เรียนรู้จากข้อมูลที่มี bias อาจนำไปสู่การเลือกปฏิบัติ เช่น ระบบพิจารณาสินเชื่ออาจปฏิเสธคำขอจากกลุ่มผู้หญิงหรือคนผิวสีมากกว่ากลุ่มอื่น แม้ว่าจะมีคุณสมบัติใกล้เคียงกันในด้านเศรษฐกิจหรือความเสี่ยง
  </p>
  <p className="mb-4 leading-relaxed">
    ปัญหานี้มักเกิดจาก historical bias ที่ฝังอยู่ในข้อมูลการอนุมัติสินเชื่อในอดีต ซึ่ง AI นำมาใช้เป็นต้นแบบการเรียนรู้ และสะท้อนพฤติกรรมที่ไม่เป็นธรรมกลับออกมาโดยไม่รู้ตัว
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">2. การจำกัดโอกาสทางสังคม</h3>
  <p className="mb-4 leading-relaxed">
    AI ที่ใช้คัดเลือกผู้สมัครงาน หรือแนะนำตำแหน่งงาน อาจจำกัดโอกาสของผู้สมัครบางกลุ่ม เช่น หากโมเดลเรียนรู้จากประวัติการจ้างงานที่มีแนวโน้มรับพนักงานชายมากกว่าหญิง โมเดลอาจเสนอแนะหรือคัดออกผู้หญิงจากตำแหน่งที่มีศักยภาพ โดยไม่รู้ว่าพฤติกรรมดังกล่าวเป็นการจำกัดสิทธิ
  </p>
  <p className="mb-4 leading-relaxed">
    สิ่งนี้อาจส่งผลให้ผู้สมัครกลุ่มหนึ่งไม่สามารถเข้าถึงตำแหน่งงานที่ดี หรือถูกลดศักยภาพโดยไม่เป็นธรรม เป็นการสร้างอุปสรรคเชิงโครงสร้างที่เกิดจากเทคโนโลยีโดยตรง
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">3. ความเสี่ยงต่อชีวิตและสุขภาพ</h3>
  <p className="mb-4 leading-relaxed">
    ในวงการแพทย์ ระบบวินิจฉัยด้วย AI ที่ไม่ได้รับการตรวจสอบเรื่อง fairness อาจประเมินอาการผิดพลาดกับบางกลุ่ม เช่น วินิจฉัยโรคหัวใจกับผู้หญิงได้ช้ากว่าผู้ชาย เพราะอาการทางกายภาพที่แสดงออกแตกต่างกัน แต่โมเดลเรียนรู้จากข้อมูลของผู้ชายมากกว่า
  </p>
  <p className="mb-4 leading-relaxed">
    ความผิดพลาดดังกล่าวไม่ได้เป็นเพียงข้อผิดทางเทคนิค แต่มีผลต่อชีวิตโดยตรง อาจทำให้เกิดการรักษาที่ไม่เหมาะสม หรือพลาดการวินิจฉัยที่สำคัญ
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">4. การทำลายความเชื่อมั่นต่อเทคโนโลยี</h3>
  <p className="mb-4 leading-relaxed">
    เมื่อผู้ใช้งานพบว่า AI ทำงานอย่างไม่เป็นธรรม ความเชื่อมั่นต่อระบบอัตโนมัติจะลดลงอย่างรวดเร็ว ไม่ว่าจะเป็นแอปพลิเคชันภาครัฐหรือเอกชน การขาด transparency และ accountability จะกลายเป็นอุปสรรคในการนำเทคโนโลยีมาใช้ในวงกว้าง
  </p>
  <p className="mb-4 leading-relaxed">
    ความไม่เชื่อมั่นนี้อาจขยายไปถึงสถาบันที่ใช้เทคโนโลยีนั้น เช่น หน่วยงานรัฐที่ใช้ AI ตรวจคนเข้าเมือง หรือธนาคารที่ใช้ AI อนุมัติสินเชื่อ อาจถูกตั้งคำถามในระดับนโยบาย
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">5. ผลกระทบด้านกฎหมายและการฟ้องร้อง</h3>
  <p className="mb-4 leading-relaxed">
    มีหลายกรณีที่ผู้เสียหายจากการตัดสินใจของ AI ฟ้องร้องหน่วยงานหรือบริษัทที่ใช้ระบบเหล่านั้น เช่น คดีในยุโรปที่ผู้สมัครกู้เงินยื่นฟ้องต่อธนาคาร เนื่องจากการปฏิเสธโดยไม่มีคำอธิบายที่เป็นธรรม หรือการใช้ระบบ facial recognition ที่มี bias ต่อคนผิวสีและก่อให้เกิดการจับกุมผิดตัว
  </p>
  <p className="mb-4 leading-relaxed">
    กรณีเหล่านี้เริ่มถูกจับตามองจากนักกฎหมายและองค์กรสิทธิมนุษยชนทั่วโลก โดยเฉพาะในประเทศที่มีการออกกฎหมายควบคุม AI อย่างจริงจัง เช่น EU AI Act
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">6. การทำลายความหลากหลายและความเท่าเทียม</h3>
  <p className="mb-4 leading-relaxed">
    หากโมเดล AI ส่งเสริมผลลัพธ์แบบซ้ำ ๆ จากกลุ่มที่มีลักษณะเหมือนกัน เช่น แนะนำเฉพาะเพลง ศิลปิน หรือคอร์สเรียนที่เป็นที่นิยมของกลุ่มใหญ่เพียงกลุ่มเดียว ความหลากหลายทางวัฒนธรรม ความคิดเห็น หรือความรู้ จะถูกลดทอนอย่างเงียบ ๆ
  </p>
  <p className="mb-4 leading-relaxed">
    นี่คือความเสี่ยงต่อการพัฒนาทางปัญญาของผู้ใช้ และต่อ ecosystem ของเนื้อหาหรือบริการ ซึ่งอาจส่งผลให้เกิดการผูกขาดเชิงอัลกอริทึมโดยไม่ตั้งใจ
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6">7. ผลกระทบระยะยาวต่อความยุติธรรมในสังคม</h3>
  <p className="mb-4 leading-relaxed">
    หากปล่อยให้ระบบ AI ทำงานอย่างมี bias ต่อเนื่อง จะกลายเป็นการสร้าง “ความเหลื่อมล้ำเชิงอัตโนมัติ” (Automated Inequality) ซึ่งสืบทอดและขยายความไม่เท่าเทียมในระดับโครงสร้าง โดยที่ผู้พัฒนาไม่รู้ตัว
  </p>
  <p className="mb-4 leading-relaxed">
    สิ่งนี้จะส่งผลกระทบต่อการเข้าถึงโอกาส การยอมรับจากสังคม และการเติบโตในระดับบุคคลและชุมชน ในท้ายที่สุด ระบบ AI ที่ไม่เป็นธรรมจะกลายเป็นเครื่องมือที่ตอกย้ำความไม่เท่าเทียม มากกว่าการแก้ปัญหา
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p className="mb-2">
      AI ไม่ได้เป็นเพียงเครื่องมือทางเทคนิค แต่เป็นกลไกเชิงนโยบายที่ส่งผลต่อชีวิตผู้คนโดยตรง หากละเลยเรื่อง fairness และ bias ระบบ AI จะกลายเป็นภาระต่อสังคม มากกว่าจะเป็นเครื่องมือสร้างประโยชน์ร่วม
    </p>
    <p>
      การออกแบบที่คำนึงถึงผลกระทบจาก bias จึงไม่ใช่ความหรูหรา แต่เป็นความจำเป็นที่ต้องมีตั้งแต่การวางระบบไปจนถึงการนำไปใช้งานจริง
    </p>
  </div>
</section>

<section id="bias-types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">1. ประเภทของ Bias ที่พบบ่อยใน AI</h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <p className="mb-4 leading-relaxed">
    Bias ในระบบ AI ไม่ได้เกิดขึ้นจากความบังเอิญ แต่เกิดจากกระบวนการต่าง ๆ ที่ฝังอยู่ในข้อมูล วิธีการฝึกโมเดล และแม้แต่ขั้นตอนการออกแบบอัลกอริทึม ความเข้าใจในประเภทของ Bias ที่พบบ่อยจะช่วยให้สามารถออกแบบระบบที่ยุติธรรมและลดผลกระทบเชิงลบต่อผู้ใช้งานในระยะยาว
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">1. Historical Bias</h3>
  <p className="mb-4 leading-relaxed">
    เป็น Bias ที่ฝังอยู่ในข้อมูลตั้งแต่ต้น เช่น ระบบที่ใช้ข้อมูลทางประวัติศาสตร์ ซึ่งอาจมีอคติทางเชื้อชาติ เพศ หรืออายุอยู่แล้ว เช่น ในอดีตอาจมีการให้กู้เฉพาะผู้ชายที่ทำงานประจำ เมื่อใช้ข้อมูลเหล่านี้ฝึกโมเดล ก็จะเกิดการเรียนรู้อคติแบบเดิมซ้ำอีกครั้ง
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">2. Sampling Bias</h3>
  <p className="mb-4 leading-relaxed">
    เกิดจากการเลือกกลุ่มตัวอย่างในการเก็บข้อมูลไม่ครอบคลุม ทำให้โมเดลเรียนรู้จากข้อมูลที่ไม่สะท้อนความเป็นจริง เช่น การฝึกโมเดลตรวจจับใบหน้าด้วยรูปภาพที่ส่วนใหญ่เป็นคนผิวขาว จะทำให้โมเดลมีประสิทธิภาพต่ำกับกลุ่มคนผิวสี
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">3. Measurement Bias</h3>
  <p className="mb-4 leading-relaxed">
    เกิดจากวิธีการวัดหรือการบันทึกข้อมูลที่ไม่ถูกต้องหรือไม่เท่าเทียม เช่น เครื่องมือวัดความดันที่ให้ค่าคลาดเคลื่อนกับกลุ่มอายุหนึ่งมากกว่ากลุ่มอื่น หรือการวัดผลสัมฤทธิ์ทางการศึกษาที่ประเมินนักเรียนโดยไม่คำนึงถึงสภาพแวดล้อมการเรียนรู้
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">4. Label Bias</h3>
  <p className="mb-4 leading-relaxed">
    เกิดจากกระบวนการกำหนด Label โดยผู้เชี่ยวชาญที่อาจมีมุมมองหรือประสบการณ์ที่แตกต่างกัน เช่น การตีความว่าข้อความหนึ่งเป็น "ความรุนแรง" หรือ "แค่ล้อเล่น" อาจขึ้นอยู่กับวัฒนธรรมหรือบริบทส่วนบุคคล
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">5. Algorithmic Bias</h3>
  <p className="mb-4 leading-relaxed">
    แม้ข้อมูลจะไม่มี Bias แต่การออกแบบอัลกอริทึมหรือฟังก์ชันเป้าหมาย (objective function) อาจสร้าง Bias ได้ เช่น การ Optimize ให้ Accuracy สูงสุดในภาพรวม อาจทำให้โมเดลเลือกกลุ่มเสียงข้างมากเป็นหลัก โดยละเลยกลุ่มเสียงข้างน้อย
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">6. Representation Bias</h3>
  <p className="mb-4 leading-relaxed">
    การเลือกเฉพาะข้อมูลหรือกลุ่มตัวอย่างที่ "เห็นได้บ่อย" มาใช้ในการฝึกโมเดล อาจทำให้โมเดลไม่สามารถเรียนรู้จากกรณีที่หายาก เช่น โมเดลทางการแพทย์ที่ไม่สามารถตรวจจับโรคหายากได้ เพราะไม่มีข้อมูลโรคเหล่านั้นในการฝึก
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">7. Confirmation Bias</h3>
  <p className="mb-4 leading-relaxed">
    เมื่อผู้พัฒนา AI มีแนวโน้มเลือกข้อมูลที่สอดคล้องกับความเชื่อหรือสมมุติฐานเดิม เช่น การเก็บตัวอย่างผู้ใช้งานจากกลุ่มที่เห็นด้วยกับแนวคิดของระบบโดยไม่เจตนา ทำให้โมเดลไม่สามารถรับมือกับความคิดเห็นที่หลากหลายได้
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">8. Temporal Bias</h3>
  <p className="mb-4 leading-relaxed">
    เกิดจากการใช้ข้อมูลที่ล้าสมัย โดยไม่อัปเดตตามพฤติกรรมหรือบริบทของโลกที่เปลี่ยนไป เช่น โมเดลพยากรณ์เศรษฐกิจที่อิงจากข้อมูลช่วงก่อนวิกฤต COVID-19 อาจทำนายผิดพลาดในบริบทปัจจุบัน
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">9. Feedback Loop Bias</h3>
  <p className="mb-4 leading-relaxed">
    เมื่อโมเดลมีผลต่อข้อมูลที่ใช้ในอนาคต เช่น ระบบแนะนำเพลงที่แนะนำเฉพาะแนวเพลงเดิม ทำให้ผู้ใช้ฟังเพลงแคบลง และระบบเรียนรู้ซ้ำจากข้อมูลนั้น ส่งผลให้ไม่สามารถเสนอเนื้อหาหลากหลายได้
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">10. Societal Bias</h3>
  <p className="mb-4 leading-relaxed">
    Bias ที่สะท้อนอคติทางสังคม เช่น การเหมารวมเพศอาชีพ (เช่น พยาบาลต้องเป็นผู้หญิง) หรืออาชญากรรมต้องเกี่ยวกับบางเชื้อชาติ หากโมเดลเรียนรู้จากข้อมูลที่สะท้อนความเชื่อนี้ จะเสริมสร้างอคติทางสังคมต่อไปอีก
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      ประเภทของ Bias ที่กล่าวมานี้ไม่ได้เกิดจากความตั้งใจเสมอไป แต่อาจซ่อนอยู่ในทุกขั้นตอนของการพัฒนา AI ตั้งแต่การเก็บข้อมูลจนถึงการใช้งานจริง การตรวจสอบและเข้าใจ Bias ในแต่ละรูปแบบจึงเป็นสิ่งสำคัญ เพื่อสร้างระบบที่ยุติธรรม โปร่งใส และไม่สร้างผลกระทบเชิงลบต่อผู้คนหรือกลุ่มที่เปราะบางในสังคม
    </p>
  </div>
</section>


<section id="bias-analysis" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">2. การวิเคราะห์ Bias ในโมเดล</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <p className="mb-4 leading-relaxed">
    การวิเคราะห์ Bias ในโมเดล AI ไม่ใช่แค่การตรวจสอบผลลัพธ์รวม แต่เป็นการแยกดูผลลัพธ์ของโมเดลในแต่ละกลุ่มย่อยที่แตกต่างกัน เช่น เพศ เชื้อชาติ อายุ รายได้ หรือภูมิภาค เพื่อประเมินว่าโมเดลปฏิบัติต่อทุกกลุ่มอย่างเป็นธรรม หรือมีอคติแฝงอยู่ในกระบวนการเรียนรู้หรือไม่
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-8">การแบ่งกลุ่มเพื่อวิเคราะห์</h3>
  <p className="mb-4 leading-relaxed">
    ขั้นตอนแรกของการวิเคราะห์คือการกำหนดกลุ่มที่มีความเสี่ยงต่อการเกิด bias เช่น กลุ่มเปราะบาง (vulnerable groups) หรือกลุ่มที่อาจได้รับผลกระทบจากการตัดสินใจของระบบ AI ตัวอย่างเช่น ผู้หญิงในอุตสาหกรรมเทคโนโลยี คนผิวดำในระบบยุติธรรม หรือผู้สูงอายุในระบบสุขภาพ
  </p>

  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>แบ่งข้อมูลตาม group เช่น gender: male/female/non-binary</li>
    <li>ตรวจสอบความแม่นยำ (Accuracy) ของแต่ละกลุ่ม</li>
    <li>เปรียบเทียบค่า Recall, Precision, F1 score รายกลุ่ม</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">การใช้ Confusion Matrix รายกลุ่ม (Group-based Confusion Matrix)</h3>
  <p className="mb-4 leading-relaxed">
    Confusion Matrix คือเครื่องมือหลักที่ใช้ตรวจสอบประสิทธิภาพของโมเดลใน classification task โดยเมื่อใช้แบบ group-based จะช่วยให้เห็นว่าโมเดลมีความไม่เท่าเทียมในการทำนายแต่ละ class สำหรับแต่ละกลุ่มหรือไม่ เช่น อัตราการ False Positive สูงในกลุ่ม A มากกว่ากลุ่ม B อย่างชัดเจน
  </p>

  <div className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto mb-4">
    from sklearn.metrics import confusion_matrix<br/>
    for group in groups:
    &nbsp;&nbsp;cm = confusion_matrix(y_true[group], y_pred[group])
  </div>

  <p className="mb-4 leading-relaxed">
    ค่าจาก Confusion Matrix สามารถใช้ต่อยอดไปวิเคราะห์ค่าเช่น TPR, FPR, FNR เพื่อประเมินความเท่าเทียมของการทำนายในแต่ละกลุ่ม
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-8">Metric สำคัญในการตรวจจับ Bias</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Demographic Parity:</strong> โมเดลควรให้โอกาสผลลัพธ์เชิงบวก (เช่น ให้กู้ผ่าน) เท่าเทียมกันในทุกกลุ่ม</li>
    <li><strong>Equal Opportunity:</strong> โมเดลควรมี True Positive Rate เท่า ๆ กันระหว่างกลุ่ม</li>
    <li><strong>Equalized Odds:</strong> ความน่าจะเป็นของ True Positive และ False Positive ควรใกล้เคียงกันในทุกกลุ่ม</li>
    <li><strong>Disparate Impact:</strong> วัดอัตราส่วนของผลลัพธ์ระหว่างกลุ่มหลักและกลุ่มเปรียบเทียบ ค่าที่ห่าง 1 มากแสดงถึงความไม่เท่าเทียม</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">การใช้ Fairness Dashboards</h3>
  <p className="mb-4 leading-relaxed">
    Fairness Dashboard คือชุดเครื่องมือ visualization ที่ช่วยให้ทีมพัฒนาสามารถวิเคราะห์และเปรียบเทียบ performance และ fairness ของโมเดลในแต่ละกลุ่มได้อย่างชัดเจน โดยมีการแสดงผลในรูปแบบเช่น distribution graph, score breakdown และ metric per group ที่เข้าใจง่าย
  </p>

  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>Microsoft Fairlearn Dashboard</li>
    <li>IBM AI Fairness 360 Toolkit (AIF360)</li>
    <li>Google What-If Tool</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">Insight สรุปจากการวิเคราะห์</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">การวิเคราะห์ Bias ไม่ใช่แค่การวัด performance ของโมเดล แต่คือการยืนยันว่าโมเดลไม่สร้างความเหลื่อมล้ำให้กับกลุ่มใดกลุ่มหนึ่ง</p>
    <p className="mb-2">Fairness analysis ต้องทำตั้งแต่ขั้นตอนวางแผนจนถึง deployment โดยมีเป้าหมายเพื่อสร้างระบบ AI ที่ยุติธรรม และไม่ตอกย้ำความไม่เท่าเทียมในสังคม</p>
    <p className="mb-2">การแสดงผลลัพธ์แยกกลุ่ม ช่วยให้เห็นจุดอ่อนที่อาจไม่ปรากฏจาก metric รวม และนำไปสู่การปรับปรุงที่ตรงจุดมากขึ้น</p>
  </div>
</section>


       
<section id="bias-solutions" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">3. วิธีลด Bias ตั้งแต่ต้นน้ำถึงปลายน้ำ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <p className="mb-4 leading-relaxed">
    การลดอคติ (Bias) ในระบบ AI ไม่ใช่แค่การแก้ไขที่จุดใดจุดหนึ่ง แต่เป็นการออกแบบเชิงระบบที่เริ่มตั้งแต่ขั้นตอนการรวบรวมข้อมูลไปจนถึงการแสดงผลลัพธ์กับผู้ใช้งานจริง แนวทางที่ครอบคลุมประกอบด้วย Preprocessing, In-processing และ Post-processing โดยแต่ละขั้นตอนมีวิธีที่หลากหลายขึ้นอยู่กับปัญหาและบริบทของการใช้งาน
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-8">Preprocessing Techniques</h3>
  <p className="mb-4 leading-relaxed">
    ขั้นตอนก่อนการฝึกโมเดลคือช่วงที่สามารถจัดการ bias ได้ง่ายที่สุด เนื่องจากสามารถควบคุมคุณภาพข้อมูลที่ป้อนให้โมเดลได้ หากข้อมูลมี bias อยู่แล้ว โมเดลก็มีแนวโน้มที่จะเรียนรู้ bias นั้นโดยไม่รู้ตัว
  </p>

  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>Sampling แบบสมดุล (Balanced Sampling):</strong> ปรับจำนวนข้อมูลในแต่ละกลุ่มให้ใกล้เคียงกัน เช่น จำนวนตัวอย่างเพศชายและหญิง เพื่อป้องกันไม่ให้โมเดลเรียนรู้จากข้อมูลกลุ่มใดกลุ่มหนึ่งมากเกินไป</li>
    <li><strong>Reweighing:</strong> ให้ weight กับตัวอย่างข้อมูลในกลุ่มที่มีน้อยกว่าหรือด้อยโอกาส เพื่อชดเชยความไม่สมดุลในข้อมูล โดยเฉพาะใน classification tasks</li>
    <li><strong>การลบ sensitive attribute:</strong> ลบ features ที่เกี่ยวข้องกับกลุ่มเปราะบาง เช่น เพศ เชื้อชาติ หรือศาสนา เพื่อลดโอกาสที่โมเดลจะใช้ข้อมูลเหล่านั้นในการตัดสินใจ</li>
    <li><strong>Data Augmentation:</strong> สร้างข้อมูลเสริมสำหรับกลุ่มที่มีน้อย เช่น การใช้เทคนิคการสังเคราะห์ข้อมูลใน NLP หรือการหมุน/ปรับภาพในงาน computer vision</li>
    <li><strong>Disentangled Representation:</strong> ใช้ representation learning เพื่อแยก feature ที่มีผลกับ label ออกจาก feature ที่เกี่ยวข้องกับ sensitive attribute</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">In-processing Techniques</h3>
  <p className="mb-4 leading-relaxed">
    เทคนิคในขั้นตอนการฝึกโมเดลจะเข้าไปปรับ loss function หรือโครงสร้างโมเดลโดยตรง เพื่อใส่เงื่อนไขเกี่ยวกับ fairness เข้าไป
  </p>

  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>Fairness-aware Loss Functions:</strong> ปรับ loss function ให้รวม penalty สำหรับการละเมิด fairness เช่น เพิ่ม term ที่ควบคุม disparate impact หรือ equal opportunity</li>
    <li><strong>Adversarial Debiasing:</strong> ใช้ adversarial network เพื่อบังคับให้โมเดลหลักไม่สามารถเดา sensitive attribute ได้จาก latent representation ทำให้ representation ที่ได้เป็นกลางมากขึ้น</li>
    <li><strong>Representation Learning with Constraints:</strong> ใช้ embedding หรือ latent space ที่ควบคุมไม่ให้ feature เฉพาะกลุ่มส่งผลต่อผลลัพธ์ เช่นผ่านการกำหนด regularizer</li>
    <li><strong>Gradient Reversal Layer (GRL):</strong> ใช้เพื่อสกัด gradient ที่จะนำไปสู่การเรียนรู้ความแตกต่างระหว่างกลุ่ม ทำให้โมเดลเรียนรู้ feature ที่ไม่ขึ้นกับ sensitive attributes</li>
    <li><strong>Distribution Matching:</strong> ปรับ distribution ของ representation ของกลุ่มต่าง ๆ ให้มีลักษณะคล้ายกันมากขึ้นโดยใช้ MMD หรือ KL-divergence</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">Post-processing Techniques</h3>
  <p className="mb-4 leading-relaxed">
    แม้ว่าโมเดลจะฝึกเสร็จแล้ว แต่ยังสามารถปรับ output หรือ decision threshold เพื่อให้ผลลัพธ์เป็นธรรมมากขึ้นได้ เทคนิคเหล่านี้มักใช้งานง่ายและสามารถปรับใช้กับโมเดลใดก็ได้
  </p>

  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>Calibrated Equalized Odds:</strong> ปรับ threshold ของแต่ละกลุ่มให้ความน่าจะเป็นในการผ่านเงื่อนไขต่าง ๆ เท่ากัน เช่น True Positive Rate</li>
    <li><strong>Reject Option Classification:</strong> ปรับการตัดสินใจในบริเวณไม่มั่นใจ (uncertainty zone) เพื่อให้กลุ่มที่เสียเปรียบมีโอกาสมากขึ้น โดยยังรักษา performance โดยรวมไว้</li>
    <li><strong>Threshold Optimization by Group:</strong> ใช้ threshold แยกกันระหว่างกลุ่มต่าง ๆ เช่น threshold สำหรับเพศหญิงอาจต่างจากเพศชาย หากพบว่า distribution แตกต่างกันมาก</li>
    <li><strong>Fair Decision Boundary:</strong> ปรับ boundary ของ classification ให้ไม่เอนเอียง เช่น ใช้ constraint optimization เพื่อลด disparate impact</li>
    <li><strong>Confidence Adjustment:</strong> ปรับค่าความมั่นใจของ prediction หลัง inference เช่น ลด score สำหรับกลุ่มที่โมเดลมั่นใจผิดบ่อย</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-8">การเลือกใช้เทคนิค</h3>
  <p className="mb-4 leading-relaxed">
    การเลือกใช้เทคนิคต้องพิจารณาจากหลายปัจจัย เช่น ความซับซ้อนของโมเดล ลักษณะของข้อมูล และข้อจำกัดของระบบงาน เช่น หากต้องการ deploy เร็วและใช้โมเดลที่มีอยู่แล้ว อาจเลือก post-processing แต่ถ้าสามารถแก้ที่ต้นน้ำได้ควรเลือก preprocessing เพราะมีความเสี่ยงต่ำกว่า
  </p>

  <p className="mb-4 leading-relaxed">
    ไม่ควรใช้เทคนิคใดเทคนิคหนึ่งเพียงอย่างเดียว แต่ควรมองเป็นชุดของกระบวนการที่เชื่อมโยงกัน โดยในระบบใหญ่ ๆ การใช้ dashboard เพื่อตรวจสอบ bias ระหว่างการพัฒนาและทดสอบจะช่วยสร้างความมั่นใจให้กับทั้งทีมงานและผู้ใช้งาน
  </p>

  <div className="bg-green-100 dark:bg-green-900 text-black dark:text-green-100 p-6 rounded-xl border-l-4 border-green-500 shadow mt-6">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      การลด Bias อย่างแท้จริงไม่ได้หมายถึงการลบความต่างทั้งหมดออกจากข้อมูล แต่หมายถึงการออกแบบระบบที่เคารพในความแตกต่าง และไม่ตัดสินผู้ใช้งานจากปัจจัยที่ไม่เกี่ยวข้องกับความสามารถหรือคุณสมบัติที่แท้จริง
    </p>
  </div>
</section>

<section id="fairness-testing" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">4. การทดลองและวัด Fairness</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <p className="mb-4 leading-relaxed">
    การวัด Fairness ของโมเดล AI ไม่ใช่แค่การดูว่าโมเดลให้ผลลัพธ์ที่ถูกต้องหรือไม่ แต่ยังหมายถึงการตรวจสอบว่าโมเดลมีพฤติกรรมที่เป็นธรรมกับทุกกลุ่มประชากรหรือไม่ เช่น กลุ่มเพศ เชื้อชาติ หรือช่วงอายุต่าง ๆ การทดลองและวัด Fairness จำเป็นต้องอาศัยทั้งการแยกวิเคราะห์ผลลัพธ์รายกลุ่มและการใช้ metric เฉพาะที่ออกแบบมาเพื่อตรวจจับอคติในระบบอัตโนมัติ
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 1. แบ่งกลุ่มข้อมูลเพื่อประเมิน</h3>
  <p className="mb-4 leading-relaxed">
    ขั้นแรกในการตรวจสอบ Fairness คือการแบ่งข้อมูลออกเป็นกลุ่มตาม demographic attribute ที่เกี่ยวข้อง เช่น gender (ชาย/หญิง/อื่น ๆ), race (กลุ่มเชื้อชาติ), หรือ age group (วัยรุ่น/ผู้ใหญ่/ผู้สูงอายุ) เพื่อดูว่าโมเดลมี performance ที่ต่างกันระหว่างกลุ่มหรือไม่
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 2. การใช้ Confusion Matrix รายกลุ่ม</h3>
  <p className="mb-4 leading-relaxed">
    สร้าง confusion matrix แยกตามกลุ่มประชากร แล้วเปรียบเทียบค่าเช่น True Positive Rate (TPR), False Positive Rate (FPR), Precision, Recall ว่าโมเดลให้ความยุติธรรมกับกลุ่มใดมากหรือน้อยเป็นพิเศษ เช่น หาก TPR ของผู้หญิงต่ำกว่าผู้ชายอย่างมีนัยสำคัญ แสดงว่าโมเดลอาจมีอคติกับกลุ่มผู้หญิง
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 3. Metrics ที่ใช้วัด Fairness</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>Demographic Parity:</strong> โอกาสที่โมเดลจะทำนายผลลัพธ์เชิงบวก (positive prediction) ควรใกล้เคียงกันในแต่ละกลุ่ม</li>
    <li><strong>Equal Opportunity:</strong> ค่า True Positive Rate ควรเท่ากันในแต่ละกลุ่ม</li>
    <li><strong>Equalized Odds:</strong> ทั้ง TPR และ FPR ควรใกล้เคียงกันในแต่ละกลุ่ม</li>
    <li><strong>Disparate Impact:</strong> เปรียบเทียบสัดส่วน positive outcome ระหว่างกลุ่มที่ถูกปกป้อง (protected group) และกลุ่มควบคุม</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 4. ตัวอย่างโค้ดเบื้องต้นด้วย Fairlearn</h3>
  <p className="mb-4 leading-relaxed">
    Fairlearn เป็นไลบรารีสำหรับการวัดและลดอคติในโมเดล Machine Learning มีฟังก์ชันในการวิเคราะห์ fairness dashboard และการปรับแต่งโมเดลให้ยุติธรรมมากขึ้น ตัวอย่างการใช้งานเบื้องต้น:
  </p>

  <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto">
{`
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from sklearn.metrics import accuracy_score

metric_frame = MetricFrame(
  metrics={"accuracy": accuracy_score, "selection_rate": selection_rate, "TPR": true_positive_rate},
  y_true=y_test,
  y_pred=predictions,
  sensitive_features=df_test["gender"]
)

print(metric_frame.by_group)
`}
  </pre>

  <p className="mb-4 leading-relaxed">
    ค่าที่ได้จะบอกว่าโมเดลมี performance อย่างไรในแต่ละกลุ่ม เช่น ความแม่นยำในผู้หญิงเทียบกับผู้ชาย หรืออัตราการเลือกกลุ่มที่แตกต่างกันอย่างมีนัยสำคัญหรือไม่
  </p>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 5. ทดลองเปรียบเทียบผลลัพธ์จริง</h3>
  <p className="mb-4 leading-relaxed">
    เพื่อดูว่าโมเดลมี fairness หรือไม่ ควรทดลองให้โมเดลทำนายผลลัพธ์กับข้อมูลที่มาจากกลุ่มต่าง ๆ โดยใช้ข้อมูลชุดเดียวกัน และแยกวิเคราะห์ผลลัพธ์ เช่น:
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ทดสอบความแม่นยำของโมเดลแยกตามเชื้อชาติ → พบว่า TPR ในกลุ่ม A สูงกว่า B</li>
    <li>ทดสอบ Recall ระหว่างอายุ 18–25 และ 60–70 → หากต่างกันมากอาจต้องพิจารณาวิธีปรับสมดุล</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 6. การสร้าง Fairness Report</h3>
  <p className="mb-4 leading-relaxed">
    ควรสร้างรายงานที่ระบุว่าโมเดลผ่านหรือไม่ผ่านเกณฑ์ด้าน Fairness โดยใช้ metrics ข้างต้น พร้อมสรุปความแตกต่างของ performance ระหว่างกลุ่ม รายงานนี้ควรประกอบด้วย:
  </p>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>ตาราง metric รายกลุ่ม</li>
    <li>Visualization เช่น bar chart, parity plot</li>
    <li>ข้อเสนอแนะในการปรับปรุง fairness</li>
    <li>เก็บเป็น audit trail เพื่อใช้ในการตรวจสอบภายหลัง</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3 mt-6"> 7. กรณีศึกษา</h3>
  <p className="mb-4 leading-relaxed">
    มีกรณีศึกษาหลายกรณีที่การวัด Fairness มีบทบาทสำคัญ เช่น โมเดลคัดกรองผู้สมัครงานที่มี accuracy สูงแต่คัดผู้หญิงออกมากกว่าผู้ชาย หลังจากวัด Equal Opportunity พบว่า TPR ของผู้หญิงต่ำกว่าเกณฑ์ จึงนำไปสู่การ retrain ด้วยเทคนิค reweighing และ post-processing เพื่อเพิ่ม fairness
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การวัด Fairness ไม่ใช่ขั้นตอนเสริมหลังการพัฒนา แต่ควรเป็นส่วนหนึ่งของกระบวนการประเมินคุณภาพโมเดลในทุกระบบ AI ที่มีผลกระทบต่อมนุษย์โดยตรง
  </div>
</section>
<section id="ethics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">5. จริยธรรมของ AI และการออกแบบที่รับผิดชอบ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <p className="mb-4 leading-relaxed">
    การพัฒนา AI ไม่ได้เป็นเพียงเรื่องของประสิทธิภาพเชิงเทคนิคหรือความสามารถในการทำนายที่แม่นยำ แต่ยังเกี่ยวข้องกับการตัดสินใจที่มีผลกระทบต่อสังคม มนุษย์ และโครงสร้างของระบบในระยะยาว จริยธรรมของ AI (Ethics in AI) คือกรอบแนวคิดที่ใช้ในการกำหนดว่าอะไรควรทำและไม่ควรทำในการออกแบบ พัฒนา และใช้งานระบบปัญญาประดิษฐ์
  </p>

  <h3 className="text-xl font-semibold mb-3">หลักการสำคัญของ AI ที่มีจริยธรรม (Ethical AI Principles)</h3>
  <ul className="list-disc pl-6 space-y-3 mb-6">
    <li>
      <strong>Transparency:</strong> ระบบ AI ควรมีความโปร่งใส สามารถอธิบายการตัดสินใจและวิธีการทำงานของโมเดลได้ ไม่ควรเป็นกล่องดำที่ปิดบังเบื้องหลังการวิเคราะห์
    </li>
    <li>
      <strong>Fairness:</strong> AI ต้องไม่เลือกปฏิบัติต่อกลุ่มใดกลุ่มหนึ่งโดยไม่มีเหตุผล เช่น การให้สิทธิ์หรือปฏิเสธโอกาสกับบุคคลตามเชื้อชาติ เพศ หรือศาสนา
    </li>
    <li>
      <strong>Accountability:</strong> ผู้พัฒนาและองค์กรต้องรับผิดชอบต่อผลกระทบของ AI ที่ถูกใช้งาน โดยไม่โยนความรับผิดให้กับระบบอัตโนมัติ
    </li>
    <li>
      <strong>Safety:</strong> ระบบ AI ต้องมีความปลอดภัย ไม่ก่อให้เกิดความเสียหายกับผู้ใช้งานหรือสิ่งแวดล้อม และต้องมีระบบตรวจจับข้อผิดพลาดได้ทันท่วงที
    </li>
    <li>
      <strong>Privacy:</strong> ข้อมูลที่ใช้ฝึกหรือประมวลผลต้องได้รับการจัดการอย่างระมัดระวังตามหลักการคุ้มครองข้อมูลส่วนบุคคล
    </li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">ตัวอย่าง Framework ที่เป็นมาตรฐานระดับโลก</h3>
  <div className="grid md:grid-cols-2 gap-6 mb-6">
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow border text-sm">
      <h4 className="font-semibold mb-2">Google AI Principles</h4>
      <p>
        ประกาศหลักการ 7 ข้อ เช่น ความยุติธรรม การหลีกเลี่ยงอาวุธ การเคารพความเป็นส่วนตัว และการทดสอบระบบเพื่อความปลอดภัย ถูกใช้เพื่อประเมินทุกโครงการภายในก่อนอนุมัติให้ใช้งานจริง
      </p>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow border text-sm">
      <h4 className="font-semibold mb-2">EU AI Act</h4>
      <p>
        กฎหมายใหม่ของสหภาพยุโรปที่จัดประเภทความเสี่ยงของระบบ AI เช่น high-risk AI ต้องมีการตรวจสอบโดยผู้เชี่ยวชาญ มี document การทำงาน และควบคุมความโปร่งใสอย่างเข้มงวด
      </p>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow border text-sm">
      <h4 className="font-semibold mb-2">OECD AI Guidelines</h4>
      <p>
        แนวทางระดับนานาชาติที่ส่งเสริมการใช้ AI อย่างมีความรับผิดชอบ ครอบคลุมทั้งด้านสิทธิมนุษยชน ความเท่าเทียม และการกระจายผลประโยชน์ของเทคโนโลยี
      </p>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3">จริยธรรมเชิงปฏิบัติ: กรณีตัวอย่างในองค์กร</h3>
  <p className="mb-4 leading-relaxed">
    หลายองค์กรได้นำหลักจริยธรรมของ AI มาปรับใช้ในระดับปฏิบัติ เช่น ตั้งคณะกรรมการตรวจสอบอัลกอริทึม สร้างกระบวนการ feedback จากผู้ใช้ และกำหนดนโยบายการใช้ข้อมูลที่ชัดเจน ซึ่งไม่เพียงช่วยลดความเสี่ยงเชิงกฎหมาย แต่ยังสร้างความไว้วางใจในระดับสังคม
  </p>

  <h3 className="text-xl font-semibold mb-3">การฝังแนวคิด Ethical AI ในวงจรการพัฒนา</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>เริ่มจากการกำหนดวัตถุประสงค์ที่ไม่ละเมิดสิทธิหรือเลือกปฏิบัติ</li>
    <li>ออกแบบระบบด้วยข้อมูลที่สะท้อนความหลากหลายของผู้ใช้งาน</li>
    <li>เลือกโมเดลที่สามารถตรวจสอบได้หรือใช้ explainability tool ประกอบ</li>
    <li>ตั้ง checkpoint ด้านจริยธรรมในแต่ละ phase ของการพัฒนา</li>
    <li>ทำ fairness testing และ privacy check ก่อน deploy</li>
    <li>ตั้งกระบวนการแจ้งเตือนเมื่อระบบทำงานผิดปกติ (Monitoring & Alert)</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">แนวคิดใหม่: Human-Centered AI</h3>
  <p className="mb-4 leading-relaxed">
    แนวคิด Human-Centered AI เน้นว่า AI ควรเป็นเครื่องมือเสริมมนุษย์ ไม่ใช่แทนที่มนุษย์ทั้งหมด การออกแบบจึงควรคำนึงถึงประสบการณ์ของผู้ใช้ ความเข้าใจง่าย และการควบคุมได้ ซึ่งสอดคล้องกับหลักจริยธรรมที่ว่าระบบควรให้มนุษย์เป็นผู้ตัดสินใจสุดท้าย (Human-in-the-loop)
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      ระบบ AI ที่รับผิดชอบต้องไม่เพียงแค่คำนวณได้แม่นยำ แต่ต้องคำนึงถึงคุณค่า มนุษยธรรม และผลกระทบต่อสังคม การมีจริยธรรมฝังอยู่ในการออกแบบตั้งแต่ต้นจะนำไปสู่เทคโนโลยีที่ยั่งยืนและเป็นที่ยอมรับอย่างแท้จริง
    </p>
  </div>
</section>


<section id="human-values" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    6. การออกแบบระบบ AI ที่เคารพความเป็นมนุษย์
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <p className="mb-4 leading-relaxed">
    การออกแบบระบบ AI ที่เคารพความเป็นมนุษย์ไม่ได้หมายถึงเพียงแค่การทำให้ระบบใช้งานง่ายหรือมี UI ที่เป็นมิตรเท่านั้น แต่ต้องรวมไปถึงหลักการเชิงจริยธรรมที่ตระหนักถึงศักดิ์ศรี เสรีภาพ และความเป็นส่วนตัวของบุคคล ระบบ AI ควรมีโครงสร้างที่ยึดถือสิทธิมนุษยชนเป็นหลัก พร้อมสร้างสมดุลระหว่างประสิทธิภาพและผลกระทบต่อชีวิตของผู้ใช้
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">1. ความยินยอม (Consent)</h3>
  <p className="mb-4 leading-relaxed">
    ทุกระบบที่ใช้ข้อมูลส่วนบุคคลควรมีการขอความยินยอมอย่างชัดเจนก่อนการเก็บ ใช้งาน หรือวิเคราะห์ข้อมูล ไม่ใช่แค่ checkbox เชิงเทคนิค แต่ต้องอธิบายอย่างเข้าใจง่ายว่าเก็บข้อมูลอะไร ใช้เพื่ออะไร และสามารถยกเลิกได้ตลอดเวลา ความยินยอมต้องเกิดจากการเลือกด้วยความสมัครใจ ไม่ใช่การบังคับกลาย ๆ
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">2. Right to Explanation</h3>
  <p className="mb-4 leading-relaxed">
    ผู้ใช้มีสิทธิที่จะได้รับคำอธิบายว่าระบบ AI ตัดสินใจอย่างไร โดยเฉพาะหากการตัดสินใจนั้นมีผลต่อสิทธิ์หรือโอกาส เช่น การอนุมัติเงินกู้ การคัดเลือกผู้สมัครงาน หรือการประเมินสุขภาพ การให้เหตุผลต้องเข้าใจง่าย ไม่ใช่แค่แสดงค่า probability หรือ confidence score อย่างเดียว
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">3. ความเป็นส่วนตัวโดยการออกแบบ (Privacy-by-Design)</h3>
  <p className="mb-4 leading-relaxed">
    ระบบควรวางโครงสร้างให้ข้อมูลส่วนบุคคลถูกปกป้องตั้งแต่ขั้นตอนแรก ไม่ใช่เป็นเพียงส่วนเสริมทีหลัง เช่น ใช้การ anonymization, differential privacy, หรือ encryption แบบ end-to-end ตั้งแต่การเก็บข้อมูลจนถึงการประมวลผล เพื่อป้องกันการรั่วไหลแม้ในกรณีที่ระบบถูกเจาะ
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">4. การป้องกันการเลือกปฏิบัติ (Anti-Discrimination)</h3>
  <p className="mb-4 leading-relaxed">
    ระบบ AI ไม่ควรตัดสินใจจากคุณลักษณะที่อาจนำไปสู่การเลือกปฏิบัติ เช่น เพศ เชื้อชาติ ศาสนา หรือสถานะทางสังคม เว้นแต่มีความจำเป็นชัดเจนเชิงกฎหมายหรือจริยธรรม ควรใช้ fairness constraint หรือ algorithm debiasing เพื่อลดความลำเอียงที่อาจเกิดขึ้นโดยไม่ตั้งใจ
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">5. ความปลอดภัยจากการใช้ผิดวัตถุประสงค์ (Misuse Prevention)</h3>
  <p className="mb-4 leading-relaxed">
    ทุกระบบ AI มีความเสี่ยงที่จะถูกนำไปใช้ในทางที่ผิด เช่น deepfake, social scoring หรือ surveillance ที่ละเมิดสิทธิส่วนบุคคล จึงต้องมีการออกแบบระบบที่จำกัดขอบเขตการใช้งาน มี logging, monitoring, และการจำกัด access อย่างชัดเจน รวมถึงมีแผนรับมือหากเกิดการใช้งานที่ไม่ตรงตามวัตถุประสงค์ดั้งเดิม
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">6. ความสอดคล้องกับเป้าหมายมนุษย์ (AI Alignment)</h3>
  <p className="mb-4 leading-relaxed">
    AI ควรมีจุดมุ่งหมายที่สอดคล้องกับคุณค่า ความต้องการ และจุดประสงค์ของมนุษย์ ซึ่งต้องมีการกำหนดไว้ล่วงหน้าในระบบ ไม่ใช่แค่ให้ AI “เรียนรู้เอง” อย่างไม่มีขอบเขต เช่น AI ที่แนะนำข่าว ควรคำนึงถึงสุขภาพจิตและการรับรู้ของผู้ใช้ ไม่ใช่แค่ engagement หรือ click-through rate
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">7. การเปิดโอกาสให้มนุษย์ควบคุมได้เสมอ (Human-in-the-Loop)</h3>
  <p className="mb-4 leading-relaxed">
    แม้ระบบจะมี automation สูง แต่ต้องมีช่องทางให้มนุษย์เข้าแทรกแซงหรือตัดสินใจสุดท้ายในกรณีที่มีความเสี่ยงสูง เช่น ในระบบแพทย์หรือยุติธรรม เพื่อให้มั่นใจว่า AI เป็นเครื่องมือช่วย ไม่ใช่สิ่งที่แทนที่ความรับผิดชอบของมนุษย์
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">8. ความชัดเจนในการแจ้งเตือนว่าเป็น AI</h3>
  <p className="mb-4 leading-relaxed">
    ผู้ใช้งานควรรับรู้ว่าตนกำลังโต้ตอบกับ AI ไม่ใช่มนุษย์ เช่น chatbot หรือ voice assistant ควรมีการแจ้งให้ชัดเจน เพื่อป้องกันความเข้าใจผิดที่อาจส่งผลต่อพฤติกรรม การให้ข้อมูล หรือการตัดสินใจ
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">9. ความเป็นธรรมทางวัฒนธรรม (Cultural Fairness)</h3>
  <p className="mb-4 leading-relaxed">
    ระบบที่นำไปใช้ในสังคมต่างวัฒนธรรมต้องปรับให้เหมาะสมกับบริบท ไม่ใช้โมเดลที่ถูกฝึกมาในประเทศหนึ่งแล้วนำไปใช้อีกประเทศโดยตรง เช่น ระบบประเมินพฤติกรรมการเรียนควรอิงกับบริบทท้องถิ่น ไม่ใช่มาตรฐานจากตะวันตกเพียงอย่างเดียว
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p className="mb-2">
      การออกแบบระบบ AI ที่เคารพมนุษย์คือการสร้างระบบที่ “รับฟัง เข้าใจ และปกป้อง” ไม่ใช่แค่ “คำนวณและตอบกลับ” เท่านั้น
    </p>
    <p>
      สิ่งสำคัญไม่ใช่แค่ผลลัพธ์ แต่คือ “วิธีที่ระบบเดินทางไปถึงผลลัพธ์นั้น” ว่าสอดคล้องกับคุณค่าของมนุษย์หรือไม่
    </p>
  </div>
</section>


        
<section id="workflow" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">7. การฝัง Fairness เข้าไปใน Workflow ทีมพัฒนา</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <p className="mb-4 leading-relaxed">
    การสร้างระบบ AI ที่เป็นธรรมไม่ได้เป็นเพียงหน้าที่ของนักวิจัยด้านอัลกอริทึมเท่านั้น แต่ต้องเริ่มตั้งแต่ระดับองค์กรและกระบวนการทำงานของทีมพัฒนา ระบบที่เคารพ Fairness จำเป็นต้องถูกฝังไว้ตั้งแต่ต้นน้ำของการพัฒนา ตั้งแต่การกำหนดเป้าหมาย การเก็บข้อมูล ไปจนถึงการนำระบบไปใช้จริง
  </p>

  <h3 className="text-xl font-semibold mb-3">1. การกำหนดนิยามของ Fairness ร่วมกันในทีม</h3>
  <p className="mb-4 leading-relaxed">
    Fairness ไม่ได้มีนิยามเดียวสำหรับทุกบริบท ทีมพัฒนาควรร่วมกันกำหนดนิยามของ Fairness ที่สอดคล้องกับบริบทของโปรเจกต์ เช่น ในระบบการให้กู้ยืม นิยามของความยุติธรรมอาจหมายถึงการที่ผู้กู้ทุกกลุ่มมีโอกาสได้รับอนุมัติในอัตราที่ใกล้เคียงกันเมื่อเงื่อนไขทางการเงินใกล้เคียงกัน
  </p>

  <h3 className="text-xl font-semibold mb-3">2. การฝึกอบรมและสร้างความเข้าใจในทีม</h3>
  <p className="mb-4 leading-relaxed">
    ทีมพัฒนาทั้งฝั่งเทคนิคและธุรกิจควรได้รับการอบรมเกี่ยวกับประเด็น Fairness และ Bias ใน AI เพื่อให้เข้าใจว่าแต่ละการตัดสินใจในงานวิจัย วิศวกรรม หรือ UX/UI สามารถส่งผลต่อความเป็นธรรมของระบบได้ทั้งหมด
  </p>

  <h3 className="text-xl font-semibold mb-3">3. การสร้าง Feedback Loop จากผู้ใช้งานจริง</h3>
  <p className="mb-4 leading-relaxed">
    ผู้ใช้งานควรมีช่องทางในการส่งฟีดแบ็ค เช่น การรายงานผลลัพธ์ที่รู้สึกว่าไม่ยุติธรรม ระบบควรเก็บข้อมูลเหล่านี้ไว้ใน Log และส่งต่อให้ทีมตรวจสอบอย่างเป็นระบบ เช่น การรวบรวมตัวอย่างที่ถูกปฏิเสธการให้กู้โดยไม่ทราบสาเหตุ
  </p>

  <h3 className="text-xl font-semibold mb-3">4. การประเมิน Fairness ในทุกขั้นตอนของ ML Pipeline</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li><strong>ขั้นตอนเก็บข้อมูล:</strong> ตรวจสอบว่าไม่มี bias จากแหล่งข้อมูล เช่น ความถี่ของข้อมูลจากกลุ่มใดกลุ่มหนึ่งมากเกินไป</li>
    <li><strong>ขั้นตอน preprocessing:</strong> พิจารณาว่า feature ที่เลือกใช้มีความเกี่ยวข้องกับ sensitive attribute หรือไม่</li>
    <li><strong>ขั้นตอน model training:</strong> ใช้ fairness-aware loss function หรือ constraints</li>
    <li><strong>ขั้นตอน evaluation:</strong> ใช้ fairness metrics เช่น Equalized Odds, Demographic Parity</li>
    <li><strong>ขั้นตอน deployment:</strong> มีระบบ monitoring เพื่อตรวจสอบความเสถียรของ fairness เมื่อมีข้อมูลใหม่</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">5. ใช้ Audit Trail และ Logging สำหรับความโปร่งใส</h3>
  <p className="mb-4 leading-relaxed">
    ทุกการตัดสินใจของระบบ AI ควรมีการบันทึกอย่างเป็นระบบ โดยเฉพาะการตัดสินใจที่ส่งผลต่อชีวิตผู้ใช้ เช่น การให้สินเชื่อ การเลือกผู้สมัครงาน ควรมี log ของ input/output, confidence score และเหตุผลที่เลือกตัดสินใจแบบนั้น เพื่อให้สามารถย้อนตรวจสอบได้
  </p>

  <h3 className="text-xl font-semibold mb-3">6. ตั้ง Ethical Review Board ในโครงการที่มีผลกระทบสูง</h3>
  <p className="mb-4 leading-relaxed">
    โครงการ AI ขนาดใหญ่ โดยเฉพาะที่เกี่ยวข้องกับสาธารณชน เช่น ภาครัฐ การแพทย์ หรือการเงิน ควรมีคณะกรรมการตรวจสอบด้านจริยธรรม (Ethical Review Board) ที่มีตัวแทนจากหลายฝ่าย เพื่อกลั่นกรองโมเดลก่อน deployment จริง
  </p>

  <h3 className="text-xl font-semibold mb-3">7. กำหนด Owner สำหรับ Fairness</h3>
  <p className="mb-4 leading-relaxed">
    เช่นเดียวกับ Security หรือ Performance ควรมีเจ้าหน้าที่หรือผู้รับผิดชอบหลักที่ดูแล Fairness ในระบบ ตั้งแต่การกำกับเป้าหมาย การเลือกชุดข้อมูล การติดตาม metric ไปจนถึงการสื่อสารกับ Stakeholder
  </p>

  <h3 className="text-xl font-semibold mb-3">8. การประเมินผลกระทบต่อสังคม (Social Impact Assessment)</h3>
  <p className="mb-4 leading-relaxed">
    ทุกครั้งที่พัฒนา AI ใหม่ โดยเฉพาะในระดับโปรดักชัน ควรมีการวิเคราะห์ว่าโมเดลนี้จะส่งผลต่อใครบ้าง มีใครที่อาจได้รับผลกระทบเชิงลบ เช่น กลุ่มผู้ใช้ที่ไม่ใช่ภาษาอังกฤษ, ผู้หญิง, กลุ่มชาติพันธุ์ หรือผู้พิการ
  </p>

  <h3 className="text-xl font-semibold mb-3">9. ตัวอย่าง Best Practice จากองค์กรระดับโลก</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li><strong>Google:</strong> มี Responsible AI team พร้อมแนวทาง Model Cards และ Dataset Documentation</li>
    <li><strong>Microsoft:</strong> สร้าง Fairlearn library สำหรับปรับ bias ใน ML pipeline</li>
    <li><strong>IBM:</strong> สร้าง AI Fairness 360 (AIF360) สำหรับวิเคราะห์และลด bias</li>
  </ul>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      Fairness ต้องถูกออกแบบเป็น “กระบวนการ” ไม่ใช่ “คุณสมบัติเพิ่มเติม” เมื่อฝังความเป็นธรรมไว้ในทุกขั้นตอนของ workflow จะช่วยให้ระบบ AI มีความน่าเชื่อถือ โปร่งใส และยั่งยืนในการใช้งานจริง
    </p>
  </div>
</section>

<section id="case-study" className="mb-16 scroll-mt-32 min-h-[400px]">
    <h2 className="text-2xl font-semibold mb-4 text-center">8. Case Study: ระบบคัดกรองผู้สมัครงาน</h2>
    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

    <p className="mt-6 mb-4 leading-relaxed">
      ระบบคัดกรองผู้สมัครงานที่ใช้ AI ได้รับความนิยมเพิ่มขึ้น เนื่องจากสามารถช่วยประหยัดเวลา ลดภาระของฝ่ายบุคคล และคัดเลือกผู้สมัครได้อย่างรวดเร็ว อย่างไรก็ตาม หากโมเดลไม่ได้รับการออกแบบอย่างรอบคอบ อาจส่งผลให้เกิดอคติ (bias) หรือการเลือกปฏิบัติที่ไม่ยุติธรรม โดยเฉพาะเมื่อข้อมูลที่ใช้ในการฝึกโมเดลมีความเอนเอียงมาตั้งแต่ต้น
    </p>

    <h3 className="text-xl font-semibold mb-3">ปัญหาที่พบในระบบจริง</h3>
    <p className="mb-4 leading-relaxed">
      บริษัทเทคโนโลยีแห่งหนึ่งใช้ AI ในการวิเคราะห์เรซูเม่ของผู้สมัครงาน เพื่อคัดกรองเบื้องต้นและจัดลำดับความเหมาะสม โมเดลได้รับการฝึกจากข้อมูลเรซูเม่และประวัติการจ้างงานในอดีตขององค์กร ซึ่งโดยมากเป็นข้อมูลของผู้ชายในตำแหน่งวิศวกรซอฟต์แวร์ที่เคยได้รับคัดเลือก
    </p>
    <p className="mb-4 leading-relaxed">
      ผลที่ตามมาคือโมเดลเริ่มลดคะแนนของเรซูเม่ที่มีคำว่า "หญิงล้วน" หรือกิจกรรมที่เกี่ยวข้องกับผู้หญิง และให้คะแนนสูงกว่ากับคำที่พบในเรซูเม่ของผู้ชาย เช่น "ฟุตบอล", "ช่างเทคนิค", หรือโรงเรียนชายล้วน โดยไม่เกี่ยวข้องกับคุณสมบัติทางเทคนิคที่แท้จริง
    </p>

    <h3 className="text-xl font-semibold mb-3">ขั้นตอนการวิเคราะห์และแก้ไข</h3>
    <ul className="list-disc pl-6 space-y-2 mb-6">
      <li>ตรวจสอบ distribution ของคะแนนที่โมเดลให้กับผู้สมัครแยกตามเพศ</li>
      <li>ใช้ Group Confusion Matrix เปรียบเทียบอัตราความแม่นยำระหว่างผู้หญิงและผู้ชาย</li>
      <li>วิเคราะห์ SHAP value เพื่อตรวจสอบว่า feature ไหนมีผลต่อคะแนนมากที่สุด</li>
      <li>พบว่าคำบางคำในเรซูเม่ที่ไม่เกี่ยวข้องกับทักษะมีผลต่อ prediction</li>
      <li>ประเมิน Disparate Impact Ratio เพื่อวัดผลกระทบของ bias ที่มีต่อการคัดเลือก</li>
    </ul>

    <h3 className="text-xl font-semibold mb-3">การปรับปรุงเพื่อเพิ่ม Fairness</h3>
    <div className="grid md:grid-cols-2 gap-6 mb-6">
      <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border shadow">
        <h4 className="font-semibold mb-2">Preprocessing</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ลบ feature ที่เกี่ยวข้องกับเพศออกจาก training data</li>
          <li>ใช้ sampling แบบสมดุลเพื่อให้กลุ่มเพศมี representation ที่เท่าเทียม</li>
          <li>ใช้ reweighing เพื่อปรับน้ำหนักตัวอย่างตามความหลากหลายของกลุ่ม</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border shadow">
        <h4 className="font-semibold mb-2">In-processing & Post-processing</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ใช้ fairness-aware loss function เพื่อ penalize การตัดสินใจที่ไม่เป็นธรรม</li>
          <li>ฝึก adversarial model เพื่อลบความสามารถของโมเดลในการแยกเพศ</li>
          <li>ปรับค่าความมั่นใจในขั้นตอน post-processing เพื่อ balance group outcome</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mb-3">ผลลัพธ์หลังจากการปรับปรุง</h3>
    <p className="mb-4 leading-relaxed">
      หลังการปรับโมเดล พบว่าอัตราการคัดเลือกผู้สมัครในแต่ละเพศมีความใกล้เคียงกันมากขึ้น จากเดิมที่ผู้ชายได้รับการคัดเลือก 72% และผู้หญิงเพียง 28% กลายเป็น 52% และ 48% ตามลำดับ ในขณะที่ accuracy โดยรวมลดลงจาก 87% เป็น 83% ซึ่งถือเป็น trade-off ที่องค์กรสามารถยอมรับได้
    </p>
    <p className="mb-4 leading-relaxed">
      นอกจากนี้ ยังมีการจัดทำ Fairness Dashboard เพื่อตรวจสอบค่าความเป็นธรรมเป็นประจำ และใช้ audit log ในการบันทึกคำแนะนำหรือเหตุผลการคัดกรองของระบบอย่างโปร่งใส
    </p>

    <h3 className="text-xl font-semibold mb-3">ประเด็นเพิ่มเติมที่ได้เรียนรู้</h3>
    <ul className="list-disc pl-6 space-y-2 mb-6">
      <li>แม้จะลบข้อมูล sensitive ออก แต่โมเดลอาจยังเรียนรู้ bias ผ่าน proxy เช่น ชื่อโรงเรียน</li>
      <li>ต้องใช้หลายเทคนิคร่วมกันทั้งก่อน ระหว่าง และหลังการฝึกโมเดล เพื่อให้เกิด fairness จริง</li>
      <li>ควรมีการนิยามความหมายของความยุติธรรมร่วมกันระหว่างทีม Data, HR และ Legal</li>
      <li>การให้ feedback จากผู้ใช้ระบบคัดกรองช่วยชี้จุดอ่อนของโมเดลที่อาจมองข้าม</li>
      <li>Fairness ไม่ได้หมายถึงการให้ผลลัพธ์เท่ากันเสมอ แต่หมายถึงการพิจารณาผลลัพธ์อย่างเป็นธรรม</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow">
      <strong className="block mb-2">Insight:</strong>
      การสร้างระบบ AI ที่ยุติธรรมไม่ใช่แค่การเขียนโค้ดให้ดี แต่ต้องคำนึงถึงที่มาของข้อมูล โครงสร้างสังคม และการมีส่วนร่วมของทุกภาคส่วนที่เกี่ยวข้อง การออกแบบโมเดลด้วยความเข้าใจบริบทและเป้าหมายขององค์กรเป็นกุญแจสำคัญในการสร้างความยั่งยืนในระยะยาว
    </div>
  </section>

  <section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">9. Insight: Accuracy vs Fairness</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในการพัฒนาและนำ AI ไปใช้จริง หนึ่งในความท้าทายที่สำคัญคือการหาสมดุลระหว่าง "ความแม่นยำของโมเดล (Accuracy)" และ "ความยุติธรรม (Fairness)" ทั้งสองแนวคิดนี้แม้จะดูแยกขาดกัน แต่กลับส่งผลซึ่งกันและกันอย่างซับซ้อน โดยเฉพาะอย่างยิ่งในระบบที่มีผลกระทบต่อชีวิตผู้คน เช่น การแพทย์ การจ้างงาน หรือการพิจารณาสินเชื่อ
  </p>

  <h3 className="text-xl font-semibold mb-3">ความแม่นยำ ≠ ความยุติธรรม</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่มี accuracy สูง อาจไม่ได้หมายความว่าผลลัพธ์นั้นเป็นธรรม ตัวอย่างเช่น หากโมเดลทำนายได้ถูกต้อง 95% แต่ความผิดพลาด 5% กลับกระจุกอยู่ในกลุ่มประชากรบางกลุ่ม (เช่น ผู้หญิง หรือชนกลุ่มน้อย) สิ่งนี้อาจถือเป็นการเลือกปฏิบัติ แม้ในภาพรวมจะดู "ดีพอ" ก็ตาม
  </p>

  <p className="mb-4 leading-relaxed">
    ในหลายกรณี โมเดลที่ optimize เพื่อให้ได้ accuracy สูงสุด อาจเลือก feature ที่มี bias แฝงอยู่ เพราะ feature เหล่านั้นช่วยให้โมเดลแม่นยำขึ้นใน training set โดยไม่ได้พิจารณาผลกระทบทางสังคมหรือจริยธรรม
  </p>

  <h3 className="text-xl font-semibold mb-3">Fairness ที่มากขึ้น อาจลด Accuracy บางกลุ่ม</h3>
  <p className="mb-4 leading-relaxed">
    การใช้ fairness constraint หรือการปรับ distribution ให้สมดุล อาจทำให้ accuracy โดยรวมลดลงเล็กน้อย เช่น การลดน้ำหนักของ feature ที่มี bias หรือลบ attribute ที่เป็น sensitive ออก ส่งผลให้โมเดลสูญเสียข้อมูลบางอย่างที่เคยช่วยให้ทำนายแม่น แต่สิ่งนี้เป็น trade-off ที่ยอมรับได้ หากช่วยให้ผลลัพธ์โดยรวมมีความยุติธรรมมากขึ้น
  </p>

  <p className="mb-4 leading-relaxed">
    ในเชิงเทคนิค วิธีการอย่าง Reweighing, Adversarial Debiasing หรือ Reject Option Classification ล้วนพยายามควบคุมไม่ให้ accuracy สูงสุดเป็นเป้าหมายเพียงอย่างเดียว แต่เน้น fairness เป็นเงื่อนไขร่วมในการ optimize
  </p>

  <h3 className="text-xl font-semibold mb-3">Case Study: ระบบคัดกรองผู้สมัครงาน</h3>
  <p className="mb-4 leading-relaxed">
    ในระบบคัดเลือกผู้สมัครงานของบริษัทหนึ่ง โมเดล ML ถูกฝึกจากข้อมูลผู้สมัครในอดีต ซึ่งมีการรับผู้ชายในตำแหน่งเทคนิคมากกว่าผู้หญิงถึง 80% โมเดลจึงเรียนรู้ว่าการเป็น "เพศชาย" สัมพันธ์กับโอกาสได้งาน ผลคือผู้หญิงจำนวนมากที่มีคุณสมบัติเหมาะสมกลับถูกตัดออก
  </p>
  <p className="mb-4 leading-relaxed">
    หลังจากวิเคราะห์ confusion matrix รายกลุ่ม พบว่า precision และ recall ของผู้หญิงต่ำกว่าผู้ชายอย่างมีนัยสำคัญ ทีมงานจึงปรับระบบโดยการลบ gender ออกจาก feature, ทำ sampling ให้สมดุล และใช้ fairness-aware loss function ผลลัพธ์คือ accuracy โดยรวมลดลง 1.2% แต่ fairness index เพิ่มขึ้นอย่างมาก
  </p>

  <h3 className="text-xl font-semibold mb-3">Fairness เป็นเรื่องของเจตนาและผลลัพธ์</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ถูกออกแบบมาให้แม่นยำ อาจไม่ได้ผิดโดยเจตนา แต่หากผลลัพธ์สุดท้ายสร้างความเหลื่อมล้ำในสังคม หรือ reinforce ความไม่ยุติธรรมเดิม เช่น ขยายช่องว่างทางเศรษฐกิจ หรือเลือกปฏิบัติโดยไม่รู้ตัว โมเดลนั้นก็ไม่ควรถูกมองว่า "ดีพอ"
  </p>

  <h3 className="text-xl font-semibold mb-3">ตัวชี้วัดที่ใช้ในการประเมิน</h3>
  <div className="grid md:grid-cols-2 gap-6 mb-6">
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-gray-300 dark:border-gray-700 shadow text-sm">
      <h4 className="font-semibold mb-2">ตัวชี้วัดทาง Accuracy</h4>
      <ul className="list-disc pl-5 space-y-2">
        <li>Accuracy</li>
        <li>Precision / Recall</li>
        <li>F1 Score</li>
        <li>ROC-AUC</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-gray-300 dark:border-gray-700 shadow text-sm">
      <h4 className="font-semibold mb-2">ตัวชี้วัดทาง Fairness</h4>
      <ul className="list-disc pl-5 space-y-2">
        <li>Demographic Parity</li>
        <li>Equal Opportunity</li>
        <li>Equalized Odds</li>
        <li>Disparate Impact</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3">การเลือกเป้าหมายตามบริบท</h3>
  <p className="mb-4 leading-relaxed">
    ในบางงาน เช่น ระบบแนะนำสินค้า อาจเน้น accuracy มากกว่า เพราะผลกระทบต่อผู้ใช้ไม่สูง แต่ในระบบประเมินสินเชื่อหรือคัดเลือกผู้สมัครงาน ความยุติธรรมควรถูกยกระดับเป็นเป้าหมายหลัก เพื่อหลีกเลี่ยงการเลือกปฏิบัติทางสังคม
  </p>

  <p className="mb-4 leading-relaxed">
    การเลือกโมเดลหรือปรับระบบจึงไม่ควรยึดติดกับ accuracy เพียงอย่างเดียว แต่ควรพิจารณาความเป็นธรรมและผลกระทบเชิงจริยธรรมในระยะยาวควบคู่กันไป
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">Insight สรุป</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">โมเดลที่แม่นที่สุดอาจไม่ใช่โมเดลที่ยุติธรรมที่สุด</p>
    <p className="mb-2">Fairness เป็นเป้าหมายที่ต้องการความตั้งใจ ไม่ใช่ผลพลอยได้จากการ optimize</p>
    <p className="mb-2">การประเมิน AI ต้องมองทั้งผลลัพธ์และผลกระทบ</p>
    <p className="italic text-sm">การพัฒนา AI ที่ดี ต้องพร้อมแลก accuracy เล็กน้อย เพื่อความยุติธรรมที่มากขึ้น</p>
  </div>
</section>


<section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">10. สรุปท้ายบท</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <p className="mb-4 leading-relaxed">
    การพัฒนา AI ในยุคปัจจุบันไม่อาจพิจารณาเฉพาะด้านเทคนิคอย่างความแม่นยำ (Accuracy) หรือประสิทธิภาพ (Performance) เพียงอย่างเดียวอีกต่อไป ความเป็นธรรม (Fairness) และจริยธรรม (Ethics) ได้กลายมาเป็นเสาหลักที่ขาดไม่ได้ หาก AI ต้องการอยู่ร่วมกับมนุษย์ในระบบสังคมที่ซับซ้อนและมีผลกระทบในระดับกว้าง
  </p>

  <p className="mb-4 leading-relaxed">
    ตลอดบทเรียนนี้ได้กล่าวถึงองค์ประกอบสำคัญของ Fairness ใน AI ตั้งแต่แนวคิดพื้นฐาน ไปจนถึงเครื่องมือที่ใช้ตรวจสอบความเป็นธรรม และแนวทางออกแบบระบบที่รับผิดชอบ ตั้งแต่ต้นน้ำของข้อมูล ไปจนถึงการออกแบบกระบวนการพัฒนา AI ที่โปร่งใสและตรวจสอบได้จริง
  </p>

  <p className="mb-4 leading-relaxed">
    หนึ่งในประเด็นสำคัญคือ AI ไม่ได้เป็นกลางโดยธรรมชาติ พฤติกรรมของโมเดลสะท้อนข้อมูลที่ใช้ในการฝึก หากข้อมูลมีอคติ โมเดลก็มีแนวโน้มจะรับอคตินั้นไปด้วยอย่างไม่รู้ตัว การทำความเข้าใจประเภทของ Bias เช่น Historical Bias, Measurement Bias, Algorithmic Bias จึงเป็นกุญแจสำคัญที่จะทำให้ทีมพัฒนารับมือกับความไม่เป็นธรรมได้อย่างเป็นระบบ
  </p>

  <p className="mb-4 leading-relaxed">
    Fairness ไม่ใช่แนวคิดนามธรรม แต่เป็นสิ่งที่ตรวจสอบและวัดได้ เช่น การใช้ Demographic Parity หรือ Equal Opportunity Metric ในการเปรียบเทียบผลลัพธ์ของโมเดลระหว่างกลุ่มต่าง ๆ อย่างเป็นธรรม การสร้าง Group-based Confusion Matrix ก็สามารถเผยให้เห็นจุดอ่อนที่ซ่อนอยู่ของโมเดลที่ดูเหมือนแม่นยำแต่ไม่ยุติธรรมในบางกลุ่ม
  </p>

  <p className="mb-4 leading-relaxed">
    นอกจากนี้ ยังสามารถเลือกใช้เทคนิคการแก้ไขที่เหมาะสมกับขั้นตอนการพัฒนา ไม่ว่าจะเป็นการจัดการข้อมูลล่วงหน้า (Preprocessing), การฝังแนวคิด Fairness เข้าไปในโมเดลระหว่างการฝึก (In-processing), หรือการแก้ไขผลลัพธ์หลังการพยากรณ์ (Post-processing) 
  </p>

  <p className="mb-4 leading-relaxed">
    สิ่งที่สำคัญไม่แพ้กันคือการฝัง Fairness เข้าไปในวัฒนธรรมองค์กร ไม่ว่าจะเป็นการสร้าง Feedback Loop จากผู้ใช้จริง การตั้ง Ethical Review Board ที่มีอำนาจตรวจสอบอย่างแท้จริง หรือการออกแบบ Audit Trail เพื่อเก็บบันทึกทุกการตัดสินใจของระบบ AI เพื่อการตรวจสอบย้อนหลังในอนาคต
  </p>

  <p className="mb-4 leading-relaxed">
    หลายกรณีศึกษาแสดงให้เห็นว่า แม้ความพยายามในการทำให้ระบบเป็นธรรมอาจแลกมากับการลดลงของ Accuracy บางส่วน แต่การลด Bias และสร้างความไว้วางใจจากผู้ใช้งานกลับมีผลกระทบเชิงบวกที่ยั่งยืนต่อทั้งองค์กรและสังคม
  </p>

  <p className="mb-4 leading-relaxed">
    ความท้าทายที่แท้จริงของ AI คือการสร้างระบบที่ไม่เพียง “ตอบถูก” แต่ “คิดถูก” และ “ตัดสินใจโดยคำนึงถึงผลกระทบต่อมนุษย์” ในบริบทจริง การนำหลักการของ Fairness และ Ethics ไปใช้ในการออกแบบ AI เปรียบเสมือนการสร้างภูมิคุ้มกันทางปัญญาให้กับเทคโนโลยี เพื่อให้สามารถอยู่ร่วมกับสังคมมนุษย์อย่างยั่งยืนและปลอดภัย
  </p>

  <p className="mb-4 leading-relaxed">
    ในระยะยาว AI ที่ประสบความสำเร็จจะไม่ใช่ AI ที่ทำนายได้แม่นที่สุดเพียงอย่างเดียว แต่คือ AI ที่ได้รับความไว้วางใจจากสังคม มีความโปร่งใสในการตัดสินใจ ปกป้องข้อมูลส่วนบุคคลของผู้ใช้ และแสดงถึงความรับผิดชอบต่อผลกระทบที่เกิดขึ้นจากการทำงานของมัน
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight สรุปท้ายบท:</p>
    <ul className="list-disc pl-6 space-y-2 text-sm">
      <li>AI ไม่สามารถหลีกเลี่ยง Bias ได้หากไม่ถูกออกแบบอย่างใส่ใจ</li>
      <li>Fairness คือองค์ประกอบพื้นฐานของความรับผิดชอบ ไม่ใช่แค่คุณสมบัติเสริม</li>
      <li>การตรวจสอบความเป็นธรรมต้องอยู่ในทุกขั้นตอน ตั้งแต่เก็บข้อมูลจนถึงการใช้งาน</li>
      <li>ความโปร่งใสและสิทธิของผู้ใช้งานคือหัวใจของระบบ AI ที่มีจริยธรรม</li>
      <li>การยอมลด Accuracy เล็กน้อยเพื่อเพิ่ม Fairness อาจเป็นสิ่งที่คุ้มค่าในระยะยาว</li>
      <li>การสร้างวัฒนธรรมของ Fair AI ต้องอาศัยทั้งกระบวนการ เทคนิค และความตั้งใจ</li>
    </ul>
  </div>
</section>


        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[400px]">
          <MiniQuiz_Day14 theme={theme} />
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
        <ScrollSpy_Ai_Day14 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day14_ModelFairness;
