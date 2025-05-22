import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day39 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day39";
import MiniQuiz_Day39 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day39";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day39_BestPractices = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day39_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day39_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day39_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day39_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day39_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day39_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day39_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day39_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day39_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day39_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day39_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day39_12").format("auto").quality("auto").resize(scale().width(501));
  const img13 = cld.image("Day39_13").format("auto").quality("auto").resize(scale().width(501));
  const img14 = cld.image("Day39_14").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 39: Supervised Learning Best Practices</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    1. บทนำ: ทำไม Best Practices จึงสำคัญใน Supervised Learning?
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className=" prose-lg  max-w-none space-y-6 text-base leading-relaxed">
    <h3>บทบาทของ Best Practices ในบริบทของการเรียนรู้แบบมีผู้สอน</h3>
    <p>
      Supervised Learning เป็นแกนหลักของโมเดล Machine Learning จำนวนมาก ไม่ว่าจะเป็นการจำแนกภาพ การรู้จำเสียง หรือการทำนายข้อมูลเชิงตัวเลข
      อย่างไรก็ตาม ผลลัพธ์ของโมเดลไม่ได้ขึ้นอยู่กับอัลกอริทึมเพียงอย่างเดียว แต่ขึ้นอยู่กับความแม่นยำในการกำหนดกระบวนการทำงานตั้งแต่ต้นน้ำถึงปลายน้ำ
      ซึ่งเรียกรวมกันว่า Best Practices
    </p>

    <h3>ผลกระทบของการขาดแนวทางที่ดี</h3>
    <ul className="list-disc pl-6">
      <li>โมเดลอาจเรียนรู้จากข้อมูลที่ไม่มีคุณภาพ ส่งผลให้ผลลัพธ์มี bias</li>
      <li>ความสามารถในการ generalize ไปยังข้อมูลจริงนอกชุด training ลดลง</li>
      <li>กระบวนการพัฒนาโมเดลกลายเป็นแบบ trial-and-error ที่ไม่มีโครงสร้าง</li>
    </ul>

    <h3>แนวคิดจากงานวิจัยระดับโลก</h3>
    <p>
      งานวิจัยของ Andrew Ng (Stanford) และทีมจาก DeepMind เน้นย้ำว่า best practices ไม่เพียงช่วยให้โมเดลเรียนรู้ได้ดีขึ้น
      แต่ยังช่วยลดทรัพยากรการฝึก (compute) และความซับซ้อนของ pipeline ได้อย่างมาก
    </p>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner">
      <p className="font-semibold">Insight Box</p>
      <p className="mt-2">
        "โมเดลที่ดีเกิดจากการวางแผนที่ดี ไม่ใช่แค่จาก parameter ที่ถูกปรับจูน" — จากบันทึกการสอนของ Prof. Christopher Ré (Stanford AI Lab)
      </p>
    </div>

    <h3>กรณีเปรียบเทียบ</h3>
    <table className="table-auto w-full text-left border mt-6">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="px-4 py-2">หัวข้อ</th>
          <th className="px-4 py-2">ไม่มี Best Practice</th>
          <th className="px-4 py-2">มี Best Practice</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การจัดการข้อมูล</td>
          <td className="border px-4 py-2">สุ่มข้อมูลโดยไม่วิเคราะห์</td>
          <td className="border px-4 py-2">มีขั้นตอนตรวจสอบข้อมูล, กำจัด outlier</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การฝึกโมเดล</td>
          <td className="border px-4 py-2">ปรับ hyperparameter แบบเดาสุ่ม</td>
          <td className="border px-4 py-2">ใช้ cross-validation และ grid search</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความสามารถของโมเดล</td>
          <td className="border px-4 py-2">มัก overfit และไม่เสถียร</td>
          <td className="border px-4 py-2">เรียนรู้ได้ลึกและเสถียรระยะยาว</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner mt-8">
      <p className="font-semibold">Highlight</p>
      <p className="mt-2">
        การตั้งระบบ Best Practices ที่ชัดเจนตั้งแต่ต้นเป็นหนึ่งในเครื่องมือที่สำคัญที่สุดในการทำให้โมเดลสามารถ deploy ได้ใน production environment
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Ng, Andrew. “Machine Learning Yearning.” deeplearning.ai, 2018</li>
      <li>Ré, C. et al. “Snorkel: Rapid Training Data Creation with Weak Supervision.” VLDB, 2017</li>
      <li>Amodei, D. et al. “Concrete Problems in AI Safety.” arXiv:1606.06565</li>
    </ul>
  </div>
</section>


     <section id="problem" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. วางโจทย์ให้ถูกตั้งแต่ต้น</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-6 text-base leading-relaxed">
    <h3>นิยามโจทย์ที่สอดคล้องกับวัตถุประสงค์เชิงธุรกิจ</h3>
    <p>
      การวางโจทย์ปัญหาที่ชัดเจนถือเป็นจุดเริ่มต้นที่สำคัญใน Supervised Learning โมเดลไม่สามารถแก้ปัญหาที่ไม่ถูกนิยามได้อย่างมีประสิทธิภาพ
      งานวิจัยจาก Harvard Business School ระบุว่าโมเดล Machine Learning ที่ล้มเหลวจำนวนมากมีสาเหตุจากการนิยามปัญหาผิด (problem misalignment)
    </p>

    <h3>ประเภทของโจทย์และผลกระทบต่อการออกแบบ</h3>
    <ul className="list-disc pl-6">
      <li>Classification: ใช้กับปัญหาที่มี class ชัดเจน เช่น spam detection</li>
      <li>Regression: ใช้กับปัญหาที่มีค่าต่อเนื่อง เช่น price prediction</li>
      <li>Ranking: ใช้กับระบบแนะนำ เช่น movie recommendations</li>
    </ul>

    <h3>ตัวอย่างจากงานวิจัยของ Stanford</h3>
    <p>
      ทีม Stanford ML Group ภายใต้การนำของ Andrew Ng ได้ย้ำในรายงานปี 2021 ว่าโจทย์ผิดจะทำให้โมเดลดีแค่ไหนก็ไม่สามารถใช้งานจริงได้ เช่น การพยายามทำนาย diagnosis โดยไม่แยก causal feature กับ correlational noise
    </p>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner">
      <p className="font-semibold">Insight Box</p>
      <p className="mt-2">
        "การออกแบบโจทย์ต้องเริ่มจากผลลัพธ์ที่ต้องการ ไม่ใช่จาก dataset ที่มี" — จากบันทึก Stanford CS229 Lecture
      </p>
    </div>

    <h3>การใช้ Problem Tree ในการวิเคราะห์</h3>
    <p>
      การใช้แผนผังปัญหา (Problem Tree) ช่วยให้เห็นสาเหตุและผลลัพธ์ของปัญหาอย่างเป็นระบบ ช่วยให้ทีมสามารถเจาะลึกว่าควรวาง prediction target ที่จุดใด
    </p>

    <h3>เปรียบเทียบ: โจทย์ที่นิยามถูกกับผิด</h3>
    <table className="table-auto w-full text-left border mt-6">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="px-4 py-2">ประเด็น</th>
          <th className="px-4 py-2">นิยามโจทย์ผิด</th>
          <th className="px-4 py-2">นิยามโจทย์ถูก</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Input Feature</td>
          <td className="border px-4 py-2">เลือก feature ตามความสะดวก</td>
          <td className="border px-4 py-2">เลือกจาก causal path ที่ตรวจสอบแล้ว</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Target Output</td>
          <td className="border px-4 py-2">กำหนด label ตามการมีอยู่ของข้อมูล</td>
          <td className="border px-4 py-2">กำหนดจากเป้าหมายเชิงกลยุทธ์</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ประสิทธิภาพโมเดล</td>
          <td className="border px-4 py-2">accuracy สูงแต่ใช้งานจริงไม่ได้</td>
          <td className="border px-4 py-2">generalize ได้ดีใน environment จริง</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner mt-8">
      <p className="font-semibold">Highlight</p>
      <p className="mt-2">
        การวางโจทย์ที่ดีช่วยลดค่าใช้จ่ายในการ annotation, training และ tuning ได้มากถึง 60% จากผลการศึกษาของ Carnegie Mellon University (CMU)
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Ng, Andrew. “CS229 Lecture Notes.” Stanford University</li>
      <li>Obermeyer, Z. et al. “Dissecting Bias in Predictive Algorithms.” Science, 2019</li>
      <li>CMU ML Department. “Best Practices in Problem Definition.” ML Systems 2020</li>
    </ul>
  </div>
</section>


     <section id="data-strategy" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Data Collection & Labeling Strategy</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8">
    <h3>แนวทางการเก็บข้อมูลอย่างเป็นระบบ</h3>
    <p>
      การเก็บข้อมูลในระบบ supervised learning จำเป็นต้องมีความสมดุลทางสถิติและครอบคลุมบริบทการใช้งานที่แท้จริง (real-world distribution)
      การเก็บข้อมูลแบบสุ่มอย่างไม่มีโครงสร้าง มักนำไปสู่ model bias และ overfitting จากความไม่สมดุลของข้อมูล (data imbalance) ซึ่งเป็นข้อผิดพลาดที่เกิดขึ้นซ้ำในงานวิจัยหลายชิ้น
      ตัวอย่างเช่น การเก็บข้อมูลจากผู้ใช้เพียงกลุ่มเดียวในงาน health AI อาจทำให้โมเดลไม่สามารถใช้งานได้กับประชากรวงกว้าง
    </p>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยจาก Stanford (2021) ระบุว่า 80% ของ performance degradation ใน real-world deployment มาจากการขาด representativeness ใน data collection phase
      </p>
    </div>

    <h3>กลยุทธ์ในการเก็บข้อมูล</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ stratified sampling เพื่อให้ข้อมูลทุกคลาสถูกเก็บในสัดส่วนที่เท่ากัน</li>
      <li>ใช้ active learning เพื่อเลือกตัวอย่างที่ยังไม่มี label โดยอิงจาก uncertainty ของโมเดล</li>
      <li>ออกแบบ data pipeline ที่รองรับการติดตามแหล่งที่มาของข้อมูล (data provenance)</li>
    </ul>

    <h3>Labeling: ปัจจัยที่ส่งผลต่อคุณภาพ</h3>
    <p>
      การ labeling ต้องคำนึงถึง consistency, accuracy และ inter-rater agreement โดยเฉพาะในข้อมูลที่มีความกำกวม เช่น ใน NLP หรือภาพทางการแพทย์
    </p>

    <table className="table-auto border border-gray-300 w-full text-sm">
      <thead>
        <tr className="bg-gray-200">
          <th className="border px-4 py-2">แนวทาง Labeling</th>
          <th className="border px-4 py-2">ข้อดี</th>
          <th className="border px-4 py-2">ข้อควรระวัง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Expert Labeling</td>
          <td className="border px-4 py-2">คุณภาพสูง เชื่อถือได้</td>
          <td className="border px-4 py-2">ต้นทุนสูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Crowdsourcing</td>
          <td className="border px-4 py-2">ประหยัด สามารถ scale ได้</td>
          <td className="border px-4 py-2">อาจเกิด inconsistency ได้สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Weak Supervision</td>
          <td className="border px-4 py-2">ประหยัดเวลา ใช้ label จาก rule หรือ model เดิม</td>
          <td className="border px-4 py-2">อาจมี noise สูง ต้องใช้การ clean ซ้ำ</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-700 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานวิจัยจาก MIT และ CMU แนะนำการใช้ human-in-the-loop + programmatic labeling เพื่อลด bias และเพิ่ม label quality โดยเฉพาะใน dataset ขนาดใหญ่ที่ต้องการ scale อย่างรวดเร็ว
      </p>
    </div>

    <h3>ตัวอย่างการนำไปใช้</h3>
    <p>
      ในโปรเจกต์ด้าน autonomous driving อย่าง Waymo หรือ Tesla, การเก็บข้อมูลบนรถจำนวนมากและการ label ด้วย semi-automated tools ช่วยให้ได้ annotated dataset ที่ใหญ่และมีคุณภาพสูงในเวลาอันสั้น
    </p>

    <h3>สรุปแนวปฏิบัติที่แนะนำ</h3>
    <ul className="list-disc pl-6">
      <li>ออกแบบแผน data acquisition ล่วงหน้าให้รองรับความหลากหลายของข้อมูล</li>
      <li>ใช้หลายวิธีในการ labeling เพื่อ balance cost vs quality</li>
      <li>ควร log ข้อมูล meta-data เช่น location, device, annotator เพื่อวิเคราะห์ภายหลัง</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Stanford HAI 2021: "The Foundation of Trustworthy AI is Data"</li>
      <li>MIT-IBM Watson Lab: Weak Supervision Pipeline for Enterprise AI</li>
      <li>CMU ML Department (2022): Data Provenance in Large-scale ML Systems</li>
      <li>Science Journal (2021): The Cost of Bias in Human-Annotated Datasets</li>
    </ul>
  </div>
</section>


      <section id="split" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    4. Data Splitting ที่ถูกต้อง
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none text-base leading-relaxed space-y-10">
    <h3>4.1 ความสำคัญของการแบ่งข้อมูลอย่างเป็นระบบ</h3>
    <p>
      การแบ่งข้อมูลเป็นขั้นตอนที่มีอิทธิพลโดยตรงต่อความสามารถของโมเดลในการเรียนรู้และการ generalize ไปยังข้อมูลใหม่
      โดยเฉพาะอย่างยิ่งในงานที่ต้องการความแม่นยำสูง เช่น การวินิจฉัยทางการแพทย์หรือการตรวจจับความผิดปกติในระบบการเงิน
    </p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-500 shadow">
      <p className="font-semibold">Insight:</p>
      <p>
        หากการแบ่งข้อมูลไม่สะท้อน distribution ที่แท้จริง อาจทำให้โมเดล overfit หรือ underfit โดยไม่สามารถประเมินประสิทธิภาพจริงได้
      </p>
    </div>

    <h3>4.2 ประเภทของ Data Splitting</h3>
    <ul className="list-disc pl-6">
      <li><strong>Training Set:</strong> ใช้ในการเรียนรู้พารามิเตอร์ของโมเดล</li>
      <li><strong>Validation Set:</strong> ใช้ในการปรับ hyperparameters และ early stopping</li>
      <li><strong>Test Set:</strong> ใช้ในการประเมินประสิทธิภาพสุดท้ายของโมเดล</li>
    </ul>

    <h3>4.3 เทคนิคการแบ่งข้อมูล</h3>
    <table className="table-auto w-full border border-gray-300 text-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border px-4 py-2 text-left">วิธี</th>
          <th className="border px-4 py-2 text-left">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Random Split</td>
          <td className="border px-4 py-2">สุ่มแบ่งข้อมูลออกเป็น training/validation/test ตามสัดส่วนที่กำหนด เช่น 70/15/15</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stratified Split</td>
          <td className="border px-4 py-2">แบ่งข้อมูลโดยรักษาสัดส่วนของคลาส (label distribution) ให้เท่ากันในทุกชุด</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Time-based Split</td>
          <td className="border px-4 py-2">ใช้ในข้อมูล time series โดยข้อมูลในอดีตจะเป็น training และข้อมูลในอนาคตเป็น validation/test</td>
        </tr>
      </tbody>
    </table>

    <h3>4.4 Best Practices จากสถาบันชั้นนำ</h3>
    <ul className="list-disc pl-6">
      <li>Stanford ML Course แนะนำให้ใช้ stratified split ใน classification task เพื่อป้องกัน label imbalance</li>
      <li>MIT Deep Learning รายงานว่าการ random split โดยไม่มี shuffling อาจทำให้ test set ซ้ำกับ training set ในบางกรณี</li>
      <li>IEEE Paper (2022) เสนอว่า time-based split เหมาะสมที่สุดกับงาน forecasting และ anomaly detection</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-500 shadow">
      <p className="font-semibold">Highlight:</p>
      <p>
        การแบ่งข้อมูลต้องสอดคล้องกับธรรมชาติของข้อมูลต้นทาง โดยเฉพาะใน domain-specific task เช่น NLP หรือ Medical Imaging
      </p>
    </div>

    <h3>4.5 สรุป</h3>
    <p>
      การเลือกเทคนิคการแบ่งข้อมูลที่เหมาะสมไม่เพียงส่งผลต่อผลลัพธ์โมเดล แต่ยังสะท้อนถึงความเข้าใจเชิงลึกของนักพัฒนา
      การนำแนวทางจากมหาวิทยาลัยชั้นนำและองค์กรวิจัยมาใช้ จะช่วยยกระดับคุณภาพของโมเดลใน production environment ได้อย่างมีนัยสำคัญ
    </p>
  </div>
</section>


 <section id="preprocessing" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    5. Preprocessing & Feature Engineering
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>การเตรียมข้อมูล: ขั้นตอนสำคัญก่อนป้อนสู่โมเดล</h3>
    <p>
      การ Preprocessing หรือการเตรียมข้อมูลก่อนฝึกโมเดลมีผลต่อประสิทธิภาพอย่างมีนัยสำคัญ โดยเฉพาะเมื่อข้อมูลมีลักษณะแตกต่างหรือผิดปกติ เช่น missing values, outliers หรือ scale ที่ไม่สม่ำเสมอ การเตรียมข้อมูลที่ดีจะช่วยให้โมเดลเรียนรู้ได้เร็วขึ้น และมีแนวโน้ม generalize ได้ดีกว่า
    </p>

    <ul className="list-disc pl-6">
      <li>การจัดการ missing values เช่น การใช้ mean imputation หรือ KNN-based</li>
      <li>การ normalize/standardize ข้อมูล เพื่อปรับ scale ให้อยู่ในช่วงเดียวกัน</li>
      <li>การ encoding ข้อมูลประเภท categorical เช่น One-hot encoding, Label encoding</li>
      <li>การจัดการกับ outliers ด้วย z-score หรือ IQR</li>
    </ul>

    <h3>การสร้างฟีเจอร์ที่มีคุณภาพ (Feature Engineering)</h3>
    <p>
      ในการเรียนรู้แบบมีผู้สอน (Supervised Learning) ฟีเจอร์ที่ดีคือกุญแจสำคัญในการพยากรณ์ที่แม่นยำ การสร้างฟีเจอร์ต้องคำนึงถึง domain knowledge และความสัมพันธ์เชิงเชิงสถิติระหว่างตัวแปรต้นกับผลลัพธ์
    </p>

    <ul className="list-decimal pl-6">
      <li>การสร้างฟีเจอร์ใหม่จากข้อมูลเดิม เช่น interaction terms หรือ polynomial features</li>
      <li>การเลือกฟีเจอร์โดยอิงจาก mutual information, correlation, หรือ recursive feature elimination</li>
      <li>การใช้ embeddings สำหรับ categorical features ที่มี cardinality สูง</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight:</p>
      <p>
        จากงานวิจัยของ Stanford (P. Liang, 2022) พบว่า 80% ของเวลาในการพัฒนาโมเดลถูกใช้ไปกับการ Preprocessing และ Feature Engineering มากกว่าการจูนโมเดลเสียอีก นี่แสดงให้เห็นว่าขั้นตอนนี้ไม่ใช่เรื่องเล็กน้อย แต่คือพื้นฐานที่กำหนด performance ของระบบทั้งหมด
      </p>
    </div>

    <h3>เปรียบเทียบเทคนิคการเตรียมข้อมูลเบื้องต้น</h3>
    <div className="overflow-x-auto">
      <table className="min-w-full table-auto border border-gray-300 text-sm">
        <thead className="bg-gray-200 dark:bg-gray-700">
          <tr>
            <th className="px-4 py-2 text-left">เทคนิค</th>
            <th className="px-4 py-2 text-left">ข้อดี</th>
            <th className="px-4 py-2 text-left">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800">
          <tr>
            <td className="px-4 py-2">StandardScaler</td>
            <td className="px-4 py-2">เหมาะกับข้อมูลที่แจกแจงปกติ</td>
            <td className="px-4 py-2">ไวต่อ outlier</td>
          </tr>
          <tr>
            <td className="px-4 py-2">MinMaxScaler</td>
            <td className="px-4 py-2">คงค่าในช่วง 0–1</td>
            <td className="px-4 py-2">อ่อนไหวต่อค่าสุดขั้ว</td>
          </tr>
          <tr>
            <td className="px-4 py-2">RobustScaler</td>
            <td className="px-4 py-2">ต้าน outlier ได้ดี</td>
            <td className="px-4 py-2">ข้อมูลที่ไม่กระจายตัวมากอาจไม่มีผลชัด</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ตัวอย่างโค้ด Feature Engineering</h3>
    <pre>
      <code className="language-python">
{`from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.DataFrame({"x1": [1,2,3], "x2": [4,5,6]})
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df)
print(X_poly)`}
      </code>
    </pre>

    <h3>แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS229: Machine Learning (Lecture Notes)</li>
      <li>MIT OpenCourseWare: Feature Engineering for ML</li>
      <li>arXiv: 1801.09414 - Practical Recommendations for Feature Engineering</li>
      <li>IEEE Transactions on Knowledge and Data Engineering, 2021</li>
    </ul>
  </div>
</section>


   <section id="dev-debug" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Model Development & Debugging</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความสำคัญของกระบวนการพัฒนาโมเดล</h3>
    <p>
      การพัฒนาโมเดล (Model Development) เป็นหัวใจสำคัญของการเรียนรู้แบบมีผู้สอน โดยรวมถึงการเลือกสถาปัตยกรรม, การกำหนด loss function, และการฝึกโมเดลผ่านกระบวนการ optimization การพัฒนาโมเดลไม่ควรเป็นกระบวนการที่ทำแบบสุ่ม แต่ต้องอาศัยการออกแบบและทดสอบเชิงระบบ
    </p>

    <h3>การ Debug โมเดลเชิงโครงสร้าง</h3>
    <p>
      เมื่อโมเดลมีพฤติกรรมที่ไม่คาดคิด การ Debug คือการวิเคราะห์ว่าโมเดลเข้าใจสิ่งใดผิด การตรวจสอบเชิงเชิงโครงสร้างควรรวมถึง:
    </p>
    <ul className="list-disc pl-6">
      <li>ตรวจสอบ input/output shapes และค่า gradients</li>
      <li>ตรวจสอบ training loss เทียบกับ validation loss</li>
      <li>ใช้ weight initialization แบบเหมาะสม</li>
      <li>วิเคราะห์ layer activation เพื่อดูความกระจายของฟีเจอร์</li>
    </ul>

    <div className="bg-blue-700 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยของ MIT CSAIL (Zhang et al., 2022) เสนอว่าโมเดลที่ออกแบบมาอย่างดีแต่มีข้อมูลฝึกไม่ครบถ้วนจะให้ผลลัพธ์ต่ำกว่าโมเดลที่ออกแบบไม่ดีแต่ใช้ debugging pipeline อย่างมีระบบ
      </p>
    </div>

    <h3>เทคนิค Debugging เชิงวิศวกรรม</h3>
    <p>
      สำหรับการ Debug โมเดลในสภาพแวดล้อมจริง จำเป็นต้องใช้เทคนิคที่มีความสามารถในการวิเคราะห์ runtime เช่น:
    </p>
    <ul className="list-disc pl-6">
      <li>การใช้ unit test สำหรับ preprocessing pipeline และ evaluation metric</li>
      <li>การ track training dynamics ด้วย TensorBoard หรือ Weights & Biases</li>
      <li>การใช้ assert statements เพื่อตรวจสอบค่าที่ไม่ควรเกิดขึ้น เช่น NaN</li>
    </ul>

    <h3>กรณีศึกษาการ Debug โมเดล</h3>
    <div className="overflow-x-auto">
      <table className="min-w-full table-auto border border-gray-300 text-sm">
        <thead className="bg-gray-200 dark:bg-gray-700">
          <tr>
            <th className="px-4 py-2 text-left">ปัญหา</th>
            <th className="px-4 py-2 text-left">อาการ</th>
            <th className="px-4 py-2 text-left">วิธีแก้ไข</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-800">
          <tr>
            <td className="px-4 py-2">Vanishing Gradients</td>
            <td className="px-4 py-2">Loss ไม่ลด</td>
            <td className="px-4 py-2">ใช้ ReLU และ BatchNorm</td>
          </tr>
          <tr>
            <td className="px-4 py-2">Overfitting</td>
            <td className="px-4 py-2">Accuracy สูงเฉพาะ training</td>
            <td className="px-4 py-2">เพิ่ม regularization เช่น dropout</td>
          </tr>
          <tr>
            <td className="px-4 py-2">Underfitting</td>
            <td className="px-4 py-2">Accuracy ต่ำทั้ง training และ validation</td>
            <td className="px-4 py-2">เพิ่ม model capacity</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ตัวอย่างโค้ด Debug แบบพื้นฐาน</h3>
    <pre>
      <code className="language-python">
{`import torch
assert not torch.isnan(loss), "Loss contains NaN"
if epoch % 10 == 0:
    print(f"Epoch {epoch}: Loss = {loss.item()}")`}
      </code>
    </pre>

    <h3>แหล่งข้อมูลอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning Book: Chapter on Optimization and Debugging</li>
      <li>arXiv: 2003.05247 - Troubleshooting Deep Learning Models</li>
      <li>IEEE Transactions on Neural Networks, 2021</li>
    </ul>
  </div>
</section>


    <section id="eval" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Cross-validation & Evaluation</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert space-y-8">
    <h3>แนวคิดพื้นฐานของ Cross-validation</h3>
    <p>Cross-validation เป็นเทคนิคที่ใช้เพื่อประเมินประสิทธิภาพของโมเดลในลักษณะที่เป็นกลาง โดยไม่พึ่งพา subset ใด subset หนึ่งมากเกินไป เทคนิคที่ใช้บ่อยที่สุดคือ K-Fold Cross-validation ซึ่งแบ่งข้อมูลออกเป็น k ส่วนเท่า ๆ กัน แล้วทำการฝึกและทดสอบ k ครั้ง โดยในแต่ละครั้งจะใช้ชุดหนึ่งเป็น test set และอีก k-1 ชุดเป็น training set</p>

    <div className="bg-yellow-700 p-4 rounded-lg border-l-4 border-yellow-500">
      <p className="font-semibold">Insight:</p>
      <p>การใช้ cross-validation อย่างถูกต้องช่วยลด variance ในการประเมินผลโมเดล และทำให้การเลือกโมเดลมีความน่าเชื่อถือมากขึ้น</p>
    </div>

    <h3>เทคนิคการประเมินผลที่ใช้กันทั่วไป</h3>
    <ul className="list-disc pl-6">
      <li><strong>Accuracy:</strong> เหมาะสำหรับข้อมูลที่มี class balance</li>
      <li><strong>Precision & Recall:</strong> ใช้เมื่อ class imbalance สำคัญ เช่นในงาน medical diagnosis</li>
      <li><strong>F1-score:</strong> ค่าเฉลี่ยแบบ harmonic ของ precision และ recall</li>
      <li><strong>ROC-AUC:</strong> ใช้วัด performance ของโมเดลใน binary classification</li>
    </ul>

    <h3>การเลือก Metric ที่เหมาะสม</h3>
    <p>การเลือก evaluation metric ที่เหมาะสมขึ้นอยู่กับบริบทของปัญหา เช่นในกรณีของ fraud detection จะต้องให้ความสำคัญกับ recall มากกว่า precision เพื่อไม่ให้พลาดกรณีสำคัญ</p>

    <div className="bg-blue-700 p-4 rounded-lg border-l-4 border-blue-500">
      <p className="font-semibold">Highlight:</p>
      <p>ไม่ควรใช้เพียงแค่ accuracy ในกรณีที่ข้อมูลไม่สมดุล เพราะอาจนำไปสู่การสรุปผลที่ผิดพลาดได้</p>
    </div>

    <h3>ตารางเปรียบเทียบ Evaluation Metrics</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border-collapse">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="border px-4 py-2">Metric</th>
            <th className="border px-4 py-2">เหมาะกับสถานการณ์</th>
            <th className="border px-4 py-2">ข้อควรระวัง</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Accuracy</td>
            <td className="border px-4 py-2">Class balance</td>
            <td className="border px-4 py-2">ลวงตาใน class imbalance</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Precision</td>
            <td className="border px-4 py-2">False positives สำคัญ</td>
            <td className="border px-4 py-2">อาจลด recall</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Recall</td>
            <td className="border px-4 py-2">False negatives สำคัญ</td>
            <td className="border px-4 py-2">อาจลด precision</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">F1-score</td>
            <td className="border px-4 py-2">สมดุล precision/recall</td>
            <td className="border px-4 py-2">ไม่เข้าใจง่าย</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ปัญหาที่พบบ่อยในการประเมินผล</h3>
    <ul className="list-disc pl-6">
      <li>Leakage: ข้อมูลจาก test set หลุดเข้า training set</li>
      <li>Overfitting metric: tuning เพื่อให้คะแนนสูงเกินจริง</li>
      <li>การประเมินจาก subset ที่ไม่ representative</li>
    </ul>

    <h3>ตัวอย่างจากงานวิจัย</h3>
    <p>งานจาก Stanford และ Google Research ได้แสดงให้เห็นว่า Cross-validation มีความสัมพันธ์อย่างแนบแน่นกับ generalization error ในบริบทของ deep learning โดยเฉพาะเมื่อใช้ techniques เช่น stratified sampling และ repeated k-fold</p>

    <ul className="list-disc pl-6">
      <li>Stanford CS229 Lecture Notes (Ng, 2021)</li>
      <li>“Evaluating Machine Learning Models” - Google Developers</li>
      <li>“A Survey on Model Evaluation Metrics” - IEEE Transactions on Neural Networks</li>
    </ul>
  </div>
</section>


    <section id="ensembling" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Model Selection & Ensembling</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8">
    <h3>การเลือกโมเดลที่เหมาะสม (Model Selection)</h3>
    <p>
      การเลือกโมเดลที่เหมาะสมมีผลโดยตรงต่อประสิทธิภาพของระบบ Supervised Learning โดยทั่วไปควรพิจารณาจากลักษณะของข้อมูล (เช่น ขนาด ความไม่สมดุล ความซับซ้อนของความสัมพันธ์) และวัตถุประสงค์ของการทำนาย (classification, regression, ranking เป็นต้น)
    </p>
    <p>
      ขั้นตอนเบื้องต้นสำหรับการเลือกโมเดลประกอบด้วย:
    </p>
    <ul>
      <li>เริ่มต้นจากโมเดลพื้นฐาน (Baseline Model) เช่น Logistic Regression หรือ Decision Tree</li>
      <li>เปรียบเทียบกับโมเดลเชิงลึกขึ้น เช่น Random Forest, XGBoost, หรือ Neural Network</li>
      <li>ประเมินด้วย Cross-validation ที่มั่นคง</li>
      <li>ตรวจสอบ trade-off ระหว่าง performance และ interpretability</li>
    </ul>

    <h3>เทคนิคการรวมโมเดล (Model Ensembling)</h3>
    <p>
      Ensembling คือกระบวนการรวมการพยากรณ์จากหลายโมเดลเข้าด้วยกันเพื่อลด variance และเพิ่ม robustness ให้กับระบบ โดยแบ่งได้เป็นหลายรูปแบบ:
    </p>
    <ul>
      <li><strong>Bagging:</strong> เช่น Random Forest ใช้การเทรนโมเดลหลายชุดด้วยข้อมูลที่สุ่มต่างกัน</li>
      <li><strong>Boosting:</strong> เช่น AdaBoost, XGBoost ใช้การเทรนโมเดลลำดับที่ปรับตามความผิดพลาด</li>
      <li><strong>Stacking:</strong> ใช้ meta-model รวมผลลัพธ์จากหลายโมเดลพื้นฐาน</li>
    </ul>

    <div className="bg-blue-700 rounded-lg p-4">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยจาก CMU และ Microsoft แสดงให้เห็นว่า Stacking สามารถเพิ่ม performance ได้สูงถึง 10-15% ในกรณีที่โมเดลพื้นฐานมี diversity เพียงพอ (Zhou et al., 2021)
      </p>
    </div>

    <h3>เกณฑ์การเลือกวิธี Ensembling</h3>
    <ul>
      <li>ชุดข้อมูลมี noise หรือ imbalance สูง → ใช้ Bagging</li>
      <li>ข้อมูลมีโครงสร้างลำดับหรือ hierarchical → ใช้ Boosting</li>
      <li>มีโมเดลที่หลากหลายและผลลัพธ์แตกต่างกัน → ใช้ Stacking</li>
    </ul>

    <h3>ตัวอย่างการเปรียบเทียบผลลัพธ์</h3>
    <table className="table-auto w-full border border-gray-400 text-sm">
      <thead>
        <tr>
          <th className="border px-4 py-2">Model</th>
          <th className="border px-4 py-2">Accuracy</th>
          <th className="border px-4 py-2">F1-Score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Random Forest</td>
          <td className="border px-4 py-2">0.82</td>
          <td className="border px-4 py-2">0.79</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">XGBoost</td>
          <td className="border px-4 py-2">0.85</td>
          <td className="border px-4 py-2">0.83</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stacked Ensemble</td>
          <td className="border px-4 py-2">0.88</td>
          <td className="border px-4 py-2">0.86</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-700 rounded-lg p-4">
      <p className="font-semibold">Highlight:</p>
      <p>
        ใน Kaggle Competition ส่วนใหญ่ ทีมที่ชนะล้วนใช้ Ensembling โดยเฉพาะการ Stacking หลายชั้นเพื่อเพิ่มความแม่นยำ
      </p>
    </div>

    <h3>ข้อควรระวังในการทำ Ensembling</h3>
    <ul>
      <li>Overfitting หาก stacking ซับซ้อนเกินไป</li>
      <li>เพิ่มเวลาในการเทรนและ deploy</li>
      <li>ยากต่อการตีความผลลัพธ์ในระบบที่ต้องอธิบายได้</li>
    </ul>

    <h3>สรุปและแหล่งอ้างอิง</h3>
    <ul>
      <li>Stanford CS229 Lecture Notes on Model Selection</li>
      <li>MIT OpenCourseWare: Advanced Machine Learning</li>
      <li>IEEE Transactions on Pattern Analysis and Machine Intelligence</li>
      <li>Zhou, Z. et al., "Ensemble Methods: Foundations and Algorithms," 2021</li>
    </ul>
  </div>
</section>


   <section id="regularization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Regularization & Avoiding Overfitting</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert">
    <h3>ความสำคัญของ Regularization ในโมเดล Supervised Learning</h3>
    <p>
      ปัญหา Overfitting เป็นความท้าทายหลักในกระบวนการฝึกโมเดล Supervised Learning โดยเฉพาะเมื่อโมเดลมีความซับซ้อนสูงและข้อมูลฝึกมีขนาดจำกัด Regularization เป็นเทคนิคสำคัญที่ช่วยลดความเสี่ยงนี้โดยเพิ่มข้อจำกัดให้กับพารามิเตอร์ของโมเดล ส่งผลให้โมเดลสามารถเรียนรู้ได้อย่างมีเสถียรภาพและมีความสามารถในการ generalize สูงขึ้น
    </p>

    <h3>ประเภทของ Regularization</h3>
    <ul>
      <li><strong>L1 Regularization (Lasso):</strong> เพิ่มค่าสัมบูรณ์ของน้ำหนักเข้าไปในฟังก์ชัน loss ทำให้บางพารามิเตอร์ลดเหลือศูนย์ ช่วยให้เกิดการเลือก feature อัตโนมัติ</li>
      <li><strong>L2 Regularization (Ridge):</strong> เพิ่มค่ายกกำลังสองของน้ำหนัก ช่วยให้โมเดลไม่เรียนรู้ค่าพารามิเตอร์ที่สูงเกินไป</li>
      <li><strong>Elastic Net:</strong> ผสมผสานระหว่าง L1 และ L2 เพื่อสร้างสมดุลระหว่างการเลือก feature และความเสถียรของโมเดล</li>
    </ul>

    <h3>เทคนิคเพิ่มเติมในการหลีกเลี่ยง Overfitting</h3>
    <ul>
      <li><strong>Early Stopping:</strong> หยุดการฝึกเมื่อ validation loss เริ่มเพิ่มขึ้น</li>
      <li><strong>Dropout:</strong> ใช้ใน neural networks เพื่อลดการพึ่งพาร่วมกันของ neurons โดยสุ่มปิดบาง node ชั่วคราวระหว่าง training</li>
      <li><strong>Data Augmentation:</strong> ขยายขนาดข้อมูลโดยแปลงข้อมูลเดิม ช่วยให้โมเดลได้เรียนรู้จากความหลากหลายที่มากขึ้น</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-md my-6 border-l-4 border-yellow-400">
      <h4 className="font-semibold mb-2">Insight Box</h4>
      <p>
        Regularization ไม่ได้เพียงช่วยลดความซับซ้อนของโมเดลเท่านั้น แต่ยังช่วยให้นักวิจัยเข้าใจลักษณะของข้อมูลและพฤติกรรมของโมเดลได้ดีขึ้น โดยเฉพาะเมื่อทำงานกับชุดข้อมูลขนาดใหญ่หรือไม่สมดุล
      </p>
    </div>

    <h3>ตารางเปรียบเทียบ Regularization Techniques</h3>
    <div className="overflow-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800">
          <tr>
            <th className="px-4 py-2 text-left">Technique</th>
            <th className="px-4 py-2 text-left">Key Feature</th>
            <th className="px-4 py-2 text-left">Use Case</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t">
            <td className="px-4 py-2">L1 (Lasso)</td>
            <td className="px-4 py-2">Sparse weights</td>
            <td className="px-4 py-2">Feature selection</td>
          </tr>
          <tr className="border-t">
            <td className="px-4 py-2">L2 (Ridge)</td>
            <td className="px-4 py-2">Penalize large weights</td>
            <td className="px-4 py-2">General model stabilization</td>
          </tr>
          <tr className="border-t">
            <td className="px-4 py-2">Elastic Net</td>
            <td className="px-4 py-2">Combines L1 and L2</td>
            <td className="px-4 py-2">When data has high dimensionality</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="mt-10">การอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Ng, A. (Stanford). "Regularization in ML Models." CS229 Lecture Notes.</li>
      <li>Zou, H., & Hastie, T. (2005). "Regularization and variable selection via the elastic net." Journal of the Royal Statistical Society.</li>
      <li>Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR.</li>
      <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
    </ul>
  </div>
</section>

   <section id="deployment" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Monitoring & Deployment Readiness</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความสำคัญของ Monitoring หลังการ Deploy</h3>
    <p>
      แม้ว่าโมเดลจะผ่านการฝึกฝนอย่างเข้มข้นและประเมินผลมาอย่างครบถ้วนในขั้นตอนก่อนหน้า แต่ในโลกจริงยังคงมีปัจจัยแวดล้อมจำนวนมากที่ทำให้โมเดลอาจไม่สามารถคงประสิทธิภาพไว้ได้ เช่น concept drift, data drift, หรือ user behavior shift ซึ่งจำเป็นต้องมีระบบ monitoring เพื่อคอยตรวจสอบอย่างต่อเนื่อง
    </p>

    <div className="bg-yellow-700 rounded-lg p-4 border-l-4 border-yellow-500">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยจาก Google Research ("Hidden Technical Debt in Machine Learning Systems") พบว่าต้นทุนในการดูแลระบบ ML หลัง deploy สูงกว่าการพัฒนา initial model หลายเท่าตัว
      </p>
    </div>

    <h3>องค์ประกอบหลักของระบบ Monitoring</h3>
    <ul className="list-disc pl-6">
      <li><strong>Prediction Drift Detection:</strong> ตรวจจับการเปลี่ยนแปลงของ distribution ของ prediction</li>
      <li><strong>Input Feature Drift:</strong> ตรวจสอบการเปลี่ยนแปลงของ distribution ของ input features</li>
      <li><strong>Performance Monitoring:</strong> ติดตามค่า metric เช่น accuracy, precision, recall จากข้อมูลจริง (ถ้ามี label)</li>
      <li><strong>Logging & Alert:</strong> ระบบต้องมี log ที่ละเอียด พร้อมตั้งค่าการแจ้งเตือนเมื่อเกิด anomaly</li>
    </ul>

    <h3>การเตรียมความพร้อมก่อน Deploy</h3>
    <ul className="list-disc pl-6">
      <li>บันทึก model version และ metadata ด้วยระบบ MLOps เช่น MLflow</li>
      <li>สร้างชุด test case ครอบคลุมทุก edge case และข้อมูลผิดปกติ</li>
      <li>ใช้ shadow deployment หรือ canary release เพื่อทดสอบใน production จริง</li>
    </ul>

    <div className="bg-blue-700 rounded-lg p-4 border-l-4 border-blue-500">
      <p className="font-semibold">Best Practice:</p>
      <p>
        สำหรับองค์กรขนาดใหญ่ ควรมีระบบ model rollback ที่สามารถ revert ไปยังเวอร์ชันก่อนหน้าได้ทันทีหากตรวจพบปัญหาใน production
      </p>
    </div>

    <h3>การเชื่อมโยงกับระบบ Data Pipeline</h3>
    <p>
      ระบบ deployment ที่ดีต้องสามารถเชื่อมต่อกับ pipeline ที่ทำหน้าที่ preprocessing, validation, และการส่งข้อมูลเข้าโมเดลได้อย่าง real-time หรือ batch ตามลักษณะการใช้งาน
    </p>

    <h3>แนวทางในการวัดความพร้อมของโมเดลก่อน Deploy</h3>
    <table className="table-auto border border-collapse w-full text-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border px-4 py-2">หัวข้อ</th>
          <th className="border px-4 py-2">คำถามที่ควรตรวจสอบ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Robustness</td>
          <td className="border px-4 py-2">โมเดลสามารถรับมือกับ noise หรือ input ที่ไม่สมบูรณ์ได้หรือไม่?</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Latency</td>
          <td className="border px-4 py-2">เวลาในการ response อยู่ในระดับที่ผู้ใช้ยอมรับได้หรือไม่?</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Scalability</td>
          <td className="border px-4 py-2">ระบบสามารถรองรับจำนวนผู้ใช้งานที่เพิ่มขึ้นได้หรือไม่?</td>
        </tr>
      </tbody>
    </table>

    <h3>บทสรุป</h3>
    <p>
      การวางแผนด้าน Monitoring และ Deployment ไม่ควรถูกมองว่าเป็นขั้นตอนท้ายสุดที่ไม่สำคัญ แต่กลับเป็นหัวใจหลักที่จะทำให้โมเดลทำงานได้อย่างมีประสิทธิภาพต่อเนื่อง และสามารถตรวจสอบปัญหาได้อย่างรวดเร็วเมื่อเกิดเหตุการณ์ผิดปกติในระบบจริง
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>"Hidden Technical Debt in Machine Learning Systems", D. Sculley et al., Google Research, NIPS 2015</li>
      <li>"Reliable Machine Learning", Chip Huyen, Stanford CS329S (2023)</li>
      <li>"MLOps: Continuous Delivery and Automation Pipelines in Machine Learning", Mark Treveil et al., O'Reilly</li>
    </ul>
  </div>
</section>


       <section id="feedback" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    11. Feedback Loop & Continuous Learning
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert">
    <h3>บทนำสู่ Feedback Loop ในระบบเรียนรู้</h3>
    <p>
      การพัฒนาโมเดล Machine Learning ที่มีประสิทธิภาพสูงในระยะยาวจำเป็นต้องมีระบบ Feedback Loop ที่ออกแบบอย่างรัดกุม โดยเฉพาะในระบบที่เปลี่ยนแปลงตามเวลา เช่นระบบ recommendation, fraud detection หรือ NLP-based assistant ซึ่งข้อมูลใหม่อาจสะท้อนพฤติกรรมที่เปลี่ยนไปอย่างรวดเร็ว
    </p>

    <div className="bg-yellow-700 rounded-md p-4 my-6">
      <p className="text-sm font-medium">
        🔍 Insight: ระบบ feedback loop ที่ไม่มีการควบคุมอาจทำให้โมเดลตอกย้ำ bias เดิม ซึ่งนำไปสู่การเสื่อมประสิทธิภาพในระบบเรียนรู้ระยะยาว (MIT 2022)
      </p>
    </div>

    <h3>ประเภทของ Feedback</h3>
    <ul>
      <li><strong>Explicit Feedback</strong>: ข้อมูลจากผู้ใช้โดยตรง เช่น การให้คะแนน</li>
      <li><strong>Implicit Feedback</strong>: ข้อมูลพฤติกรรม เช่น การคลิก, เวลาที่ใช้กับเนื้อหา</li>
      <li><strong>Corrective Feedback</strong>: การ label ใหม่เมื่อโมเดลพลาด เช่น การ report spam</li>
    </ul>

    <h3>แนวทาง Continuous Learning ที่ได้รับการยอมรับ</h3>
    <p>สถาบันอย่าง CMU และ Oxford เสนอแนวทางที่สำคัญในการออกแบบ Continuous Learning ดังนี้:</p>
    <ul>
      <li><strong>Incremental Learning:</strong> ปรับโมเดลเมื่อมีข้อมูลใหม่โดยไม่ต้อง retrain ทั้งหมด</li>
      <li><strong>Online Learning:</strong> อัปเดตโมเดลต่อเนื่องทุก instance เช่น streaming data</li>
      <li><strong>Active Learning:</strong> เลือกตัวอย่างที่โมเดลไม่มั่นใจเพื่อให้มนุษย์ label เพิ่ม</li>
    </ul>

    <h3>ข้อควรระวังในการใช้ Feedback Loop</h3>
    <div className="bg-blue-700 rounded-md p-4 my-6">
      <p>
        ⚠️ การเรียนรู้จาก feedback ที่มาจากโมเดลเดิม (self-labeling) อาจทำให้เกิด feedback bias ได้
        การใช้งาน techniques อย่าง self-training ควรมีการ validate ด้วยชุดข้อมูลภายนอกเสมอ
      </p>
    </div>

    <h3>การประเมินระบบ Continuous Learning</h3>
    <p>โมเดลที่เรียนรู้อย่างต่อเนื่องต้องมีการประเมินต่อเนื่องเช่นกัน:</p>
    <ul>
      <li>ใช้ sliding window accuracy เพื่อตรวจสอบ performance ระยะสั้น</li>
      <li>เปรียบเทียบ performance กับ baseline static model</li>
      <li>ตรวจสอบ concept drift: ตรวจจับการเปลี่ยนแปลง distribution</li>
    </ul>

    <h3>Best Practice: สร้าง Feedback Infrastructure</h3>
    <table className="table-auto border border-gray-300 text-sm my-6">
      <thead>
        <tr>
          <th className="border px-4 py-2">องค์ประกอบ</th>
          <th className="border px-4 py-2">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Logging</td>
          <td className="border px-4 py-2">บันทึกการตัดสินใจของโมเดลเพื่อวิเคราะห์ย้อนหลัง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Feedback API</td>
          <td className="border px-4 py-2">ช่องทางรับ feedback จากผู้ใช้หรือแหล่งข้อมูลอื่น</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Version Control</td>
          <td className="border px-4 py-2">บันทึก model version และ data schema ที่ใช้งาน</td>
        </tr>
      </tbody>
    </table>

    <h3>แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS229 Lecture Notes on Online and Incremental Learning (2022)</li>
      <li>IEEE Transactions on Neural Networks and Learning Systems, Vol. 32, No. 4</li>
      <li>arXiv:2203.07568 - A Survey on Continual Learning and Feedback Loops</li>
      <li>MIT CSAIL: Risks of Self-Reinforcing Feedback Loops in AI (2023)</li>
    </ul>
  </div>
</section>


<section id="checklist" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Best Practices Checklist</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>วัตถุประสงค์ของ Checklist</h3>
    <p>
      การมีรายการตรวจสอบ (Checklist) ช่วยลดความผิดพลาดที่อาจเกิดขึ้นจากกระบวนการเรียนรู้ที่ซับซ้อนใน Supervised Learning โดยเฉพาะในงานจริงซึ่งต้องประสานหลายฝ่าย ตั้งแต่ data engineer, researcher ไปจนถึง deployment engineer
    </p>

    <h3>รายการตรวจสอบก่อนเริ่มพัฒนา</h3>
    <ul className="list-disc pl-6">
      <li>นิยามปัญหาให้ชัดเจนเป็น Classification, Regression หรืออื่น ๆ</li>
      <li>มี baseline model สำหรับเปรียบเทียบหรือไม่</li>
      <li>ข้อมูลครบถ้วนหรือยัง — ต้องมี data profiling เบื้องต้น</li>
      <li>ระบุ evaluation metric ที่เหมาะสมไว้ล่วงหน้า</li>
    </ul>

    <h3>ระหว่างการพัฒนาโมเดล</h3>
    <ul className="list-decimal pl-6">
      <li>ใช้เทคนิค feature selection และ feature engineering อย่างมีระบบ</li>
      <li>มีการจัดการกับ missing values และ outliers อย่างเหมาะสม</li>
      <li>ทำ cross-validation แบบ stratified หรือ k-fold เพื่อลด bias</li>
      <li>ใช้ regularization (เช่น L1, L2) หากจำเป็น</li>
    </ul>

    <h3>หลังการเทรนและก่อน Deploy</h3>
    <ul className="list-disc pl-6">
      <li>ตรวจสอบ overfitting และ underfitting ผ่าน learning curve</li>
      <li>ประเมินโมเดลด้วย hold-out set ที่ไม่เคยใช้มาก่อน</li>
      <li>วิเคราะห์ error ด้วย confusion matrix และ metric เฉพาะ task</li>
      <li>ทดสอบ performance consistency บนข้อมูลจากหลาย distribution</li>
    </ul>

    <div className="bg-blue-700 dark:bg-blue-900 text-blue-900 dark:text-blue-700 rounded-lg p-4">
      <strong>Highlight:</strong> งานวิจัยจาก MIT และ CMU แสดงให้เห็นว่า checklist ที่มีโครงสร้างแบบลำดับเวลา (temporal pipeline checklist) ช่วยลดเวลา debug ได้มากกว่า 25% และเพิ่ม reproducibility ใน production deployment
    </div>

    <h3>Deployment และการวัดผลภายหลัง</h3>
    <ul className="list-disc pl-6">
      <li>บันทึก parameter และเวอร์ชันของโค้ดที่ใช้สำหรับเทรนไว้เสมอ</li>
      <li>มีระบบ monitor latency และ accuracy ของโมเดลแบบ real-time</li>
      <li>มี feedback loop จากผู้ใช้งานจริงเข้ามาช่วยปรับปรุงโมเดล</li>
    </ul>

    <h3>การใช้ Checklist แบบอัตโนมัติ</h3>
    <p>
      ในระบบ production ขนาดใหญ่ การใช้ ML pipeline frameworks เช่น MLflow, Kubeflow หรือ TFX ช่วยทำให้ checklist เหล่านี้กลายเป็นระบบอัตโนมัติที่สามารถติดตามและควบคุมได้ง่าย
    </p>

    <div className="bg-yellow-700 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-700 rounded-lg p-4">
      <strong>Insight:</strong> Checklist ไม่ได้ใช้เพื่อลดอิสระของนักวิจัย แต่ช่วยให้ทุกฝ่ายในทีมทำงานร่วมกันได้อย่างมีระบบ โดยเฉพาะเมื่อต้อง maintain โมเดลในระยะยาว
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>"Hidden Technical Debt in Machine Learning Systems" — Google Research</li>
      <li>"ML Test Score: A Rubric for Production Readiness" — Google Brain</li>
      <li>"The ML Checklist Manifesto" — Stanford CS229 Lecture Notes</li>
      <li>"Efficient ML Workflow Management" — MIT CSAIL, 2020</li>
    </ul>
  </div>
</section>

<section id="case" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">13. Case Studies</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>กรณีศึกษาจากอุตสาหกรรมเทคโนโลยี: Google และ AutoML</h3>
    <p>
      Google ได้พัฒนาแนวทางการใช้ Automated Machine Learning (AutoML) ซึ่งรวมถึงขั้นตอนของการเลือกโมเดล, การปรับแต่ง hyperparameters, และการวัดผลแบบอัตโนมัติ เพื่อให้ได้ประสิทธิภาพสูงสุดในระบบ Production. หนึ่งในตัวอย่างที่สำคัญคือ AutoML Vision ซึ่งใช้ Neural Architecture Search (NAS) สำหรับค้นหาโมเดลที่เหมาะสมสำหรับงาน image classification บนชุดข้อมูลขนาดใหญ่.
    </p>

    <div className="bg-yellow-700 rounded-lg p-4">
      <strong>Highlight:</strong> แนวทางของ Google AutoML ใช้ Bayesian Optimization ร่วมกับการค้นหาเชิงวิวัฒนาการ (evolutionary search) ทำให้สามารถปรับแต่งสถาปัตยกรรมได้แม่นยำโดยไม่ต้องพึ่งวิศวกรโมเดลโดยตรง
    </div>

    <h3>กรณีศึกษาในภาคการแพทย์: Stanford’s Deep Learning in Radiology</h3>
    <p>
      งานวิจัยจาก Stanford แสดงให้เห็นว่าการประยุกต์ใช้ Convolutional Neural Networks (CNNs) เพื่อวิเคราะห์ภาพเอกซเรย์ทรวงอกสามารถตรวจพบโรคหลายชนิดได้อย่างแม่นยำ เทียบเคียงได้กับรังสีแพทย์ผู้เชี่ยวชาญ โดยโครงการ CheXNet ซึ่งตีพิมพ์ในปี 2017 ได้รับการยอมรับอย่างกว้างขวาง
    </p>

    <div className="bg-blue-700 rounded-lg p-4">
      <strong>Insight Box:</strong> ทีมวิจัยได้เน้นย้ำถึงความสำคัญของการใช้ dataset ที่มี annotation คุณภาพสูง และการควบคุม class imbalance ผ่านการ sampling แบบชาญฉลาด
    </div>

    <h3>การประยุกต์ใช้ในธุรกิจ: Netflix และการแนะนำคอนเทนต์</h3>
    <p>
      Netflix ใช้ supervised learning เพื่อวิเคราะห์พฤติกรรมของผู้ใช้งาน โดยนำข้อมูลเช่น ประวัติการดู, การให้คะแนน, และเวลาที่ดู มาใช้ในการ train โมเดล recommendation system ที่สามารถปรับเปลี่ยนแบบ dynamic ตามบริบทของผู้ใช้
    </p>

    <ul className="list-disc pl-5">
      <li>ใช้ Matrix Factorization ร่วมกับ neural networks สำหรับ contextual embedding</li>
      <li>มีการทดลอง A/B Testing อย่างต่อเนื่องเพื่อ validate โมเดล</li>
      <li>ใช้ Feedback loop แบบ near real-time เพื่อปรับปรุง recommendation</li>
    </ul>

    <h3>แนวโน้มในระดับองค์กรและงานวิจัย</h3>
    <p>
      องค์กรอย่าง Microsoft, Amazon และ Meta ได้พัฒนา best practices สำหรับการ deploy ML models ในระดับองค์กร เช่น MLOps, monitoring pipeline, และ reproducibility ของระบบ
    </p>

    <div className="bg-yellow-700 rounded-lg p-4">
      <strong>Highlight:</strong> งานของ Microsoft ในเรื่อง Responsible AI แสดงให้เห็นว่า Best Practices ไม่ได้จำกัดแค่ความแม่นยำของโมเดล แต่รวมถึงความโปร่งใส (transparency) และความเป็นธรรม (fairness) ของระบบด้วย
    </div>

    <h3>การอ้างอิง</h3>
    <ul className="list-disc pl-5">
      <li>Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. arXiv preprint arXiv:1603.02754.</li>
      <li>Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225</li>
      <li>Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946</li>
      <li>Netflix Research Blog: https://netflixtechblog.com</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day39 theme={theme} />
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
        <ScrollSpy_Ai_Day39 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day39_BestPractices;
