// โครงสร้างเริ่มต้น Day13: Model Interpretability & Explainability
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day13 from "./scrollspy/ScrollSpy_Ai_Day13";
import MiniQuiz_Day13 from "./miniquiz/MiniQuiz_Day13";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../components/common/sidebar/AiSidebar";

const Day13_ModelExplainability = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: 'dxtnq9fxw' } });

  const img1 = cld.image('Interpretabilit1').format('auto').quality('auto').resize(scale().width(500));
  const img2 = cld.image('Interpretabilit2').format('auto').quality('auto').resize(scale().width(500));
  const img3 = cld.image('Interpretabilit3').format('auto').quality('auto').resize(scale().width(500));
  const img4 = cld.image('Interpretabilit4').format('auto').quality('auto').resize(scale().width(500));
  const img5 = cld.image('Interpretabilit5').format('auto').quality('auto').resize(scale().width(500));
  const img6 = cld.image('Interpretabilit6').format('auto').quality('auto').resize(scale().width(500));
  const img7 = cld.image('Interpretabilit7').format('auto').quality('auto').resize(scale().width(500));
  const img8 = cld.image('Interpretabilit8').format('auto').quality('auto').resize(scale().width(500));
  const img9 = cld.image('Interpretabilit9').format('auto').quality('auto').resize(scale().width(500));
  const img10 = cld.image('Interpretabilit10').format('auto').quality('auto').resize(scale().width(500));
  const img11 = cld.image('Interpretabilit11').format('auto').quality('auto').resize(scale().width(500));
  const img12 = cld.image('Interpretabilit12').format('auto').quality('auto').resize(scale().width(500));
  const img13 = cld.image('Interpretabilit13').format('auto').quality('auto').resize(scale().width(500));

  

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 13: Model Interpretability & Explainability</h1>

        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center"> ทำไม Interpretability ถึงสำคัญ?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในโลกของ AI และ Machine Learning ที่มีการใช้งานแพร่หลายในหลากหลายอุตสาหกรรม เช่น การแพทย์ การเงิน และกฎหมาย ความสามารถในการอธิบายเหตุผลเบื้องหลังการตัดสินใจของโมเดลจึงมีความสำคัญอย่างยิ่ง ไม่ใช่แค่เพียงเรื่องของความโปร่งใส แต่รวมถึงเรื่องของความน่าเชื่อถือ ความปลอดภัย และจริยธรรม
  </p>

  <p className="mb-4 leading-relaxed">
    เมื่อโมเดลให้ผลลัพธ์ เราไม่ควรถามแค่ว่า “ผลลัพธ์นี้ถูกหรือไม่” แต่ควรถามว่า “เพราะอะไรจึงได้ผลลัพธ์นี้” — นั่นคือหัวใจของ Interpretability
  </p>

  <div className="grid md:grid-cols-2 gap-6 my-6">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-3"> ประโยชน์ของ Interpretability</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>สร้างความไว้วางใจให้กับผู้ใช้และ Stakeholder</li>
        <li>ช่วยตรวจจับ Bias ที่แอบแฝงในข้อมูล</li>
        <li>วินิจฉัยพฤติกรรมโมเดลที่ผิดปกติ เช่น Overfitting กับ Feature ไม่สำคัญ</li>
        <li>อธิบายเหตุผลที่อยู่เบื้องหลังการตัดสินใจของโมเดลได้</li>
        <li>จำเป็นต่อการทำ Compliance หรือการตรวจสอบทางกฎหมาย (เช่น GDPR)</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-3"> เมื่อไม่มี Interpretability</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ไม่สามารถรู้ว่าโมเดลตัดสินใจจากข้อมูลใด</li>
        <li>เสี่ยงต่อการเลือกใช้งานโมเดลผิดวัตถุประสงค์</li>
        <li>ไม่สามารถตรวจสอบได้ว่าโมเดลทำงานตามที่คาดไว้หรือไม่</li>
        <li>ผู้ใช้อาจไม่มั่นใจแม้ Accuracy จะสูง</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mb-3 mt-10"> ตัวอย่างจากสถานการณ์จริง</h3>
  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">
      ในงานด้านการแพทย์ AI อาจช่วยในการตรวจหามะเร็งจากภาพ MRI แต่หากโมเดลตัดสินใจจาก Artifact หรือจุดแสงที่ไม่เกี่ยวข้อง — แม้โมเดลจะมี Accuracy สูง ก็ยังเป็นอันตรายได้
    </p>
    <p className="mb-2">
      ในด้านการเงิน โมเดลให้กู้เฉพาะเพศชายโดยอิงจากข้อมูลที่มี Bias — สิ่งนี้อาจนำไปสู่การเลือกปฏิบัติอย่างไม่ยุติธรรมโดยไม่รู้ตัว
    </p>
    <p>
      สิ่งเหล่านี้ชี้ให้เห็นว่า เราจำเป็นต้องรู้ว่า “เหตุผลเบื้องหลัง” การตัดสินใจคืออะไร ไม่ใช่แค่ “คำตอบคืออะไร”
    </p>
  </div>

  <h3 className="text-xl font-semibold mb-3 mt-10"> Global vs Local Interpretability</h3>
  <div className="grid md:grid-cols-2 gap-6 mt-6">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-medium mb-2"> Global Interpretability</h4>
      <p className="text-sm">
        วิเคราะห์พฤติกรรมของโมเดลโดยรวม เช่น Feature ไหนสำคัญที่สุดสำหรับการตัดสินใจโดยรวมของโมเดล
      </p>
    </div>
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="font-medium mb-2"> Local Interpretability</h4>
      <p className="text-sm">
        อธิบายการตัดสินใจเฉพาะตัวอย่าง เช่น เหตุใดโมเดลจึงตัดสินใจให้ผู้ใช้นี้กู้เงิน (เฉพาะราย) หรือไม่
      </p>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-10 mb-3 text-center"> Insight สรุป</h3>
  <div className="bg-green-100 dark:bg-green-900 text-black dark:text-green-100 p-5 rounded-xl border-l-4 border-green-500 shadow">
    <p className="mb-2">Interpretability ไม่ได้เป็นแค่เครื่องมือทางเทคนิค แต่เป็นหลักฐานของความรับผิดชอบ (Accountability)</p>
    <p>โมเดลที่ดีไม่ใช่แค่แม่น แต่ต้องอธิบายได้ — และเข้าใจได้โดยมนุษย์</p>
  </div>
</section>

<section id="blackbox" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">ปัญหาที่เกิดจากโมเดล Black-box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในยุคที่โมเดล Machine Learning และ Deep Learning มีความซับซ้อนมากขึ้น การที่โมเดลสามารถให้ผลลัพธ์ที่แม่นยำเพียงอย่างเดียวไม่เพียงพอ โดยเฉพาะในงานที่เกี่ยวข้องกับการตัดสินใจของมนุษย์ เช่น การแพทย์ การเงิน หรือกระบวนการยุติธรรม ความเข้าใจว่าทำไมโมเดลจึงตัดสินใจเช่นนั้นมีความสำคัญอย่างยิ่ง
  </p>

  <p className="mb-4 leading-relaxed">
    โมเดลประเภท "Black-box" คือโมเดลที่แม้จะสามารถทำนายผลได้อย่างแม่นยำ แต่ภายในกระบวนการทำงานกลับยากต่อการตีความ ยกตัวอย่างเช่น Neural Network ที่มีหลายชั้น (Deep Neural Networks) หรือ Ensemble Models อย่าง XGBoost หรือ Random Forest ซึ่งรวมหลายโมเดลเข้าด้วยกัน จึงทำให้การตรวจสอบการทำงานภายในเป็นเรื่องท้าทาย
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">1. ขาดความโปร่งใสในการตัดสินใจ</h3>
  <p className="mb-4 leading-relaxed">
    หากระบบ ML ตัดสินใจโดยไม่มีคำอธิบายชัดเจน เช่น ปฏิเสธคำขอกู้เงินของผู้สมัครรายหนึ่ง แต่ไม่สามารถบอกเหตุผลได้ ผู้ใช้จะรู้สึกไม่ยุติธรรม และอาจตั้งคำถามถึงความน่าเชื่อถือของระบบ
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">2. ตรวจจับ Bias ได้ยาก</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ฝึกจากข้อมูลที่มีอคติ อาจสะท้อนอคตินั้นกลับออกมา เช่น หากชุดข้อมูลมีการเลือกปฏิบัติต่อเพศหรือเชื้อชาติ โมเดลก็อาจเรียนรู้อคตินั้นโดยไม่รู้ตัว การขาด interpretability ทำให้ยากต่อการตรวจสอบ bias เหล่านี้อย่างเป็นระบบ
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">3. ปัญหาด้านกฎหมายและความรับผิดชอบ</h3>
  <p className="mb-4 leading-relaxed">
    ในหลายประเทศ การใช้ AI ในระบบที่มีผลกระทบสูงต้องสามารถอธิบายได้ เช่น กฎหมาย GDPR ของยุโรประบุว่า ผู้ใช้งานมีสิทธิได้รับคำอธิบายของการตัดสินใจที่มีผลต่อพวกเขา ซึ่งขัดแย้งกับการใช้โมเดลที่ไม่สามารถตีความได้
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">4. ยากต่อการ Debug และปรับปรุง</h3>
  <p className="mb-4 leading-relaxed">
    หากโมเดลให้ผลลัพธ์ที่ไม่ตรงตามคาด เช่น แนะนำผลิตภัณฑ์ผิดกลุ่มเป้าหมาย นักพัฒนาไม่สามารถระบุได้ว่าเกิดจากฟีเจอร์ใดหรือกระบวนการใดของโมเดล ส่งผลให้การพัฒนาเชิงซ้ำกลายเป็นการลองผิดลองถูก
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">5. ความไม่มั่นใจจากผู้มีส่วนได้เสีย</h3>
  <p className="mb-4 leading-relaxed">
    ผู้บริหาร นักวิเคราะห์ และลูกค้า มักต้องการเข้าใจเหตุผลเบื้องหลังการคาดการณ์ของโมเดล การนำเสนอผลลัพธ์ที่ตีความไม่ได้จะลดความมั่นใจและขัดขวางการนำ AI ไปใช้งานจริงในองค์กร
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">6. ตัวอย่างปัญหาที่เกิดขึ้นจริง</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ระบบคัดกรองใบสมัครงานที่ให้คะแนนสูงกับผู้ชายมากกว่าผู้หญิง เนื่องจากโมเดลถูกฝึกจากข้อมูลประวัติการรับสมัครเดิม</li>
    <li>ระบบวินิจฉัยโรคจากภาพถ่ายทางการแพทย์ที่ตัดสินผลจาก background noise ไม่ใช่ตัวโรค</li>
    <li>ระบบให้คะแนนความเสี่ยงของผู้กู้ที่แยกแยะไม่ได้ว่าคะแนนมาจากประวัติการเงิน หรือมาจากรหัสไปรษณีย์ (ซึ่งอาจมี bias เชิงพื้นที่)</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-3">7. ความเสี่ยงต่อชื่อเสียงองค์กร</h3>
  <p className="mb-4 leading-relaxed">
    การตัดสินใจของ AI ที่ไม่สามารถอธิบายได้ อาจนำไปสู่การวิพากษ์วิจารณ์จากสื่อหรือสังคม โดยเฉพาะหากผลการตัดสินใจนั้นสร้างผลกระทบเชิงลบต่อผู้ใช้ การขาด transparency จึงอาจกลายเป็นจุดอ่อนร้ายแรงในระยะยาว
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      แม้ว่าโมเดล Black-box จะให้ผลลัพธ์ที่ดีในแง่ของความแม่นยำ แต่ความสามารถในการอธิบายเบื้องหลังผลลัพธ์นั้นเป็นสิ่งที่จำเป็นอย่างยิ่งสำหรับการนำ AI ไปใช้ในโลกจริง โดยเฉพาะในบริบทที่มีผลกระทบต่อชีวิตผู้คนหรือความน่าเชื่อถือขององค์กร
    </p>
  </div>
</section>

<section id="global" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">เทคนิค Global Interpretability</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <p className="mb-4 leading-relaxed">
    Global Interpretability คือการอธิบายพฤติกรรมโดยรวมของโมเดล เช่น โมเดลนี้ให้ความสำคัญกับ feature อะไรมากที่สุด หรือโมเดลตัดสินใจอย่างไรกับข้อมูลทั้งหมด ไม่ใช่แค่ตัวอย่างใดตัวอย่างหนึ่ง จุดแข็งของวิธีนี้คือช่วยให้เราวิเคราะห์โครงสร้างการเรียนรู้ของโมเดลในระดับภาพรวมได้ดี เหมาะกับการวิเคราะห์การเรียนรู้ผิดพลาด หรือการปรับปรุง preprocessing
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">1. Feature Importance (เหมาะกับ Tree-Based Models)</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลประเภทเช่น Decision Tree, Random Forest และ XGBoost สามารถบอกความสำคัญของ feature ได้ตรง ๆ ผ่านค่าความถี่หรือผลกระทบที่ feature นั้นมีต่อการ split node ซึ่งช่วยให้เราเข้าใจว่าโมเดลเรียนรู้อะไร
  </p>

  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 text-sm font-mono overflow-auto mb-4">
    from sklearn.ensemble import RandomForestClassifier<br />
    model = RandomForestClassifier()<br />
    model.fit(X, y)<br />
    print(model.feature_importances_)
  </div>

  <p className="mb-4 leading-relaxed">
    ค่าที่ได้จะเป็น weight ของแต่ละ feature ซึ่งสามารถนำไป plot bar chart เพื่อดูว่าสิ่งที่โมเดลโฟกัสคืออะไร เช่น เพศ อายุ รายได้ ฯลฯ
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">2. Permutation Importance</h3>
  <p className="mb-4 leading-relaxed">
    เทคนิคนี้จะสุ่มสลับค่าในแต่ละฟีเจอร์ แล้วดูว่า performance ของโมเดลลดลงแค่ไหน หาก performance ลดลงมาก แสดงว่า feature นั้นสำคัญ
  </p>

  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 text-sm font-mono overflow-auto mb-4">
    from sklearn.inspection import permutation_importance<br />
    result = permutation_importance(model, X, y, n_repeats=10)<br />
    print(result.importances_mean)
  </div>

  <p className="mb-4 leading-relaxed">
    จุดเด่นของวิธีนี้คือใช้ได้กับทุกโมเดล ไม่จำเป็นต้องเป็น tree-based และยังช่วยให้เข้าใจว่าฟีเจอร์มีผลต่อ model performance อย่างไรโดยไม่ต้องอิงแค่ค่า coefficient
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">3. Coefficients จาก Linear Models</h3>
  <p className="mb-4 leading-relaxed">
    สำหรับโมเดลเชิงเส้น เช่น Linear Regression หรือ Logistic Regression เราสามารถอธิบายผลกระทบของแต่ละฟีเจอร์ได้จากค่า coefficient โดยตรง ซึ่งแสดงให้เห็นว่าฟีเจอร์แต่ละตัวส่งผลเป็นบวกหรือลบต่อ output
  </p>

  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 text-sm font-mono overflow-auto mb-4">
    from sklearn.linear_model import LogisticRegression<br />
    model = LogisticRegression()<br />
    model.fit(X, y)<br />
    print(model.coef_)
  </div>

  <p className="mb-4 leading-relaxed">
    ข้อควรระวัง: ค่า coefficient อาจแปลผลผิดถ้าไม่ได้ทำ feature scaling ล่วงหน้า หรือถ้า feature มี multicollinearity
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">4. Partial Dependence Plot (PDP)</h3>
  <p className="mb-4 leading-relaxed">
    PDP แสดงผลกระทบของ feature หนึ่ง (หรือสอง) ต่อการพยากรณ์ของโมเดล โดยการเปลี่ยนค่าฟีเจอร์นั้น ๆ ทีละค่าแล้วดูว่าผลลัพธ์ของโมเดลเปลี่ยนอย่างไร
  </p>

  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 text-sm font-mono overflow-auto mb-4">
    from sklearn.inspection import plot_partial_dependence<br />
    plot_partial_dependence(model, X, [0, 1])  # ฟีเจอร์ 0 และ 1
  </div>

  <p className="mb-4 leading-relaxed">
    เหมาะสำหรับ visualizing ความสัมพันธ์ของฟีเจอร์กับ prediction และเข้าใจการตัดสินใจของโมเดลในมุมที่ลึกขึ้น
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">5. Tree Visualization</h3>
  <p className="mb-4 leading-relaxed">
    หากใช้ Decision Tree สามารถ visualize การ split ของ tree ได้เลย ทำให้เข้าใจลำดับการตัดสินใจ เช่นอายุ &gt; 35 → รายได้ &gt; 50K → ทำนายว่า “ผ่าน”
  </p>

  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 text-sm font-mono overflow-auto mb-4">
    from sklearn.tree import plot_tree<br />
    import matplotlib.pyplot as plt<br />
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)<br />
    plt.show()
  </div>

  <p className="mb-4 leading-relaxed">
    การแสดงผล tree visualization ช่วยให้ทั้งทีมเทคนิคและ non-technical เข้าใจ logic ที่โมเดลใช้ในการตัดสินใจ และสามารถตรวจสอบความสมเหตุสมผลได้
  </p>

  <h3 className="text-xl font-semibold mt-6 mb-3">Insight</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">
      Global Interpretability เหมาะสำหรับการสื่อสารกับทีมวิจัย วิศวกร หรือเพื่อทำ Data Debugging แต่ไม่เพียงพอในการเข้าใจการตัดสินใจ “เฉพาะเคส” ซึ่งต้องใช้เทคนิค Local Interpretability ร่วมด้วย
    </p>
  </div>
</section>


<section id="local" className="mb-16 scroll-mt-32 min-h-[400px]">
        <h2 className="text-2xl font-semibold mb-4 text-center">เทคนิค Local Interpretability</h2>
        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img5} />
        </div>

        <p className="mb-4 leading-relaxed">
          Local Interpretability คือการอธิบายการตัดสินใจของโมเดลในระดับรายตัวอย่าง โดยมุ่งเน้นไปที่การตอบคำถามว่า "ทำไมโมเดลถึงตัดสินใจแบบนี้กับอินพุตเฉพาะนี้" ซึ่งต่างจาก Global Interpretability ที่วิเคราะห์พฤติกรรมโดยรวมของโมเดลทั้งหมด
        </p>

        <h3 className="text-xl font-semibold mb-3 text-center">LIME (Local Interpretable Model-Agnostic Explanations)</h3>
        <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img6} />
        </div>
        <p className="mb-4 leading-relaxed">
          LIME เป็นเทคนิคที่ใช้สร้างโมเดลง่าย ๆ (local surrogate model) เพื่อเลียนแบบพฤติกรรมของโมเดลที่ซับซ้อนเฉพาะบริเวณรอบ ๆ อินพุตที่ต้องการอธิบาย จากนั้นจะวิเคราะห์ว่า feature ใดส่งผลต่อผลลัพธ์มากที่สุด
        </p>

        <p className="mb-4 leading-relaxed">
          ตัวอย่างเช่น ถ้าโมเดลพยากรณ์ว่า "ลูกค้ารายนี้จะไม่ชำระหนี้" LIME จะช่วยชี้ว่าฟีเจอร์ใด (เช่น รายได้, อายุ, จำนวนหนี้สิน) มีผลต่อการตัดสินใจนี้ และแสดงค่า weight ของแต่ละฟีเจอร์
        </p>

        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 mb-6 overflow-auto text-sm">
          <pre>
{`
from lime import lime_tabular
explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=features, class_names=classes, discretize_continuous=True)

exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=5)
exp.show_in_notebook()
`}
          </pre>
        </div>

        <p className="mb-4 leading-relaxed">
          LIME รองรับข้อมูลหลากหลายรูปแบบ เช่น Tabular, Text และ Image โดยในกรณีของ Image จะใช้การ perturb pixel บางจุดเพื่อดูผลกระทบต่อผลลัพธ์ของโมเดล
        </p>

        <h3 className="text-xl font-semibold mb-3 text-center">SHAP (SHapley Additive exPlanations)</h3>
        <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img7} />
        </div>
        <p className="mb-4 leading-relaxed">
          SHAP ใช้แนวคิดจากทฤษฎีเกม โดยคำนวณว่าแต่ละ feature มีส่วนช่วยในการเปลี่ยนผลลัพธ์ prediction อย่างไรในเชิงปริมาณ ค่าที่ได้จะเป็นค่า SHAP value ที่บอกว่าแต่ละ feature "เพิ่ม" หรือ "ลด" ค่าการพยากรณ์เท่าใด
        </p>

        <p className="mb-4 leading-relaxed">
          ความพิเศษของ SHAP คือความยุติธรรมและเป็นกลางทางคณิตศาสตร์ ทำให้สามารถเปรียบเทียบได้แม่นยำว่า feature ใดมีผลมากที่สุดในตัวอย่างนั้น ๆ
        </p>

        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700 mb-6 overflow-auto text-sm">
          <pre>
{`
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[0])
shap.plots.bar(shap_values)
`}
          </pre>
        </div>

        <h3 className="text-xl font-semibold mb-3">การเปรียบเทียบ LIME กับ SHAP</h3>
        <div className="overflow-auto">
          <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-600">
            <thead className="bg-gray-200 dark:bg-gray-700 text-left">
              <tr>
                <th className="px-4 py-2 border-b">คุณสมบัติ</th>
                <th className="px-4 py-2 border-b">LIME</th>
                <th className="px-4 py-2 border-b">SHAP</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t">
                <td className="px-4 py-2">รองรับข้อมูลหลายประเภท</td>
                <td className="px-4 py-2">✔</td>
                <td className="px-4 py-2">✔</td>
              </tr>
              <tr>
                <td className="px-4 py-2">ค่าที่ได้มีความยุติธรรมทางคณิตศาสตร์</td>
                <td className="px-4 py-2">✘</td>
                <td className="px-4 py-2">✔</td>
              </tr>
              <tr>
                <td className="px-4 py-2">ความเร็วในการคำนวณ</td>
                <td className="px-4 py-2">เร็วกว่า</td>
                <td className="px-4 py-2">ช้ากว่า</td>
              </tr>
              <tr>
                <td className="px-4 py-2">เข้าใจง่ายสำหรับผู้ใช้งานทั่วไป</td>
                <td className="px-4 py-2">✔</td>
                <td className="px-4 py-2">✔</td>
              </tr>
              <tr>
                <td className="px-4 py-2">สามารถใช้ได้กับโมเดล Black-box</td>
                <td className="px-4 py-2">✔</td>
                <td className="px-4 py-2">✔</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
          <p className="font-semibold mb-2">Insight:</p>
          <p>
            เทคนิค Local Interpretability ไม่เพียงช่วยให้เข้าใจการตัดสินใจเฉพาะจุด แต่ยังสามารถใช้ในงานตรวจสอบความเป็นธรรม การหาจุดบกพร่องของโมเดล และสร้างความโปร่งใสในระบบ AI ได้อย่างแท้จริง
          </p>
        </div>
      </section>

      <section id="image-text" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">การอธิบายสำหรับ Image และ Text</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <p className="mb-4 leading-relaxed">
    โมเดล Machine Learning โดยเฉพาะ Deep Learning อย่าง CNN และ Transformer มักถูกมองว่าเป็น "กล่องดำ" โดยเฉพาะเมื่อใช้งานกับข้อมูลประเภทภาพและข้อความ ดังนั้นการเข้าใจว่าทำไมโมเดลถึงตัดสินใจแบบใดจึงเป็นเรื่องสำคัญ โดยเฉพาะในแอปพลิเคชันที่เกี่ยวกับความปลอดภัย สุขภาพ หรือการเงิน
  </p>

  <h3 className="text-xl font-semibold mb-3 text-center">Grad-CAM สำหรับข้อมูลภาพ (CNN)</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <p className="mb-4 leading-relaxed">
    Grad-CAM (Gradient-weighted Class Activation Mapping) เป็นเทคนิคที่ใช้ในการสร้าง heatmap บนภาพ โดยอาศัย gradient ที่ไหลย้อนจาก class prediction ไปยัง convolutional layer สุดท้าย เพื่อดูว่าโมเดลให้ความสนใจส่วนไหนของภาพในการตัดสินใจ
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>ช่วยให้เห็นว่าจุดโฟกัสของโมเดลสอดคล้องกับวัตถุจริงหรือไม่</li>
    <li>สามารถนำไปใช้ตรวจสอบว่ามี bias หรือ artifact ที่โมเดลเรียนรู้ผิดหรือไม่</li>
    <li>ใช้งานได้ดีโดยเฉพาะกับโมเดล CNN ที่มีลักษณะเป็น hierarchical feature extractor</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">การใช้งาน Grad-CAM</h3>
  <p className="mb-4 leading-relaxed">
    การสร้าง heatmap ด้วย Grad-CAM จะต้องมีการเข้าถึง activations และ gradients ของ convolutional layer สุดท้าย ตัวอย่างโค้ดใน PyTorch มีดังนี้:
  </p>
  <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto">
{`
import torch
import torchvision.models as models
from gradcam import GradCAM
from gradcam.utils import visualize_cam

model = models.resnet50(pretrained=True)
model.eval()

# กำหนด layer ที่ต้องการดู
target_layer = model.layer4[-1]

gradcam = GradCAM(model, target_layer)
mask, _ = gradcam(img_tensor)
heatmap, result = visualize_cam(mask, img_tensor)
`}
  </pre>

  <h3 className="text-xl font-semibold mb-3">Attention Visualization สำหรับ Text (NLP)</h3>
  <p className="mb-4 leading-relaxed">
    สำหรับงานด้าน NLP โมเดลเช่น BERT และ Transformer จะมี attention layer ที่สามารถดึงมา visualize ได้ เพื่อดูว่า token ไหนให้ผลต่อการตัดสินใจของโมเดลมากที่สุด
  </p>
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li>สามารถใช้วิเคราะห์ว่าคำไหนในประโยคที่โมเดลให้ความสำคัญ</li>
    <li>ช่วยตรวจจับ bias เช่น โมเดลให้ความสำคัญกับคำที่ไม่ควรมีผล</li>
    <li>สามารถใช้ตรวจสอบ explainability ในงานแปลภาษา การสรุป หรือการจัดกลุ่มข้อความ</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">ตัวอย่างการดู Attention</h3>
  <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto">
{`
from transformers import BertTokenizer, BertModel
import torch

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Explainability in AI is important", return_tensors="pt")
outputs = model(**inputs)

attentions = outputs.attentions  # list of attention matrices
`}
  </pre>
  <p className="mb-4 leading-relaxed">
    ค่า attentions ที่ได้เป็น tensor ขนาด [batch, num_heads, seq_len, seq_len] ซึ่งสามารถนำไป plot ด้วย Heatmap เช่น Matplotlib หรือ Seaborn เพื่อดูว่า token ไหนสนใจกับ token ใดมากที่สุด
  </p>

  <h3 className="text-xl font-semibold mb-3">Insight: ความสำคัญของการอธิบายข้อมูลภาพและข้อความ</h3>
  <p className="mb-4 leading-relaxed">
    การที่เราสามารถเห็นว่าโมเดลให้ความสำคัญกับส่วนใดของ input ไม่เพียงแต่ช่วยเพิ่มความมั่นใจในการ deploy เท่านั้น แต่ยังเป็นการเปิดโอกาสให้ปรับปรุงโมเดลในด้าน fairness, robustness และ ethical AI ได้ด้วย
  </p>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การอธิบายโมเดลบนข้อมูลภาพและข้อความไม่เพียงช่วยตรวจสอบว่าโมเดล “คิดถูกทาง” หรือไม่ แต่ยังทำให้ AI กลายเป็นเครื่องมือที่โปร่งใสและไว้วางใจได้มากขึ้น
  </div>
</section>


<section id="tradeoff" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Accuracy vs Explainability</h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <p className="mb-4 leading-relaxed">
    ในการเลือกใช้โมเดลสำหรับปัญหาทาง Machine Learning หรือ AI หนึ่งในสิ่งที่ต้องพิจารณาอย่างรอบคอบคือการสร้างสมดุลระหว่าง "ความแม่นยำของโมเดล (Accuracy)" และ "ความสามารถในการอธิบายผลลัพธ์ (Explainability)" ซึ่งเป็นสิ่งที่มักจะสวนทางกันในหลายกรณี โดยเฉพาะเมื่อใช้โมเดลที่ซับซ้อน
  </p>

  <h3 className="text-xl font-semibold mt-8 mb-3">ทำไมจึงต้องแลกเปลี่ยน?</h3>
  <p className="mb-4 leading-relaxed">
    โมเดลที่ซับซ้อน เช่น Neural Network มักจะมี performance สูง แต่เข้าใจยากในระดับรายละเอียด ว่าเหตุใดจึงให้ผลลัพธ์เช่นนั้น ขณะที่โมเดลที่เรียบง่ายอย่าง Linear Regression หรือ Decision Tree แม้อธิบายง่าย แต่บางครั้งก็ไม่แม่นยำเพียงพอสำหรับปัญหาที่ซับซ้อน
  </p>

  <div className="overflow-x-auto">
    <table className="table-auto w-full border-collapse border text-sm text-left">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="px-4 py-2 border">Model</th>
          <th className="px-4 py-2 border">Accuracy</th>
          <th className="px-4 py-2 border">Explainability</th>
          <th className="px-4 py-2 border">เหมาะกับงาน</th>
        </tr>
      </thead>
      <tbody>
        <tr className="hover:bg-gray-100 dark:hover:bg-gray-800">
          <td className="px-4 py-2 border">Linear Regression</td>
          <td className="px-4 py-2 border">⭐️⭐️</td>
          <td className="px-4 py-2 border">⭐️⭐️⭐️⭐️⭐️</td>
          <td className="px-4 py-2 border">วิเคราะห์เชิงธุรกิจทั่วไป, งานที่ต้องอธิบายต่อ stakeholder</td>
        </tr>
        <tr className="hover:bg-gray-100 dark:hover:bg-gray-800">
          <td className="px-4 py-2 border">Decision Tree</td>
          <td className="px-4 py-2 border">⭐️⭐️⭐️</td>
          <td className="px-4 py-2 border">⭐️⭐️⭐️⭐️</td>
          <td className="px-4 py-2 border">ปัญหาที่ต้องการ logic ที่เข้าใจง่าย</td>
        </tr>
        <tr className="hover:bg-gray-100 dark:hover:bg-gray-800">
          <td className="px-4 py-2 border">Random Forest</td>
          <td className="px-4 py-2 border">⭐️⭐️⭐️⭐️</td>
          <td className="px-4 py-2 border">⭐️⭐️</td>
          <td className="px-4 py-2 border">classification ที่ไม่ต้องการ interpret รายตัว</td>
        </tr>
        <tr className="hover:bg-gray-100 dark:hover:bg-gray-800">
          <td className="px-4 py-2 border">Neural Network</td>
          <td className="px-4 py-2 border">⭐️⭐️⭐️⭐️⭐️</td>
          <td className="px-4 py-2 border">⭐️</td>
          <td className="px-4 py-2 border">งานภาพ เสียง ข้อความ ที่ต้องการ performance สูง</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-8 mb-3 text-center">เกณฑ์ในการเลือกโมเดล</h3>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>หากใช้ในระบบที่ผู้ใช้ต้องเชื่อมั่น เช่น ระบบการแพทย์หรือการเงิน ควรเลือกโมเดลที่อธิบายได้</li>
    <li>ถ้าเป้าหมายคือลด error ให้ต่ำที่สุด เช่น งาน prediction ที่ไม่มีผลกระทบต่อผู้คนโดยตรง อาจใช้โมเดลที่แม่นแต่ไม่อธิบายได้</li>
    <li>ในงานที่ต้องขออนุมัติจากภายนอก เช่น กฎหมาย หรือรัฐ ควรใช้โมเดลที่ตีความได้ง่ายและตรวจสอบได้</li>
    <li>ระบบ recommendation อาจใช้ hybrid model โดยมี black-box ใน backend และ logic ชัดเจนใน frontend</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-3">แนวทางเพื่อสร้างสมดุล</h3>
  <ul className="list-disc pl-6 space-y-2 mb-4">
    <li>ใช้ XAI (Explainable AI) เช่น SHAP, LIME เพื่ออธิบายผลลัพธ์ของโมเดลที่ไม่สามารถแปลได้ตรง ๆ</li>
    <li>ใช้ Surrogate Model เช่น Decision Tree เพื่อ approximate ผลลัพธ์จาก black-box model</li>
    <li>สร้างระบบ visual dashboard ให้ผู้ใช้อธิบายได้ เช่น highlight feature ที่มีผลต่อการตัดสินใจ</li>
    <li>บันทึกเหตุผลการตัดสินใจของโมเดลไว้ทุกครั้งที่มีการ deploy หรือ retrain</li>
    <li>ฝึกทีมให้เข้าใจการทำงานของโมเดลและเลือกเครื่องมือให้เหมาะกับความสามารถของผู้ใช้งาน</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8 mb-3">Insight สรุป</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow">
    <p className="mb-2">Explainability ไม่ใช่ตัวเลือกเสริม แต่เป็นหนึ่งในเงื่อนไขความรับผิดชอบของ AI ที่จะนำไปใช้จริง</p>
    <p className="mb-2">การเลือกโมเดลที่โปร่งใสไม่เพียงเพิ่มความไว้วางใจจากผู้ใช้ แต่ยังช่วยให้วิเคราะห์และพัฒนาโมเดลได้ต่อเนื่องในระยะยาว</p>
    <p className="italic">ในอนาคต ความสามารถในการอธิบายอาจกลายเป็นข้อบังคับทางกฎหมายในหลายอุตสาหกรรม</p>
  </div>
</section>

<section id="best-practice" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Best Practice สำหรับ Explainability</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <p className="mb-4 leading-relaxed">
    การออกแบบระบบ Machine Learning ที่สามารถอธิบายได้ ไม่ได้เป็นเพียงความต้องการเชิงเทคนิค แต่ยังเกี่ยวข้องกับความโปร่งใส ความน่าเชื่อถือ และการยอมรับจากผู้ใช้งานจริง
  </p>

  <h3 className="text-xl font-semibold mb-3">1. เริ่มวางแผน Explainability ตั้งแต่ต้น</h3>
  <p className="mb-4 leading-relaxed">
    เลือกโมเดลและสถาปัตยกรรมที่สามารถตรวจสอบการตัดสินใจได้ง่าย เช่น Tree-based models หรือ Linear models พร้อมวางแผนเพื่อเก็บ Feature Importance หลังการ Train
  </p>

  <h3 className="text-xl font-semibold mb-3">2. ใช้เทคนิคที่เหมาะกับข้อมูล</h3>
  <ul className="list-disc pl-6 mb-4 space-y-2">
    <li>Text: Attention Weights, SHAP (Transformer, BERT)</li>
    <li>Image: Grad-CAM, Saliency Map</li>
    <li>Tabular: SHAP, Permutation Importance, Coefficients</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">3. สร้าง Dashboard ที่ตีความได้</h3>
  <p className="mb-4 leading-relaxed">
    จัดทำ Visualization UI เพื่อให้คนในทีม Business หรือผู้ใช้งานสามารถดูเหตุผลการตัดสินใจของโมเดล เช่น Highlight Feature ที่มีผล, กราฟแสดงการเปรียบเทียบความสำคัญ
  </p>

  <h3 className="text-xl font-semibold mb-3">4. ตรวจสอบ Bias อย่างสม่ำเสมอ</h3>
  <p className="mb-4 leading-relaxed">
    ใช้ Explainability Tools เพื่อตรวจว่าโมเดลใช้ข้อมูลที่ sensitive เช่น เพศ หรือเชื้อชาติ มากเกินไปหรือไม่ ถ้ามี ควรพิจารณาเปลี่ยน Feature หรือใส่ constraint
  </p>

  <h3 className="text-xl font-semibold mb-3">5. ตรวจสอบผลลัพธ์หลัง Retrain</h3>
  <p className="mb-4 leading-relaxed">
    หลัง Retrain ควรบันทึกเปรียบเทียบว่า Feature ที่โมเดลให้ความสำคัญมีการเปลี่ยนแปลงหรือไม่ และเหตุผลในการทำนายยังคงเสถียรหรือไม่ เพื่อป้องกัน Drift
  </p>

  <h3 className="text-xl font-semibold mb-3">6. ให้ Feedback Loop จากผู้ใช้</h3>
  <p className="mb-4 leading-relaxed">
    ให้ผู้ใช้สามารถโต้ตอบกับผลลัพธ์ได้ เช่น กดรายงานผลลัพธ์ที่ไม่เหมาะสมหรืออธิบายผิด เพื่อใช้ข้อมูลนี้มาปรับปรุงโมเดลในรุ่นถัดไป
  </p>

  <h3 className="text-xl font-semibold mb-3">7. สร้างชุดข้อมูลสำหรับการตรวจสอบ</h3>
  <p className="mb-4 leading-relaxed">
    นอกจาก Test Set ทั่วไป ควรมีชุด “Test for Reasoning” ที่ใช้สำหรับตรวจว่าโมเดลอธิบายด้วยเหตุผลที่ถูกต้อง เช่น ตัวอย่างที่มีความหมายแต่ได้ผลลัพธ์ที่ผิดแปลก
  </p>

  <h3 className="text-xl font-semibold mb-3">8. ฝังระบบ Explainability ใน Workflow</h3>
  <p className="mb-4 leading-relaxed">
    การแสดงเหตุผลควรอยู่ในทุกหน้าที่ผู้ใช้ตัดสินใจ เช่น ระบบคัดคนเข้าทำงานควรบอกเหตุผลประกอบ ไม่ใช่แค่คะแนนเพื่อให้โปร่งใสและตรวจสอบได้
  </p>

  <h3 className="text-xl font-semibold mb-3">9. เลือกโมเดลตามความจำเป็น</h3>
  <p className="mb-4 leading-relaxed">
    บางงานเช่น วิจัย อาจใช้โมเดลซับซ้อนได้ แต่ระบบที่ใช้งานจริงควรเลือกโมเดลที่อธิบายได้ เช่น Tree, Logistic Regression เพื่อความน่าเชื่อถือ
  </p>

  <h3 className="text-xl font-semibold mb-3">10. ให้ความรู้ทีมงานเรื่อง Explainability</h3>
  <p className="mb-4 leading-relaxed">
    อบรมทีมที่เกี่ยวข้องให้เข้าใจว่า Explainability สำคัญอย่างไร เพื่อให้มีความร่วมมือและทิศทางเดียวกันในการพัฒนาโมเดลที่ตรวจสอบได้จริง
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">Insight:</p>
    <p>
      Explainability ที่ดีไม่ใช่แค่แสดงข้อมูล แต่ต้องเชื่อมโยงกับความเข้าใจของมนุษย์ และสามารถตรวจสอบ ปรับปรุง และตัดสินใจได้อย่างมั่นใจในโลกความเป็นจริง
    </p>
  </div>
</section>


<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">Insight เปรียบเทียบ: ความแม่นยำ vs ความเข้าใจ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <p className="mb-4 leading-relaxed">
    ความแม่นยำ (Accuracy) และความสามารถในการอธิบาย (Explainability) ไม่ใช่แค่ตัวชี้วัดเชิงเทคนิค แต่คือ "หลักคิด" สองแบบที่กำหนดแนวทางการออกแบบ AI ให้เหมาะกับวัตถุประสงค์ในแต่ละบริบท
  </p>

  <div className="grid md:grid-cols-2 gap-6 my-6">
    <div className="bg-white dark:bg-gray-800 p-5 border rounded-xl shadow text-sm">
      <h4 className="font-semibold mb-2">มุมมองด้าน Accuracy</h4>
      <ul className="list-disc pl-5 space-y-2">
        <li>เน้นผลลัพธ์ที่ถูกต้องแม่นยำ</li>
        <li>เหมาะกับงานที่ต้อง optimize performance เช่น การทำนาย</li>
        <li>ใช้วัดความ "เก่ง" ของโมเดลในเชิงตัวเลข</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-800 p-5 border rounded-xl shadow text-sm">
      <h4 className="font-semibold mb-2">มุมมองด้าน Explainability</h4>
      <ul className="list-disc pl-5 space-y-2">
        <li>เน้นเข้าใจ "เหตุผลเบื้องหลัง"</li>
        <li>เหมาะกับงานที่มีผลกระทบสูง เช่น การแพทย์ การเงิน ความยุติธรรม</li>
        <li>ช่วยสร้างความไว้วางใจ ความโปร่งใส และความรับผิดชอบ</li>
      </ul>
    </div>
  </div>

  <p className="mb-4 leading-relaxed">
    ในหลายกรณี ความแม่นยำสูงไม่ได้แปลว่าโมเดล “ปลอดภัย” หรือ “ยุติธรรม” หากโมเดลพยากรณ์แม่นยำโดยอิงจากฟีเจอร์ที่มี bias หรือไม่เหมาะสม ความแม่นยำจะกลายเป็นภาพลวงตาที่ซ่อนปัญหาไว้ข้างใน
  </p>

  <p className="mb-4 leading-relaxed">
    ในขณะที่ Explainability ทำให้เราสามารถมองเห็นภายในว่าโมเดลใช้ข้อมูลใดในการตัดสินใจ ช่วยให้เราสร้างโมเดลที่ไม่เพียง “ให้คำตอบที่ถูก” แต่ “คิดอย่างถูกวิธี”
  </p>

  <p className="mb-4 leading-relaxed">
    ความท้าทายคือการสร้างสมดุลระหว่างสองสิ่งนี้ — ไม่เลือกแค่โมเดลที่ดีที่สุดบนกระดาษ แต่เลือกโมเดลที่ดีที่สุดในบริบทของ “มนุษย์” และ “ความเป็นจริง”
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-500 shadow mt-8">
    <p className="font-semibold mb-2">สรุป Insight:</p>
    <p className="mb-2">Accuracy คือคำตอบที่แม่น</p>
    <p className="mb-2">Explainability คือเหตุผลที่ดี</p>
    <p className="mb-2">AI ที่น่าเชื่อถือในโลกจริง ต้อง “ทั้งแม่น และเข้าใจได้”</p>
    <p className="mt-2 italic text-sm">เพราะความโปร่งใส ไม่ได้เป็นแค่คุณสมบัติของโมเดล — แต่คือหลักการของการออกแบบเทคโนโลยีที่รับผิดชอบ</p>
  </div>
</section>



        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day13 theme={theme} />
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
        <ScrollSpy_Ai_Day13 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day13_ModelExplainability;
