import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day61 from "../scrollspy/scrollspyDay61-80/ScrollSpy_Ai_Day61.jsx";
import MiniQuiz_Day61 from "../miniquiz/miniquizDay61-80/MiniQuiz_Day61";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day61_IntroRL = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  // Image assets (ตามลำดับหัวข้อ)
  const img1 = cld.image("Day61_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day61_2").format("auto").quality("auto").resize(scale().width(660));
  const img3 = cld.image("Day61_3").format("auto").quality("auto").resize(scale().width(660));
  const img4 = cld.image("Day61_4").format("auto").quality("auto").resize(scale().width(660));
  const img5 = cld.image("Day61_5").format("auto").quality("auto").resize(scale().width(660));
  const img6 = cld.image("Day61_6").format("auto").quality("auto").resize(scale().width(660));
  const img7 = cld.image("Day61_7").format("auto").quality("auto").resize(scale().width(660));
  const img8 = cld.image("Day61_8").format("auto").quality("auto").resize(scale().width(660));
  const img9 = cld.image("Day61_9").format("auto").quality("auto").resize(scale().width(660));
  const img10 = cld.image("Day61_10").format("auto").quality("auto").resize(scale().width(660));
  const img11 = cld.image("Day61_11").format("auto").quality("auto").resize(scale().width(660));
  const img12 = cld.image("Day61_12").format("auto").quality("auto").resize(scale().width(660));
  const img13 = cld.image("Day61_13").format("auto").quality("auto").resize(scale().width(660));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      {/* Sidebar */}
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>
    


      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 61: Introduction to Reinforcement Learning</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

       <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: Reinforcement Learning คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">แนวคิดหลักของ Reinforcement Learning</h3>
    <p>
      Reinforcement Learning (RL) คือกรอบการเรียนรู้ที่มีการปฏิสัมพันธ์ระหว่าง Agent และ Environment ซึ่ง Agent จะเรียนรู้ผ่านกระบวนการลองผิดลองถูก เพื่อให้สามารถตัดสินใจได้ดีขึ้นในอนาคต โดยเป้าหมายของ RL คือการเรียนรู้กลยุทธ์ (policy) ที่ทำให้ Agent ได้รับผลตอบแทนรวมสูงสุด (maximum cumulative reward)
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>Reinforcement Learning แตกต่างจากการเรียนรู้แบบมีผู้สอน (Supervised Learning) ตรงที่ไม่มีการบอกคำตอบที่ถูกต้องให้กับ Agent แต่ให้ผลตอบแทนแทน</p>
    </div>

    <h3 className="text-xl font-semibold">องค์ประกอบพื้นฐานของ RL</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>State (S):</strong> สภาพแวดล้อมที่ Agent สังเกตเห็น ณ เวลาหนึ่ง</li>
      <li><strong>Action (A):</strong> ชุดการกระทำที่ Agent สามารถเลือกได้</li>
      <li><strong>Reward (R):</strong> ผลตอบแทนที่ได้รับหลังจากดำเนิน action</li>
      <li><strong>Policy (π):</strong> กลยุทธ์ที่ Agent ใช้เลือกการกระทำในแต่ละสถานะ</li>
      <li><strong>Value Function (V):</strong> การประมาณค่าว่าการอยู่ในสถานะใดจะได้ผลตอบแทนรวมในอนาคตเท่าใด</li>
      <li><strong>Model (optional):</strong> แบบจำลองของ environment ที่ Agent ใช้ทำนายผลลัพธ์ของ action</li>
    </ul>

    <h3 className="text-xl font-semibold">สูตรผลตอบแทนสะสมแบบลดค่า</h3>
    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
      <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
        <code>{`G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^{∞} γ^k R_{t+k+1}`}</code>
      </pre>
    </div>

    <p>โดยที่ γ (gamma) เป็น discount factor ในช่วง 0 ถึง 1 ซึ่งใช้ลดความสำคัญของรางวัลที่ได้รับในอนาคต</p>

    <h3 className="text-xl font-semibold">ตัวอย่างการประยุกต์ใช้งาน RL</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>AlphaGo ของ DeepMind ใช้ RL ในการเรียนรู้กลยุทธ์การเล่นหมากล้อมที่เหนือกว่ามนุษย์</li>
      <li>การควบคุมหุ่นยนต์เดินในพื้นที่ไม่แน่นอน</li>
      <li>การบริหารจัดการทรัพยากรในคลาวด์แบบอัตโนมัติ</li>
      <li>การสร้างนโยบายการลงทุนแบบปรับตัวได้</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Highlight:</strong>
      <p>RL เป็นพื้นฐานของระบบอัตโนมัติที่เรียนรู้ได้เอง เช่น autonomous vehicles, industrial robotics และ adaptive healthcare systems</p>
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับ Learning รูปแบบอื่น</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">รูปแบบการเรียนรู้</th>
          <th className="border px-4 py-2">ลักษณะ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Supervised Learning</td>
          <td className="border px-4 py-2">เรียนรู้จากคู่ข้อมูล-คำตอบ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Unsupervised Learning</td>
          <td className="border px-4 py-2">ค้นหาโครงสร้างในข้อมูล</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Reinforcement Learning</td>
          <td className="border px-4 py-2">เรียนรู้จากการปฏิสัมพันธ์และรางวัลที่ได้รับ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.</li>
      <li>Silver, D. et al. (2016). Mastering the game of Go with deep neural networks and tree search. Nature.</li>
      <li>Arulkumaran, K. et al. (2017). Deep reinforcement learning: A brief survey. IEEE Signal Processing Magazine.</li>
    </ul>
  </div>
</section>

       <section id="comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. เปรียบเทียบกับ Learning แบบอื่น</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">กรอบแนวคิดของการเรียนรู้ใน Machine Learning</h3>
    <p>
      การเรียนรู้ในบริบทของปัญญาประดิษฐ์สามารถแบ่งออกได้เป็นสามรูปแบบหลัก ได้แก่ Supervised Learning, Unsupervised Learning และ Reinforcement Learning ซึ่งแต่ละรูปแบบมีวิธีการเรียนรู้ ความต้องการข้อมูล และลักษณะการใช้งานที่แตกต่างกันอย่างมีนัยสำคัญ
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        แม้ว่า Supervised Learning จะเป็นกระแสหลักในงานเชิงพาณิชย์ แต่ Reinforcement Learning กลับมีศักยภาพที่โดดเด่นในการสร้างระบบอัตโนมัติที่สามารถปรับตัวได้ในสภาพแวดล้อมที่เปลี่ยนแปลงและไม่แน่นอน
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Learning รูปแบบต่าง ๆ</h3>
    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600">
  <table className="min-w-[800px] table-auto border-collapse text-sm sm:text-base">
    <thead className="bg-gray-200 dark:bg-gray-700 text-left">
      <tr>
        <th className="border px-4 py-2">ประเภทการเรียนรู้</th>
        <th className="border px-4 py-2">คำอธิบาย</th>
        <th className="border px-4 py-2">ข้อมูลที่ต้องการ</th>
        <th className="border px-4 py-2">ลักษณะผลลัพธ์</th>
        <th className="border px-4 py-2">การประยุกต์ใช้</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">Supervised Learning</td>
        <td className="border px-4 py-2">เรียนรู้จากข้อมูลที่มีป้ายกำกับ (label)</td>
        <td className="border px-4 py-2">คู่ข้อมูล (X, Y)</td>
        <td className="border px-4 py-2">จำแนก/ทำนายค่าที่รู้ล่วงหน้า</td>
        <td className="border px-4 py-2">การรู้จำภาพ, การวิเคราะห์ข้อความ</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Unsupervised Learning</td>
        <td className="border px-4 py-2">ค้นหารูปแบบจากข้อมูลที่ไม่มี label</td>
        <td className="border px-4 py-2">เฉพาะ X (ไม่มี Y)</td>
        <td className="border px-4 py-2">กลุ่ม/โครงสร้างซ่อนเร้น</td>
        <td className="border px-4 py-2">การจัดกลุ่มลูกค้า, anomaly detection</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Reinforcement Learning</td>
        <td className="border px-4 py-2">เรียนรู้จากการปฏิสัมพันธ์กับ environment</td>
        <td className="border px-4 py-2">สถานะ, การกระทำ, รางวัล</td>
        <td className="border px-4 py-2">กลยุทธ์ (policy) ที่เหมาะสมที่สุด</td>
        <td className="border px-4 py-2">เกม, robotics, autonomous driving</td>
      </tr>
    </tbody>
  </table>
</div>

    <h3 className="text-xl font-semibold">ตัวอย่างสถานการณ์ที่เหมาะกับแต่ละรูปแบบ</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Supervised:</strong> การจำแนกประเภทอีเมลว่าเป็น spam หรือไม่</li>
      <li><strong>Unsupervised:</strong> การหากลุ่มลูกค้าที่มีพฤติกรรมคล้ายกัน</li>
      <li><strong>RL:</strong> การเรียนรู้วิธีควบคุมหุ่นยนต์ให้เดินได้ในภูมิประเทศใหม่</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อได้เปรียบและข้อจำกัดของ RL</h3>
    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        RL มีจุดแข็งในการแก้ปัญหาที่ไม่มีคำตอบที่แน่นอน แต่ต้องอาศัยการสำรวจและการเรียนรู้แบบ adaptive อย่างต่อเนื่อง อย่างไรก็ตาม RL มีข้อจำกัดเช่น computational cost สูง และต้องใช้การทดลองซ้ำจำนวนมาก
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้ RL ในกรณีที่ SL/UL ไม่สามารถทำได้</h3>
    <p>
      ในระบบควบคุมโดรนอัตโนมัติที่ต้องบินผ่านพื้นที่ที่ไม่เคยรู้จักมาก่อน การใช้ supervised model ที่ถูก train มาก่อนอาจไม่มีประสิทธิภาพเพียงพอ ในขณะที่ RL สามารถเรียนรู้และปรับกลยุทธ์การบินตาม feedback ที่ได้รับจากสภาพแวดล้อมโดยตรง
    </p>

    <h3 className="text-xl font-semibold">บทสรุปการเปรียบเทียบเชิงแนวคิด</h3>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`Supervised → คำตอบมีให้ → ลดความไม่แน่นอนของโมเดล
Unsupervised → ไม่มีคำตอบ → ค้นหาโครงสร้างในข้อมูล
Reinforcement → ไม่มีคำตอบแน่นอน → เรียนรู้ผ่านรางวัลจากการทดลองซ้ำ`}</code></pre>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
      <li>Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.</li>
      <li>Ng, A. (2023). CS229: Machine Learning. Stanford University.</li>
      <li>Arora, S. et al. (2022). "Theoretical Insights into Supervised vs. RL Paradigms", arXiv:2211.00101</li>
    </ul>
  </div>
</section>

       <section id="components" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. ส่วนประกอบพื้นฐานของ RL</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">กรอบแนวคิด Markov Decision Process (MDP)</h3>
    <p>
      การกำหนดปัญหาใน Reinforcement Learning ส่วนใหญ่จะอิงจาก Markov Decision Process (MDP) ซึ่งเป็นกรอบทางคณิตศาสตร์สำหรับโมเดลของระบบที่มีความไม่แน่นอนและลำดับการตัดสินใจ MDP ประกอบด้วย:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>𝒮 (States):</strong> เซตของสถานะทั้งหมดที่ environment สามารถอยู่ได้</li>
      <li><strong>𝒜 (Actions):</strong> เซตของการกระทำที่ agent สามารถเลือกในแต่ละสถานะ</li>
      <li><strong>𝒫 (Transition Probabilities):</strong> ความน่าจะเป็นในการย้ายจากสถานะหนึ่งไปอีกสถานะหนึ่งหลังจากทำ action</li>
      <li><strong>ℛ (Reward Function):</strong> ฟังก์ชันที่ให้รางวัลเมื่อมีการเปลี่ยนแปลงสถานะ</li>
      <li><strong>γ (Discount Factor):</strong> ตัวลดค่าความสำคัญของรางวัลในอนาคต</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        MDP เป็นรากฐานของ RL ที่เปิดโอกาสให้โมเดลสามารถเรียนรู้และวางแผนการตัดสินใจในสภาวะที่ผลลัพธ์มีความไม่แน่นอนและยืดหยุ่นตามเวลาจริง
      </p>
    </div>

    <h3 className="text-xl font-semibold">องค์ประกอบหลักของระบบ RL</h3>
    <p>โมเดลพื้นฐานของ RL ประกอบด้วยองค์ประกอบหลักดังนี้:</p>

    <ul className="list-disc list-inside space-y-2">
      <li><strong>Agent:</strong> ตัวเรียนรู้หรือสิ่งที่ตัดสินใจเลือก action</li>
      <li><strong>Environment:</strong> โลกที่ agent เข้าไปปฏิสัมพันธ์ด้วย</li>
      <li><strong>Policy (π):</strong> กลยุทธ์ของ agent ที่ใช้เลือก action จากสถานะ</li>
      <li><strong>Reward Signal (r):</strong> ตัววัดว่าการกระทำนั้น "ดี" แค่ไหนในบริบทปัจจุบัน</li>
      <li><strong>Value Function (V):</strong> การประเมินค่าความดีของสถานะ หรือคู่สถานะ-การกระทำ</li>
      <li><strong>Model of Environment:</strong> (ถ้ามี) ใช้สำหรับการเรียนรู้แบบ model-based</li>
    </ul>

    <h3 className="text-xl font-semibold">นโยบาย (Policy)</h3>
    <p>
      นโยบาย (Policy) คือนิยามของการเลือกการกระทำในแต่ละสถานะ โดยอาจเป็นแบบ deterministic หรือ stochastic:
    </p>

    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 mb-6">
      <pre className="whitespace-pre text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
        <code>{`π(s) = a             # deterministic policy
π(a|s) = P(a|s)     # stochastic policy (probability distribution)`}</code>
      </pre>
    </div>

    <h3 className="text-xl font-semibold">ฟังก์ชันคุณค่า (Value Functions)</h3>
    <p>
      Value Function ช่วยให้ agent ประเมินความน่าจะเป็นในการได้รับรางวัลในอนาคตเมื่ออยู่ในสถานะ s หรือเลือก action a ในสถานะ s โดยสามารถแบ่งเป็น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>V(s):</strong> ค่าคาดหวังของ cumulative reward เมื่อเริ่มต้นที่สถานะ s</li>
      <li><strong>Q(s, a):</strong> ค่าคาดหวังของ cumulative reward เมื่อเริ่มที่ s และเลือก a</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        Q-learning เป็นเทคนิคยอดนิยมที่ใช้ประมาณ Q(s, a) โดยไม่ต้องใช้โมเดลของ environment และเป็นหัวใจสำคัญของหลายอัลกอริธึม deep RL ในปัจจุบัน
      </p>
    </div>

    <h3 className="text-xl font-semibold">Reward Signal และ Goal Optimization</h3>
    <p>
      เป้าหมายของ RL คือการหานโยบาย π ที่ทำให้ผลรวมของ reward สูงที่สุด ซึ่ง reward signal ที่กำหนดมีอิทธิพลโดยตรงต่อพฤติกรรมที่ agent จะเรียนรู้:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Reward ต้องมีความสอดคล้องกับ objective function ที่ต้องการในระบบจริง</li>
      <li>Reward ที่กำหนดไม่เหมาะ อาจนำไปสู่การเรียนรู้พฤติกรรมที่ไม่พึงประสงค์</li>
    </ul>

    <h3 className="text-xl font-semibold">Model vs. Model-Free RL</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">ลักษณะ</th>
          <th className="border px-4 py-2">Model-Based RL</th>
          <th className="border px-4 py-2">Model-Free RL</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ใช้แบบจำลอง</td>
          <td className="border px-4 py-2">มีการเรียนรู้หรือให้โมเดลของ environment</td>
          <td className="border px-4 py-2">ไม่ใช้โมเดล จำลองผ่านการลองจริง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความยืดหยุ่น</td>
          <td className="border px-4 py-2">สามารถวางแผนล่วงหน้าได้</td>
          <td className="border px-4 py-2">เรียนรู้ผ่าน trial-and-error</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">อัลกอริธึมเด่น</td>
          <td className="border px-4 py-2">Dyna-Q, MuZero</td>
          <td className="border px-4 py-2">Q-learning, DQN, A3C</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">สรุปองค์ประกอบทั้งหมดในรูปแบบ flow</h3>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`[Environment] 
     ↑   ↓ reward/observation
[Agent]
     ↓ action
[Policy] → [Value Function] → (optional: Model)`}</code></pre>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Sutton, R. S., & Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i>. MIT Press.</li>
      <li>Silver, D. (2020). <i>Deep RL Lecture Series</i>, University College London.</li>
      <li>Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms", arXiv:1707.06347</li>
      <li>Hafner, D. et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination", ICLR.</li>
    </ul>
  </div>
</section>


 <section id="loop" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Loop การทำงานของ RL (Agent–Environment Interaction)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">โครงสร้างพื้นฐานของ Agent–Environment Loop</h3>
    <p>
      กระบวนการเรียนรู้ใน Reinforcement Learning (RL) ถูกนิยามผ่านรูปแบบการปฏิสัมพันธ์ระหว่าง <strong>Agent</strong> และ <strong>Environment</strong> โดย Agent เป็นผู้ตัดสินใจเลือก action และ Environment เป็นผู้ตอบสนองด้วย state และ reward ตามกฎของระบบ โดยกระบวนการนี้เกิดขึ้นแบบวนลูปในลักษณะต่อเนื่อง
    </p>

    <h3 className="text-xl font-semibold">ขั้นตอนของ Interaction Loop</h3>
        <ul className="list-decimal list-inside space-y-2">
        <li><strong>Agent สังเกตสภาพแวดล้อม</strong> โดยรับค่า state ปัจจุบัน sₜ</li>
        <li><strong>Agent เลือก action</strong> ตามนโยบาย π(a|s)</li>
        <li><strong>Environment เปลี่ยน state</strong> ไปยัง sₜ₊₁</li>
        <li><strong>Environment ส่ง reward</strong> rₜ₊₁ ให้กับ Agent</li>
        <li>Agent นำข้อมูล (sₜ, aₜ, rₜ₊₁, sₜ₊₁) ไปอัปเดต policy</li>
        </ul>


    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        ความงามของ RL อยู่ที่การเรียนรู้ผ่านการกระทำจริง ไม่ได้อิงข้อมูลที่มี label ล่วงหน้า ซึ่งทำให้ระบบสามารถค้นหากลยุทธ์ใหม่ในบริบทที่ไม่แน่นอนหรือเปลี่ยนแปลงได้ตลอดเวลา
      </p>
    </div>

    <h3 className="text-xl font-semibold">แสดงการไหลของ Interaction</h3>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`while not done:
    s_t = observe_environment()
    a_t = select_action(s_t)
    s_{t+1}, r_{t+1} = environment.step(a_t)
    update_policy(s_t, a_t, r_{t+1}, s_{t+1})`}</code></pre>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบการทำงาน Agent และ Environment</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">Component</th>
          <th className="border px-4 py-2">หน้าที่</th>
          <th className="border px-4 py-2">ข้อมูลที่เกี่ยวข้อง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Agent</td>
          <td className="border px-4 py-2">เลือกการกระทำและอัปเดต policy</td>
          <td className="border px-4 py-2">s_t, a_t, π(a|s)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Environment</td>
          <td className="border px-4 py-2">เปลี่ยนแปลงสถานะและให้รางวัล</td>
          <td className="border px-4 py-2">sₜ₊₁, rₜ₊₁</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ตัวอย่างในโลกจริง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ในเกม: Agent เลือกการเคลื่อนไหว → เกมเปลี่ยนสถานะ → ให้คะแนน</li>
      <li>ในระบบหุ่นยนต์: หุ่นยนต์เลือกการขยับ → เซนเซอร์ตรวจ feedback → ระบบเรียนรู้จากผลลัพธ์</li>
      <li>ในระบบ recommendation: Agent แนะนำเนื้อหา → ผู้ใช้ตอบสนอง → ระบบอัปเดตความน่าจะเป็น</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        การทำความเข้าใจ interaction loop อย่างลึกซึ้ง เป็นกุญแจสำคัญในการออกแบบ RL agent ที่สามารถปรับตัวและพัฒนาได้อย่างมีประสิทธิภาพในบริบทที่หลากหลาย
      </p>
    </div>

    <h3 className="text-xl font-semibold">การวัดผลลัพธ์ในแต่ละ loop</h3>
    <p>
      หลังจากแต่ละ episode หรือ loop การเรียนรู้ มักมีการประเมิน performance ผ่าน metrics ต่าง ๆ เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Average Return:</strong> รางวัลรวมเฉลี่ยต่อ episode</li>
      <li><strong>Success Rate:</strong> ความถี่ของการบรรลุเป้าหมาย</li>
      <li><strong>Learning Curve:</strong> กราฟแสดงการพัฒนา policy</li>
    </ul>

    <h3 className="text-xl font-semibold">กรณีการวน loop แบบ online vs. batch</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">ลักษณะ</th>
          <th className="border px-4 py-2">Online RL</th>
          <th className="border px-4 py-2">Batch RL</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">การอัปเดต</td>
          <td className="border px-4 py-2">อัปเดตทุก step</td>
          <td className="border px-4 py-2">เรียนรู้จากชุดข้อมูลที่เก็บไว้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">การใช้งาน</td>
          <td className="border px-4 py-2">real-time systems</td>
          <td className="border px-4 py-2">offline training เช่น healthcare</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Sutton, R. S., & Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i>. MIT Press.</li>
      <li>Li, L. (2017). <i>A Short Course on RL</i>, Microsoft Research.</li>
      <li>Achiam, J. (2018). <i>Spinning Up in Deep RL</i>, OpenAI.</li>
      <li>Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning", <i>Nature</i>.</li>
    </ul>
  </div>
</section>


  <section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Types of Reinforcement Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">การจำแนกประเภทของ RL</h3>
    <p>
      Reinforcement Learning (RL) สามารถจำแนกออกเป็นหลายประเภทตามลักษณะการเรียนรู้ วิธีการประเมินนโยบาย หรือกลยุทธ์ในการสำรวจ (exploration strategy) โดยทั่วไปแล้วการจำแนกที่พบบ่อย ได้แก่:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Model-Based vs Model-Free RL</li>
      <li>Value-Based vs Policy-Based vs Actor-Critic Methods</li>
      <li>On-policy vs Off-policy Learning</li>
      <li>Discrete vs Continuous Action Spaces</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        ความเข้าใจในชนิดของ RL ที่หลากหลายเป็นกุญแจสำคัญในการเลือกอัลกอริธึมที่เหมาะสมกับสภาพแวดล้อม เช่น robotics, finance หรือ healthcare
      </p>
    </div>

    <h3 className="text-xl font-semibold">Model-Free vs Model-Based</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Model-Free</th>
          <th className="border px-4 py-2">Model-Based</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ใช้แบบจำลอง Environment</td>
          <td className="border px-4 py-2">ไม่ใช้</td>
          <td className="border px-4 py-2">ใช้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ตัวอย่างอัลกอริธึม</td>
          <td className="border px-4 py-2">Q-learning, DQN</td>
          <td className="border px-4 py-2">Dyna-Q, MuZero</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ความสามารถในการวางแผนล่วงหน้า</td>
          <td className="border px-4 py-2">ต่ำ</td>
          <td className="border px-4 py-2">สูง</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">Value-Based vs Policy-Based vs Actor-Critic</h3>
    <p>
      การเลือกกลยุทธ์ในการเรียนรู้ขึ้นอยู่กับว่าเน้นการเรียนรู้ค่าของ action (Q-value), การเรียนรู้ policy โดยตรง, หรือทั้งสองอย่างผสมกัน:
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li><strong>Value-Based:</strong> ใช้ Q(s,a) ในการประมาณค่าการกระทำ → เช่น Q-learning</li>
      <li><strong>Policy-Based:</strong> ปรับ policy โดยตรง → เช่น REINFORCE</li>
      <li><strong>Actor-Critic:</strong> ใช้ทั้ง policy และ value function → เช่น A2C, PPO</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        Actor-Critic เป็นทางสายกลางที่ทรงพลัง โดยใช้ “Actor” ในการเลือก action และ “Critic” ในการประเมินว่าดีหรือไม่ จึงเหมาะกับงานที่มี action space ต่อเนื่อง
      </p>
    </div>

    <h3 className="text-xl font-semibold">On-policy vs Off-policy Learning</h3>
    <p>
      การจำแนกตามนโยบายที่ใช้เก็บข้อมูล:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>On-policy:</strong> ใช้นโยบายเดียวกับที่กำลังเรียนรู้ เช่น SARSA</li>
      <li><strong>Off-policy:</strong> เรียนรู้จากนโยบายอื่น เช่น Q-learning, DDPG</li>
    </ul>

    <h3 className="text-xl font-semibold">Discrete vs Continuous Action Spaces</h3>
    <p>
      ลักษณะของ action space มีผลต่อโครงสร้างของอัลกอริธึมที่เลือกใช้:
    </p>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">ลักษณะ</th>
          <th className="border px-4 py-2">Discrete</th>
          <th className="border px-4 py-2">Continuous</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">จำนวน action</td>
          <td className="border px-4 py-2">จำกัด เช่น ซ้าย/ขวา</td>
          <td className="border px-4 py-2">ไม่จำกัด เช่น องศาการหมุน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">อัลกอริธึมที่เหมาะสม</td>
          <td className="border px-4 py-2">DQN, Q-learning</td>
          <td className="border px-4 py-2">DDPG, SAC, PPO</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ตัวอย่างการจับคู่ Use Case กับประเภทของ RL</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>เกม Atari:</strong> ใช้ DQN (Value-Based, Off-Policy, Discrete)</li>
      <li><strong>หุ่นยนต์แขนกล:</strong> ใช้ PPO หรือ SAC (Actor-Critic, Continuous)</li>
      <li><strong>Recommendation System:</strong> ใช้ Off-policy method เช่น Batch Q-learning</li>
    </ul>

    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`# ตัวอย่างการเรียนรู้แบบ Off-policy:
Q(s, a) = Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]`}</code></pre>

    <h3 className="text-xl font-semibold">สรุปภาพรวมแบบ Infographic (อธิบายด้วยโครงสร้าง)</h3>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`RL
├── Model-Free
│   ├── Value-Based → Q-learning, DQN
│   ├── Policy-Based → REINFORCE
│   └── Actor-Critic → PPO, A2C
└── Model-Based → Dyna-Q, MuZero`}</code></pre>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Silver, D. et al. (2016). <i>Mastering the game of Go</i>, Nature.</li>
      <li>Schulman, J. et al. (2017). <i>Proximal Policy Optimization Algorithms</i>, arXiv:1707.06347.</li>
      <li>Hafner, D. et al. (2021). <i>Mastering Atari with Discrete World Models</i>, ICLR.</li>
      <li>Ng, A. (2023). <i>CS229 Lecture Notes</i>, Stanford University.</li>
    </ul>
  </div>
</section>


   <section id="examples" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ตัวอย่างปัญหา RL ที่เป็นมาตรฐาน</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ความสำคัญของ Benchmark ใน RL</h3>
    <p>
      ในการวิจัยและพัฒนา Reinforcement Learning ระบบ benchmark ที่มีมาตรฐานช่วยให้สามารถวัดผล เปรียบเทียบ และประเมินประสิทธิภาพของอัลกอริธึมได้อย่างเป็นระบบ Benchmark ต่าง ๆ จึงถูกออกแบบให้มีลักษณะที่หลากหลาย ตั้งแต่งานง่ายอย่าง CartPole ไปจนถึงสภาพแวดล้อมแบบ high-dimensional เช่น MuJoCo และ StarCraft II
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        Benchmark RL environment ไม่ใช่เพียงแค่ “สนามทดลอง” แต่เป็นเครื่องมือสำคัญในการพัฒนาอัลกอริธึมที่สามารถ generalize และทำงานได้ในสถานการณ์จริงที่ซับซ้อน
      </p>
    </div>

    <h3 className="text-xl font-semibold">ประเภทของ Benchmark ที่พบบ่อย</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Classic Control:</strong> งานควบคุมแบบง่าย เช่น CartPole, MountainCar, Acrobot</li>
      <li><strong>Atari Games:</strong> งานที่ใช้ภาพ input และ action space แบบจำกัด เช่น Breakout, Pong</li>
      <li><strong>Robotic Simulation:</strong> ใช้ฟิสิกส์จริง เช่น MuJoCo, PyBullet, Isaac Gym</li>
      <li><strong>Strategic Planning:</strong> เช่น Go, Chess, StarCraft II</li>
      <li><strong>Navigation/Exploration:</strong> เช่น MiniGrid, DeepMind Lab, VizDoom</li>
    </ul>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Benchmark สำคัญ</h3>
    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600">
  <table className="min-w-[800px] table-auto border-collapse text-sm sm:text-base">
    <thead className="bg-gray-200 dark:bg-gray-700 text-left">
      <tr>
        <th className="border px-4 py-2">Benchmark</th>
        <th className="border px-4 py-2">ลักษณะเด่น</th>
        <th className="border px-4 py-2">ใช้ในงาน</th>
        <th className="border px-4 py-2">ประเภท Input</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="border px-4 py-2">CartPole</td>
        <td className="border px-4 py-2">ปัญหา balancing เบื้องต้น</td>
        <td className="border px-4 py-2">การสอนเบื้องต้น</td>
        <td className="border px-4 py-2">Low-dimension</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Atari 2600</td>
        <td className="border px-4 py-2">การควบคุมผ่านภาพ</td>
        <td className="border px-4 py-2">DQN, A3C, Rainbow</td>
        <td className="border px-4 py-2">Pixel-based</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">MuJoCo</td>
        <td className="border px-4 py-2">ฟิสิกส์แบบต่อเนื่อง</td>
        <td className="border px-4 py-2">Control, Robotics</td>
        <td className="border px-4 py-2">Continuous state</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">MiniGrid</td>
        <td className="border px-4 py-2">Navigation + Partial Observability</td>
        <td className="border px-4 py-2">Exploration, memory-based learning</td>
        <td className="border px-4 py-2">Grid-based</td>
      </tr>
    </tbody>
  </table>
</div>

    <h3 className="text-xl font-semibold">ตัวอย่าง: Deep Q-Network (DQN) กับ Atari</h3>
    <p>
      Deep Q-Network (DQN) ของ DeepMind ได้รับความสนใจอย่างสูงจากความสามารถในการเล่นเกม Atari โดยใช้ raw pixel เป็น input และ action space แบบจำกัด DQN ใช้ replay buffer และ target network เพื่อทำให้การเรียนรู้มีเสถียรภาพ
    </p>

    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]`}</code></pre>

    <h3 className="text-xl font-semibold">ตัวอย่าง: PPO กับ MuJoCo</h3>
    <p>
      Proximal Policy Optimization (PPO) เหมาะสำหรับงาน continuous control ที่ต้องการความเสถียรสูง เช่น หุ่นยนต์เดินสองขา (HalfCheetah, Walker2d) ใน MuJoCo PPO ใช้ clipping objective function เพื่อลด variance และเพิ่ม sample efficiency
    </p>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        MuJoCo เป็น benchmark ที่นิยมใช้วัด performance ของอัลกอริธึม Actor-Critic ที่ออกแบบมาสำหรับ action space แบบต่อเนื่อง โดยเฉพาะอย่างยิ่ง PPO และ SAC
      </p>
    </div>

    <h3 className="text-xl font-semibold">การเลือก Benchmark ให้เหมาะสมกับงานวิจัย</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>หากงานเน้น perception: ควรเลือก Atari หรือ DeepMind Lab</li>
      <li>หากเน้นการควบคุมทางกายภาพ: ควรเลือก MuJoCo หรือ Isaac Gym</li>
      <li>หากเน้น memory หรือ planning: เลือก MiniGrid, DMLab, หรือ GridWorld</li>
    </ul>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning". <i>Nature</i>.</li>
      <li>Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms". <i>arXiv:1707.06347</i>.</li>
      <li>Brockman, G. et al. (2016). "OpenAI Gym". <i>arXiv:1606.01540</i>.</li>
      <li>Chebotar, Y. et al. (2021). "Closing the Sim-to-Real Loop". <i>IEEE Robotics and Automation Letters</i>.</li>
    </ul>
  </div>
</section>


    <section id="advantages" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. จุดแข็งของ RL</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ความสามารถในการเรียนรู้ผ่านการทดลอง</h3>
    <p>
      Reinforcement Learning (RL) มีจุดแข็งสำคัญในการเรียนรู้จากการปฏิสัมพันธ์โดยตรงกับสภาพแวดล้อม โดยไม่ต้องมีคำตอบล่วงหน้า (label) ซึ่งต่างจาก Supervised Learning ที่ต้องอาศัยข้อมูลคำตอบที่ถูกต้อง RL สามารถสร้างกลยุทธ์ (policy) ผ่านกระบวนการสำรวจ (exploration) และใช้รางวัล (reward) เป็น feedback
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>สามารถปรับกลยุทธ์จากผลลัพธ์ที่เกิดขึ้นจริงในระบบ</li>
      <li>รองรับการเรียนรู้แบบ trial-and-error</li>
      <li>ปรับตัวได้แม้ข้อมูลใน environment เปลี่ยนแปลง</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        ความสามารถของ RL ในการเรียนรู้จากรางวัลโดยตรง ทำให้ระบบสามารถ "ค้นพบ" พฤติกรรมที่ไม่เคยมีมนุษย์ป้อนให้ได้ เช่น กลยุทธ์ใหม่ในการเล่น Go หรือการควบคุมหุ่นยนต์แบบพลิกแนวคิดเดิม
      </p>
    </div>

    <h3 className="text-xl font-semibold">การปรับตัวต่อสภาพแวดล้อมแบบ dynamic</h3>
    <p>
      RL มีความสามารถเด่นในการจัดการกับสภาพแวดล้อมที่ไม่แน่นอนหรือเปลี่ยนแปลงได้ เช่น ในระบบ real-time หรือ multi-agent systems:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ใช้กับระบบที่มี state space ขนาดใหญ่และเปลี่ยนแปลงตลอดเวลา</li>
      <li>เรียนรู้กลยุทธ์ใหม่โดยไม่ต้อง re-train ทั้งระบบ</li>
      <li>ใช้กับปัญหาที่ไม่มีโมเดลเชิงวิเคราะห์ได้โดยตรง เช่น การเจรจาอัตโนมัติ หรือ autonomous trading</li>
    </ul>

    <h3 className="text-xl font-semibold">ความสามารถในการ generalize และ scale</h3>
    <p>
      ด้วยการผสานกับ Deep Learning ทำให้ RL สามารถจัดการกับ state space ที่เป็น high-dimensional เช่น รูปภาพหรือเสียง และสามารถ scale ไปยังงานขนาดใหญ่ได้:
    </p>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        Deep RL เช่น AlphaStar และ OpenAI Five แสดงให้เห็นถึงศักยภาพของ RL ในการแข่งขันในสภาพแวดล้อมที่ซับซ้อนและหลากหลายระดับแบบ multi-agent ได้ในระดับที่เหนือกว่ามนุษย์
      </p>
    </div>

    <h3 className="text-xl font-semibold">การควบคุมที่มีการเพิ่มรางวัลแบบต่อเนื่อง (Long-Term Planning)</h3>
    <p>
      RL เหมาะสำหรับปัญหาที่ต้องใช้การวางแผนระยะยาว เช่น การบริหารจัดการทรัพยากรหรือวางกลยุทธ์:
    </p>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200">
      <code>{`G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}`}</code>
    </pre>
    <p>
      โดยที่ γ (discount factor) ใช้ควบคุมความสำคัญของรางวัลในอนาคต ทำให้ RL ไม่เพียงมองที่ผลลัพธ์ปัจจุบัน แต่พิจารณาผลลัพธ์ที่ตามมาด้วย
    </p>

    <h3 className="text-xl font-semibold">การประยุกต์ใช้งานจริงที่แสดงจุดแข็งของ RL</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">อุตสาหกรรม</th>
          <th className="border px-4 py-2">ตัวอย่างการใช้งาน</th>
          <th className="border px-4 py-2">จุดแข็งของ RL ที่ใช้</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Healthcare</td>
          <td className="border px-4 py-2">Personalized Treatment Planning</td>
          <td className="border px-4 py-2">การวางแผนแบบลำดับเวลา</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Autonomous Systems</td>
          <td className="border px-4 py-2">Self-driving Vehicles</td>
          <td className="border px-4 py-2">Real-time interaction</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Energy Management</td>
          <td className="border px-4 py-2">Smart Grid Optimization</td>
          <td className="border px-4 py-2">การควบคุมแบบ dynamic</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับการเรียนรู้แบบอื่น</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Supervised Learning:</strong> ต้องมี label เต็ม → ใช้ได้ดีใน classification แต่ไม่สามารถวางแผนล่วงหน้าได้</li>
      <li><strong>Unsupervised Learning:</strong> เหมาะกับการหาโครงสร้าง แต่ไม่มีแนวคิดเรื่อง reward หรือ policy</li>
      <li><strong>Reinforcement Learning:</strong> เหมาะกับงานที่มีเป้าหมายเป็นผลสะสม (cumulative objective)</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อสรุป</h3>
    <p>
      จุดแข็งของ RL สะท้อนให้เห็นถึงศักยภาพในการเรียนรู้ที่มีความยืดหยุ่น ปรับตัว และตอบสนองต่อสภาพแวดล้อมได้โดยไม่ต้องอาศัยคำตอบล่วงหน้า จึงเป็นหัวใจของระบบอัตโนมัติและ AI ยุคใหม่
    </p>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Sutton, R. S., & Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i>. MIT Press.</li>
      <li>Silver, D. et al. (2016). <i>Mastering the game of Go with deep neural networks and tree search</i>. Nature.</li>
      <li>Arulkumaran, K. et al. (2017). <i>Deep Reinforcement Learning: A Brief Survey</i>. IEEE Signal Processing Magazine.</li>
      <li>OpenAI (2019). <i>OpenAI Five: Mastering Dota 2</i>. arXiv:1912.06680</li>
    </ul>
  </div>
</section>


 <section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. ข้อจำกัดของ RL</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">บริบทของข้อจำกัดใน Reinforcement Learning</h3>
    <p>
      แม้ Reinforcement Learning (RL) จะมีศักยภาพสูงในการแก้ปัญหาแบบ sequential decision making แต่ก็ยังมีข้อจำกัดหลายด้าน ทั้งในแง่ของการใช้งานจริง ความซับซ้อนของการฝึกสอน และการประเมินผลที่ยังไม่สมบูรณ์ ข้อจำกัดเหล่านี้มักส่งผลต่อทั้งด้าน computational cost และความสามารถในการนำไปใช้ในระบบจริง (deployment gap)
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        การเข้าใจข้อจำกัดของ RL ช่วยให้สามารถออกแบบอัลกอริธึมและ environment ได้ดีขึ้น ลดการใช้ทรัพยากรเกินความจำเป็น และเพิ่มประสิทธิภาพในการฝึกและประยุกต์ใช้งานจริง
      </p>
    </div>

    <h3 className="text-xl font-semibold">1. Sample Inefficiency</h3>
    <p>
      RL ต้องการข้อมูลจำนวนมากในการเรียนรู้ โดยเฉพาะใน environment ที่มี stochasticity สูง หรือ state-action space ขนาดใหญ่ เช่น:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>การเรียนรู้ใน Atari game อาจต้องการ episode นับล้านครั้งเพื่อให้ agent เก่งพอ</li>
      <li>ในงาน robotics การทดลองซ้ำจริงมีต้นทุนสูง จึงไม่สามารถฝึกได้แบบ brute-force</li>
    </ul>

    <h3 className="text-xl font-semibold">2. Exploration-Exploitation Trade-off</h3>
    <p>
      การเลือกว่าจะ “ลองสิ่งใหม่” หรือ “ทำสิ่งเดิมที่ดีอยู่แล้ว” เป็นปัญหาที่ยังไม่มีวิธีแก้ที่สมบูรณ์ โดยเฉพาะใน environment ที่ sparse reward:
    </p>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        อัลกอริธึมที่มีการ explore ไม่เพียงพออาจติดอยู่ใน local optimum ขณะที่การ explore มากเกินไปอาจไม่ converge
      </p>
    </div>

    <h3 className="text-xl font-semibold">3. Delayed Reward & Credit Assignment</h3>
    <p>
      ปัญหาสำคัญอีกประการหนึ่งคือการระบุว่า action ใดที่ส่งผลต่อรางวัลในอนาคต:
    </p>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`R_t = f(a_{t-k}, ..., a_t)`}</code></pre>
    <p>
      ในสถานการณ์ที่ reward มาช้า เช่น ในเกมเชิงกลยุทธ์หรือการวางแผนทางธุรกิจ การระบุว่า action ไหนที่ควรได้ “เครดิต” เป็นเรื่องยาก และส่งผลให้การเรียนรู้ไม่เสถียร
    </p>

    <h3 className="text-xl font-semibold">4. ความเปราะบางต่อ Hyperparameters</h3>
    <p>
      RL มีความไวต่อการตั้งค่า hyperparameter เช่น learning rate, discount factor, exploration strategy ฯลฯ:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>ค่า gamma ที่สูงเกินไปจะให้ความสำคัญกับอนาคตมากเกิน ทำให้ไม่ converge</li>
      <li>ค่า epsilon ต่ำเกินไปทำให้ explore ไม่พอ</li>
      <li>ค่า learning rate ที่ไม่เหมาะสมอาจทำให้เกิด divergence</li>
    </ul>

    <h3 className="text-xl font-semibold">5. ความไม่เสถียรของการฝึกด้วย Neural Network</h3>
    <p>
      ใน Deep RL การใช้ function approximator อย่าง neural network มีความไม่เสถียร เนื่องจากข้อมูลมี correlation และ reward distribution ไม่ stationary:
    </p>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">สาเหตุ</th>
          <th className="border px-4 py-2">ผลกระทบ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Non-i.i.d. Data</td>
          <td className="border px-4 py-2">โมเดลเรียนรู้ bias จาก sequence เดิม</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Bootstrapping</td>
          <td className="border px-4 py-2">เกิด feedback loop ของ error</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Reward Clipping</td>
          <td className="border px-4 py-2">ลด sensitivity ของ signal</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">6. Deployment Gap</h3>
    <p>
      ปัญหาที่ฝึกใน simulation แล้วไม่สามารถนำไปใช้ใน real world ได้อย่างราบรื่น เช่นใน robotics หรือ healthcare ซึ่งอาจเกิดจาก:
    </p>
    <ul className="list-disc list-inside space-y-2">
      <li>Environment จริงแตกต่างจาก simulator</li>
      <li>ไม่มีการ model uncertainty อย่างเหมาะสม</li>
      <li>ไม่สามารถเชื่อถือได้ใน safety-critical systems</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Highlight:</strong>
      <p>
        แนวทางใหม่ เช่น Offline RL, Safe RL และ Sim2Real ถูกพัฒนาเพื่อแก้ปัญหาการใช้งานจริงของ RL โดยเฉพาะในระบบที่ต้องการความน่าเชื่อถือสูง
      </p>
    </div>

    <h3 className="text-xl font-semibold">7. Computational Cost</h3>
    <p>
      การฝึก RL บางรูปแบบโดยเฉพาะใน high-dimensional space ต้องใช้ทรัพยากรสูง เช่น GPU cluster, multi-agent rollout, และ long training time:
    </p>
    <pre className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-900 p-4 text-sm sm:text-base leading-relaxed text-gray-800 dark:text-gray-200"><code>{`# Dota 2 AI Training (OpenAI Five)
128,000 CPU cores + 256 GPUs
ต่อเนื่องนานหลายสัปดาห์`}</code></pre>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Henderson, P. et al. (2018). <i>Deep RL That Matters</i>. arXiv:1709.06560</li>
      <li>Achiam, J. et al. (2017). <i>Constrained Policy Optimization</i>. arXiv:1705.10528</li>
      <li>Kumar, A. et al. (2020). <i>Conservative Q-Learning for Offline RL</i>. arXiv:2006.04779</li>
      <li>OpenAI (2019). <i>OpenAI Five Technical Report</i>. arXiv:1912.06680</li>
    </ul>
  </div>
</section>


     <section id="real-world" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. ใช้งานในโลกจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ภาพรวมการใช้งาน RL นอกห้องทดลอง</h3>
    <p>
      การประยุกต์ใช้ Reinforcement Learning (RL) ในโลกจริงได้เริ่มแพร่หลายมากขึ้นในช่วงทศวรรษที่ผ่านมา โดยเฉพาะในระบบที่ต้องการการตัดสินใจแบบ sequential, adaptive, และแบบ real-time อย่างไรก็ตาม การนำ RL ออกจากห้องทดลองไปสู่ระบบ production-level ต้องผ่านการปรับจูนอย่างเข้มข้น ทั้งด้าน robust learning, cost efficiency และ safe deployment
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        การใช้งาน RL ในโลกจริงต้องอาศัยการออกแบบ environment, reward shaping และ policy constraints อย่างพิถีพิถัน มิใช่แค่ฝึก agent ด้วย model เท่านั้น
      </p>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างจากอุตสาหกรรมหลัก</h3>
    <div className="overflow-x-auto rounded-lg shadow-md">
  <table className="w-full min-w-[700px] table-auto border-collapse text-sm sm:text-base">
    <thead className="bg-gray-900 dark:bg-gray-800 text-white">
      <tr className="divide-x divide-gray-700">
        <th className="px-4 py-3 text-left font-semibold">อุตสาหกรรม</th>
        <th className="px-4 py-3 text-left font-semibold">การประยุกต์ใช้</th>
        <th className="px-4 py-3 text-left font-semibold">เทคโนโลยี RL ที่เกี่ยวข้อง</th>
      </tr>
    </thead>
    <tbody className="divide-y divide-gray-300 dark:divide-gray-700">
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">พลังงาน</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Demand response, smart grid optimization</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Deep Q-Networks, Safe RL</td>
      </tr>
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">หุ่นยนต์</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Robot locomotion, manipulation tasks</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Model-free RL, Proximal Policy Optimization</td>
      </tr>
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">บริการดิจิทัล</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Personalized recommendation, ad placement</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Off-policy RL, Contextual Bandits</td>
      </tr>
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">การเงิน</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Portfolio management, algorithmic trading</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Multi-Agent RL, Risk-Sensitive RL</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold">กรณีศึกษา: Waymo และการควบคุมรถอัตโนมัติ</h3>
    <p>
      ในระบบ self-driving cars อย่าง Waymo หรือ Tesla Autopilot ส่วนหนึ่งของ decision making pipeline อาศัย RL สำหรับการเรียนรู้ policy ที่ซับซ้อนภายใต้เงื่อนไขด้านความปลอดภัย เช่น การเปลี่ยนเลน, การเว้นระยะห่างจากรถคันหน้า โดย RL จะต้อง integrate กับ perception model และ safety module อย่างแนบแน่น
    </p>

    <h3 className="text-xl font-semibold">กรณีศึกษา: การวางแผนการรักษาเฉพาะบุคคล</h3>
    <p>
      ในงาน healthcare เช่น การวางแผนการให้ยา หรือการรักษาผู้ป่วยเรื้อรัง (เช่น diabetic control) RL ถูกนำมาใช้เพื่อเรียนรู้กลยุทธ์การรักษาที่ personalized โดยอิงจากข้อมูลผู้ป่วยจริง ทำให้การรักษาปรับตัวตามสภาวะของผู้ป่วยในแต่ละช่วงเวลา
    </p>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        Offline RL และ Batch-Constrained Q-Learning (BCQ) เป็นแนวทางที่นิยมใช้ใน healthcare เนื่องจากสามารถเรียนรู้จาก historical medical data โดยไม่ต้องทำ trial-and-error จริง
      </p>
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มการใช้งาน RL ในระบบจริง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li><strong>Sim2Real Transfer:</strong> ฝึกใน simulator แล้ว deploy สู่ระบบจริง เช่น robot arm</li>
      <li><strong>RL as API:</strong> นำ RL มาเป็น service layer ที่เรียนรู้จาก feedback ผู้ใช้แบบ real-time</li>
      <li><strong>Human-in-the-loop RL:</strong> มีมนุษย์ช่วยให้ feedback แก่ agent เช่น ในระบบการศึกษา</li>
    </ul>

    <h3 className="text-xl font-semibold">ความท้าทายที่ยังต้องแก้ไข</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>การควบคุมความเสี่ยงและความปลอดภัย (Safety Constraints)</li>
      <li>การทำให้ RL มี Explainability</li>
      <li>การจัดการกับ distribution shift และ non-stationarity</li>
      <li>การนำไปใช้กับ multi-objective และ multi-agent environments</li>
    </ul>

    <h3 className="text-xl font-semibold">แนวทางการวิจัยที่เกิดขึ้นเพื่อสนับสนุนการใช้งานจริง</h3>
    <table className="w-full table-auto border-collapse border border-gray-300 dark:border-gray-600 text-sm sm:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700 text-left">
        <tr>
          <th className="border px-4 py-2">แนวทาง</th>
          <th className="border px-4 py-2">เป้าหมาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Safe RL</td>
          <td className="border px-4 py-2">ป้องกันไม่ให้ agent เลือก action ที่เสี่ยง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Meta-RL</td>
          <td className="border px-4 py-2">เรียนรู้ให้เร็วขึ้นโดยใช้ประสบการณ์จากหลาย task</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Explainable RL</td>
          <td className="border px-4 py-2">อธิบาย policy ที่ได้ในรูปแบบเข้าใจง่าย</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Rajeswaran, A. et al. (2020). <i>Review of RL in Robotics</i>, arXiv:1708.05866</li>
      <li>Gottesman, O. et al. (2019). <i>Guidelines for RL in Healthcare</i>, Nature Medicine</li>
      <li>Chen, M. et al. (2021). <i>Decision Transformer: Reinforcement Learning as Sequence Modeling</i>, arXiv:2106.01345</li>
      <li>DeepMind (2023). <i>RL in Industry Report</i>, https://deepmind.com/blog</li>
    </ul>
  </div>
</section>


<section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">RL กับ Paradigm Shift ด้านการเรียนรู้ของ Machine</h3>
    <p>
      Reinforcement Learning (RL) ถือเป็นหนึ่งในสามเสาหลักของการเรียนรู้ของเครื่อง ร่วมกับ Supervised และ Unsupervised Learning แต่แตกต่างด้วยแนวคิดสำคัญคือ <strong>การเรียนรู้จากผลลัพธ์ของการกระทำ</strong> มากกว่าการเรียนรู้จากข้อมูลที่ป้าย label ไว้ล่วงหน้า การที่ RL สามารถเรียนรู้ผ่าน interaction แบบ trial-and-error ทำให้เหมาะสำหรับปัญหาที่มีลักษณะ sequential decision making ซึ่งเป็นพื้นฐานของระบบอัจฉริยะจำนวนมาก
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <strong className="block font-semibold mb-2">Insight:</strong>
      <p>
        RL คือการเรียนรู้แบบ "ผลักดันโดยการทดลอง" (exploration-driven learning) ซึ่งต่างจาก supervised learning ที่ยึดติดกับการเลียนแบบแบบฝึกหัดที่มีเฉลยล่วงหน้า
      </p>
    </div>

    <h3 className="text-xl font-semibold">ทำไม RL จึงเป็นก้าวต่อไปของ AI Systems?</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>RL ช่วยให้ระบบ AI ตัดสินใจในสภาวะแวดล้อมที่เปลี่ยนแปลงตลอดเวลา</li>
      <li>สามารถเรียนรู้จาก feedback ที่เกิดขึ้นในระยะยาว ไม่ใช่เฉพาะผลลัพธ์ทันที</li>
      <li>เหมาะกับงานที่มนุษย์ไม่สามารถให้ label หรือ supervision ได้โดยตรง เช่น การเล่นเกมหรือควบคุมหุ่นยนต์</li>
      <li>ใช้สำหรับ optimization problems ที่ไม่มี solution ตายตัว เช่น dynamic resource allocation</li>
    </ul>

    <h3 className="text-xl font-semibold">ความเชื่อมโยงระหว่าง RL และ Cognitive Science</h3>
    <p>
      หลายการศึกษาจาก Stanford และ MIT ชี้ให้เห็นว่า RL มีความสอดคล้องกับการทำงานของสมองมนุษย์ โดยเฉพาะระบบ dopamine reward ซึ่งมีบทบาทในการเรียนรู้พฤติกรรมที่ให้ผลตอบแทน งานของ Sutton & Barto (1998) ได้อธิบายว่าการเรียนรู้ของ agent ใน RL มีความใกล้เคียงกับการปรับตัวของสมองใน biological learning systems อย่างมีนัยสำคัญ
    </p>

    <h3 className="text-xl font-semibold">RL และ Emergent Intelligence</h3>
    <p>
      ผลงานจาก DeepMind และ OpenAI แสดงให้เห็นว่า RL สามารถทำให้ agent เกิดพฤติกรรมที่ซับซ้อนได้โดยไม่มีการโปรแกรมไว้ล่วงหน้า เช่น การเล่นเกม Dota2 หรือ StarCraft II ด้วยทักษะระดับแชมป์โลก หรือการสร้าง multi-agent behaviors ที่คล้ายกับ social dynamics ในธรรมชาติ
    </p>

    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        การประยุกต์ RL กับ environment ที่มีหลาย agent สามารถนำไปสู่ emergent strategy เช่น การแบ่งบทบาท, การช่วยเหลือซึ่งกันและกัน หรือแม้แต่การ “เจรจา” ระหว่าง agent ที่ไม่มีโค้ดส่วนสนทนาเลย
      </p>
    </div>

    <h3 className="text-xl font-semibold">แผนผังเปรียบเทียบ: RL vs Supervised vs Unsupervised</h3>
 <div className="overflow-x-auto rounded-lg shadow-md">
  <table className="min-w-[700px] w-full border-collapse text-sm sm:text-base">
    <thead className="bg-gray-900 dark:bg-gray-800 text-white">
      <tr className="divide-x divide-gray-700">
        <th className="px-4 py-3 text-left font-semibold">คุณสมบัติ</th>
        <th className="px-4 py-3 text-left font-semibold">Supervised Learning</th>
        <th className="px-4 py-3 text-left font-semibold">Unsupervised Learning</th>
        <th className="px-4 py-3 text-left font-semibold">Reinforcement Learning</th>
      </tr>
    </thead>
    <tbody className="divide-y divide-gray-700">
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">การป้อนข้อมูล</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">มี Label</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">ไม่มี Label</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">มี State และ Reward</td>
      </tr>
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">Output</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Predictive value</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Cluster/Pattern</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Policy (action function)</td>
      </tr>
      <tr className="even:bg-gray-100 dark:even:bg-gray-900 divide-x divide-gray-300 dark:divide-gray-700">
        <td className="px-4 py-2 text-gray-900 dark:text-gray-100">การประเมิน</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Loss function</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Intra-cluster similarity</td>
        <td className="px-4 py-2 text-gray-800 dark:text-gray-200">Reward function</td>
      </tr>
    </tbody>
  </table>
</div>




    <h3 className="text-xl font-semibold">แง่มุมทางจริยธรรมและความปลอดภัย</h3>
    <p>
      ขณะที่ RL มีพลังมหาศาลในการสร้างการเรียนรู้แบบอัตโนมัติ แต่งานของ Amodei et al. จาก OpenAI และ DeepMind ได้เตือนถึงความเสี่ยงที่เกิดจาก "Reward Hacking" และ "Specifying the wrong objective" ซึ่งสามารถทำให้ agent เลือกพฤติกรรมที่ไม่พึงประสงค์เพียงเพราะมัน maximize reward ได้
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-300/20 border-l-4 border-yellow-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        การออกแบบ reward function ที่ผิดพลาดสามารถนำไปสู่ agent ที่ “โกงระบบ” แทนที่จะเรียนรู้เป้าหมายที่แท้จริง เช่น การชนวัตถุเพื่อ reset environment แล้วได้ reward ซ้ำ
      </p>
    </div>

    <h3 className="text-xl font-semibold">สรุปบทเรียนจาก Insight Box</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>RL คือแนวทางการเรียนรู้ที่ทรงพลังที่สุดสำหรับปัญหา sequential decision</li>
      <li>นำไปสู่พฤติกรรมแบบ emergent intelligence ได้จริงเมื่อออกแบบ environment และ reward ดีพอ</li>
      <li>เชื่อมโยงกับ neuroscience, robotics และ real-world applications อย่างลึกซึ้ง</li>
      <li>มีความเสี่ยงหากออกแบบ goal และ reward ไม่เหมาะสม — ต้องใช้เทคนิค Safe RL ควบคู่</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Sutton, R. & Barto, A. (2018). <i>Reinforcement Learning: An Introduction</i>, MIT Press</li>
      <li>Amodei, D. et al. (2016). <i>Concrete Problems in AI Safety</i>, arXiv:1606.06565</li>
      <li>Silver, D. et al. (2016). <i>Mastering the game of Go with deep neural networks and tree search</i>, Nature</li>
      <li>Levine, S. et al. (2020). <i>Offline Reinforcement Learning</i>, arXiv:2005.01643</li>
      <li>Lake, B. et al. (2017). <i>Building Machines that Learn and Think like People</i>, Behavioral and Brain Sciences</li>
    </ul>

  </div>
</section>


    <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Academic References</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="text-base leading-relaxed space-y-10">

    <h3 className="text-xl font-semibold">ภาพรวมของแหล่งอ้างอิง</h3>
    <p>
      Reinforcement Learning (RL) ถือเป็นแขนงสำคัญในสาขาวิทยาการปัญญาประดิษฐ์ที่มีงานวิจัยเชิงลึกและครอบคลุมจากมหาวิทยาลัยชั้นนำของโลก เช่น Stanford, MIT, CMU และสถาบันวิจัยที่มีบทบาทสูงอย่าง DeepMind, OpenAI และ Berkeley AI Research (BAIR) การเรียนรู้ผ่าน interaction กับสภาพแวดล้อมได้กลายเป็นกรอบแนวคิดหลักสำหรับการออกแบบระบบอัจฉริยะรุ่นใหม่
    </p>

    <h3 className="text-xl font-semibold">ประเภทของเอกสารที่อ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>หนังสือเรียนมาตรฐานจาก MIT Press และ Morgan Kaufmann</li>
      <li>บทความวิชาการใน Nature, Science และ arXiv</li>
      <li>สไลด์คอร์สเรียนจาก Stanford CS234, MIT 6.S191 และ CMU 10-703</li>
      <li>รายงานจากหน่วยวิจัย AI เช่น DeepMind, OpenAI, Facebook AI</li>
      <li>ระบบ benchmark และ dataset ที่ใช้ในการประเมิน RL agent</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างแหล่งอ้างอิงสำคัญ</h3>
    <ul className="list-decimal list-inside space-y-4">
      <li>
        Sutton, R. S., & Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction (2nd Ed.)</i>, MIT Press.
        <p className="text-gray-600">หนังสือพื้นฐานที่สุดในสาย RL ใช้ในการเรียนการสอนระดับมหาวิทยาลัยชั้นนำ</p>
      </li>
      <li>
        Silver, D. et al. (2016). <i>Mastering the game of Go with deep neural networks and tree search</i>, Nature, 529(7587), 484–489.
        <p className="text-gray-600">แสดงให้เห็นการประยุกต์ RL ร่วมกับ deep learning ใน AlphaGo</p>
      </li>
      <li>
        Amodei, D. et al. (2016). <i>Concrete Problems in AI Safety</i>, arXiv:1606.06565.
        <p className="text-gray-600">ชี้ให้เห็นความเสี่ยงของ reward hacking และประเด็นด้านความปลอดภัย</p>
      </li>
      <li>
        Lillicrap, T. P. et al. (2015). <i>Continuous control with deep reinforcement learning</i>, arXiv:1509.02971.
        <p className="text-gray-600">แนะนำ DDPG ซึ่งเป็น RL algorithm สำหรับ action space แบบต่อเนื่อง</p>
      </li>
      <li>
        Schulman, J. et al. (2017). <i>Proximal Policy Optimization Algorithms</i>, arXiv:1707.06347.
        <p className="text-gray-600">PPO เป็นอัลกอริธึมที่มีความสมดุลระหว่างประสิทธิภาพและความเสถียร</p>
      </li>
      <li>
        Mnih, V. et al. (2015). <i>Human-level control through deep reinforcement learning</i>, Nature, 518(7540), 529–533.
        <p className="text-gray-600">บทความที่เริ่มต้นกระแส deep RL ด้วย Deep Q-Network (DQN)</p>
      </li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งสื่อการเรียนรู้ระดับมหาวิทยาลัย</h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-blue-100 dark:bg-blue-300/20 p-4 rounded-lg shadow">
        <h4 className="font-semibold text-lg"> Stanford CS234</h4>
        <p className="text-sm">คอร์ส RL ระดับกลาง-สูง โดย Prof. Emma Brunskill พร้อมสไลด์, video และ assignment</p>
        <a href="https://web.stanford.edu/class/cs234/" target="_blank" className="text-blue-600 underline text-sm">cs234 - Stanford</a>
      </div>
      <div className="bg-yellow-100 dark:bg-yellow-300/20 p-4 rounded-lg shadow">
        <h4 className="font-semibold text-lg"> MIT 6.S191</h4>
        <p className="text-sm">คอร์ส AI ที่สอนรวมทั้ง deep RL, policy gradients และการใช้ gym environment</p>
        <a href="http://introtodeeplearning.com/" target="_blank" className="text-blue-600 underline text-sm">Intro to Deep Learning</a>
      </div>
      <div className="bg-blue-100 dark:bg-blue-300/20 p-4 rounded-lg shadow">
        <h4 className="font-semibold text-lg"> CMU 10-703</h4>
        <p className="text-sm">คอร์ส Machine Learning Theory โดย Prof. Zico Kolter มีหัวข้อ RL เชิงคณิตศาสตร์</p>
        <a href="http://www.cs.cmu.edu/~10703/" target="_blank" className="text-blue-600 underline text-sm">CMU 10-703</a>
      </div>
      <div className="bg-yellow-100 dark:bg-yellow-300/20 p-4 rounded-lg shadow">
        <h4 className="font-semibold text-lg"> Berkeley CS285</h4>
        <p className="text-sm">คอร์ส Deep Reinforcement Learning ระดับสูงโดย Sergey Levine</p>
        <a href="https://rail.eecs.berkeley.edu/deeprlcourse/" target="_blank" className="text-blue-600 underline text-sm">Berkeley CS285</a>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Insight: ความสำคัญของการศึกษาแหล่งอ้างอิงที่ถูกต้อง</h3>
    <div className="bg-blue-100 dark:bg-blue-300/20 border-l-4 border-blue-400 p-4 rounded-lg shadow-inner text-sm sm:text-base">
      <p>
        การศึกษาแนวคิดจากแหล่งอ้างอิงที่มีมาตรฐาน ไม่เพียงช่วยให้เข้าใจทฤษฎีได้ลึกซึ้ง แต่ยังสามารถต่อยอดสู่การวิจัย การประยุกต์ใช้งานจริง และการแก้ปัญหาด้านความปลอดภัยหรือความเป็นธรรมในระบบ RL ได้อย่างมีประสิทธิภาพ
      </p>
    </div>

    <h3 className="text-xl font-semibold">แนวทางต่อยอดหลังจากอ่าน Section นี้</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>เลือกศึกษา 1 คอร์สจากรายการด้านบนและเรียนให้ครบทั้งภาคเรียน</li>
      <li>ทดลองใช้ DQN หรือ PPO ใน OpenAI Gym และวิเคราะห์ผลการเรียนรู้</li>
      <li>อ่าน Paper ด้าน Safe RL และลองออกแบบ environment พร้อม reward ที่ปราศจากการโกง</li>
      <li>นำ RL ไปใช้ในแอปพลิเคชันจริง เช่น optimization, game AI, robotics หรือ recommender systems</li>
    </ul>
  </div>
</section>


    


        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day61 theme={theme} />
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
        <ScrollSpy_Ai_Day61 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day61_IntroRL;
