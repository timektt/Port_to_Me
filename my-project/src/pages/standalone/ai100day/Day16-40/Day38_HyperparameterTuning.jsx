import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day38 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day38";
import MiniQuiz_Day38 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day38";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day38_HyperparameterTuning = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day38_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day38_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day38_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day38_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day38_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day38_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day38_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day38_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day38_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day38_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day38_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day38_12").format("auto").quality("auto").resize(scale().width(501));
  const img13 = cld.image("Day38_13").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 38: Hyperparameter Tuning Strategies</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

       <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Hyperparameter จึงสำคัญ?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ภาพรวมของ Hyperparameter ใน Deep Learning</h3>
    <p>
      ในระบบการเรียนรู้ของโมเดลเชิงลึก (deep learning systems) ความสามารถของโมเดลไม่ได้พึ่งพาเพียงแค่โครงสร้างของเครือข่ายประสาทเทียม แต่ขึ้นอยู่กับการตั้งค่าพารามิเตอร์ภายนอกที่เรียกว่า <strong>Hyperparameters</strong> ซึ่งไม่ได้เรียนรู้โดยตรงจากข้อมูล แต่ต้องกำหนดไว้ล่วงหน้าก่อนการฝึกโมเดล
    </p>
    <p>
      ตัวอย่างของ Hyperparameters ได้แก่ learning rate, batch size, จำนวนชั้นของโมเดล, จำนวนยูนิตต่อชั้น, dropout rate และค่าพารามิเตอร์เฉพาะใน optimization เช่น beta1/beta2 ใน Adam หรือ momentum ใน SGD
    </p>

    <h3>การเปรียบเทียบ Hyperparameter กับ Parameter</h3>
    <ul className="list-disc pl-6">
      <li><strong>Parameter:</strong> น้ำหนัก (weights) และ bias ที่เรียนรู้ได้จากข้อมูลผ่านการฝึกโมเดล</li>
      <li><strong>Hyperparameter:</strong> ค่าที่ควบคุมการฝึก เช่น learning rate และ architecture ซึ่งต้องกำหนดก่อนการฝึก</li>
    </ul>

    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">Parameter</th>
            <th className="border px-4 py-2">Hyperparameter</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">น้ำหนักของ layer</td>
            <td className="border px-4 py-2">Learning rate</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Bias term</td>
            <td className="border px-4 py-2">Batch size</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Embedding matrix</td>
            <td className="border px-4 py-2">Dropout rate</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ผลกระทบเชิงลึกของการจูน Hyperparameter</h3>
    <p>
      งานวิจัยของ Bergstra และ Bengio (2012) พบว่าโมเดล deep learning ที่มี hyperparameter เหมาะสมสามารถ outperform โมเดลที่ใช้สถาปัตยกรรมเดียวกันแต่จูนไม่ดีได้ถึงหลายเท่าตัว ทั้งในด้าน accuracy และการ generalize ไปยังข้อมูลใหม่ โดยเฉพาะในงานที่มีข้อมูลจำกัด
    </p>
    <p>
      Hyperparameter บางค่ามีผลเชิง exponential ต่อ performance เช่น learning rate ที่มากเกินไปอาจทำให้ loss ไม่ลดลง ขณะที่น้อยเกินไปอาจทำให้ convergence ช้าและติด local minima
    </p>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        การเลือก Hyperparameter ที่เหมาะสมมีผลต่อความสามารถของโมเดลมากกว่าการเพิ่ม layer หรือ parameter ในหลายกรณี การเปลี่ยน learning rate จาก 0.01 → 0.001 สามารถเพิ่ม test accuracy ได้มากกว่า 20% ในโมเดล CNN บน CIFAR-10 (Bengio et al., 2012)
      </p>
    </div>

    <h3>ความซับซ้อนของการจูนในระบบสมัยใหม่</h3>
    <ul className="list-disc pl-6">
      <li>จำนวน Hyperparameter มากขึ้นในโมเดลขนาดใหญ่ เช่น Transformer หรือ LLM</li>
      <li>Hyperparameter บางตัวมี dependency ระหว่างกัน เช่น learning rate กับ batch size</li>
      <li>ต้องการ compute สูงขึ้นในการ evaluate configuration แต่ละชุด</li>
    </ul>

    <h3>กรณีศึกษาเบื้องต้นจากอุตสาหกรรม</h3>
    <p>
      บริษัทอย่าง OpenAI, Google และ Meta ได้ลงทุนทรัพยากรจำนวนมากในระบบ hyperparameter tuning อัตโนมัติ เช่น Google Vizier, Optuna และ Ray Tune โดยให้ความสำคัญกับการใช้ Bayesian optimization, population-based training และ early stopping เพื่อให้ได้ประสิทธิภาพสูงสุดภายใต้ resource constraint
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        การศึกษาจาก Google Research (Golovin et al., 2017) ชี้ว่า การใช้ AutoML เพื่อตัดสินใจการตั้งค่า Hyperparameter สามารถลดเวลาในการฝึกโมเดลได้มากกว่า 40% พร้อมเพิ่มความแม่นยำในการ deploy บน production
      </p>
    </div>

    <h3>สรุป</h3>
    <p>
      Hyperparameter ถือเป็นหนึ่งในปัจจัยที่กำหนดเพดานของ performance สำหรับโมเดล AI ในยุคใหม่ และการจูนที่เหมาะสมสามารถเป็นความแตกต่างระหว่างโมเดลระดับวิจัยกับโมเดลระดับ production การเข้าใจบทบาทของ Hyperparameter อย่างลึกซึ้งจึงเป็นพื้นฐานสำคัญก่อนเข้าสู่การเรียนรู้กลยุทธ์การจูนที่หลากหลายในส่วนถัดไป
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR.</li>
      <li>Golovin, D. et al. (2017). Google Vizier: A Service for Black-Box Optimization. KDD.</li>
      <li>Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization. NIPS.</li>
      <li>Hutter, F., Kotthoff, L., & Vanschoren, J. (2019). AutoML: Methods, Systems, Challenges. Springer.</li>
      <li>Stanford CS229 Lecture Notes: Hyperparameter Tuning</li>
    </ul>
  </div>
</section>


        <section id="examples" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ตัวอย่าง Hyperparameters ที่มักต้องจูน</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>การจำแนกประเภทของ Hyperparameters</h3>
    <p>
      Hyperparameters สามารถแบ่งออกเป็นหลายกลุ่มตามบทบาทในกระบวนการเรียนรู้ เช่น optimization, regularization, architectural, และ training dynamics โดยการทำความเข้าใจความสัมพันธ์ระหว่างกลุ่มเหล่านี้มีความสำคัญต่อการกำหนดกลยุทธ์การจูนที่มีประสิทธิภาพ
    </p>

    <h3>กลุ่มที่ 1: Optimization Hyperparameters</h3>
    <ul className="list-disc pl-6">
      <li><strong>Learning Rate (lr):</strong> ควบคุมขนาดของก้าวในการปรับ parameter แต่ละรอบ</li>
      <li><strong>Momentum:</strong> ช่วยเพิ่มความเร็วในการฝึก โดยใช้ประวัติ gradient</li>
      <li><strong>Weight Decay:</strong> ทำหน้าที่ regularization โดยลดขนาดของ weight</li>
      <li><strong>Beta1 / Beta2:</strong> ใช้ใน Adam optimizer สำหรับการควบคุม moment estimate</li>
    </ul>

    <h3>กลุ่มที่ 2: Regularization Hyperparameters</h3>
    <ul className="list-disc pl-6">
      <li><strong>Dropout Rate:</strong> ป้องกัน overfitting โดยสุ่มปิด neuron บางตัวขณะฝึก</li>
      <li><strong>Label Smoothing:</strong> ลดความมั่นใจเกินไปของโมเดล</li>
      <li><strong>Early Stopping Patience:</strong> ควบคุมการหยุดฝึกหาก performance ไม่พัฒนา</li>
    </ul>

    <h3>กลุ่มที่ 3: Structural Hyperparameters</h3>
    <ul className="list-disc pl-6">
      <li><strong>Number of Layers:</strong> ความลึกของเครือข่ายมีผลต่อ capacity และ gradient flow</li>
      <li><strong>Number of Units per Layer:</strong> ควบคุม dimensionality ของ hidden representation</li>
      <li><strong>Kernel Size / Filter Count:</strong> เฉพาะใน CNN สำหรับควบคุม receptive field</li>
    </ul>

    <h3>กลุ่มที่ 4: Training Hyperparameters</h3>
    <ul className="list-disc pl-6">
      <li><strong>Batch Size:</strong> ขนาดข้อมูลที่ใช้ต่อรอบการปรับพารามิเตอร์</li>
      <li><strong>Epochs:</strong> จำนวนรอบในการผ่านข้อมูลทั้งหมด</li>
      <li><strong>Shuffle:</strong> การสับลำดับข้อมูลแต่ละ epoch</li>
    </ul>

    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">Hyperparameter</th>
            <th className="border px-4 py-2">กลุ่ม</th>
            <th className="border px-4 py-2">ผลต่อโมเดล</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Learning Rate</td>
            <td className="border px-4 py-2">Optimization</td>
            <td className="border px-4 py-2">กำหนดความเร็วในการเรียนรู้</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Dropout Rate</td>
            <td className="border px-4 py-2">Regularization</td>
            <td className="border px-4 py-2">ลด overfitting</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Number of Layers</td>
            <td className="border px-4 py-2">Structural</td>
            <td className="border px-4 py-2">กำหนดความลึกของโมเดล</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Batch Size</td>
            <td className="border px-4 py-2">Training</td>
            <td className="border px-4 py-2">มีผลต่อ stability และ memory</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยของ Smith et al. (2018) จาก Johns Hopkins พบว่า learning rate กับ batch size มี dependency แบบ non-linear ซึ่งต้องใช้ adaptive strategy ร่วมกัน เช่น Cyclical Learning Rate หรือ Linear Scaling Rule เพื่อให้ได้ประสิทธิภาพที่ดีในการฝึกโมเดล deep learning ขนาดใหญ่
      </p>
    </div>

    <h3>ความท้าทายในการเลือก Hyperparameter</h3>
    <ul className="list-disc pl-6">
      <li>บางค่ามี interaction ระหว่างกัน เช่น lr กับ optimizer, dropout กับ regularization</li>
      <li>บางค่ามี sensitivity สูง เช่น learning rate</li>
      <li>ค่าบางตัวไม่สามารถจูนได้เชิง linear เช่น batch size มีผลกระทบต่อ distribution ของ gradient</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        จากรายงานของ Stanford CS231n การเลือก Hyperparameter ที่ดีไม่จำเป็นต้องครอบคลุมทุกค่า แต่ควรโฟกัสเฉพาะกลุ่มที่มีผลเชิงนัยสำคัญ เช่น learning rate, regularization strength และ architectural depth ซึ่งมักเป็นค่า key influencer ในโมเดลขนาดใหญ่
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Smith, L. N. et al. (2018). A disciplined approach to neural network hyper-parameters. arXiv:1803.09820</li>
      <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press</li>
      <li>Stanford CS231n Lecture Notes - Hyperparameter Tuning</li>
      <li>Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures. arXiv:1206.5533</li>
    </ul>
  </div>
</section>


<section id="manual" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Manual Tuning (ยุคแรก)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>แนวคิดเบื้องต้นของการปรับค่าด้วยตนเอง</h3>
    <p>
      ก่อนที่จะมีการพัฒนาเทคนิคเชิงอัตโนมัติ การปรับค่า hyperparameters มักอาศัยความรู้ ความชำนาญ และประสบการณ์ของนักวิจัย โดยกระบวนการนี้เรียกว่า <strong>Manual Tuning</strong> ซึ่งอาจรวมถึงการทดลองค่าต่าง ๆ ทีละค่าผ่านกระบวนการ trial-and-error เพื่อหาค่าที่ได้ผลลัพธ์ดีที่สุด
    </p>

    <h3>ขั้นตอนทั่วไปของ Manual Tuning</h3>
    <ul className="list-disc pl-6">
      <li>กำหนดช่วงค่าที่ต้องการทดสอบ</li>
      <li>ประเมินผลลัพธ์บน validation set</li>
      <li>วิเคราะห์ผลการทดลองแต่ละครั้ง</li>
      <li>ปรับค่าใหม่โดยใช้ heuristic หรือ intuition</li>
    </ul>

    <h3>ข้อดีของการจูนแบบแมนนวล</h3>
    <ul className="list-disc pl-6">
      <li>ช่วยให้เข้าใจพฤติกรรมของโมเดลอย่างลึกซึ้ง</li>
      <li>ไม่จำเป็นต้องใช้ทรัพยากรคอมพิวเตอร์สูงในบางกรณี</li>
      <li>เหมาะกับโมเดลขนาดเล็กหรืองานที่ไม่ซับซ้อน</li>
    </ul>

    <h3>ข้อจำกัดของ Manual Tuning</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">ข้อจำกัด</th>
          <th className="border px-4 py-2">รายละเอียด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ไม่เป็นระบบ</td>
          <td className="border px-4 py-2">พึ่งพาประสบการณ์ของผู้วิจัยโดยไม่มีหลักฐานเชิงระบบ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ใช้เวลานาน</td>
          <td className="border px-4 py-2">ต้องทดลองค่าหลายชุดทีละตัวด้วยมือ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ไม่สามารถปรับได้ในกรณีที่ parameter space มีมิติสูง</td>
          <td className="border px-4 py-2">ความซับซ้อนเพิ่มขึ้นแบบทวีคูณตามจำนวนพารามิเตอร์</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยจาก Stanford (Bergstra & Bengio, 2012) ชี้ให้เห็นว่าการจูนแบบสุ่ม (Random Search) ยังมีประสิทธิภาพมากกว่า Manual Tuning เสียอีกในหลายกรณี โดยเฉพาะเมื่อมีพารามิเตอร์ที่ไม่สำคัญหลายตัวกระจายอยู่ใน space
      </p>
    </div>

    <h3>กรณีศึกษาทางประวัติศาสตร์</h3>
    <p>
      ในยุคต้นของ Deep Learning เช่น LeNet-5 (1998) หรือแม้กระทั่ง AlexNet (2012) การเลือกค่าเช่น learning rate, batch size และ momentum ถูกเลือกโดยนักวิจัยผ่านการลองผิดลองถูก ซึ่งสะท้อนให้เห็นข้อจำกัดของแนวทางแบบ manual นี้ได้อย่างชัดเจน
    </p>

    <h3>การนำไปใช้ในระบบปัจจุบัน</h3>
    <ul className="list-disc pl-6">
      <li>เหมาะกับโมเดลที่สามารถรันได้รวดเร็ว เช่น Logistic Regression, Decision Tree</li>
      <li>ใช้สำหรับปรับค่าตั้งต้นเบื้องต้นก่อนเข้าสู่กระบวนการ tuning แบบอัตโนมัติ</li>
      <li>อาจใช้ร่วมกับ Domain Expertise เพื่อสร้าง prior ที่ดีสำหรับ Bayesian Optimization</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        การทำ Manual Tuning มีบทบาทสำคัญในช่วงต้นของการพัฒนา AI แต่ในระบบที่มีความซับซ้อนสูงและต้องการ scaling ในระดับใหญ่ การใช้วิธีเชิงอัตโนมัติจะให้ผลลัพธ์ที่มีเสถียรภาพและประสิทธิภาพมากกว่าในระยะยาว
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research.</li>
      <li>LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE.</li>
      <li>Krizhevsky, A. et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.</li>
      <li>Stanford CS229 Lecture Notes: Optimization and Hyperparameter Tuning</li>
    </ul>
  </div>
</section>


      <section id="grid-search" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Grid Search</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>แนวคิดพื้นฐานของ Grid Search</h3>
    <p>
      Grid Search คือวิธีการจูนค่าพารามิเตอร์แบบครบทุกค่าที่เป็นไปได้ในกรอบที่กำหนด โดยจะสร้างตารางของค่าทุก combination ที่สามารถเกิดขึ้นจากชุดของ hyperparameters ที่ต้องการสำรวจ วิธีนี้เป็นหนึ่งในแนวทางที่เก่าแก่และตรงไปตรงมามากที่สุดในงานวิจัยด้าน Machine Learning และ Deep Learning โดยเฉพาะในการเลือกค่าที่เหมาะสมของพารามิเตอร์ เช่น learning rate, number of units, kernel size เป็นต้น
    </p>

    <h3>ข้อดีของ Grid Search</h3>
    <ul className="list-disc pl-6">
      <li>รับประกันว่าทุก combination ที่อยู่ในกรอบการสำรวจจะถูกทดสอบ</li>
      <li>เหมาะสำหรับชุดพารามิเตอร์จำนวนน้อย</li>
      <li>ง่ายต่อการวิเคราะห์ผลเชิงสถิติ</li>
    </ul>

    <h3>ข้อจำกัดที่สำคัญ</h3>
    <ul className="list-disc pl-6">
      <li>ต้องใช้เวลาและทรัพยากรมาก เนื่องจากจำนวน combination เติบโตแบบกำลังสองหรือมากกว่า</li>
      <li>ไม่มีการให้ความสำคัญกับพื้นที่ที่มีแนวโน้มว่าจะได้ผลลัพธ์ที่ดี</li>
      <li>ไม่สามารถขยายสเกลได้ดีในกรณีที่มี hyperparameters จำนวนมาก (curse of dimensionality)</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานของ Bergstra & Bengio (2012) ระบุว่า Grid Search มักมีประสิทธิภาพต่ำเมื่อเปรียบเทียบกับ Random Search ในกรณีที่มี hyperparameters ที่ไม่มีความสำคัญเท่ากัน เนื่องจาก Grid Search แบ่งพื้นที่ค้นหาแบบเท่ากัน โดยไม่สนใจว่า parameter ไหนมีอิทธิพลมากกว่ากัน
      </p>
    </div>

    <h3>ตัวอย่างการใช้งาน Grid Search ใน Scikit-Learn</h3>
    <pre><code className="language-python">from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

<pre className="bg-gray-900 text-white text-sm rounded-lg p-4 overflow-x-auto my-6">
  <code>
    {`param_grid = {
  'n_estimators': [100, 200, 300],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10]
}`}
  </code>
</pre>


model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)</code></pre>

    <h3>การจัดการ Computational Resource ในการใช้ Grid Search</h3>
    <ul className="list-disc pl-6">
      <li>การใช้ Parallelization เช่นผ่าน joblib หรือ Dask</li>
      <li>ใช้ Cloud-based AutoML ที่รองรับ distributed grid search</li>
      <li>จำกัดขอบเขตของ parameter space เช่น range หรือ step ที่เหมาะสม</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ในการแข่งขัน Kaggle หลายรายการพบว่า Grid Search สามารถสร้าง baseline model ที่ดีพอสมควรได้อย่างรวดเร็ว โดยเฉพาะเมื่อมี resource จำกัดและยังไม่มีโมเดล deep learning ที่ซับซ้อนมากนัก
      </p>
    </div>

    <h3>เปรียบเทียบกับเทคนิคอื่น</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">วิธีการ</th>
          <th className="border px-4 py-2">การสำรวจ Parameter Space</th>
          <th className="border px-4 py-2">จุดเด่น</th>
          <th className="border px-4 py-2">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Grid Search</td>
          <td className="border px-4 py-2">แบบสุ่มอย่างครอบคลุม</td>
          <td className="border px-4 py-2">ครบทุกค่า, วิเคราะห์ง่าย</td>
          <td className="border px-4 py-2">ช้าเมื่อ parameter เยอะ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Random Search</td>
          <td className="border px-4 py-2">สุ่มแบบมีการควบคุม</td>
          <td className="border px-4 py-2">เร็วขึ้น, scale ได้ดี</td>
          <td className="border px-4 py-2">อาจพลาดค่าที่ดีที่สุด</td>
        </tr>
      </tbody>
    </table>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research.</li>
      <li>Stanford CS229: Lecture Notes on Hyperparameter Optimization</li>
      <li>Scikit-learn Documentation: GridSearchCV</li>
      <li>Oxford Machine Learning Group. (2021). ML Optimization Techniques</li>
    </ul>
  </div>
</section>

     <section id="random-search" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Random Search</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>แนวคิดพื้นฐานของ Random Search</h3>
    <p>
      Random Search เป็นเทคนิคการค้นหาค่าที่เหมาะสมของ hyperparameters โดยอาศัยการสุ่มค่าจาก space ของ parameter แต่ละตัวแทนการทดสอบแบบครอบคลุมทุกจุดอย่าง Grid Search เทคนิคนี้ถูกนำเสนอโดย Bergstra และ Bengio (2012) ซึ่งพบว่าการสุ่มแบบมีแบบแผนสามารถค้นหาค่าที่มีประสิทธิภาพได้ดีกว่า grid search โดยเฉพาะเมื่อพารามิเตอร์บางตัวมีอิทธิพลต่อประสิทธิภาพมากกว่าตัวอื่น
    </p>

    <h3>ข้อดีเมื่อเทียบกับ Grid Search</h3>
    <ul className="list-disc pl-6">
      <li>ประหยัดเวลา: ไม่จำเป็นต้องประเมินทุกจุดในพารามิเตอร์ space</li>
      <li>เหมาะกับ high-dimensional space ที่ grid search ไม่สามารถทำได้อย่างครอบคลุม</li>
      <li>เปิดโอกาสให้ค้นหาค่าที่ดีในช่วงที่ไม่ได้อยู่ใน grid fix</li>
    </ul>

    <h3>การใช้งานจริง</h3>
    <p>
      ในการฝึก deep learning model เช่น CNN หรือ LSTM บน dataset ขนาดใหญ่ เช่น ImageNet หรือ NLP corpus อย่าง GLUE benchmark, Random Search ช่วยให้สามารถระบุค่า learning rate, batch size, หรือ dropout ได้อย่างมีประสิทธิภาพมากกว่า grid search
    </p>

    <h3>ตัวอย่างโค้ด</h3>
    <pre><code className="language-python">from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

<pre className="bg-gray-900 text-white text-sm rounded-lg p-4 overflow-x-auto my-6">
  <code>
    {`param_dist = {
  'n_estimators': [50, 100, 200],
  'max_depth': [10, 20, 30, None],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}`}
  </code>
</pre>

clf = RandomForestClassifier()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=3)
random_search.fit(X_train, y_train)</code></pre>

    <h3>ข้อควรระวัง</h3>
    <ul className="list-disc pl-6">
      <li>สุ่มไม่สม่ำเสมอใน space ที่มีขนาดใหญ่มาก หากไม่มีการ normalize</li>
      <li>อาจพลาดจุดที่ดีที่สุดได้หาก n_iter ต่ำ</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานของ Bergstra และ Bengio (2012) แสดงให้เห็นว่าใน hyperparameter space ที่มีขนาดใหญ่ Random Search สามารถให้ performance ที่ดีกว่า Grid Search ได้โดยใช้จำนวน iteration น้อยกว่าหลายเท่า
      </p>
    </div>

    <h3>การเปรียบเทียบกับ Grid Search</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Grid Search</th>
          <th className="border px-4 py-2">Random Search</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">จำนวนจุดที่ประเมิน</td>
          <td className="border px-4 py-2">ครอบคลุมทั้งหมด</td>
          <td className="border px-4 py-2">สุ่มบางจุด</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ประหยัดเวลา</td>
          <td className="border px-4 py-2">น้อย</td>
          <td className="border px-4 py-2">มาก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">เหมาะกับ space ใหญ่</td>
          <td className="border px-4 py-2">ไม่เหมาะ</td>
          <td className="border px-4 py-2">เหมาะมาก</td>
        </tr>
      </tbody>
    </table>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research.</li>
      <li>Stanford CS231n Lecture Notes on Hyperparameter Tuning</li>
      <li>Scikit-learn documentation on RandomizedSearchCV</li>
    </ul>
  </div>
</section>

  <section id="bayesian" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Bayesian Optimization</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>หลักการพื้นฐานของ Bayesian Optimization</h3>
    <p>
      Bayesian Optimization เป็นกระบวนการหาค่าที่เหมาะสมที่สุดของ hyperparameters โดยใช้แบบจำลองเชิงความน่าจะเป็น (probabilistic model) เพื่อคาดการณ์ค่าผลลัพธ์ของฟังก์ชันวัตถุประสงค์ (objective function) จากค่าของ hyperparameters โดยไม่ต้องประเมินฟังก์ชันจริงทั้งหมด วิธีนี้มีประโยชน์มากเมื่อต้นทุนการฝึกโมเดลสูง และการทดลองแต่ละครั้งมีค่าใช้จ่ายมาก
    </p>

    <h3>องค์ประกอบหลักของ Bayesian Optimization</h3>
    <ul className="list-disc pl-6">
      <li>Surrogate Model: ใช้สำหรับประมาณค่าฟังก์ชัน เช่น Gaussian Process (GP)</li>
      <li>Acquisition Function: ใช้ตัดสินใจว่าจะลองค่าต่อไปที่ใด เช่น Expected Improvement (EI), Upper Confidence Bound (UCB)</li>
      <li>Exploration vs Exploitation: ปรับสมดุลระหว่างการค้นหาค่าที่ไม่เคยลองกับค่าที่น่าจะดีที่สุด</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ในงานของ Brochu et al. (2010) และ Snoek et al. (2012) Bayesian Optimization แสดงให้เห็นถึงประสิทธิภาพในการหาค่าที่ดีที่สุดของโมเดล deep learning ด้วยการลดจำนวนรอบการฝึกลงได้มากกว่า 50% เมื่อเทียบกับ Grid Search หรือ Random Search
      </p>
    </div>

    <h3>การเปรียบเทียบกับเทคนิคอื่น</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">เทคนิค</th>
          <th className="border px-4 py-2">จุดเด่น</th>
          <th className="border px-4 py-2">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Grid Search</td>
          <td className="border px-4 py-2">ครอบคลุมทุกความเป็นไปได้</td>
          <td className="border px-4 py-2">สิ้นเปลือง compute มาก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Random Search</td>
          <td className="border px-4 py-2">รวดเร็วกว่า และมีโอกาสเจอค่าดี</td>
          <td className="border px-4 py-2">ไม่ใช้ประโยชน์จากข้อมูลเดิม</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Bayesian Optimization</td>
          <td className="border px-4 py-2">ใช้ประวัติข้อมูลเพื่อเลือกค่าต่อไป</td>
          <td className="border px-4 py-2">ซับซ้อนกว่าและต้องเลือก surrogate</td>
        </tr>
      </tbody>
    </table>

    <h3>ตัวอย่างการใช้งานในงานวิจัย</h3>
    <ul className="list-disc pl-6">
      <li>Hyperparameter tuning ใน Deep Belief Networks โดยใช้ Gaussian Process (Snoek et al., 2012)</li>
      <li>การฝึกโมเดล CNN ด้วย Hyperopt และ Tree-structured Parzen Estimator (Bergstra et al., 2013)</li>
      <li>AutoML frameworks เช่น Auto-sklearn และ Google Vizier ใช้ Bayesian Optimization เป็นแกนหลัก</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        การใช้ Bayesian Optimization ทำให้การทดลองในปริมาณจำกัดสามารถให้ผลลัพธ์ที่มีคุณภาพสูงได้เทียบเท่าหรือดีกว่าการทดลองแบบ brute-force โดยเฉพาะในบริบทที่ต้นทุน compute มีความสำคัญ เช่น การฝึกโมเดลบน GPU cluster ขนาดใหญ่
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Brochu, E., Cora, V. M., & de Freitas, N. (2010). A tutorial on Bayesian optimization of expensive cost functions. arXiv:1012.2599</li>
      <li>Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS.</li>
      <li>Bergstra, J., et al. (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. JMLR.</li>
      <li>Feurer, M., et al. (2015). Efficient and Robust Automated Machine Learning. NeurIPS.</li>
    </ul>
  </div>
</section>


    <section id="halving" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Successive Halving & Hyperband</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>แนวคิดพื้นฐานของ Successive Halving</h3>
    <p>
      Successive Halving (SH) เป็นอัลกอริทึมที่ได้รับแรงบันดาลใจจากแนวคิดของการแบ่งทรัพยากรแบบขั้นตอน (iterative resource allocation) โดยเริ่มจากการสุ่มชุด hyperparameters จำนวนมาก แล้วค่อยๆ ลดจำนวนลงตาม performance ที่วัดได้ในรอบแรก ๆ โดยใช้ทรัพยากรน้อยก่อน เช่น จำนวน epoch น้อย แล้วค่อยเพิ่มในการรอบถัดไปสำหรับผู้ผ่านเข้ารอบ
    </p>

    <h3>ขั้นตอนของ Successive Halving</h3>
    <ul className="list-disc pl-6">
      <li>สุ่มเลือก hyperparameters จำนวน <code>n</code> ชุด</li>
      <li>ฝึกแต่ละชุดด้วยทรัพยากรเริ่มต้น เช่น 1 epoch</li>
      <li>คัดเลือกเพียง top-<code>1/η</code> ที่ดีที่สุดเพื่อเข้าสู่รอบถัดไป</li>
      <li>เพิ่มทรัพยากร เช่น 3, 9, 27 epoch ตามระดับขั้น</li>
      <li>วนจนกว่าจะเหลือชุดเดียวที่ดีที่สุด</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        Successive Halving ช่วยลดเวลาและงบประมาณในการค้นหา hyperparameter ได้อย่างมีประสิทธิภาพ เมื่อเทียบกับ Grid Search และ Random Search โดยเฉพาะเมื่อโมเดลมีเวลาฝึกยาวนาน
      </p>
    </div>

    <h3>Hyperband: การต่อยอดจาก SH</h3>
    <p>
      Hyperband คืออัลกอริทึมที่นำ Successive Halving มาผสมกับแนวคิดของ multi-armed bandit เพื่อให้กระจายงบประมาณการฝึกได้อย่างยืดหยุ่น โดยสร้างหลายรอบการคัดเลือกที่มี <strong>budget</strong> ต่างกัน เพื่อให้สามารถทดลองได้ทั้งชุดพารามิเตอร์จำนวนมากที่ฝึกน้อย และชุดที่ฝึกนานแต่จำนวนน้อย
    </p>

    <h3>โครงสร้างของ Hyperband</h3>
    <ul className="list-disc pl-6">
      <li>กำหนดงบรวม <code>B</code> และ factor <code>η</code></li>
      <li>แบ่ง budget เป็นหลาย brackets โดยใช้ Successive Halving แต่ละชุด</li>
      <li>เปรียบเทียบผลแต่ละ bracket เพื่อหาค่าที่ดีที่สุด</li>
    </ul>

    <h3>ตารางเปรียบเทียบ SH กับ Hyperband</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">หัวข้อ</th>
            <th className="border px-4 py-2">Successive Halving</th>
            <th className="border px-4 py-2">Hyperband</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">แนวทาง</td>
            <td className="border px-4 py-2">ลดชุดทดลองทีละรอบ</td>
            <td className="border px-4 py-2">รวมหลาย SH ด้วย budget ต่างกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ข้อดี</td>
            <td className="border px-4 py-2">ประหยัด compute</td>
            <td className="border px-4 py-2">สำรวจได้กว้างและลึกพร้อมกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การใช้งานจริง</td>
            <td className="border px-4 py-2">AutoML, Deep Learning</td>
            <td className="border px-4 py-2">Ray Tune, Optuna</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานของ Li et al. จาก CMU (2017) ได้แสดงให้เห็นว่า Hyperband มีประสิทธิภาพมากกว่า Random Search และ SH อย่างมีนัยสำคัญในหลาย benchmark โดยสามารถหาค่า hyperparameter ที่ดีกว่าได้ในเวลาน้อยกว่า 1/3 ของ Grid Search
      </p>
    </div>

    <h3>เครื่องมือที่รองรับ Hyperband</h3>
    <ul className="list-disc pl-6">
      <li><code>Ray Tune</code>: มี API ที่ใช้งานง่ายและรองรับ Hyperband/ASHA</li>
      <li><code>Optuna</code>: รองรับ pruning ที่คล้ายกับ Hyperband</li>
      <li><code>scikit-optimize</code>: มี interface สำหรับ Successive Halving</li>
    </ul>

    <h3>อ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. arXiv:1603.06560</li>
      <li>Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research</li>
      <li>Stanford CS329: Topics in ML - Lecture on Hyperparameter Optimization (2023)</li>
    </ul>
  </div>
</section>


       <section id="pbt" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Population-Based Training (PBT)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>แนวคิดพื้นฐานของ Population-Based Training</h3>
    <p>
      Population-Based Training (PBT) เป็นเทคนิค optimization ที่พัฒนาโดย DeepMind โดยผสมผสานแนวคิดจาก evolutionary algorithms และ online model adaptation เข้าด้วยกัน แนวทางนี้มุ่งหมายเพื่อหลีกเลี่ยงข้อจำกัดของ static hyperparameters โดยใช้ population ของ model ที่เรียนรู้พร้อมกันและปรับตัวได้แบบ dynamic ตลอดกระบวนการฝึก
    </p>

    <h3>องค์ประกอบสำคัญของ PBT</h3>
    <ul className="list-disc pl-6">
      <li><strong>Population:</strong> โมเดลจำนวนหลายตัวที่ถูกฝึกพร้อมกัน โดยมี hyperparameters ที่หลากหลาย</li>
      <li><strong>Exploit:</strong> คัดลอกพารามิเตอร์จากสมาชิกที่มี performance สูงกว่าไปยังสมาชิกที่อ่อนกว่า</li>
      <li><strong>Explore:</strong> ทำการสุ่มเปลี่ยนค่า hyperparameters หลังจาก copy พารามิเตอร์</li>
      <li><strong>Asynchronous Evolution:</strong> ไม่มีขั้นตอน sync ทั่วทั้ง population ส่งผลให้สามารถขยายขนาดได้ดีในระบบกระจาย</li>
    </ul>

    <h3>เปรียบเทียบ PBT กับวิธีการจูนทั่วไป</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">คุณสมบัติ</th>
            <th className="border px-4 py-2">Grid/Random/Bayesian</th>
            <th className="border px-4 py-2">PBT</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Fixed Schedule</td>
            <td className="border px-4 py-2">ใช่</td>
            <td className="border px-4 py-2">ไม่ใช่ (adaptive)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ต้องเรียนทุก config จนจบ</td>
            <td className="border px-4 py-2">ใช่</td>
            <td className="border px-4 py-2">ไม่ใช่</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Online adaptation</td>
            <td className="border px-4 py-2">ไม่มี</td>
            <td className="border px-4 py-2">มี</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานของ DeepMind (Jaderberg et al., 2017) แสดงให้เห็นว่า PBT สามารถเอาชนะ hyperparameter tuning แบบ static ในหลาย task ได้อย่างมีนัยสำคัญ โดยเฉพาะในงาน reinforcement learning และ language modeling
      </p>
    </div>

    <h3>ตัวอย่างการประยุกต์ใช้งานจริง</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ฝึก AlphaStar ของ DeepMind สำหรับเกม StarCraft II</li>
      <li>นำมาใช้ใน AutoML และงานที่ต้องการ adaptation ต่อสภาพแวดล้อม</li>
      <li>ในงาน NLP เช่นการฝึก Transformer สำหรับ pretraining ที่มี budget จำกัด</li>
    </ul>

    <h3>ข้อจำกัดและข้อพิจารณา</h3>
    <ul className="list-disc pl-6">
      <li>ต้องใช้ resource สูงในช่วงเริ่มต้น</li>
      <li>อาจต้องออกแบบ strategy ในการ exploit/explore ให้เหมาะสมกับ task</li>
      <li>ต้องมีระบบจัดการ asynchronous execution อย่างมีประสิทธิภาพ</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        ความสามารถของ PBT ในการปรับตัวแบบต่อเนื่องทำให้สามารถลดเวลา training ลงได้มากเมื่อเทียบกับ grid search โดยเฉพาะในระบบที่มี budget และ compute จำกัด เป็นเทคนิคที่เหมาะอย่างยิ่งในงาน real-world
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Jaderberg, M. et al. (2017). Population Based Training of Neural Networks. arXiv:1711.09846</li>
      <li>Li, L. et al. (2020). A System for Massively Parallel Hyperparameter Tuning. HPCA.</li>
      <li>DeepMind Blog: https://deepmind.google/alpha-star</li>
      <li>OpenAI: Scaling Laws and PBT Exploration (2022)</li>
    </ul>
  </div>
</section>


   <section id="nas" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Neural Architecture Search (NAS)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความหมายและบทบาทของ NAS</h3>
    <p>
      Neural Architecture Search (NAS) เป็นกระบวนการค้นหาโครงสร้างของโครงข่ายประสาทเทียม (neural network architecture) ที่ให้ผลลัพธ์ดีที่สุดสำหรับ task เฉพาะ โดยไม่ต้องอาศัยการออกแบบจากมนุษย์ทั้งหมด วิธีการนี้เกิดขึ้นเพื่อรับมือกับปัญหาความซับซ้อนในการเลือกสถาปัตยกรรมที่เหมาะสม เช่น จำนวนเลเยอร์, รูปร่างของ convolution block, skip connections และอื่น ๆ
    </p>

    <h3>ประเภทของ NAS</h3>
    <ul className="list-disc pl-6">
      <li><strong>Reinforcement Learning-based NAS:</strong> ใช้ agent ในการเรียนรู้ policy เพื่อสร้างและปรับปรุงสถาปัตยกรรม</li>
      <li><strong>Evolutionary Algorithms:</strong> ใช้แนวคิดทางพันธุกรรมเช่น mutation, crossover เพื่อคัดเลือกโมเดลที่ดีขึ้น</li>
      <li><strong>Gradient-based NAS (e.g., DARTS):</strong> ใช้การไล่ระดับ (differentiable architecture search) เพื่อปรับโครงสร้างแบบต่อเนื่อง</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยจาก Google Brain (Zoph & Le, 2017) แสดงให้เห็นว่า NAS สามารถค้นหาโครงสร้างที่ outperform สถาปัตยกรรมที่มนุษย์ออกแบบ เช่น Inception และ ResNet บน CIFAR-10 และ ImageNet
      </p>
    </div>

    <h3>กระบวนการทำงานของ NAS</h3>
    <ol className="list-decimal pl-6">
      <li>กำหนด search space เช่น ความลึก, ขนาด kernel, type ของ layer</li>
      <li>ใช้ search strategy เช่น RL หรือ gradient-based เพื่อค้นหา candidate architecture</li>
      <li>ประเมิน candidate โดยใช้ proxy task หรือการฝึกแบบ full training</li>
    </ol>

    <h3>การปรับปรุง NAS สมัยใหม่</h3>
    <p>
      เพื่อให้ NAS ใช้งานได้จริงในระดับ production มีการพัฒนาเทคนิคใหม่ เช่น:
    </p>
    <ul className="list-disc pl-6">
      <li><strong>One-shot NAS:</strong> ใช้ supernet ที่แชร์ weight และทดสอบทุก candidate โดยไม่ต้องฝึกใหม่ทุกรอบ</li>
      <li><strong>Zero-cost NAS:</strong> ใช้ metric ที่คำนวณได้ทันที (e.g., gradient norm, FLOPS) เพื่อจัดอันดับโมเดล</li>
    </ul>

    <h3>เปรียบเทียบ NAS กับการออกแบบแบบมนุษย์</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">เกณฑ์</th>
          <th className="border px-4 py-2">Human Design</th>
          <th className="border px-4 py-2">Neural Architecture Search</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Speed</td>
          <td className="border px-4 py-2">เร็ว</td>
          <td className="border px-4 py-2">ช้า (แต่ลดลงเมื่อใช้ One-shot)</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Performance</td>
          <td className="border px-4 py-2">ดีถ้าผู้ออกแบบมีประสบการณ์</td>
          <td className="border px-4 py-2">ดีกว่าในหลายกรณี</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Scalability</td>
          <td className="border px-4 py-2">จำกัด</td>
          <td className="border px-4 py-2">ปรับใช้กับ task อื่นได้ง่ายกว่า</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm mt-8">
      <p className="font-semibold">Insight Box:</p>
      <p>
        แม้ NAS จะถูกมองว่าใช้ทรัพยากรมากเกินไปในยุคแรก แต่ปัจจุบันแนวทางอย่าง ProxylessNAS และ Zero-cost NAS ช่วยให้สามารถใช้งาน NAS ในงานจริงได้อย่างมีประสิทธิภาพมากขึ้น
      </p>
    </div>

    <h3>ข้อจำกัดและแนวทางในอนาคต</h3>
    <ul className="list-disc pl-6">
      <li>ใช้เวลา compute สูงเมื่อใช้ RL หรือ Evolutionary</li>
      <li>ต้องการการออกแบบ search space อย่างมีประสิทธิภาพ</li>
      <li>อนาคตเน้น interpretability และการผนวกกับ meta-learning</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. arXiv:1611.01578</li>
      <li>Pham, H. et al. (2018). Efficient Neural Architecture Search via Parameter Sharing. ICML</li>
      <li>Liu, H. et al. (2019). DARTS: Differentiable Architecture Search. ICLR</li>
      <li>Real, E. et al. (2019). Regularized Evolution for Image Classifier Architecture Search. AAAI</li>
      <li>Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural Architecture Search: A Survey. Journal of Machine Learning Research</li>
    </ul>
  </div>
</section>

   <section id="scheduling" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Smart Scheduling & Early Stopping</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>แนวคิดของการวางแผนทรัพยากร (Smart Scheduling)</h3>
    <p>
      ในการฝึกโมเดลขนาดใหญ่ ปริมาณทรัพยากรคอมพิวเตอร์ เช่น GPU และเวลาในการฝึก ถือเป็นข้อจำกัดหลัก นักวิจัยจาก Harvard และ CMU จึงได้พัฒนาแนวทางการจัดสรรทรัพยากรอย่างชาญฉลาดโดยพิจารณาความก้าวหน้าในการฝึก เช่น ความเร็วในการลด loss หรือการเปลี่ยนแปลง gradient เพื่อตัดสินใจว่าจะหยุดหรือเร่งรันกระบวนการใดก่อน
    </p>

    <h3>Early Stopping คืออะไร?</h3>
    <p>
      Early Stopping เป็นเทคนิคในการหยุดการฝึกโมเดลล่วงหน้าก่อนที่จะถึงจำนวน epoch ที่กำหนดไว้ หาก performance บน validation set ไม่ดีขึ้นเป็นระยะเวลาหนึ่ง เทคนิคนี้ช่วยลด overfitting และประหยัดเวลาในการฝึกได้อย่างมีนัยสำคัญ
    </p>

    <h3>เงื่อนไขที่พบบ่อยสำหรับ Early Stopping</h3>
    <ul className="list-disc pl-6">
      <li>Validation loss ไม่ลดลงต่อเนื่องเกินจำนวนรอบที่กำหนด (patience)</li>
      <li>Accuracy หรือ metric อื่น ๆ บน validation set หยุดนิ่ง</li>
      <li>ไม่มีการปรับค่า learning rate ตาม scheduler ที่กำหนดไว้</li>
    </ul>

    <h3>เทคนิค Scheduling ที่นิยม</h3>
    <ul className="list-disc pl-6">
      <li><strong>Step Decay:</strong> ลด learning rate ทุก ๆ k รอบ</li>
      <li><strong>Exponential Decay:</strong> learning rate ลดลงอย่างต่อเนื่องตามพารามิเตอร์</li>
      <li><strong>Cyclic Learning Rate:</strong> ปรับขึ้นลงตามวัฏจักร เพื่อออกจาก local minima</li>
      <li><strong>Cosine Annealing:</strong> ใช้ฟังก์ชัน cosine ในการลดค่า learning rate อย่างนุ่มนวล</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ในงานของ Loshchilov & Hutter (2016) จาก ICLR พบว่า การใช้ Cosine Annealing พร้อม Early Stopping สามารถเร่งการ converge และลดการใช้ GPU ได้สูงสุดถึง 40% โดยไม่สูญเสีย performance ของโมเดล
      </p>
    </div>

    <h3>การประยุกต์ใช้ร่วมกับ Hyperparameter Tuning</h3>
    <p>
      Smart Scheduling ถูกนำไปใช้ร่วมกับเทคนิคเชิง optimization อื่น เช่น Successive Halving และ Hyperband เพื่อให้สามารถหยุด trial ที่ performance แย่ได้ล่วงหน้า ช่วยให้การค้นหา hyperparameter มีประสิทธิภาพมากขึ้นโดยใช้ทรัพยากรน้อยลง
    </p>

    <table className="w-full table-auto border border-gray-300 text-sm mt-6">
      <thead className="bg-gray-700 text-white">
        <tr>
          <th className="px-4 py-2 border">เทคนิค</th>
          <th className="px-4 py-2 border">ลักษณะการทำงาน</th>
          <th className="px-4 py-2 border">ข้อดี</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Early Stopping</td>
          <td className="border px-4 py-2">หยุดฝึกหากไม่มีการพัฒนา</td>
          <td className="border px-4 py-2">ลดเวลา ป้องกัน overfitting</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Cosine Annealing</td>
          <td className="border px-4 py-2">ลด learning rate แบบโค้งนุ่ม</td>
          <td className="border px-4 py-2">ช่วยให้ converge เร็ว</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Hyperband + Early Stop</td>
          <td className="border px-4 py-2">เลือก trial ที่ดีที่สุดเร็วขึ้น</td>
          <td className="border px-4 py-2">ใช้ทรัพยากรอย่างมีประสิทธิภาพ</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 mt-6">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ในการแข่งขัน AI อย่าง NeurIPS AutoML Challenge (2022) ทีมที่ชนะล้วนใช้ Early Stopping และ Scheduling ร่วมกันเพื่อจัดการทรัพยากร GPU ให้คุ้มค่าสูงสุด โดยใช้กลยุทธ์ adaptive scheduling ที่ออกแบบเฉพาะ task ด้วย reinforcement learning
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Loshchilov & Hutter (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR.</li>
      <li>Li et al. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. ICLR.</li>
      <li>Smith (2017). Cyclical Learning Rates for Training Neural Networks. IEEE WACV.</li>
      <li>Stanford CS231n: Lecture Notes on Training Optimization Techniques</li>
      <li>NeurIPS 2022 AutoML Challenge Leaderboard and Reports</li>
    </ul>
  </div>
</section>


    <section id="metrics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. การใช้ Metric ที่เหมาะสม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ความสำคัญของ Metric ในการจูน Hyperparameter</h3>
    <p>
      ในกระบวนการปรับแต่ง Hyperparameters การเลือกใช้ metric ที่เหมาะสมมีผลอย่างยิ่งต่อคุณภาพของโมเดลที่ได้ เนื่องจาก metric ที่ใช้จะเป็นตัวชี้วัดหลักที่ algorithm optimization ยึดถือเป็นเป้าหมายในการค้นหาค่าที่ดีที่สุด หากเลือก metric ไม่เหมาะสม อาจทำให้ได้โมเดลที่แม่นยำต่ำ หรือมี bias ต่อกลุ่มข้อมูลบางประเภท
    </p>

    <h3>ประเภทของ Metric ที่นิยมใช้</h3>
    <ul className="list-disc pl-6">
      <li><strong>Accuracy:</strong> เหมาะกับ dataset ที่สมดุลทุก class</li>
      <li><strong>Precision / Recall / F1-Score:</strong> เหมาะกับ dataset ที่มี class imbalance เช่น fraud detection หรือ medical diagnosis</li>
      <li><strong>AUC-ROC:</strong> ใช้เปรียบเทียบความสามารถในการแยก class โดยไม่ขึ้นกับ threshold</li>
      <li><strong>Log Loss (Cross-Entropy):</strong> ใช้วัด probabilistic output ของโมเดล โดย penalize ความมั่นใจที่ผิด</li>
      <li><strong>RMSE / MAE:</strong> เหมาะกับ regression tasks</li>
    </ul>

    <h3>การใช้ Metric ที่สัมพันธ์กับ Business Objective</h3>
    <p>
      ในงานจริง ควรเลือก metric ที่สะท้อนเป้าหมายทางธุรกิจ เช่น:
    </p>
    <ul className="list-disc pl-6">
      <li>ในระบบ recommendation — ใช้ Precision@k หรือ NDCG</li>
      <li>ในงานคัดกรองผู้ป่วย — เน้น Recall เพื่อลด false negative</li>
      <li>ใน fraud detection — ใช้ F1-score หรือ AUCPR</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        การเลือก metric ที่ไม่สัมพันธ์กับปัญหาทางธุรกิจอาจทำให้โมเดลมี performance สูงในเชิงเทคนิค แต่ล้มเหลวในภาคปฏิบัติ เช่น การ optimize accuracy ในระบบคัดกรองโรคที่ false negative มีความเสี่ยงสูง
      </p>
    </div>

    <h3>การเลือก Metric ในงานที่มีหลายเป้าหมาย</h3>
    <p>
      บางระบบต้อง optimize หลาย objective พร้อมกัน เช่น ความแม่นยำ และ latency ในกรณีนี้ควร:
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ multi-objective optimization เช่น Pareto Front</li>
      <li>สร้าง Composite Metric เช่น Weighted F1 หรือ Custom Utility Function</li>
    </ul>

    <h3>การวิเคราะห์ Sensitivity ของ Metric</h3>
    <p>
      ควรทดสอบว่า metric ที่เลือกมีความไวต่อการเปลี่ยนแปลงของ hyperparameter หรือไม่ หาก metric เปลี่ยนแปลงเพียงเล็กน้อยแม้ค่า hyperparameter เปลี่ยนมาก แสดงว่า metric อาจไม่เหมาะกับ task ดังกล่าว
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Stanford CS229 Lecture Notes: Evaluation Metrics</li>
      <li>MIT OpenCourseWare: Machine Learning — Model Evaluation</li>
      <li>Saito, T. & Rehmsmeier, M. (2015). The Precision-Recall Plot is More Informative than the ROC Plot when Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE.</li>
      <li>Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.</li>
    </ul>
  </div>
</section>

<section id="case-studies" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. แนวคิดจากโลกจริง (Case Studies)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <h3 className="text-xl font-semibold mb-4">กรณีศึกษา: Google Vizier และการค้นหาค่า Hyperparameter ที่เหมาะสม</h3>
  <p className="mb-4">
    หนึ่งในแพลตฟอร์มที่ได้รับการยอมรับในระดับอุตสาหกรรมสำหรับการปรับจูน Hyperparameter คือ Google Vizier ซึ่งเป็นระบบภายในของ Google ที่ใช้สำหรับการทดลองเชิง optimization แบบอัตโนมัติ โดย Vizier สนับสนุนการใช้ Bayesian Optimization แบบ Tree-structured Parzen Estimator (TPE) และ Gaussian Process (GP) ในบริบทที่ซับซ้อนและมีมิติสูง
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-md mb-6">
    <strong>Insight:</strong> Google พบว่าแม้จะมีงบประมาณทรัพยากรสูง การใช้ Bayesian Optimization ที่ดีมีประสิทธิภาพกว่าการใช้ Random Search และ Grid Search ถึงหลายเท่าในการหาค่าที่เหมาะสมในหลายโมเดล เช่น XGBoost, DNN, และโมเดล NLP ขนาดใหญ่
  </div>

  <h3 className="text-xl font-semibold mb-4">กรณีศึกษา: Facebook และการปรับ Learning Rate</h3>
  <p className="mb-4">
    ในงานของ Facebook AI Research (FAIR) มีการทดลองอย่างเข้มข้นเกี่ยวกับการปรับ learning rate สำหรับการเทรนโมเดล Transformer ซึ่งพบว่าการใช้ Learning Rate Schedule ที่ออกแบบอย่างเป็นระบบ เช่น Cosine Annealing หรือ OneCycle Policy ช่วยให้ convergence เร็วขึ้น และหลีกเลี่ยง local minima ได้ดีกว่า
  </p>

  <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-md mb-6">
    <strong>Highlight:</strong> ในการเทรน BART และ RoBERTa มีการใช้ learning rate schedule ร่วมกับ warm-up strategy ซึ่งช่วยให้สามารถลดเวลาการฝึกโมเดลลงกว่า 30% เมื่อเทียบกับค่า learning rate คงที่
  </div>

  <h3 className="text-xl font-semibold mb-4">กรณีศึกษา: OpenAI และ Scaling Laws</h3>
  <p className="mb-4">
    ในงานวิจัย Scaling Laws ของ OpenAI มีการระบุชัดเจนว่า hyperparameter ที่สำคัญที่สุดในกรณีของ LLM คือ learning rate และ batch size โดยความสัมพันธ์ของพารามิเตอร์เหล่านี้กับจำนวน parameter ในโมเดลสามารถทำให้ประสิทธิภาพของการฝึกเพิ่มขึ้นแบบ non-linear
  </p>

  <table className="table-auto w-full border border-gray-300 dark:border-gray-700 mb-6 text-sm">
    <thead>
      <tr className="bg-gray-200 dark:bg-gray-800">
        <th className="px-4 py-2 border">Hyperparameter</th>
        <th className="px-4 py-2 border">ผลกระทบหลัก</th>
        <th className="px-4 py-2 border">Scaling Behavior</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="px-4 py-2 border">Learning Rate</td>
        <td className="px-4 py-2 border">ควบคุมการเคลื่อนที่ของ gradient</td>
        <td className="px-4 py-2 border">∝ 1/√batch_size</td>
      </tr>
      <tr>
        <td className="px-4 py-2 border">Batch Size</td>
        <td className="px-4 py-2 border">ส่งผลต่อ noise และ memory</td>
        <td className="px-4 py-2 border">↑ batch_size → ↓ noise</td>
      </tr>
    </tbody>
  </table>

  <h3 className="text-xl font-semibold mb-4">การเรียนรู้จากกรณีจริง</h3>
  <p className="mb-4">
    การศึกษากรณีจากองค์กรระดับโลกสะท้อนถึงแนวโน้มในการใช้ระบบอัตโนมัติ เช่น AutoML, Bayesian Tuning, และ Reinforcement Learning เพื่อจัดการกับ hyperparameter ที่มีหลายมิติและซับซ้อน โดยเฉพาะอย่างยิ่งเมื่อเป้าหมายคือการลดเวลาและค่าใช้จ่ายในการฝึกโมเดลในสเกลใหญ่
  </p>

  <h3 className="text-xl font-semibold mt-10 mb-4">แหล่งอ้างอิง</h3>
  <ul className="list-disc list-inside text-sm space-y-1">
    <li>Golovin et al., "Google Vizier: A Service for Black-Box Optimization," arXiv:1706.04473</li>
    <li>Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," arXiv:1706.02677</li>
    <li>Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361</li>
    <li>He et al., "Delving Deep into Rectifiers," arXiv:1502.01852</li>
  </ul>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day38 theme={theme} />
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
        <ScrollSpy_Ai_Day38 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day38_HyperparameterTuning;
