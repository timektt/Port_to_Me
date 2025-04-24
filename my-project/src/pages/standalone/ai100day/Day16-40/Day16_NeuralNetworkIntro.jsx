// โครงสร้างเริ่มต้น Day16: Introduction to Neural Networks
import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day16 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day16";
import MiniQuiz_Day16 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day16";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day16_NeuralNetworkIntro = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("neural1").format("auto").quality("auto").resize(scale().width(600));
  const img2 = cld.image("neural2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("neural3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("neural4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("neural5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("neural6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("neural7").format("auto").quality("auto").resize(scale().width(400));
  const img8 = cld.image("neural8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("neural9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("neural10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("neural11").format("auto").quality("auto").resize(scale().width(600));
  const img12 = cld.image("neural12").format("auto").quality("auto").resize(scale().width(600));
  const img13 = cld.image("neural13").format("auto").quality("auto").resize(scale().width(600));
  const img14 = cld.image("neural14").format("auto").quality("auto").resize(scale().width(600));
  const img15 = cld.image("neural15").format("auto").quality("auto").resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 16: Introduction to Neural Networks</h1>

        <section id="overview" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">1. Overview: ทำไมต้อง Neural Networks</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img1} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Neural Networks ถือเป็นหนึ่งในนวัตกรรมที่ทรงพลังที่สุดในโลกของ Machine Learning และปัญญาประดิษฐ์ ถูกออกแบบมาเพื่อแก้ปัญหาที่ไม่สามารถจัดการได้อย่างมีประสิทธิภาพด้วยโมเดลเชิงเส้น (Linear Models) แบบดั้งเดิม ความสามารถหลักที่ทำให้ Neural Network โดดเด่นคือความสามารถในการเรียนรู้ฟังก์ชันที่ไม่เป็นเชิงเส้น (Non-linear Functions) และการเป็นตัวแทนของ Universal Function Approximator
    </p>

    <p>
      Linear Models เช่น Linear Regression หรือ Logistic Regression นั้นแม้จะมีความเรียบง่ายและตีความได้ง่าย แต่ก็มีข้อจำกัดด้านการเรียนรู้ฟังก์ชันที่ซับซ้อน ในหลายกรณี เช่น การจดจำภาพ การวิเคราะห์ภาษา หรือการทำนายพฤติกรรมที่มีความซับซ้อนสูง โมเดลเชิงเส้นไม่สามารถแยกแยะความสัมพันธ์เชิงซ้อนในข้อมูลได้อย่างเพียงพอ
    </p>

    <div className="grid md:grid-cols-2 gap-6 mt-8">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
        <h3 className="text-lg font-semibold mb-3">ข้อจำกัดของ Linear Models</h3>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ไม่สามารถแยก class ที่มี decision boundary แบบโค้งได้</li>
          <li>ไม่สามารถเรียนรู้ feature interaction ที่ซับซ้อนได้โดยตรง</li>
          <li>มี bias สูงในกรณีที่ข้อมูลมี non-linear patterns</li>
        </ul>
      </div>
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
        <h3 className="text-lg font-semibold mb-3">กรณีศึกษาที่ Linear Model ล้มเหลว</h3>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>การทำนายภาพจำแนกหมวดหมู่ (Image Classification)</li>
          <li>ระบบวิเคราะห์ความรู้สึกในข้อความ (Sentiment Analysis)</li>
          <li>การแปลภาษา (Machine Translation)</li>
          <li>การแยกแยะความสัมพันธ์แบบ XOR</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-5 text-center">Neural Network: การก้าวข้ามขีดจำกัด</h3>
    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
    <p>
      Neural Network ออกแบบให้มีโครงสร้างแบบชั้น (Layered Architecture) ซึ่งแต่ละ layer ทำหน้าที่แปลงและแสดงลักษณะของข้อมูลในรูปแบบที่ซับซ้อนขึ้น ผ่านการรวมกันของน้ำหนัก (weights), การคูณเมทริกซ์, และฟังก์ชันการกระตุ้น (Activation Functions) ที่ช่วยสร้าง non-linearity ให้กับโมเดล
    </p>

    <p>
      ความสามารถของ Neural Network ในการเป็น Universal Function Approximator ได้รับการพิสูจน์ทางทฤษฎีว่า เพียงแค่มี Hidden Layer เพียงหนึ่งชั้นที่ใหญ่เพียงพอ ก็สามารถประมาณค่าฟังก์ชันที่ไม่เป็นเชิงเส้นใด ๆ ได้ภายใต้เงื่อนไขบางประการ อย่างไรก็ตาม โครงสร้างแบบลึก (Deep Architecture) ที่มีหลายชั้นมักให้ประสิทธิภาพดีกว่า เนื่องจากสามารถเรียนรู้ hierarchical features ได้โดยตรง
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-800 text-yellow-900 dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400">
      <p className="font-medium mb-1">Insight:</p>
      <p className="text-sm">
        ความแตกต่างหลักระหว่าง Linear Models กับ Neural Networks ไม่ใช่เพียงรูปแบบสมการ แต่คือ "ความสามารถในการแสดงออก" (Expressive Power) ซึ่ง Neural Networks มีความยืดหยุ่นสูงกว่าในการเรียนรู้ฟังก์ชันที่มีโครงสร้างซับซ้อนหรือมีความสัมพันธ์เชิงลึกระหว่างข้อมูล
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">เมื่อไหร่ควรใช้ Neural Networks?</h3>
    <ul className="list-disc pl-6 text-sm space-y-2">
      <li>ข้อมูลมีความไม่เชิงเส้นสูง เช่น รูปภาพ, เสียง, ภาษา</li>
      <li>ปัญหามีขนาดข้อมูลใหญ่และต้องการการเรียนรู้เชิงลึก</li>
      <li>ต้องการ performance สูงกว่า Linear Models อย่างมีนัยสำคัญ</li>
      <li>งานต้องการการแยกแยะฟีเจอร์ที่มีความซับซ้อนแบบหลายมิติ</li>
    </ul>

    <div className="grid md:grid-cols-2 gap-6 mt-10">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
        <h4 className="text-lg font-medium mb-2">โมเดลที่ใช้ Neural Networks</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>Convolutional Neural Networks (CNN) สำหรับภาพ</li>
          <li>Recurrent Neural Networks (RNN) สำหรับลำดับเวลา</li>
          <li>Transformer สำหรับภาษาและ sequence-to-sequence tasks</li>
          <li>Deep Reinforcement Learning สำหรับการควบคุม</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
        <h4 className="text-lg font-medium mb-2">แนวทางการเรียนรู้</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>เริ่มจากปัญหาง่าย ๆ ที่ Linear Model ทำไม่ได้ เช่น XOR</li>
          <li>ใช้ไลบรารี เช่น Keras หรือ PyTorch เพื่อทดลอง Network</li>
          <li>วิเคราะห์การไหลของข้อมูล (Forward Pass) และการเรียนรู้ (Backward Pass)</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10">สรุปประเด็นสำคัญ</h3>
    <ul className="list-disc pl-6 text-sm space-y-2">
      <li>Linear Models มีข้อจำกัดในการเรียนรู้ข้อมูลที่ไม่เป็นเชิงเส้น</li>
      <li>Neural Networks สามารถแสดงออกและจำลองฟังก์ชันซับซ้อนได้</li>
      <li>Neural Networks เป็น Universal Function Approximator</li>
      <li>มีโครงสร้างหลายแบบที่เหมาะกับข้อมูลประเภทต่าง ๆ</li>
      <li>เป็นฐานรากของเทคโนโลยี AI สมัยใหม่ เช่น GPT, DALL·E, AlphaGo</li>
    </ul>

    <div className="bg-green-100 dark:bg-green-800 text-green-900 dark:text-green-100 p-5 rounded-xl border-l-4 border-green-400 mt-8">
      <p className="font-semibold mb-2">คำกล่าวจาก Yann LeCun:</p>
      <p className="text-sm italic">
        “The beauty of neural nets is not in their architecture, but in their ability to learn features directly from raw data.”
      </p>
    </div>
  </div>
</section>


<section id="neuron" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    2. ส่วนประกอบหลักของ Neural Network
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Neural Network ถูกออกแบบโดยอิงแรงบันดาลใจจากระบบประสาทของมนุษย์ โดยมีหน่วยพื้นฐานคือ <strong>Neuron</strong> หรือบางครั้งเรียกว่า <em>Perceptron</em> ซึ่งเป็นกลไกที่จำลองการทำงานของเซลล์ประสาทจริงในสมองมนุษย์ โดยรับข้อมูลจากหลายแหล่ง ประมวลผล และส่งผลลัพธ์ต่อไปยังเซลล์ถัดไป
    </p>

    <h3 className="text-xl font-semibold mt-8">Neuron ทำงานอย่างไร?</h3>
    <p>
      Neuron หนึ่งตัวจะรับค่า <code>input</code> หลายมิติจาก layer ก่อนหน้า แล้วนำค่าเหล่านั้นมาคูณกับ <code>weight</code> ที่สัมพันธ์กับแต่ละ input ก่อนจะบวกค่าคงที่ <code>bias</code> แล้วจึงส่งผ่านฟังก์ชันไม่เชิงเส้น (activation function) เพื่อสร้างผลลัพธ์ที่มีความยืดหยุ่นมากกว่า linear function
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded-xl p-4 overflow-auto">
{`// สมการของ neuron
Z = W1 * X1 + W2 * X2 + ... + Wn * Xn + b
Z = W • X + b
A = f(Z) // f คือ activation function เช่น ReLU, Sigmoid`}
    </pre>

    <p>
      ค่าที่ได้จากสมการ <code>Z</code> เป็นค่าเชิงเส้นที่รวม input ทั้งหมดเข้าด้วยกัน ส่วน <code>A</code> เป็นค่าที่ผ่านการแปลงด้วย activation function เพื่อสร้างความไม่เชิงเส้นให้กับระบบ ช่วยให้โมเดลสามารถเรียนรู้ฟังก์ชันซับซ้อนที่ไม่สามารถแทนด้วยเส้นตรงได้
    </p>

    <h3 className="text-xl font-semibold mt-8">Activation Function ที่ใช้บ่อย</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-5 border rounded-xl shadow text-sm">
        <h4 className="font-semibold mb-2">ReLU (Rectified Linear Unit)</h4>
        <p>ส่งค่าบวกกลับมาเท่าเดิม ถ้า input น้อยกว่า 0 จะส่ง 0 ออกไป ใช้บ่อยที่สุดใน hidden layers</p>
        <code>f(x) = max(0, x)</code>
      </div>
      <div className="bg-white dark:bg-gray-800 p-5 border rounded-xl shadow text-sm">
        <h4 className="font-semibold mb-2">Sigmoid</h4>
        <p>เปลี่ยนค่าใด ๆ ให้อยู่ระหว่าง 0 ถึง 1 ใช้ใน output layer สำหรับ classification</p>
        <code>f(x) = 1 / (1 + e<sup>-x</sup>)</code>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-8">Hidden Layer คืออะไร?</h3>
    <p>
      Hidden Layer หมายถึง layer ที่อยู่ระหว่าง input และ output ซึ่งเป็นจุดที่โมเดลเรียนรู้ pattern และความสัมพันธ์ของข้อมูล แต่ละ layer จะประกอบด้วย neuron หลายตัว ซึ่งทำงานประมวลผลข้อมูลจาก layer ก่อนหน้าและส่งผลลัพธ์ต่อไปยัง layer ถัดไป
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow">
      <p className="font-semibold mb-2">Insight:</p>
      <p className="text-sm">
        การเพิ่มจำนวน hidden layers และจำนวน neuron ต่อ layer จะช่วยให้โมเดลสามารถเข้าใจความสัมพันธ์ที่ซับซ้อนขึ้นได้ แต่ก็เสี่ยงต่อ overfitting หากไม่มีการ regularization หรือ dataset ที่เพียงพอ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8">Output Layer</h3>
    <p>
      Output layer เป็นชั้นสุดท้ายของโมเดล ทำหน้าที่ส่งผลลัพธ์สุดท้ายที่ใช้ในการตัดสินใจ เช่น การทำนายผลลัพธ์ การจัดหมวดหมู่ โดยจำนวน neuron ใน output layer จะขึ้นอยู่กับลักษณะของปัญหา เช่น:
    </p>
    <ul className="list-disc pl-6 text-sm">
      <li>1 neuron พร้อม Sigmoid activation → ปัญหา binary classification</li>
      <li>หลาย neuron พร้อม Softmax → ปัญหา multi-class classification</li>
      <li>ไม่มี activation → ปัญหา regression</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">การวางโครงสร้างให้เรียนรู้ได้ดี</h3>
    <p>
      โครงสร้างของ Neural Network ต้องพิจารณาให้เหมาะกับลักษณะข้อมูล เช่น ข้อมูลที่มีความสัมพันธ์เชิงลำดับเวลา อาจต้องใช้ recurrent network หรือ temporal-based architecture ส่วนข้อมูลภาพ อาจใช้ convolutional network ร่วมกับ dense layer เพื่อประมวลผลขั้นสุดท้าย
    </p>
  </div>
</section>


<section id="architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. พื้นฐานโครงสร้างของ Neural Network</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Neural Network ถูกสร้างขึ้นจากชั้นของหน่วยที่เชื่อมต่อกัน เรียกว่า <strong>Neuron</strong> โดยองค์ประกอบพื้นฐานที่สุดในโครงสร้างของ Neural Network คือ <strong>Dense Layer</strong> หรือที่เรียกว่า Fully Connected Layer ซึ่งในแต่ละชั้นนั้น neuron ทุกตัวจะเชื่อมโยงกับ neuron ทุกตัวในชั้นถัดไปผ่านพารามิเตอร์ที่ปรับค่าได้ เรียกว่า <strong>Weights</strong>
    </p>

    <p>
      การเชื่อมโยงแต่ละคู่ neuron จะมี <strong>weight (W)</strong> กำกับอยู่ และ neuron แต่ละตัวจะมี <strong>bias (b)</strong> ของตนเอง พารามิเตอร์เหล่านี้จะถูกเรียนรู้ระหว่างการฝึกโมเดล และมีบทบาทสำคัญในการควบคุมอิทธิพลของ input หรือค่า activation ต่อผลลัพธ์สุดท้าย การคำนวณหลักภายใน neuron คือการรวมเชิงเส้นของข้อมูลเข้าตามด้วยการแปลงแบบไม่เชิงเส้นด้วย activation function
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <p className="text-sm">
        <strong>สมการ:</strong> Z = W<sub>i</sub>X + b<sub>i</sub>, และ A = activation(Z)
      </p>
    </div>

    <p>
      กระบวนการส่งข้อมูลจากชั้นหนึ่งไปยังอีกชั้นหนึ่งเรียกว่า <strong>Forward Pass</strong> ในขั้นตอนนี้ ข้อมูลจะถูกแปลงชั้นต่อชั้น ผ่านการคูณน้ำหนักและการใช้ activation function จนได้ผลลัพธ์สุดท้ายที่ชั้น output การแปลงนี้ทำให้โมเดลสามารถจับความสัมพันธ์ที่ซับซ้อนระหว่าง feature และเป้าหมายได้
    </p>

    <h3 className="text-xl font-semibold mt-8">ประเภทของ Layer และบทบาทของแต่ละชั้น</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Input Layer:</strong> รับข้อมูลดิบ โดย neuron แต่ละตัวจะแทน feature หนึ่งตัวจาก dataset
      </li>
      <li>
        <strong>Hidden Layers:</strong> ทำหน้าที่แปลงข้อมูลแบบซ่อนอยู่ เป็นที่ที่โมเดลเรียนรู้ลักษณะของข้อมูล
      </li>
      <li>
        <strong>Output Layer:</strong> สร้างผลลัพธ์สุดท้าย เช่น การจำแนกหรือการคาดการณ์ โดยมักใช้ activation เช่น softmax หรือ sigmoid
      </li>
    </ul>

    <p>
      Dense Layer มีความสามารถในการแทนฟังก์ชันใด ๆ ได้ตามทฤษฎี จึงถือว่าเป็น <strong>Universal Function Approximator</strong> อย่างไรก็ตาม ความสามารถในการแทนฟังก์ชันจะขึ้นอยู่กับความลึก (depth) และความกว้าง (width) ของโมเดล รวมถึงการเลือก activation function และการกำหนดค่าเริ่มต้นของ weight
    </p>

    <div className="grid md:grid-cols-2 gap-6 mt-6">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">ข้อดีของ Dense Layer</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>สามารถเรียนรู้รูปแบบที่ไม่เชิงเส้นได้ดี</li>
          <li>ใช้งานง่ายด้วยไลบรารีมาตรฐาน เช่น Keras, PyTorch</li>
          <li>เหมาะกับข้อมูลแบบตารางหรือข้อมูลที่มีโครงสร้าง</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">ข้อจำกัด</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ไม่สามารถเข้าใจโครงสร้างพื้นที่ เช่น รูปภาพ ได้ดี</li>
          <li>มีความเสี่ยง overfit หากใหญ่เกินไปและไม่มีการ regularization</li>
          <li>พารามิเตอร์จำนวนมากใน input ที่มีมิติสูง (High-dimensional)</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-8">ลำดับขั้นตอนของ Forward Pass</h3>
    <ol className="list-decimal pl-6 space-y-2">
      <li>เริ่มจาก input vector ดิบ X</li>
      <li>คูณ X กับ weight matrix W ของชั้นแรก</li>
      <li>บวกกับ bias vector b</li>
      <li>ใช้ activation function (เช่น ReLU)</li>
      <li>ทำซ้ำสำหรับทุก hidden layer</li>
      <li>สร้าง output สุดท้าย Y ที่ชั้นสุดท้าย</li>
    </ol>

    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

    <h3 className="text-xl font-semibold mt-10">Insight</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <p className="text-sm">
        แม้จะมีสถาปัตยกรรมเฉพาะทางเกิดขึ้นมากมาย แต่ Fully Connected Layer ยังคงเป็นพื้นฐานสำคัญของ Deep Learning การเข้าใจบทบาทของ weight, bias และ activation function ใน Forward Pass เป็นสิ่งจำเป็นต่อการแปลผลและแก้ไขพฤติกรรมของโมเดล
      </p>
    </div>
  </div>
</section>


<section id="feedforward" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    4. From Input → Output: Feedforward Process
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-5">
    <p>
      Feedforward คือกระบวนการพื้นฐานของ Neural Network ที่ข้อมูลจะถูกส่งต่อจากชั้นอินพุต (Input Layer) ผ่านชุดของชั้นซ่อน (Hidden Layers)
      ไปยังชั้นเอาต์พุต (Output Layer) โดยไม่มีการย้อนกลับของข้อมูล กระบวนการนี้เป็นแกนกลางของการคำนวณในโมเดลประเภท Fully Connected Network
    </p>

    <p>
      ในแต่ละ Layer ข้อมูลจะถูกแปลงเชิงคณิตศาสตร์ผ่านฟังก์ชันแบบเชิงเส้น (Linear Transformation) ตามสมการ:
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg shadow border dark:border-gray-700 text-sm overflow-auto">
      <code className="block text-center">
        Z = W × X + b<br />
        A = Activation(Z)
      </code>
    </div>

    <p>
      โดยที่ W คือ weight matrix, X คือ input vector, b คือ bias vector และ A คือค่าผลลัพธ์หลังผ่านฟังก์ชันกระตุ้น (Activation Function) การคำนวณนี้จะทำในแต่ละ node ของ Layer และค่าที่ได้จะถูกส่งต่อไปยังชั้นถัดไป
    </p>

    <h3 className="text-xl font-semibold mt-8">ลำดับของกระบวนการ Feedforward</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>รับข้อมูลดิบ (เช่น ตัวเลข, vector ของ feature) ที่ถูก normalize แล้ว</li>
      <li>คูณข้อมูลกับ weight matrix ของชั้นแรก พร้อมบวก bias</li>
      <li>นำผลลัพธ์ไปผ่าน Activation Function เพื่อเพิ่มความไม่เชิงเส้น (non-linearity)</li>
      <li>ค่าที่ได้จะถูกส่งต่อไปยัง Layer ถัดไป ทำซ้ำขั้นตอนเดิม</li>
      <li>ผลลัพธ์สุดท้ายของ Layer สุดท้ายจะถูกส่งไปยัง Output Layer</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">ตัวอย่าง: XOR Problem</h3>
    <p>
      ปัญหา XOR เป็นตัวอย่างคลาสสิกที่แสดงให้เห็นว่าโมเดล Linear ไม่สามารถแยกข้อมูลแบบไม่เชิงเส้นได้ จึงต้องใช้ Neural Network แบบมี Hidden Layer
    </p>

    <div className="flex justify-center my-6">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border shadow w-full md:w-3/4">
      <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
        <p className="text-center text-sm text-gray-500 mt-2">
          แผนภาพโครงสร้าง XOR Neural Network (2-2-1)
        </p>
      </div>
    </div>

    <p>
      โครงสร้าง XOR Neural Network ประกอบด้วย 2 input neurons, 2 hidden neurons และ 1 output neuron โดยใช้ Activation Function เช่น ReLU หรือ Sigmoid
    </p>

    <h3 className="text-xl font-semibold mt-8">ตัวอย่างการคำนวณ Feedforward อย่างง่าย</h3>

    <div className="bg-gray-800 text-white text-sm p-4 rounded-lg overflow-auto">
      <pre>{`
import numpy as np

# Input: XOR [1, 0]
X = np.array([1, 0])

# Weights และ Bias ของ Hidden Layer
W1 = np.array([[1, 1], [1, 1]])     # shape: (2, 2)
b1 = np.array([0, -1])              # shape: (2,)

# Activation Function (Sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hidden Layer
Z1 = np.dot(W1, X) + b1             # Linear
A1 = sigmoid(Z1)                    # Non-linear

# Output Layer
W2 = np.array([1, -2])              # shape: (2,)
b2 = 0

Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

print("Output:", A2)
      `}</pre>
    </div>

    <p>
      โค้ดด้านบนแสดงให้เห็นกระบวนการ feedforward อย่างง่าย โดยคำนวณผ่าน Hidden Layer ไปยัง Output Layer
      ซึ่งค่าผลลัพธ์สุดท้าย A2 คือค่าที่ได้จากโมเดล
    </p>

    <h3 className="text-xl font-semibold mt-10">Feedforward เป็นพื้นฐานของการเรียนรู้</h3>
    <p>
      การทำ Feedforward ไม่เพียงแต่ใช้ในการคำนวณผลลัพธ์ แต่ยังเป็นส่วนสำคัญในการ Training โดยจะถูกใช้ควบคู่กับการคำนวณ Loss Function และการอัปเดตค่าผ่าน Backpropagation (ซึ่งจะอธิบายเพิ่มเติมใน Day ถัดไป)
    </p>

    <div className="bg-green-100 dark:bg-green-900 text-black dark:text-green-100 p-5 rounded-xl border-l-4 border-green-500 mt-8 shadow">
      <h4 className="font-semibold mb-2">Insight:</h4>
      <p className="text-sm">
        การออกแบบ Feedforward ที่ดีควรเลือกจำนวน Layer, Activation Function และโครงสร้างของ Hidden Units ให้เหมาะสมกับลักษณะของปัญหา โดยเฉพาะปัญหาที่ไม่สามารถแยกด้วยเส้นตรง จำเป็นต้องใช้ Layer ที่ลึกพอจะจับความซับซ้อนได้
      </p>
    </div>
  </div>
</section>

<section id="learning" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    5. การเรียนรู้: จาก Error สู่การอัปเดต
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การเรียนรู้ของ Neural Network เริ่มจากการเปรียบเทียบค่าที่โมเดลทำนายกับค่าจริงผ่านฟังก์ชัน Loss เพื่อวัดว่าโมเดลพลาดไปมากน้อยเพียงใด หลังจากนั้นระบบจะใช้แนวทาง Backpropagation เพื่อถ่ายทอดข้อผิดพลาดย้อนกลับไปยัง Layer ก่อนหน้า โดยอาศัยหลักการ Chain Rule ใน Calculus จากนั้นจะใช้ Gradient Descent เพื่อปรับค่าพารามิเตอร์ของโมเดลให้เหมาะสมขึ้นในแต่ละรอบของการเรียนรู้
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-xl border border-gray-300 dark:border-gray-700">
      <h3 className="text-xl font-semibold mb-2">Loss Function คืออะไร?</h3>
      <p>
        Loss Function คือฟังก์ชันที่ใช้วัดข้อผิดพลาดของโมเดล ตัวอย่างเช่น Mean Squared Error (MSE) ใช้ในงาน Regression หรือ Cross Entropy Loss ใช้ในงาน Classification ยิ่งค่าของ Loss ต่ำ แสดงว่าโมเดลมีความแม่นยำสูงขึ้น
      </p>
      <pre className="bg-gray-900 text-white text-sm p-4 rounded-md overflow-x-auto">
{`# ตัวอย่างการใช้ Loss Function ใน PyTorch
import torch
import torch.nn as nn

# สำหรับงาน Classification
loss_fn = nn.CrossEntropyLoss()
pred = torch.tensor([[2.0, 1.0, 0.1]])  # logits
label = torch.tensor([0])  # class ที่ถูกต้องคือ index 0
loss = loss_fn(pred, label)
print('Loss:', loss.item())`}
      </pre>
    </div>

    <div className="bg-white dark:bg-gray-900 p-5 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h3 className="text-xl font-semibold mb-2">Backpropagation ทำงานอย่างไร?</h3>
      <p>
        เมื่อได้ค่าความผิดพลาดจาก Loss Function ระบบจะถ่ายทอดค่าดังกล่าวย้อนกลับไปยัง Layer ก่อนหน้าทีละชั้นโดยใช้ Gradient (อนุพันธ์) ของแต่ละ Layer ผ่าน Chain Rule เพื่อระบุว่าแต่ละพารามิเตอร์ควรเปลี่ยนไปในทิศทางใดจึงจะลดข้อผิดพลาดได้
      </p>
      <p>
        การทำ Backpropagation ไม่เพียงแค่ถ่ายทอดข้อผิดพลาด แต่ยังต้องคำนวณ Gradient ให้ถูกต้องในแต่ละ Layer โดย Framework เช่น PyTorch จะใช้ Autograd ทำหน้าที่นี้โดยอัตโนมัติ
      </p>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 p-5 rounded-xl border border-gray-300 dark:border-gray-700">
      <h3 className="text-xl font-semibold mb-2">Gradient Descent: ปรับพารามิเตอร์อย่างไร?</h3>
      <p>
        Gradient Descent คืออัลกอริทึมที่ใช้ปรับค่าพารามิเตอร์ในทิศทางตรงข้ามกับ Gradient เพื่อให้ค่าความผิดพลาดลดลงในแต่ละรอบของการฝึก โมเดลจะอัปเดต Weight และ Bias ทีละนิด ๆ อย่างค่อยเป็นค่อยไป
      </p>
      <pre className="bg-gray-900 text-white text-sm p-4 rounded-md overflow-x-auto">
{`# อัปเดตพารามิเตอร์ด้วย Gradient Descent
learning_rate = 0.01
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad`}
      </pre>
    </div>

    <div className="grid md:grid-cols-2 gap-6 mt-6">
      <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-2">Hyperparameter ที่สำคัญ</h4>
        <ul className="list-disc pl-5 space-y-1 text-sm">
          <li>Learning Rate: ความเร็วในการปรับค่าพารามิเตอร์</li>
          <li>Batch Size: จำนวนข้อมูลที่ใช้ต่อรอบการอัปเดต</li>
          <li>Epoch: จำนวนรอบที่โมเดลเรียนรู้ข้อมูลทั้งหมด</li>
          <li>Momentum: การสะสม Gradient เพื่อให้การปรับค่าราบรื่นขึ้น</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-300 dark:border-gray-700">
        <h4 className="font-semibold mb-2">ความเข้าใจผิดที่พบบ่อย</h4>
        <ul className="list-disc pl-5 space-y-1 text-sm">
          <li>คิดว่า Loss ต่ำที่สุดคือเป้าหมายเดียว ทั้งที่อาจ Overfit</li>
          <li>ใช้ Learning Rate สูงเกินไป ทำให้โมเดลไม่ converge</li>
          <li>เข้าใจว่า Backpropagation คือการปรับค่าทันที ทั้งที่เพียงแค่คำนวณ Gradient</li>
        </ul>
      </div>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500">
      <h4 className="font-semibold mb-2">Insight:</h4>
      <p className="text-sm">
        แนวทางการเรียนรู้ของ Neural Network เปรียบได้กับกระบวนการปรับตัวในสิ่งมีชีวิตที่ตอบสนองต่อสิ่งแวดล้อมอย่างค่อยเป็นค่อยไป และพัฒนาศักยภาพในการทำนายให้แม่นยำยิ่งขึ้นในทุกครั้งที่ได้รับข้อมูลใหม่
      </p>
    </div>
  </div>
</section>

<section id="depth-width" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">
    6. Network Depth vs Width
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-5">
    <p>
      การออกแบบโครงสร้างของ Neural Network มีองค์ประกอบสำคัญ 2 ประการ ได้แก่
      ความลึก (Depth) ซึ่งหมายถึงจำนวนชั้นของ Layer และความกว้าง (Width)
      ซึ่งหมายถึงจำนวน Neurons ในแต่ละ Layer โดยทั้งสองปัจจัยนี้มีผลต่อความสามารถของโมเดลในการเรียนรู้ฟังก์ชันที่ซับซ้อน
    </p>

    <h3 className="text-xl font-semibold">Shallow Neural Network คืออะไร?</h3>
    <p>
      Shallow Network คือโมเดลที่มี Hidden Layer เพียง 1 ชั้นเท่านั้น แม้จะสามารถใช้แก้ปัญหาง่าย ๆ ได้ดี เช่นการจำแนกรูปแบบเบื้องต้น แต่ไม่สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนได้อย่างมีประสิทธิภาพ เช่น รูปแบบของภาษา หรือภาพที่มีโครงสร้างหลายชั้น
    </p>

    <h3 className="text-xl font-semibold">Deep Neural Network คืออะไร?</h3>
    <p>
      Deep Network มีหลายชั้นของ Hidden Layer โดยชั้นลึก ๆ ช่วยเรียนรู้ Feature ที่มีความเป็นนามธรรมสูง เช่น การตรวจจับเส้นขอบในภาพ → รูปร่างของวัตถุ → การจำแนกวัตถุ ความลึกของโมเดลจึงสัมพันธ์กับความสามารถในการ "เข้าใจ" ข้อมูลที่ซับซ้อน
    </p>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-6 border border-gray-300 dark:border-gray-700 rounded-xl shadow">
        <h4 className="font-semibold text-lg mb-2">ข้อดีของ Shallow Network</h4>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>ฝึกได้เร็วกว่า ใช้ทรัพยากรน้อย</li>
          <li>เหมาะกับข้อมูลที่มี Feature ชัดเจน</li>
          <li>โครงสร้างเข้าใจง่าย ตรวจสอบได้</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 border border-gray-300 dark:border-gray-700 rounded-xl shadow">
        <h4 className="font-semibold text-lg mb-2">ข้อดีของ Deep Network</h4>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>สามารถเรียนรู้ Feature ที่ซับซ้อนได้</li>
          <li>รองรับงานด้านภาพ เสียง ภาษา ได้ดีกว่า</li>
          <li>สามารถทำ Transfer Learning ได้</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-6">ปัญหาของการเพิ่ม Layer มากเกินไป</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Overfitting:</strong> โมเดลมีพารามิเตอร์มากจนจำข้อมูลเทรนได้เป๊ะเกินไป ทำให้ไม่สามารถ generalize กับข้อมูลใหม่ได้
      </li>
      <li>
        <strong>Vanishing/Exploding Gradients:</strong> ค่าที่ส่งต่อระหว่าง Layer อาจหายไปหรือระเบิด ทำให้ฝึกโมเดลไม่ได้
      </li>
      <li>
        <strong>Training Time:</strong> ยิ่งลึก ยิ่งใช้เวลาและพลังงานมากในการเทรน
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-6">ความสัมพันธ์ของ Width กับ Performance</h3>
    <p>
      ในบางกรณี การเพิ่มจำนวน Neurons ในแต่ละ Layer (เพิ่ม Width) อาจช่วยเพิ่มขีดความสามารถในการเรียนรู้ของโมเดล แต่ถ้ามากเกินไปจะนำไปสู่ Overfitting เช่นเดียวกับการเพิ่มความลึก โดยเฉพาะในข้อมูลที่มี Noise หรือมีขนาดเล็ก
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 dark:border-yellow-600">
      <p className="font-semibold mb-2">Insight:</p>
      <p className="text-sm">
        โมเดลลึกมากไม่จำเป็นต้องดีกว่าเสมอไป การออกแบบสถาปัตยกรรมควรสัมพันธ์กับปริมาณข้อมูล ความซับซ้อนของ Feature และเป้าหมายของงาน เช่น ในการวิเคราะห์เสียงพูด Deep Network อาจจำเป็น แต่ในงานจำแนกตัวเลข MNIST เพียง 2–3 ชั้นก็เพียงพอ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-6">แนวทางการเลือก Depth และ Width อย่างมีเหตุผล</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เริ่มจากโครงสร้างเล็กแล้วค่อยเพิ่มเมื่อจำเป็น</li>
      <li>ใช้ Validation Set เพื่อตรวจสอบการ Overfitting</li>
      <li>วิเคราะห์ Training Curve เพื่อตรวจจับปัญหาความลึก</li>
      <li>ใช้ Regularization (เช่น Dropout, L2) เพื่อลด Overfitting</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6 text-center">Visual เปรียบเทียบ</h3>
    <p>
      ด้านล่างเป็นภาพเปรียบเทียบระหว่าง Shallow กับ Deep Network พร้อมแสดงข้อดีข้อเสีย และตัวอย่างการเรียนรู้ข้อมูล:
    </p>

    <div className="flex justify-center my-6">
    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
    </div>

    <p>
      โดยสรุป การตัดสินใจว่าจะเพิ่ม Layer หรือ Neuron ควรทำบนฐานของหลักการทางข้อมูลและผลลัพธ์จากการทดลอง ไม่ใช่การเดา หรือยึดติดกับโครงสร้างใดโครงสร้างหนึ่งเป็นหลัก การเข้าใจว่าโมเดลจะซับซ้อนแค่ไหนจึงเป็นหัวใจของการออกแบบระบบที่มีประสิทธิภาพ
    </p>
  </div>
</section>

<section id="activation-choice" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">7. การเลือก Activation Function ในแต่ละ Layer</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-4">
    <p>
      Activation Function คือหัวใจของ Neural Network ที่ทำให้โมเดลสามารถเรียนรู้รูปแบบที่ไม่เชิงเส้นได้ หากไม่มี Activation Function ระบบจะกลายเป็นเพียง linear transformation ซึ่งไม่สามารถจำลองฟังก์ชันที่ซับซ้อนได้ การเลือก Activation Function ที่เหมาะสมในแต่ละ Layer จึงมีผลโดยตรงต่อประสิทธิภาพและการเรียนรู้ของโมเดล
    </p>

    <h3 className="text-xl font-semibold mt-8">Activation Function ที่ใช้บ่อยที่สุด</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>ReLU (Rectified Linear Unit):</strong> ใช้งานง่ายและคำนวณเร็ว ช่วยลดปัญหา vanishing gradient ได้ดี นิยมใช้ใน Hidden Layer</li>
      <li><strong>Sigmoid:</strong> ใช้เปลี่ยนค่าให้อยู่ในช่วง [0, 1] เหมาะกับ Output Layer สำหรับ binary classification</li>
      <li><strong>Tanh (Hyperbolic Tangent):</strong> เปลี่ยนค่าให้อยู่ในช่วง [-1, 1] มีสมมาตรศูนย์ แต่ยังคงปัญหา gradient หายได้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">แนวทางการเลือกใช้งานในแต่ละ Layer</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h4 className="text-lg font-semibold mb-3">Input Layer</h4>
        <p className="text-sm">
          โดยทั่วไป Input Layer ไม่ใช้ Activation Function เพราะข้อมูลยังไม่ผ่านการแปลงเชิงคณิตศาสตร์ อย่างไรก็ตามการ Normalize ข้อมูลก่อนเข้าสู่ Layer นี้มีความสำคัญ เช่นใช้ StandardScaler หรือ MinMaxScaler
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h4 className="text-lg font-semibold mb-3">Hidden Layers</h4>
        <p className="text-sm">
          ใช้ ReLU เป็นค่าตั้งต้นในงานส่วนใหญ่ เนื่องจากคำนวณเร็วและช่วยให้ Network ลึกเรียนรู้ได้ดีกว่า Sigmoid หรือ Tanh โดยเฉพาะในงาน image recognition หรือ deep learning
        </p>
        <p className="text-sm mt-2">
          ในบางกรณี เช่น RNN อาจใช้ Tanh เพื่อช่วยให้ค่าผลลัพธ์คงอยู่ในช่วงที่สมมาตรและไม่ explode
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h4 className="text-lg font-semibold mb-3">Output Layer สำหรับ Classification</h4>
        <p className="text-sm">
          ถ้าเป็น Binary Classification → ใช้ Sigmoid เพื่อแปลงผลลัพธ์ให้อยู่ในช่วง [0, 1] ซึ่งสามารถตีความเป็นความน่าจะเป็นของแต่ละ class ได้
        </p>
        <p className="text-sm mt-2">
          ถ้าเป็น Multi-class Classification → ใช้ Softmax เพื่อกระจายค่าเป็นความน่าจะเป็นรวมกันเท่ากับ 1 สำหรับทุก class
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h4 className="text-lg font-semibold mb-3">Output Layer สำหรับ Regression</h4>
        <p className="text-sm">
          ไม่ควรใช้ Activation Function ใน Output Layer หากต้องการ output เป็นค่าต่อเนื่อง เช่นในการทำนายราคาบ้านหรือคะแนนสอบ เพราะต้องการให้โมเดล output ค่าตามจริงโดยไม่มีการจำกัดช่วง
        </p>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10 text-center">เปรียบเทียบเชิงภาพ</h3>
    <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

    <p>
      ReLU มีลักษณะเป็นเส้นตรงด้านบวกและตัดค่าเชิงลบให้เป็น 0 ซึ่งช่วยให้การเรียนรู้มีความเร็วขึ้น ในขณะที่ Sigmoid และ Tanh มีลักษณะโค้งที่จำกัดช่วงค่า output ซึ่งอาจนำไปสู่ปัญหา gradient vanishing หากใช้กับ layer ที่ลึกมาก
    </p>

    <h3 className="text-xl font-semibold mt-10">ตัวอย่างโค้ดการเลือก Activation ใน Keras</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto">
{`
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, input_shape=(10,), activation='relu'),  # Hidden Layer
    Dense(32, activation='relu'),                     # Hidden Layer
    Dense(1, activation='sigmoid')                    # Output for binary classification
])
`}
    </pre>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600 mt-6">
      <p className="font-medium mb-1">Insight:</p>
      <p className="text-sm">
        การเลือก Activation Function ไม่ควรใช้แบบเหมารวม ต้องพิจารณาจากลักษณะของข้อมูล ประเภทของปัญหา และผลกระทบเชิงคณิตศาสตร์ของแต่ละฟังก์ชันใน layer ต่าง ๆ อย่างรอบคอบ การเลือกที่ดีสามารถช่วยให้โมเดล converge ได้เร็วขึ้นและได้ผลลัพธ์ที่แม่นยำมากขึ้น
      </p>
    </div>
  </div>
</section>

<section id="draw" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-4 text-center">8. ลองวาด Neural Network ด้วยมือ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การวาดโครงสร้างของ Neural Network ด้วยมือเป็นกระบวนการเรียนรู้ที่ช่วยให้เข้าใจแก่นของโครงสร้างเชิงคณิตศาสตร์และแนวคิดการไหลของข้อมูลจากอินพุตไปสู่ผลลัพธ์ การทำเช่นนี้ไม่เพียงแต่ช่วยให้จำองค์ประกอบหลักของโมเดลได้เท่านั้น แต่ยังทำให้สามารถวิเคราะห์หรือออกแบบระบบได้ดีขึ้นเมื่อลงมือเขียนโค้ดจริง
    </p>

    <h3 className="text-xl font-semibold mt-6">โจทย์ที่ใช้ในการฝึกวาด</h3>
    <p>
      สมมติว่ากำลังออกแบบ Neural Network อย่างง่ายที่ใช้ในการทำนายคะแนนสอบของนักเรียน โดยใช้ข้อมูล 3 ตัวแปร ได้แก่:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>จำนวนชั่วโมงที่อ่านหนังสือต่อวัน</li>
      <li>จำนวนชั่วโมงการนอน</li>
      <li>จำนวนครั้งที่ฝึกทำข้อสอบในสัปดาห์</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6">โครงสร้างพื้นฐานของโมเดล</h3>
    <p>
      โครงสร้างของโมเดลอาจประกอบด้วย:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li>Input Layer: มี 3 Neurons สำหรับรับข้อมูลจาก Feature แต่ละตัว</li>
      <li>Hidden Layer 1: มี 4 Neurons ใช้ Activation Function เช่น ReLU</li>
      <li>Output Layer: มี 1 Neuron ใช้ Activation Function เป็น Linear เพื่อให้ผลลัพธ์เป็นค่าคะแนน</li>
    </ul>

    <h3 className="text-xl font-semibold mt-6">คำแนะนำสำหรับการวาดด้วยมือ</h3>
    <p>
      การวาดสามารถทำในกระดาษหรือใช้เครื่องมือออนไลน์ เช่น draw.io หรือ Excalidraw โดยควรทำตามขั้นตอนดังนี้:
    </p>
    <ol className="list-decimal pl-6 space-y-2">
      <li>เริ่มจากวาดวงกลม 3 วงแทน Input Neurons</li>
      <li>วาด Hidden Layer ถัดไป โดยวาง 4 Neurons และเชื่อมโยง Input → Hidden Layer</li>
      <li>วาด Output Layer เชื่อมต่อกับ Hidden Layer ทั้งหมด</li>
      <li>กำกับชื่อ Feature, Activation Function และค่าใด ๆ ที่ต้องการ</li>
    </ol>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
        <h4 className="text-lg font-semibold mb-3">คำถามเสริมระหว่างการวาด</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>เหตุใดจึงต้องมี Hidden Layer?</li>
          <li>ถ้ามี Hidden Layer มากขึ้น ผลจะเปลี่ยนแปลงอย่างไร?</li>
          <li>จำนวน Neurons ที่เพิ่มขึ้นอาจนำไปสู่ Overfitting หรือไม่?</li>
          <li>การเลือก Activation Function มีผลอย่างไรต่อพฤติกรรมของโมเดล?</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
        <h4 className="text-lg font-semibold mb-3">ตัวอย่างรูปแบบ Schema (แนะนำให้ลองวาด)</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>Input → Hidden Layer → Output</li>
          <li>3 ⟶ 4 ⟶ 1</li>
          <li>ใช้เส้นเชื่อมโยงจากทุก Node ไปยังทุก Node ชั้นถัดไป</li>
          <li>วงกลมแทน Neuron, เส้นแทน Weight</li>
        </ul>
      </div>
    </div>

    <div className="bg-green-50 dark:bg-green-800 text-green-900 dark:text-green-100 p-4 rounded-xl border-l-4 border-green-400 dark:border-green-600 mt-6">
      <p className="font-medium">Insight Box</p>
      <p className="text-sm">
        การฝึกวาด Neural Network ด้วยตนเองช่วยกระตุ้นความเข้าใจเกี่ยวกับการคำนวณภายในแต่ละชั้น และเสริมความเข้าใจในหลักการคณิตศาสตร์เบื้องหลังการเรียนรู้ของโมเดล โดยเฉพาะการเปลี่ยนแปลงค่าด้วย Weight และ Bias ตลอดจนการทำงานของ Activation Function
      </p>
    </div>
  </div>
</section>


<section id="library" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Neural Network ในไลบรารียอดนิยม</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img14} />
  </div>

  <div className="w-full overflow-x-auto">
  <div className="grid md:grid-cols-2 gap-6 min-w-[320px]">
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h3 className="text-xl font-bold mb-3">Keras: Sequential & Dense</h3>
      <p className="text-sm leading-relaxed mb-3">
        Keras เป็น high-level API ที่ใช้ในการสร้างและฝึก Neural Network ได้อย่างรวดเร็ว เหมาะกับผู้ที่ต้องการพัฒนาโมเดล Deep Learning ด้วยความยืดหยุ่นและความสะดวกในการใช้งาน โดยมักใช้ร่วมกับ TensorFlow เป็น backend
      </p>
      <ul className="list-disc pl-5 space-y-2 text-sm mb-4">
        <li><strong>Sequential:</strong> ใช้สำหรับสร้างโมเดลแบบ linear stack ของ layers</li>
        <li><strong>Dense:</strong> คือ fully connected layer ซึ่งใช้ matrix multiplication และ bias vector</li>
      </ul>
      <pre className="bg-gray-900 text-white text-xs p-4 rounded-xl overflow-auto mb-4">
{`from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, input_shape=(3,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`}
      </pre>
      <p className="text-sm leading-relaxed">
        โค้ดด้านบนเป็นตัวอย่างการสร้างโมเดลสำหรับการจำแนกข้อมูลแบบ binary โดยใช้ input ขนาด 3 ฟีเจอร์ และมี hidden layer 2 ชั้น พร้อมการ compile ที่ใช้ optimizer และ loss function ที่เหมาะสม
      </p>
    </div>

    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h3 className="text-xl font-bold mb-3">PyTorch: nn.Sequential & nn.Linear</h3>
      <p className="text-sm leading-relaxed mb-3">
        PyTorch เป็นไลบรารีที่ให้ความยืดหยุ่นสูงและเหมาะสำหรับการทำวิจัย โดยสามารถควบคุมโครงสร้างภายในได้ละเอียด โดยเฉพาะในการเขียน training loop ด้วยตนเอง ทั้งนี้ nn.Sequential และ nn.Linear เป็น abstraction ที่ใช้บ่อยในงานที่เน้นความง่าย
      </p>
      <ul className="list-disc pl-5 space-y-2 text-sm mb-4">
        <li><strong>nn.Sequential:</strong> เป็น container สำหรับ stack layers อย่างง่าย</li>
        <li><strong>nn.Linear:</strong> ใช้สร้าง fully connected layer โดยรับ input และ output dimension</li>
      </ul>
      <pre className="bg-gray-900 text-white text-xs p-4 rounded-xl overflow-auto mb-4">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`}
      </pre>
      <p className="text-sm leading-relaxed">
        โมเดลใน PyTorch นี้สร้างขึ้นโดยใช้โครงสร้าง Sequential ที่เรียงลำดับของ layer โดยมี activation function และ output layer ที่เหมาะกับปัญหา binary classification
      </p>
    </div>
  </div>
</div>


  <div className="mt-10 text-sm leading-relaxed space-y-4">
    <h3 className="text-xl font-semibold text-center mb-3">การเปรียบเทียบโดยรวม</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Keras</strong> เหมาะสำหรับ rapid prototyping และเริ่มต้นใช้งานได้ง่ายด้วย abstraction ที่ชัดเจน</li>
      <li><strong>PyTorch</strong> เหมาะสำหรับการปรับแต่งลึกระดับ research และควบคุม process ได้แบบ manual</li>
      <li>ทั้งสอง framework รองรับการใช้ GPU และสามารถขยายไปใช้ model ที่ซับซ้อนได้</li>
      <li>สามารถ integrate กับ TensorBoard (Keras) และ TorchMetrics (PyTorch) เพื่อการวิเคราะห์ performance</li>
    </ul>
    <p>
      การเลือกไลบรารีขึ้นอยู่กับเป้าหมายการพัฒนา หากเน้นการ deploy ที่รวดเร็วและ production-ready Keras คือทางเลือกที่ดี หากเน้นการทดลองหรือวิจัย PyTorch มีความยืดหยุ่นและตอบโจทย์การควบคุมภายในได้ดีกว่า
    </p>
  </div>
</section>

<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight: ทำไม Neural Networks ถึงปฏิวัติวงการ AI</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img15} />
  </div>

  <div className="grid md:grid-cols-2 gap-6 mb-10">
    <div className="bg-white dark:bg-gray-800 border rounded-xl p-6 shadow">
      <h3 className="text-lg font-semibold mb-3">Traditional Machine Learning</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ต้องการ Feature Engineering จากผู้เชี่ยวชาญเพื่อแยกแยะข้อมูล</li>
        <li>โมเดลอย่าง SVM, Decision Tree, Logistic Regression ทำงานได้ดีในปริมาณข้อมูลจำกัด</li>
        <li>มักจำกัดอยู่ที่การเรียนรู้แบบเส้นตรง หรือมีความสามารถในเชิงจำกัดในการจำลองฟังก์ชันที่ซับซ้อน</li>
        <li>ประสิทธิภาพลดลงอย่างรวดเร็วเมื่อมีการเพิ่ม Feature หรือมีข้อมูลที่ไม่เป็นเชิงเส้น</li>
        <li>ไม่สามารถเรียนรู้ Feature โดยอัตโนมัติจากข้อมูลดิบ เช่น รูปภาพหรือเสียง</li>
      </ul>
    </div>

    <div className="bg-white dark:bg-gray-800 border rounded-xl p-6 shadow">
      <h3 className="text-lg font-semibold mb-3">Deep Learning & Neural Networks</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>สามารถเรียนรู้ Feature โดยอัตโนมัติผ่านหลายชั้นของการแปลงข้อมูล (representation learning)</li>
        <li>ทำงานได้ดีกับข้อมูลขนาดใหญ่ที่ไม่มีโครงสร้าง เช่น รูปภาพ เสียง และข้อความ</li>
        <li>สามารถแทนที่ Feature Engineering ได้หลายกรณี</li>
        <li>มีความสามารถในการจำลองฟังก์ชันที่ซับซ้อนได้แบบไม่จำกัดด้วยจำนวนชั้น (Universal Approximator)</li>
        <li>สามารถขยายขนาดได้ผ่านการขนานการประมวลผล (GPU/TPU) และ Model Parallelism</li>
      </ul>
    </div>
  </div>

  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-5">
    <p>
      ความก้าวหน้าของ Neural Networks มาจากการเปลี่ยนแปลงเชิงโครงสร้าง ที่ทำให้ระบบสามารถเรียนรู้ลำดับของการแปลงข้อมูลได้อย่างลึกซึ้ง โดยไม่ต้องพึ่งพา Domain Knowledge จากมนุษย์ทั้งหมด การมี Layer ซ้อนกันจำนวนมากทำให้โมเดลสามารถเรียนรู้ลักษณะของข้อมูลจากพื้นฐาน (raw input) ไปยัง Feature ที่ซับซ้อนมากขึ้นโดยอัตโนมัติ
    </p>

    <p>
      ก่อนการมาถึงของ Deep Learning การประมวลผลภาพต้องใช้เทคนิคเชิงคณิตศาสตร์ เช่น Edge Detection, Histogram Equalization และ Handcrafted Feature Extraction ขณะที่ Neural Networks สามารถแทนสิ่งเหล่านั้นด้วย Convolution Layers ที่เรียนรู้ได้เองจากข้อมูลแบบ End-to-End
    </p>

    <p>
      ในด้านการประมวลผลภาษาแบบธรรมชาติ (NLP) ก่อนหน้านี้มีการใช้ n-gram และ Bag-of-Words ซึ่งมีข้อจำกัดในการรักษาโครงสร้างของภาษา แต่เมื่อมีการใช้ Recurrent Neural Networks (RNN) และต่อมากลายเป็น Transformer-based models อย่าง BERT และ GPT ความสามารถของโมเดลก็เพิ่มขึ้นในระดับที่สามารถเข้าใจบริบทและความสัมพันธ์ในประโยคได้อย่างแม่นยำ
    </p>

    <p>
      Neural Networks ยังสามารถใช้ได้กับปัญหาที่ไม่ได้เป็นเชิงเส้น เช่น การทำนายพฤติกรรมผู้ใช้ การจำแนกเสียงพูด หรือการควบคุมหุ่นยนต์ในโลกจริง ซึ่งแบบจำลองเชิงเส้นแบบเดิมไม่สามารถรองรับได้
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">ปัจจัยที่ทำให้ Neural Networks เป็น Game Changer</h3>
    <ul className="list-disc pl-6 space-y-2 text-sm">
      <li>ความสามารถในการเรียนรู้ Representation โดยไม่ต้องพึ่งการออกแบบ Feature ด้วยมือ</li>
      <li>รองรับการฝึกบนข้อมูลขนาดใหญ่ด้วย Hardware สมัยใหม่</li>
      <li>ขยายขนาดของโมเดลได้ตามความต้องการ (Scalability)</li>
      <li>มีโครงสร้างที่ยืดหยุ่นและสามารถปรับใช้ได้กับปัญหาหลากหลาย</li>
      <li>มี Framework ที่สนับสนุนการใช้งานจริง เช่น TensorFlow, PyTorch, Keras</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-2">ตัวอย่างการปฏิวัติในงานจริง</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-blue-50 dark:bg-blue-900 p-5 rounded-xl border-l-4 border-blue-400 dark:border-blue-600 text-sm">
        <h4 className="font-bold mb-2">Image Recognition</h4>
        <p>
          จากความแม่นยำ 70-80% ในยุคก่อน Deep Learning สู่โมเดลที่สามารถจำแนกวัตถุได้เหนือมนุษย์ในหลายกรณี เช่น ResNet, EfficientNet
        </p>
      </div>
      <div className="bg-green-50 dark:bg-green-900 p-5 rounded-xl border-l-4 border-green-400 dark:border-green-600 text-sm">
        <h4 className="font-bold mb-2">Language Models</h4>
        <p>
          จาก Rule-based System สู่ GPT, BERT ที่สามารถสร้างข้อความ ตอบคำถาม และแปลภาษาได้ในระดับใกล้เคียงมนุษย์
        </p>
      </div>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 mt-8">
      <h4 className="font-semibold mb-2">Insight:</h4>
      <p className="text-sm">
        Neural Networks ไม่ได้แค่เพิ่มความแม่นยำของโมเดล แต่เปลี่ยนวิธีคิดเกี่ยวกับ AI จากระบบที่กำหนดกฎโดยมนุษย์ สู่ระบบที่เรียนรู้จากข้อมูลและสร้างกฎของตัวเองได้ — นี่คือเหตุผลที่ Deep Learning ได้กลายเป็นแกนกลางของปัญญาประดิษฐ์สมัยใหม่
      </p>
    </div>
  </div>
</section>


        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day16 theme={theme} />
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
        <ScrollSpy_Ai_Day16 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day16_NeuralNetworkIntro;
