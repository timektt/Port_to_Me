import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day17 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day17";
import MiniQuiz_Day17 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day17";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day17_PerceptronMLP = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

   const img1 = cld.image("perceptron1").format("auto").quality("auto").resize(scale().width(650));
   const img2 = cld.image("perceptron2").format("auto").quality("auto").resize(scale().width(600));
   const img3 = cld.image("perceptron3").format("auto").quality("auto").resize(scale().width(600));
   const img4 = cld.image("perceptron4").format("auto").quality("auto").resize(scale().width(600));
   const img5 = cld.image("perceptron5").format("auto").quality("auto").resize(scale().width(600));
   const img6 = cld.image("perceptron6").format("auto").quality("auto").resize(scale().width(600));
   const img7 = cld.image("perceptron7").format("auto").quality("auto").resize(scale().width(600));
   const img8 = cld.image("perceptron8").format("auto").quality("auto").resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6 ">Day 17: Perceptron & Multi-Layer Perceptron</h1>
         <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

        <section id="overview" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: จากสมองสู่โครงข่ายประสาทเทียม</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img8} />
          </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      โครงข่ายประสาทเทียม (Artificial Neural Networks - ANN) ได้รับแรงบันดาลใจมาจากโครงสร้างทางชีววิทยาของระบบประสาทของมนุษย์ โดยเฉพาะเซลล์ประสาท (neurons) และรูปแบบการส่งสัญญาณไฟฟ้าระหว่างกัน จากการศึกษาในระดับมหาวิทยาลัยชั้นนำ เช่น MIT และ Stanford พบว่า Neuron หนึ่งตัวในสมองของมนุษย์มีศักยภาพในการรับข้อมูลจากเซลล์ประสาทอื่น ๆ หลายพันเซลล์และส่งต่อสัญญาณไปยังเซลล์ถัดไปผ่าน synapse ซึ่งเป็นจุดเชื่อมโยงระหว่าง neuron
    </p>

    <p>
      แนวคิดนี้ถูกนำมาปรับใช้ในระบบคอมพิวเตอร์ โดยพัฒนาขึ้นเป็นโครงข่ายประสาทเทียม ซึ่งเลียนแบบการทำงานแบบ distributed และ parallel processing ของสมอง การส่งสัญญาณในโครงข่ายประสาทเทียมเปรียบเสมือนกับการส่งศักย์ไฟฟ้าระหว่าง neuron ผ่านค่าที่เรียกว่า weight และการตัดสินใจของแต่ละ neuron จะขึ้นอยู่กับผลรวมของ input ที่ผ่าน weight และค่าคงที่ที่เรียกว่า bias
    </p>

    <p>
      จากงานวิจัยของ University of Oxford และรายงานขององค์กรระดับโลกอย่าง DeepMind ได้มีการกล่าวถึงประสิทธิภาพของ Neural Networks ในการประมวลผลข้อมูลที่ไม่เป็นเชิงเส้น เช่น ภาพ เสียง หรือภาษา ซึ่งถือเป็นการก้าวข้ามขีดจำกัดของแบบจำลองเชิงเส้นดั้งเดิม การทำงานของ ANN มีลักษณะเป็นระบบแบบ adaptive ที่สามารถปรับปรุงการเรียนรู้ได้ตลอดเวลา ผ่านกลไกการปรับ weight และ bias ระหว่างการฝึก (training)
    </p>

    <p>
      โครงข่ายประสาทเทียมแบบดั้งเดิมประกอบด้วยสามชั้นหลัก ได้แก่ ชั้นนำเข้าข้อมูล (Input Layer), ชั้นซ่อน (Hidden Layers) และชั้นผลลัพธ์ (Output Layer) ซึ่งในแต่ละชั้นประกอบด้วย neuron หลายตัวที่ทำหน้าที่แปลงข้อมูลไปยังระดับที่ซับซ้อนขึ้น โดยเฉพาะในชั้นซ่อนซึ่งถือเป็นหัวใจของการเรียนรู้รูปแบบที่ซับซ้อน
    </p>

    <p>
      ความสามารถในการเรียนรู้ของ ANN ไม่ได้ขึ้นอยู่กับการเพิ่มข้อมูลเพียงอย่างเดียว แต่ยังต้องการการออกแบบโครงสร้าง (architecture) ที่เหมาะสม เช่น จำนวนชั้นซ่อน จำนวน neuron ต่อชั้น และการเลือกใช้ activation function ที่เหมาะสม เช่น ReLU, Tanh หรือ Sigmoid ซึ่งมีผลต่อความสามารถในการแยกแยะรูปแบบและลดปัญหาทางคณิตศาสตร์ เช่น vanishing gradient
    </p>

    <p>
      จากการศึกษาเปรียบเทียบโดย Harvard University พบว่า ANN สามารถเรียนรู้และทำงานได้อย่างมีประสิทธิภาพเทียบเท่ากับการประมวลผลของสมองมนุษย์ในบางกรณี เช่น การจดจำภาพใบหน้า การรู้จำเสียงพูด หรือการจำแนกหมวดหมู่ข้อมูลที่มีความซับซ้อนสูง
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 text-blue-900 dark:text-blue-100 p-5 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <h3 className="font-semibold mb-2">ประวัติศาสตร์ของ Neural Network</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li><strong>1943:</strong> Warren McCulloch และ Walter Pitts เสนอโมเดล neuron เชิงคณิตศาสตร์ตัวแรก</li>
        <li><strong>1958:</strong> Frank Rosenblatt เสนอโมเดล Perceptron สำหรับการจำแนกข้อมูลเชิงเส้น</li>
        <li><strong>1986:</strong> Geoffrey Hinton, David Rumelhart และ Ronald Williams เผยแพร่แนวคิด Backpropagation ทำให้ Neural Network เรียนรู้ได้หลายชั้น</li>
        <li><strong>2012:</strong> AlexNet ชนะการแข่งขัน ImageNet ด้วยความแม่นยำเหนือกว่าระบบเดิมเกือบ 10% นำไปสู่ยุค Deep Learning</li>
      </ul>
    </div>

    <p>
      ปัจจุบัน Neural Networks กลายเป็นพื้นฐานสำคัญของระบบปัญญาประดิษฐ์ (AI) ที่ใช้ในหลายอุตสาหกรรม เช่น การแพทย์ (วินิจฉัยโรค), การเงิน (ตรวจจับการฉ้อโกง), พลังงาน (ระบบทำนายโหลดไฟฟ้า) และยานยนต์ (รถยนต์ไร้คนขับ) โดยเฉพาะเทคโนโลยีอย่าง Deep Learning และ Reinforcement Learning ล้วนพัฒนาขึ้นจากรากฐานของ ANN ทั้งสิ้น
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h3 className="font-semibold mb-2">Insight Box</h3>
      <p className="text-sm">
        โครงข่ายประสาทเทียมไม่ได้เป็นเพียงการจำลองสมอง แต่คือการออกแบบระบบที่สามารถเรียนรู้ ปรับตัว และคาดการณ์ได้จากข้อมูลดิบแบบ end-to-end อย่างมีประสิทธิภาพ ด้วยการพัฒนาโครงสร้างทางคณิตศาสตร์และการฝึกฝนที่มีหลักการทางวิทยาศาสตร์รองรับอย่างชัดเจน
      </p>
    </div>
  </div>
</section>

<section id="perceptron" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Perceptron คืออะไร?</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img2} />
          </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Perceptron คือโมเดลพื้นฐานของโครงข่ายประสาทเทียม (Neural Network) ที่ถูกเสนอโดย Frank Rosenblatt ในปี ค.ศ. 1958 ข้อมูลจาก Cornell Aeronautical Laboratory ชี้ให้เห็นว่าโมเดลนี้ได้รับแรงบันดาลใจจากการทำงานของเซลล์ประสาทในสมองมนุษย์ โดยออกแบบให้สามารถเรียนรู้และแยกแยะข้อมูลที่มีลักษณะแตกต่างกันผ่านกระบวนการปรับน้ำหนัก
    </p>

    <p>
      Perceptron เป็นโมเดลการจำแนกเชิงเส้น (Linear Classifier) ซึ่งประกอบด้วยหน่วยประมวลผลหลักเพียงหนึ่งตัว (Single Layer Perceptron) โดยมีโครงสร้างประกอบด้วย:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Input Vector (x):</strong> ข้อมูลที่ป้อนเข้ามาในรูปแบบเวกเตอร์</li>
      <li><strong>Weight Vector (w):</strong> ค่าน้ำหนักที่เชื่อมโยงกับแต่ละ input</li>
      <li><strong>Bias (b):</strong> ค่าคงที่ที่เพิ่มความยืดหยุ่นให้กับการตัดสินใจ</li>
      <li><strong>Activation Function:</strong> ฟังก์ชันที่ตัดสินว่า output จะเป็น 1 หรือ 0</li>
    </ul>

    <p>
      การคำนวณของ Perceptron มีลักษณะเป็นการรวมเชิงเส้นของ input และ weight แล้วนำผลลัพธ์ที่ได้มาเปรียบเทียบกับ threshold ด้วย activation function ที่เรียกว่า <strong>Step Function</strong> หรือ <strong>Heaviside function</strong> ตามสมการ:
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <code className="text-sm block">
        z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b<br />
        y = 1 if z ≥ 0 else 0
      </code>
    </div>

    <p>
      หากค่า z มีค่ามากกว่าหรือเท่ากับ 0 โมเดลจะให้ค่าผลลัพธ์เป็น 1 (เช่น คลาสบวก) หากไม่ถึงก็จะให้ผลลัพธ์เป็น 0 การตัดสินใจนี้สามารถใช้ในการจำแนกข้อมูลแบบสองกลุ่ม (binary classification) เช่น:
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h4 className="text-lg font-medium mb-2">ตัวอย่างที่ Perceptron ใช้ได้ผล:</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>การแยกข้อมูล AND (0 AND 0 = 0, 1 AND 1 = 1)</li>
          <li>การแยกข้อมูล OR (0 OR 1 = 1, 0 OR 0 = 0)</li>
          <li>การตรวจจับภาพเบื้องต้นที่สามารถแยกเส้นตรงได้</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-red-300 dark:border-red-700">
        <h4 className="text-lg font-medium mb-2">ข้อจำกัดของ Perceptron:</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ไม่สามารถแยกข้อมูลที่ไม่เป็นเชิงเส้น เช่น ปัญหา XOR</li>
          <li>ไม่มีความสามารถในการเรียนรู้ลักษณะซับซ้อนหรือ hierarchical feature</li>
          <li>การตัดสินใจขึ้นอยู่กับเส้นตรงเพียงเส้นเดียวในพื้นที่ข้อมูล</li>
        </ul>
      </div>
    </div>

    <p>
      ตัวอย่างคลาสสิกของข้อจำกัดของ Perceptron ที่ปรากฏในการศึกษาของ Marvin Minsky และ Seymour Papert ในหนังสือ Perceptrons ปี 1969 คือ XOR Problem ซึ่งไม่สามารถแยกได้ด้วยเส้นตรง เพราะข้อมูลถูกกระจายอยู่ในพื้นที่ที่ไม่มีเส้นตรงใดสามารถแยกได้อย่างถูกต้อง
    </p>

    <h3 className="text-xl font-semibold mt-10">การเรียนรู้ของ Perceptron</h3>
    <p>
      Perceptron ใช้หลักการ <strong>Perceptron Learning Rule</strong> ในการอัปเดตน้ำหนักระหว่างการฝึก โดยปรับ weight ตาม error ที่เกิดขึ้นในแต่ละรอบการเทรน (epoch) ซึ่งสามารถแสดงได้ดังนี้:
    </p>

    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`# Pseudocode
initialize weights and bias = 0
for each data point (x, y):
    compute output y_hat = step(w • x + b)
    compute error = y - y_hat
    update weights: w = w + learning_rate * error * x
    update bias: b = b + learning_rate * error`}
    </pre>

    <p>
      การอัปเดตน้ำหนักนี้จะทำไปเรื่อย ๆ จนกว่าโมเดลจะสามารถแยกข้อมูลได้ถูกต้องทั้งหมด หรือถึงจำนวนรอบที่กำหนดไว้ จุดเด่นคือสามารถ converge ได้ในกรณีที่ข้อมูลสามารถแยกด้วยเส้นตรง
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <p className="text-sm">
        แม้ Perceptron จะเป็นเพียงโมเดลพื้นฐาน แต่ถือว่าเป็นจุดเริ่มต้นสำคัญของการพัฒนา Neural Networks ทั้งหมด การเข้าใจ Perceptron อย่างลึกซึ้งจึงเป็นกุญแจสำคัญในการต่อยอดสู่โมเดลที่ซับซ้อนกว่า เช่น Multi-Layer Perceptron, CNN และ Transformer
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">สรุป Perceptron</h3>
    <ul className="list-disc pl-6 space-y-2 text-sm">
      <li>เป็น binary classifier ที่ใช้การตัดสินใจผ่าน step function</li>
      <li>สามารถเรียนรู้ได้จากข้อมูลที่แยกได้แบบเชิงเส้นเท่านั้น</li>
      <li>โครงสร้างเรียบง่าย เหมาะสำหรับเรียนรู้เบื้องต้น</li>
      <li>ข้อจำกัดสำคัญคือไม่สามารถแยกข้อมูล XOR ได้</li>
      <li>เป็นรากฐานของโครงข่ายแบบหลายชั้น (MLP) ที่จะกล่าวถึงใน section ถัดไป</li>
    </ul>
  </div>
</section>


<section id="mlp" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Multi-Layer Perceptron (MLP)</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img3} />
          </div>
    <p>
      Multi-Layer Perceptron (MLP) เป็นโครงสร้างพื้นฐานของ Artificial Neural Networks ที่ได้รับการพัฒนาเพื่อแก้ไขข้อจำกัดของ Perceptron แบบดั้งเดิมในการจำแนกข้อมูลที่ไม่สามารถแยกได้เชิงเส้น (non-linearly separable data) เช่น ปัญหา XOR ซึ่งไม่สามารถแก้ไขได้ด้วย Single-Layer Perceptron ตามข้อพิสูจน์ของ Marvin Minsky และ Seymour Papert ในปี 1969
    </p>

    <p>
      MLP ประกอบด้วยสามส่วนหลัก ได้แก่ Input Layer, Hidden Layer(s) และ Output Layer โดยแต่ละชั้นประกอบด้วยหน่วยประมวลผลพื้นฐานที่เรียกว่า Neuron ซึ่งมีการเชื่อมต่อแบบเต็ม (Fully Connected) ผ่านน้ำหนัก (weights) และค่าคงที่ (biases) ซึ่งเป็นพารามิเตอร์ที่สามารถเรียนรู้ได้ระหว่างการฝึกฝนโมเดล
    </p>

    <h3 className="text-xl font-semibold">สถาปัตยกรรมของ MLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Input Layer:</strong> รับข้อมูลดิบเข้าสู่ระบบ โดยแต่ละ Neuron จะรับหนึ่ง Feature จากข้อมูล เช่น พิกเซลของภาพ หรือค่าจากเซ็นเซอร์
      </li>
      <li>
        <strong>Hidden Layers:</strong> มีตั้งแต่หนึ่งชั้นขึ้นไป ทำหน้าที่แปลงข้อมูลเชิงเส้นเป็นการแทนค่าที่ซับซ้อนมากขึ้นผ่านฟังก์ชัน Activation เพื่อให้สามารถจำลองฟังก์ชันไม่เชิงเส้นได้
      </li>
      <li>
        <strong>Output Layer:</strong> ส่งผลลัพธ์สุดท้ายของโมเดล เช่น การจำแนกประเภท หรือการพยากรณ์ค่าต่อเนื่อง โดยใช้ฟังก์ชัน Activation ที่เหมาะสมกับประเภทของปัญหา
      </li>
    </ul>

    <h3 className="text-xl font-semibold">ฟังก์ชัน Activation ที่ใช้ใน MLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>ReLU (Rectified Linear Unit):</strong> เพิ่มความเร็วในการฝึกและลดปัญหา Vanishing Gradient ทำให้เป็นฟังก์ชันที่นิยมใช้ใน Hidden Layers</li>
      <li><strong>Sigmoid:</strong> ใช้ในงาน Binary Classification เนื่องจากเปลี่ยนค่าให้อยู่ในช่วง [0,1]</li>
      <li><strong>Tanh (Hyperbolic Tangent):</strong> เหมาะสำหรับปัญหาที่ต้องการค่าผลลัพธ์สมมาตรระหว่าง -1 ถึง 1</li>
    </ul>

    <h3 className="text-xl font-semibold">พารามิเตอร์สำคัญใน MLP</h3>
    <p>
      แต่ละการเชื่อมต่อระหว่าง Neurons ประกอบด้วย Weight (W) และ Bias (b) ซึ่งมีการเรียนรู้ผ่านกระบวนการ Backpropagation และ Optimization เช่น Gradient Descent โดยอาศัยการคำนวณ Loss ที่บ่งชี้ว่าผลลัพธ์ของโมเดลห่างจากค่าจริงเพียงใด
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <p className="text-sm">
        <strong>สูตรหลักของแต่ละ Neuron:</strong> Z = W \u00d7 X + b, A = Activation(Z)
      </p>
    </div>

    <h3 className="text-xl font-semibold">ความแตกต่างระหว่าง Perceptron กับ MLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>Perceptron มีเพียงหนึ่งชั้นและสามารถจัดการได้เฉพาะข้อมูลที่แยกได้เชิงเส้น</li>
      <li>MLP มีหลายชั้น (Hidden Layers) สามารถจำลองฟังก์ชันที่ซับซ้อนได้ เช่น การจดจำลายมือ หรือการจำแนกภาพ</li>
      <li>MLP ใช้ Activation Function ที่ไม่เชิงเส้นเพื่อเพิ่มความสามารถในการเรียนรู้</li>
    </ul>

    <h3 className="text-xl font-semibold">การใช้งานจริงของ MLP</h3>
    <p>
      MLP ถูกนำมาใช้ในหลายสาขา เช่น การรู้จำเสียงพูด (Speech Recognition), การวิเคราะห์ภาพถ่ายทางการแพทย์, การคัดกรองสินเชื่อทางการเงิน และการคาดการณ์แนวโน้มในตลาดหุ้น โดยเฉพาะงานที่ข้อมูลเป็นโครงสร้างแบบตาราง (Structured Data)
    </p>

    <h3 className="text-xl font-semibold">จุดเด่นและข้อจำกัดของ MLP</h3>
    <div className="grid md:grid-cols-2 gap-6 mt-6">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">ข้อดีของ MLP</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>สามารถเรียนรู้รูปแบบที่ไม่เชิงเส้นได้</li>
          <li>รองรับงานทั้ง Classification และ Regression</li>
          <li>ใช้งานง่ายผ่านไลบรารีเช่น TensorFlow และ PyTorch</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">ข้อจำกัดของ MLP</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ต้องการพลังประมวลผลสูงเมื่อข้อมูลมีมิติสูง</li>
          <li>เสี่ยงต่อ Overfitting หากไม่มีการ Regularization</li>
          <li>ไม่เหมาะกับข้อมูลที่มีโครงสร้างพื้นที่ เช่น ภาพ (ควรใช้ CNN แทน)</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Insight จากการวิจัยสถาบันชั้นนำ</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <p className="text-sm">
        งานวิจัยจาก Stanford และ MIT ยืนยันว่า MLP เป็นรากฐานสำคัญของ Neural Networks ที่ลึกกว่า (Deep Neural Networks) โดยการเพิ่มจำนวน Hidden Layers และ Units อย่างมีเหตุผลสามารถขยายขีดความสามารถของโมเดลในการเรียนรู้ฟังก์ชันที่ซับซ้อนในรูปแบบข้อมูลที่หลากหลาย
      </p>
    </div>
  </div>
</section>

<section id="mlplearn" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การเรียนรู้ใน MLP</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img4} />
          </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      กระบวนการเรียนรู้ใน Multi-Layer Perceptron (MLP) ประกอบด้วยขั้นตอนหลักที่ช่วยให้โมเดลสามารถเข้าใจและเรียนรู้จากข้อมูลได้อย่างมีประสิทธิภาพ ได้แก่ การส่งข้อมูลแบบ Forward Propagation การคำนวณค่า Loss การถ่ายทอดข้อผิดพลาดด้วย Backpropagation และการอัปเดตน้ำหนักผ่านอัลกอริทึม Gradient Descent
    </p>

    <h3 className="text-xl font-semibold">การส่งข้อมูลแบบ Forward Propagation</h3>
    <p>
      ในขั้นตอนนี้ ข้อมูลจาก Input Layer จะถูกส่งต่อผ่าน Hidden Layers ไปยัง Output Layer โดยแต่ละชั้นจะคำนวณค่าผลลัพธ์จากการคูณเวกเตอร์ของอินพุตกับน้ำหนัก (Weight Matrix) และบวกค่าคงที่ (Bias Vector) จากนั้นจึงนำไปผ่าน Activation Function เพื่อสร้างค่าผลลัพธ์ที่ไม่เชิงเส้น
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <p className="text-sm">
        <strong>สูตร:</strong> Z = W × X + b, A = Activation(Z)
      </p>
    </div>

    <h3 className="text-xl font-semibold">การคำนวณค่า Loss</h3>
    <p>
      เมื่อโมเดลทำนายผลลัพธ์จากข้อมูลอินพุตแล้ว ค่าที่ได้จะถูกนำไปเปรียบเทียบกับค่าจริง (Ground Truth) ด้วยฟังก์ชัน Loss เพื่อตรวจสอบความคลาดเคลื่อน โดย Loss Function ที่นิยมใช้ ได้แก่:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Mean Squared Error (MSE):</strong> ใช้กับปัญหา Regression โดยวัดค่าความต่างระหว่างค่าจริงและค่าที่ทำนาย ยกกำลังสอง</li>
      <li><strong>Cross-Entropy Loss:</strong> ใช้กับปัญหา Classification โดยเฉพาะ Binary และ Multi-class Classification</li>
    </ul>

    <h3 className="text-xl font-semibold">การอัปเดตน้ำหนักด้วย Backpropagation</h3>
    <p>
      Backpropagation คือกระบวนการถ่ายทอดข้อผิดพลาดย้อนกลับจาก Output Layer ไปยัง Hidden Layer แต่ละชั้น โดยอาศัยหลักการ Chain Rule ใน Calculus เพื่อคำนวณ Gradient ของค่าพารามิเตอร์ (Weights และ Biases) ที่ส่งผลต่อค่า Loss มากที่สุด
    </p>
    <p>
      การคำนวณ Gradient อย่างถูกต้องในแต่ละชั้นของโมเดลมีความสำคัญต่อการอัปเดตพารามิเตอร์ให้มีประสิทธิภาพสูงสุด โดย Framework เช่น TensorFlow และ PyTorch มีระบบ Auto-differentiation ที่ช่วยทำงานนี้โดยอัตโนมัติ
    </p>

    <h3 className="text-xl font-semibold">การใช้ Gradient Descent กับ Network หลายชั้น</h3>
    <p>
      เมื่อทราบค่าของ Gradient แล้ว ขั้นตอนถัดไปคือการปรับค่าพารามิเตอร์ของโมเดลผ่านอัลกอริทึม Gradient Descent โดยปรับน้ำหนักในทิศทางตรงกันข้ามกับ Gradient เพื่อลดค่า Loss ในรอบถัดไป การฝึกโมเดลหลายชั้นจำเป็นต้องใช้ Learning Rate ที่เหมาะสม เพื่อป้องกันปัญหาเช่น Vanishing Gradient หรือ Exploding Gradient
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">สิ่งที่ต้องระวังในการฝึก MLP</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>Learning Rate สูงเกินไปอาจทำให้โมเดลไม่เสถียร</li>
          <li>Gradient หายหรือระเบิดในเครือข่ายที่ลึก</li>
          <li>Overfitting เมื่อโมเดลซับซ้อนเกินไป</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">แนวทางแก้ไข</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ใช้ Regularization เช่น Dropout, L2</li>
          <li>ใช้ Batch Normalization เพื่อรักษาเสถียรภาพ</li>
          <li>ใช้ Optimizer ที่ทันสมัย เช่น Adam หรือ RMSprop</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Insight จากการวิจัย</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <p className="text-sm">
        งานวิจัยจาก University of Toronto และ DeepMind พบว่าการฝึกโมเดลด้วย Backpropagation ร่วมกับเทคนิคเช่น Adaptive Gradient และ Early Stopping มีผลอย่างมากในการเพิ่มประสิทธิภาพและลดความเสี่ยงจาก Overfitting ใน MLP หลายชั้น
      </p>
    </div>
  </div>
</section>



<section id="real-world" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. การฝึก MLP บนงานจริง</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img5} />
          </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      การนำ Multi-Layer Perceptron (MLP) ไปใช้กับปัญหาในโลกจริงเป็นหนึ่งในก้าวสำคัญที่ผลักดันให้โครงข่ายประสาทเทียมกลายเป็นเครื่องมือมาตรฐานของ Machine Learning สมัยใหม่ MLP ถูกใช้งานหลากหลายในงานที่ข้อมูลมีรูปแบบเป็นตาราง (tabular data), ข้อมูลแบบ vector หรือข้อมูล structured ทั่วไป เช่น งานจำแนกประเภท (classification), การทำนายค่า (regression) และระบบแนะนำ (recommendation systems)
    </p>

    <h3 className="text-xl font-semibold mt-8">การประยุกต์ใช้งาน: ตัวอย่าง MNIST</h3>
    <p>
      ฐานข้อมูล MNIST (Modified National Institute of Standards and Technology) เป็นชุดข้อมูลที่ประกอบด้วยภาพตัวเลข 28x28 พิกเซล รวม 60,000 ภาพสำหรับฝึกฝน และ 10,000 ภาพสำหรับทดสอบ เป้าหมายคือการแยกประเภทตัวเลข 0–9 โดยใช้ MLP
    </p>

    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto">
{`from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and preprocess
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)`}
    </pre>

    <p>
      โค้ดตัวอย่างนี้ใช้ไลบรารี Keras เพื่อสร้าง MLP ที่สามารถเรียนรู้จากภาพดิจิไบน์ขนาด 784 พิกเซล และจำแนกออกเป็น 10 ประเภทได้อย่างแม่นยำ ความเรียบง่ายของ MLP ทำให้เป็นจุดเริ่มต้นที่ดีในงานที่ยังไม่ซับซ้อนเกินไป
    </p>

    <h3 className="text-xl font-semibold mt-8">จำนวน Hidden Layer และ Units ที่เหมาะสม</h3>
    <p>
      จากคำแนะนำของ Andrew Ng (Stanford University) และ Deep Learning Specialization โดย Deeplearning.ai การเลือกจำนวน Hidden Layers และจำนวน Neurons ต่อชั้นควรพิจารณาจากความซับซ้อนของข้อมูล หากข้อมูลมีความเป็นเชิงเส้นต่ำ อาจใช้ Hidden Layer เพียง 1-2 ชั้นก็เพียงพอ แต่หากต้องการให้โมเดลแยกแยะ pattern ที่ซับซ้อนมากขึ้น ควรเพิ่มความลึกและความกว้างของเครือข่าย
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>เริ่มจากโครงสร้างง่าย เช่น 1 Hidden Layer ที่มี 64 Units</li>
      <li>ใช้ cross-validation เพื่อปรับแต่งจำนวน Layer และ Units</li>
      <li>หลีกเลี่ยงการออกแบบให้ใหญ่เกินไปซึ่งอาจนำไปสู่ Overfitting</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">การจัดการปัญหา Overfitting</h3>
    <p>
      MLP ที่มีพารามิเตอร์จำนวนมากมีแนวโน้มจะจดจำข้อมูลการฝึกฝนมากเกินไป โดยไม่สามารถประยุกต์ใช้กับข้อมูลใหม่ได้ดี วิธีป้องกันที่แนะนำโดย Google Research และ MIT AI Lab ได้แก่:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Dropout:</strong> การสุ่มปิดการทำงานของ Neuron บางตัวระหว่างการฝึก</li>
      <li><strong>Early Stopping:</strong> หยุดการฝึกหาก Validation Loss เริ่มเพิ่มขึ้น</li>
      <li><strong>L2 Regularization:</strong> เพิ่มโทษใน Loss Function หากพารามิเตอร์มีค่าสูงเกิน</li>
      <li><strong>Data Augmentation:</strong> ขยายข้อมูลฝึกให้หลากหลายมากขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">Vanishing Gradient ใน MLP</h3>
    <p>
      การใช้ Activation Function ที่มี Gradient แคบ เช่น Sigmoid หรือ Tanh ในชั้นลึก ๆ อาจทำให้เกิดปัญหา Vanishing Gradient ซึ่งส่งผลให้การอัปเดตน้ำหนักใน Layer ล่าง ๆ แทบไม่มีผล วิธีแก้คือ:
    </p>

    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ ReLU หรือ LeakyReLU แทน Sigmoid/Tanh</li>
      <li>ใช้ Batch Normalization เพื่อลดความผันผวนของ Gradient</li>
      <li>เริ่มต้นน้ำหนักด้วย He Initialization หรือ Xavier Initialization</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">การแสดง Decision Boundary</h3>
    <p>
      เพื่อเข้าใจว่า MLP แยกข้อมูลอย่างไร การแสดง Decision Boundary ช่วยให้เห็นภาพความสามารถของโมเดลในการแบ่งขอบเขตระหว่างกลุ่มข้อมูล เช่น ในชุดข้อมูล XOR, Spiral หรือข้อมูล 2 มิติที่กำหนดเอง โดยใช้ Matplotlib และ sklearn สร้างภาพได้ดังนี้:
    </p>

    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-auto">
{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500)
mlp.fit(X_train, y_train)

# Plot decision boundary
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("MLP Decision Boundary")
plt.show()`}
    </pre>

    <p>
      การวิเคราะห์ Decision Boundary เป็นวิธีที่มีประสิทธิภาพในการตรวจสอบว่าโมเดลสามารถเข้าใจโครงสร้างข้อมูลเชิงพื้นที่และการแยกกลุ่มได้ดีเพียงใด
    </p>

    <div className="bg-green-50 dark:bg-green-900 text-green-900 dark:text-green-100 p-5 rounded-xl border-l-4 border-green-400 dark:border-green-600">
      <p className="font-medium">Insight Box</p>
      <p className="text-sm">
        แม้ MLP จะไม่ใช่สถาปัตยกรรมที่ทันสมัยที่สุดใน Deep Learning แต่ยังคงมีบทบาทสำคัญในงานที่ข้อมูลมีโครงสร้างแน่นอน การทำความเข้าใจการทำงานของ MLP ในงานจริงช่วยสร้างรากฐานสำหรับการพัฒนาโมเดลที่ซับซ้อนกว่า เช่น CNN, RNN และ Transformer
      </p>
    </div>
  </div>
</section>

<section id="advantages" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. ข้อดีและข้อจำกัดของ MLP</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img6} />
          </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Multi-Layer Perceptron (MLP) เป็นหนึ่งในสถาปัตยกรรมพื้นฐานของ Neural Network ที่มีบทบาทสำคัญในงานด้าน AI และ Machine Learning มานาน โดยเฉพาะในยุคเริ่มต้นของ Deep Learning ถึงแม้จะมีโครงสร้างไม่ซับซ้อนเท่ากับสถาปัตยกรรมเฉพาะทาง เช่น CNN หรือ Transformer แต่ MLP ก็ยังมีคุณสมบัติหลายประการที่ทำให้เหมาะสำหรับงานประเภทต่าง ๆ โดยเฉพาะข้อมูลที่เป็น structured หรือ tabular data
    </p>

    <h3 className="text-xl font-semibold mt-8">ข้อดีของ MLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Universal Function Approximator:</strong> ตามทฤษฎีของ Hornik et al. (1989) ระบุว่า MLP ที่มี Hidden Layer อย่างน้อยหนึ่งชั้นที่กว้างเพียงพอ สามารถประมาณค่าฟังก์ชันไม่เชิงเส้นได้ทุกชนิดภายใต้เงื่อนไขที่เหมาะสม
      </li>
      <li>
        <strong>ความยืดหยุ่นในการใช้งาน:</strong> สามารถใช้กับงาน Classification, Regression, Time Series, และ Signal Processing ได้หลากหลาย
      </li>
      <li>
        <strong>เหมาะกับข้อมูลแบบ Structured:</strong> MLP เหมาะสำหรับข้อมูลตาราง เช่น ฐานข้อมูลจากองค์กร ธนาคาร การแพทย์ หรือธุรกิจ ซึ่งข้อมูลไม่อยู่ในรูปภาพหรือข้อความยาว
      </li>
      <li>
        <strong>ฝึกฝนง่ายด้วย Framework ทั่วไป:</strong> ไลบรารีอย่าง TensorFlow, Keras และ PyTorch สนับสนุนการสร้าง MLP อย่างมีประสิทธิภาพและยืดหยุ่น
      </li>
      <li>
        <strong>สามารถใช้ Transfer Learning:</strong> ในบางงาน MLP สามารถต่อยอดจากการเรียนรู้ฟีเจอร์ที่ได้จากโมเดลอื่น เช่น CNN หรือ Encoder จาก Transformer ได้
      </li>
      <li>
        <strong>มีความเข้าใจง่าย:</strong> โครงสร้างตรงไปตรงมา ทำให้เหมาะสำหรับการเรียนรู้เบื้องต้นและการสอนพื้นฐานเกี่ยวกับ Neural Networks
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">ข้อจำกัดของ MLP</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>ไม่สามารถเรียนรู้โครงสร้างเชิงพื้นที่ได้ดี:</strong> เช่น รูปภาพหรือวิดีโอที่มี spatial structure จำเป็นต้องใช้ CNN หรือ Attention-based Network แทน
      </li>
      <li>
        <strong>จำนวนพารามิเตอร์สูง:</strong> การเชื่อมต่อแบบ Fully Connected ทำให้มีพารามิเตอร์มาก โดยเฉพาะใน input ที่มีมิติสูง ส่งผลต่อเวลาในการฝึกและหน่วยความจำ
      </li>
      <li>
        <strong>มีแนวโน้ม Overfitting:</strong> โดยเฉพาะเมื่อมีจำนวน Layer หรือ Neuron มากเกินไป และไม่มีการใช้เทคนิค regularization เช่น Dropout หรือ L2 Penalty
      </li>
      <li>
        <strong>ไม่เหมาะกับข้อมูล sequence หรือ temporal:</strong> ข้อมูลที่มีลำดับเวลา เช่น ข้อความหรือ series จำเป็นต้องใช้ RNN, LSTM หรือ Transformer ที่ออกแบบมาสำหรับงานประเภทนี้
      </li>
      <li>
        <strong>Training Efficiency ต่ำในบางกรณี:</strong> MLP อาจเรียนรู้ได้ช้าหรือประสิทธิภาพต่ำในงานที่มีข้อมูลหลากหลายมิติหรือมีโครงสร้างซับซ้อนโดยธรรมชาติ
      </li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">เมื่อไรควรใช้ MLP?</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เมื่อข้อมูลเป็นตาราง เช่น รายการลูกค้า ฐานข้อมูลการเงิน ข้อมูลทางการแพทย์ เป็นต้น</li>
      <li>เมื่อปัญหามีมิติของ input ไม่สูงเกินไป และไม่ต้องการการแยกโครงสร้างเชิงพื้นที่</li>
      <li>เมื่อเน้นความเรียบง่าย และต้องการโมเดลที่อธิบายได้ง่าย</li>
      <li>เมื่อใช้เป็น baseline ก่อนจะเปรียบเทียบกับโมเดลอื่นที่ซับซ้อนกว่า</li>
    </ul>

    <div className="grid md:grid-cols-2 gap-6 mt-10">
      <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
        <h4 className="text-lg font-semibold mb-3">ตัวอย่างงานที่เหมาะกับ MLP</h4>
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>การจำแนกกลุ่มลูกค้าตามพฤติกรรม</li>
          <li>การพยากรณ์ความเสี่ยงทางการเงิน</li>
          <li>การตรวจจับการทุจริตจากข้อมูลเชิงพฤติกรรม</li>
          <li>การวิเคราะห์ข้อมูลด้านสุขภาพ (Health Records)</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 p-6 rounded-xl shadow">
        <h4 className="text-lg font-semibold mb-3">กรณีที่ควรหลีกเลี่ยงการใช้ MLP</h4>
        <ul className="list-disc pl-5 space-y-2 text-sm">
          <li>งานด้าน Computer Vision (ควรใช้ CNN)</li>
          <li>งานด้าน NLP เช่น Machine Translation (ควรใช้ Transformer)</li>
          <li>ข้อมูลเสียงและสัญญาณที่มี Time-dependency (ควรใช้ RNN/LSTM)</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <p className="text-sm">
        MLP อาจไม่ใช่ทางเลือกที่ดีที่สุดในยุคของสถาปัตยกรรม Deep Learning ที่ทันสมัย แต่ยังคงเป็นจุดเริ่มต้นสำคัญของระบบ AI ที่ทันสมัยและสามารถทำงานได้ดีในบริบทที่เหมาะสม ความเข้าใจพื้นฐานเกี่ยวกับ MLP จึงเป็นกุญแจสำคัญในการเข้าใจสถาปัตยกรรมที่ซับซ้อนในลำดับถัดไป เช่น Deep CNN หรือ Transformer
      </p>
    </div>
  </div>
</section>


<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Insight Box</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img7} />
          </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h3 className="text-xl font-semibold mb-4">การเปรียบเทียบ Perceptron และ Logistic Regression</h3>
      <p>
        Perceptron และ Logistic Regression ต่างเป็นโมเดลที่มีพื้นฐานทางคณิตศาสตร์ในการจำแนกข้อมูลแบบสองคลาส (Binary Classification) แต่มีจุดแตกต่างที่สำคัญในเชิงทฤษฎีและการนำไปใช้จริง
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          Perceptron ใช้หลักการ Threshold-based Decision โดยการเปรียบค่าผลรวมแบบเชิงเส้นกับเกณฑ์ตัดสินใจ (Threshold) และออกผลลัพธ์แบบ Hard Decision (0 หรือ 1)
        </li>
        <li>
          Logistic Regression ใช้ฟังก์ชัน Sigmoid แปลงผลรวมเชิงเส้นเป็นค่าความน่าจะเป็นระหว่าง 0 ถึง 1 ก่อนจะนำไปตัดสินใจ จึงสามารถให้การตีความได้ทางสถิติและมีความนุ่มนวลมากกว่า
        </li>
        <li>
          ในเชิง Optimization, Perceptron ใช้วิธีการ Perceptron Learning Rule ในขณะที่ Logistic Regression ใช้ Maximum Likelihood Estimation (MLE) เพื่อหาค่า Optimal Parameters
        </li>
        <li>
          Logistic Regression มีการวัดความผิดพลาดด้วย Cross Entropy Loss ซึ่งเป็นฟังก์ชันต่อเนื่อง ทำให้สามารถใช้อัลกอริทึม Gradient Descent ได้อย่างมีประสิทธิภาพ
        </li>
      </ul>
    </div>

    <div className="bg-green-50 dark:bg-green-900 text-green-900 dark:text-green-100 p-6 rounded-xl border-l-4 border-green-400 dark:border-green-600">
      <h3 className="text-xl font-semibold mb-4">MLP คือรากฐานของ Deep Neural Networks</h3>
      <p>
        Multi-Layer Perceptron (MLP) คือรูปแบบที่วางรากฐานสำคัญให้กับสถาปัตยกรรมของ Neural Networks สมัยใหม่ โดยมีคุณสมบัติที่สถาบันชั้นนำ เช่น Stanford University และ Massachusetts Institute of Technology (MIT) ยืนยันความสำคัญในหลักสูตร Machine Learning และ Deep Learning ดังนี้
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          MLP ประกอบด้วยหลายชั้นของหน่วยประมวลผล (Hidden Layers) ที่สามารถเรียนรู้การเปลี่ยนแปลงข้อมูลผ่านการรวมเชิงเส้นและการใช้ Activation Function ที่ไม่เชิงเส้น
        </li>
        <li>
          ความลึกของ MLP ช่วยให้โมเดลสามารถจับโครงสร้างข้อมูลที่ซับซ้อน เช่น การจำแนกภาพ เสียง และข้อความ ได้ดีกว่าโมเดลเชิงเส้นทั่วไป
        </li>
        <li>
          ตามทฤษฎี Universal Approximation Theorem จาก Cybenko (1989) และ Hornik (1991) ระบุว่า MLP ที่มี Hidden Layer เพียงชั้นเดียว แต่มีจำนวน Neuron เพียงพอ สามารถประมาณค่าฟังก์ชันไม่เชิงเส้นใด ๆ ได้ภายใต้เงื่อนไขบางประการ
        </li>
        <li>
          การเพิ่มจำนวน Hidden Layers ช่วยให้โมเดลเรียนรู้ Feature Hierarchy หรือลำดับชั้นของฟีเจอร์อย่างเป็นระบบ เช่น การจำแนกรูปภาพสามารถเริ่มต้นจากการตรวจจับขอบ → รูปร่างพื้นฐาน → วัตถุเฉพาะเจาะจง
        </li>
        <li>
          MLP ถือเป็นบันไดขั้นแรกที่นำไปสู่ Deep Learning ซึ่งต่อมาพัฒนาเป็นโครงข่ายแบบ Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), และ Transformer Architectures
        </li>
      </ul>
    </div>

    <div className="bg-blue-50 dark:bg-blue-900 text-blue-900 dark:text-blue-100 p-6 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <h3 className="text-xl font-semibold mb-4">จุดเปลี่ยนสำคัญ: Backpropagation Algorithm</h3>
      <p>
        การถือกำเนิดของ Backpropagation Algorithm ในช่วงทศวรรษ 1980 โดยการทำงานของนักวิจัยอย่าง Geoffrey Hinton, David E. Rumelhart และ Ronald J. Williams ได้เปลี่ยนแนวทางการฝึก Neural Networks อย่างสิ้นเชิง โดยอัลกอริทึมนี้ได้รับการยืนยันจากการตีพิมพ์ในวารสาร Nature และ Science ว่าเป็นหนึ่งในจุดเปลี่ยนสำคัญของ AI
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          ก่อนยุค Backpropagation การฝึกโมเดลหลายชั้นเป็นไปได้ยากเนื่องจากการคำนวณ Gradient ที่ซับซ้อนและการไหลของข้อมูลย้อนกลับที่ไม่มีประสิทธิภาพ
        </li>
        <li>
          Backpropagation ทำให้สามารถคำนวณ Gradient ของพารามิเตอร์แต่ละตัวอย่างมีประสิทธิภาพ โดยใช้หลักการ Chain Rule ใน Calculus ทำให้การฝึก MLP และ Deep Neural Networks เป็นไปได้จริงในเชิงคำนวณ
        </li>
        <li>
          กระบวนการนี้ทำให้สามารถอัปเดตค่าพารามิเตอร์แบบ End-to-End ได้อย่างมีประสิทธิภาพในหลายชั้น ซึ่งเป็นหัวใจสำคัญของการฝึกโมเดลที่มีหลายล้านพารามิเตอร์ในปัจจุบัน
        </li>
      </ul>
    </div>

    <div className="bg-purple-50 dark:bg-purple-900 text-purple-900 dark:text-purple-100 p-6 rounded-xl border-l-4 border-purple-400 dark:border-purple-600">
      <h3 className="text-xl font-semibold mb-4">การเรียนรู้แบบ End-to-End จาก Perceptron สู่ Deep Learning</h3>
      <p>
        หนึ่งในวิสัยทัศน์ที่ได้รับการเน้นย้ำในรายงานจาก DeepMind และ Stanford AI Lab คือการพัฒนาโมเดลที่สามารถเรียนรู้แบบ End-to-End หรือการเรียนรู้โดยตรงจากข้อมูลดิบสู่ผลลัพธ์ โดยไม่ต้องมีขั้นตอน Feature Engineering แบบแมนนวลมาก่อน
      </p>
      <ul className="list-disc pl-6 space-y-2">
        <li>
          แนวคิดนี้เริ่มจาก Perceptron ซึ่งสามารถ Mapping จาก Input → Output ได้โดยตรง และขยายสู่ MLP และ Deep Neural Networks ที่สามารถเรียนรู้ Representation ที่ซับซ้อนมากขึ้น
        </li>
        <li>
          ความสามารถในการเรียนรู้แบบ End-to-End ช่วยลดการพึ่งพา Domain Knowledge จากมนุษย์ และทำให้ระบบสามารถค้นพบ Feature ที่เหมาะสมที่สุดสำหรับงานโดยอัตโนมัติ
        </li>
        <li>
          ตัวอย่างที่เห็นได้ชัดคือการเปลี่ยนจากการใช้ Manual Feature Extraction ในงาน Image Classification สู่การใช้ Convolutional Layers ที่เรียนรู้ Feature เองจากข้อมูลภาพดิบ
        </li>
      </ul>
    </div>

  </div>
</section>


        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day17 theme={theme} />
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
        <ScrollSpy_Ai_Day17 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day17_PerceptronMLP;
