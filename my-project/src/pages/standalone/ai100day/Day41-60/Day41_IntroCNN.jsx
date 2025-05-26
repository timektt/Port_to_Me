import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day41 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day41";
import MiniQuiz_Day41 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day41";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day41_IntroCNN = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day41_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day41_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day41_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day41_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day41_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day41_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day41_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day41_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day41_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day41_10").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 41: Introduction to CNNs</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

     <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้อง CNN?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none text-base leading-relaxed space-y-8">
    <h3>ความท้าทายของข้อมูลเชิงภาพ</h3>
    <p>
      ข้อมูลภาพประกอบด้วยข้อมูลหลายมิติและมีโครงสร้างเชิงพื้นที่ (spatial structure) ที่ซับซ้อน เช่น ความสัมพันธ์ของ pixel ที่อยู่ใกล้เคียงกัน หรือ pattern ที่เกิดซ้ำในบริเวณต่าง ๆ ของภาพ ซึ่งไม่สามารถจัดการได้อย่างมีประสิทธิภาพโดย model ทั่วไปอย่าง Multilayer Perceptron (MLP) ที่ไม่มีโครงสร้างเชิงพื้นที่
    </p>

    <h3>การเกิดขึ้นของ Convolutional Neural Networks</h3>
    <p>
      Convolutional Neural Networks (CNNs) ถูกพัฒนาขึ้นเพื่อจัดการกับปัญหาดังกล่าว โดยใช้ layer พิเศษที่เรียกว่า convolutional layers ซึ่งสามารถเรียนรู้ pattern จากพื้นที่เล็ก ๆ ของภาพ แล้วค่อย ๆ สร้างความเข้าใจระดับสูงขึ้นเมื่อข้อมูลผ่าน layer ลึกขึ้นเรื่อย ๆ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded-md">
      <p className="font-semibold">Insight Box:</p>
      <p className="mt-1">
        CNNs ได้รับแรงบันดาลใจจากโครงสร้าง receptive field ของ visual cortex ในสมองมนุษย์ ซึ่งมีการประมวลผลภาพผ่าน neuron ที่รับข้อมูลจากพื้นที่จำกัดในภาพ — งานวิจัยคลาสสิกของ Hubel & Wiesel (1959)
      </p>
    </div>

    <h3>จุดแข็งของ CNN เมื่อเทียบกับ MLP</h3>
    <ul className="list-disc pl-6">
      <li><strong>Local Connectivity:</strong> CNN เรียนรู้ pattern จากกลุ่ม pixel ใกล้เคียง ทำให้เข้าใจ texture ได้ดีกว่า</li>
      <li><strong>Shared Weights:</strong> ตัวกรอง (filter) เดียวกันถูกนำไปใช้ทั่วทั้งภาพ ช่วยลดจำนวนพารามิเตอร์</li>
      <li><strong>Translation Invariance:</strong> สามารถจำแนกรูปแบบเดียวกันในตำแหน่งที่ต่างกันของภาพได้</li>
    </ul>

    <h3>ตัวอย่างการใช้งานจริงที่ CNN ได้เปรียบ</h3>
    <ul className="list-disc pl-6">
      <li>การตรวจจับใบหน้าในกล้องมือถือ</li>
      <li>การแยกโรคจากภาพ X-ray หรือ MRI</li>
      <li>การจำแนกสินค้าจากกล้องในระบบหุ่นยนต์คลังสินค้า</li>
    </ul>

    <h3>การเปลี่ยนแปลงครั้งใหญ่จาก AlexNet</h3>
    <p>
      งานวิจัย AlexNet (Krizhevsky et al., 2012) เป็นจุดเริ่มต้นของยุค Deep Learning ในงานประมวลผลภาพ โดยใช้ CNN ความลึก 8 ชั้นเข้าแข่งขัน ImageNet Challenge และลด error ได้กว่า 10% จากงานวิจัยก่อนหน้า
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded-md">
      <p className="font-semibold">Highlight:</p>
      <p className="mt-1">
        หลังจากความสำเร็จของ AlexNet โครงข่าย CNN ถูกพัฒนาต่อเนื่องเป็น ResNet, Inception, EfficientNet และอื่น ๆ ซึ่งต่างมุ่งเน้นการเพิ่มความลึกและลดพารามิเตอร์ เพื่อความแม่นยำสูงขึ้นในระดับอุตสาหกรรม
      </p>
    </div>

    <h3>สรุป</h3>
    <p>
      CNN กลายเป็นรากฐานของงานด้าน Computer Vision ที่สามารถจัดการข้อมูลเชิงภาพได้อย่างมีประสิทธิภาพ ไม่ว่าจะเป็นการจำแนก การตรวจจับ หรือการวิเคราะห์ภาพเชิงลึก โดยยังเป็นหนึ่งในสาขาที่มีการพัฒนาอย่างต่อเนื่อง
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition (2023)</li>
      <li>MIT 6.S191: Deep Learning for Self-Driving Cars, Lecture 3 - CNNs</li>
      <li>Hubel, D.H., & Wiesel, T.N. (1959). Receptive fields of single neurons in the cat's striate cortex. Journal of Physiology.</li>
    </ul>
  </div>
</section>


 <section id="convolution" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. พื้นฐาน Convolution</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>นิยามของ Convolution ในบริบท Deep Learning</h3>
    <p>
      การทำ Convolution คือกระบวนการทางคณิตศาสตร์ที่มีต้นกำเนิดจาก signal processing โดยในระบบ Deep Learning จะนำแนวคิดนี้มาใช้เพื่อดึงลักษณะเฉพาะ (features) จากภาพหรือข้อมูลเชิงพื้นที่ผ่าน kernel หรือ filter ซึ่งเป็น weight matrix ขนาดเล็กที่เลื่อนผ่าน input tensor เพื่อคำนวณค่าออกมาเป็น feature map
    </p>

    <h3>รูปแบบของ Convolution Operation</h3>
    <ul className="list-disc pl-6">
      <li><strong>2D Convolution:</strong> ใช้กับข้อมูลภาพทั่วไป เช่น grayscale หรือ RGB image</li>
      <li><strong>1D Convolution:</strong> ใช้กับข้อมูลลำดับ เช่น time-series หรือ audio</li>
      <li><strong>3D Convolution:</strong> ใช้กับข้อมูลภาพสามมิติ เช่น MRI scan</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p className="mt-2">
        ในงานวิจัยของ Stanford (CS231n) การทำ Convolution ถือเป็นหัวใจสำคัญที่ทำให้ CNN สามารถลดจำนวนพารามิเตอร์และเรียนรู้ spatial hierarchy ได้อย่างมีประสิทธิภาพโดยไม่ต้องใช้ fully connected layer จำนวนมาก
      </p>
    </div>

    <h3>การคำนวณ Convolution เบื้องต้น</h3>
    <p>ตัวอย่างการคำนวณ 2D Convolution ด้วย kernel 3x3 บน input 5x5 โดย stride = 1 และไม่มี padding:</p>
    <pre><code className="language-text">
Input (5x5):                Kernel (3x3):
[[1 2 3 0 1]                [[0 -1 0]
 [4 5 6 1 2]                 [1  0 -1]
 [7 8 9 0 0]                 [0  1  0]]
 [1 3 2 4 5]
 [0 6 7 8 9]]

→ Output (3x3): Feature Map ที่เกิดจากการ dot product และ sum
    </code></pre>

    <h3>Padding และ Stride</h3>
    <ul className="list-disc pl-6">
      <li><strong>Padding:</strong> ช่วยให้ output มีขนาดเท่า input หรือคงข้อมูลขอบภาพ</li>
      <li><strong>Stride:</strong> ควบคุมการเลื่อนของ kernel ทำให้สามารถลดขนาด spatial resolution ได้</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Highlight Box:</p>
      <p>
        การตั้งค่า stride มากกว่า 1 จะลด resolution ของภาพขาออก และอาจใช้แทน pooling ได้ในบางกรณี แต่ต้องระวังเรื่องการสูญเสียข้อมูลสำคัญในภาพ
      </p>
    </div>

    <h3>Properties ที่ทำให้ Convolution เหมาะกับภาพ</h3>
    <ul className="list-disc pl-6">
      <li><strong>Local connectivity:</strong> kernel จะเห็นเพียง local region ในแต่ละครั้ง ทำให้เข้าใจ context เฉพาะจุดได้ดี</li>
      <li><strong>Parameter sharing:</strong> kernel เดียวกันเลื่อนไปทั่วภาพ ทำให้ลดจำนวนพารามิเตอร์</li>
      <li><strong>Translation equivariance:</strong> ผลลัพธ์ยังคงลักษณะสำคัญเมื่อวัตถุถูกเลื่อนภายในภาพ</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Stanford CS231n Lecture Notes: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Introduction to Deep Learning (2023)</li>
      <li>Deep Learning Book, Ian Goodfellow et al., Chapter 9: Convolutional Networks</li>
      <li>arXiv:1603.07285 — A guide to convolution arithmetic for deep learning</li>
    </ul>
  </div>
</section>


<section id="layers" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Layer ประเภทต่าง ๆ ใน CNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>Overview: สถาปัตยกรรมแบบเลเยอร์ใน CNN</h3>
    <p>
      โครงสร้างของ Convolutional Neural Networks (CNNs) ประกอบด้วยชุดของเลเยอร์ที่ทำงานร่วมกันเพื่อเรียนรู้ hierarchical representations จากข้อมูลภาพ โดยแต่ละเลเยอร์มีหน้าที่เฉพาะและมีผลต่อการเรียนรู้ฟีเจอร์ต่าง ๆ ในลักษณะที่แตกต่างกันอย่างเป็นระบบ
    </p>

    <h3>1. Convolutional Layer</h3>
    <p>
      เป็นเลเยอร์หลักที่รับผิดชอบในการเรียนรู้ spatial patterns โดยใช้ kernel หรือ filter ที่เลื่อนผ่านอินพุต และคำนวณค่าด้วยการทำ dot product
    </p>
    <ul className="list-disc pl-6">
      <li>สามารถจับ pattern เช่น ขอบ, เส้น, หรือ texture</li>
      <li>ใช้การแชร์พารามิเตอร์ทำให้จำนวนพารามิเตอร์ลดลงมาก</li>
    </ul>

    <h3>2. Activation Layer (เช่น ReLU)</h3>
    <p>
      ใช้ฟังก์ชันไม่เชิงเส้น (non-linear activation function) เพื่อทำให้โมเดลสามารถเรียนรู้ pattern ที่ซับซ้อนได้ เช่น ReLU(x) = max(0, x)
    </p>

    <h3>3. Pooling Layer</h3>
    <p>
      ลดขนาด spatial dimension ของ feature map เพื่อควบคุม overfitting และเพิ่ม robustness ต่อการเปลี่ยนแปลงในภาพ
    </p>
    <ul className="list-disc pl-6">
      <li><strong>Max Pooling:</strong> เลือกค่าที่มากที่สุดในแต่ละ region</li>
      <li><strong>Average Pooling:</strong> คำนวณค่าเฉลี่ยของแต่ละ region</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานวิจัยจาก Stanford (CS231n) แสดงให้เห็นว่า การใช้ max pooling ช่วยเน้น feature ที่สำคัญและส่งผลให้ network มีความสามารถในการ generalize ได้ดีขึ้นโดยเฉพาะในงาน classification
      </p>
    </div>

    <h3>4. Fully Connected Layer (Dense Layer)</h3>
    <p>
      ใช้ในช่วงท้ายของ CNN เพื่อแปลง feature map ที่เรียนรู้มาให้กลายเป็น output vector ที่เหมาะกับ task เช่น classification
    </p>

    <h3>5. Dropout Layer</h3>
    <p>
      เป็น regularization technique ที่ช่วยลด overfitting โดยการสุ่มตัดการเชื่อมต่อระหว่าง neurons บางส่วนในแต่ละรอบการฝึก
    </p>

    <h3>สรุปโครงสร้างทั่วไปของ CNN</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border text-sm">
        <thead className="bg-gray-200 dark:bg-gray-700">
          <tr>
            <th className="border px-4 py-2 text-left">Layer Type</th>
            <th className="border px-4 py-2 text-left">Function</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Convolutional</td>
            <td className="border px-4 py-2">ดึงฟีเจอร์จาก input ด้วย kernel</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Activation (ReLU)</td>
            <td className="border px-4 py-2">เพิ่ม non-linearity ให้ network</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Pooling</td>
            <td className="border px-4 py-2">ลด dimensionality และเพิ่ม robustness</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Fully Connected</td>
            <td className="border px-4 py-2">สร้าง decision boundary จาก features</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        ความลึกของ CNN ไม่ได้หมายถึงจำนวน layer เท่านั้น แต่ยังรวมถึงความสามารถในการเรียนรู้ feature ที่ abstract มากขึ้นในแต่ละเลเยอร์ลึก ๆ ด้วย
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning Lecture Series</li>
      <li>Goodfellow et al. (2016), Deep Learning, MIT Press</li>
      <li>arXiv:1409.1556 - Visualizing and Understanding Convolutional Networks</li>
    </ul>
  </div>
</section>


    <section id="learning" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. กระบวนการเรียนรู้ของ CNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>โครงสร้างพื้นฐานของการเรียนรู้ใน Convolutional Neural Networks</h3>
    <p>
      การเรียนรู้ของ CNN (Convolutional Neural Networks) เป็นกระบวนการที่เกิดขึ้นผ่านการปรับปรุงค่าพารามิเตอร์ของเลเยอร์ต่าง ๆ ด้วยการย้อนแพร่ค่าความผิดพลาด (backpropagation) ซึ่งส่งผลโดยตรงต่อความสามารถในการจดจำรูปแบบ (pattern recognition) จากข้อมูลที่มีโครงสร้างเชิงพื้นที่ เช่น ภาพหรือวิดีโอ
    </p>

    <h3>ขั้นตอนการเรียนรู้ของ CNN</h3>
    <ul className="list-disc pl-6">
      <li><strong>Forward Propagation:</strong> อินพุตถูกส่งผ่าน convolutional layers, activation functions, และ pooling layers เพื่อสร้าง feature map</li>
      <li><strong>Loss Calculation:</strong> ใช้ loss function เช่น cross-entropy หรือ MSE เพื่อวัดความคลาดเคลื่อนระหว่างค่าที่พยากรณ์กับค่าจริง</li>
      <li><strong>Backpropagation:</strong> คำนวณ gradient ของ loss กับพารามิเตอร์ทั้งหมด โดยใช้ chain rule</li>
      <li><strong>Parameter Update:</strong> ใช้ optimizer เช่น SGD หรือ Adam ในการอัปเดตพารามิเตอร์เพื่อให้ loss ลดลง</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ใน CNN การเรียนรู้ไม่ได้จำกัดเฉพาะ weight ของ kernel แต่รวมถึง bias และลำดับชั้นของ features ที่ถูกดึงออกมาในระดับต่าง ๆ — จาก low-level ไปจนถึง high-level representation
      </p>
    </div>

    <h3>การทำ Backpropagation ใน Convolutional Layers</h3>
    <p>
      แตกต่างจาก fully connected layers ตรงที่ convolutional layers มีการแชร์น้ำหนัก (weight sharing) และการเคลื่อนที่ของ kernel (sliding window) ทำให้การคำนวณ gradient ซับซ้อนมากขึ้น แต่ประหยัดพารามิเตอร์ได้มหาศาล
    </p>
   <p>สูตรพื้นฐานของการคำนวณ gradient สำหรับ kernel ใน convolutional layer คือ:</p>

<pre>
<code className="language-python">
∇W = ∑ (i=1 ถึง m) [ δᵢ * xᵢ ]
</code>
</pre>

<p>
โดย δᵢ คือ error term ที่ส่งกลับมาจากเลเยอร์ถัดไป และ xᵢ คืออินพุตในตำแหน่งที่ kernel เคลื่อนไป
</p>


    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยจาก Stanford (CS231n) แนะนำให้ใช้การ normalize gradient และการใช้ learning rate schedule เพื่อเพิ่มความเสถียรในการฝึก CNN ที่มีความลึกมาก
      </p>
    </div>

    <h3>การใช้ Optimizer ในการเรียนรู้ของ CNN</h3>
    <table className="table-auto w-full text-sm border border-gray-300">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2 text-left">Optimizer</th>
          <th className="border px-4 py-2 text-left">ลักษณะเด่น</th>
          <th className="border px-4 py-2 text-left">ข้อควรระวัง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">SGD</td>
          <td className="border px-4 py-2">Simple, converges with proper tuning</td>
          <td className="border px-4 py-2">ต้องใช้ learning rate decay</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Adam</td>
          <td className="border px-4 py-2">เร็ว, ปรับ learning rate อัตโนมัติ</td>
          <td className="border px-4 py-2">อาจไม่ generalize ได้ดีเสมอไป</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">RMSProp</td>
          <td className="border px-4 py-2">ดีสำหรับ RNNs หรือ signal with noise</td>
          <td className="border px-4 py-2">ต้องจับคู่กับ learning rate ที่เหมาะสม</td>
        </tr>
      </tbody>
    </table>

    <h3>บทบาทของ Batch Size และ Learning Rate</h3>
    <ul className="list-disc pl-6">
      <li><strong>Batch Size:</strong> batch ที่ใหญ่ช่วยให้ gradient มีความนิ่ง แต่ใช้ memory สูง</li>
      <li><strong>Learning Rate:</strong> หากสูงเกินไปอาจ overshoot จุด minimum หากต่ำเกินไปจะ convergence ช้า</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning for Self-Driving Cars, 2021</li>
      <li>arXiv:1609.04747 — "Understanding the difficulty of training deep feedforward neural networks"</li>
      <li>IEEE Transactions on Neural Networks, 2020</li>
    </ul>
  </div>
</section>


 <section id="hierarchies" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Feature Hierarchies</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>แนวคิดของ Feature Hierarchy ใน CNN</h3>
    <p>
      ในสถาปัตยกรรมของ Convolutional Neural Networks (CNNs) การเรียนรู้ลักษณะหรือ feature ของข้อมูลจะดำเนินไปในลักษณะลำดับชั้น (hierarchical representation) โดยชั้นต้น ๆ ของโมเดลจะเรียนรู้ลักษณะที่มีความเฉพาะน้อย เช่น edge, color, หรือ texture ในขณะที่ชั้นลึกขึ้นจะเรียนรู้ลักษณะที่ซับซ้อนมากขึ้น เช่น รูปร่างหรือส่วนประกอบของวัตถุ ซึ่งถือเป็นกลไกสำคัญที่ทำให้ CNN ประสบความสำเร็จในงานด้านการรู้จำภาพ
    </p>

    <h3>ตัวอย่าง Feature ในแต่ละชั้น</h3>
    <ul className="list-disc pl-6">
      <li><strong>ชั้นแรก:</strong> ตรวจจับ edge แนวตั้ง แนวนอน หรือแนวทแยง</li>
      <li><strong>ชั้นกลาง:</strong> เรียนรู้รูปทรงเช่น circle, rectangle, curve</li>
      <li><strong>ชั้นลึก:</strong> จับองค์ประกอบที่มีความหมาย เช่น หน้า, หู, ปีก</li>
    </ul>

    <h3>ตารางแสดงลำดับของ Feature Hierarchies</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-600 dark:bg-gray-800">
          <tr>
            <th className="px-4 py-2 border">ระดับชั้น</th>
            <th className="px-4 py-2 border">ประเภทของ Feature</th>
            <th className="px-4 py-2 border">ตัวอย่าง</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900">
          <tr>
            <td className="px-4 py-2 border">Low-Level</td>
            <td className="px-4 py-2 border">Edges, Textures</td>
            <td className="px-4 py-2 border">Sobel, Gabor Filters</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border">Mid-Level</td>
            <td className="px-4 py-2 border">Shapes, Motifs</td>
            <td className="px-4 py-2 border">Curves, Rectangles</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border">High-Level</td>
            <td className="px-4 py-2 border">Object Parts</td>
            <td className="px-4 py-2 border">Faces, Hands</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-600 p-4 rounded-md border-l-4 border-blue-400">
      <p className="font-semibold">Insight Box</p>
      <p>
        การเรียนรู้แบบลำดับชั้นของ CNN ช่วยให้โมเดลสามารถ generalize ได้ดีขึ้นในงานที่ซับซ้อน เช่น การตรวจจับวัตถุหลากหลายขนาดและมุมมอง แม้จะมีข้อมูลจำนวนน้อยก็ตาม
      </p>
    </div>

    <h3>ประโยชน์ของ Feature Hierarchy</h3>
    <ul className="list-disc pl-6">
      <li>ลดการพึ่งพาการออกแบบ feature ด้วยมือ (manual feature engineering)</li>
      <li>เพิ่มความสามารถในการเรียนรู้ลักษณะที่ซับซ้อนจากข้อมูลดิบ</li>
      <li>เหมาะกับการทำ Transfer Learning เนื่องจากสามารถ reuse feature บางส่วนได้</li>
    </ul>

    <div className="bg-yellow-600 p-4 rounded-md border-l-4 border-yellow-500">
      <p className="font-semibold">Highlight Box</p>
      <p>
        งานวิจัยจาก MIT และ DeepMind (Nature, 2021) แสดงให้เห็นว่า CNN ที่มีความลึกมากขึ้นจะมีการจัดระเบียบ feature ที่สอดคล้องกับลำดับการประมวลผลภาพในสมองมนุษย์ ซึ่งสนับสนุนทฤษฎีของ feature hierarchy
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature.</li>
      <li>MIT CSAIL: Hierarchical Representation in Vision, 2022</li>
      <li>DeepMind Research, 2021. Visual Processing and Cortex-like Feature Hierarchy</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
    </ul>
  </div>
</section>


<section id="parameters" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. การลดจำนวนพารามิเตอร์</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความสำคัญของการลดพารามิเตอร์</h3>
    <p>
      จำนวนพารามิเตอร์ใน Convolutional Neural Networks (CNNs) มีผลโดยตรงต่อประสิทธิภาพในการเรียนรู้ ความเร็วในการ inference และความสามารถในการ deploy บนอุปกรณ์ที่จำกัดทรัพยากร เช่น edge devices หรือ embedded systems การลดจำนวนพารามิเตอร์อย่างมีประสิทธิภาพจึงเป็นหัวข้อสำคัญในงานวิจัยด้าน deep learning
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานวิจัยจาก Stanford ML Group พบว่าโมเดลที่ผ่านการ prune พารามิเตอร์สามารถลดขนาดได้กว่า 90% โดยไม่สูญเสีย accuracy อย่างมีนัยสำคัญ (Han et al., 2015)
      </p>
    </div>

    <h3>เทคนิคการลดพารามิเตอร์หลัก</h3>
    <ul className="list-disc pl-6">
      <li><strong>Pruning:</strong> การตัดพารามิเตอร์ที่มีน้ำหนักใกล้ศูนย์ทิ้ง เช่น weight magnitude pruning</li>
      <li><strong>Quantization:</strong> ลด precision ของพารามิเตอร์ เช่น จาก float32 → int8 เพื่อประหยัดหน่วยความจำ</li>
      <li><strong>Knowledge Distillation:</strong> ฝึกโมเดลเล็ก (student) ให้เลียนแบบผลลัพธ์ของโมเดลใหญ่ (teacher)</li>
      <li><strong>Depthwise Separable Convolution:</strong> แทน convolution ปกติด้วยการแยก channel และ spatial convolution</li>
    </ul>

    <h3>ตัวอย่างการลดพารามิเตอร์ด้วย Depthwise Separable Convolution</h3>
    <table className="table-auto w-full border text-sm overflow-x-auto">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">ประเภท Convolution</th>
          <th className="border px-4 py-2">จำนวน Parameters</th>
          <th className="border px-4 py-2">Inference Speed</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Standard Convolution</td>
          <td className="border px-4 py-2">~1.2M</td>
          <td className="border px-4 py-2">1x</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Depthwise Separable</td>
          <td className="border px-4 py-2">~0.2M</td>
          <td className="border px-4 py-2">~3-5x เร็วกว่า</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        Depthwise separable convolution เป็นรากฐานสำคัญของโมเดลที่เน้น lightweight เช่น MobileNet และ EfficientNet
      </p>
    </div>

    <h3>เปรียบเทียบโมเดลแบบ Full กับ Compressed</h3>
    <ul className="list-disc pl-6">
      <li><strong>ResNet-50:</strong> ~25M parameters, accuracy สูง, latency สูง</li>
      <li><strong>MobileNetV2:</strong> ~3.4M parameters, accuracy รองลงมา, latency ต่ำกว่า ~5x</li>
      <li><strong>SqueezeNet:</strong> ~1.2M parameters, ขนาดเล็กมาก, suitable for IoT</li>
    </ul>

    <h3>ข้อควรระวังในการลดพารามิเตอร์</h3>
    <ul className="list-disc pl-6">
      <li>การลดพารามิเตอร์มากเกินไปอาจทำให้โมเดล underfit</li>
      <li>เทคนิคบางอย่างเช่น quantization อาจทำให้เกิด loss ของ precision โดยเฉพาะใน regression tasks</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Han, Song, et al. "Learning both weights and connections for efficient neural network." NeurIPS 2015.</li>
      <li>Howard et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861</li>
      <li>Chollet, F. "Xception: Deep Learning with Depthwise Separable Convolutions." CVPR 2017.</li>
      <li>Stanford CS231n, Lecture 12: Model Compression & Acceleration</li>
    </ul>
  </div>
</section>


<section id="compare" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. เปรียบเทียบ CNN กับ MLP</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>
  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>พื้นฐานของ CNN และ MLP</h3>
    <p>
      Convolutional Neural Networks (CNNs) และ Multilayer Perceptrons (MLPs) เป็นสองสถาปัตยกรรมหลักใน Deep Learning ที่มีลักษณะโครงสร้างและการนำไปใช้งานที่แตกต่างกันอย่างชัดเจน โดย MLP เป็นโครงข่าย fully-connected ทั่วไปที่ทุก node เชื่อมต่อกันทั้งหมด ขณะที่ CNN ออกแบบมาเพื่อจัดการข้อมูลเชิงพื้นที่ เช่น ภาพ ด้วยการใช้ convolutional filters
    </p>

    <h3>ตารางเปรียบเทียบความแตกต่าง</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-600 dark:bg-gray-700">
          <tr>
            <th className="border px-4 py-2">ลักษณะ</th>
            <th className="border px-4 py-2">CNN</th>
            <th className="border px-4 py-2">MLP</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900">
          <tr>
            <td className="border px-4 py-2">การเชื่อมต่อ</td>
            <td className="border px-4 py-2">Local connection + shared weights</td>
            <td className="border px-4 py-2">Fully connected</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Parameter efficiency</td>
            <td className="border px-4 py-2">สูง (เนื่องจาก weight sharing)</td>
            <td className="border px-4 py-2">ต่ำ (จำนวน parameter มาก)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การจัดการกับภาพ</td>
            <td className="border px-4 py-2">เหมาะสมมาก</td>
            <td className="border px-4 py-2">ไม่เหมาะต้อง flatten ภาพก่อน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Overfitting</td>
            <td className="border px-4 py-2">น้อยกว่า</td>
            <td className="border px-4 py-2">มากกว่า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-600 p-4 border-l-4 border-yellow-500 rounded">
      <p className="font-semibold">Highlight Box:</p>
      <p>
        การใช้ CNN ช่วยลดจำนวนพารามิเตอร์และยังคงความสามารถในการจับ pattern เชิงพื้นที่ได้ดีกว่า MLP อย่างมาก ทำให้เป็นสถาปัตยกรรมหลักในงาน image recognition
      </p>
    </div>

    <h3>ประสิทธิภาพในการใช้งานจริง</h3>
    <ul className="list-disc pl-6">
      <li>CNN ได้รับการพิสูจน์ว่ามี performance ดีกว่า MLP อย่างมากในการจัดการกับข้อมูลภาพ</li>
      <li>MLP เหมาะกับข้อมูลเชิง tabular หรือข้อมูลที่ไม่มี spatial structure</li>
      <li>MLP อาจยังใช้เป็น layer สุดท้ายของ CNN เพื่อทำ classification</li>
    </ul>

    <div className="bg-blue-600 p-4 border-l-4 border-blue-400 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานวิจัยของ LeCun et al. (1998) เป็นจุดเริ่มต้นของการใช้ CNN แทน MLP ในงาน OCR โดยแสดงให้เห็นว่า CNN สามารถเรียนรู้ features จากภาพได้แบบอัตโนมัติ ซึ่งเปลี่ยนแนวทางของ deep learning ไปอย่างสิ้นเชิง
      </p>
    </div>

    <h3>ข้อจำกัดของแต่ละสถาปัตยกรรม</h3>
    <ul className="list-disc pl-6">
      <li>CNN อาจไม่เหมาะกับข้อมูลแบบ sequential หรือ symbolic</li>
      <li>MLP ไม่สามารถจับ spatial correlation ได้ดี</li>
      <li>ทั้งสองแบบสามารถปรับใช้ร่วมกันได้ใน hybrid architecture</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition". Proceedings of the IEEE.</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning Lecture Series (2022)</li>
      <li>arXiv:2009.10875 - CNN vs Fully Connected Networks</li>
    </ul>
  </div>
</section>


  <section id="usecases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Use Cases ของ CNN ในโลกจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none text-base leading-relaxed space-y-10">
    <h3>ภาพรวมของการประยุกต์ใช้ CNN ในอุตสาหกรรม</h3>
    <p>
      Convolutional Neural Networks (CNNs) ได้รับการนำไปใช้อย่างแพร่หลายในงานประมวลผลภาพ เสียง และข้อมูลแบบมีโครงสร้างที่มีลักษณะเป็นลำดับสองมิติ ความสามารถของ CNN ในการเรียนรู้ spatial features โดยไม่ต้องออกแบบ feature โดยมนุษย์ทำให้โมเดลประเภทนี้กลายเป็นหัวใจสำคัญของหลายระบบอัจฉริยะในยุคปัจจุบัน
    </p>

    <h3>1. การรู้จำภาพและวัตถุ (Image Classification & Object Detection)</h3>
    <ul className="list-disc pl-6">
      <li>ระบบรู้จำวัตถุแบบ real-time (YOLO, SSD, Faster R-CNN)</li>
      <li>การจัดหมวดหมู่วัตถุในภาพถ่าย (เช่น ภาพแมว, สุนัข, รถยนต์)</li>
      <li>การตรวจจับใบหน้าและวิเคราะห์อารมณ์ (Facial Recognition, Emotion Detection)</li>
    </ul>

    <h3>2. การประมวลผลทางการแพทย์ (Medical Imaging)</h3>
    <ul className="list-disc pl-6">
      <li>ตรวจจับโรคจากภาพ MRI, CT scan, หรือภาพ X-ray</li>
      <li>โมเดล CheXNet จาก Stanford ใช้ CNN ในการตรวจโรคปอด</li>
      <li>Segmentation ภาพทางการแพทย์เพื่อช่วยแพทย์วินิจฉัย</li>
    </ul>

    <h3>3. ยานยนต์ไร้คนขับ (Autonomous Vehicles)</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ CNN สำหรับ Lane Detection และ Road Sign Recognition</li>
      <li>การวิเคราะห์วัตถุโดยรอบจากกล้องความละเอียดสูง</li>
      <li>การควบคุมพฤติกรรมของรถโดยอิงจากภาพแบบ end-to-end</li>
    </ul>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        รายงานจาก McKinsey (2023) ระบุว่า 65% ของบริษัทในกลุ่มอุตสาหกรรมยานยนต์ที่ใช้ CNNs ในระบบ perception ของรถยนต์สามารถลดอุบัติเหตุจากการทดสอบลงได้กว่า 40% ในระยะเวลา 12 เดือน
      </p>
    </div>

    <h3>4. ระบบตรวจสอบความปลอดภัย (Security & Surveillance)</h3>
    <ul className="list-disc pl-6">
      <li>วิเคราะห์วิดีโอจากกล้องวงจรปิดเพื่อจับกิจกรรมผิดปกติ</li>
      <li>การระบุบุคคลต้องสงสัยโดยใช้ face embedding + CNN</li>
    </ul>

    <h3>5. ระบบแนะนำสินค้า (Product Recommendation)</h3>
    <p>
      CNN ถูกนำมาใช้ในการวิเคราะห์ภาพของสินค้าในระบบ E-commerce เพื่อช่วยในการจับคู่สินค้าที่คล้ายกัน หรือสร้างระบบแนะนำ (Visual Similarity) โดยอิงจากภาพเป็นหลัก เช่น Amazon และ Zalando
    </p>

    <h3>6. การสร้างภาพใหม่จากโครงสร้าง (Image Generation)</h3>
    <ul className="list-disc pl-6">
      <li>ระบบ Super-resolution เพื่อเพิ่มความละเอียดของภาพ</li>
      <li>การเติมส่วนที่ขาดในภาพ (Inpainting)</li>
      <li>ใช้ร่วมกับ GANs เพื่อสร้างภาพจากข้อความหรือโครงร่าง</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        CNNs ไม่ได้จำกัดแค่การวิเคราะห์ภาพนิ่ง แต่ยังสามารถใช้กับวิดีโอ เสียง และข้อมูล 3D ได้ เช่นในงาน Video Action Recognition หรือ Volumetric Data Analysis
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>arXiv: CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays</li>
      <li>IEEE Transactions on Image Processing</li>
      <li>MIT Deep Learning Lectures 6 & 8 (CNNs in practice)</li>
      <li>McKinsey Global Institute: AI Impact on Automotive Sector, 2023</li>
    </ul>
  </div>
</section>


     <section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>บทเรียนสำคัญจากการออกแบบสถาปัตยกรรม CNN</h3>
    <p>
      การออกแบบ Convolutional Neural Networks (CNNs) ไม่ใช่เพียงแค่การกำหนดจำนวนเลเยอร์หรือค่าพารามิเตอร์ แต่คือการผสมผสานความเข้าใจในหลักการเรียนรู้เชิงลึก การประมวลผลภาพ และกลยุทธ์เชิงคณิตศาสตร์ในการลดมิติ ขยายขนาด และทำให้โมเดลเรียนรู้ได้อย่างมีประสิทธิภาพ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box</p>
      <p className="mt-2">
        ความสำเร็จของ CNN มาจากการ "เข้าใจโครงสร้างข้อมูลภาพ" และ "ใช้ประโยชน์จาก spatial locality" โดยการลดจำนวนพารามิเตอร์ผ่านการแชร์น้ำหนัก ทำให้โมเดลสามารถเรียนรู้จากภาพความละเอียดสูงโดยไม่ต้องใช้ทรัพยากรมากเกินไป
      </p>
    </div>

    <h3>บทวิเคราะห์จากมหาวิทยาลัยชั้นนำ</h3>
    <ul className="list-disc pl-6">
      <li><strong>Stanford:</strong> CNN ถูกใช้เป็นกรณีศึกษาหลักในวิชา CS231n เพื่อสาธิตแนวคิดของ Feature Hierarchies และ Receptive Field</li>
      <li><strong>MIT CSAIL:</strong> แนะนำให้มอง CNN เป็นระบบที่เรียนรู้จากการจำแนกลักษณะเชิงโครงสร้าง (structural pattern) ไม่ใช่แค่ความเข้มของพิกเซล</li>
      <li><strong>CMU:</strong> ชี้ให้เห็นว่า CNN สามารถขยายได้ถึงสเกลระดับ ResNet-152 โดยยังคง generalize ได้ดี หากมี regularization และ normalization ที่เหมาะสม</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p className="mt-2">
        งานของ He et al. (2016) กับ ResNet ชี้ให้เห็นว่าการเพิ่มความลึกไม่จำเป็นต้องทำให้โมเดล overfit เสมอไป หากมีการใช้ residual connection ที่เหมาะสม เป็นหนึ่งในจุดเปลี่ยนของการออกแบบ CNN ยุคใหม่
      </p>
    </div>

    <h3>ความเข้าใจผิดที่พบบ่อย</h3>
    <ul className="list-disc pl-6">
      <li>คิดว่าเพิ่มจำนวนเลเยอร์อย่างเดียวจะทำให้โมเดลแม่นยำขึ้นเสมอ — ความจริงคือปัญหา vanishing gradient และ overfitting ต้องได้รับการจัดการ</li>
      <li>ใช้ kernel ขนาดใหญ่ตลอดเวลา — งานวิจัยแนะนำให้ใช้ kernel ขนาดเล็ก (3x3) แบบต่อเนื่องหลายชั้น แทน kernel ใหญ่ที่ขาด fine-grained features</li>
      <li>ละเลย Batch Normalization — ข้อมูลจาก Google Brain พบว่า BN ช่วยให้โมเดลเรียนรู้เร็วขึ้นและเสถียรขึ้น</li>
    </ul>

    <h3>แนวโน้มการพัฒนา CNN ในอนาคต</h3>
    <ul className="list-disc pl-6">
      <li><strong>Hybrid CNN-Transformer:</strong> ผสาน CNN กับ Self-Attention เพื่อเข้าใจภาพทั้ง local และ global context</li>
      <li><strong>EfficientNet & Lightweight Models:</strong> ใช้ compound scaling เพื่อสร้างโมเดลที่แม่นยำและใช้ทรัพยากรน้อยลง</li>
      <li><strong>Self-supervised Learning:</strong> ลดการพึ่งพา label โดยฝึกโมเดลให้เข้าใจโครงสร้างข้อมูลด้วยตัวเอง</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT CSAIL: "Understanding CNNs Through Feature Maps", 2021</li>
      <li>arXiv:1512.03385 - Deep Residual Learning for Image Recognition (He et al.)</li>
      <li>IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day41 theme={theme} />
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
        <ScrollSpy_Ai_Day41 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day41_IntroCNN;
