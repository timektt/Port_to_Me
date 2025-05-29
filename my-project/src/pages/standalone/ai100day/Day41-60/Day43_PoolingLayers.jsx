import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day43 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day43";
import MiniQuiz_Day43 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day43";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day43_PoolingLayers = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day43_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day43_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day43_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day43_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day43_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day43_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day43_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day43_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day43_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day43_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day43_11").format("auto").quality("auto").resize(scale().width(500));
  const img12 = cld.image("Day43_12").format("auto").quality("auto").resize(scale().width(500));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 43: Pooling Layers & Spatial Reduction</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

   <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้องมี Pooling Layer?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert text-base leading-relaxed space-y-8">
    <p>
      ในสถาปัตยกรรมของ Convolutional Neural Networks (CNNs) การจัดการกับขนาดของข้อมูลที่ส่งผ่านเลเยอร์ต่าง ๆ ถือเป็นหัวใจสำคัญในการออกแบบโมเดลที่มีประสิทธิภาพ Pooling Layer จึงถูกพัฒนาขึ้นเพื่อทำหน้าที่ลดขนาดของ Feature Maps อย่างเป็นระบบ โดยไม่ทำลายคุณสมบัติที่สำคัญของข้อมูลเดิม ทั้งนี้ยังช่วยเพิ่มประสิทธิภาพในการคำนวณและลดความซับซ้อนของโมเดลได้อย่างมาก
    </p>

    <h3>ความจำเป็นของ Pooling Layer</h3>
    <p>
      Pooling ถูกออกแบบมาเพื่อช่วยลด dimensionality ของข้อมูลที่ถูกส่งมาจาก Convolutional Layers โดยคงไว้ซึ่งข้อมูลเชิงพื้นที่ที่สำคัญ เทคนิคนี้มีความสำคัญต่อ:
    </p>
    <ul>
      <li>การควบคุม Overfitting โดยการลดจำนวนพารามิเตอร์ในโมเดล</li>
      <li>การสร้างความทนทานต่อการแปรผันเชิงตำแหน่ง (translation invariance)</li>
      <li>การเพิ่มประสิทธิภาพการคำนวณระหว่างกระบวนการ training และ inference</li>
    </ul>

    <div className="p-4 rounded-lg border border-yellow-400 bg-yellow-600 dark:bg-yellow-800 dark:border-yellow-600">
      <p className="font-semibold text-yellow-900 dark:text-yellow-600">Insight Box:</p>
      <p className="text-yellow-800 dark:text-yellow-50">
        งานวิจัยของ LeCun et al. (1998) แสดงให้เห็นว่าการใช้ subsampling (หรือ pooling ในยุคปัจจุบัน) ช่วยให้โมเดลมีประสิทธิภาพในการเรียนรู้ภาพที่มีการแปรผันตำแหน่งสูง โดยเฉพาะในโมเดลเช่น LeNet-5
      </p>
    </div>

    <h3>แนวคิดพื้นฐาน</h3>
    <p>
      Pooling คือกระบวนการที่เลือกค่าสถิติบางอย่าง (เช่น ค่าสูงสุด หรือค่าเฉลี่ย) จากกลุ่มของพิกเซลในพื้นที่ย่อยของ Feature Map โดยทั่วไปจะใช้กรอบ 2x2 หรือ 3x3 ซึ่งเลื่อนไปทีละสเต็ปตามค่า stride ที่กำหนดไว้
    </p>
    <pre className="bg-gray-600 dark:bg-gray-800 p-4 rounded-md overflow-auto text-sm">
<code>
Input Feature Map:
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]

Max Pooling (2x2):
[[5, 6],
 [8, 9]]
</code>
    </pre>

    <h3>เปรียบเทียบกับ Convolution</h3>
    <p>
      แม้ว่า Pooling จะมีโครงสร้างคล้ายกับ Convolution แต่มีความแตกต่างสำคัญคือไม่มีพารามิเตอร์ที่ต้องเรียนรู้ และไม่ได้ออกแบบมาเพื่อแยกฟีเจอร์ใหม่ แต่เน้นการย่อขนาดข้อมูลพร้อมรักษา pattern สำคัญ
    </p>
    <table className="table-auto w-full border border-gray-300 dark:border-gray-600 text-sm">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">คุณสมบัติ</th>
          <th className="border px-4 py-2">Convolution</th>
          <th className="border px-4 py-2">Pooling</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">มีพารามิเตอร์</td>
          <td className="border px-4 py-2">ใช่</td>
          <td className="border px-4 py-2">ไม่ใช่</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">สร้างฟีเจอร์ใหม่</td>
          <td className="border px-4 py-2">ใช่</td>
          <td className="border px-4 py-2">ไม่ใช่</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ลดขนาดข้อมูล</td>
          <td className="border px-4 py-2">บางกรณี</td>
          <td className="border px-4 py-2">ใช่</td>
        </tr>
      </tbody>
    </table>

    <h3>ผลกระทบต่อโมเดล</h3>
    <ul>
      <li>เพิ่มประสิทธิภาพของโมเดลโดยลดปริมาณข้อมูล</li>
      <li>เสริมความสามารถในการ generalize ของโมเดล</li>
      <li>ลดความต้องการในหน่วยความจำระหว่าง training</li>
    </ul>

    <div className="p-4 rounded-lg border border-blue-400 bg-blue-600 dark:bg-blue-900 dark:border-blue-500">
      <p className="font-semibold text-blue-900 dark:text-blue-600">Highlight:</p>
      <p className="text-blue-800 dark:text-blue-50">
        การใช้ Pooling ที่เหมาะสมสามารถลดการคำนวณได้ถึง 75% โดยไม่สูญเสียความสามารถในการเรียนรู้ ซึ่งเป็นประเด็นสำคัญในงานวิจัยของ He et al. (2015) ใน ResNet architecture
      </p>
    </div>

    <h3>ข้อควรระวังในการออกแบบ</h3>
    <ul>
      <li>Pooling มากเกินไปอาจทำให้สูญเสียข้อมูลเชิงตำแหน่งสำคัญ</li>
      <li>Pooling ที่ไม่เหมาะกับลักษณะของข้อมูลจะลดคุณภาพของฟีเจอร์</li>
      <li>อาจใช้เทคนิคอื่นทดแทน เช่น Strided Convolution หรือ Attention-based Reduction</li>
    </ul>

    <h3>อ้างอิง</h3>
    <ul>
      <li>LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.</li>
      <li>He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. CVPR.</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
    </ul>
  </div>
</section>


   <section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ประเภทของ Pooling</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert text-base leading-relaxed space-y-8">
    <p>
      Pooling เป็นกระบวนการสำคัญใน Convolutional Neural Networks (CNNs) ที่ทำหน้าที่ลดขนาดของ Feature Map
      เพื่อควบคุมความซับซ้อนของโมเดลและปรับปรุงประสิทธิภาพในการเรียนรู้ โดยทั่วไปสามารถแบ่ง Pooling ออกได้หลายประเภทตามวิธีการคัดเลือกค่าจากกลุ่มของพิกเซลในพื้นที่ย่อยของภาพ
    </p>

    <h3>Max Pooling</h3>
    <p>
      Max Pooling เป็นรูปแบบที่นิยมมากที่สุด โดยจะเลือกค่าที่มากที่สุดในแต่ละ receptive field ของ Feature Map
      วิธีนี้ช่วยให้โมเดลสามารถเน้นเฉพาะ feature ที่เด่นชัด และลดผลกระทบของ noise จากภาพ
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-4 rounded-md overflow-auto text-sm">
<code>
Input: 
[[1, 3, 2], 
 [5, 6, 1], 
 [4, 2, 9]]

2x2 Max Pooling (stride=1): 
[[6, 6], 
 [6, 9]]
</code>
    </pre>

    <h3>Average Pooling</h3>
    <p>
      Average Pooling จะคำนวณค่าเฉลี่ยจากแต่ละ receptive field แทนที่จะเลือกค่าสูงสุด โดยวิธีนี้ช่วยให้ได้ Feature Map ที่มีความเรียบเนียน และไม่เน้นเฉพาะจุดเด่นมากเกินไป
    </p>

    <ul className="list-disc list-inside">
      <li>ช่วยรักษาความต่อเนื่องของฟีเจอร์</li>
      <li>ลดความเสี่ยงของ overfitting สำหรับข้อมูลที่มี noise น้อย</li>
    </ul>

    <h3>Global Pooling</h3>
    <p>
      Global Pooling เป็นการลดมิติของ Feature Map ทั้งหมดใน spatial dimension เหลือเพียงค่าเดียวต่อ channel โดยวิธีนี้ใช้ได้ทั้งแบบ Global Max Pooling และ Global Average Pooling
      นิยมใช้ในเลเยอร์สุดท้ายก่อนเชื่อมกับ Fully Connected Layer
    </p>

    <table className="table-auto w-full border border-gray-300 dark:border-gray-600 text-sm">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">ประเภท</th>
          <th className="border px-4 py-2">ลักษณะ</th>
          <th className="border px-4 py-2">ข้อดี</th>
          <th className="border px-4 py-2">ตัวอย่างใช้งาน</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Max Pooling</td>
          <td className="border px-4 py-2">เลือกค่ามากที่สุดในแต่ละ window</td>
          <td className="border px-4 py-2">เน้น feature ที่เด่นชัด</td>
          <td className="border px-4 py-2">VGGNet, ResNet</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Average Pooling</td>
          <td className="border px-4 py-2">คำนวณค่าเฉลี่ยใน window</td>
          <td className="border px-4 py-2">ให้ representation ที่สมดุล</td>
          <td className="border px-4 py-2">LeNet, AlexNet</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Global Average Pooling</td>
          <td className="border px-4 py-2">ลดภาพลงเป็นค่าหนึ่งต่อ channel</td>
          <td className="border px-4 py-2">ลดจำนวนพารามิเตอร์อย่างมาก</td>
          <td className="border px-4 py-2">GoogLeNet, MobileNet</td>
        </tr>
      </tbody>
    </table>

    <div className="p-4 rounded-lg border border-yellow-400 bg-yellow-600 dark:bg-yellow-800 dark:border-yellow-600">
      <p className="font-semibold text-yellow-900 dark:text-yellow-600">Insight Box:</p>
      <p className="text-yellow-800 dark:text-yellow-50">
        จากการศึกษาของ Zhou et al. (2016) พบว่า Global Average Pooling สามารถลดความซับซ้อนของโมเดลได้โดยไม่ทำให้ความแม่นยำลดลง
        อีกทั้งยังช่วยลด overfitting ในงานจำแนกภาพที่มี class จำนวนมาก
      </p>
    </div>

    <h3>Comparison: Max vs Average Pooling</h3>
    <ul className="list-disc list-inside">
      <li><strong>Max Pooling:</strong> ทำให้โมเดล robust ต่อ noise และสามารถจับ pattern เด่นชัดได้ดี</li>
      <li><strong>Average Pooling:</strong> สะท้อน distribution โดยรวมของ feature ในแต่ละ window</li>
    </ul>

    <div className="p-4 rounded-lg border border-blue-400 bg-blue-600 dark:bg-blue-900 dark:border-blue-500">
      <p className="font-semibold text-blue-900 dark:text-blue-600">Highlight:</p>
      <p className="text-blue-800 dark:text-blue-50">
        ข้อมูลจาก Stanford CS231n ยืนยันว่า Max Pooling เหมาะสำหรับ early layer ที่ต้องการจับ edges หรือ patterns ที่ contrast สูง
        ขณะที่ Average Pooling มักใช้ใน mid- หรือ late-layer เพื่อความสม่ำเสมอ
      </p>
    </div>

    <h3>การเลือกใช้ Pooling ให้เหมาะกับงาน</h3>
    <ul className="list-disc list-inside">
      <li>สำหรับงาน Image Classification: Max Pooling เป็นตัวเลือกหลัก</li>
      <li>สำหรับ Feature Aggregation: Global Average Pooling ใช้ใน stage สุดท้าย</li>
      <li>สำหรับ Time Series หรือ Signal: อาจใช้ Average Pooling เพื่อรักษารูปแบบความต่อเนื่อง</li>
    </ul>

    <h3>อ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Zhou, B. et al. (2016). Learning Deep Features for Discriminative Localization. CVPR.</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
      <li>Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks. ICLR.</li>
    </ul>
  </div>
</section>


       <section id="parameters" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Pooling Parameters</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert text-base leading-relaxed space-y-8">
    <p>
      Pooling Layer ใน Convolutional Neural Networks (CNNs) มีพารามิเตอร์หลายชนิดที่มีผลต่อพฤติกรรมการย่อขนาดข้อมูลใน Feature Map พารามิเตอร์เหล่านี้ไม่ได้เป็นพารามิเตอร์ที่ต้องเรียนรู้เหมือนกับ weight ของ convolution แต่ต้องกำหนดอย่างเหมาะสมเพื่อให้การประมวลผลภาพเป็นไปอย่างมีประสิทธิภาพสูงสุด
    </p>

    <h3>1. Kernel Size</h3>
    <p>
      Kernel หรือ filter size คือขนาดของหน้าต่างที่ใช้ในการเลือกค่าภายในแต่ละ pooling operation โดยทั่วไปแล้วจะใช้ขนาด 2x2 หรือ 3x3
    </p>
    <ul className="list-disc list-inside">
      <li>2x2: เป็นค่ามาตรฐาน เหมาะสำหรับลดขนาดอย่างต่อเนื่อง</li>
      <li>3x3: ครอบคลุมข้อมูลมากขึ้น เหมาะกับข้อมูลที่มี texture ซับซ้อน</li>
    </ul>

    <pre className="bg-gray-600 dark:bg-gray-800 p-4 rounded-md overflow-auto text-sm">
<code>
# ตัวอย่างการใช้ใน PyTorch
nn.MaxPool2d(kernel_size=2)
</code>
    </pre>

    <h3>2. Stride</h3>
    <p>
      Stride คือจำนวนพิกเซลที่กรอบของ kernel เลื่อนในแต่ละขั้น หาก stride เท่ากับ kernel size จะไม่มีการ overlap แต่หาก stride น้อยกว่า kernel size จะเกิดการ overlap ซึ่งอาจเก็บรายละเอียดได้มากกว่า
    </p>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <h4 className="font-semibold text-lg">Stride = 2 (Non-overlapping)</h4>
        <p>ขนาดของ output ลดลงเร็ว</p>
      </div>
      <div>
        <h4 className="font-semibold text-lg">Stride = 1 (Overlapping)</h4>
        <p>ขนาดลดลงช้ากว่า แต่คง feature ได้ละเอียดกว่า</p>
      </div>
    </div>

    <h3>3. Padding</h3>
    <p>
      Padding คือการเพิ่มกรอบรอบ Feature Map ก่อนทำ pooling ซึ่งอาจช่วยให้ขนาดของ output คงที่ หรือช่วยให้ขอบภาพมีส่วนร่วมในกระบวนการคำนวณมากขึ้น
    </p>
    <ul>
      <li><strong>Valid Padding:</strong> ไม่เติม padding เลย (output เล็กลง)</li>
      <li><strong>Same Padding:</strong> เติม padding เพื่อให้ output มีขนาดเท่า input</li>
    </ul>

    <div className="p-4 rounded-lg border border-yellow-400 bg-yellow-600 dark:bg-yellow-800 dark:border-yellow-600">
      <p className="font-semibold text-yellow-900 dark:text-yellow-600">Insight Box:</p>
      <p className="text-yellow-800 dark:text-yellow-50">
        Padding อาจดูเหมือนไม่สำคัญใน pooling แต่จากงานของ Springenberg et al. (2015) พบว่า padding ที่เหมาะสมช่วยให้ convolutional models เรียนรู้ข้อมูลบริเวณขอบภาพได้ดีขึ้น โดยเฉพาะในการจำแนกวัตถุในภาพที่มีการจัดวางไม่สมมาตร
      </p>
    </div>

    <h3>4. Dilation (ในบาง framework)</h3>
    <p>
      แม้จะไม่ใช่พารามิเตอร์พื้นฐานของ pooling แต่บางระบบเช่น TensorFlow รองรับ dilation pooling ซึ่งมีช่องว่างระหว่างจุดที่ sampling โดยจะใช้กรณีพิเศษในการทำ pooling ขยายขอบเขตของ kernel
    </p>

    <pre className="bg-gray-600 dark:bg-gray-800 p-4 rounded-md overflow-auto text-sm">
<code>
tf.nn.max_pool2d(input, 
                 ksize=[1, 2, 2, 1], 
                 strides=[1, 2, 2, 1], 
                 padding='SAME', 
                 dilations=[1, 2, 2, 1])
</code>
    </pre>

    <h3>5. Output Size และ Adaptive Pooling</h3>
    <p>
      Adaptive Pooling ใช้ในกรณีที่ต้องการให้ output มีขนาดที่กำหนดล่วงหน้าโดยไม่สนใจ input size ซึ่งเหมาะกับการรวม input ที่มีขนาดไม่แน่นอนก่อนป้อนเข้า Fully Connected Layer
    </p>
    <ul>
      <li>ใช้ใน architecture สมัยใหม่ เช่น ResNet, DenseNet</li>
      <li>PyTorch รองรับ nn.AdaptiveAvgPool2d((h, w))</li>
    </ul>

    <div className="p-4 rounded-lg border border-blue-400 bg-blue-600 dark:bg-blue-900 dark:border-blue-500">
      <p className="font-semibold text-blue-900 dark:text-blue-600">Highlight:</p>
      <p className="text-blue-800 dark:text-blue-50">
        Adaptive Pooling คือหนึ่งในเทคนิคที่ช่วยให้ CNNs สามารถรับ input ได้หลายขนาดโดยไม่ต้องเปลี่ยนโครงสร้างของโมเดล ซึ่งเป็นหลักการที่อยู่เบื้องหลังความยืดหยุ่นของหลายโมเดล state-of-the-art
      </p>
    </div>

    <h3>ตารางเปรียบเทียบพารามิเตอร์</h3>
    <table className="table-auto w-full border border-gray-300 dark:border-gray-600 text-sm">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">Parameter</th>
          <th className="border px-4 py-2">Default</th>
          <th className="border px-4 py-2">ผลต่อ Output</th>
          <th className="border px-4 py-2">เหมาะสำหรับ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Kernel Size</td>
          <td className="border px-4 py-2">2x2</td>
          <td className="border px-4 py-2">ลดขนาดครึ่งหนึ่ง</td>
          <td className="border px-4 py-2">ทั่วไป</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stride</td>
          <td className="border px-4 py-2">2</td>
          <td className="border px-4 py-2">ย่อภาพเร็ว</td>
          <td className="border px-4 py-2">Efficiency</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Padding</td>
          <td className="border px-4 py-2">Valid</td>
          <td className="border px-4 py-2">ลด output size</td>
          <td className="border px-4 py-2">ขอบไม่สำคัญ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Adaptive</td>
          <td className="border px-4 py-2">N/A</td>
          <td className="border px-4 py-2">คงที่ตามกำหนด</td>
          <td className="border px-4 py-2">Input dynamic</td>
        </tr>
      </tbody>
    </table>

    <h3>อ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Springenberg et al. (2015). Striving for Simplicity: All Convolutional Net. arXiv:1412.6806</li>
      <li>MIT Deep Learning 6.S191</li>
      <li>PyTorch & TensorFlow official documentation</li>
    </ul>
  </div>
</section>


     <section id="vs-convolution" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    4. Pooling vs Convolution
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <p>
      ในการออกแบบสถาปัตยกรรมของ Convolutional Neural Networks (CNNs) นั้น Pooling Layer และ Convolutional Layer เป็นสองกลไกหลักที่มีหน้าที่แตกต่างกัน แม้จะทำงานร่วมกันเพื่อเพิ่มประสิทธิภาพในการเรียนรู้ข้อมูลเชิงพื้นที่ แต่ความแตกต่างในเชิงคณิตศาสตร์และจุดประสงค์ในการใช้งานจำเป็นต้องเข้าใจอย่างถ่องแท้เพื่อการออกแบบระบบที่แม่นยำและมีประสิทธิภาพสูงสุด
    </p>

    <div className="bg-blue-600 p-4 rounded-lg border border-blue-300 shadow">
      <p className="font-medium">
        Highlight: บทบาทของ Pooling และ Convolution ไม่สามารถทดแทนกันได้ แต่เป็นการทำงานร่วมที่เสริมสร้างความสามารถในการเรียนรู้ฟีเจอร์เชิงลึก
      </p>
    </div>

    <h3 className="text-xl font-semibold">การทำงานของ Convolutional Layer</h3>
    <p>
      Convolutional Layer ทำหน้าที่ในการดึงฟีเจอร์ (Feature Extraction) จากอินพุตผ่านการคำนวณเชิงเส้นและไม่เชิงเส้น โดยใช้ Kernel (หรือ Filter) เลื่อนผ่านอินพุตและคำนวณค่าจากการคูณและบวกค่าที่ตำแหน่งนั้น ๆ ซึ่งมักจะตามด้วยการใช้ฟังก์ชัน activation เช่น ReLU เพื่อเพิ่ม non-linearity ให้กับระบบ
    </p>

    <h3 className="text-xl font-semibold">การทำงานของ Pooling Layer</h3>
    <p>
      ในทางตรงกันข้าม Pooling Layer ไม่มีพารามิเตอร์ที่เรียนรู้ได้ (non-learnable) และใช้สำหรับลดขนาดเชิงพื้นที่ของ feature map เพื่อลดจำนวนพารามิเตอร์และภาระในการคำนวณ อีกทั้งยังช่วยลดความเสี่ยงจาก overfitting โดยการเก็บค่าที่สำคัญที่สุดในแต่ละกลุ่มของข้อมูล
    </p>

    <h3 className="text-xl font-semibold">การเปรียบเทียบเชิงโครงสร้าง</h3>
    <table className="table-auto w-full text-left border mt-4">
      <thead>
        <tr className="bg-gray-600 dark:bg-gray-800">
          <th className="px-4 py-2 border">คุณสมบัติ</th>
          <th className="px-4 py-2 border">Convolution</th>
          <th className="px-4 py-2 border">Pooling</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Learnable Parameters</td>
          <td className="border px-4 py-2">มี (Kernel weights)</td>
          <td className="border px-4 py-2">ไม่มี</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-900">
          <td className="border px-4 py-2">Function</td>
          <td className="border px-4 py-2">เรียนรู้ pattern</td>
          <td className="border px-4 py-2">ลดขนาดข้อมูล</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Invariance</td>
          <td className="border px-4 py-2">บางส่วน</td>
          <td className="border px-4 py-2">สูง (เช่น shift invariance)</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-900">
          <td className="border px-4 py-2">หลักการคำนวณ</td>
          <td className="border px-4 py-2">Dot product + bias</td>
          <td className="border px-4 py-2">Max หรือ Average</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-600 p-4 rounded-lg border border-yellow-300 shadow mt-8">
      <p className="font-medium">
        Insight: แม้ Convolutional Layers จะสามารถใช้ stride เพื่อลดขนาด spatial ได้เช่นเดียวกับ Pooling แต่ Pooling มีลักษณะที่คงความเด่นของฟีเจอร์ในลักษณะที่ robust กว่า
      </p>
    </div>

    <h3 className="text-xl font-semibold">แนวโน้มในงานวิจัยสมัยใหม่</h3>
    <p>
      ในหลายงานวิจัยสมัยใหม่เช่น Network in Network และ GoogLeNet มีการทดลองใช้ Strided Convolution แทน Pooling Layer เพื่อให้การลดขนาดยังคงสามารถเรียนรู้ได้ แต่ก็มีข้อเสียเรื่องความซับซ้อนและจำนวนพารามิเตอร์ที่เพิ่มขึ้น จึงขึ้นกับการออกแบบว่าในแต่ละกรณีใดเหมาะสมกว่ากัน
    </p>

    <ul className="list-disc pl-6">
      <li>Network in Network: ลด Pooling เพื่อใช้ Global Average Pooling แทน</li>
      <li>ResNet: ใช้ combination ระหว่าง Pooling และ Strided Conv</li>
      <li>MobileNet: ลดความลึกของ Pooling เพื่อให้เหมาะกับอุปกรณ์พกพา</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc pl-6">
      <li>LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature.</li>
      <li>Szegedy, C. et al. (2015). Going Deeper with Convolutions. CVPR.</li>
      <li>Lin, M., Chen, Q., & Yan, S. (2013). Network in Network. arXiv.</li>
      <li>He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.</li>
      <li>Howard, A. G. et al. (2017). MobileNets: Efficient Convolutional Neural Networks. arXiv.</li>
    </ul>
  </div>
</section>


<section id="spatial-reduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    5. Spatial Reduction Strategy
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">
    <p>
      การจัดการขนาดของข้อมูลเชิงพื้นที่ (spatial dimensions) เป็นหนึ่งในแกนหลักของการออกแบบสถาปัตยกรรม Deep Learning โดยเฉพาะใน Convolutional Neural Networks (CNNs) และสถาปัตยกรรมภาพอื่น ๆ
      การลดมิติของ spatial feature map ให้เหมาะสมสามารถเพิ่มประสิทธิภาพการเรียนรู้ ลดเวลาการฝึก และเพิ่ม generalization ได้อย่างมาก
    </p>

    <div className="bg-yellow-600 p-4 rounded-lg border border-yellow-300 shadow">
      <p className="font-medium">
        Insight: การวางกลยุทธ์ในการลดขนาด spatial dimension มีผลโดยตรงต่อการอนุมานเชิงความเร็ว (inference speed), ความสามารถในการจับฟีเจอร์, และความสามารถในการปรับใช้โมเดลจริง
      </p>
    </div>

    <h3 className="text-xl font-semibold">เป้าหมายของการลดมิติ Spatial</h3>
    <ul className="list-disc pl-6">
      <li>ลดขนาดของ feature map เพื่อลดจำนวนคำนวณ</li>
      <li>ลด overfitting โดยลดจำนวนพารามิเตอร์และหน่วยประมวลผล</li>
      <li>เพิ่ม receptive field ให้สามารถจับบริบทกว้างขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">เทคนิคหลักที่ใช้ในการลด spatial dimension</h3>
    <p>เทคนิคที่นิยมใช้มีอยู่หลายวิธี โดยแต่ละวิธีมีลักษณะเฉพาะที่ต้องเลือกใช้ให้เหมาะกับสถาปัตยกรรมของโมเดล</p>

    <h4 className="text-lg font-semibold">1. Max Pooling / Average Pooling</h4>
    <p>
      เป็นเทคนิคแบบ deterministic ที่เลือกค่ามากที่สุดหรือค่าเฉลี่ยจากกลุ่มของพิกเซลใน feature map ตามขนาด kernel ที่กำหนด การใช้ pooling layers เหล่านี้ช่วยลด resolution โดยไม่ต้องมีพารามิเตอร์ใด ๆ เพิ่ม
    </p>

    <h4 className="text-lg font-semibold">2. Strided Convolution</h4>
    <p>
      ใช้ Convolutional layer ที่มี stride มากกว่า 1 ในการลดขนาด spatial dimension ในขณะที่ยังสามารถเรียนรู้ pattern ไปพร้อมกันได้ ข้อดีคือสามารถเรียนรู้การลดมิติแบบ adaptive มากกว่า pooling
    </p>

    <h4 className="text-lg font-semibold">3. Global Pooling</h4>
    <p>
      เทคนิคเช่น Global Average Pooling (GAP) หรือ Global Max Pooling ใช้เพื่อยุบ spatial dimension ทั้งหมดลงเป็นเวกเตอร์เดียว โดยเฉพาะอย่างยิ่งใช้ในขั้นตอนท้ายของสถาปัตยกรรม CNN ก่อน fully connected layer
    </p>

    <div className="bg-blue-600 p-4 rounded-lg border border-blue-300 shadow">
      <p className="font-medium">
        Highlight: Global Average Pooling ถูกใช้แทน fully connected layers ในสถาปัตยกรรมเช่น Network-in-Network และ GoogLeNet เพื่อลดจำนวนพารามิเตอร์ลงอย่างมหาศาล
      </p>
    </div>

    <h4 className="text-lg font-semibold">4. Adaptive Pooling</h4>
    <p>
      Adaptive Pooling ปรับขนาด spatial dimension ให้อัตโนมัติให้มีขนาดที่กำหนดไว้ล่วงหน้า ใช้ประโยชน์มากในงานที่ต้องการ output คงที่ เช่นการเชื่อมต่อกับ RNN หรือ Transformer
    </p>

    <h4 className="text-lg font-semibold">5. Attention-based Downsampling</h4>
    <p>
      โมเดลบางรุ่น เช่น Vision Transformer (ViT) และ Swin Transformer ใช้การลด spatial โดยอิงกับ attention weights หรือการแบ่ง patch ที่คำนวณบน grid ซึ่งให้ flexibility สูงกว่าการใช้ kernel คงที่
    </p>

    <h3 className="text-xl font-semibold">เปรียบเทียบเทคนิคต่าง ๆ</h3>
    <table className="table-auto w-full border mt-4 text-left">
      <thead>
        <tr className="bg-gray-600 dark:bg-gray-800">
          <th className="px-4 py-2 border">เทคนิค</th>
          <th className="px-4 py-2 border">มีพารามิเตอร์</th>
          <th className="px-4 py-2 border">สามารถเรียนรู้ได้</th>
          <th className="px-4 py-2 border">ความยืดหยุ่น</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border">Max/Average Pooling</td>
          <td className="px-4 py-2 border">ไม่มี</td>
          <td className="px-4 py-2 border">ไม่</td>
          <td className="px-4 py-2 border">ต่ำ</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-900">
          <td className="px-4 py-2 border">Strided Conv</td>
          <td className="px-4 py-2 border">มี</td>
          <td className="px-4 py-2 border">ใช่</td>
          <td className="px-4 py-2 border">กลาง</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Global Pooling</td>
          <td className="px-4 py-2 border">ไม่มี</td>
          <td className="px-4 py-2 border">ไม่</td>
          <td className="px-4 py-2 border">กลาง</td>
        </tr>
        <tr className="bg-gray-50 dark:bg-gray-900">
          <td className="px-4 py-2 border">Adaptive Pooling</td>
          <td className="px-4 py-2 border">ไม่มี</td>
          <td className="px-4 py-2 border">ไม่</td>
          <td className="px-4 py-2 border">สูง</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border">Attention-based</td>
          <td className="px-4 py-2 border">มี</td>
          <td className="px-4 py-2 border">ใช่</td>
          <td className="px-4 py-2 border">สูงมาก</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ผลกระทบต่อโมเดลในระยะยาว</h3>
    <ul className="list-disc pl-6">
      <li>การลด spatial dimension อย่างรุนแรงเร็วเกินไปอาจทำให้สูญเสียฟีเจอร์สำคัญใน early layers</li>
      <li>การลดช้าหรือไม่ลดเลยอาจทำให้โมเดล overfit หรือใช้ทรัพยากรมากเกินจำเป็น</li>
      <li>การใช้ hybrid strategy เช่น combination ระหว่าง Pooling และ Strided Conv ช่วยสร้างความสมดุล</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc pl-6">
      <li>He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.</li>
      <li>Szegedy, C. et al. (2015). Going Deeper with Convolutions. CVPR.</li>
      <li>Lin, M. et al. (2013). Network in Network. arXiv.</li>
      <li>Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition. ICLR.</li>
      <li>Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer. ICCV.</li>
    </ul>
  </div>
</section>


      <section id="overlap" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    6. Overlapping vs Non-Overlapping Pooling
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <p>
      กลยุทธ์ของการ Pooling ใน Convolutional Neural Networks (CNNs) ถูกออกแบบมาเพื่อลดขนาดของ feature maps
      และควบคุมการ overfitting โดยการสกัดสาระสำคัญจาก spatial dimension ของข้อมูลภาพอย่างมีประสิทธิภาพ
      ในทางปฏิบัติ การออกแบบลักษณะของ pooling layer มีผลต่อคุณภาพของการเรียนรู้ โดยเฉพาะความแตกต่างระหว่าง
      <strong>Overlapping Pooling</strong> และ <strong>Non-Overlapping Pooling</strong> ซึ่งสะท้อนให้เห็นถึงวิธีการเคลื่อน window
      ในการสรุปค่า
    </p>

    <h3 className="text-xl font-semibold">6.1 นิยามและโครงสร้างของการ Pooling</h3>
    <p>
      Non-overlapping pooling คือการเลือกขนาด kernel เท่ากับ stride (เช่น kernel=2, stride=2) ทำให้ไม่เกิดการซ้อนทับของ region
      ในขณะที่ overlapping pooling ใช้ค่า stride ที่น้อยกว่าขนาดของ kernel (เช่น kernel=3, stride=2) ซึ่งทำให้ region ของการ pooling
      ซ้อนกันบางส่วน และมีโอกาสเกิดการเรียนรู้ feature ที่ละเอียดขึ้น
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-500 p-4 rounded-md shadow-sm">
      <p className="font-medium">
        Highlight:
      </p>
      <p>
        Overlapping pooling ทำให้ network เก็บรักษารายละเอียดในพื้นที่เชิงพื้นที่ (spatial details) ได้ดีกว่า
        โดยเฉพาะในชั้นต้นของ CNN ที่ข้อมูลยังคงมีความละเอียดสูง
      </p>
    </div>

    <h3 className="text-xl font-semibold">6.2 ข้อดีของ Overlapping Pooling</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ลดความเสี่ยงจากข้อมูลสูญหายระหว่าง pooling เนื่องจากมีการซ้อนทับ</li>
      <li>ให้ spatial invariance ในขนาดที่ดีขึ้นโดยเฉพาะภาพที่มีลักษณะ feature ที่คล้ายกันบริเวณใกล้เคียง</li>
      <li>ช่วยให้โมเดลเรียนรู้ texture และ pattern ละเอียดขึ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">6.3 จุดอ่อนของ Overlapping Pooling</h3>
    <p>
      แม้จะมีข้อดีด้านการเก็บรายละเอียด การใช้ overlapping pooling ยังมีข้อเสียในเรื่อง computational cost ที่เพิ่มขึ้น
      และการเพิ่มความซับซ้อนในการออกแบบ model ที่อาจนำไปสู่ overfitting หากไม่มีการ regularization ที่เพียงพอ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-500 p-4 rounded-md shadow-sm">
      <p className="font-medium">Insight Box:</p>
      <p>
        การเลือกใช้ Overlapping หรือ Non-Overlapping Pooling ขึ้นอยู่กับลักษณะของข้อมูล
        เช่น งานที่เกี่ยวข้องกับ texture หรือ pattern ที่ละเอียดมาก (เช่น medical imaging, satellite imagery)
        มักได้ผลดีกว่าด้วย overlapping pooling
      </p>
    </div>

    <h3 className="text-xl font-semibold">6.4 เปรียบเทียบระหว่างสองกลยุทธ์</h3>
    <table className="table-auto w-full text-left border border-gray-300">
      <thead className="bg-gray-600">
        <tr>
          <th className="border px-4 py-2">Feature</th>
          <th className="border px-4 py-2">Overlapping Pooling</th>
          <th className="border px-4 py-2">Non-Overlapping Pooling</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Stride vs Kernel</td>
          <td className="border px-4 py-2">Stride &lt; Kernel</td>
          <td className="border px-4 py-2">Stride = Kernel</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Spatial Detail</td>
          <td className="border px-4 py-2">ดีกว่า</td>
          <td className="border px-4 py-2">ต่ำกว่า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Computation</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">ต่ำ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Risk of Overfitting</td>
          <td className="border px-4 py-2">มากขึ้น</td>
          <td className="border px-4 py-2">น้อยลง</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">6.5 แนวทางจากงานวิจัย</h3>
    <p>
      งานวิจัยจาก Krizhevsky et al. (2012) ใน AlexNet ใช้ overlapping pooling และได้ผลลัพธ์ที่ดีกว่า non-overlapping
      ในการแข่งขัน ImageNet ซึ่งชี้ให้เห็นถึงประสิทธิภาพของเทคนิคนี้
    </p>

    <h3 className="text-xl font-semibold">6.6 สรุปและข้อเสนอแนะ</h3>
    <p>
      การเลือกใช้ Overlapping หรือ Non-Overlapping Pooling ควรเป็นการตัดสินใจเชิงกลยุทธ์
      โดยพิจารณาถึงลักษณะของข้อมูล ปริมาณ computation ที่มีอยู่ และความซับซ้อนของ model ที่ยอมรับได้
    </p>

    <h3 className="text-xl font-semibold">6.7 แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NIPS.</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition.</li>
      <li>He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning. arXiv.</li>
    </ul>

  </div>
</section>


    <section id="translation-invariance" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Effect of Pooling on Translation Invariance</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <p>
      หนึ่งในคุณสมบัติสำคัญที่ทำให้ Convolutional Neural Networks (CNNs) ทรงพลังในด้านการประมวลผลภาพคือความสามารถในการรองรับการแปรผันของตำแหน่ง (translation invariance) ซึ่งหมายถึงการที่โมเดลยังสามารถจดจำลักษณะเด่นของวัตถุในภาพได้แม้ตำแหน่งจะเปลี่ยนไปเล็กน้อย โดย pooling layers โดยเฉพาะ max pooling มีบทบาทสำคัญในการช่วยให้เครือข่ายมีคุณสมบัตินี้เพิ่มขึ้นอย่างเป็นธรรมชาติ
    </p>

    <h3 className="text-xl font-semibold">พื้นฐานของ Translation Invariance ใน CNNs</h3>
    <p>
      ในระบบที่ไม่ใช้ pooling เช่น MLPs หรือ convolutional layer อย่างเดียว การเปลี่ยนตำแหน่งของวัตถุในภาพอาจส่งผลให้ feature map เปลี่ยนไปอย่างมีนัยสำคัญ เพราะ spatial structure มีบทบาทโดยตรงในการเรียนรู้ feature แต่การใส่ pooling layer เข้ามาจะช่วยลดความไวต่อการเปลี่ยนตำแหน่งของ pixel ในลักษณะ localized
    </p>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
      <strong>Insight Box:</strong>
      <p className="mt-2">
        การใช้ pooling layer อย่าง max pooling ช่วยเน้น feature ที่เด่นที่สุดภายใน receptive field และลด sensitivity ต่อการเลื่อนของวัตถุใน input image ส่งผลให้เครือข่ายสามารถ generalize ได้ดีขึ้นต่อภาพที่ไม่ได้จัดวางในตำแหน่งเดียวกันกับ training set
      </p>
    </div>

    <h3 className="text-xl font-semibold">ผลกระทบของ Pooling ต่อการเรียนรู้ Feature</h3>
    <p>
      แม้ pooling จะช่วยเรื่อง invariance แต่ก็มี trade-off เพราะ pooling ยังทำให้โมเดลเสีย spatial precision ไปบางส่วน โดยเฉพาะเมื่อ pooling size มีขนาดใหญ่เกินไป และ stride สูงเกินไป จนอาจทำให้บาง feature หายไปในระหว่างการ downsample
    </p>

    <table className="table-auto w-full border border-gray-300 text-left">
      <thead>
        <tr className="bg-gray-600 dark:bg-gray-800">
          <th className="px-4 py-2 border">Pooling Type</th>
          <th className="px-4 py-2 border">Translation Invariance</th>
          <th className="px-4 py-2 border">Spatial Precision</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Max Pooling</td>
          <td className="border px-4 py-2">สูง</td>
          <td className="border px-4 py-2">ปานกลาง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Average Pooling</td>
          <td className="border px-4 py-2">ปานกลาง</td>
          <td className="border px-4 py-2">สูงกว่า Max Pooling</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Global Pooling</td>
          <td className="border px-4 py-2">สูงมาก</td>
          <td className="border px-4 py-2">ต่ำ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">การใช้งานในระบบจริง</h3>
    <ul className="list-disc pl-5">
      <li>การจดจำใบหน้าที่อยู่ในตำแหน่งต่าง ๆ ของภาพ</li>
      <li>การตรวจจับวัตถุในภาพจากกล้องที่เคลื่อนไหว</li>
      <li>การวิเคราะห์ภาพทางการแพทย์ที่อาจมีการขยับของกล้องจุลทรรศน์</li>
    </ul>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <strong>Highlight:</strong>
      <p className="mt-2">
        Translation invariance ที่เกิดจาก pooling เป็นหนึ่งในปัจจัยหลักที่ทำให้ CNNs เหมาะสมกับการประมวลผลภาพมากกว่าโครงข่ายแบบ fully-connected
      </p>
    </div>

    <h3 className="text-xl font-semibold">การทดแทน Pooling ด้วย Convolution</h3>
    <p>
      มีการศึกษาบางส่วน เช่นงานวิจัยของ Springenberg et al. (2015) ที่เสนอให้ใช้ strided convolution แทนการ pooling เพื่อให้เครือข่ายเรียนรู้การ downsample อย่างยืดหยุ่นโดยไม่สูญเสีย spatial precision มากเกินไป ซึ่งสามารถใช้ร่วมกับการเพิ่ม data augmentation เพื่อช่วยให้เกิด translation invariance ในอีกมิติหนึ่ง
    </p>

    <h3 className="text-xl font-semibold">สรุปผลกระทบและแนวทางในอนาคต</h3>
    <p>
      แม้ pooling จะมีข้อดีในด้านการสร้าง translation invariance แต่การใช้อย่างเหมาะสม รวมถึงการทดลองกับ convolutional alternatives อาจช่วยเพิ่มความสามารถในการจดจำภาพโดยไม่สูญเสีย resolution และความแม่นยำมากนัก
    </p>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc pl-5">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>Springenberg et al., "Striving for Simplicity: The All Convolutional Net", arXiv:1412.6806</li>
      <li>LeCun, Y., Bengio, Y., Hinton, G. "Deep learning". Nature 521, 436–444 (2015)</li>
      <li>MIT 6.S191: Introduction to Deep Learning - Lecture on CNNs</li>
    </ul>

  </div>
</section>


    <section id="alternatives" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Pooling Alternatives (Modern View)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <p>
      ในระบบ Convolutional Neural Networks (CNNs) แบบดั้งเดิม การใช้ pooling layer เช่น max pooling หรือ average pooling เป็นส่วนสำคัญที่ช่วยในการลดขนาดเชิงพื้นที่ของ feature maps เพื่อเพิ่มประสิทธิภาพในการประมวลผล และลด overfitting อย่างไรก็ตาม งานวิจัยสมัยใหม่ได้เริ่มท้าทายแนวคิดนี้ และเสนอวิธีการอื่นที่สามารถทดแทนหรือเสริมแทน pooling layers ได้ โดยมุ่งเน้นให้การเรียนรู้เชิงลึกสามารถ retain spatial information ได้ดีกว่าเดิม และทำงานได้อย่างมีประสิทธิภาพมากขึ้น
    </p>

    <h3 className="text-xl font-semibold">ข้อจำกัดของ Pooling Layers แบบดั้งเดิม</h3>
    <p>
      แม้ว่า pooling จะช่วยให้เครือข่ายสามารถบรรลุ translation invariance และลดขนาด feature map ได้ แต่ข้อเสียของ pooling คือการทำลายรายละเอียดเชิงพื้นที่ที่อาจมีความสำคัญต่อการตัดสินใจในระดับสูง โดยเฉพาะในงานที่ต้องการ spatial precision เช่น semantic segmentation หรือ medical image analysis
    </p>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-xl">
      <strong>Insight Box:</strong>
      <p className="mt-2">
        ในงานวิจัยเช่น "Striving for Simplicity: The All Convolutional Net" โดย Springenberg et al. (2015) ได้เสนอให้ลบ pooling ออกทั้งหมดและแทนที่ด้วย convolutional layers ที่มี stride มากขึ้น ซึ่งช่วยให้เครือข่ายสามารถเรียนรู้การ downsample ได้โดยตรง
      </p>
    </div>

    <h3 className="text-xl font-semibold">1. Strided Convolutions</h3>
    <p>
      แทนการใช้ pooling layer นักวิจัยบางกลุ่มเลือกใช้ convolutional layer ที่มี stride มากกว่า 1 (เช่น stride = 2) เพื่อทำการลดขนาดของ feature map ข้อดีของวิธีนี้คือสามารถเรียนรู้การลดขนาดที่เหมาะสมได้โดยไม่ทำลาย spatial features แบบ abrupt
    </p>

    <pre className="bg-gray-600 rounded-lg p-4 overflow-auto text-sm">
      <code>
        {`# ตัวอย่าง pseudocode ใน PyTorch
nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)`}
      </code>
    </pre>

    <ul className="list-disc pl-5">
      <li>ช่วยให้ network สามารถเรียนรู้ spatially-aware downsampling</li>
      <li>ลด hyperparameter เพิ่มเติม เช่น kernel size ของ pooling</li>
      <li>ทำให้ network มีความเป็น differentiable ทั้งหมด</li>
    </ul>

    <h3 className="text-xl font-semibold">2. Dilated Convolutions (Atrous Convolutions)</h3>
    <p>
      Dilated convolutions เป็นเทคนิคที่ช่วยเพิ่ม receptive field โดยไม่เพิ่มจำนวนพารามิเตอร์ หรือความซับซ้อนของโมเดล วิธีนี้ใช้การแทรก zero ระหว่าง kernel weights และช่วยให้สามารถควบคุม resolution ของ output ได้โดยไม่สูญเสีย context
    </p>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-xl">
      <strong>Highlight:</strong>
      <p className="mt-2">
        การใช้ dilated convolutions นิยมอย่างแพร่หลายใน segmentation models เช่น DeepLab และ WaveNet โดยเฉพาะในงานด้าน temporal และ spatial sequence modeling
      </p>
    </div>

    <h3 className="text-xl font-semibold">3. Attention-Based Downsampling</h3>
    <p>
      แนวทางสมัยใหม่ที่เกิดขึ้นในยุคของ Transformer คือการใช้ attention mechanism เพื่อทำการลดขนาด feature map โดยเน้นที่ตำแหน่งที่สำคัญที่สุดแทนการลดขนาดแบบคงที่ เช่น pooling หรือ convolution แบบ fix kernel
    </p>

    <p>
      เทคนิคนี้ช่วยให้เครือข่ายเรียนรู้ที่จะโฟกัสไปยัง features ที่สำคัญจริง ๆ แทนที่จะ treat ทุกตำแหน่งใน feature map อย่างเท่าเทียมกัน
    </p>

    <ul className="list-disc pl-5">
      <li>เหมาะกับการประมวลผลภาพที่มีหลายวัตถุ หรือ complexity สูง</li>
      <li>ใช้พลังการคำนวณสูงกว่าการ pooling ปกติ</li>
      <li>ให้ผลลัพธ์ที่แม่นยำและมี contextual understanding สูงกว่า</li>
    </ul>

    <h3 className="text-xl font-semibold">4. Learnable Pooling Methods</h3>
    <p>
      แทนการใช้ max หรือ average pooling แบบคงที่ นักวิจัยบางกลุ่มเสนอให้ใช้ pooling ที่เรียนรู้ได้ เช่น Lp pooling, gated pooling หรือ methods ที่ใช้ neural networks มาช่วยในการเลือกค่าภายใน receptive field
    </p>

    <table className="table-auto w-full border border-gray-300 text-left">
      <thead>
        <tr className="bg-gray-600 dark:bg-gray-800">
          <th className="px-4 py-2 border">Method</th>
          <th className="px-4 py-2 border">Learnable</th>
          <th className="px-4 py-2 border">Adaptivity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Max Pooling</td>
          <td className="border px-4 py-2">No</td>
          <td className="border px-4 py-2">Low</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Gated Pooling</td>
          <td className="border px-4 py-2">Yes</td>
          <td className="border px-4 py-2">High</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Attention Pooling</td>
          <td className="border px-4 py-2">Yes</td>
          <td className="border px-4 py-2">Very High</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ผลกระทบต่อการออกแบบสถาปัตยกรรม</h3>
    <p>
      การเปลี่ยนไปใช้ alternatives เหล่านี้ต้องพิจารณาด้าน computational cost, compatibility กับ architecture ที่มีอยู่, และความเข้าใจในลักษณะข้อมูล เพราะแม้จะเพิ่มประสิทธิภาพในบางด้าน แต่ก็อาจทำให้ latency เพิ่มขึ้นในงานที่ต้อง real-time
    </p>

    <h3 className="text-xl font-semibold">เอกสารอ้างอิง</h3>
    <ul className="list-disc pl-5">
      <li>Springenberg et al., "Striving for Simplicity: The All Convolutional Net", arXiv:1412.6806</li>
      <li>Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions", arXiv:1511.07122</li>
      <li>Hu et al., "Squeeze-and-Excitation Networks", IEEE CVPR 2018</li>
      <li>Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021</li>
      <li>MIT 6.S191: Deep Learning - Architectural Advances in CNNs</li>
    </ul>

  </div>
</section>


      <section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Visualization: Before vs After Pooling</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <p>
      การทำความเข้าใจผลกระทบของการใช้ Pooling ในเครือข่าย Convolutional Neural Networks (CNNs)
      สามารถอธิบายได้อย่างชัดเจนผ่านการวิเคราะห์เชิงภาพหรือการทำ Visualization ซึ่งช่วยให้สามารถเห็นการเปลี่ยนแปลงของ Feature Maps อย่างเป็นรูปธรรม
      โดยเฉพาะอย่างยิ่ง การเปรียบเทียบระหว่างภาพก่อนและหลังการ Pooling แสดงให้เห็นถึงการสูญเสียข้อมูลบางส่วน ควบคู่กับการสกัดคุณลักษณะสำคัญ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Insight:</strong> Visualization ช่วยให้สามารถตรวจสอบว่า Feature Map ยังคงรักษาโครงสร้างของวัตถุไว้ได้แม้จะมีการลดขนาดผ่าน Pooling โดยเฉพาะอย่างยิ่งใน Max Pooling
        ซึ่งเน้นเฉพาะลักษณะที่เด่นที่สุดในแต่ละ Receptive Field
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1. Feature Map ก่อนการ Pooling</h3>
    <p>
      ภาพก่อนการ Pooling มีความละเอียดสูง โดยมีข้อมูลเชิงพื้นที่ครบถ้วนในระดับ Pixel-Level ทำให้สามารถตรวจจับขอบ รูปทรง และลักษณะเฉพาะของวัตถุได้อย่างละเอียด
    </p>

    <ul className="list-disc list-inside">
      <li>มี Spatial Resolution สูง</li>
      <li>สามารถระบุตำแหน่งของ Feature ได้แม่นยำ</li>
      <li>ความซับซ้อนในการคำนวณยังค่อนข้างสูง</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2. Feature Map หลังการ Pooling</h3>
    <p>
      หลังผ่านการ Pooling (เช่น Max หรือ Average Pooling) ข้อมูลบางส่วนจะถูกลดรูป (downsampled)
      โดยเหลือเฉพาะ Feature ที่สำคัญ ทำให้ภาพมีขนาดเล็กลง ลดจำนวนพารามิเตอร์และเพิ่ม Robustness ต่อการเคลื่อนย้าย (translation)
    </p>

    <ul className="list-disc list-inside">
      <li>ลดขนาดภาพเพื่อเพิ่มความเร็วในการประมวลผล</li>
      <li>เพิ่มความต้านทานต่อ Noise หรือการเลื่อนตำแหน่ง</li>
      <li>อาจสูญเสียข้อมูลสำคัญบางส่วน</li>
    </ul>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Highlight:</strong> การเลือกขนาดของ Kernel และ Stride มีผลโดยตรงต่อความสมดุลระหว่างความแม่นยำและความสามารถในการลดมิติของข้อมูล
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">3. ตัวอย่างเชิงเปรียบเทียบ</h3>
    <p>
      พิจารณาภาพขนาด 32x32 Pixel เมื่อผ่าน Max Pooling ขนาด 2x2 ด้วย Stride 2 จะกลายเป็นขนาด 16x16 ซึ่งลดลงถึง 75% ของจำนวน Pixel
    </p>

    <table className="table-auto w-full text-left border-collapse mt-4">
      <thead>
        <tr>
          <th className="border px-4 py-2">ระดับ</th>
          <th className="border px-4 py-2">ขนาด Feature Map</th>
          <th className="border px-4 py-2">ข้อดี</th>
          <th className="border px-4 py-2">ข้อเสีย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">ก่อน Pooling</td>
          <td className="border px-4 py-2">32x32</td>
          <td className="border px-4 py-2">รายละเอียดสูง</td>
          <td className="border px-4 py-2">คำนวณช้า</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">หลัง Pooling</td>
          <td className="border px-4 py-2">16x16</td>
          <td className="border px-4 py-2">เร็วขึ้น, ทนต่อการเลื่อนตำแหน่ง</td>
          <td className="border px-4 py-2">สูญเสียรายละเอียดบางส่วน</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-10">4. การนำ Visualization ไปใช้ในงานวิจัย</h3>
    <p>
      หลายงานวิจัยชั้นนำ เช่น Zeiler & Fergus (2014) และ Yosinski et al. (2015) ใช้เทคนิค Visualization เพื่อตรวจสอบว่า CNN เรียนรู้อะไรในแต่ละ Layer
      โดยการสร้างภาพที่แสดงว่า Neuron ใดตอบสนองกับ Feature แบบใด
    </p>

    <p>
      งานเหล่านี้ช่วยเปิดเผยว่า Pooling Layer ส่งผลต่อความสามารถในการจับโครงสร้างภาพระดับสูง โดยลด Noise และเพิ่มความนัยทางเชิงโครงสร้างให้เด่นชัด
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Insight:</strong> Visualization ไม่เพียงแต่ใช้เพื่อ Debug โมเดลเท่านั้น แต่ยังช่วยสร้างความเข้าใจเชิงลึกต่อการทำงานของโครงข่ายประสาทเทียมในเชิงทฤษฎีอีกด้วย
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">5. สรุปและแหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In ECCV.</li>
      <li>Yosinski, J., et al. (2015). Understanding neural networks through deep visualization. In arXiv:1506.06579</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Deep Learning for Self-Driving Cars (Lectures on Visual Perception)</li>
    </ul>

  </div>
</section>


   <section id="use-cases" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Use Cases ที่ใช้ Pooling</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <p>
      ในระบบ Deep Learning โดยเฉพาะโครงข่าย Convolutional Neural Networks (CNNs) การใช้ Pooling Layer
      ไม่ได้มีเพียงเป้าหมายเพื่อการลดขนาดของข้อมูลเท่านั้น แต่ยังมีบทบาทเชิงกลยุทธ์ในหลากหลายกรณีศึกษาและงานวิจัย
      ที่ครอบคลุมทั้งด้าน Computer Vision, Medical Imaging, Natural Language Processing และ Edge Computing
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Insight:</strong> Pooling Layer คือกลไกที่ช่วยให้โมเดลมีความทนทานต่อการแปรผันทางตำแหน่ง (Translation Invariance)
        และลดความซับซ้อนของข้อมูล โดยไม่สูญเสีย feature ที่สำคัญ
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">1. Image Classification</h3>
    <p>
      หนึ่งใน use case ที่โด่งดังที่สุดของการใช้ Pooling คือในการสร้างระบบจำแนกภาพ (Image Classification)
      เช่น AlexNet (2012) และ VGGNet (2014) ซึ่งใช้ Max Pooling เป็นแกนกลางในกระบวนการลดขนาดของ Feature Maps
      ทำให้โมเดลสามารถเรียนรู้ภาพเชิงลึกโดยมีพารามิเตอร์น้อยลงและลดความเสี่ยงของ Overfitting
    </p>

    <ul className="list-disc list-inside">
      <li>ใช้ Max Pooling หลัง Convolution เพื่อคง feature สำคัญ</li>
      <li>เพิ่มความสามารถในการจำแนกวัตถุแม้มีการเปลี่ยนมุมมองเล็กน้อย</li>
      <li>ใช้ร่วมกับ Fully Connected Layers เพื่อสร้าง Final Prediction</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">2. Object Detection & Localization</h3>
    <p>
      ในงานตรวจจับวัตถุ เช่น YOLO และ Faster R-CNN การ Pooling ถูกใช้เพื่อรักษาคุณลักษณะเชิงตำแหน่งในขณะที่ลดขนาดข้อมูล
      โดยเฉพาะใน ROI Pooling (Region of Interest Pooling) ซึ่งเป็นการนำ Pooling มาใช้กับ Feature Map ในส่วนที่สนใจ
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Highlight:</strong> ROI Pooling ช่วยให้สามารถนำ Feature Maps ไปประมวลผลด้วย Dense Layers
        ได้โดยไม่สูญเสียบริบทของตำแหน่งที่เกี่ยวข้องกับวัตถุเป้าหมาย
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">3. Semantic Segmentation</h3>
    <p>
      แม้ Semantic Segmentation ต้องการการรักษาความละเอียด แต่ Global Pooling อย่าง Global Average Pooling (GAP)
      ก็ถูกใช้ในช่วงท้ายของบางสถาปัตยกรรม เช่น DeepLab และ PSPNet เพื่อสกัด Contextual Features ทั่วภาพ
    </p>

    <ul className="list-disc list-inside">
      <li>GAP ช่วยลดขนาด Feature Map เป็นค่าค่าเฉลี่ยภาพรวม</li>
      <li>สามารถใช้ในการตัดสินการจัดประเภทของภาพระดับ global</li>
      <li>ลดปัญหา overfitting ที่พบใน Fully Connected Layers</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">4. Medical Imaging</h3>
    <p>
      ในงานเช่น MRI, CT Scan หรือ Histopathological Image Analysis การใช้ Pooling Layer มีส่วนช่วยให้โมเดลมีความ Generalizable มากขึ้น
      โดยลดความละเอียดที่ไม่สำคัญ และช่วยลดจำนวน Feature Dimensions ก่อนนำไปประมวลผลต่อ
    </p>

    <div className="bg-yellow-600 border-l-4 border-yellow-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Insight:</strong> Max Pooling ถูกใช้เพื่อเน้นบริเวณที่มีความเข้มของสัญญาณสูง ซึ่งมักสัมพันธ์กับบริเวณที่มีความเสี่ยงทางพยาธิวิทยา
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">5. Natural Language Processing (NLP)</h3>
    <p>
      แม้ Pooling จะถูกออกแบบมาสำหรับภาพ แต่ก็มีการนำมาใช้ใน NLP เช่น 1D Max Pooling หรือ Temporal Pooling
      บนลำดับของ Embeddings โดยเฉพาะในงาน Sentiment Classification หรือ Text Matching เช่นใน CNN for Sentence Classification (Kim, 2014)
    </p>

    <ul className="list-disc list-inside">
      <li>ใช้ 1D Pooling บนลำดับคำเพื่อสกัด keyword ที่เด่นที่สุด</li>
      <li>ช่วยลด Sequence Length ก่อนป้อนเข้า Dense Layer</li>
    </ul>

    <h3 className="text-xl font-semibold mt-10">6. Edge & Mobile Computing</h3>
    <p>
      ในสภาพแวดล้อมที่มีทรัพยากรจำกัด เช่น IoT หรือ Mobile Devices การใช้ Pooling Layer
      ช่วยให้โมเดลสามารถลดขนาด input ได้ตั้งแต่ต้นน้ำ และช่วยให้สามารถ deploy โมเดลขนาดเล็กลงได้
    </p>

    <div className="bg-blue-600 border-l-4 border-blue-400 p-4 rounded-lg">
      <p className="text-sm">
        <strong>Highlight:</strong> MobileNet ใช้ Depthwise Separable Convolution ควบคู่กับ Average Pooling เพื่อให้การประมวลผลเร็วและแม่นยำมากขึ้นในอุปกรณ์ขนาดเล็ก
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-10">7. Sound & Audio Recognition</h3>
    <p>
      CNN สำหรับ Audio Recognition ใช้ Pooling Layer เพื่อจัดการกับ Spectrogram หรือ MFCC Features
      โดยลดขนาด temporal/spatial resolution ของเสียงเพื่อให้โมเดลโฟกัสที่ pattern สำคัญ เช่นเสียงพูดหรือเสียงสิ่งแวดล้อม
    </p>

    <table className="table-auto w-full text-left border-collapse mt-4">
      <thead>
        <tr>
          <th className="border px-4 py-2">Domain</th>
          <th className="border px-4 py-2">Type of Pooling</th>
          <th className="border px-4 py-2">Purpose</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Image Classification</td>
          <td className="border px-4 py-2">Max Pooling</td>
          <td className="border px-4 py-2">Retain strong features</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Semantic Segmentation</td>
          <td className="border px-4 py-2">Global Average Pooling</td>
          <td className="border px-4 py-2">Contextual summary</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">NLP</td>
          <td className="border px-4 py-2">1D Pooling</td>
          <td className="border px-4 py-2">Keyword emphasis</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Medical Imaging</td>
          <td className="border px-4 py-2">Max Pooling</td>
          <td className="border px-4 py-2">Highlight anomalies</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.</li>
      <li>Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556</li>
      <li>Girshick, R. (2015). Fast R-CNN. ICCV.</li>
      <li>Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.</li>
      <li>Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861</li>
      <li>Stanford CS231n, MIT 6.S191, Oxford Visual Geometry Group</li>
    </ul>
  </div>
</section>


   <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className=" prose-lg max-w-none text-base leading-relaxed space-y-10">

    <p>
      หนึ่งในความเข้าใจเชิงลึกที่สำคัญเกี่ยวกับ Convolutional Neural Networks (CNNs) คือการออกแบบที่ต้องคำนึงถึงทั้งโครงสร้างเชิงพื้นที่และการจัดการขนาดของข้อมูล (dimensionality reduction) อย่างเหมาะสม การใช้เทคนิค Pooling เป็นเพียงหนึ่งในเครื่องมือ แต่ Insight ที่แท้จริงอยู่ที่การประสานหลายองค์ประกอบร่วมกันเพื่อให้ได้สถาปัตยกรรมที่มีประสิทธิภาพสูง
    </p>

    <h3>มุมมองเชิงโครงสร้างจากงานวิจัยระดับโลก</h3>
    <p>
      จากงานวิจัยของ He et al. (2015) ที่เสนอ Residual Networks (ResNet) ได้มีการเน้นว่า "การเรียนรู้เชิงลึกที่มีชั้นมากอาจไม่ดีขึ้นเสมอไปหากไม่มีกลไกในการส่งต่อข้อมูลที่มีประสิทธิภาพ" นี่เป็น Insight สำคัญที่ผลักดันให้งานวิจัยต่อมามองการออกแบบ CNN เป็นเรื่องของ balance ระหว่างความลึกและความสามารถในการรักษาข้อมูลสำคัญระหว่าง layer
    </p>

    <h3>กล่อง Insight สำคัญ: Feature Hierarchy vs Abstraction Level</h3>
    <div className="bg-yellow-600 border border-yellow-300 rounded-lg p-6 shadow-sm">
      <h4 className="text-lg font-semibold mb-2">Feature Hierarchy</h4>
      <p className="mb-2">
        CNNs สร้างลำดับชั้นของ feature โดยอัตโนมัติ — ชั้นต้นๆ มักเรียนรู้ edge และ pattern พื้นฐาน ขณะที่ชั้นลึกจะเรียนรู้ semantic-level feature เช่น วัตถุหรือหน้าคน
      </p>
      <p>
        การรักษา hierarchy ที่เหมาะสมด้วยเทคนิคอย่าง Strided Convolution, Dilated Convolution, และ Skip Connection ช่วยเพิ่ม abstraction โดยไม่ลด resolution ของข้อมูลเกินไป
      </p>
    </div>

    <h3>การใช้ Pooling ร่วมกับเทคนิคอื่น</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>
        <strong>Global Average Pooling (GAP):</strong> นิยมในโมเดลเช่น MobileNet และ EfficientNet ซึ่งลดขนาด feature map ให้เป็น vector โดยไม่ต้องใช้ Fully Connected Layer
      </li>
      <li>
        <strong>Attention-guided Pooling:</strong> ใช้ Attention Map เป็นน้ำหนักการรวม spatial feature ตามความสำคัญ
      </li>
      <li>
        <strong>Adaptive Pooling:</strong> ปรับขนาด output ตาม target resolution โดยใช้ interpolation-based pooling
      </li>
    </ul>

    <h3>เปรียบเทียบภาพรวม</h3>
    <table className="table-auto w-full text-left border-collapse mt-6">
      <thead>
        <tr>
          <th className="border px-4 py-2 bg-gray-600">เทคนิค</th>
          <th className="border px-4 py-2 bg-gray-600">ข้อดี</th>
          <th className="border px-4 py-2 bg-gray-600">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Max Pooling</td>
          <td className="border px-4 py-2">ลดขนาดและ noise ได้ดี</td>
          <td className="border px-4 py-2">อาจสูญเสีย pattern สำคัญ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Average Pooling</td>
          <td className="border px-4 py-2">เก็บภาพรวม feature ได้ดีกว่า</td>
          <td className="border px-4 py-2">ลดความคมชัดของ pattern</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Attention-based Pooling</td>
          <td className="border px-4 py-2">เรียนรู้บริบทแบบ dynamic</td>
          <td className="border px-4 py-2">ต้องการพารามิเตอร์เพิ่ม</td>
        </tr>
      </tbody>
    </table>

    <h3>ประยุกต์ใช้ Insight ในโมเดลใหม่</h3>
    <p>
      การออกแบบ CNN สมัยใหม่ เช่น ConvNeXt (Liu et al., 2022) ได้ผสมผสานแนวคิดจาก Transformer และใช้ global pooling แทน local pooling เพื่อจับ pattern ระยะไกล โดย insight จากการทดลองแสดงว่า pooling ที่ดีต้องปรับตาม task และ distribution ของข้อมูล
    </p>

    <div className="bg-blue-600 border border-blue-300 rounded-lg p-6 shadow-sm">
      <h4 className="text-lg font-semibold mb-2">แนวโน้มการพัฒนา</h4>
      <p>
        งานวิจัยใหม่เริ่มมอง CNN เป็น modular unit ที่สามารถถูกฝึกให้เข้าใจ context แบบกว้างโดยไม่จำกัดแค่ receptive field แบบเดิม อีกทั้งการใช้ pooling แบบ dynamic ที่เรียนรู้ได้อัตโนมัติกำลังเป็นแนวทางที่เติบโต
      </p>
    </div>

    <h3>การอ้างอิง</h3>
    <ul className="list-disc pl-6 space-y-1">
      <li>He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In *CVPR*.</li>
      <li>Liu, Z., Mao, H., Wu, C., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In *CVPR*.</li>
      <li>Wang, H., & Yu, H. (2021). Attention-based Pooling in CNNs: A Survey. In *IEEE Transactions on Pattern Analysis and Machine Intelligence*.</li>
      <li>Lin, M., Chen, Q., & Yan, S. (2013). Network In Network. arXiv:1312.4400</li>
    </ul>

  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day43 theme={theme} />
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
        <ScrollSpy_Ai_Day43 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day43_PoolingLayers;
