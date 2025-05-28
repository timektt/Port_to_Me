import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day42 from "../scrollspy/scrollspyDay41-60/ScrollSpy_Ai_Day42";
import MiniQuiz_Day42 from "../miniquiz/miniquizDay41-60/MiniQuiz_Day42";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day42_CNNFiltersFeatureMaps = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day42_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day42_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day42_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day42_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day42_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day42_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day42_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day42_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day42_9").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 42: CNN Filters & Feature Maps</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          <section id="filters-overview" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ภาพรวมของ Filter ใน CNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert space-y-6">
    <p>
      ในระบบ Convolutional Neural Networks (CNN) ฟิลเตอร์ (Filters) หรือที่เรียกว่า Kernel คือองค์ประกอบสำคัญในการเรียนรู้ Feature ของข้อมูลภาพ
      การทำงานของฟิลเตอร์ใน CNN คือการเลื่อนหน้าต่างกรอบเล็ก ๆ ไปตามพื้นที่ภาพอินพุต เพื่อดึงข้อมูลเชิงพื้นที่ (Spatial Features) ที่มีความสำคัญ เช่น ขอบภาพ ลวดลาย หรือรูปทรงเรขาคณิต
    </p>

    <h3 className="text-xl font-semibold">แนวคิดหลักของ Filter</h3>
    <ul className="list-disc list-inside">
      <li>ฟิลเตอร์คือเมทริกซ์ขนาดเล็ก เช่น 3x3 หรือ 5x5 ที่เรียนรู้ได้</li>
      <li>เลื่อนฟิลเตอร์ผ่านภาพอินพุตเพื่อคำนวณค่า Dot Product</li>
      <li>ผลลัพธ์จากการคูณนี้คือ Feature Map ซึ่งแสดงให้เห็นว่า Feature ใด ๆ ปรากฏในตำแหน่งใดของภาพ</li>
    </ul>

    <div className="bg-yellow-600 p-4 rounded-md border-l-4 border-yellow-500">
      <strong>Insight:</strong> หน่วยพื้นฐานของการเรียนรู้ใน CNN ไม่ใช่พิกเซลเดี่ยว แต่คือกลุ่มพิกเซลที่สัมพันธ์กันซึ่งฟิลเตอร์สามารถตรวจจับลวดลายหรือโครงสร้างร่วมได้
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบ Filter กับ Edge Detection</h3>
    <p>
      ในภาพคลาสสิกของการประมวลผลภาพ ฟิลเตอร์เช่น Sobel หรือ Laplacian ถูกออกแบบด้วยมือเพื่อจับขอบภาพหรือการเปลี่ยนแปลงความเข้มของพิกเซลอย่างเฉียบพลัน CNN นำแนวคิดนี้มาขยายให้สามารถเรียนรู้ฟิลเตอร์ได้เองจากข้อมูลจำนวนมาก
    </p>

    <table className="table-auto w-full border mt-6">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border px-4 py-2">Traditional Filter</th>
          <th className="border px-4 py-2">CNN Filter</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">กำหนดโดยมนุษย์</td>
          <td className="border px-4 py-2">เรียนรู้โดยอัตโนมัติจากข้อมูล</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">เหมาะกับงานเฉพาะเจาะจง</td>
          <td className="border px-4 py-2">ยืดหยุ่นและปรับใช้ได้กับภาพหลายประเภท</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ต้องใช้ความรู้เฉพาะทาง</td>
          <td className="border px-4 py-2">พึ่งพาข้อมูลในการปรับพารามิเตอร์</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">ฟิลเตอร์หลายชุดและความลึกของข้อมูล</h3>
    <p>
      CNN สมัยใหม่มักใช้ฟิลเตอร์หลายชุดพร้อมกันในแต่ละเลเยอร์เพื่อเรียนรู้ Feature ที่หลากหลาย เช่น ขอบภาพ, พื้นผิว, ลวดลายซ้ำ หรือแม้แต่ลักษณะเฉพาะใบหน้า
      ยิ่งใช้ฟิลเตอร์จำนวนมากขึ้นเท่าใด ความสามารถในการจำแนก Feature ที่ซับซ้อนก็ยิ่งดีขึ้น
    </p>

    <div className="bg-blue-600 p-4 rounded-md border-l-4 border-blue-500">
      <strong>Highlight:</strong> ฟิลเตอร์หลายตัวทำงานร่วมกันแบบขนาน ช่วยให้โมเดลมีมิติข้อมูลลึกขึ้น ซึ่งเป็นกุญแจสำคัญของ Deep Learning
    </div>

    <h3 className="text-xl font-semibold">การทำงานแบบเรียนรู้</h3>
    <p>
      แต่ละฟิลเตอร์มีพารามิเตอร์ที่ถูกอัปเดตผ่านกระบวนการ Backpropagation ด้วย Optimizer เช่น SGD หรือ Adam ฟิลเตอร์เหล่านี้จะปรับตัวเพื่อเรียนรู้ Feature ที่แยกแยะได้ดีที่สุดสำหรับงานจำแนกภาพ
    </p>

    <ul className="list-disc list-inside">
      <li>ฟิลเตอร์ชุดแรกมักจับ Feature พื้นฐาน เช่น ขอบหรือรูปทรง</li>
      <li>ฟิลเตอร์ในเลเยอร์ลึกจะเริ่มจับ Feature ที่ซับซ้อน เช่น โครงสร้างของใบหน้า หรือวัตถุเฉพาะ</li>
    </ul>

    <h3 className="text-xl font-semibold">อ้างอิงจากงานวิจัยและแหล่งข้อมูล</h3>
    <ul className="list-disc list-inside">
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning Lecture Series, 6.S191</li>
      <li>LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature.</li>
      <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
    </ul>
  </div>
</section>

      <section id="layerwise-filters" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ฟิลเตอร์ในเลเยอร์ต่าง ๆ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose lg:prose-lg max-w-none dark:prose-invert">
    <h3>โครงสร้างของ CNN: จากเลเยอร์แรกถึงเลเยอร์ลึก</h3>
    <p>
      ในสถาปัตยกรรมของ Convolutional Neural Networks (CNNs) ฟิลเตอร์ในแต่ละเลเยอร์มีหน้าที่แตกต่างกันอย่างชัดเจน โดยการออกแบบฟิลเตอร์ในลำดับเลเยอร์ที่ต่างกันมีความเกี่ยวข้องกับระดับความซับซ้อนของ feature ที่ต้องการเรียนรู้จากข้อมูล
    </p>

    <h3>ลักษณะของฟิลเตอร์ในเลเยอร์เริ่มต้น</h3>
    <p>
      ฟิลเตอร์ในเลเยอร์แรกมักจะเรียนรู้ feature ที่มีโครงสร้างพื้นฐาน เช่น edge, corner, และ blob ซึ่งเป็น pattern เบื้องต้นที่พบได้ทั่วไปในภาพ เช่น เส้นตรงแนวนอนและแนวตั้ง ซึ่งคล้ายคลึงกับการตรวจจับ edge ใน edge detection algorithm แบบ Sobel หรือ Prewitt
    </p>

    <div className="bg-blue-600 p-4 rounded-xl border border-blue-300 my-6">
      <p className="font-medium">Insight Box</p>
      <p>
        งานวิจัยจาก University of Oxford (Zeiler & Fergus, 2014) พบว่าเลเยอร์ต้นของ CNN มีความคล้ายคลึงกับการตอบสนองของเซลล์ประสาทในชั้น V1 ของระบบสายตาในมนุษย์ ซึ่งบ่งชี้ว่าระบบเรียนรู้โครงสร้างภาพขั้นต้นแบบเป็นธรรมชาติ
      </p>
    </div>

    <h3>ฟิลเตอร์ในเลเยอร์กลาง</h3>
    <p>
      ฟิลเตอร์ในเลเยอร์กลางเริ่มเรียนรู้โครงสร้างที่ซับซ้อนมากขึ้น เช่น รูปทรงเรขาคณิต, ลวดลายซ้ำ, และส่วนประกอบของวัตถุ โดยอิงจากการรวมกันของ feature ในเลเยอร์ต้น เป็นการสร้าง representation ที่มี semantics มากขึ้น
    </p>

    <h3>ฟิลเตอร์ในเลเยอร์ลึก</h3>
    <p>
      เลเยอร์ลึกใน CNN จะจับโครงสร้างระดับ high-level ที่แสดงถึงวัตถุเฉพาะ เช่น หน้า, ล้อรถ, หรือหูของสัตว์ โดยเฉพาะในโมเดลที่มีโครงสร้างลึก เช่น ResNet-152 หรือ EfficientNet ซึ่งสามารถเรียนรู้ representation ที่สลับซับซ้อน
    </p>

    <div className="bg-yellow-600 p-4 rounded-xl border border-yellow-300 my-6">
      <p className="font-medium">Highlight Box</p>
      <p>
        ลักษณะการเปลี่ยนแปลงของฟิลเตอร์ในแต่ละเลเยอร์นี้ถูกใช้อธิบายในแนวคิดของ "Feature Hierarchy" ซึ่งเป็นแกนหลักของการเรียนรู้เชิงลำดับขั้น (hierarchical learning) ใน deep learning สมัยใหม่
      </p>
    </div>

    <h3>ตารางเปรียบเทียบลักษณะฟิลเตอร์ในแต่ละเลเยอร์</h3>
    <div className="overflow-x-auto my-6">
      <table className="table-auto w-full border border-gray-300">
        <thead>
          <tr className="bg-gray-600 dark:bg-gray-800">
            <th className="px-4 py-2 border">ระดับเลเยอร์</th>
            <th className="px-4 py-2 border">ลักษณะฟิลเตอร์</th>
            <th className="px-4 py-2 border">ตัวอย่าง Feature</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-4 py-2 border">เลเยอร์ต้น</td>
            <td className="px-4 py-2 border">Edge, Line, Corner</td>
            <td className="px-4 py-2 border">ขอบสี, เส้นตรง</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border">เลเยอร์กลาง</td>
            <td className="px-4 py-2 border">Texture, Pattern</td>
            <td className="px-4 py-2 border">ลวดลาย, รูปทรงเรขาคณิต</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border">เลเยอร์ลึก</td>
            <td className="px-4 py-2 border">Object Part Representation</td>
            <td className="px-4 py-2 border">ใบหน้า, วัตถุเฉพาะ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>การเรียนรู้ฟิลเตอร์ในระดับเลเยอร์</h3>
    <p>
      ฟิลเตอร์ในแต่ละเลเยอร์จะได้รับการเรียนรู้ผ่านการ backpropagation และ optimization โดยใช้ gradient descent เพื่อปรับ weight ของ kernel ให้เหมาะสมกับการดึง feature จาก input ตัวอย่างที่หลากหลาย ฟิลเตอร์ในเลเยอร์ลึกจึงมักต้องการข้อมูลและ parameter มากขึ้น
    </p>

    <h3>สรุป</h3>
    <p>
      ความเข้าใจถึงโครงสร้างของฟิลเตอร์ในแต่ละเลเยอร์เป็นปัจจัยสำคัญที่ช่วยให้สามารถออกแบบสถาปัตยกรรมของ CNN ได้อย่างมีประสิทธิภาพ และสามารถเลือกใช้เลเยอร์ให้เหมาะสมกับลักษณะของข้อมูลในงานต่าง ๆ
    </p>

    <h3 className="mt-10">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV.</li>
      <li>Yosinski, J. et al. (2015). Understanding Neural Networks Through Deep Visualization. arXiv:1506.06579</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191: Introduction to Deep Learning</li>
    </ul>
  </div>
</section>


    <section id="feature-map" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Feature Map คืออะไร?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-neutral dark:prose-invert max-w-none">
    <h3>นิยามของ Feature Map</h3>
    <p>
      Feature Map คือผลลัพธ์ของกระบวนการ Convolution ระหว่างฟิลเตอร์ (kernel) และ input data เช่นภาพ โดยจะมีการเลื่อนฟิลเตอร์ไปตามพื้นที่ของอินพุตเพื่อสร้างการตอบสนองที่เป็นลักษณะเด่นของข้อมูลในแต่ละตำแหน่ง ซึ่งช่วยให้โมเดลสามารถแยกลักษณะสำคัญ เช่น เส้นขอบ รูปร่าง หรือพื้นผิว ได้อย่างเป็นระบบ
    </p>

    <h3>หลักการสร้าง Feature Map</h3>
    <ul className="list-disc pl-6">
      <li>เริ่มต้นด้วยการวางฟิลเตอร์ลงบนตำแหน่งเริ่มต้นของภาพ</li>
      <li>คำนวณการ dot product ระหว่างฟิลเตอร์และพื้นที่ภาพที่ทับซ้อนกัน</li>
      <li>เลื่อนฟิลเตอร์ไปในแนวแกน X และ Y ตามค่า stride</li>
      <li>บันทึกค่าที่ได้จากแต่ละตำแหน่งในเมทริกซ์ใหม่ ซึ่งเรียกว่า Feature Map</li>
    </ul>

    <div className="bg-yellow-600 rounded-xl p-4 my-6 border border-yellow-300">
      <h4 className="text-lg font-semibold mb-2">Insight Box</h4>
      <p className="text-sm leading-relaxed">
        Feature Map ไม่ใช่ภาพต้นฉบับที่ถูกลดขนาดลง แต่เป็นการแทนค่าของลักษณะเฉพาะ (features) ที่ฟิลเตอร์จับได้ ซึ่งช่วยลดมิติข้อมูลพร้อมคงสาระสำคัญของข้อมูลไว้
      </p>
    </div>

    <h3>ตัวอย่าง Feature Map ในแต่ละเลเยอร์</h3>
    <p>
      แต่ละเลเยอร์ใน CNN จะมีฟิลเตอร์หลายตัวซึ่งสร้าง Feature Maps หลายแผนที่ในหนึ่งอินพุต โดยในเลเยอร์แรก มักจะจับลักษณะง่าย เช่น เส้นขอบ ในขณะที่เลเยอร์ลึกจะจับรูปร่างหรือวัตถุที่ซับซ้อนขึ้น
    </p>

    <table className="table-auto border border-gray-600 w-full text-sm text-left mt-6">
      <thead className="bg-gray-600">
        <tr>
          <th className="border px-4 py-2">เลเยอร์</th>
          <th className="border px-4 py-2">ขนาด Input</th>
          <th className="border px-4 py-2">จำนวน Filters</th>
          <th className="border px-4 py-2">ขนาด Feature Map</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Conv1</td>
          <td className="border px-4 py-2">224×224×3</td>
          <td className="border px-4 py-2">64</td>
          <td className="border px-4 py-2">112×112×64</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Conv2</td>
          <td className="border px-4 py-2">112×112×64</td>
          <td className="border px-4 py-2">128</td>
          <td className="border px-4 py-2">56×56×128</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-600 rounded-xl p-4 my-6 border border-blue-300">
      <h4 className="text-lg font-semibold mb-2">Highlight Box</h4>
      <p className="text-sm leading-relaxed">
        ความละเอียดของ Feature Map จะลดลงตามจำนวน stride และการใช้ pooling แต่จำนวน channel จะเพิ่มขึ้นตามจำนวนฟิลเตอร์ ส่งผลให้โมเดลสามารถเรียนรู้คุณสมบัติที่ลึกขึ้น
      </p>
    </div>

    <h3>ผลของ Feature Map ต่อการเรียนรู้</h3>
    <p>
      Feature Maps เป็นตัวแทนของ “การเรียนรู้” ในแต่ละชั้นของโมเดล CNN หากไม่มี Feature Map โมเดลจะไม่สามารถเก็บข้อมูลลักษณะเด่นในลำดับชั้นได้ ซึ่งสำคัญต่อความแม่นยำในการจดจำภาพหรือวัตถุ
    </p>

    <h3 className="mt-8">เอกสารอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n - Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT 6.S191 - Deep Learning Lectures</li>
      <li>Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. ECCV.</li>
      <li>Springenberg, J. T., et al. (2015). Striving for simplicity: The all convolutional net. arXiv.</li>
    </ul>
  </div>
</section>


    <section id="filter-training" className="mb-20 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl md:text-3xl font-semibold mb-8 text-center text-gray-900 dark:text-gray-100">
    4. การเรียนรู้ Filters ระหว่าง Training
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-sm sm:prose-base lg:prose-lg dark:prose-invert max-w-none">
    <h3>บทนำ</h3>
    <p>
      การเรียนรู้ของฟิลเตอร์ใน Convolutional Neural Networks (CNNs) เป็นหัวใจสำคัญที่ส่งผลต่อความสามารถของโมเดลในการแยกแยะลักษณะของข้อมูลภาพ...
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-800/30 border-l-4 border-yellow-500 dark:border-yellow-400 p-4 rounded-md my-6">
      <p className="font-semibold text-yellow-800 dark:text-yellow-200">Insight:</p>
      <p className="text-yellow-900 dark:text-yellow-100">
        Filters ใน CNN ไม่ได้ถูกกำหนดล่วงหน้า แต่เรียนรู้ขึ้นจากข้อมูลโดยตรงผ่าน optimization algorithm เช่น SGD หรือ Adam
      </p>
    </div>

    <h3>กลไกการเรียนรู้</h3>
    <p>
      ระหว่างการฝึก โมเดลจะใช้ loss function เพื่อวัด error แล้วส่งผลลัพธ์ผ่าน backpropagation มายัง layer ต่าง ๆ รวมถึง convolutional filters
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded-md overflow-x-auto p-4 my-4">
      <code className="language-python">
{`# ตัวอย่าง pseudo-code ของการอัปเดต filter
for filter in conv_filters:
    gradient = compute_gradient(loss, filter)
    filter -= learning_rate * gradient`}
      </code>
    </pre>

    <h3>การเรียนรู้ลักษณะเฉพาะในแต่ละเลเยอร์</h3>
    <ul>
      <li><strong>เลเยอร์ต้น:</strong> มักเรียนรู้ edge, corners, และ texture เบื้องต้น</li>
      <li><strong>เลเยอร์กลาง:</strong> เรียนรู้ feature ที่ซับซ้อนมากขึ้น เช่น shape</li>
      <li><strong>เลเยอร์ลึก:</strong> เรียนรู้ object-level representation เช่น หน้าคน ดวงตา</li>
    </ul>

    <h3>ภาพจำลองการเปลี่ยนแปลงของฟิลเตอร์</h3>
    <p>
      การวิจัยจาก MIT และ Stanford แสดงให้เห็นว่า filters มีลักษณะเปลี่ยนแปลงจากแบบ random noise → structure-based pattern ภายในไม่กี่ epoch แรก
    </p>

    <div className="bg-blue-100 dark:bg-blue-900/30 border-l-4 border-blue-500 dark:border-blue-400 p-4 rounded-md my-6">
      <p className="font-semibold text-blue-800 dark:text-blue-200">Highlight:</p>
      <p className="text-blue-900 dark:text-blue-100">
        การใช้ learning rate ที่เหมาะสม และ optimizer ที่มี momentum เช่น Adam หรือ RMSProp ช่วยให้ filter convergence ได้ดีขึ้น
      </p>
    </div>

    <h3>การ Regularize การเรียนรู้ Filters</h3>
    <p>เพื่อป้องกันการเรียนรู้ที่ overfitting นักวิจัยนิยมใช้เทคนิคต่าง ๆ เช่น:</p>
    <ul>
      <li><strong>L2 Regularization:</strong> ลดขนาด weight ของ filter</li>
      <li><strong>Dropout:</strong> ปิดการทำงานบางส่วนของ layer เพื่อเพิ่มความ generalization</li>
      <li><strong>Batch Normalization:</strong> ควบคุม distribution ของ activation ภายในแต่ละ mini-batch</li>
    </ul>

    <h3>การใช้ Pretrained Filters</h3>
    <p>
      บางกรณีโมเดลจะเริ่มต้นจาก pretrained filters จากโมเดลขนาดใหญ่ เช่น VGG หรือ ResNet เพื่อช่วยให้ convergence เร็วขึ้น และเพิ่ม accuracy
    </p>

    <pre className="bg-gray-800 text-white text-sm rounded-md overflow-x-auto p-4 my-4">
      <code className="language-python">
{`# โหลด pretrained model
model = torchvision.models.resnet18(pretrained=True)`}
      </code>
    </pre>

    <h3>การทดสอบและวัดผลฟิลเตอร์ที่เรียนรู้</h3>
    <ul>
      <li>Visualize filters และ activations</li>
      <li>วิเคราะห์ feature importance ด้วย saliency map</li>
      <li>ใช้ transfer learning เพื่อตรวจสอบประสิทธิภาพ</li>
    </ul>

    <h3>การเปรียบเทียบการเรียนรู้ Filter ใน Optimizers ต่าง ๆ</h3>
    <div className="overflow-x-auto my-4">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-800 text-left">
            <th className="px-4 py-2 border border-gray-300 dark:border-gray-700">Optimizer</th>
            <th className="px-4 py-2 border border-gray-300 dark:border-gray-700">ความเร็วในการ convergence</th>
            <th className="px-4 py-2 border border-gray-300 dark:border-gray-700">ความเสถียรของ filter</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">SGD</td>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">ช้า</td>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">ไม่เสถียรเมื่อ learning rate สูง</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">Adam</td>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">เร็วมาก</td>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">เสถียรสูง</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">RMSProp</td>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">ปานกลาง</td>
            <td className="px-4 py-2 border border-gray-300 dark:border-gray-700">เสถียรในข้อมูล noisy</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition</li>
      <li>MIT Deep Learning for Self-Driving Cars, Lecture 4</li>
      <li>Zeiler & Fergus (2014), Visualizing and Understanding Convolutional Networks, ECCV</li>
      <li>arXiv:1706.03850 - A disciplined approach to neural network hyper-parameters</li>
    </ul>
  </div>
</section>


  <section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center text-gray-900 dark:text-white">
    5. Visualization ของ Filters และ Feature Maps
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose dark:prose-invert text-base max-w-none space-y-8">
    <h3>การมองเห็นภายในโมเดล</h3>
    <p>Visualization ช่วยให้เข้าใจว่า CNNs มุ่งเน้น feature ใดในแต่ละ layer และสามารถตรวจจับ overfitting หรือ bias ได้</p>

    <div className="bg-yellow-100 dark:bg-yellow-700 p-4 rounded-md border-l-4 border-yellow-400">
      <p className="font-semibold">Insight:</p>
      <p>Visualization ของ filters เพิ่มความเข้าใจในการออกแบบโมเดล โดยไม่ต้องทดลองแบบ brute-force</p>
    </div>

    <h3>เทคนิคยอดนิยม</h3>
    <ul>
      <li><strong>Filter Visualization:</strong> วิเคราะห์ weight ใน conv layer</li>
      <li><strong>Feature Map:</strong> ดูผลลัพธ์การ convolution</li>
      <li><strong>Activation Maximization:</strong> input ที่ maximize neuron</li>
      <li><strong>Saliency Maps:</strong> คำนวณ gradient เพื่อหา pixel สำคัญ</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-700 p-4 rounded-md border-l-4 border-blue-400">
      <p className="font-semibold">Highlight:</p>
      <p>Grad-CAM ช่วยอธิบายการตัดสินใจของโมเดลด้วยการเน้นจุดสำคัญบนภาพ</p>
    </div>

    <h3>เปรียบเทียบ Layer</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border">
        <thead className="bg-gray-200 dark:bg-gray-700">
          <tr>
            <th className="px-4 py-2">Layer</th>
            <th className="px-4 py-2">Feature Type</th>
            <th className="px-4 py-2">Visualization Meaning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-4 py-2 border">ต้น</td>
            <td className="px-4 py-2 border">Edge / Texture</td>
            <td className="px-4 py-2 border">เรียนรู้เส้นตรงหรือโค้งพื้นฐาน</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border">กลาง</td>
            <td className="px-4 py-2 border">โครงสร้าง</td>
            <td className="px-4 py-2 border">Feature ระดับสูงขึ้น</td>
          </tr>
          <tr>
            <td className="px-4 py-2 border">ลึก</td>
            <td className="px-4 py-2 border">Object-level</td>
            <td className="px-4 py-2 border">รวม feature เพื่อตัดสินใจ</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ข้อควรระวัง</h3>
    <ul>
      <li>ภาพที่สร้างอาจไม่สะท้อนความเข้าใจจริงของโมเดล</li>
      <li>บาง filter อาจไม่มีการใช้จริง</li>
      <li>Layer ลึกยากต่อการตีความ</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-700 p-4 rounded-md border-l-4 border-yellow-400">
      <p className="font-semibold">Insight:</p>
      <p>Visualization คือเครื่องมือทั้งวินิจฉัยและออกแบบโมเดลให้ align กับความเข้าใจของมนุษย์</p>
    </div>

    <h3>การประยุกต์ใช้งาน</h3>
    <ul>
      <li>วิเคราะห์ AI ทางการแพทย์</li>
      <li>ออกแบบ architecture ใหม่</li>
      <li>อธิบายผลลัพธ์กับผู้ใช้งาน</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul>
      <li>Yosinski et al. (2015), arXiv:1506.06579</li>
      <li>Selvaraju et al., Grad-CAM, ICCV 2017</li>
      <li>Stanford CS231n Lecture</li>
      <li>MIT 6.S191 Deep Learning 2024</li>
    </ul>
  </div>
</section>


<section id="channel-wise" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. การทำงานแบบ Channel-wise</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose dark:prose-invert text-base leading-relaxed space-y-10 max-w-none">
    <p>
      การทำงานแบบ <strong>Channel-wise</strong> ใน Convolutional Neural Networks (CNNs) เป็นหนึ่งในองค์ประกอบหลักที่ทำให้โมเดลสามารถเรียนรู้คุณลักษณะจากหลายมุมมองของข้อมูลได้ โดยเฉพาะในกรณีของข้อมูลภาพที่ประกอบด้วยหลายช่องสัญญาณ (channels) เช่น RGB
    </p>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-lg">
      <strong>Insight:</strong> แนวคิด channel-wise convolution ช่วยให้โมเดลสามารถเรียนรู้ความสัมพันธ์เฉพาะของแต่ละช่อง เช่น ขอบในช่อง R, G, หรือ B โดยไม่เกิดการผสมสัญญาณที่ไม่จำเป็นในช่วงแรกของการเรียนรู้
    </div>

    <h3 className="text-xl font-semibold mt-8">6.1 พื้นฐานการแยก Channel</h3>
    <p>
      ในภาพสีทั่วไปจะประกอบด้วย 3 ช่องคือ Red, Green และ Blue เมื่อข้อมูลนี้ถูกป้อนเข้า CNN ในเลเยอร์แรก ฟิลเตอร์แต่ละตัวจะมีขนาด (k × k × c) โดยที่ c คือจำนวน channel เช่น (3 × 3 × 3)
    </p>

    <pre><code className="language-python">
# โครงสร้าง filter channel-wise
# ตัวอย่าง filter ขนาด 3x3x3
filter = [
  [[w11, w12, w13], [w14, w15, w16], [w17, w18, w19]],  # Channel R
  [[w21, w22, w23], [w24, w25, w26], [w27, w28, w29]],  # Channel G
  [[w31, w32, w33], [w34, w35, w36], [w37, w38, w39]],  # Channel B
]
    </code></pre>

    <p>
      จากนั้นจะทำการคูณ (dot product) ทีละ channel แยกกัน แล้วรวมผลลัพธ์ก่อนส่งผ่าน activation function
    </p>

    <h3 className="text-xl font-semibold mt-8">6.2 การเรียนรู้ Channel-wise Features</h3>
    <ul className="list-disc pl-6">
      <li>ใน layer แรก โมเดลสามารถตรวจจับ edges, gradients หรือ patterns เฉพาะในแต่ละ channel</li>
      <li>เมื่อรวม features เข้าด้วยกัน จะได้ representation ที่ซับซ้อนขึ้นใน spatial dimension</li>
    </ul>

    <div className="bg-yellow-600 border border-yellow-300 p-4 rounded-lg">
      <strong>Highlight:</strong> ในเลเยอร์ที่ลึกขึ้น การรวม channel-wise features เข้ากับ spatial encoding ทำให้ CNN สามารถแยกแยะวัตถุหรือ texture ได้อย่างแม่นยำ
    </div>

    <h3 className="text-xl font-semibold mt-8">6.3 เปรียบเทียบ: Channel-wise vs Depthwise Separable</h3>
    <div className="overflow-x-auto my-6">
  <table className="table-auto min-w-[600px] w-full border border-gray-300 dark:border-gray-600 text-sm">
    <thead>
      <tr className="bg-gray-600 dark:bg-gray-800 text-white">
        <th className="border px-4 py-2">ประเภท</th>
        <th className="border px-4 py-2">ลักษณะ</th>
        <th className="border px-4 py-2">ข้อดี</th>
        <th className="border px-4 py-2">ตัวอย่างใช้งาน</th>
      </tr>
    </thead>
    <tbody>
      <tr className="bg-white dark:bg-gray-900">
        <td className="border px-4 py-2">Channel-wise</td>
        <td className="border px-4 py-2">เรียนรู้ทุก channel พร้อมกัน</td>
        <td className="border px-4 py-2">แม่นยำสูง</td>
        <td className="border px-4 py-2">VGGNet, ResNet</td>
      </tr>
      <tr className="bg-gray-50 dark:bg-gray-800">
        <td className="border px-4 py-2">Depthwise Separable</td>
        <td className="border px-4 py-2">แยก convolution เป็น depthwise + pointwise</td>
        <td className="border px-4 py-2">ลดพารามิเตอร์</td>
        <td className="border px-4 py-2">MobileNet</td>
      </tr>
    </tbody>
  </table>
</div>


    <h3 className="text-xl font-semibold mt-8">6.4 ผลกระทบต่อ Parameter และ Computation</h3>
    <p>
      ฟิลเตอร์แบบ channel-wise ทำให้โมเดลต้องเรียนรู้ weights จำนวนมาก โดยหากมี input 3 channel และฟิลเตอร์ขนาด 3 × 3 จะมี weight ทั้งหมด 3 × 3 × 3 = 27 weights ต่อฟิลเตอร์ ซึ่งส่งผลต่อ:
    </p>
    <ul className="list-disc pl-6">
      <li>จำนวน parameter ทั้งหมดในโมเดล</li>
      <li>การใช้หน่วยความจำ (memory usage)</li>
      <li>เวลาในการฝึกสอน (training time)</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">6.5 บทบาทของ Channel-wise ในการเรียนรู้เชิงลึก</h3>
    <p>
      การออกแบบโมเดล CNN ที่มีประสิทธิภาพมักเริ่มต้นด้วยการใช้ channel-wise convolution เพื่อแยกแยะลักษณะพื้นฐานของข้อมูล และเพิ่มความสามารถในการแยกแยะความซับซ้อนทางเชิงพื้นที่ใน layer ถัดไป
    </p>

    <div className="bg-blue-600 border border-blue-300 p-4 rounded-lg">
      <strong>Insight:</strong> งานวิจัยจาก MIT และ Stanford ระบุว่า channel-wise convolution เป็นส่วนสำคัญของการ encode features ที่แตกต่างกัน เช่น ความแตกต่างของแสงเงาในช่อง R เทียบกับความชัดของ texture ในช่อง G
    </div>

    <h3 className="text-xl font-semibold mt-8">6.6 การปรับปรุงและเทคนิคเสริม</h3>
    <p>
      นักวิจัยได้พัฒนาเทคนิคต่าง ๆ เพื่อเพิ่มประสิทธิภาพของ channel-wise เช่น:
    </p>
    <ul className="list-disc pl-6">
      <li><strong>Channel Attention Mechanism</strong> - ให้ความสำคัญกับช่องที่ให้ข้อมูลเชิงลึกมากกว่า</li>
      <li><strong>Squeeze-and-Excitation (SE)</strong> - เทคนิคเพิ่มน้ำหนักเฉพาะช่องที่สำคัญ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8">6.7 แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS231n Lecture Notes: Convolutional Neural Networks for Visual Recognition</li>
      <li>Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” arXiv:1704.04861</li>
      <li>Hu et al., “Squeeze-and-Excitation Networks,” CVPR 2018</li>
    </ul>
  </div>
</section>



     <section id="filter-depth" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl sm:text-3xl font-semibold mb-6 text-center text-gray-900 dark:text-gray-100">
    7. Depth of Filters กับพลังของ CNN
  </h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert text-base leading-relaxed space-y-10">

    <p>
      หนึ่งในปัจจัยที่ส่งผลโดยตรงต่อความสามารถในการเรียนรู้ของ Convolutional Neural Networks (CNNs) คือความลึกของฟิลเตอร์ (Filter Depth)
      ซึ่งหมายถึงจำนวนของ channel ที่ใช้ใน kernel แต่ละชุดใน convolutional layer โดยทั่วไป filter depth จะสัมพันธ์กับจำนวน feature map ที่ต้องการสร้าง
      และความซับซ้อนของ pattern ที่โมเดลสามารถรับรู้ได้
    </p>

    <div className="p-4 rounded-lg border border-yellow-400 bg-yellow-100 dark:bg-yellow-800 dark:border-yellow-600">
      <p className="font-semibold text-yellow-900 dark:text-yellow-100">Insight:</p>
      <p className="text-yellow-800 dark:text-yellow-50">
        การเพิ่ม filter depth อย่างมีระบบสามารถขยายขีดความสามารถของโมเดลในการตรวจจับฟีเจอร์ตั้งแต่ระดับ low-level เช่น ขอบภาพ ไปจนถึง high-level เช่น โครงสร้างเฉพาะของวัตถุ.
      </p>
    </div>

    <h3>ความหมายของ Filter Depth</h3>
    <ul className="list-disc pl-6">
      <li>ในเลเยอร์แรก: ฟิลเตอร์จะตรวจจับฟีเจอร์พื้นฐาน เช่น ขอบ แนวตั้ง แนวนอน</li>
      <li>ในเลเยอร์ถัดไป: ฟิลเตอร์จะผสมผสานฟีเจอร์พื้นฐานเหล่านั้นเพื่อตรวจจับ pattern ที่ซับซ้อนขึ้น</li>
      <li>ในเลเยอร์ลึกมากขึ้น: ฟิลเตอร์สามารถจำแนกรูปร่างที่เฉพาะเจาะจง เช่น ใบหน้า หรือวัตถุเฉพาะได้</li>
    </ul>

    <h3>ตัวอย่างการใช้ Filter Depth ใน CNN Architectures</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 dark:border-gray-600 text-left text-sm">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="border px-4 py-2">สถาปัตยกรรม</th>
            <th className="border px-4 py-2">Filter Depth (แต่ละเลเยอร์)</th>
            <th className="border px-4 py-2">ลักษณะเด่น</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">VGG-16</td>
            <td className="border px-4 py-2">64 → 128 → 256 → 512</td>
            <td className="border px-4 py-2">ลึกและเป็นระเบียบ, ใช้ฟิลเตอร์ขนาดเล็ก</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ResNet-50</td>
            <td className="border px-4 py-2">64 → 256 → 512 → 1024</td>
            <td className="border px-4 py-2">ใช้ residual connections เพื่อหลีกเลี่ยง vanishing gradient</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">EfficientNet-B0</td>
            <td className="border px-4 py-2">32 → 128 → 320</td>
            <td className="border px-4 py-2">ใช้ compound scaling เพิ่ม depth, width และ resolution พร้อมกัน</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ความสัมพันธ์ระหว่าง Depth และ Capacity</h3>
    <p>
      ความลึกของฟิลเตอร์ส่งผลโดยตรงต่อ capacity หรือความสามารถในการเรียนรู้ของโมเดล ดังนี้:
    </p>
    <ul className="list-decimal pl-6">
      <li>โมเดลที่มี filter depth ต่ำ: มักเรียนรู้เฉพาะฟีเจอร์พื้นฐาน ทำให้จำแนกวัตถุได้ไม่แม่นยำ</li>
      <li>โมเดลที่มี filter depth สูง: มีศักยภาพในการเข้าใจบริบทเชิงซ้อน แต่ต้องการข้อมูลและพลังประมวลผลสูงกว่า</li>
    </ul>

    <h3>ตัวอย่างโค้ดที่แสดงผลของ Filter Depth</h3>
    <pre><code className="language-python">
# ตัวอย่างจาก PyTorch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
    nn.ReLU()
)
    </code></pre>

    <h3>ข้อดีของการเพิ่ม Filter Depth</h3>
    <ul className="list-disc pl-6">
      <li>เพิ่ม expressiveness ของโมเดล</li>
      <li>ตรวจจับข้อมูลเชิงความหมายได้ลึกขึ้น</li>
      <li>เหมาะกับงานเช่น object detection และ face recognition</li>
    </ul>

    <h3>ข้อควรระวัง</h3>
    <div className="p-4 rounded-lg border border-blue-400 bg-blue-100 dark:bg-blue-900 dark:border-blue-500">
      <p className="font-semibold text-blue-900 dark:text-blue-100">Highlight:</p>
      <p className="text-blue-800 dark:text-blue-50">
        การเพิ่ม filter depth โดยไม่ควบคุมจำนวนพารามิเตอร์อาจทำให้เกิด overfitting และทำให้ inference latency สูงขึ้นอย่างไม่จำเป็น
      </p>
    </div>

    <h3>แนวทางออกแบบ Filter Depth อย่างมีประสิทธิภาพ</h3>
    <ul className="list-disc pl-6">
      <li>เริ่มจาก low → mid → high depth เพื่อค่อย ๆ สร้าง feature hierarchy</li>
      <li>ใช้ residual block หรือ bottleneck เพื่อลดความซับซ้อนเชิงคำนวณ</li>
      <li>ควบคุมจำนวนพารามิเตอร์โดยเลือก kernel size และ stride อย่างเหมาะสม</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Simonyan & Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition”, arXiv:1409.1556</li>
      <li>He et al., “Deep Residual Learning for Image Recognition”, CVPR 2016</li>
      <li>Tan & Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”, ICML 2019</li>
      <li>Stanford CS231n, Convolutional Neural Networks – Lecture Notes</li>
    </ul>

  </div>
</section>


<section id="feature-hierarchy" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. การวิเคราะห์ Feature Hierarchy</h2>

  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert space-y-6">
    <p>
      ในสถาปัตยกรรมของ Convolutional Neural Networks (CNNs) โครงสร้างเชิงลำดับชั้น (Feature Hierarchy)
      เป็นแกนหลักที่ทำให้โมเดลสามารถเรียนรู้และประมวลผลข้อมูลเชิงลึกได้อย่างมีประสิทธิภาพ โดย CNNs
      จะสร้างลำดับของฟีเจอร์จากระดับพื้นฐานไปจนถึงระดับนามธรรม เช่น ขอบ → รูปร่าง → วัตถุ → บริบท
    </p>

    <h3 className="text-xl font-semibold">ลักษณะของ Hierarchical Features</h3>
    <ul className="list-disc list-inside">
      <li>เลเยอร์ต้น: ตรวจจับ edge, blob, gradient</li>
      <li>เลเยอร์กลาง: แยกรูปร่างเรขาคณิต, ลวดลาย</li>
      <li>เลเยอร์ลึก: เข้าใจ object class, context</li>
    </ul>

    <div className="bg-yellow-600 p-4 rounded-md border-l-4 border-yellow-500">
      <strong>Insight:</strong> การเข้าใจลำดับขั้นของฟีเจอร์ช่วยให้นักพัฒนาออกแบบโมเดลได้แม่นยำ และวิเคราะห์สิ่งที่โมเดลเรียนรู้ในแต่ละเลเยอร์
    </div>

    <h3 className="text-xl font-semibold">การใช้งาน Feature Hierarchy</h3>
    <p>
      Feature Hierarchy มีบทบาทสำคัญใน Transfer Learning, การทำ Visualization และการ Fine-tuning โมเดล
    </p>
    <ul className="list-disc list-inside">
      <li><strong>Transfer Learning:</strong> ใช้เลเยอร์ต้นที่เรียนรู้ feature ทั่วไปแล้ว</li>
      <li><strong>Interpretation:</strong> วิเคราะห์เลเยอร์เพื่อเข้าใจการเรียนรู้</li>
      <li><strong>Fine-tuning:</strong> Freeze layer ที่ไม่เกี่ยวข้อง ปรับเฉพาะเลเยอร์ลึก</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่าง Feature Hierarchy</h3>
    <pre className=" dark:bg-gray-800 p-4 rounded-md overflow-auto text-sm">
<code>
Layer 1 → Edge detectors{'\n'}
Layer 2 → Curve and corner patterns{'\n'}
Layer 3 → Object parts (eye, wheel){'\n'}
Layer 4 → Object identity (face, car)
</code>
    </pre>

    <div className="bg-blue-600 p-4 rounded-md border-l-4 border-blue-500">
      <strong>Highlight:</strong> งานวิจัยของ Zeiler & Fergus (2014) ใช้ deconvnet แสดงให้เห็นว่าฟีเจอร์ในแต่ละเลเยอร์เป็นลำดับขั้นจริง
    </div>

    <h3 className="text-xl font-semibold">เปรียบเทียบลักษณะฟีเจอร์ในแต่ละเลเยอร์</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border mt-6">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-700">
            <th className="border px-4 py-2">Layer</th>
            <th className="border px-4 py-2">Feature Scope</th>
            <th className="border px-4 py-2">Use in Transfer Learning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Early</td>
            <td className="border px-4 py-2">Edges, Textures</td>
            <td className="border px-4 py-2">Reusable</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Mid</td>
            <td className="border px-4 py-2">Patterns, Shapes</td>
            <td className="border px-4 py-2">Semi-specific</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Deep</td>
            <td className="border px-4 py-2">Semantics, Objects</td>
            <td className="border px-4 py-2">Fine-tuning</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">แนวทางการวิเคราะห์ Hierarchy</h3>
    <ul className="list-disc list-inside">
      <li>Saliency Maps (e.g., Grad-CAM)</li>
      <li>Deconvolutional Network</li>
      <li>Layer-wise Relevance Propagation</li>
    </ul>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิง</h3>
    <ul className="list-disc list-inside">
      <li>Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding CNNs. ECCV.</li>
      <li>Yosinski, J. et al. (2014). Transferable Features in DNNs. NIPS.</li>
      <li>Simonyan, K. et al. (2013). Deep Inside CNNs. arXiv.</li>
    </ul>
  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day42 theme={theme} />
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
        <ScrollSpy_Ai_Day42 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day42_CNNFiltersFeatureMaps;
