import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day24 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day24";
import MiniQuiz_Day24 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day24";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day24_CNNVision = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("CNNVision1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("CNNVision2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("CNNVision3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("CNNVision4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("CNNVision5").format("auto").quality("auto").resize(scale().width(590));
  const img6 = cld.image("CNNVision6").format("auto").quality("auto").resize(scale().width(590));
  const img7 = cld.image("CNNVision7").format("auto").quality("auto").resize(scale().width(590));
  const img8 = cld.image("CNNVision8").format("auto").quality("auto").resize(scale().width(590));



  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 24: CNN in Computer Vision</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำสู่ Convolutional Neural Networks (CNNs) ในระบบประมวลผลภาพ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Convolutional Neural Networks (CNNs) กลายเป็นรากฐานของระบบคอมพิวเตอร์วิทัศน์ยุคใหม่
      โครงสร้างของ CNN ได้แรงบันดาลใจจากการจัดเรียงเซลล์ในเยื่อหุ้มสมองส่วนที่เกี่ยวข้องกับการมองเห็นของสัตว์เลี้ยงลูกด้วยนม
      โดยถูกออกแบบให้สามารถเรียนรู้ลำดับชั้นของคุณลักษณะเชิงพื้นที่ (spatial hierarchies) ได้อย่างอัตโนมัติผ่านการถ่ายทอดย้อนกลับ (backpropagation)
    </p>

    <p>
      สถาปัตยกรรมแบบเป็นชั้น (layered architecture) ทำให้โมเดลสามารถเข้าใจรูปแบบพื้นฐาน เช่น ขอบ (edge), เนื้อผิว (texture) ในเลเยอร์แรก
      และค่อย ๆ สร้างความเข้าใจเชิงนามธรรม เช่น วัตถุ ใบหน้า หรือฉากภาพ ในเลเยอร์ลึก
    </p>

    <p>
      ความก้าวหน้าครั้งสำคัญเกิดขึ้นในปี 2012 เมื่อโมเดล AlexNet (Krizhevsky et al.) ชนะการแข่งขัน ILSVRC ด้วยอัตราความผิดพลาดเพียง 15.3%
      ซึ่งต่ำกว่าวิธีที่ดีที่สุดก่อนหน้านั้นมาก แสดงให้เห็นถึงพลังของ CNN บนงานจำแนกภาพขนาดใหญ่
    </p>

    <p>
      วิชา Stanford CS231n ระบุว่าจุดแข็งของ CNN อยู่ที่ความสามารถในการใช้โครงสร้างพื้นที่ (spatial locality) และการแชร์พารามิเตอร์ (shared weights)
      ซึ่งช่วยให้โมเดลตรวจจับรูปแบบที่ซับซ้อนโดยไม่ต้องพึ่งการออกแบบคุณลักษณะด้วยมือ (handcrafted features)
    </p>

    <h3 className="text-xl font-semibold">แนวคิดหลักของ CNN ในระบบประมวลผลภาพ</h3>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Convolutional Layers:</strong> ทำหน้าที่สกัดคุณลักษณะผ่านฟิลเตอร์ที่เรียนรู้ได้</li>
      <li><strong>Activation Functions:</strong> สร้างความไม่เชิงเส้น เช่น ReLU เพื่อให้โมเดลมีความสามารถในการประมาณฟังก์ชันซับซ้อน</li>
      <li><strong>Pooling Layers:</strong> ลดขนาดข้อมูลเพื่อเพิ่มความทนทานต่อการเปลี่ยนแปลงเชิงพื้นที่</li>
      <li><strong>Fully Connected Layers:</strong> ใช้ในการตัดสินใจขั้นสุดท้ายจากคุณลักษณะที่สกัดได้</li>
    </ul>

    <p>CNN อาศัยหลักการพื้นฐาน 3 ประการของการเรียนรู้ภาพ ได้แก่:</p>
    <ul className="list-decimal list-inside ml-6 space-y-2">
      <li>การรับข้อมูลเฉพาะจุด (Local receptive fields)</li>
      <li>การใช้ฟิลเตอร์ร่วมกันทั่วทั้งภาพ (Shared weights)</li>
      <li>การลดข้อมูลเชิงพื้นที่เพื่อเพิ่มความไม่ไวต่อการแปล (Spatial subsampling)</li>
    </ul>

    <h3 className="text-xl font-semibold">เหตุผลที่ CNN เหมาะสมกับข้อมูลภาพ</h3>

    <p>
      ภาพมีโครงสร้างเชิงพื้นที่สูง พิกเซลที่อยู่ใกล้กันมักมีความสัมพันธ์สูง
      CNN สามารถจำลองความสัมพันธ์นี้ผ่านฟิลเตอร์เชิงพื้นที่ที่เคลื่อนที่ไปทั่วภาพได้อย่างมีประสิทธิภาพ
    </p>

    <p>
      ในรายวิชา MIT 6.S191 CNN ถือเป็น inductive prior สำหรับข้อมูลภาพ
      โดยการใช้ฟิลเตอร์ร่วมและการเชื่อมโยงเฉพาะท้องถิ่น โมเดลสามารถลดจำนวนพารามิเตอร์ลงได้อย่างมหาศาล
      ทำให้เหมาะกับข้อมูลภาพที่มีความละเอียดสูง
    </p>

    <p>
      CNN ที่ลึกสามารถเรียนรู้ลำดับชั้นของคุณลักษณะ โดยชั้นต้นจับขอบและพื้นผิว ชั้นกลางจับรูปร่างหรือองค์ประกอบ และชั้นลึกเข้าใจความหมายของวัตถุ
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งานจริง</h3>

    <p>
      ในระบบรู้จำใบหน้า เลเยอร์ต้นของ CNN ตรวจจับตา จมูก ปาก
      เลเยอร์กลางเข้าใจโครงสร้างของใบหน้า ส่วนเลเยอร์ลึกสามารถระบุตัวตนแม้มีการเปลี่ยนแสง มุมมอง หรือมีบางส่วนถูกบัง
    </p>

    <p>
      ในรถยนต์ไร้คนขับ CNN ประมวลผลภาพจากกล้องแบบเรียลไทม์เพื่อระบุเลนถนน ป้ายจราจร คนเดินถนน และพาหนะอื่น ๆ ได้อย่างแม่นยำ
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>CNN แทนที่การออกแบบฟีเจอร์แบบ manual ด้วยการเรียนรู้แบบ end-to-end</li>
        <li>โครงสร้างการแชร์พารามิเตอร์และความสัมพันธ์เชิงพื้นที่ ทำให้ CNN เหมาะกับข้อมูลที่มีโครงสร้างตำแหน่ง</li>
        <li>สถาปัตยกรรมใหม่ ๆ เช่น ResNet, DenseNet, EfficientNet ล้วนพัฒนาจาก CNN พื้นฐาน</li>
        <li>งานวิจัยจาก DeepMind และ FAIR ยืนยันว่า residual connections ช่วยให้โมเดลฝึกได้ลึกขึ้นและมีประสิทธิภาพ</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">พัฒนาการสำคัญทางประวัติศาสตร์</h3>

    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">ปี</th>
          <th className="border px-4 py-2">โมเดล</th>
          <th className="border px-4 py-2">ความสำเร็จ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">1998</td>
          <td className="border px-4 py-2">LeNet-5</td>
          <td className="border px-4 py-2">ใช้ในการจำแนกตัวเลข พร้อมระบบ backpropagation</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">2012</td>
          <td className="border px-4 py-2">AlexNet</td>
          <td className="border px-4 py-2">นำ GPU และ ReLU มาใช้ ทำให้ deep learning ฟื้นตัว</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">2014</td>
          <td className="border px-4 py-2">VGGNet</td>
          <td className="border px-4 py-2">สถาปัตยกรรมลึกและสม่ำเสมอ ใช้ kernel ขนาด 3×3</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">2015</td>
          <td className="border px-4 py-2">ResNet</td>
          <td className="border px-4 py-2">ใช้ residual connections ฝึกโมเดลลึกกว่า 150 ชั้นได้สำเร็จ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">2019</td>
          <td className="border px-4 py-2">EfficientNet</td>
          <td className="border px-4 py-2">พัฒนาแนวคิด compound scaling เพิ่มประสิทธิภาพการฝึก</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">บทสรุป</h3>

    <p>
      CNN เปลี่ยนระบบประมวลผลภาพจากแบบใช้กฎมาเป็นแบบขับเคลื่อนด้วยข้อมูล
      ความสามารถในการปรับตัวและความมีประสิทธิภาพ ทำให้ CNN เหมาะกับงานจำแนก แยกวัตถุ การทำ segmentation ไปจนถึงการสร้างภาพใหม่
      ด้วยฮาร์ดแวร์และสถาปัตยกรรมที่พัฒนาอย่างต่อเนื่อง CNN ยังคงเป็นแกนหลักของระบบ AI ในสายตาของโลกยุคใหม่
    </p>

  </div>
</section>


<section id="feature-extraction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. การสกัดคุณลักษณะและลำดับชั้นของข้อมูล</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การสกัดคุณลักษณะ (Feature Extraction) เป็นหัวใจสำคัญในระบบประมวลผลภาพเชิงลึก โดยหมายถึงกระบวนการแปลงข้อมูลภาพดิบให้กลายเป็นคุณลักษณะที่มีโครงสร้างและสามารถใช้งานในการตัดสินใจหรือการเรียนรู้ได้
      ในอดีต การออกแบบคุณลักษณะทำด้วยมือ เช่น SIFT, HOG, หรือ SURF ซึ่งต้องอาศัยความเชี่ยวชาญและสัญชาตญาณของมนุษย์สูง
      แต่ Convolutional Neural Networks (CNNs) ได้เปลี่ยนแนวทางดังกล่าว โดยให้ระบบเรียนรู้คุณลักษณะจากข้อมูลเองอย่างเป็นลำดับชั้น
    </p>

    <p>
      ตามแนวทางของวิชา Stanford CS231n CNN จะเรียนรู้ฟิลเตอร์ที่มีโครงสร้างท้องถิ่น และพัฒนาจากรูปแบบง่ายในเลเยอร์ต้นๆ ไปสู่แนวคิดนามธรรมมากขึ้นในเลเยอร์ลึก
    </p>

    <h3 className="text-xl font-semibold">ลำดับชั้นของคุณลักษณะใน CNN</h3>

    <p>
      CNN มีการจัดเรียงเลเยอร์แบบลำดับชั้น โดยเลเยอร์แต่ละชั้นจะมีบทบาทในการจับรูปแบบเฉพาะเจาะจงในระดับที่ต่างกัน โดยลำดับชั้นสามารถแบ่งได้ดังนี้:
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>เลเยอร์ที่ 1:</strong> ตรวจจับขอบ เส้นตรง และสีพื้นฐาน เช่น ขอบแนวตั้ง แนวนอน หรือจุดตัด
      </li>
      <li>
        <strong>เลเยอร์ที่ 2:</strong> จับรูปทรงพื้นฐาน เช่น โค้ง มุม และพื้นผิวซ้ำ
      </li>
      <li>
        <strong>เลเยอร์ที่ 3:</strong> ตรวจจับส่วนประกอบของวัตถุ เช่น ดวงตา ล้อ หรือประตู
      </li>
      <li>
        <strong>เลเยอร์สุดท้าย:</strong> สร้างความเข้าใจในระดับวัตถุ เช่น จำแนกประเภทของวัตถุหรือฉาก
      </li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อดีของการเรียนรู้แบบลำดับชั้น</h3>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ฟีเจอร์ถูกปรับแต่งตามภารกิจเฉพาะ เช่น การจำแนกภาพ การแยกวัตถุ หรือการรู้จำใบหน้า</li>
      <li>สามารถรับมือกับความซับซ้อนของภาพ เช่น แสง มุมมอง หรือสิ่งรบกวนได้ดี</li>
      <li>นำกลับมาใช้ใหม่ได้ โดยเฉพาะใน Transfer Learning ซึ่งชั้นต้นของโมเดลมักสามารถใช้ร่วมกันได้ข้ามโดเมน</li>
    </ul>

    <p>
      รายวิชา MIT 6.S191 อธิบายไว้อย่างชัดเจนว่า โครงสร้างการเรียนรู้ลำดับชั้นของ CNN นั้นสอดคล้องกับโครงสร้างของสมองมนุษย์ โดยมีการจัดระเบียบจากภาพง่ายไปสู่ความหมายที่ลึกซึ้ง ผ่านการใช้คอนโวลูชัน การเปิดใช้งาน (Activation) และการลดขนาด (Pooling)
    </p>

    <h3 className="text-xl font-semibold">การมองภาพภายในเลเยอร์</h3>

    <p>
      งานวิจัยของ Zeiler & Fergus (2014) ใช้การ Visualize ฟีเจอร์เพื่อตรวจสอบสิ่งที่แต่ละเลเยอร์เรียนรู้ พบว่าเลเยอร์ต้นเรียนรู้ลักษณะขอบและพื้นผิว ส่วนเลเยอร์ลึกเข้าใจความหมายของวัตถุได้ดี
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>การจัดเรียงเลเยอร์ช่วยให้โมเดลเข้าใจโลกภาพแบบองค์รวม</li>
        <li>รูปแบบลำดับชั้นใน CNN คล้ายกับระบบการมองเห็นของสัตว์เลี้ยงลูกด้วยนม</li>
        <li>การถ่ายทอดการเรียนรู้ (Transfer Learning) ใช้ชั้นล่างของ CNN เพื่อเร่งการฝึก</li>
        <li>คุณลักษณะระดับสูงที่ได้จากเลเยอร์ลึกสามารถใช้ในหลายงาน เช่น NLP, Medical Imaging, Robotics</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">กรณีศึกษา: โฟลวของ ResNet-50</h3>

    <p>
      ResNet-50 มีการแบ่งเลเยอร์ออกเป็น 5 สเตจ โดยแต่ละสเตจมีโครงสร้าง Residual Blocks ซึ่งช่วยให้การไหลของ Gradient ไม่ถูกทำลาย ทำให้สามารถสร้างโมเดลลึกได้โดยไม่เกิดปัญหา Vanishing Gradient
    </p>

    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">สเตจ</th>
          <th className="border px-4 py-2">ขนาด Output</th>
          <th className="border px-4 py-2">ชนิดของคุณลักษณะ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Conv1</td>
          <td className="border px-4 py-2">112×112</td>
          <td className="border px-4 py-2">ขอบและพื้นผิวพื้นฐาน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Conv2_x</td>
          <td className="border px-4 py-2">56×56</td>
          <td className="border px-4 py-2">รูปทรงง่าย</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Conv3_x</td>
          <td className="border px-4 py-2">28×28</td>
          <td className="border px-4 py-2">รูปแบบและลวดลายซ้ำ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Conv4_x</td>
          <td className="border px-4 py-2">14×14</td>
          <td className="border px-4 py-2">องค์ประกอบของวัตถุ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Conv5_x</td>
          <td className="border px-4 py-2">7×7</td>
          <td className="border px-4 py-2">ความหมายระดับสูง</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">เปรียบเทียบกับระบบคลาสสิก</h3>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ระบบคลาสสิกต้องออกแบบฟีเจอร์ด้วยมือ CNN เรียนรู้ได้เอง</li>
      <li>ระบบเก่าไม่สามารถปรับตัวกับโดเมนใหม่ได้ดี CNN มีความยืดหยุ่นสูง</li>
      <li>ฟีเจอร์จาก CNN สามารถย้ายข้ามงานได้ ในขณะที่ระบบคลาสสิกทำไม่ได้</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุป</h3>

    <p>
      โครงข่าย CNN ได้เปลี่ยนรูปแบบการสกัดคุณลักษณะจากการพึ่งมนุษย์ไปสู่การเรียนรู้อย่างลำดับชั้น
      แนวคิดนี้นำไปสู่ความก้าวหน้าในการประมวลผลภาพอย่างมีประสิทธิภาพ และสามารถต่อยอดไปยังสาขาอื่นได้อีกมากมาย
    </p>

  </div>
</section>

<section id="classification" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. การจำแนกประเภทด้วย Convolutional Neural Networks (CNNs)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      หนึ่งในเป้าหมายหลักของการประมวลผลภาพคือการจำแนกประเภท (classification)
      Convolutional Neural Networks (CNNs) ได้กลายเป็นแกนหลักของงานประเภทนี้ด้วยโครงสร้างที่สามารถเรียนรู้ลำดับชั้นของคุณลักษณะจากดิบถึงเชิงนามธรรม
      ระบบการเรียนรู้แบบลึกช่วยให้การแยกแยะภาพที่ซับซ้อนสามารถทำได้อย่างมีประสิทธิภาพ
    </p>

    <p>
      CNN ทำงานโดยเปลี่ยนอินพุตภาพให้กลายเป็นเวกเตอร์ลักษณะ (feature vector) ผ่านชุดของเลเยอร์ ได้แก่ convolution, activation, pooling และ fully connected layers
      สุดท้าย feature vector นี้จะถูกส่งต่อไปยังเลเยอร์สุดท้ายที่ทำหน้าที่ในการตัดสินใจจำแนกคลาส
    </p>

    <h3 className="text-xl font-semibold">ลำดับกระบวนการจำแนกภาพด้วย CNN</h3>
    <ol className="list-decimal list-inside ml-6 space-y-2">
      <li><strong>Convolution:</strong> สกัดลักษณะเฉพาะจากตำแหน่งต่าง ๆ ของภาพด้วยฟิลเตอร์ที่เรียนรู้ได้</li>
      <li><strong>Activation (ReLU):</strong> เพิ่มความไม่เชิงเส้นให้กับระบบ เพื่อให้สามารถแยกข้อมูลที่ซับซ้อนได้ดีขึ้น</li>
      <li><strong>Pooling:</strong> ลดมิติของข้อมูล เพื่อเพิ่มความทนทานต่อการเลื่อนตำแหน่งหรือความเบี้ยวของวัตถุ</li>
      <li><strong>Flattening:</strong> เปลี่ยนข้อมูลแบบเชิงพื้นที่ให้กลายเป็นเวกเตอร์เดียว</li>
      <li><strong>Fully Connected Layer:</strong> วิเคราะห์ feature vector และทำการตัดสินใจว่าภาพนั้นจัดอยู่ในหมวดหมู่ใด</li>
    </ol>

    <p>
      โดยทั่วไปเลเยอร์สุดท้ายของ CNN ที่ทำ classification จะใช้ softmax function เพื่อแปลงค่าความมั่นใจ (logits) ให้กลายเป็นความน่าจะเป็นในแต่ละคลาส
      คลาสที่มีความน่าจะเป็นสูงสุดจะถูกเลือกเป็นผลลัพธ์
    </p>

    <h3 className="text-xl font-semibold">Softmax Function</h3>
    <p>
      Softmax เป็นฟังก์ชัน activation ที่มักใช้ในเลเยอร์สุดท้ายของโมเดล classification
      โดยรับเวกเตอร์ของค่าความมั่นใจ และแปลงให้กลายเป็นความน่าจะเป็นที่รวมกันได้ 1
    </p>
    <pre>
{`Softmax(z_i) = exp(z_i) / Σ(exp(z_j))  for j = 1 to K`}
    </pre>

    <p>
      ข้อดีของ Softmax คือสามารถตีความผลลัพธ์เป็นการแจกแจงความน่าจะเป็น ซึ่งเหมาะสำหรับการจำแนกภาพที่มีหลายคลาส
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างจาก ImageNet Classification</h3>
    <p>
      ในการแข่งขัน ImageNet โมเดล CNN จะได้รับภาพขนาด 224x224x3 เป็นอินพุต แล้วผ่านกระบวนการสกัดคุณลักษณะหลายเลเยอร์
      ในที่สุดโมเดลจะให้เวกเตอร์ขนาด 1x1000 ซึ่งแสดงความน่าจะเป็นของแต่ละคลาสใน ImageNet ทั้งหมด
    </p>

    <p>
      CNN ที่ประสบความสำเร็จเช่น AlexNet, VGG, GoogLeNet และ ResNet ได้แสดงให้เห็นว่าโครงสร้างลึก (deep architecture)
      ช่วยให้การจำแนกประเภทสามารถครอบคลุมรายละเอียดได้ดีขึ้น
    </p>

    <h3 className="text-xl font-semibold">การเรียนรู้พารามิเตอร์ผ่าน Backpropagation</h3>
    <p>
      พารามิเตอร์ของฟิลเตอร์ในแต่ละเลเยอร์จะถูกปรับด้วยการใช้ gradient descent และ backpropagation
      โดยเปรียบเทียบผลลัพธ์กับ ground truth แล้วคำนวณค่าความผิดพลาด (loss) ซึ่งจะถูกนำไปใช้ในการอัปเดตพารามิเตอร์
    </p>

    <p>
      ฟังก์ชัน loss ที่ใช้บ่อยใน classification ได้แก่ cross-entropy loss:
    </p>

    <pre>
{`Loss = -Σ y_i * log(p_i)`}
    </pre>

    <p>
      โดยที่ y_i คือค่าความจริง (0 หรือ 1), p_i คือค่าความน่าจะเป็นจาก softmax
    </p>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>เลเยอร์ convolution ช่วยให้โมเดลสามารถเข้าใจรูปแบบท้องถิ่น เช่น ขอบหรือพื้นผิว</li>
        <li>การใช้ pooling ทำให้โมเดลมีความสามารถในการต้านทานการเปลี่ยนแปลงของภาพ</li>
        <li>Softmax ช่วยให้โมเดลสามารถตัดสินใจโดยพิจารณาความน่าจะเป็นของทุกคลาส</li>
        <li>การใช้ cross-entropy loss ร่วมกับ backpropagation ทำให้โมเดลเรียนรู้ได้อย่างแม่นยำ</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">สถาปัตยกรรมที่ใช้สำหรับ Classification</h3>
    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-700 text-sm sm:text-base">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="border px-4 py-2">ชื่อโมเดล</th>
          <th className="border px-4 py-2">ปีที่เปิดตัว</th>
          <th className="border px-4 py-2">จุดเด่น</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">AlexNet</td>
          <td className="border px-4 py-2">2012</td>
          <td className="border px-4 py-2">ใช้ GPU และ ReLU บนข้อมูลขนาดใหญ่</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">VGGNet</td>
          <td className="border px-4 py-2">2014</td>
          <td className="border px-4 py-2">โครงสร้างลึก ใช้ convolution 3x3</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ResNet</td>
          <td className="border px-4 py-2">2015</td>
          <td className="border px-4 py-2">ใช้ residual connections เพื่อฝึกโมเดลลึกได้</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">EfficientNet</td>
          <td className="border px-4 py-2">2019</td>
          <td className="border px-4 py-2">เพิ่มประสิทธิภาพผ่านการ scaling เชิงซ้อน</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">สรุป</h3>
    <p>
      CNN กลายเป็นเครื่องมือหลักในการจำแนกประเภทภาพในหลากหลายสาขา
      การรวม convolutional layers, activation, pooling, และ fully connected layers ร่วมกัน
      ทำให้สามารถแปลงภาพเป็นการตัดสินใจที่แม่นยำและมีความน่าเชื่อถือ
      ปัจจุบัน CNN ยังคงเป็นพื้นฐานสำคัญในการพัฒนาโมเดลที่ใช้งานจริงในระบบประมวลผลภาพและ AI
    </p>

  </div>
</section>


<section id="applications" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การประยุกต์ใช้ในโลกความจริง</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    
    <p>
      เครือข่ายประสาทเทียมแบบคอนโวลูชัน (CNNs) ได้กลายเป็นแกนหลักของระบบวิสัยทัศน์คอมพิวเตอร์ในโลกแห่งความจริง
      จากการประยุกต์ใช้ในการแพทย์จนถึงยานยนต์ไร้คนขับ การออกแบบโมเดลที่สามารถเรียนรู้ลักษณะของภาพจากข้อมูลโดยตรงได้พลิกโฉมวิธีการที่ปัญญาประดิษฐ์มองเห็นโลก
    </p>

    <h3 className="text-xl font-semibold">การวิเคราะห์ทางการแพทย์</h3>

    <p>
      งานวิจัยจาก Stanford และ MIT ยืนยันว่า CNN สามารถตรวจพบมะเร็งผิวหนังจากภาพถ่ายด้วยความแม่นยำเทียบเท่าผู้เชี่ยวชาญทางผิวหนัง
      โดยการฝึกจากภาพที่มีการติดป้ายกำกับเฉพาะทาง CNN สามารถแยกแยะลักษณะเฉพาะของเซลล์ผิดปกติได้
    </p>

    <p>
      นอกจากนี้ ในการวินิจฉัยโรคปอดอักเสบจากภาพเอกซเรย์ CNN ที่ได้รับการฝึกบนชุดข้อมูล ChestX-ray14 ของ NIH สามารถตรวจพบจุดผิดปกติและทำหน้าที่เป็นเครื่องมือช่วยแพทย์ในการตัดสินใจ
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ตรวจจับเนื้องอกในสมองจากภาพ MRI</li>
      <li>วิเคราะห์โครงสร้างของหัวใจจากภาพอัลตราซาวด์</li>
      <li>ตรวจจับภาวะเบาหวานขึ้นจอประสาทตาจากภาพเรตินา</li>
    </ul>

    <h3 className="text-xl font-semibold">ยานยนต์ไร้คนขับ</h3>

    <p>
      CNN เป็นส่วนประกอบสำคัญของระบบรับรู้ภาพในรถยนต์ไร้คนขับ โดยทำหน้าที่ประมวลผลภาพจากกล้องเพื่อตรวจจับป้ายจราจร ทางม้าลาย คนเดินถนน และรถคันอื่น
    </p>

    <p>
      ตามหลักสูตรของ MIT 6.S191, CNN ถูกฝึกให้รู้จำวัตถุในสภาพแวดล้อมจริงที่ซับซ้อน เช่น เวลากลางคืน หรือเมื่อมีแสงจ้าเข้ากล้อง
    </p>

    <p>
      ตัวอย่างจาก Tesla และ Waymo แสดงให้เห็นว่า CNN ถูกใช้ควบคู่กับ LIDAR และเรดาร์ เพื่อรวมข้อมูลหลายมิติในการตัดสินใจของระบบนำทาง
    </p>

    <h3 className="text-xl font-semibold">ระบบความปลอดภัยสาธารณะ</h3>

    <p>
      CNN ถูกนำไปใช้ในการรู้จำใบหน้าในสนามบิน สถานีรถไฟ และระบบตรวจสอบกล้องวงจรปิด โดยสามารถจับคู่ใบหน้าในฐานข้อมูลกับผู้ที่ปรากฏในภาพได้ในเวลาจริง
    </p>

    <p>
      ในประเทศจีน มีการประยุกต์ CNN สำหรับตรวจจับพฤติกรรมต้องสงสัยในที่สาธารณะ เช่น การวิเคราะห์ท่าทางจากกล้องเพื่อส่งสัญญาณเตือนเมื่อพบพฤติกรรมผิดปกติ
    </p>

    <h3 className="text-xl font-semibold">การเกษตรและสิ่งแวดล้อม</h3>

    <p>
      การวิเคราะห์ภาพถ่ายดาวเทียมเพื่อประเมินผลผลิตทางการเกษตรและติดตามการเปลี่ยนแปลงของสภาพแวดล้อมอาศัย CNN ในการประมวลผลภาพขนาดใหญ่จากหลายแหล่ง
    </p>

    <p>
      การตรวจจับการรั่วไหลของน้ำมัน การเฝ้าระวังไฟป่า และการตรวจวัดมลพิษในแม่น้ำเป็นกรณีศึกษาที่แสดงถึงการใช้ CNN ในงานสิ่งแวดล้อมระดับโลก
    </p>

    <h3 className="text-xl font-semibold">การวิเคราะห์เนื้อหาภาพและวิดีโอ</h3>

    <p>
      แพลตฟอร์มเช่น YouTube และ TikTok ใช้ CNN ในการตรวจสอบเนื้อหาไม่เหมาะสมหรือการละเมิดลิขสิทธิ์
      โดยระบบจะวิเคราะห์เฟรมวิดีโอและแยกแยะวัตถุหรือโลโก้เพื่อเปรียบเทียบกับฐานข้อมูล
    </p>

    <p>
      ในการสืบสวนด้านนิติเวช CNN ถูกใช้ในการตรวจจับการแก้ไขภาพ (image forgery) และตรวจสอบความแท้ของภาพถ่ายหรือวิดีโอ
    </p>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>
          CNN ช่วยลดการพึ่งพาคุณลักษณะที่มนุษย์นิยามไว้ล่วงหน้า และแทนที่ด้วยการเรียนรู้จากข้อมูลจริง
        </li>
        <li>
          สถาบันชั้นนำ เช่น DeepMind และ Carnegie Mellon ใช้ CNN ในงานวิจัยการตีความภาพระดับสูง เช่น การคาดเดาความลึกจากภาพเพียงภาพเดียว
        </li>
        <li>
          เทคโนโลยี CNN กำลังถูกปรับใช้ในอุปกรณ์ edge เช่น โทรศัพท์มือถือและกล้องวงจรปิดที่มีหน่วยประมวลผลเฉพาะ
        </li>
        <li>
          ความสามารถในการทำงานแบบ end-to-end ของ CNN ทำให้สามารถสร้างระบบอัตโนมัติที่เชื่อถือได้ในภาคอุตสาหกรรม
        </li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">สรุป</h3>

    <p>
      CNN ได้รับการพิสูจน์แล้วว่ามีประสิทธิภาพในการรับรู้ภาพในโลกจริง ไม่ว่าจะเป็นในทางการแพทย์ การคมนาคม หรือความมั่นคง
      การเรียนรู้เชิงลำดับชั้น (hierarchical learning) และการใช้ทรัพยากรที่มีประสิทธิภาพทำให้ CNN เป็นโครงสร้างพื้นฐานของระบบ AI ที่ต้องการการมองเห็น
    </p>

  </div>
</section>


<section id="case-studies" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. กรณีศึกษาและความท้าทายในระบบวิชันด้วย CNN</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การประยุกต์ใช้งาน CNN ในโลกจริงได้นำไปสู่การพัฒนาอย่างก้าวกระโดดในด้านวิทยาการคอมพิวเตอร์ วิศวกรรม และชีวการแพทย์
      อย่างไรก็ตาม การนำ CNN ไปใช้ในระดับ production ยังคงพบอุปสรรคในหลายมิติ เช่น ความเข้าใจโมเดล, ความสามารถในการ generalize, ความต้องการข้อมูลจำนวนมาก และประเด็นด้านความยุติธรรมและความโปร่งใส
    </p>

    <h3 className="text-xl font-semibold">กรณีศึกษา 1: การตรวจจับมะเร็งผิวหนังด้วย CNN</h3>
    <p>
      งานวิจัยของมหาวิทยาลัย Stanford (Esteva et al., 2017) ใช้ CNN เพื่อจำแนกรอยโรคผิวหนังมากกว่า 130,000 ภาพ ครอบคลุมกว่า 2,000 ชนิด
      ผลลัพธ์ของโมเดลเทียบเท่าหรือดีกว่าแพทย์ผิวหนังที่มีประสบการณ์ โดยใช้โครงข่าย Inception v3 เป็นแกนกลาง และปรับ fine-tune ด้วยข้อมูลเฉพาะทาง
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>โมเดลเรียนรู้จากภาพระดับ consumer-grade ได้โดยไม่ต้องใช้การกำกับอย่างละเอียด</li>
      <li>ใช้การ balance dataset อย่างระมัดระวังเพื่อป้องกัน class imbalance</li>
      <li>การประเมินโมเดลใช้ ROC-AUC และ Precision-Recall เพื่อความแม่นยำเชิงคลินิก</li>
    </ul>

    <h3 className="text-xl font-semibold">กรณีศึกษา 2: ระบบรู้จำใบหน้าบนระดับประเทศ (FaceNet)</h3>
    <p>
      Google Research พัฒนา FaceNet (Schroff et al., 2015) เพื่อเรียนรู้การฝัง (embedding) ใบหน้าในระยะเชิงเวกเตอร์
      ด้วยการฝึก CNN ที่เรียนรู้ความแตกต่างระหว่างบุคคล แทนการจำแนกโดยตรง โดยใช้ loss แบบ triplet loss เพื่อแยกเวกเตอร์ใบหน้าระหว่างบุคคลให้ห่างที่สุด
    </p>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ลดความต้องการในการสร้างคลาสใหม่ทุกครั้งที่เพิ่มบุคคลใหม่</li>
      <li>ใช้ Deep Metric Learning ในสเกลข้อมูลหลายล้านภาพ</li>
      <li>มีความแม่นยำสูงบน Labeled Faces in the Wild (LFW) dataset ถึง 99.63%</li>
    </ul>

    <h3 className="text-xl font-semibold">กรณีศึกษา 3: การวิเคราะห์ภาพทางการแพทย์ระดับ 3 มิติ</h3>
    <p>
      ในการวิเคราะห์ MRI และ CT scans จำเป็นต้องใช้ CNN แบบ 3 มิติ เช่น V-Net หรือ MedicalNet
      เพื่อรองรับข้อมูลที่มีมิติความลึก เช่น สมอง หลอดเลือด และเนื้อเยื่อภายใน
      การฝึกโมเดลลักษณะนี้มีความท้าทายมากขึ้นจากการใช้ทรัพยากร GPU และปริมาณข้อมูลจำกัด
    </p>

    <p>
      Harvard และ MIT ใช้เทคนิค semi-supervised learning และ data augmentation เพื่อเพิ่มความสามารถในการเรียนรู้จากข้อมูลจำกัด เช่น self-training และ pseudo-labeling
    </p>

    <h3 className="text-xl font-semibold">ความท้าทายหลักในการนำ CNN ไปใช้งานจริง</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>1. Overfitting:</strong> เกิดขึ้นง่ายหากใช้ dataset เล็กหรือมี class imbalance โดยเฉพาะในข้อมูลทางการแพทย์</li>
      <li><strong>2. Adversarial Attacks:</strong> ภาพที่ถูกปรับค่าพิกเซลเล็กน้อยสามารถหลอกโมเดลให้ทำนายผิดได้</li>
      <li><strong>3. Explainability:</strong> CNN เป็นโมเดลที่ตีความยากในหลายกรณี เช่น การวิเคราะห์ผิดพลาดในงานยุติธรรม</li>
      <li><strong>4. Bias & Fairness:</strong> ข้อมูลที่ฝึกอาจส่งผลให้โมเดลมีอคติต่อเพศ สีผิว หรือกลุ่มชาติพันธุ์</li>
      <li><strong>5. Computational Demand:</strong> CNN ขนาดใหญ่ต้องการ GPU ที่มีหน่วยความจำสูงและใช้พลังงานมาก</li>
    </ul>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>CNN เป็นเครื่องมือทรงพลังในการแก้ปัญหาภาพ แต่ต้องใช้ข้อมูลและทรัพยากรอย่างเหมาะสม</li>
        <li>ความสำเร็จของโมเดลขึ้นอยู่กับการเตรียมข้อมูล การประเมินผล และกลยุทธ์ลด overfitting</li>
        <li>การพัฒนาโมเดลที่ตีความได้ (interpretable CNNs) เช่น Grad-CAM หรือ Explainable AI เป็นแนวทางสำคัญในอนาคต</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ข้อเสนอแนะเชิงกลยุทธ์จากงานวิจัย</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ใช้ Transfer Learning เพื่อลดเวลาและข้อมูลที่ต้องใช้ในการฝึก</li>
      <li>ทำ Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล โดยเฉพาะในงานทางการแพทย์</li>
      <li>ใช้ techniques เช่น Dropout, BatchNorm และ Early Stopping เพื่อป้องกัน overfitting</li>
      <li>หากต้องการ deploy บน edge device เช่น IoT ควรใช้ MobileNet หรือ EfficientNet-lite</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      แม้ CNN ได้รับการพิสูจน์ว่าทรงพลังในหลายบริบทจริง แต่การใช้งานระดับอุตสาหกรรมต้องการการออกแบบที่รอบคอบในทุกขั้นตอน ตั้งแต่การเตรียมข้อมูลไปจนถึงการ deploy
      งานวิจัยและการพัฒนาในทศวรรษหน้าอาจมุ่งเน้นไปที่ความเข้าใจเชิงลึก (interpretability), การลดการใช้พลังงาน และการประยุกต์กับ multimodal systems เช่น vision-language models
    </p>

  </div>
</section>


<section id="research" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. งานวิจัยเด่นด้าน CNN ในวิสัยทัศน์คอมพิวเตอร์</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      งานวิจัยในช่วงทศวรรษที่ผ่านมาได้ผลักดันขีดความสามารถของ Convolutional Neural Networks (CNNs) ไปไกลกว่าการจำแนกภาพแบบดั้งเดิม
      โดยมุ่งเน้นทั้งการเพิ่มประสิทธิภาพของโครงสร้าง การลดค่าใช้จ่ายในการคำนวณ และการเพิ่มความเข้าใจเชิงเหตุผลจากภาพ
      ต่อไปนี้คือไฮไลต์ของผลงานวิจัยที่มีอิทธิพลสูงจากมหาวิทยาลัยและองค์กรวิจัยระดับโลก
    </p>

    <h3 className="text-xl font-semibold">1. Residual Networks (ResNet) - Microsoft Research (2015)</h3>
    <p>
      งานวิจัยโดย He et al. ที่ Microsoft Research นำเสนอ ResNet ซึ่งเป็นโครงสร้างแบบ residual ที่สามารถฝึกโมเดลลึกมากกว่า 150 ชั้นได้สำเร็จโดยไม่เกิดปัญหา gradient vanishing
      แนวคิดของ residual connection ที่ให้ข้อมูลไหลผ่านแบบ identity shortcut กลายเป็นมาตรฐานสำหรับโครงสร้าง deep learning ทุกรูปแบบ
    </p>

    <h3 className="text-xl font-semibold">2. Visual Attention Mechanisms - Stanford & Google Brain</h3>
    <p>
      การเพิ่ม attention module เข้าไปใน CNN ช่วยให้โมเดลสามารถโฟกัสกับพื้นที่สำคัญในภาพได้แบบ adaptive
      งานจาก Stanford (Xu et al.) และ Google Brain (Bahdanau-style attention adapted for vision) ทำให้ CNN เข้าใจภาพในเชิงความหมายได้ลึกขึ้น
    </p>

    <h3 className="text-xl font-semibold">3. EfficientNet - Google AI (2019)</h3>
    <p>
      Tan & Le จาก Google AutoML นำเสนอ EfficientNet ซึ่งใช้เทคนิค compound scaling ปรับขนาด depth, width และ resolution พร้อมกัน
      ส่งผลให้โมเดลขนาดเล็กมีประสิทธิภาพเหนือกว่าโมเดลใหญ่ในหลาย benchmark โดยใช้พารามิเตอร์น้อยลงกว่าครึ่ง
    </p>

    <h3 className="text-xl font-semibold">4. Visual Transformers (ViT) - Google Brain (2020)</h3>
    <p>
      Dosovitskiy et al. เสนอการใช้ Transformer แทน CNN โดยแบ่งภาพเป็น patch แล้วใช้ self-attention เหมือน NLP
      แม้ว่า ViT ไม่ใช่ CNN โดยตรง แต่มีผลต่อการพัฒนา CNN รุ่นใหม่ให้มี self-attention module ผสม (เช่น ConvNeXt, BoTNet)
    </p>

    <h3 className="text-xl font-semibold">5. Grad-CAM: Explainable CNNs - Georgia Tech</h3>
    <p>
      Selvaraju et al. จาก Georgia Tech พัฒนา Grad-CAM ซึ่งสามารถสร้างแผนที่ความร้อน (heatmap) บ่งบอกว่าพื้นที่ใดในภาพที่ CNN ใช้ตัดสินใจ
      เทคนิคนี้กลายเป็นมาตรฐานสำหรับการทำ interpretability ในงานวิสัยทัศน์คอมพิวเตอร์
    </p>

    <h3 className="text-xl font-semibold">6. Self-Supervised Learning for CNNs - Facebook AI Research (FAIR)</h3>
    <p>
      งานอย่าง MoCo, SimCLR และ BYOL จาก FAIR พัฒนาการฝึก CNN โดยไม่ต้องมี label ผ่าน contrastive learning และ momentum encoders
      การเรียนรู้แบบ self-supervised ทำให้ CNN สามารถใช้ข้อมูล unlabeled ได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">7. CNNs สำหรับ 3D Vision - Berkeley AI Research (BAIR)</h3>
    <p>
      งานจาก BAIR เช่น PointNet, 3D ConvNet และ voxel CNN แสดงให้เห็นว่า CNN สามารถประยุกต์กับข้อมูล 3 มิติ เช่น point cloud และ voxel grid ได้
      ซึ่งเป็นพื้นฐานสำหรับงานเช่น AR/VR และการสร้างภาพ 3 มิติจากกล้องหลายมุม
    </p>

    <h3 className="text-xl font-semibold">8. CNNs ในภาพทางการแพทย์ - Harvard Medical School & MIT CSAIL</h3>
    <p>
      การใช้งาน CNN ในการวินิจฉัยจากภาพ CT, MRI และ X-ray มีการวิจัยจาก Harvard และ MIT ที่พัฒนาโครงสร้างเฉพาะด้าน เช่น U-Net และ Attention U-Net
      เพื่อช่วยในงาน segmentation, detection และการจำแนกโรคอย่างแม่นยำ
    </p>

    <h3 className="text-xl font-semibold">9. Neuromorphic CNNs - MIT Media Lab</h3>
    <p>
      MIT Media Lab พัฒนาแนวทางที่จำลองการประมวลผลแบบสมองจริง เช่น SNN (Spiking Neural Networks) ร่วมกับ CNN
      เพื่อให้สามารถประมวลผลแบบประหยัดพลังงานบนอุปกรณ์ edge ได้
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>ResNet ทำให้การฝึกโมเดลลึกหลายร้อยชั้นเป็นไปได้จริง</li>
        <li>Visual Attention ทำให้ CNN เข้าใจภาพในเชิงบริบทได้ดีขึ้น</li>
        <li>EfficientNet พิสูจน์ว่าโมเดลที่ออกแบบอย่างเหมาะสมสามารถเล็กและเร็วได้พร้อมกัน</li>
        <li>Grad-CAM ช่วยให้ตรวจสอบและตีความ CNN ได้แบบโปร่งใส</li>
        <li>Self-Supervised CNN เรียนรู้จากข้อมูล unlabeled ได้อย่างทรงพลัง</li>
        <li>CNN สำหรับภาพ 3 มิติเปิดทางสู่งานวิจัย AR/VR และหุ่นยนต์เชิงพื้นที่</li>
        <li>การประยุกต์ในแพทย์แม่นยำขึ้นด้วยการออกแบบโครงสร้างเฉพาะ</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">แหล่งอ้างอิงงานวิจัย</h3>
    <ul className="list-disc list-inside ml-6 space-y-2 text-sm">
      <li>He et al., "Deep Residual Learning for Image Recognition", CVPR 2016</li>
      <li>Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019</li>
      <li>Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021</li>
      <li>Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017</li>
      <li>Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020</li>
      <li>Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015</li>
    </ul>

    <p>
      งานวิจัยเหล่านี้ไม่เพียงแต่ผลักดันขีดความสามารถของ CNN ในเชิงความแม่นยำ แต่ยังขยายขอบเขตการประยุกต์ของโมเดล
      จากการรู้จำภาพไปสู่งานที่ซับซ้อนและมีผลกระทบต่อมนุษย์โดยตรง เช่น การแพทย์ ยานยนต์ และปัญญาประดิษฐ์ระดับระบบ
    </p>
  </div>
</section>



<section id="insight-recap" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. บทสรุปเชิงลึก (Insight Recap)</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img8} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Convolutional Neural Networks (CNNs) ได้กลายเป็นโครงสร้างหลักในงานวิสัยทัศน์คอมพิวเตอร์ โดยความสามารถในการสกัดคุณลักษณะเชิงลึกและการเข้าใจลำดับชั้นของภาพทำให้ CNN มีบทบาทสำคัญในแอปพลิเคชันทั้งทางวิชาการและอุตสาหกรรม เช่น การรู้จำใบหน้า, การขับเคลื่อนอัตโนมัติ และการวินิจฉัยทางการแพทย์
    </p>

    <ul className="list-disc list-inside space-y-2">
      <li>
        สถาปัตยกรรม CNN ประกอบด้วยเลเยอร์ต่าง ๆ ที่ทำงานร่วมกันเพื่อเรียนรู้จากข้อมูลภาพ ทั้งในระดับต่ำ (edges, textures) ไปจนถึงระดับสูง (วัตถุ, รูปแบบเชิงนามธรรม)
      </li>
      <li>
        แนวคิดเรื่องการเรียนรู้ลำดับชั้นและการใช้ฟิลเตอร์ร่วมกัน (shared weights) ทำให้โมเดลสามารถลดจำนวนพารามิเตอร์และยังคงประสิทธิภาพสูง
      </li>
      <li>
        การใช้งานจริงของ CNN มีอยู่ในทุกมิติของระบบ AI ที่ต้องเข้าใจข้อมูลภาพ เช่น การตรวจจับวัตถุ การแบ่งส่วนภาพ และการติดตามการเคลื่อนไหว
      </li>
      <li>
        งานวิจัยจาก MIT, Stanford, Google Brain และ FAIR ได้พัฒนาเทคนิคใหม่ เช่น Residual Networks, Attention, Efficient Scaling และ Explainability ซึ่งช่วยเพิ่มความลึก ความเร็ว และความโปร่งใสของโมเดล
      </li>
      <li>
        CNNs ไม่ได้จำกัดอยู่เพียงภาพ 2 มิติ แต่ยังสามารถประยุกต์กับข้อมูล 3 มิติ, วิดีโอ, และแม้แต่ภาพจากเครื่องมือแพทย์ผ่านโครงสร้างเฉพาะด้าน
      </li>
      <li>
        การเรียนรู้แบบ Self-Supervised ช่วยให้โมเดล CNN ใช้ประโยชน์จากข้อมูลที่ไม่มีป้ายกำกับ ทำให้ลดต้นทุนการฝึกและเปิดโอกาสให้ใช้ข้อมูลในโลกจริงได้มากขึ้น
      </li>
    </ul>

    <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box: แนวโน้มของ CNN ในอนาคต</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          โมเดล CNN รุ่นใหม่เริ่มผสานความสามารถของ Transformer เพื่อสร้าง Hybrid Models ที่แม่นยำและตีความได้ดีขึ้น
        </li>
        <li>
          เทคโนโลยี Edge AI ทำให้มีการพัฒนา CNN ที่เบาและเร็ว เพื่อทำงานได้บนอุปกรณ์พกพาหรือ IoT
        </li>
        <li>
          การตีความ (Interpretability) จะกลายเป็นองค์ประกอบสำคัญของระบบ CNN ที่นำไปใช้งานจริง โดยเฉพาะในภาคการแพทย์และความปลอดภัย
        </li>
        <li>
          งานวิจัยต่อไปจะมุ่งเน้นการผสมผสานหลาย Modal (ภาพ, เสียง, ข้อความ) ซึ่งต้องอาศัย CNN เป็นรากฐานของการประมวลผลเชิงภาพ
        </li>
      </ul>
    </div>

    <p>
      การทำความเข้าใจโครงสร้างของ CNN อย่างลึกซึ้ง รวมถึงการติดตามงานวิจัยล่าสุด จะเป็นปัจจัยสำคัญในการพัฒนาโมเดลที่ทั้งแม่นยำ ยืดหยุ่น และเหมาะสมกับบริบทของการใช้งานจริงในยุคของ AI ที่เปลี่ยนแปลงอย่างรวดเร็ว
    </p>

  </div>
</section>




          {/* Quiz Section */}
          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day24 theme={theme} />
          </section>

          {/* Tags Section */}
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

          {/* Comments */}
          <Comments theme={theme} />
        </div>
      </div>

      {/* ScrollSpy Sidebar */}
      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day24 />
      </div>

      {/* Support Me Button */}
      <SupportMeButton />
    </div>
  );
};

export default Day24_CNNVision;
