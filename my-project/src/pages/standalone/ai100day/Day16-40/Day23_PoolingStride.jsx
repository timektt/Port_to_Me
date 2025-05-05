import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day23 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day23";
import MiniQuiz_Day23 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day23";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day23_PoolingStride = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });
  

  const img1 = cld.image("Pooling1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("Pooling2").format("auto").quality("auto").resize(scale().width(600));
  const img3 = cld.image("Pooling3").format("auto").quality("auto").resize(scale().width(600));
  const img4 = cld.image("Pooling4").format("auto").quality("auto").resize(scale().width(600));
  const img5 = cld.image("Pooling5").format("auto").quality("auto").resize(scale().width(600));
  const img6 = cld.image("Pooling6").format("auto").quality("auto").resize(scale().width(600));
  const img7 = cld.image("Pooling7").format("auto").quality("auto").resize(scale().width(600));
  const img8 = cld.image("Pooling8").format("auto").quality("auto").resize(scale().width(600));
  const img9 = cld.image("Pooling9").format("auto").quality("auto").resize(scale().width(600));
  const img10 = cld.image("Pooling10").format("auto").quality("auto").resize(scale().width(600));
  const img11 = cld.image("Pooling11").format("auto").quality("auto").resize(scale().width(590));
  const img12 = cld.image("Pooling12").format("auto").quality("auto").resize(scale().width(590));
  const img13 = cld.image("Pooling13").format("auto").quality("auto").resize(scale().width(590));


  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 23: Pooling & Stride Techniques</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
          <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
        </div>

       {/* Section 1 */}
       <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไมต้อง Pool และ Stride?</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img2} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      ในการประมวลผลภาพหรือข้อมูลเชิงพื้นที่ผ่านโครงข่ายประสาทเทียมประเภท Convolutional Neural Networks (CNNs)
      ข้อมูลที่ใช้มีลักษณะเป็น Tensor ที่มีขนาดใหญ่ เช่น ภาพความละเอียดสูงหรือข้อมูลแบบ multichannel
      การคงไว้ซึ่งขนาดของข้อมูลในแต่ละเลเยอร์จะทำให้เกิดปัญหาเชิงประสิทธิภาพทั้งในด้านหน่วยความจำและเวลาในการคำนวณ
    </p>

    <p>
      Pooling และ Stride คือสองกลไกหลักที่ใช้เพื่อควบคุมขนาดของ Tensor ที่ไหลผ่านโครงข่าย
      โดยทั้งสองเทคนิคถูกออกแบบมาเพื่อลดความละเอียดของข้อมูลลงอย่างเป็นระบบโดยไม่สูญเสียลักษณะสำคัญของข้อมูล
      ซึ่งแนวคิดนี้มีรากฐานจากแนวคิดทางคณิตศาสตร์ของ signal compression และ pattern abstraction
    </p>

    <p>
      งานวิจัยจาก Stanford University (CS231n) ระบุว่าการลดขนาดเชิงพื้นที่ของข้อมูลด้วย Pooling มีบทบาทในการเพิ่ม Robustness
      ของโมเดลต่อการเปลี่ยนตำแหน่งของวัตถุในภาพ เช่น การเลื่อน (Translation), การบิดเบือนเล็กน้อย (Distortion) หรือมุมมองที่ต่างกัน
      สิ่งนี้ช่วยให้โมเดลสามารถเรียนรู้ Feature ที่ "สำคัญจริง" แทนที่จะจดจำรายละเอียดเชิงพิกเซล
    </p>

    <p>
      การใช้ Stride ที่มีค่ามากกว่า 1 ใน Convolutional Layer จะช่วยลดจำนวนตำแหน่งที่ทำการ Convolve ได้อย่างมาก
      ส่งผลให้ Feature Map มีขนาดเล็กลง ซึ่งนำไปสู่การลดภาระทางคำนวณในลำดับขั้นถัดไป
      กลยุทธ์นี้เรียกว่า Strided Convolution และถูกใช้เป็นทางเลือกแทน Pooling ในบางสถาปัตยกรรม เช่น ResNet หรือ GoogLeNet
    </p>

    <p>
      ข้อดีหลักของการทำ Pooling ได้แก่:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>ลดจำนวนพารามิเตอร์ในโมเดลลงอย่างมีนัยสำคัญ</li>
      <li>ลดโอกาส Overfitting โดยการสรุปลักษณะเด่นและตัดข้อมูลที่ไม่จำเป็น</li>
      <li>เพิ่ม Invariance ต่อการเลื่อนตำแหน่งของ Feature ในภาพ</li>
    </ul>

    <p>
      การศึกษาโดย University of Toronto ซึ่งเป็นต้นกำเนิดของ AlexNet แสดงให้เห็นว่า Max Pooling แบบ 2×2 พร้อม Stride = 2
      มีผลอย่างมากต่อความสามารถในการเรียนรู้ Feature ที่มีความหมายในระดับสูง และสามารถลดขนาด Feature Map ลง 75% ในแต่ละเลเยอร์
    </p>

    <p>
      Pooling มีหลายรูปแบบ เช่น Max Pooling, Average Pooling และ Global Pooling ซึ่งแต่ละแบบมีข้อดีที่แตกต่างกัน
      โดย Max Pooling จะเก็บค่าที่สูงสุดในแต่ละ Region เพื่อเน้นลักษณะเด่นที่สุด
      ส่วน Average Pooling จะหาค่าเฉลี่ยซึ่งเหมาะกับการรักษาลักษณะทั่วไปของภาพ
    </p>

    <p>
      ในการออกแบบโครงข่ายแบบลึก เช่น VGGNet หรือ ResNet การใช้ Pooling และการกำหนดค่า Stride
      จำเป็นต้องมีการวางแผนอย่างรัดกุม เพราะหากลดขนาดเร็วเกินไปอาจทำให้ข้อมูลสูญหายมากเกินไป
      ในทางกลับกัน หากไม่ลดขนาดเลยจะทำให้โมเดลใช้ทรัพยากรมากและเกิด Overfitting ได้ง่าย
    </p>

    <p>
      จากข้อมูลในงานวิจัยของ Ioffe & Szegedy (Batch Normalization, 2015) และ He et al. (ResNet, 2016)
      การออกแบบตำแหน่งของ Pooling Layer และการใช้ Strided Convolution ต้องมีความสมดุลระหว่างการสกัด Feature
      และการคงไว้ซึ่ง Spatial Resolution ที่เพียงพอ
    </p>

    <p>
      ภาพประกอบด้านล่างแสดงให้เห็นถึงผลของการใช้ Max Pooling กับ Filter ขนาด 2×2 บนภาพขนาด 4×4 โดยมี Stride = 2
      จะได้ผลลัพธ์เป็นภาพขนาด 2×2 ซึ่งช่วยลดข้อมูลลงได้ 75% แต่ยังคงเก็บลักษณะเด่นของข้อมูลไว้ได้อย่างมีประสิทธิภาพ
    </p>

          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img3} />
          </div>

    <p>
      โดยสรุป Pooling และ Stride คือเครื่องมือสำคัญในการควบคุมขนาดของข้อมูลระหว่างทางใน Deep Neural Network
      ทั้งสองกลไกนี้ไม่เพียงแต่ช่วยให้โมเดลประมวลผลได้เร็วขึ้นและมีขนาดเล็กลง แต่ยังช่วยให้โมเดลสามารถจับลักษณะสำคัญของข้อมูล
      ได้อย่างมีประสิทธิภาพมากขึ้นในสถานการณ์ที่ Feature มีการเปลี่ยนตำแหน่งเล็กน้อยหรือมีการบิดเบือน
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          การทำ Max Pooling มีลักษณะคล้ายการสกัดลักษณะเด่นเชิงตำแหน่งจากภาพ โดยการเลือกค่าที่ “โดดเด่นที่สุด”
        </li>
        <li>
          การกำหนดค่า Stride มีผลโดยตรงต่อ Spatial Downsampling และอัตราการลดของ Feature Map ต้องควบคุมอย่างรอบคอบ
        </li>
        <li>
          การลดขนาดอย่างรุนแรงในช่วงต้นของโมเดลอาจทำให้สูญเสียข้อมูลละเอียดที่สำคัญ เช่น ขอบวัตถุหรือ texture
        </li>
        <li>
          ในงาน Edge AI และ Mobile Deployment เช่น MobileNet, การเลือกใช้ Global Average Pooling แทน Fully Connected Layer
          ช่วยลดพารามิเตอร์และพลังงานได้อย่างมาก
        </li>
      </ul>
    </div>
  </div>
</section>


<section id="max-vs-avg" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Max Pooling vs Average Pooling</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img4} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การทำ Pooling มีบทบาทสำคัญในการลดขนาดเชิงพื้นที่ของ Feature Maps ในเครือข่าย Convolutional Neural Networks (CNNs) โดยไม่ทำลายข้อมูลสำคัญของภาพ
      รูปแบบที่ใช้บ่อยที่สุดในงานวิจัยและการใช้งานจริงมีสองประเภท คือ Max Pooling และ Average Pooling ซึ่งแต่ละประเภทมีคุณสมบัติที่แตกต่างกันในเชิงการสกัด Feature
    </p>

    <p>
      Max Pooling คือการเลือกค่าที่มีขนาดมากที่สุดภายใน Region ที่กำหนด เช่น 2×2 หรือ 3×3 ซึ่งจะช่วยเน้นลักษณะเด่นที่รุนแรงที่สุดในบริเวณนั้น ๆ
      โดยเชื่อว่าค่าที่สูงสุดในแต่ละ Region คือค่าที่แสดงถึงการปรากฏของ Feature ที่สำคัญมากที่สุด เช่น ขอบหรือ texture ที่ชัดเจน
    </p>

    <p>
      Average Pooling จะคำนวณค่าเฉลี่ยของค่าภายใน Region ทั้งหมด ซึ่งให้ผลลัพธ์ที่เป็นการแทนความเข้มเฉลี่ยของ Feature ภายในพื้นที่นั้น
      เทคนิคนี้เหมาะกับการรักษาภาพรวมของข้อมูลเชิงพื้นที่ และมักถูกใช้ในช่วงท้ายของโมเดล เช่น Global Average Pooling ในเครือข่ายเชิงลึกหลายชั้น เช่น Inception และ MobileNet
    </p>

    <p>
      งานวิจัยจาก Stanford (CS231n) ชี้ให้เห็นว่า Max Pooling มีแนวโน้มให้ผลลัพธ์ที่ดีกว่า Average Pooling ในงานที่เน้นการแยกแยะ Feature เช่น การจำแนกภาพหรือการตรวจจับวัตถุ
      เนื่องจาก Max Pooling สามารถเก็บ Feature ที่ชัดเจนได้ดีกว่าโดยไม่ถูกเฉลี่ยทับกับค่าที่ไม่สำคัญ
    </p>

    <p>
      ข้อเปรียบเทียบในแง่ต่าง ๆ ของ Max Pooling และ Average Pooling มีดังนี้:
    </p>

    <div className="overflow-x-auto">
      <table className="table-auto w-full text-left border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="px-4 py-2">คุณสมบัติ</th>
            <th className="px-4 py-2">Max Pooling</th>
            <th className="px-4 py-2">Average Pooling</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-gray-300 dark:border-gray-700">
            <td className="px-4 py-2">ลักษณะ</td>
            <td className="px-4 py-2">เลือกค่าสูงสุด</td>
            <td className="px-4 py-2">คำนวณค่าเฉลี่ย</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-700">
            <td className="px-4 py-2">ความไวต่อ Feature</td>
            <td className="px-4 py-2">สูง</td>
            <td className="px-4 py-2">ต่ำ</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-700">
            <td className="px-4 py-2">ความทนทานต่อ Noise</td>
            <td className="px-4 py-2">ต่ำกว่า</td>
            <td className="px-4 py-2">สูงกว่า</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-700">
            <td className="px-4 py-2">ความสามารถในการเก็บ Feature</td>
            <td className="px-4 py-2">ดีมากสำหรับ Feature ชัดเจน</td>
            <td className="px-4 py-2">อ่อนกว่าเมื่อ Feature เบาบาง</td>
          </tr>
          <tr className="border-t border-gray-300 dark:border-gray-700">
            <td className="px-4 py-2">ใช้งานทั่วไป</td>
            <td className="px-4 py-2">การจำแนกภาพ, ตรวจจับวัตถุ</td>
            <td className="px-4 py-2">Global Feature Summary, Model Compression</td>
          </tr>
        </tbody>
      </table>
    </div>

    <p>
      ในงานวิจัยของ LeCun et al. (1998) ซึ่งเป็นหนึ่งในผู้พัฒนา CNN ยุคแรก ๆ ได้มีการใช้ Average Pooling ภายใต้ชื่อ Subsampling
      แต่ต่อมา Max Pooling ได้รับความนิยมมากขึ้นจากผลลัพธ์ในงาน ImageNet Classification (Krizhevsky et al., 2012) ที่ใช้ Max Pooling คู่กับ ReLU เพื่อดึง Feature ที่เด่นชัดมากที่สุด
    </p>

    <p>
      ผลลัพธ์จากการทดลองของ Google Brain พบว่าในงานบางประเภท เช่น Image Super Resolution หรือ Denoising การใช้ Average Pooling
      สามารถช่วยลดความผิดเพี้ยนจาก Noise ได้ดี เนื่องจากการเฉลี่ยจะลดความรุนแรงของค่าผิดปกติในข้อมูล
    </p>

    <p>
      ในทางกลับกัน Max Pooling เหมาะกับงานที่เน้นความสามารถในการจดจำ Feature ที่เด่นและจำเพาะ เช่นการตรวจจับใบหน้าหรือการวิเคราะห์วัตถุในฉากที่ซับซ้อน
    </p>
 

    <p>
      จากมุมมองการวิเคราะห์ข้อมูลเชิงสถิติ Max Pooling ทำหน้าที่คล้ายกับการเลือกค่า Outlier ที่โดดเด่น
      ในขณะที่ Average Pooling ทำหน้าที่เป็นการลดค่าความแปรปรวนเพื่อเน้นค่าเฉลี่ยของลักษณะโดยรวม
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          Max Pooling มีประสิทธิภาพสูงในการแยกแยะ Feature ที่มีการแสดงออกเด่น เช่น ขอบภาพ เส้น และ texture ที่ชัดเจน
        </li>
        <li>
          Average Pooling ช่วยให้โมเดลมีความทนทานต่อ Noise โดยลดการเน้น Feature ที่อาจไม่สำคัญหรือผิดปกติ
        </li>
        <li>
          ในเครือข่ายแบบ Modern CNN เช่น ResNet และ MobileNet การใช้ Global Average Pooling แทน Fully Connected Layer ช่วยลดจำนวนพารามิเตอร์ได้หลายล้าน
        </li>
        <li>
          การเลือกใช้ Max หรือ Average Pooling ควรพิจารณาตามบริบทของงาน และลักษณะเฉพาะของข้อมูล เช่น ความซับซ้อนของ Feature และระดับของ Noise
        </li>
      </ul>
    </div>
  </div>
</section>


{/* Section 3 */}
<section id="pooling-config" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. วิธีการตั้งค่า Pooling</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img5} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การตั้งค่า Pooling Layer อย่างถูกต้องมีความสำคัญต่อการสกัด Feature และการลดขนาดของ Tensor
      ที่ไหลผ่านโครงข่าย Convolutional Neural Network (CNN) อย่างมีประสิทธิภาพ
      การตั้งค่าที่พบบ่อย ได้แก่ ขนาดของ Kernel (Filter), ค่า Stride, Padding และประเภทของ Pooling
      โดยทุกพารามิเตอร์เหล่านี้ส่งผลต่อขนาด Output และคุณลักษณะของ Feature Map โดยตรง
    </p>

    <h3 className="text-xl font-semibold">3.1 ขนาดของ Kernel (Filter Size)</h3>
    <p>
      ขนาดของ Kernel เป็นปัจจัยหลักที่กำหนดขอบเขตของข้อมูลที่ Pooling Layer จะพิจารณาในแต่ละขั้นตอน
      ค่าที่นิยมใช้คือ 2×2 หรือ 3×3 โดยมีข้อดีแตกต่างกัน:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li><span className="font-semibold">2×2:</span> ลดขนาดภาพลงครึ่งหนึ่งเมื่อใช้ร่วมกับ Stride = 2, ช่วยลดขนาดข้อมูลอย่างรวดเร็ว</li>
      <li><span className="font-semibold">3×3:</span> มีความละเอียดมากกว่าในการสกัด Feature แต่ลดขนาดได้น้อยกว่าต่อรอบ</li>
    </ul>
    <p>
      การศึกษาใน AlexNet และ VGGNet นิยมใช้ Kernel ขนาด 2×2 และ 3×3 เพื่อรักษา Feature สำคัญและลดขนาดอย่างเป็นระบบ
    </p>

    <h3 className="text-xl font-semibold">3.2 ค่า Stride</h3>
    <p>
      Stride คือจำนวนพิกเซลที่ Kernel เคลื่อนที่ในแต่ละขั้นตอน ค่าปกติคือ 1 หรือ 2
      การเพิ่มค่า Stride จะลดขนาดของ Output ได้รวดเร็วขึ้น:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li><span className="font-semibold">Stride = 1:</span> ลดขนาดน้อยที่สุด แต่คงรายละเอียดได้มาก</li>
      <li><span className="font-semibold">Stride = 2:</span> ลดขนาดได้ครึ่งหนึ่ง เหมาะกับการ Downsample ข้อมูล</li>
    </ul>
    <p>
      ในการออกแบบเครือข่ายแบบลึก เช่น ResNet หรือ MobileNet ค่า Stride ที่เหมาะสมช่วยให้เครือข่ายประมวลผลข้อมูลอย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">3.3 Padding (การขยายขอบ)</h3>
    <p>
      ในบางกรณี เมื่อ Kernel ถูกเลื่อนในภาพที่มีขนาดไม่หารลงตัวด้วย Kernel และ Stride อาจต้องมีการ Padding
      เพื่อป้องกันการสูญเสียข้อมูลขอบภาพ
    </p>
    <ul className="list-disc list-inside ml-6">
      <li><span className="font-semibold">Valid Padding:</span> ไม่มีการเติมขอบ Output มีขนาดเล็กลง</li>
      <li><span className="font-semibold">Same Padding:</span> มีการเติมขอบเพื่อให้ Output มีขนาดเท่ากับ Input</li>
    </ul>
    <p>
      แบบ "Same" padding ถูกใช้ในเครือข่ายที่ต้องการให้ขนาด Feature Map คงที่เพื่อจับ Feature ได้ลึกตลอดโครงข่าย
    </p>

    <h3 className="text-xl font-semibold">3.4 ประเภทของ Pooling</h3>
    <ul className="list-disc list-inside ml-6">
      <li><span className="font-semibold">Max Pooling:</span> เลือกค่ามากที่สุดในแต่ละ Region</li>
      <li><span className="font-semibold">Average Pooling:</span> คำนวณค่าเฉลี่ยของแต่ละ Region</li>
      <li><span className="font-semibold">Global Pooling:</span> คำนวณค่าสูงสุดหรือค่าเฉลี่ยจากทั้ง Feature Map</li>
    </ul>
    <p>
      งานวิจัยจาก DeepMind และ Facebook AI Research พบว่า Max Pooling มีประสิทธิภาพสูงในการจับ Feature ที่เด่นชัด
      ขณะที่ Global Average Pooling ถูกนำมาใช้แทน Fully Connected Layer ใน MobileNet และ EfficientNet
      เพื่อลดจำนวนพารามิเตอร์และเพิ่มความยืดหยุ่นต่อขนาด Input
    </p>

    <h3 className="text-xl font-semibold">3.5 ตัวอย่างการคำนวณ Output Size</h3>
    <p>
      สมมุติให้ Input Image มีขนาด 32×32, ใช้ Max Pooling ด้วย Kernel ขนาด 2×2, Stride = 2, Padding = 0
    </p>
    <p>
      สูตรการคำนวณ:  
      <br />
      Output = floor((Input − Kernel + 2 * Padding) / Stride + 1)
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
      <code>
        Output = floor((32 − 2 + 0) / 2 + 1) = 16
      </code>
    </pre>
    <p>
      Output Feature Map มีขนาด 16×16 ซึ่งเท่ากับลดขนาดลง 75% เมื่อเทียบกับต้นฉบับ
    </p>

    <h3 className="text-xl font-semibold">3.6 แนวทางการตั้งค่าในโมเดลจริง</h3>
    <ul className="list-disc list-inside ml-6">
      <li><span className="font-semibold">VGGNet:</span> ใช้ Max Pooling ขนาด 2×2 Stride = 2 ทุก 2 Conv Layer</li>
      <li><span className="font-semibold">GoogLeNet:</span> ใช้ทั้ง Max และ Average Pooling ใน Inception Module</li>
      <li><span className="font-semibold">ResNet:</span> ใช้ Strided Convolution แทน Pooling และเพิ่ม Global Average Pooling ตอนท้าย</li>
    </ul>


    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          การกำหนดขนาด Kernel และค่า Stride มีผลต่อความสามารถในการสรุป Feature และลดข้อมูลอย่างสมดุล
        </li>
        <li>
          การใช้ Padding แบบ Same เหมาะสำหรับโครงข่ายลึกที่ต้องการให้ขนาด Feature Map คงที่
        </li>
        <li>
          Global Average Pooling ช่วยลดจำนวนพารามิเตอร์ และเพิ่มความสามารถในการ Generalize
        </li>
        <li>
          ในการออกแบบโมเดลแบบ ResNet หรือ MobileNet การตัด Fully Connected Layer และใช้ Global Pooling แทน
          ช่วยลดขนาดโมเดลลงได้มากกว่า 90% โดยไม่ลดประสิทธิภาพ
        </li>
      </ul>
    </div>
  </div>
</section>

<section id="pool-placement" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การวาง Pooling Layer อย่างมีหลักการ</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img6} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การจัดวาง Pooling Layer ในโครงข่าย Convolutional Neural Networks (CNNs) ไม่ใช่เพียงการลดขนาด Tensor
      แต่ยังส่งผลโดยตรงต่อประสิทธิภาพของการเรียนรู้ ความสามารถในการสกัด Feature และความสามารถในการทำ Generalization ของโมเดล
      งานวิจัยจาก Stanford (CS231n) และ Google DeepMind ได้ให้คำแนะนำอย่างชัดเจนเกี่ยวกับตำแหน่งและความถี่ในการวาง Pooling Layer
    </p>

    <p>
      โดยทั่วไป การวาง Pooling Layer มักเกิดขึ้นหลังกลุ่มของ Convolutional Layers ที่มี Activation Function
      เช่น ReLU และ Batch Normalization การจัดเรียงในลักษณะนี้มีเป้าหมายเพื่อลด Spatial Resolution
      หลังจากที่ Feature ที่จำเป็นถูกสกัดออกมาแล้วบางส่วน ซึ่งเป็นแนวทางที่ใช้ในสถาปัตยกรรมยอดนิยม เช่น VGGNet และ ResNet
    </p>

    <h3 className="text-xl font-semibold">หลักการวาง Pooling Layer จาก VGGNet</h3>
    <p>
      VGGNet ซึ่งพัฒนาโดย University of Oxford ได้แสดงให้เห็นว่า การวาง Max Pooling Layer หลังทุก ๆ 2 Convolutional Layers
      โดยใช้ Filter ขนาด 2×2 และ Stride = 2 จะช่วยลดขนาด Feature Map อย่างมีประสิทธิภาพ พร้อมกับคงไว้ซึ่ง Feature ที่จำเป็น
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>ลด Spatial Resolution อย่างเป็นระบบจาก 224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7</li>
      <li>คงไว้ซึ่งลำดับขั้นของการเรียนรู้ Feature แบบ Low → Mid → High level</li>
      <li>เพิ่ม Robustness ต่อการเปลี่ยนตำแหน่งของ Feature</li>
    </ul>

    <h3 className="text-xl font-semibold">ความถี่ในการวาง Pooling Layer</h3>
    <p>
      การวาง Pooling Layer ที่ถี่เกินไป เช่น หลังทุก Convolution Layer อาจส่งผลให้ข้อมูลสูญหายเร็วเกินไป
      ส่งผลให้โมเดลเรียนรู้ Feature ระดับสูงได้ยาก ในทางกลับกัน หากวาง Pooling น้อยเกินไป โมเดลจะมีภาระในการเรียนรู้ Feature ที่มีขนาดใหญ่มาก
      ซึ่งไม่เหมาะสมกับความสามารถของโมเดลในช่วงเริ่มต้นของการฝึก
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <p className="font-semibold">ข้อแนะนำ:</p>
      <ul className="list-disc list-inside ml-4 space-y-1">
        <li>วาง Pooling Layer หลังทุก 2 หรือ 3 Convolutional Layers</li>
        <li>ใช้ขนาดของ Pooling Filter เท่ากับ 2×2 และ Stride = 2 เพื่อการ Downsampling ที่มีประสิทธิภาพ</li>
        <li>หลีกเลี่ยงการใช้ Pooling ซ้อนต่อกันโดยไม่มี Convolution คั่นกลาง</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">Alternative Strategy: Strided Convolution แทน Pooling</h3>
    <p>
      งานวิจัยจาก Google (GoogLeNet) และ DeepMind (AlphaGo Zero) ได้นำเสนอทางเลือกโดยใช้ Strided Convolution แทนการใช้ Pooling
      โดยการเพิ่มค่า Stride ใน Convolutional Layer แทนที่จะใช้ MaxPooling จะช่วยให้โมเดลเรียนรู้ได้แบบ end-to-end โดยไม่ต้องแยกกระบวนการลงขนาดออกจากการเรียนรู้ Feature
    </p>

    <p>
      จุดเด่นของแนวทางนี้คือ:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>โมเดลสามารถเรียนรู้วิธีการ Downsampling โดยอัตโนมัติผ่าน Weight ของ Convolution</li>
      <li>ลดความซับซ้อนของโครงสร้างโมเดล</li>
      <li>เหมาะสำหรับระบบที่ต้องการ train จากข้อมูลขนาดใหญ่แบบไม่ใช้ handcrafted component</li>
    </ul>

    <p>
      อย่างไรก็ตาม การใช้ Strided Convolution ต้องมีการปรับค่า Padding อย่างรอบคอบ และอาจต้องใช้ BatchNorm เพื่อควบคุมค่า Distribution ภายในแต่ละเลเยอร์
    </p>

    <h3 className="text-xl font-semibold">Global Average Pooling ในเลเยอร์สุดท้าย</h3>
    <p>
      สถาปัตยกรรมยุคใหม่ เช่น Inception และ MobileNet ใช้ Global Average Pooling (GAP) แทน Fully Connected Layer
      ในเลเยอร์สุดท้ายของ Feature Extractor เพื่อรวม Feature Map เป็นเวกเตอร์เดียว โดยไม่ต้อง Flatten และใช้พารามิเตอร์จำนวนมาก
    </p>
    <p>
      ข้อดีของ GAP:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>ลดจำนวนพารามิเตอร์ในโมเดล ทำให้โมเดลเล็กลงและเร็วขึ้น</li>
      <li>ลดความเสี่ยงของ Overfitting</li>
      <li>สามารถนำไปใช้ใน Edge Devices หรือโมเดลขนาดเล็กได้ง่าย</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างการวาง Pooling Layer</h3>
    <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-sm">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-800">
            <th className="p-2">ลำดับ</th>
            <th className="p-2">เลเยอร์</th>
            <th className="p-2">รายละเอียด</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="p-2">1</td>
            <td className="p-2">Conv2D + ReLU</td>
            <td className="p-2">3×3 filter, padding=1, stride=1</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-800">
            <td className="p-2">2</td>
            <td className="p-2">Conv2D + ReLU</td>
            <td className="p-2">3×3 filter, padding=1, stride=1</td>
          </tr>
          <tr>
            <td className="p-2">3</td>
            <td className="p-2">MaxPooling</td>
            <td className="p-2">2×2 filter, stride=2</td>
          </tr>
          <tr className="bg-gray-50 dark:bg-gray-800">
            <td className="p-2">4</td>
            <td className="p-2">Conv2D + ReLU</td>
            <td className="p-2">3×3 filter, padding=1, stride=1</td>
          </tr>
          <tr>
            <td className="p-2">5</td>
            <td className="p-2">Global Average Pooling</td>
            <td className="p-2">Reduce to 1×1×Channels</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          การวาง Pooling Layer อย่างมีหลักการช่วยลดขนาด Feature Map อย่างสมดุล โดยไม่ลดทอนความสามารถของโมเดลในการเรียนรู้ Feature
        </li>
        <li>
          การใช้ Global Average Pooling แทน Fully Connected Layer สามารถลดขนาดโมเดลลงอย่างมีนัยสำคัญ โดยไม่สูญเสีย Accuracy
        </li>
        <li>
          ในงานที่ต้องการประสิทธิภาพสูง เช่น Embedded AI หรือ Edge Deployment ควรเลือก Pooling ที่มีขนาดเล็กและวางห่างกัน
        </li>
        <li>
          แนวทางของ Strided Convolution ช่วยให้เกิด End-to-End Feature Learning โดยไม่ต้องแยกขั้นตอน Downsampling ออกมา
        </li>
      </ul>
    </div>
  </div>
</section>

<section id="stride-effects" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Stride: แนวคิดและผลลัพธ์</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img7} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Stride คือค่าที่กำหนดระยะการเลื่อนของ Filter หรือ Kernel ระหว่างการทำ Convolution หรือ Pooling
      หากค่า Stride เท่ากับ 1 หมายถึงการเลื่อนหนึ่งพิกเซลในแต่ละครั้ง ซึ่งให้ผลลัพธ์ Feature Map ที่มีความละเอียดสูง
      ในขณะที่ค่า Stride ที่มากกว่านั้น จะลดความละเอียดของ Feature Map ลงโดยการข้ามพิกเซลบางตำแหน่งออกจากการคำนวณ
    </p>

    <p>
      จากหลักการของ Signal Processing และงานวิจัยในสาขา Computer Vision เช่น LeNet-5, AlexNet และ ResNet
      ค่า Stride ที่เลือกจะมีผลโดยตรงต่อขนาดของ Output Tensor และลักษณะของข้อมูลที่ถูกสกัดออกมา
      การกำหนดค่า Stride เป็นหนึ่งในเทคนิคที่สำคัญในการทำ Downsampling ภายในเครือข่าย CNN โดยไม่ต้องใช้ Pooling Layer
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">ผลกระทบต่อ Feature Map</h3>
    <p>
      การกำหนดค่า Stride ส่งผลต่อขนาดของ Feature Map ตามสูตร:
    </p>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-auto">
      <code className="block whitespace-pre-wrap">
        OutputSize = ⌊(InputSize - FilterSize + 2 × Padding) / Stride⌋ + 1
      </code>
    </div>
    <p>
      ตัวอย่าง: หากมีภาพขนาด 32×32 พิกเซล ใช้ Filter ขนาด 3×3 โดยไม่มี Padding และค่า Stride เท่ากับ 2
      จะได้ขนาด Feature Map เท่ากับ ⌊(32 - 3)/2⌋ + 1 = 15
    </p>

    <p>
      การลดความละเอียดของ Feature Map อย่างมีนัยสำคัญจะช่วยให้โมเดลมีขนาดเล็กลง
      แต่ในขณะเดียวกันก็เสี่ยงต่อการสูญเสียรายละเอียดของข้อมูล ซึ่งอาจส่งผลต่อความสามารถในการจดจำ Feature ที่ซับซ้อน
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">Stride เทียบกับ Pooling</h3>
    <p>
      งานวิจัยจาก Google Brain (2017) แสดงให้เห็นว่า Strided Convolution สามารถแทนที่ Pooling ได้ในบางสถาปัตยกรรม
      โดยเฉพาะในโมเดลอย่าง All-CNN และ ResNet ที่ไม่มีการใช้ Pooling Layer แบบดั้งเดิม
      Stride ใน Convolution Layer จะควบคุมขนาด Output โดยตรง ทำให้ลดความซับซ้อนของโครงสร้างเครือข่าย
    </p>

    <p>
      จุดเด่นของการใช้ Stride แทน Pooling คือการเรียนรู้เชิงปรับตัวผ่าน Parameter ใน Filter
      ต่างจาก Pooling ที่มีพฤติกรรมตายตัว เช่น การหาค่าสูงสุดหรือค่าเฉลี่ยภายใน Window
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">ผลต่อการไหลของกราดิเอนต์</h3>
    <p>
      เมื่อใช้ Stride ที่มากขึ้น Gradient Flow ระหว่าง Backpropagation จะลดลงตามความละเอียดของ Feature Map
      ซึ่งอาจทำให้การเรียนรู้ลักษณะบางอย่างลำบากขึ้น
      เพื่อบรรเทาปัญหานี้ สถาปัตยกรรมสมัยใหม่มักใช้ Residual Connections หรือ Dense Skip Connections
      เพื่อช่วยให้ Gradient ไหลกลับได้อย่างราบรื่น
    </p>

    <p>
      งานวิจัยจาก University of Oxford (2016) ที่พัฒนาโมเดล ResNet ระบุว่า การลด Resolution อย่างรวดเร็วเกินไป
      โดยไม่มีเส้นทางลัดสำหรับ Gradient ส่งผลให้โมเดลสูญเสียศักยภาพในการเรียนรู้ Feature ลึก
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2 text-center">ตัวอย่างเปรียบเทียบภาพ</h3>
    <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img8} />
          </div>
    <p className="text-center text-sm italic">
      ซ้าย: Convolution ด้วย Stride = 1 (รายละเอียดครบ) | ขวา: Stride = 2 (ลดขนาด แต่สูญเสียรายละเอียดบางส่วน)
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">แนวทางการเลือกค่า Stride</h3>
    <ul className="list-disc list-inside ml-6">
      <li>Stride = 1: ใช้ในเลเยอร์ต้น ๆ เพื่อคงรายละเอียดของภาพไว้ให้มากที่สุด</li>
      <li>Stride = 2: ใช้สำหรับ Downsampling แทน Pooling หรือควบคู่กับ Residual Block</li>
      <li>Stride มากกว่า 2: ใช้ในกรณีพิเศษที่ต้องการลดขนาดข้อมูลอย่างรุนแรง เช่น ใน Global Feature Extraction</li>
    </ul>

    <p>
      จากการเปรียบเทียบการใช้งานจริงในสถาปัตยกรรมต่าง ๆ เช่น VGG16, Inception v3 และ EfficientNet
      พบว่าการกำหนดค่า Stride อย่างมีประสิทธิภาพสามารถลดการใช้หน่วยความจำได้กว่า 60%
      โดยยังรักษาความแม่นยำของโมเดลไว้ได้เกือบเท่าเดิม
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Stride = 2 ถือเป็นค่ามาตรฐานสำหรับ Downsampling ที่สมดุลระหว่างขนาดและคุณภาพ</li>
        <li>การใช้ Strided Convolution แทน Pooling เพิ่มความยืดหยุ่นในการเรียนรู้ผ่านการปรับค่า Parameter</li>
        <li>Stride ที่มากเกินไปอาจลดการแยกแยะ Feature ที่ละเอียด ทำให้โมเดลขาด Sensitivity</li>
        <li>โมเดลอย่าง DenseNet ใช้ Stride ควบคู่กับ Transition Layer เพื่อควบคุมความลึกและความกว้างของ Feature Map</li>
      </ul>
    </div>
  </div>
</section>


<section id="compare-stride-pool" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Comparative Analysis: Pool vs Strided Convolution</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img9} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      ในการออกแบบเครือข่ายประสาทเทียมเชิงลึกที่ประมวลผลข้อมูลเชิงภาพหรือเชิงพื้นที่ มีสองแนวทางที่ใช้สำหรับลดขนาดของข้อมูล
      ได้แก่การใช้ Pooling Layer และ Strided Convolution ทั้งสองวิธีมีเป้าหมายเดียวกันคือการลดความละเอียดเชิงพื้นที่ของ Feature Maps
      แต่มีความแตกต่างในด้านโครงสร้าง คณิตศาสตร์ และผลลัพธ์เชิงลึกต่อการเรียนรู้ของโมเดล
    </p>

    <p>
      การใช้ Pooling (เช่น Max Pooling หรือ Average Pooling) จะลดขนาดของข้อมูลผ่านการเลือกค่าภายใน Region ของข้อมูลโดยตรง
      โดยไม่ใช้พารามิเตอร์เพิ่มเติม ซึ่งช่วยให้โมเดลมีขนาดเล็กลงและลดความซับซ้อน
      ในทางกลับกัน Strided Convolution ใช้ Convolution Kernel พร้อม Stride มากกว่า 1 เพื่อลดขนาดของ Feature Map โดยตรงระหว่างการคำนวณ
      โดยที่ Weight ของ Kernel จะยังคงถูกเรียนรู้ในระหว่างกระบวนการฝึก
    </p>


    <h3 className="text-xl font-semibold mt-8 mb-2">โครงสร้างเชิงคณิตศาสตร์</h3>
    <p>
      ในเชิงคณิตศาสตร์ Pooling ทำงานบน Local Patch โดยการคำนวณค่าทางสถิติเช่น max หรือ mean โดยไม่มีการเรียนรู้พารามิเตอร์ใด
      ขณะที่ Strided Convolution เป็นการใช้ Kernel ที่มีพารามิเตอร์แบบเรียนรู้ได้ทำการ Convolve แบบเว้นระยะ
      ทำให้สามารถเรียนรู้ Transformation ของข้อมูลได้มากกว่า
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">ข้อดีของ Pooling</h3>
    <ul className="list-disc list-inside ml-6">
      <li>ไม่มีพารามิเตอร์เพิ่มเติม ทำให้โมเดลไม่เพิ่มภาระทางคำนวณ</li>
      <li>เพิ่ม Invariance ต่อการเปลี่ยนตำแหน่งของวัตถุ</li>
      <li>ช่วยลด Overfitting โดยไม่จำเป็นต้องเรียนรู้ Feature ที่ละเอียดเกินไป</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-2">ข้อดีของ Strided Convolution</h3>
    <ul className="list-disc list-inside ml-6">
      <li>สามารถเรียนรู้ Feature ที่มีความซับซ้อนสูงได้โดยตรงผ่าน Kernel</li>
      <li>ลดจำนวนเลเยอร์ได้โดยไม่ต้องเพิ่ม Block Pooling เพิ่มเติม</li>
      <li>ประหยัดเวลาและเพิ่มความเร็วในการทำ inference โดยการลดจำนวน layer</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-2">ข้อจำกัดของแต่ละแนวทาง</h3>
    <p>
      จากการวิเคราะห์ของงานวิจัยโดย Springenberg et al. (2015) ในหัวข้อ “Striving for Simplicity: The All Convolutional Net”
      พบว่าโมเดลที่ใช้ Strided Convolution แทน Pooling ในทุกชั้นสามารถให้ผลลัพธ์เทียบเท่าหรือดีกว่าในหลาย Dataset เช่น CIFAR-10 และ ImageNet
      อย่างไรก็ตาม Pooling ยังคงมีบทบาทสำคัญในสถานการณ์ที่ต้องการความคงที่ของ Feature หรือ Resilience ต่อ Noise
    </p>

    <p>
      นอกจากนี้ งานวิจัยของ Facebook AI Research (He et al., 2016) ในการพัฒนา ResNet
      แสดงให้เห็นว่า Pooling เมื่อใช้อย่างเหมาะสมสามารถช่วยรักษาความเสถียรของโมเดลและช่วยในการเรียนรู้ Feature ที่ generalizable
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          Pooling เป็นตัวกรองแบบไม่เรียนรู้ (Non-learned) ที่ช่วยลดขนาดเชิงพื้นที่และเพิ่ม Robustness โดยเฉพาะในงานที่มีการแปรเปลี่ยนตำแหน่งสูง เช่น Object Detection
        </li>
        <li>
          Strided Convolution มีข้อได้เปรียบในงานที่ต้องการ Feature Transformation ลึก เช่น Semantic Segmentation หรือ Generative Modeling
        </li>
        <li>
          ในงาน Vision ที่ต้องการความละเอียดสูง เช่น Super-Resolution หรือ Medical Imaging, การไม่ใช้ Pooling อาจให้ความละเอียดและรายละเอียดดีกว่า
        </li>
        <li>
          การออกแบบโมเดลรุ่นใหม่ เช่น EfficientNet และ MobileNet ใช้กลยุทธ์ผสมผสาน โดยเลือกใช้ Strided Convolution คู่กับ Global Pooling เพื่อเพิ่มทั้งประสิทธิภาพและความแม่นยำ
        </li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-2">ตัวอย่างสถาปัตยกรรมที่ใช้แต่ละแนวทาง</h3>
    <table className="table-auto w-full border-collapse border border-gray-300 dark:border-gray-600">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">Architecture</th>
          <th className="border px-4 py-2">Pooling Strategy</th>
          <th className="border px-4 py-2">Stride Usage</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">AlexNet</td>
          <td className="border px-4 py-2">Max Pooling 2×2</td>
          <td className="border px-4 py-2">Stride = 1</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">ResNet</td>
          <td className="border px-4 py-2">Mix: Pool + Stride</td>
          <td className="border px-4 py-2">Stride = 2 in early layers</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">All-CNN</td>
          <td className="border px-4 py-2">None</td>
          <td className="border px-4 py-2">Stride = 2</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">MobileNetV2</td>
          <td className="border px-4 py-2">Global Avg Pool</td>
          <td className="border px-4 py-2">Depthwise Strided Conv</td>
        </tr>
      </tbody>
    </table>

    <p>
      โดยสรุป การเลือกใช้ Pooling หรือ Strided Convolution ควรพิจารณาจากความลึกของเครือข่าย ลักษณะของข้อมูล และความต้องการเชิง computational efficiency
      งานวิจัยชั้นนำหลายฉบับเน้นย้ำถึงความยืดหยุ่นในการเลือกใช้ทั้งสองอย่างร่วมกัน โดยไม่มีวิธีใดที่ "ดีที่สุดเสมอไป"
    </p>
  </div>
</section>


{/* Section 7 */}
<section id="adaptive-pooling" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Adaptive Pooling</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img10} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Adaptive Pooling คือเทคนิคที่ถูกพัฒนาขึ้นเพื่อแก้ไขข้อจำกัดของ Pooling แบบคงขนาด (Fixed-size Pooling) โดยเฉพาะในสถานการณ์ที่ต้องการให้เอาต์พุตของเลเยอร์มีขนาดที่แน่นอน ไม่ว่าจะมีอินพุตขนาดใดก็ตาม เทคนิคนี้ได้รับความนิยมในงานวิจัยจาก MIT และในสถาปัตยกรรมการเรียนรู้ภาพสมัยใหม่ เช่น ResNet และ EfficientNet
    </p>

    <p>
      แนวคิดหลักของ Adaptive Pooling คือ การคำนวณขนาดของ Kernel และ Stride ให้เหมาะสมโดยอัตโนมัติ เพื่อให้ได้เอาต์พุตที่มีขนาดที่กำหนดไว้ล่วงหน้า โดยไม่ต้องระบุขนาดของ Kernel หรือ Stride เอง
    </p>

    <p>
      ตัวอย่างเช่น หากอินพุตมีขนาด 32×32 และต้องการให้เอาต์พุตมีขนาด 7×7 Adaptive Pooling จะคำนวณขนาดของ Kernel และ Stride โดยอัตโนมัติให้ตรงตามเป้าหมาย โดยไม่ต้อง hard-code ค่า
    </p>

    <p>
      ใน PyTorch การใช้งาน Adaptive Pooling สามารถทำได้ผ่านคำสั่ง เช่น
      <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">nn.AdaptiveAvgPool2d((7, 7))</code>
      ซึ่งจะทำให้ไม่ว่าขนาดอินพุตจะเป็นเท่าใด ก็ตาม เอาต์พุตจะถูกปรับให้อยู่ในขนาด 7×7 เสมอ
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">ข้อดีของ Adaptive Pooling</h3>
    <ul className="list-disc list-inside ml-6">
      <li>ความยืดหยุ่นในการออกแบบโมเดล โดยไม่ขึ้นกับขนาดอินพุต</li>
      <li>เหมาะสำหรับงานที่ข้อมูลมีขนาดไม่คงที่ เช่น วิดีโอ หรือข้อความที่แปลงเป็น Feature Map</li>
      <li>ช่วยลดการ Overfitting โดยลดขนาดเอาต์พุตให้เหมาะสมก่อนป้อนเข้าสู่ Fully Connected Layer</li>
      <li>เพิ่มความสามารถในการนำโมเดลไปใช้ใน real-world ที่ขนาดอินพุตเปลี่ยนแปลงได้</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-2">การทำงานของ Adaptive Pooling</h3>
    <p>
      การคำนวณ Kernel และ Stride จะถูกกำหนดโดยสมการ:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`stride = floor(input_size / output_size)
kernel_size = input_size - (output_size - 1) * stride`}
    </pre>
    <p>
      โดย input_size คือขนาดของอินพุตในแกนที่ต้องการ (เช่น ความสูงหรือความกว้าง) ส่วน output_size คือขนาดที่ต้องการให้เป็นผลลัพธ์
    </p>

    <p>
      ตัวอย่างการใช้ Adaptive Pooling ในการแปลงภาพ 224×224 เป็น 7×7 ก่อนเข้า Fully Connected Layer:
    </p>
    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`import torch.nn as nn

model = nn.Sequential(
  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
  nn.ReLU(),
  nn.AdaptiveAvgPool2d((7, 7)),
  nn.Flatten(),
  nn.Linear(64 * 7 * 7, 1000)
)`}
    </pre>

    <p>
      Adaptive Pooling แบ่งออกเป็น 3 ประเภทที่ใช้บ่อย:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li><span className="font-semibold">Adaptive Max Pooling:</span> เก็บค่าสูงสุดในแต่ละช่องของ Grid ที่คำนวณอัตโนมัติ</li>
      <li><span className="font-semibold">Adaptive Average Pooling:</span> คำนวณค่าเฉลี่ยในแต่ละช่อง</li>
      <li><span className="font-semibold">Global Average Pooling:</span> รูปแบบพิเศษของ Adaptive Avg Pooling ที่ผลลัพธ์มีขนาด 1×1</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-2">กรณีศึกษา: การใช้งาน Adaptive Pooling ใน ResNet</h3>
    <p>
      ใน ResNet แต่ละ Block จะลด Spatial Size ผ่าน Convolution ที่มี Stride = 2 แต่ก่อนเข้าสู่ Fully Connected Layer
      จะมี AdaptiveAvgPool2d((1, 1)) เพื่อให้ขนาด Feature Map สุดท้ายคงที่ ไม่ว่าขนาดของอินพุตจะเปลี่ยนไปแค่ไหน
    </p>

    <p>
      งานวิจัยของ Facebook AI (FAIR) พบว่า Adaptive Pooling ช่วยให้สถาปัตยกรรมสามารถ Generalize ได้ดีขึ้นเมื่อนำไปใช้กับภาพที่มี Resolution ต่างกัน
      ซึ่งเป็นสิ่งจำเป็นสำหรับงาน real-world ที่ไม่มีการกำหนดขนาดอินพุตแบบแน่นอน
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">เปรียบเทียบกับ Fixed-size Pooling</h3>
    <table className="w-full text-left border border-gray-400 dark:border-gray-600 rounded-lg overflow-hidden text-sm md:text-base">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="p-3 border border-gray-400 dark:border-gray-600">Aspect</th>
          <th className="p-3 border border-gray-400 dark:border-gray-600">Fixed-size Pooling</th>
          <th className="p-3 border border-gray-400 dark:border-gray-600">Adaptive Pooling</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="p-3 border border-gray-400 dark:border-gray-600">ความยืดหยุ่น</td>
          <td className="p-3 border border-gray-400 dark:border-gray-600">ต่ำ</td>
          <td className="p-3 border border-gray-400 dark:border-gray-600">สูง</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 dark:border-gray-600">การกำหนดขนาดเอาต์พุต</td>
          <td className="p-3 border border-gray-400 dark:border-gray-600">ต้องคำนวณ Kernel และ Stride เอง</td>
          <td className="p-3 border border-gray-400 dark:border-gray-600">ปรับอัตโนมัติ</td>
        </tr>
        <tr>
          <td className="p-3 border border-gray-400 dark:border-gray-600">เหมาะกับภาพขนาดไม่คงที่</td>
          <td className="p-3 border border-gray-400 dark:border-gray-600">ไม่เหมาะสม</td>
          <td className="p-3 border border-gray-400 dark:border-gray-600">เหมาะสม</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg mt-6">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          Adaptive Pooling ช่วยให้โมเดลสามารถจัดการกับข้อมูลขนาดไม่คงที่ได้ดี โดยไม่ต้องเขียนเงื่อนไขแยกแต่ละขนาด
        </li>
        <li>
          การใช้ Global Average Pooling แทน Fully Connected Layer ช่วยลดจำนวนพารามิเตอร์ในโมเดลได้มากกว่า 90%
        </li>
        <li>
          สถาปัตยกรรมเช่น EfficientNet ใช้ Adaptive Pooling เพื่อรองรับการประมวลผลภาพจากอุปกรณ์ที่มีความละเอียดต่างกัน
        </li>
        <li>
          Adaptive Pooling มีผลโดยตรงต่อการ deploy โมเดลบน edge devices ที่ขนาดของข้อมูลอินพุตเปลี่ยนแปลงตลอดเวลา
        </li>
      </ul>
    </div>

  </div>
</section>


{/* Section 8 */}
<section id="pool-alternatives" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Pooling Alternatives & Modern Trends</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img11} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      แม้ว่า Max Pooling และ Average Pooling จะเป็นวิธีดั้งเดิมที่ใช้แพร่หลายในการลดขนาดเชิงพื้นที่ในโครงข่าย Convolutional Neural Networks
      แต่ในช่วงหลายปีที่ผ่านมาได้มีการพัฒนาเทคนิคทางเลือกที่มีประสิทธิภาพและความยืดหยุ่นมากขึ้น
      โดยมีเป้าหมายเพื่อรักษาข้อมูลที่สำคัญ ลดการสูญเสียเชิงโครงสร้าง และรองรับงานที่ต้องการความละเอียดเชิงพื้นที่สูง
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">1. Strided Convolution</h3>
    <p>
      Strided Convolution เป็นทางเลือกแรกที่แทนที่การใช้ Pooling ด้วยการเพิ่มค่า Stride ใน Convolution Layer
      วิธีนี้ช่วยลด Spatial Resolution ในขณะที่ยังคงเรียนรู้ Feature ผ่าน Kernel ได้โดยตรง
      งานวิจัยจาก He et al. (2015, ResNet) ชี้ให้เห็นว่า Strided Convolution มีประสิทธิภาพใกล้เคียงกับ Pooling
      และสามารถใช้เพื่อควบคุมความลึกและความละเอียดของ Feature Map ได้อย่างยืดหยุ่น
    </p>

    <p className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <code className="block"># PyTorch Example:</code>
      <code className="block">nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)</code>
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">2. Dilated Convolution (Atrous Convolution)</h3>
    <p>
      เทคนิคนี้ใช้ในงาน segmentation และ recognition ที่ต้องการขยาย receptive field โดยไม่ลด resolution
      เช่นใน DeepLab จาก Google Brain
      Dilated Convolution ช่วยให้โมเดลเห็นข้อมูลในบริบทกว้างขึ้นโดยไม่ต้องเพิ่มพารามิเตอร์หรือใช้ Pooling
    </p>

    <p>
      การศึกษาโดย Yu & Koltun (2016, Multi-Scale Context Aggregation) แสดงให้เห็นว่า Dilated Convolution
      ให้ผลลัพธ์ที่ดีกว่า Pooling ในงาน semantic segmentation และสามารถขยาย receptive field ได้แบบ exponential
      โดยไม่มีการลดจำนวนพิกเซล
    </p>


    <h3 className="text-xl font-semibold mt-8 mb-2">3. Adaptive Pooling</h3>
    <p>
      Adaptive Pooling ช่วยปรับ Output ให้มีขนาดที่กำหนดไว้ล่วงหน้าโดยไม่สนใจ Input size
      วิธีนี้เหมาะกับงานที่ต้องการ Input รูปภาพหลายขนาด เช่นใน EfficientNet หรือ SqueezeNet
      Adaptive Pooling คงความสามารถของการเรียนรู้ Feature แต่สามารถยืดหยุ่นกับขนาดภาพได้
    </p>

    <p className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <code className="block"># PyTorch Example:</code>
      <code className="block">nn.AdaptiveAvgPool2d((1, 1))</code>
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">4. Global Average Pooling (GAP)</h3>
    <p>
      GAP ถูกใช้แพร่หลายในโครงข่ายสมัยใหม่ เช่น Inception, MobileNet และ ResNet
      โดยจะคำนวณค่าเฉลี่ยของ Feature Map ทั้งหมดในแต่ละ Channel เหลือเพียง 1 ค่า
      ซึ่งลดจำนวนพารามิเตอร์ลงได้มากกว่าการใช้ Fully Connected Layer
    </p>

    <p>
      งานวิจัยโดย Lin et al. (2013, Network in Network) เสนอ GAP เป็นทางเลือกที่มีความ regularization สูงและป้องกัน Overfitting
      ได้ดีในเครือข่ายที่ลึก
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">5. Attention Pooling</h3>
    <p>
      Attention-based Pooling ใช้กลไก Attention เพื่อเรียนรู้ว่าตำแหน่งใดของ Feature Map มีความสำคัญ
      เช่นใน SE-Net (Hu et al., 2018) และ Transformer-based vision models
      เทคนิคนี้ช่วยให้โมเดลโฟกัสเฉพาะ Feature ที่มีผลต่อการจำแนก โดยไม่ลด Spatial Context แบบ Pooling ทั่วไป
    </p>

    <p>
      ในระดับคณิตศาสตร์ Attention Pooling ทำงานโดยใช้ Weight Map ที่เรียนรู้ได้มาเป็นตัวถ่วงค่าเฉลี่ยของแต่ละ Spatial Unit
      และมีความสามารถในการเรียนรู้ Non-linear Spatial Importance ได้
    </p>


    <h3 className="text-xl font-semibold mt-8 mb-2">6. Learnable Pooling (LPool, Gated Pooling)</h3>
    <p>
      Learnable Pooling เป็นแนวคิดใหม่ที่เปลี่ยน Pooling ให้กลายเป็น Layer ที่สามารถเรียนรู้ได้
      เช่น LPool (Gulcehre et al.) และ Gated Pooling ซึ่งเรียนรู้วิธีรวมค่าจากแต่ละ Region
      ด้วยน้ำหนักที่ปรับได้ตลอดการฝึก
    </p>

    <p>
      ในบางกรณี Learnable Pooling สามารถเพิ่ม Accuracy ได้เหนือกว่า Max Pooling ในงาน classification และ segmentation
      โดยเฉพาะเมื่อข้อมูลมีความซับซ้อนสูง
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">7. Mixed Pooling & Stochastic Pooling</h3>
    <p>
      Mixed Pooling ใช้การสุ่มระหว่าง Max และ Average Pooling ด้วย Probabilistic Coefficient
      ช่วยลดความเอนเอียงของการเลือก Feature ในแต่ละ Region
    </p>
    <p>
      Stochastic Pooling (Zeiler et al., 2013) ใช้วิธีสุ่มเลือกค่าภายในแต่ละ Pooling Window โดยอิงตามค่าความน่าจะเป็น
      ซึ่งมีผลให้ Feature ที่ไม่ใช่ Max หรือ Mean ได้รับโอกาสมีส่วนร่วมในการเรียนรู้
    </p>

    <p className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <strong>Insight Box</strong>
      <ul className="list-disc list-inside space-y-2">
        <li>Strided Convolution ช่วยให้โมเดลเรียนรู้ Feature พร้อมกับลดขนาด Spatial โดยไม่ต้องแยก Pooling Layer ออกต่างหาก</li>
        <li>Dilated Convolution ขยาย receptive field ได้โดยไม่ลด resolution จึงเหมาะกับงาน segmentation</li>
        <li>Global Average Pooling ช่วยลดพารามิเตอร์อย่างมาก และเพิ่มความยืดหยุ่นในโมเดลสำหรับ deployment บนอุปกรณ์จำกัดพลังงาน</li>
        <li>Attention Pooling และ Learnable Pooling กำลังเป็นเทรนด์ใหม่ใน Vision Transformer และ Lightweight CNNs</li>
      </ul>
    </p>
  </div>
</section>


<section id="real-architecture" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Insight จากสถาปัตยกรรมจริง</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img12} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      ในการพัฒนาโมเดล Convolutional Neural Networks (CNNs) ชั้นนำ การเลือกใช้ Pooling และ Stride มีบทบาทสำคัญต่อความลึกของเครือข่าย,
      การลดขนาดข้อมูล, และการเพิ่มความสามารถในการเรียนรู้ Feature ที่ซับซ้อน โดยสถาปัตยกรรมชื่อดังหลายตัวได้นำเทคนิคเหล่านี้มาใช้อย่างชาญฉลาด
      เพื่อสร้างโมเดลที่ทั้งมีประสิทธิภาพและมีความสามารถในการ Generalize ได้ดี
    </p>

    <h3 className="text-xl font-semibold">AlexNet (2012)</h3>
    <p>
      AlexNet คือโมเดล CNN ที่เริ่มต้นยุคของ Deep Learning ในงานด้าน Computer Vision ด้วยการชนะการแข่งขัน ImageNet ปี 2012 อย่างท่วมท้น
      โดยโมเดลนี้ใช้ Max Pooling ขนาด 3×3 ร่วมกับ Stride = 2 หลังจากทุกชุดของ Convolutional Layers เพื่อ:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>ลดขนาด Spatial ของ Feature Map ลงครึ่งหนึ่งในแต่ละขั้น</li>
      <li>เพิ่มความทนทานต่อ Translation</li>
      <li>ลดภาระการคำนวณของ Fully Connected Layers ที่อยู่ท้ายเครือข่าย</li>
    </ul>

    <h3 className="text-xl font-semibold">VGGNet (2014)</h3>
    <p>
      งานวิจัยจาก University of Oxford โดย Simonyan และ Zisserman แสดงให้เห็นว่า
      การใช้ Convolution ขนาดเล็ก (3×3) หลายครั้งติดกัน ก่อนทำ Max Pooling ขนาด 2×2 พร้อม Stride = 2
      ให้ผลลัพธ์ที่ดีกว่าการใช้ Convolution ขนาดใหญ่ทีเดียว เช่น 5×5 หรือ 7×7
    </p>
    <p>
      โดยทุก 2–3 Convolution Layers จะตามด้วย Pooling Layer เพื่อค่อย ๆ ลดขนาด Spatial ลง
      ส่งผลให้โมเดลลึกขึ้นแต่ยังสามารถคงประสิทธิภาพการเรียนรู้ Feature ที่มีความละเอียดสูงไว้ได้
    </p>

    <h3 className="text-xl font-semibold">GoogLeNet (Inception, 2014)</h3>
    <p>
      GoogLeNet จากทีม Google ใช้แนวคิด “Network in Network” โดยมี Inception Modules ที่ประกอบด้วย Convolution หลายขนาด
      และการทำ Pooling ภายใน Module เอง ทั้ง Max และ Average Pooling
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>ใช้ Global Average Pooling แทน Fully Connected Layer เพื่อลดจำนวนพารามิเตอร์</li>
      <li>เพิ่ม Regularization Effect โดยเฉพาะในช่วงท้ายของโมเดล</li>
    </ul>


    <h3 className="text-xl font-semibold">ResNet (2015)</h3>
    <p>
      โมเดล Residual Network จาก Microsoft Research ใช้ Strided Convolution แทน Pooling ในบางส่วนของเครือข่าย
      โดยให้ Convolution Layer มีค่า Stride = 2 เพื่อให้การลดขนาดข้อมูลเกิดขึ้นไปพร้อมกับการเรียนรู้ Feature
      และหลีกเลี่ยงการสูญเสีย Gradient ผ่าน Residual Connection
    </p>
    <p>
      Insight สำคัญจาก ResNet คือ Pooling อาจไม่จำเป็นในบางกรณีหากสามารถออกแบบ Convolution ให้มี Stride ที่เหมาะสม
      และใช้ Batch Normalization ร่วมด้วยเพื่อคงเสถียรภาพของการฝึก
    </p>

    <h3 className="text-xl font-semibold">DenseNet (2017)</h3>
    <p>
      DenseNet จากทีมที่รวมมหาวิทยาลัยหลายแห่ง เช่น Cornell และ Tsinghua ใช้แนวคิดการเชื่อมต่อเลเยอร์แบบ Dense Block
      โดยใช้ Transition Layer ที่ประกอบด้วย BatchNorm → 1×1 Convolution → 2×2 Average Pooling (Stride = 2)
      เพื่อควบคุมขนาดของ Feature Map ในขณะที่ยังคงการไหลของข้อมูลจากเลเยอร์ก่อนหน้า
    </p>
    <p>
      การใช้ Average Pooling ในที่นี้ไม่เพียงแต่ช่วยลดขนาดเท่านั้น แต่ยังทำหน้าที่เป็นตัวช่วย smoothing
      ที่ลดความซับซ้อนของ Feature ก่อนส่งผ่านไปยัง Block ถัดไป
    </p>

    <h3 className="text-xl font-semibold">MobileNet (2017-2021)</h3>
    <p>
      MobileNet Series ซึ่งออกแบบมาสำหรับอุปกรณ์ขนาดเล็ก ใช้ Global Average Pooling เป็นมาตรฐาน
      และหลีกเลี่ยงการใช้ Fully Connected Layers ที่มีจำนวนพารามิเตอร์มาก
      โดยผลการวิจัยพบว่าการแทนที่ Fully Connected ด้วย Global Pooling สามารถลดพลังงานลงได้มากกว่า 70%
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>เพิ่มความเร็วในการ inference</li>
      <li>ลดขนาดโมเดลจากร้อย MB เหลือเพียงไม่กี่ MB</li>
    </ul>

    <h3 className="text-xl font-semibold">EfficientNet (2019)</h3>
    <p>
      EfficientNet ใช้เทคนิค Compound Scaling เพื่อปรับขนาดความลึก, ความกว้าง และขนาดอินพุต อย่างสมดุล
      โดย Pooling ถูกใช้เพื่อลด Spatial Resolution อย่างเป็นขั้นตอนร่วมกับ Squeeze-and-Excitation
      ซึ่งเป็น Attention Module ที่คำนึงถึง Context ของ Channel ด้วย
    </p>
    <p>
      การศึกษาจาก Google Brain พบว่า EfficientNet สามารถให้ Accuracy สูงกว่ารุ่นก่อนหน้า
      ในขณะที่ใช้พลังงานน้อยกว่า 10 เท่าเมื่อ deploy บน Cloud หรือ Mobile Devices
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>Global Average Pooling ช่วยลดพารามิเตอร์และความซับซ้อนของโมเดลโดยไม่สูญเสีย Performance</li>
        <li>Stride = 2 ใน Convolution ช่วยลดขนาด Feature Map และสามารถแทนที่ Pooling ได้ในบางกรณี</li>
        <li>การใช้ Max Pooling อย่างระมัดระวังร่วมกับ Convolution ขนาดเล็กช่วยให้โมเดลลึกและมีประสิทธิภาพสูง</li>
        <li>Average Pooling มีบทบาทสำคัญในการลดความซับซ้อนของข้อมูลก่อนส่งเข้า Layer ถัดไปในโมเดลลึก</li>
      </ul>
    </div>

    <p>
      บทเรียนจากสถาปัตยกรรมจริงเหล่านี้แสดงให้เห็นว่า Pooling และ Stride ไม่ใช่เพียงเครื่องมือในการลดขนาดข้อมูล
      แต่เป็นองค์ประกอบเชิงกลยุทธ์ที่มีผลต่อความลึก ประสิทธิภาพ และการเรียนรู้เชิงนามธรรมของโมเดลอย่างลึกซึ้ง
      การออกแบบให้เหมาะสมตั้งแต่ต้นจะส่งผลต่อทุกระดับของการประมวลผลภายในโครงข่าย
    </p>
  </div>
</section>


{/* Section 10 */}
<section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Visualization</h2>
  <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img13} />
          </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การทำความเข้าใจการทำงานภายในของโครงข่าย Convolutional Neural Networks (CNNs)
      โดยเฉพาะอย่างยิ่งในส่วนของ Layer ที่เกี่ยวข้องกับการ Pooling และการใช้ Stride
      จำเป็นต้องใช้เทคนิคการ Visualization เพื่อวิเคราะห์ว่าข้อมูลภาพผ่านการแปลงไปอย่างไรในแต่ละชั้น
    </p>

    <p>
      Visualization ช่วยให้เห็นภาพว่า Feature ที่เครือข่ายเรียนรู้จากข้อมูลภาพมีลักษณะอย่างไร
      และช่วยตรวจสอบว่า Layer ต่าง ๆ มีการจับลักษณะ (pattern) ได้อย่างถูกต้องหรือไม่
      โดยเฉพาะในขั้นตอนที่ขนาดของภาพถูกลดลง (Downsampling) จาก Pooling หรือ Stride
    </p>

    <p>
      งานของ Zeiler & Fergus (2013) จากมหาวิทยาลัย New York เป็นหนึ่งในงานคลาสสิกที่แสดงให้เห็นถึงการใช้
      Deconvolutional Network เพื่อย้อนกลับ Feature Map กลับสู่รูปแบบของ Input เพื่อแสดงสิ่งที่โมเดล “มองเห็น”
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">การใช้ Feature Map ในการทำ Visualization</h3>
    <p>
      Feature Map คือผลลัพธ์จากการใช้ Convolution และ Pooling ในแต่ละเลเยอร์
      การแสดงผลของ Feature Map ทำให้สามารถตรวจสอบได้ว่าโมเดลกำลังโฟกัสกับบริเวณใดของภาพ
      ซึ่งนิยมใช้ในงาน Classification และ Object Detection
    </p>

    <p>
      ข้อมูลจาก Stanford CS231n ระบุว่าการใช้ Max Pooling จะทำให้ Feature Map มีลักษณะ "คมชัด" มากขึ้น
      เนื่องจากเลือกเฉพาะลักษณะที่เด่นที่สุดในแต่ละบริเวณ ขณะที่ Average Pooling จะให้ภาพที่ "นุ่มนวล" มากกว่า
      โดยข้อมูลเชิงสถิติถูกเก็บไว้ครบถ้วนแต่รายละเอียดเชิงขอบลดลง
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">Gradient-Based Visualization: Saliency Maps และ Grad-CAM</h3>
    <p>
      เทคนิคเช่น Saliency Maps และ Grad-CAM (Gradient-weighted Class Activation Mapping)
      ใช้ข้อมูลกราดิเอนต์จากการไหลย้อนกลับ (Backpropagation) เพื่อแสดงว่า Feature ใดส่งผลต่อการตัดสินใจของโมเดลมากที่สุด
    </p>

    <p>
      Grad-CAM ใช้การถ่วงน้ำหนักของ Feature Map โดยอิงจากกราดิเอนต์เฉลี่ยของ Target Class และทำการ Upsample กลับเป็นขนาดเดิมของภาพ
      เพื่อแสดงบริเวณที่โมเดลให้ความสำคัญ สามารถใช้วิเคราะห์ผลของ Pooling ว่าทำให้ตำแหน่งสำคัญถูกเบลอหรือไม่
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">การเปรียบเทียบ Visualization ก่อนและหลัง Pooling</h3>
    <p>
      Visualization สามารถนำมาใช้เปรียบเทียบผลกระทบของการทำ Max Pooling, Average Pooling, หรือ Strided Convolution ได้โดยตรง
      ตัวอย่างเช่น:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>Max Pooling ส่งผลให้ข้อมูลขาดความต่อเนื่อง แต่เน้น Feature ที่ชัดเจน</li>
      <li>Average Pooling รักษารูปร่างของ Feature ได้ดีกว่า แต่ความชัดของขอบลดลง</li>
      <li>Strided Convolution สามารถแทน Pooling ได้ แต่ต้องมีการควบคุม Kernel ให้เหมาะสม</li>
    </ul>

    <p>
      จากการวิเคราะห์ภาพจริงในงานของ Google Brain พบว่าเครือข่ายที่ใช้ Strided Convolution อย่างเดียวสามารถเรียนรู้ Feature
      ได้ละเอียดกว่า Max Pooling แต่ต้องใช้ Parameter มากกว่าและมีโอกาส Overfitting
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">การใช้ T-SNE เพื่อดูการกระจายตัวของ Feature</h3>
    <p>
      T-SNE (t-distributed stochastic neighbor embedding) เป็นอีกเครื่องมือหนึ่งที่สามารถใช้วิเคราะห์ผลกระทบของ Pooling
      และ Stride โดยการลดมิติของ Feature Map และแสดงภาพในมิติที่เข้าใจง่าย (2D/3D)
      เพื่อดูว่า Feature ที่โมเดลเรียนรู้สามารถแยกแยะคลาสได้หรือไม่
    </p>


    <p>
      การกระจายตัวของ Feature หลังการทำ Pooling และการใช้ Stride จะเห็นได้ชัดว่า หากข้อมูลไม่ได้ถูกบีบมากเกินไป
      จะสามารถแยกกลุ่มได้อย่างชัดเจน แต่ถ้าการลดขนาดรุนแรงเกินไป Feature จะปะปนกันและยากต่อการแยกคลาส
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Insight Box</h3>
      <ul className="list-disc list-inside space-y-2">
        <li>
          การทำ Visualization ช่วยวิเคราะห์การไหลของข้อมูลเชิงพื้นที่ในโครงข่ายแบบลึกได้อย่างมีประสิทธิภาพ
        </li>
        <li>
          Grad-CAM เป็นเครื่องมือสำคัญที่แสดงให้เห็นว่าการ Pool ส่งผลต่อความสามารถในการโฟกัสของโมเดลอย่างไร
        </li>
        <li>
          การเปรียบเทียบ Feature Map แบบ before-after ช่วยตัดสินใจได้ว่า Pooling Technique ใดเหมาะสมกับโจทย์นั้นมากที่สุด
        </li>
        <li>
          Visualization ช่วยในการ Debug เครือข่ายในระดับลึก และตรวจสอบปัญหา เช่น การลบล้าง Feature หรือการเบลอข้อมูล
        </li>
      </ul>
    </div>
  </div>
</section>


<section id="insight" className="mb-20 scroll-mt-32 px-4 sm:px-6 lg:px-8">
  <h2 className="text-2xl font-bold text-center text-yellow-500 mb-8">Special Insight</h2>

  <div className="bg-yellow-100 dark:bg-yellow-800 rounded-xl shadow-lg p-6 sm:p-8 lg:p-10 max-w-5xl mx-auto space-y-6">
    <p className="text-center text-lg font-medium text-gray-800 dark:text-gray-100">
      “Pooling is the compression algorithm of deep learning—it lets the model focus on what matters and ignore the rest.”
    </p>

    <div className="text-gray-800 dark:text-gray-100 text-base leading-relaxed space-y-5">
      <p>
        การศึกษาจาก Stanford และ MIT-IBM Watson AI Lab อธิบายว่า Pooling คือกระบวนการสำคัญที่ช่วยกลั่นกรองเฉพาะ Feature
        ที่สำคัญจากภาพหรือตัวแทนข้อมูลขนาดใหญ่ ทำให้โมเดลเข้าใจภาพรวมได้แม่นยำโดยไม่ต้องวิเคราะห์ทุกจุด
      </p>

      <p>
        Pooling เป็นกระบวนการแบบไม่ต้องเรียนรู้พารามิเตอร์ (parameter-free) ช่วยให้โมเดลฝึกได้เร็วขึ้น และลดความเสี่ยงจากการจำมากเกินไป (Overfitting)
      </p>

      <div>
        <h3 className="text-lg font-semibold mb-2">ทำไม Pooling ถึงสำคัญ?</h3>
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>ลดขนาดของ Feature Map → ประหยัดหน่วยความจำ</li>
          <li>เพิ่มความทนต่อการแปรผันของภาพ เช่น การเลื่อนหรือย่อขยาย</li>
          <li>ช่วยเน้นส่วนที่สำคัญและตัดสิ่งรบกวน</li>
        </ul>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">Max Pooling: เน้นจุดเด่น</h3>
        <p>
          DeepMind (2021) พบว่า Max Pooling ช่วยให้โมเดลเลือกเฉพาะข้อมูลที่เด่นที่สุด เช่น เส้นขอบหรือรายละเอียดชัดเจนในภาพ
          ซึ่งสัมพันธ์กับแนวคิดใน Signal Processing ที่เน้นรักษาพลังงานสูงในข้อมูล
        </p>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">Stride คืออะไร?</h3>
        <p>
          Stride คือการกำหนดระยะการเลื่อนของ Filter หาก Stride = 1 จะละเอียดและเก็บข้อมูลครบ แต่ถ้า Stride มากกว่า 1
          จะลดจำนวนจุดตรวจสอบลง ช่วยลดการคำนวณและเวลาในการฝึกโมเดล
        </p>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">Transfer Learning เริ่มต้นที่ Pooling</h3>
        <p>
          งานวิจัยของ MIT ระบุว่า Feature จาก Pooling Layer สามารถนำไปใช้กับงานใหม่ได้อย่างมีประสิทธิภาพ
          เช่น Object Detection หรือ Style Transfer → กลายเป็นรากฐานของ Transfer Learning
        </p>
      </div>

      <div className="bg-white/50 dark:bg-black/20 p-5 rounded-lg space-y-3">
        <h4 className="font-semibold">Key Takeaways:</h4>
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li><strong>Pooling:</strong> ช่วยย่อข้อมูลโดยคงลักษณะเด่น</li>
          <li><strong>Max Pooling:</strong> คัดเลือก Feature ที่โดดเด่น</li>
          <li><strong>Stride:</strong> ปรับความถี่ของการอ่านข้อมูลในภาพ</li>
          <li><strong>Global Pooling:</strong> สรุปภาพรวมของ Feature ทั้งหมด</li>
          <li><strong>Average Pooling:</strong> ค่าเฉลี่ยเพื่อการวิเคราะห์แบบกลางๆ</li>
        </ul>
      </div>

      <p>
        การเข้าใจ Pooling และ Stride ไม่ใช่เพียงแค่เรื่องของการลดขนาดข้อมูล
        แต่เป็นรากฐานของการออกแบบโมเดลที่ฉลาด เข้าใจสิ่งสำคัญในข้อมูล และพร้อมต่อยอดในงานจริงได้หลากหลาย
      </p>
    </div>
  </div>
</section>





          {/* Quiz */}
          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day23 theme={theme} />
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

          {/* Comments */}
          <Comments theme={theme} />
        </div>
      </div>

      <div className="hidden lg:block fixed right-6 top-[185px] z-50">
        <ScrollSpy_Ai_Day23 />
      </div>
      <SupportMeButton />
    </div>
  );
};

export default Day23_PoolingStride;
