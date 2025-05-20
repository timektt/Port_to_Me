import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day37 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day37";
import MiniQuiz_Day37 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day37";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day37_DataAugmentation = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day37_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day37_2").format("auto").quality("auto").resize(scale().width(500));
  const img3 = cld.image("Day37_3").format("auto").quality("auto").resize(scale().width(500));
  const img4 = cld.image("Day37_4").format("auto").quality("auto").resize(scale().width(500));
  const img5 = cld.image("Day37_5").format("auto").quality("auto").resize(scale().width(500));
  const img6 = cld.image("Day37_6").format("auto").quality("auto").resize(scale().width(500));
  const img7 = cld.image("Day37_7").format("auto").quality("auto").resize(scale().width(500));
  const img8 = cld.image("Day37_8").format("auto").quality("auto").resize(scale().width(500));
  const img9 = cld.image("Day37_9").format("auto").quality("auto").resize(scale().width(500));
  const img10 = cld.image("Day37_10").format("auto").quality("auto").resize(scale().width(500));
  const img11 = cld.image("Day37_11").format("auto").quality("auto").resize(scale().width(500));
  const img12 = cld.image("Day37_12").format("auto").quality("auto").resize(scale().width(500));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 37: Data Augmentation Techniques</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
               <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

       <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Data Augmentation จึงสำคัญ</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ความท้าทายของข้อมูลใน Machine Learning</h3>
    <p>
      ในหลายกรณี ระบบเรียนรู้ด้วยเครื่องโดยเฉพาะ Deep Learning ต้องการข้อมูลจำนวนมากเพื่อให้สามารถเรียนรู้ pattern ที่ซับซ้อนได้อย่างถูกต้อง อย่างไรก็ตาม การเก็บข้อมูลจำนวนมาก โดยเฉพาะแบบ labeled มักมีค่าใช้จ่ายสูง ใช้เวลา และอาจมีข้อจำกัดทางกฎหมายหรือจริยธรรม เช่น ข้อมูลผู้ป่วย หรือข้อมูลส่วนตัวของผู้ใช้
    </p>

    <h3>Data Augmentation คืออะไร?</h3>
    <p>
      Data Augmentation คือกระบวนการสร้างตัวอย่างข้อมูลใหม่โดยการแปลงข้อมูลที่มีอยู่ให้หลากหลายมากขึ้น โดยไม่เปลี่ยน label ของข้อมูล เช่น การหมุนภาพ, เปลี่ยนสี, เพิ่ม noise, หรือในกรณีของข้อความ อาจใช้การแทนคำ (synonym replacement) หรือแปลกลับ (back-translation) แนวทางนี้ช่วยขยายชุดข้อมูลเทียม (synthetic data) จากข้อมูลจริง ทำให้โมเดลเรียนรู้ได้ดียิ่งขึ้น
    </p>

    <h3>ประโยชน์เชิงลึกของ Data Augmentation</h3>
    <ul className="list-disc pl-6">
      <li>เพิ่มปริมาณข้อมูลเพื่อป้องกัน overfitting</li>
      <li>ช่วยให้โมเดลเรียนรู้ feature ที่มีความยืดหยุ่น (robust features)</li>
      <li>จำลองสถานการณ์ใหม่ ๆ ที่อาจไม่ปรากฏใน training set</li>
      <li>ส่งเสริมความสามารถของโมเดลในการ generalize ไปยังข้อมูลจริง</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานของ Shorten & Khoshgoftaar (2019) พบว่า Data Augmentation ช่วยให้โมเดล convolutional neural network (CNN) เพิ่ม accuracy ได้มากกว่า 10% บน dataset ขนาดเล็ก เช่น CIFAR-10 และ MNIST โดยไม่ต้องเปลี่ยนสถาปัตยกรรม
      </p>
    </div>

    <h3>แนวโน้มการใช้งานในอุตสาหกรรม</h3>
    <p>
      ในองค์กรขนาดใหญ่ เช่น Google, Meta, และ Tesla การใช้ Data Augmentation ถูกใช้เป็นมาตรฐานในการฝึกโมเดล AI เช่น ในการประมวลผลภาพจากรถยนต์ไร้คนขับ หรือการฝึกโมเดลแปลภาษาในหลายภาษา ด้วยการสร้างตัวอย่างจากภาษาต้นทางโดยไม่ต้องพึ่งพามนุษย์ในการแปลใหม่ทั้งหมด
    </p>

    <h3>ตารางเปรียบเทียบ: โมเดลมี/ไม่มี Augmentation</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เกณฑ์</th>
            <th className="border px-4 py-2">ไม่มี Augmentation</th>
            <th className="border px-4 py-2">ใช้ Augmentation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Accuracy</td>
            <td className="border px-4 py-2">ต่ำ (60–70%)</td>
            <td className="border px-4 py-2">สูงขึ้น (75–85%)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Overfitting</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ลดลง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">การ generalize</td>
            <td className="border px-4 py-2">ต่ำ</td>
            <td className="border px-4 py-2">ดีขึ้น</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 mt-6">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ข้อมูลไม่ได้มีอยู่มากเสมอ โดยเฉพาะใน domain ที่มีความเฉพาะสูง เช่น medical imaging, legal document หรือภาษาท้องถิ่น การใช้ Data Augmentation อย่างเป็นระบบช่วยเปิดทางให้โมเดลสามารถเรียนรู้ได้อย่างมีประสิทธิภาพ แม้จะมีทรัพยากรจำกัด
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data.</li>
      <li>Wang, Y. et al. (2021). Understanding Self-supervised Augmentation Techniques in Computer Vision. arXiv:2103.05905</li>
      <li>Stanford CS231n Lecture Notes: Data Augmentation</li>
      <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
    </ul>
  </div>
</section>


      <section id="principles" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. หลักการพื้นฐานของการ Augmentation</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ความหมายของ Data Augmentation</h3>
    <p>
      Data Augmentation คือกระบวนการสร้างข้อมูลใหม่จากข้อมูลเดิมโดยการเปลี่ยนแปลงในลักษณะที่ไม่กระทบต่อ label หรือความหมายที่แท้จริง เช่น การหมุนภาพ, การแปลข้อความ, หรือการแปลงสัญญาณเสียง โดยมีเป้าหมายเพื่อเพิ่มปริมาณข้อมูลฝึกให้มากขึ้น โดยไม่ต้องเก็บข้อมูลใหม่
    </p>

    <h3>ประโยชน์ของ Data Augmentation</h3>
    <ul className="list-disc pl-6">
      <li>เพิ่ม generalization ของโมเดล ลดปัญหา overfitting</li>
      <li>เพิ่มความทนทานต่อ noise และ variation ในข้อมูลจริง</li>
      <li>ขยาย dataset โดยไม่เพิ่มต้นทุนการเก็บข้อมูล</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        ในงานของ Shorten & Khoshgoftaar (2019) ได้วิเคราะห์ว่า augmentation ช่วยเพิ่ม accuracy ได้มากกว่า 10% ในหลาย benchmark โดยเฉพาะเมื่อ dataset มีขนาดจำกัด
      </p>
    </div>

    <h3>แนวทางการแปลงข้อมูลอย่างมีประสิทธิภาพ</h3>
    <ul className="list-disc pl-6">
      <li>Transformation ต้องไม่เปลี่ยน label ของข้อมูล เช่น การกลับภาพแนวตั้งอาจไม่เหมาะกับ digit recognition</li>
      <li>Augmentation ควรสุ่มด้วยความน่าจะเป็น เพื่อไม่ให้โมเดล overfit กับ pattern ที่เพิ่มเข้าไป</li>
      <li>ควรใช้ augmentation ที่ตรงกับธรรมชาติของ task เช่น text ต้องไม่เปลี่ยนความหมาย</li>
    </ul>

    <h3>เปรียบเทียบ: Manual vs Learned Augmentation</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">รูปแบบ</th>
            <th className="border px-4 py-2">ตัวอย่าง</th>
            <th className="border px-4 py-2">ข้อดี</th>
            <th className="border px-4 py-2">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Manual</td>
            <td className="border px-4 py-2">Flip, Rotation, Noise</td>
            <td className="border px-4 py-2">เข้าใจง่าย ใช้ได้กับหลาย task</td>
            <td className="border px-4 py-2">อาจไม่ optimal ต่อ task-specific pattern</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Learned</td>
            <td className="border px-4 py-2">AutoAugment, RandAugment</td>
            <td className="border px-4 py-2">เรียนรู้ augmentation policy จากข้อมูล</td>
            <td className="border px-4 py-2">ต้องใช้ compute สูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>ความเชื่อมโยงกับ Regularization</h3>
    <p>
      Augmentation ทำหน้าที่คล้ายกับ regularization โดยบังคับให้โมเดลไม่ยึดติดกับ feature เฉพาะเจาะจงจากตัวอย่างเดิม แต่ต้องสามารถปรับตัวให้เข้าใจข้อมูลที่มี variation ได้กว้างขึ้น ส่งผลให้โมเดลมีความสามารถในการ generalize ดียิ่งขึ้นใน unseen data
    </p>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400">
      <p className="font-semibold">Highlight:</p>
      <p>
        การวิจัยจาก Google Brain (Cubuk et al., 2019) ระบุว่า การใช้ learned augmentation เช่น AutoAugment บน CIFAR-10 และ ImageNet สามารถเพิ่มความแม่นยำได้เกิน baseline อย่างมีนัยสำคัญ
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. Journal of Big Data.</li>
      <li>Cubuk, E. Z., et al. (2019). AutoAugment: Learning Augmentation Strategies from Data. CVPR.</li>
      <li>Wang, H. et al. (2021). Survey on Text Data Augmentation for Deep Learning. arXiv:2107.03158</li>
      <li>Stanford CS231n: Convolutional Neural Networks for Visual Recognition – Lecture on Data Augmentation</li>
    </ul>
  </div>
</section>


     <section id="cv" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Augmentation สำหรับ Computer Vision</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>พื้นฐานของ Augmentation ในงานภาพ</h3>
    <p>
      ในงาน Computer Vision การทำ Data Augmentation มีบทบาทสำคัญในการเพิ่มความหลากหลายของชุดข้อมูลภาพ โดยไม่ต้องเก็บภาพใหม่ ซึ่งช่วยให้โมเดลเรียนรู้จากตัวอย่างที่มี variation สูงขึ้น และลดโอกาสของ overfitting
    </p>

    <h3>เทคนิค Augmentation แบบพื้นฐาน</h3>
    <ul className="list-disc pl-6">
      <li><strong>Flipping:</strong> การพลิกภาพแนวนอนหรือแนวตั้ง</li>
      <li><strong>Rotation:</strong> หมุนภาพด้วยมุมต่าง ๆ</li>
      <li><strong>Scaling:</strong> การซูมหรือย่อขนาด</li>
      <li><strong>Translation:</strong> เลื่อนภาพในแกน X/Y</li>
      <li><strong>Color Jittering:</strong> ปรับความสว่าง ความอิ่มสี และ contrast</li>
      <li><strong>Noise Injection:</strong> ใส่ Gaussian noise หรือ salt-and-pepper noise</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยจาก Google Brain (2019) ชี้ให้เห็นว่า Augmentation ด้วย Random Flip และ Color Jitter เพียงไม่กี่เทคนิค ก็สามารถช่วยเพิ่ม accuracy ของ ResNet บน ImageNet ได้หลายเปอร์เซ็นต์
      </p>
    </div>

    <h3>Advanced Augmentation Techniques</h3>
    <p>เทคนิคขั้นสูงที่ถูกนำมาใช้ในงานวิจัยและการผลิตจริง:</p>
    <ul className="list-disc pl-6">
      <li><strong>Cutout:</strong> ลบสี่เหลี่ยมบางส่วนของภาพออกแบบสุ่ม</li>
      <li><strong>Mixup:</strong> ผสมภาพสองภาพเข้าด้วยกันโดยใช้ interpolation</li>
      <li><strong>CutMix:</strong> ผสม patch จากภาพอื่นเข้ากับภาพเป้าหมาย</li>
      <li><strong>AutoAugment:</strong> ใช้ reinforcement learning เพื่อหา policy ของการทำ augmentation ที่ดีที่สุด</li>
      <li><strong>RandAugment:</strong> ใช้ random policy จากชุด transformation โดยไม่ต้อง tune มาก</li>
    </ul>

    <h3>เปรียบเทียบวิธี Augmentation</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">เป้าหมาย</th>
            <th className="border px-4 py-2">ข้อดี</th>
            <th className="border px-4 py-2">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Cutout</td>
            <td className="border px-4 py-2">เพิ่มความต้านทานต่อ occlusion</td>
            <td className="border px-4 py-2">ง่ายและมีประสิทธิภาพ</td>
            <td className="border px-4 py-2">อาจลบข้อมูลสำคัญ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Mixup</td>
            <td className="border px-4 py-2">ปรับ decision boundary</td>
            <td className="border px-4 py-2">ลด overfitting</td>
            <td className="border px-4 py-2">ลด interpretability</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">AutoAugment</td>
            <td className="border px-4 py-2">ค้นหา policy ที่ดีที่สุด</td>
            <td className="border px-4 py-2">accuracy สูง</td>
            <td className="border px-4 py-2">ใช้เวลา compute สูง</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        การใช้เทคนิค CutMix ทำให้ EfficientNet บน ImageNet มี performance ดีกว่า baseline ถึง 1.5–2.0% โดยไม่ต้องเปลี่ยนสถาปัตยกรรมใด ๆ (Yun et al., 2019)
      </p>
    </div>

    <h3>แนวทางปฏิบัติที่แนะนำ</h3>
    <ul className="list-disc pl-6">
      <li>เริ่มจาก augmentation พื้นฐานก่อน เช่น flipping และ scaling</li>
      <li>ใช้ advanced technique เมื่อ dataset มีขนาดเล็กหรือมีปัญหา overfitting</li>
      <li>ทดสอบ performance ผ่าน cross-validation ก่อน deploy จริง</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Devries, T. & Taylor, G. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. arXiv:1708.04552</li>
      <li>Zhang, H. et al. (2017). Mixup: Beyond Empirical Risk Minimization. arXiv:1710.09412</li>
      <li>Yun, S. et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers. arXiv:1905.04899</li>
      <li>Cubuk, E. D. et al. (2019). AutoAugment: Learning Augmentation Policies. CVPR</li>
      <li>Stanford CS231n Lecture 10: Data Augmentation & Regularization</li>
    </ul>
  </div>
</section>


       <section id="nlp" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Augmentation สำหรับ NLP (Text)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ภาพรวมของ NLP Augmentation</h3>
    <p>
      ในงานประมวลผลภาษาธรรมชาติ (NLP) การสร้างข้อมูลใหม่จากข้อมูลที่มีอยู่เพื่อเพิ่มประสิทธิภาพการเรียนรู้ของโมเดลถือเป็นหนึ่งในหัวข้อที่ได้รับความสนใจอย่างมาก โดยเฉพาะเมื่อชุดข้อมูลจำกัด หรือมีปัญหา class imbalance ซึ่งส่งผลต่อความสามารถในการ generalize ของโมเดล
    </p>

    <h3>เทคนิคพื้นฐานในการ Augment ข้อความ</h3>
    <ul className="list-disc pl-6">
      <li><strong>Synonym Replacement:</strong> เปลี่ยนคำในประโยคด้วยคำพ้องความหมายจาก WordNet หรือ embedding space</li>
      <li><strong>Random Insertion:</strong> แทรกคำที่เกี่ยวข้องแบบสุ่มลงในประโยค</li>
      <li><strong>Random Deletion:</strong> ลบคำบางคำออกจากประโยคแบบสุ่ม</li>
      <li><strong>Random Swap:</strong> สลับตำแหน่งของคำสองคำภายในประโยค</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400">
      <p className="font-semibold">Insight Box:</p>
      <p>
        เทคนิค EDA (Easy Data Augmentation) ที่นำเสนอโดย Wei และ Zou (2019) แสดงให้เห็นว่าสามารถเพิ่ม accuracy ของ text classification model ได้ในหลาย dataset โดยไม่ต้องปรับสถาปัตยกรรมของโมเดลเลย
      </p>
    </div>

    <h3>Contextual Augmentation</h3>
    <p>
      การใช้โมเดลภาษาขนาดใหญ่ (เช่น BERT) เพื่อสร้างคำใหม่ในบริบทเดิม เช่น การใช้ masked language modeling เพื่อแทนที่คำบางคำในประโยค
    </p>
    <pre><code className="language-python">from nlpaug.augmenter.word import ContextualWordEmbsAug
aug = ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
augmented_text = aug.augment("The quick brown fox jumps over the lazy dog")</code></pre>

    <h3>Back Translation</h3>
    <p>
      การแปลข้อความต้นฉบับไปยังภาษาอื่น และแปลกลับมายังภาษาต้นทางอีกครั้ง ช่วยสร้างข้อความที่หลากหลายทางโครงสร้าง แต่ยังคงความหมายเดิม เช่น EN → DE → EN
    </p>
    <ul className="list-disc pl-6">
      <li>เพิ่มความหลากหลายทางภาษาศาสตร์</li>
      <li>เหมาะกับงาน machine translation และ sentiment analysis</li>
    </ul>

    <h3>Paraphrasing โดยใช้โมเดล Generative</h3>
    <p>
      การใช้โมเดลประเภท T5, GPT-3 หรือ Pegasus เพื่อสร้างข้อความที่มีความหมายคล้ายกัน แต่มีโครงสร้างแตกต่าง เช่น การใช้ prompt "Paraphrase the following: ..."
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานวิจัยโดย Kumar et al. (2022) พบว่า การใช้ back translation ร่วมกับ T5-paraphrase ทำให้ performance ของ BERT classifier เพิ่มขึ้นกว่า 3–6% ใน task ที่มี data น้อย
      </p>
    </div>

    <h3>ตารางเปรียบเทียบเทคนิคยอดนิยม</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">จุดเด่น</th>
            <th className="border px-4 py-2">ข้อควรระวัง</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">EDA</td>
            <td className="border px-4 py-2">เรียบง่ายและรวดเร็ว</td>
            <td className="border px-4 py-2">อาจเปลี่ยนความหมายของประโยค</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Back Translation</td>
            <td className="border px-4 py-2">เปลี่ยนโครงสร้างภาษามนุษย์อย่างเป็นธรรมชาติ</td>
            <td className="border px-4 py-2">ต้องใช้ทรัพยากรการแปลและเวลา</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Contextual Embedding</td>
            <td className="border px-4 py-2">สร้างคำใหม่ที่สัมพันธ์กับบริบท</td>
            <td className="border px-4 py-2">ต้องใช้โมเดลที่ฝึกล่วงหน้า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Wei, J. and Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. arXiv:1901.11196</li>
      <li>Kumar, A. et al. (2022). Augmenting Low-Resource Datasets with Paraphrasing and Back-Translation. EMNLP Findings</li>
      <li>Jiang, Z. et al. (2020). How Can We Know What Language Models Know? TACL</li>
      <li>Stanford CS224N 2023: Lecture Notes on NLP Data Augmentation</li>
    </ul>
  </div>
</section>


   <section id="time-series" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Augmentation สำหรับ Time Series / Audio</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ภาพรวมของ Time Series และ Audio Data</h3>
    <p>
      ข้อมูลแบบ Time Series และ Audio มีลักษณะเฉพาะคือมีความต่อเนื่องตามลำดับเวลา (temporal dependency) ซึ่งทำให้การทำ Augmentation ต้องระมัดระวังไม่ให้ทำลายโครงสร้างเวลาเดิม เทคนิคที่ใช้มักมุ่งเน้นการเพิ่มความหลากหลายเชิงเวลา เช่น การเปลี่ยนแปลงความเร็ว การสุ่ม noise หรือการเลื่อนเฟสของสัญญาณ
    </p>

    <h3>เทคนิคพื้นฐานของ Time Series Augmentation</h3>
    <ul className="list-disc pl-6">
      <li><strong>Time Warping:</strong> เปลี่ยนความเร็วในช่วงบางจุดของลำดับเวลา</li>
      <li><strong>Window Slicing:</strong> ตัด window ย่อยจากลำดับเดิมมาใช้เป็น sample ใหม่</li>
      <li><strong>Jittering:</strong> เติม Gaussian noise เข้าไปในค่าของลำดับเวลา</li>
      <li><strong>Permutation:</strong> สลับลำดับของ segment ย่อย ๆ</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยจาก Harvard (Um et al., 2017) แสดงให้เห็นว่าการใช้ jittering และ slicing ช่วยเพิ่มความแม่นยำของโมเดล LSTM บนชุดข้อมูล Human Activity Recognition ได้มากกว่า 10%
      </p>
    </div>

    <h3>เทคนิคพื้นฐานสำหรับ Audio Augmentation</h3>
    <ul className="list-disc pl-6">
      <li><strong>Pitch Shifting:</strong> เปลี่ยน pitch โดยไม่เปลี่ยนความเร็ว</li>
      <li><strong>Time Stretching:</strong> เปลี่ยนความเร็วโดยไม่เปลี่ยน pitch</li>
      <li><strong>Background Noise:</strong> เติมเสียง noise หรือ ambience เข้าไป</li>
      <li><strong>SpecAugment:</strong> ปรับเปลี่ยน spectrogram โดย masking frequency/time</li>
    </ul>

    <h3>เปรียบเทียบเทคนิค Augmentation สำหรับ Time Series กับ Audio</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">ใช้กับ Time Series</th>
            <th className="border px-4 py-2">ใช้กับ Audio</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Jittering</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️ (Noise Addition)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Time Warping</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">✔️ (Time Stretching)</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Window Slicing</td>
            <td className="border px-4 py-2">✔️</td>
            <td className="border px-4 py-2">—</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">SpecAugment</td>
            <td className="border px-4 py-2">—</td>
            <td className="border px-4 py-2">✔️</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        SpecAugment ซึ่งถูกนำเสนอในงานของ Google Research (Park et al., 2019) เป็นเทคนิคสำคัญที่ทำให้โมเดล ASR อย่าง LAS และ Conformer มี performance สูงขึ้นบนชุดข้อมูลเช่น LibriSpeech และ TED-LIUM
      </p>
    </div>

    <h3>ข้อควรระวัง</h3>
    <ul className="list-disc pl-6">
      <li>การ Augment โดยไม่คงความสัมพันธ์เชิงเวลาอาจทำลายลักษณะสำคัญของ sequence</li>
      <li>ควรเลือกเทคนิคให้เหมาะกับโมเดล เช่น CNN ต้องการ invariance แต่ RNN ต้องการ preservation</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Um, T.T. et al. (2017). Data Augmentation of Wearable Sensor Data for Parkinson’s Disease Monitoring. ACM IHI.</li>
      <li>Park, D.S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. arXiv:1904.08779</li>
      <li>Zhang, C. et al. (2021). A Survey on Deep Learning for Time-Series Forecasting. arXiv:2103.05848</li>
      <li>Harvard CS281 Lecture Notes: Sequence Modeling & Augmentation</li>
    </ul>
  </div>
</section>


   <section id="advanced" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Advanced Augmentation: GAN, SimCLR, Self-Supervised</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>บทนำสู่เทคนิคขั้นสูงในการเพิ่มข้อมูล</h3>
    <p>
      เมื่อการทำ Data Augmentation แบบพื้นฐานไม่เพียงพอในการเพิ่มประสิทธิภาพของโมเดล เทคนิคขั้นสูง เช่น GAN (Generative Adversarial Networks), SimCLR และ Self-Supervised Learning จึงถูกพัฒนาเพื่อสร้างตัวอย่างใหม่ที่มีคุณภาพสูงและมีความสัมพันธ์กับ distribution จริงมากยิ่งขึ้น
    </p>

    <h3>Generative Adversarial Networks (GAN)</h3>
    <p>
      GAN เป็นหนึ่งในเทคนิคที่ทรงพลังในการสร้างตัวอย่างใหม่ โดยมีโครงสร้างเป็นสองเครือข่ายคือ Generator และ Discriminator ซึ่งแข่งขันกันจนสามารถสร้างข้อมูลที่คล้ายกับข้อมูลจริงได้อย่างสมจริง
    </p>
    <ul className="list-disc pl-6">
      <li>ใช้ในงาน image synthesis, video generation, medical data augmentation</li>
      <li>สามารถควบคุมรูปแบบของข้อมูล เช่น สี, ท่าทาง, แสง</li>
    </ul>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>GAN ได้รับความนิยมอย่างแพร่หลายใน medical imaging โดยเฉพาะการเพิ่มภาพ MRI และ CT ที่มีข้อมูลจำกัด</p>
    </div>

    <h3>SimCLR และ Contrastive Learning</h3>
    <p>
      SimCLR เป็น framework สำหรับ self-supervised contrastive learning ที่มุ่งเน้นให้ representation ของตัวอย่างที่คล้ายกันอยู่ใกล้กันใน latent space และตัวอย่างที่ต่างกันอยู่ห่างกัน
    </p>
    <ul className="list-disc pl-6">
      <li>ไม่ต้องใช้ label ในการฝึก</li>
      <li>สามารถใช้กับ downstream tasks ได้หลากหลาย เช่น classification, retrieval</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        SimCLR สามารถ outperform supervised learning บนหลาย benchmark เมื่อใช้ข้อมูล unlabeled จำนวนมาก
      </p>
    </div>

    <h3>Self-Supervised Pretext Tasks</h3>
    <p>
      Self-supervised learning เป็นแนวทางที่ใช้ task ที่ตั้งขึ้นเอง (pretext tasks) เพื่อฝึกโมเดล เช่น การทำนาย patch ที่หายไปในภาพ หรือทำนายคำถัดไปในข้อความ โดยไม่ต้องมี label จริง
    </p>
    <ul className="list-disc pl-6">
      <li>Jigsaw prediction</li>
      <li>Rotation prediction</li>
      <li>Masked Language Modeling (เช่น BERT)</li>
    </ul>

    <h3>ตารางเปรียบเทียบเทคนิคขั้นสูง</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">ลักษณะเด่น</th>
            <th className="border px-4 py-2">ใช้กับ Modalities</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">GAN</td>
            <td className="border px-4 py-2">สร้างข้อมูลใหม่ที่ใกล้เคียงกับข้อมูลจริง</td>
            <td className="border px-4 py-2">Image, Video</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">SimCLR</td>
            <td className="border px-4 py-2">เรียนรู้ representation แบบไม่ใช้ label</td>
            <td className="border px-4 py-2">Image</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Self-Supervised Pretext</td>
            <td className="border px-4 py-2">สร้าง task เองจากข้อมูล</td>
            <td className="border px-4 py-2">Text, Image, Audio</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Goodfellow, I. et al. (2014). Generative Adversarial Nets. NeurIPS.</li>
      <li>Chen, T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.</li>
      <li>He, K. et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. CVPR.</li>
      <li>Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.</li>
      <li>Stanford CS231n Lecture 2023: Self-Supervised Learning and Contrastive Loss</li>
    </ul>
  </div>
</section>


  <section id="smart-strategy" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. การใช้ Augmentation อย่างชาญฉลาด (Smart Strategies)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>บทนำ: ทำไมกลยุทธ์ที่ชาญฉลาดจึงสำคัญ</h3>
    <p>
      แม้ว่า Data Augmentation จะสามารถเพิ่มปริมาณข้อมูลได้อย่างรวดเร็ว แต่หากเลือกใช้โดยขาดความเข้าใจเชิงกลยุทธ์ อาจส่งผลให้โมเดลเรียนรู้จากข้อมูลที่ไม่มีประโยชน์หรือสร้าง bias เพิ่มขึ้น การออกแบบกลยุทธ์การเพิ่มข้อมูลอย่างมีหลักการจึงมีความสำคัญต่อประสิทธิภาพของระบบ
    </p>

    <h3>ประเภทของกลยุทธ์ที่ชาญฉลาด</h3>
    <ul className="list-disc pl-6">
      <li><strong>Class-Aware Augmentation:</strong> ใช้ augmentation ที่แตกต่างกันตามแต่ละ class เพื่อเพิ่มความหลากหลายที่มีบริบท</li>
      <li><strong>Adaptive Augmentation:</strong> ปรับวิธีการ augment ตามสถานะของการฝึก เช่น loss หรือ confidence</li>
      <li><strong>Curriculum-based Augmentation:</strong> เริ่มจาก augmentation ที่ง่าย แล้วค่อยเพิ่มความซับซ้อน</li>
      <li><strong>Augmentation Policy Search:</strong> ใช้อัลกอริธึมเชิง optimization เพื่อค้นหานโยบายที่ดีที่สุด</li>
    </ul>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        ในงาน AutoAugment (Cubuk et al., 2019, Google Brain) การใช้ reinforcement learning เพื่อค้นหา augmentation policy โดยอัตโนมัติ ช่วยให้ Accuracy บน CIFAR-10 และ ImageNet เพิ่มขึ้นอย่างมีนัยสำคัญ
      </p>
    </div>

    <h3>การผสมผสานระหว่างหลายกลยุทธ์</h3>
    <p>
      การผสมหลายวิธี เช่น ใช้ Mixup ร่วมกับ CutMix และเรียนรู้นโยบายผ่าน RandAugment หรือ TrivialAugment สามารถช่วยลด overfitting และสร้าง robustness ได้ในหลายกรณี
    </p>

    <h3>ตัวอย่างการนำไปใช้ในระบบจริง</h3>
    <ul className="list-disc pl-6">
      <li>Facebook ใช้ Augmentation แบบ curriculum ในการฝึก self-supervised pretraining (BYOL, 2020)</li>
      <li>Google ใช้ AutoAugment สำหรับ EfficientNet training</li>
      <li>Amazon SageMaker นำ Policy-based Augmentation เข้า pipeline production</li>
    </ul>

    <h3>การจัดอันดับกลยุทธ์</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead className="bg-gray-700 dark:bg-gray-800 text-white">
        <tr>
          <th className="border px-4 py-2">ชื่อกลยุทธ์</th>
          <th className="border px-4 py-2">ข้อดี</th>
          <th className="border px-4 py-2">ข้อจำกัด</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">AutoAugment</td>
          <td className="border px-4 py-2">ค้นหานโยบายที่ดีที่สุดโดยอัตโนมัติ</td>
          <td className="border px-4 py-2">ต้องใช้ compute สูง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">RandAugment</td>
          <td className="border px-4 py-2">เร็วกว่าและง่ายกว่า AutoAugment</td>
          <td className="border px-4 py-2">ไม่มีการปรับแบบ class-specific</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Curriculum Augment</td>
          <td className="border px-4 py-2">จำลองลำดับการเรียนรู้</td>
          <td className="border px-4 py-2">ออกแบบลำดับยาก</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        แม้ Augmentation จะดูเป็นกระบวนการง่าย ๆ แต่การเลือกให้เหมาะกับ task, model และ data distribution คือสิ่งที่แยกโมเดลระดับ production ออกจาก baseline ทั่วไปได้อย่างแท้จริง
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Cubuk, E. et al. (2019). AutoAugment: Learning Augmentation Policies from Data. CVPR.</li>
      <li>Ho, D. et al. (2020). Population Based Augmentation Policy Learning. arXiv:2002.00361</li>
      <li>Kolesnikov, A. et al. (2020). Big Transfer (BiT): General Visual Representation Learning. ECCV</li>
      <li>Grill, J.-B. et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. NeurIPS</li>
      <li>Stanford CS231n: Lecture on Data Augmentation Strategies</li>
    </ul>
  </div>
</section>


   <section id="evaluation" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. การวัดผลของ Augmentation</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>ความสำคัญของการประเมินผลหลัง Augmentation</h3>
    <p>
      หลังจากการใช้เทคนิค Data Augmentation ไม่ว่าจะเป็นทางภาพ เสียง ข้อความ หรือข้อมูลเชิงเวลา จำเป็นต้องประเมินผลเพื่อวัดว่า augmentation นั้นช่วยให้โมเดล generalize ได้ดีขึ้นจริงหรือไม่ โดยเฉพาะในบริบทของการ production และ deployment จริงที่ไม่สามารถพึ่งพาเพียง validation accuracy ได้
    </p>

    <h3>ตัวชี้วัดหลักในการประเมิน</h3>
    <ul className="list-disc pl-6">
      <li><strong>Validation Accuracy:</strong> ตัวชี้วัดพื้นฐานที่ใช้เปรียบเทียบก่อนและหลังทำ augmentation</li>
      <li><strong>Out-of-Distribution (OOD) Accuracy:</strong> ใช้วัดความสามารถในการ generalize บนข้อมูลที่ต่างจาก training set</li>
      <li><strong>Robustness Metrics:</strong> เช่น accuracy หลังจากเพิ่ม noise, rotation หรือ adversarial attack</li>
      <li><strong>Calibration Error:</strong> วัดความมั่นใจของโมเดลหลัง augmentation เช่น Expected Calibration Error (ECE)</li>
    </ul>

    <h3>ตารางเปรียบเทียบการวัดผล</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">Metric</th>
            <th className="border px-4 py-2">คำอธิบาย</th>
            <th className="border px-4 py-2">ใช้เมื่อใด</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Accuracy</td>
            <td className="border px-4 py-2">วัดผลลัพธ์ที่ถูกต้องจากทั้งหมด</td>
            <td className="border px-4 py-2">Base case</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">F1 Score</td>
            <td className="border px-4 py-2">บาลานซ์ระหว่าง precision กับ recall</td>
            <td className="border px-4 py-2">Class imbalance</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ECE</td>
            <td className="border px-4 py-2">วัดความมั่นใจของ prediction เทียบกับความจริง</td>
            <td className="border px-4 py-2">Model calibration</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Insight Box:</p>
      <p>
        งานของ Hendrycks et al. (2020) แสดงให้เห็นว่าโมเดลที่ผ่าน Augmentation แล้วควรได้รับการประเมินบนชุดข้อมูล corrupt และชุด out-of-distribution ด้วย เพื่อทดสอบความสามารถในการ generalize จริง ไม่ใช่เพียงความสามารถบนข้อมูลที่คล้ายกับ training set
      </p>
    </div>

    <h3>ตัวอย่างวิธีประเมินในงานจริง</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ชุดข้อมูลเช่น CIFAR-10-C, ImageNet-C ซึ่งมีการเพิ่ม noise และ distortion</li>
      <li>เทียบ performance บน Clean vs Corrupted dataset เพื่อดูว่า augmentation ช่วยเพิ่ม robustness ได้หรือไม่</li>
      <li>ใช้ reliability diagram ร่วมกับ ECE metric เพื่อดูว่าความมั่นใจของโมเดลเหมาะสมหรือไม่</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Hendrycks, D., & Dietterich, T. (2020). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR.</li>
      <li>Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.</li>
      <li>Stanford CS231n Lecture Notes: Evaluation and Metrics in CV</li>
      <li>Microsoft Research: Robustness testing for deep learning models</li>
    </ul>
  </div>
</section>


      <section id="case-study" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. กรณีศึกษาระดับโลก</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>1. AutoAugment จาก Google Research</h3>
    <p>
      AutoAugment คือเทคนิคการเรียนรู้ policy การแปลงข้อมูลแบบอัตโนมัติที่ใช้ reinforcement learning ในการค้นหาการผสมผสานของ augmentation ที่เหมาะสมกับแต่ละ dataset เช่น CIFAR-10 และ ImageNet โดยสามารถเพิ่ม accuracy ได้อย่างมีนัยสำคัญโดยไม่ต้องปรับเปลี่ยนโครงสร้างของโมเดล
    </p>
    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400">
      <p className="font-semibold">Highlight:</p>
      <p>
        AutoAugment ช่วยให้ ResNet-50 ที่ฝึกบน ImageNet มี accuracy เพิ่มขึ้นถึง 2% โดยไม่เพิ่มพารามิเตอร์หรือเปลี่ยนสถาปัตยกรรมหลักเลย
      </p>
    </div>

    <h3>2. SpecAugment สำหรับงานเสียง (Google Brain)</h3>
    <p>
      งานของ Google Brain ในปี 2019 ได้นำเสนอ SpecAugment ซึ่งเป็นเทคนิคการทำ data augmentation บน spectrogram โดยใช้ time warping, frequency masking และ time masking สามารถปรับปรุงประสิทธิภาพของ speech recognition ได้อย่างมากโดยไม่ต้องเปลี่ยนโมเดล
    </p>

    <h3>3. NLP: Back-translation และ contextual augmentation</h3>
    <p>
      Facebook AI Research และ Stanford แสดงให้เห็นว่า back-translation (การแปลประโยคไป-กลับจากภาษาหนึ่ง) และการใช้ contextual embedding เช่น BERT ในการสร้างประโยคใหม่ สามารถเพิ่มข้อมูล training สำหรับงาน sentiment analysis และ QA ได้อย่างมีประสิทธิภาพ
    </p>
    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400">
      <p className="font-semibold">Insight Box:</p>
      <p>
        Back-translation ช่วยให้ระบบ translation และ sentiment classification สำหรับภาษา low-resource มี performance เทียบเท่ากับระบบภาษาอังกฤษ
      </p>
    </div>

    <h3>4. การใช้ GAN เพื่อสร้างภาพจำลองในวงการแพทย์</h3>
    <p>
      ในงานวิจัยของ Harvard Medical School และ MIT CSAIL ได้ใช้ GAN เพื่อสร้างภาพ MRI ที่หลากหลายจากตัวอย่างที่มีอยู่อย่างจำกัด โดยโมเดลที่ได้รับการฝึกด้วย augmented MRI image เหล่านี้มีประสิทธิภาพในการวินิจฉัยโรคสมองเสื่อมและมะเร็งสมองสูงกว่าระบบที่ฝึกด้วยข้อมูลจริงเพียงอย่างเดียว
    </p>

    <h3>ตารางเปรียบเทียบกรณีศึกษา</h3>
    <div className="overflow-x-auto">
      <table className="w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-700 dark:bg-gray-800 text-white">
          <tr>
            <th className="border px-4 py-2">เทคนิค</th>
            <th className="border px-4 py-2">โดเมน</th>
            <th className="border px-4 py-2">ผลลัพธ์ที่ได้</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">AutoAugment</td>
            <td className="border px-4 py-2">Computer Vision</td>
            <td className="border px-4 py-2">เพิ่ม ImageNet top-1 accuracy 2%</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">SpecAugment</td>
            <td className="border px-4 py-2">Speech Recognition</td>
            <td className="border px-4 py-2">WER ลดลง ~6%</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Back-translation</td>
            <td className="border px-4 py-2">NLP (Low-resource)</td>
            <td className="border px-4 py-2">เพิ่ม accuracy งาน Sentiment/QA</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GAN-MRI</td>
            <td className="border px-4 py-2">Medical Imaging</td>
            <td className="border px-4 py-2">เพิ่ม precision ในการตรวจโรค</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Cubuk et al. (2019). AutoAugment: Learning Augmentation Policies from Data. arXiv:1805.09501</li>
      <li>Park et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. arXiv:1904.08779</li>
      <li>Sennrich et al. (2016). Improving Neural Machine Translation Models with Monolingual Data. ACL</li>
      <li>Frid-Adar et al. (2018). GAN-based Synthetic Medical Image Augmentation. IEEE Transactions on Medical Imaging</li>
    </ul>
  </div>
</section>


<section id="code" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Code Examples (Vision & Text)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>การทำ Data Augmentation ใน Computer Vision ด้วย torchvision</h3>
    <p>
      ไลบรารี <code>torchvision.transforms</code> เป็นหนึ่งในเครื่องมือยอดนิยมสำหรับการทำ augmentation ในงานภาพ โดยสามารถผสมผสานหลายเทคนิคเข้าด้วยกันผ่าน <code>transforms.Compose</code>
    </p>
    <pre><code className="language-python">
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    </code></pre>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400">
      <p className="font-semibold">Highlight:</p>
      <p>
        การใช้ ColorJitter และ RandomRotation ทำให้โมเดลทนต่อแสงและการหมุนในภาพได้ดียิ่งขึ้น โดยเฉพาะในการเรียนรู้แบบ few-shot
      </p>
    </div>

    <h3>การใช้ Albumentations กับ Augmentation ที่ซับซ้อน</h3>
    <p>
      สำหรับงานภาพที่ต้องการความยืดหยุ่นสูง เช่น segmentation หรือ detection การใช้ <code>Albumentations</code> ช่วยให้สามารถจัดการ augmentation ได้หลากหลายและมีประสิทธิภาพ
    </p>
    <pre><code className="language-python">
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
    </code></pre>

    <h3>การทำ Text Augmentation ด้วย nlpaug และ TextAttack</h3>
    <p>
      ในงาน NLP การทำ data augmentation เป็นสิ่งท้าทาย เนื่องจากต้องรักษา semantics เดิมของข้อความ ตัวอย่างเช่นการใช้ <code>nlpaug</code> ร่วมกับ word embeddings เพื่อแทนที่คำบางคำ
    </p>
    <pre><code className="language-python">
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
text = "The quick brown fox jumps over the lazy dog."
augmented_text = aug.augment(text)
print(augmented_text)
    </code></pre>

    <p>
      หรือการใช้ TextAttack เพื่อสร้างชุดข้อมูล adversarial สำหรับ training และ evaluation:
    </p>
    <pre><code className="language-python">
from textattack.augmentation import WordNetAugmenter

augmenter = WordNetAugmenter()
text = "This movie was absolutely wonderful!"
print(augmenter.augment(text))
    </code></pre>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยของ Wei & Zou (2019) แสดงให้เห็นว่า EDA (Easy Data Augmentation) สามารถเพิ่ม accuracy ของ text classifier ได้อย่างมีนัยสำคัญใน dataset ขนาดเล็ก
      </p>
    </div>

    <h3>การใช้ HuggingFace Datasets + Augmentations</h3>
    <p>
      HuggingFace Datasets ช่วยให้สามารถปรับข้อมูลขนาดใหญ่ได้อย่างมีประสิทธิภาพผ่านการ map ด้วยฟังก์ชัน augmentation:
    </p>
  <pre><code className="language-python">{`
from datasets import load_dataset
from textattack.augmentation import EasyDataAugmenter

augmenter = EasyDataAugmenter()
dataset = load_dataset("imdb", split="train")

augmented_dataset = dataset.map(lambda x: {"text": augmenter.augment(x["text"])})
`}</code></pre>


    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Shorten, C. & Khoshgoftaar, T. (2019). A survey on Image Data Augmentation. Journal of Big Data.</li>
      <li>Wei, J. & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. arXiv:1901.11196</li>
      <li><a href="https://huggingface.co/docs" target="_blank" rel="noopener noreferrer">HuggingFace Documentation</a></li>
      <li><a href="https://albumentations.ai/docs/" target="_blank" rel="noopener noreferrer">Albumentations Docs</a></li>
      <li><a href="https://github.com/QData/TextAttack" target="_blank" rel="noopener noreferrer">TextAttack GitHub</a></li>
    </ul>
  </div>
</section>


<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ภาพรวม: บทบาทเชิงลึกของ Data Augmentation ใน AI สมัยใหม่</h3>
    <p>
      ในยุคที่ข้อมูลกลายเป็นเชื้อเพลิงสำคัญของปัญญาประดิษฐ์ การเพิ่มประสิทธิภาพของโมเดลโดยไม่ต้องเก็บข้อมูลเพิ่มเติมถือเป็นแนวทางเชิงกลยุทธ์ Data Augmentation จึงกลายเป็นเครื่องมือหลักที่ไม่ใช่เพียงการเติมข้อมูล แต่เป็นการส่งเสริม <strong>generalization, robustness</strong> และ <strong>fairness</strong> ของระบบปัญญาประดิษฐ์ให้สามารถทำงานในสภาพแวดล้อมจริงที่มีความซับซ้อนได้ดียิ่งขึ้น
    </p>

    <h3>การขับเคลื่อนด้วยงานวิจัยระดับโลก</h3>
    <p>
      งานของ Google Research, Meta AI และทีมจาก Stanford แสดงให้เห็นว่า augmentation ไม่ได้ใช้แค่สำหรับเพิ่ม accuracy แต่ยังมีบทบาทในการลด <em>bias</em>, เพิ่ม <em>resilience</em> ต่อ adversarial attacks และแม้กระทั่งการสร้างโมเดลที่สามารถใช้งานกับภาษาท้องถิ่นหรือ domain-specific ได้ โดยไม่ต้องเริ่มฝึกใหม่จากศูนย์
    </p>

    <div className="bg-yellow-100 text-gray-900 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm">
      <p className="font-semibold">Highlight:</p>
      <p>
        งานของ Hendrycks et al. (2021) แสดงให้เห็นว่าโมเดลที่ได้รับการฝึกด้วย augmentation ที่ออกแบบมาอย่างเหมาะสม มีความสามารถในการ generalize บน dataset ที่ไม่เคยเห็นมาก่อน (Out-of-distribution generalization) ได้ดีกว่าโมเดลที่ฝึกบนข้อมูลเดิมหลายเท่า
      </p>
    </div>

    <h3>แนวโน้มสู่ Augmentation แบบอัตโนมัติ (AutoAugment, RandAugment, TrivialAugment)</h3>
    <p>
      การค้นหาวิธีการ augmentation ที่เหมาะสมในอดีตมักอาศัยการทดลองซ้ำซ้อน แต่ในปัจจุบันมีแนวโน้มเพิ่มขึ้นในการใช้ search-based หรือ meta-learning approach ในการเลือกชุด augmentation แบบอัตโนมัติ ซึ่งเรียกรวมกันว่า <strong>AutoML-based Augmentation</strong> เช่น:
    </p>
    <ul className="list-disc pl-6">
      <li><strong>AutoAugment:</strong> ค้นหาชุด policy โดยใช้ reinforcement learning</li>
      <li><strong>RandAugment:</strong> ลดการค้นหา โดยใช้การสุ่มอย่างชาญฉลาด</li>
      <li><strong>TrivialAugment:</strong> วิธีง่าย ๆ แต่ได้ผลลัพธ์ดีเยี่ยมในหลาย benchmark</li>
    </ul>

    <h3>ผลกระทบต่อ fairness และความเท่าเทียม</h3>
    <p>
      Augmentation ถูกนำไปใช้เพื่อสร้างสมดุลระหว่างกลุ่มข้อมูลที่มีน้อย เช่น เพศ, สีผิว, ภาษา หรือภาวะทางร่างกาย ซึ่งช่วยลด bias และเพิ่ม inclusiveness ให้โมเดล
    </p>

    <div className="bg-blue-100 text-gray-900 p-4 rounded-lg border-l-4 border-blue-400 shadow-sm">
      <p className="font-semibold">Insight:</p>
      <p>
        งานจาก CMU (2020) แสดงให้เห็นว่า augmentation บน minority class โดยใช้ GAN หรือ oversampling-based method ช่วยลด Equal Opportunity Gap ใน task classification บางประเภทได้มากกว่า 30%
      </p>
    </div>

    <h3>แนวโน้มในอนาคต</h3>
    <ul className="list-disc pl-6">
      <li>การผนวก augmentation เข้ากับ pipeline ของ continual learning</li>
      <li>การใช้ generative foundation models (เช่น Diffusion หรือ StableLM) เพื่อสร้างข้อมูลเสมือนที่มีความเหมือนจริงสูง</li>
      <li>การเชื่อมโยง augmentation กับ symbolic reasoning เพื่อช่วยตีความ</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Hendrycks, D. et al. (2021). The Many Faces of Robustness: A Critical Analysis of Out-of-distribution Generalization. arXiv:2006.16241</li>
      <li>Cubuk, E. D. et al. (2019). AutoAugment: Learning Augmentation Strategies from Data. CVPR.</li>
      <li>Touvron, H. et al. (2021). Training Data-efficient Image Transformers & Distillation through Attention. arXiv:2012.12877</li>
      <li>Chen, T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.</li>
      <li>CMU ML Fairness Research Group. (2020). Augmentation for Equity: Bias-aware Oversampling Techniques</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day37 theme={theme} />
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
        <ScrollSpy_Ai_Day37 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day37_DataAugmentation;
