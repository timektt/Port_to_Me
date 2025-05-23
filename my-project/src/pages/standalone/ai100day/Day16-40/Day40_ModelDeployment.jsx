import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day40 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day40";
import MiniQuiz_Day40 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day40";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day40_ModelDeployment = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day40_1").format("auto").quality("auto").resize(scale().width(661));
  const img2 = cld.image("Day40_2").format("auto").quality("auto").resize(scale().width(501));
  const img3 = cld.image("Day40_3").format("auto").quality("auto").resize(scale().width(501));
  const img4 = cld.image("Day40_4").format("auto").quality("auto").resize(scale().width(501));
  const img5 = cld.image("Day40_5").format("auto").quality("auto").resize(scale().width(501));
  const img6 = cld.image("Day40_6").format("auto").quality("auto").resize(scale().width(501));
  const img7 = cld.image("Day40_7").format("auto").quality("auto").resize(scale().width(501));
  const img8 = cld.image("Day40_8").format("auto").quality("auto").resize(scale().width(501));
  const img9 = cld.image("Day40_9").format("auto").quality("auto").resize(scale().width(501));
  const img10 = cld.image("Day40_10").format("auto").quality("auto").resize(scale().width(501));
  const img11 = cld.image("Day40_11").format("auto").quality("auto").resize(scale().width(501));
  const img12 = cld.image("Day40_12").format("auto").quality("auto").resize(scale().width(501));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-700 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20" />
      <div className="">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-6">Day 40: Model Deployment Basics</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

<section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">
    1. บทนำ: ความท้าทายของการ Deploy โมเดล ML
  </h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>บริบทของการ Deploy ในระบบ Machine Learning</h3>
    <p>
      การ Deploy โมเดล Machine Learning ไม่ใช่เพียงการนำโมเดลที่ผ่านการฝึกฝนไปใช้งาน แต่เป็นกระบวนการที่เกี่ยวข้องกับความซับซ้อนของระบบจริง เช่น latency, scalability, monitoring, และ version control โดยเฉพาะอย่างยิ่งเมื่อโมเดลถูกนำไปใช้ในระบบ production ที่ต้องรองรับการใช้งานจริงแบบต่อเนื่องและปลอดภัย
    </p>

    <h3>ความแตกต่างระหว่าง Prototype กับ Production</h3>
    <ul className="list-disc pl-6">
      <li>Prototype ML มักเน้น performance สูงสุดจาก dataset ที่มีอยู่</li>
      <li>Production ML ต้องรองรับ input ที่หลากหลาย มีระบบตรวจสอบ และจัดการ fallback เมื่อโมเดลล้มเหลว</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded-lg">
      <p className="font-semibold">Insight Box</p>
      <p className="mt-2">
        "ในระบบจริง โมเดลไม่สามารถทำงานได้อย่างโดดเดี่ยว จำเป็นต้องผสานกับระบบ infrastructure และ data pipeline" — Stanford ML Systems 2022
      </p>
    </div>

    <h3>ประเด็นหลักที่ทำให้การ Deploy ยากกว่าการฝึกโมเดล</h3>
    <ul className="list-disc pl-6">
      <li><strong>Environment mismatch:</strong> training บนเครื่อง local แต่ deploy บน server ที่ต่างออกไปโดยสิ้นเชิง</li>
      <li><strong>Data drift:</strong> ข้อมูลที่เข้ามาหลัง deployment มี distribution เปลี่ยนแปลง</li>
      <li><strong>Latency requirement:</strong> ในบางระบบ เช่น real-time fraud detection ต้องตอบกลับภายในไม่กี่มิลลิวินาที</li>
      <li><strong>Model update:</strong> การเปลี่ยนเวอร์ชันของโมเดลอาจมีผลต่อ user behavior และ model performance</li>
    </ul>

    <h3>ตารางเปรียบเทียบก่อนและหลังการ Deploy</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300 text-sm">
        <thead className="bg-gray-700 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Aspect</th>
            <th className="border px-4 py-2">Pre-deployment</th>
            <th className="border px-4 py-2">Post-deployment</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Data</td>
            <td className="border px-4 py-2">จัดการแบบ batch, มี label</td>
            <td className="border px-4 py-2">streaming, ไม่มี label จริงทันที</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Evaluation</td>
            <td className="border px-4 py-2">ใช้ test set ที่รู้ผลล่วงหน้า</td>
            <td className="border px-4 py-2">ต้องใช้ proxy metrics และ feedback loop</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Resource</td>
            <td className="border px-4 py-2">ใช้ GPU, เวลาไม่จำกัด</td>
            <td className="border px-4 py-2">จำกัด latency และ memory</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แนวโน้มจากอุตสาหกรรม</h3>
    <p>
      องค์กรระดับโลก เช่น Google, Facebook, และ Netflix ใช้กระบวนการที่เรียกว่า "ML Platform Engineering" เพื่อจัดการ pipeline การ deploy และ monitor โมเดลทั้งหมดผ่านระบบอัตโนมัติ ซึ่งรวมถึง CI/CD, Model Registry, และ Canary Release
    </p>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded-lg">
      <p className="font-semibold">Highlight</p>
      <p className="mt-2">
        การ Deploy โมเดลอย่างมืออาชีพไม่ใช่เรื่องของโค้ดเพียงอย่างเดียว แต่คือเรื่องของระบบทั้งหมด ตั้งแต่ data ถึงผู้ใช้งานจริง
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Google Research. "Hidden Technical Debt in Machine Learning Systems." NIPS 2015</li>
      <li>Stanford CS329S. "ML Systems Design." (2023)</li>
      <li>IEEE Software Engineering Journal, "Best Practices for Deploying Machine Learning at Scale"</li>
      <li>Uber Engineering Blog: "Michelangelo ML Platform"</li>
    </ul>
  </div>
</section>


 <section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. ประเภทของการ Deploy โมเดล</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>การจำแนกประเภท Deployment ตามสภาพแวดล้อมการใช้งาน</h3>
    <p>
      การเลือกประเภทของการ Deploy โมเดลไม่ใช่เพียงประเด็นทางเทคนิค แต่มีผลต่อ latency, scalability และความปลอดภัยของระบบโดยรวม งานวิจัยจาก Harvard และ Google AI Infra ชี้ให้เห็นว่าความล้มเหลวในการเลือกกลยุทธ์ deployment ที่เหมาะสม อาจส่งผลให้เกิดค่าใช้จ่ายสูงโดยไม่จำเป็น หรือเกิด model degradation ใน production environment ได้
    </p>

    <ul className="list-disc pl-6">
      <li><strong>Batch Deployment:</strong> เหมาะกับงานที่ไม่ต้องตอบสนองทันที เช่น การแนะนำสินค้ารายวัน</li>
      <li><strong>Real-time Deployment:</strong> ใช้ในระบบที่ต้องตอบกลับภายในไม่กี่มิลลิวินาที เช่น fraud detection</li>
      <li><strong>Edge Deployment:</strong> ใช้กับอุปกรณ์ปลายทาง เช่น IoT และมือถือ โดยโมเดลรันในอุปกรณ์</li>
    </ul>

    <h3>รูปแบบการ Deploy ตามสถาปัตยกรรมระบบ</h3>
    <div className="overflow-x-auto">
  <table className="table-auto min-w-[700px] w-full text-sm border border-gray-300 dark:border-gray-600">
    <thead className="bg-gray-200 dark:bg-gray-700">
      <tr>
        <th className="border px-4 py-2 text-left">รูปแบบ</th>
        <th className="border px-4 py-2 text-left">คำอธิบาย</th>
        <th className="border px-4 py-2 text-left">ข้อดี</th>
        <th className="border px-4 py-2 text-left">ข้อจำกัด</th>
      </tr>
    </thead>
    <tbody className="bg-white dark:bg-gray-900">
      <tr>
        <td className="border px-4 py-2">Monolithic</td>
        <td className="border px-4 py-2">Deploy พร้อมกับ application ทั้งชุด</td>
        <td className="border px-4 py-2">ง่ายต่อการจัดการ</td>
        <td className="border px-4 py-2">ไม่ยืดหยุ่นสำหรับ scaling</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Microservices</td>
        <td className="border px-4 py-2">โมเดลแยกเป็นบริการเฉพาะ</td>
        <td className="border px-4 py-2">รองรับ scaling, dev/ops แยก</td>
        <td className="border px-4 py-2">ซับซ้อนกว่าในการ orchestration</td>
      </tr>
      <tr>
        <td className="border px-4 py-2">Serverless</td>
        <td className="border px-4 py-2">รันเฉพาะเมื่อมี request เช่น AWS Lambda</td>
        <td className="border px-4 py-2">ประหยัดทรัพยากร</td>
        <td className="border px-4 py-2">ไม่เหมาะกับ workload ที่ต้องใช้ทรัพยากรสูง</td>
      </tr>
    </tbody>
  </table>
</div>


    <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 rounded shadow-inner">
      <p className="font-semibold">Highlight:</p>
      <p className="mt-2">
        ในงานของ Google Brain (2022) การเลือกใช้ microservice-based model deployment ทำให้สามารถ update โมเดลใหม่โดยไม่ต้องหยุดระบบหลัก และลด latency ได้ถึง 35% เมื่อเทียบกับ monolithic architecture
      </p>
    </div>

    <h3>ข้อควรพิจารณาในการเลือกกลยุทธ์ Deployment</h3>
    <ul className="list-disc pl-6">
      <li>Volume ของ request และ latency ที่ต้องการ</li>
      <li>โครงสร้างของทีมงาน (DevOps, MLOps พร้อมหรือไม่)</li>
      <li>งบประมาณและความสามารถในการ scale</li>
      <li>ข้อกำหนดด้านความปลอดภัย และการควบคุมข้อมูล</li>
    </ul>

    <div className="bg-blue-700 border-l-4 border-blue-500 p-4 rounded shadow-inner">
      <p className="font-semibold">Insight Box:</p>
      <p className="mt-2">
        CMU และ Stanford แนะนำให้ทีม ML ประเมินความต้องการด้าน deployment ตั้งแต่เริ่มออกแบบระบบ เพื่อหลีกเลี่ยงการ refactor ขนานใหญ่ในภายหลัง
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>Google Research, "Effective ML Infrastructure" (2022)</li>
      <li>CMU ML Systems Lab: Deployment Architectures for Scalable ML (2021)</li>
      <li>Stanford CS329S: Machine Learning Systems Design</li>
      <li>IEEE Access, "Taxonomy of ML Deployment Strategies" (2023)</li>
    </ul>
  </div>
</section>


 <section id="export" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. การ Export โมเดลอย่างปลอดภัย</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none text-base leading-relaxed space-y-10">
    <h3>แนวคิดเบื้องต้นของการ Export โมเดล</h3>
    <p>
      การ export โมเดลหมายถึงการบันทึกโมเดลที่ผ่านการฝึกแล้วในรูปแบบที่สามารถนำไปใช้งานในภายหลัง ไม่ว่าจะเป็นเพื่อการ inference, deployment, หรือการแลกเปลี่ยนระหว่างระบบ โดยกระบวนการนี้ต้องคำนึงถึงความปลอดภัย ความเข้ากันได้ (compatibility) และความสามารถในการตรวจสอบเวอร์ชัน
    </p>

    <h3>ฟอร์แมตที่นิยมในการ Export</h3>
    <ul className="list-disc pl-6">
      <li><strong>Pickle (.pkl):</strong> ใช้ใน Python แต่มีข้อกังวลเรื่องความปลอดภัย</li>
      <li><strong>ONNX (Open Neural Network Exchange):</strong> ใช้ร่วมกันได้ระหว่าง PyTorch, TensorFlow และระบบ inference อื่น ๆ</li>
      <li><strong>SavedModel (TensorFlow):</strong> รองรับ deployment บน TensorFlow Serving และ TensorFlow Lite</li>
      <li><strong>TorchScript:</strong> รองรับการ deploy ใน PyTorch Mobile หรือ C++ runtime</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded-md border-l-4 border-yellow-400">
      <p className="font-semibold">Insight Box:</p>
      <p>
        จากเอกสารของ Stanford ML Systems Lab ระบุว่า ONNX กลายเป็นมาตรฐานกลางที่นิยมในองค์กรระดับโลกเนื่องจากสามารถเชื่อมโยงได้กับระบบ inference หลากหลายชนิดทั้ง CPU, GPU และ edge device
      </p>
    </div>

    <h3>การจัดการกับ Parameter และ Hyperparameter</h3>
    <p>
      เมื่อ export โมเดล ควรรวมข้อมูลของ hyperparameters และ configuration ทั้งหมดเพื่อการ reproducibility ที่สมบูรณ์ อาจทำได้โดยการแนบไฟล์ YAML หรือ JSON ควบคู่กับ binary model file
    </p>

    <h3>เทคนิคการ Sign และตรวจสอบไฟล์โมเดล</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ Hash (SHA256) สำหรับตรวจสอบความถูกต้องของไฟล์</li>
      <li>ใช้ digital signature สำหรับยืนยันความน่าเชื่อถือของ source</li>
      <li>บันทึก metadata เช่น version, training dataset, training time ลงในโมเดลด้วย</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded-md border-l-4 border-blue-500">
      <p className="font-semibold">Highlight:</p>
      <p>
        ความผิดพลาดที่พบบ่อยใน production system คือการ deploy โมเดลผิดเวอร์ชันหรือขาดไฟล์ configuration ที่เกี่ยวข้อง ซึ่งสามารถป้องกันได้ด้วยระบบ versioning และการ audit log ที่ดี
      </p>
    </div>

    <h3>ตัวอย่างการ Export ด้วย PyTorch และ ONNX</h3>
    <pre><code className="language-python">import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

# สร้าง dummy input
x = torch.randn(1, 3, 224, 224)

# export เป็น ONNX
torch.onnx.export(model, x, "resnet18.onnx", export_params=True, opset_version=11)</code></pre>

    <h3>แนวทางจากองค์กรระดับโลก</h3>
    <ul className="list-disc pl-6">
      <li>Meta AI: ใช้ TorchScript + ONNX สำหรับ deploy ผ่าน Caffe2 runtime</li>
      <li>Google: ใช้ TensorFlow SavedModel บน TensorFlow Serving และ TFLite</li>
      <li>Microsoft: สนับสนุน ONNX เป็นมาตรฐาน deployment</li>
    </ul>

    <h3>สรุป</h3>
    <p>
      การ export โมเดลเป็นมากกว่าการบันทึกไฟล์น้ำหนักโมเดล แต่เป็นกระบวนการที่ต้องรองรับความปลอดภัย ความสามารถในการติดตามเวอร์ชัน และการนำไปใช้งานในระบบที่แตกต่างอย่างมีประสิทธิภาพ
    </p>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford ML Systems Lab. (2022). Exporting Models for Production.</li>
      <li>PyTorch Official Docs: https://pytorch.org/tutorials/</li>
      <li>ONNX Documentation: https://onnx.ai/</li>
      <li>TensorFlow SavedModel Guide: https://www.tensorflow.org/guide/saved_model</li>
    </ul>
  </div>
</section>

 <section id="api" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. การสร้าง Model API (Serve/Wrap)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>บทนำ: ความจำเป็นของ Model API</h3>
    <p>
      ในบริบทของการนำโมเดล Machine Learning ไปใช้งานจริง จำเป็นต้องแปลงโมเดลให้สามารถตอบสนองต่อคำขอจากผู้ใช้งานหรือระบบอื่น ๆ ได้อย่างยืดหยุ่น โดยทั่วไปจะทำผ่านการสร้าง API ที่สามารถรับอินพุต ประมวลผลด้วยโมเดล และส่งผลลัพธ์กลับมาได้ในรูปแบบ JSON หรือรูปแบบที่ระบบ downstream เข้าใจ
    </p>

    <h3>หลักการพื้นฐานของการ Serve โมเดล</h3>
    <ul className="list-disc pl-6">
      <li>โหลดโมเดลจากไฟล์ที่ถูก export มาแล้ว (เช่น .pkl, .pt, .onnx)</li>
      <li>รับข้อมูลอินพุตผ่าน HTTP request (มักใช้ POST)</li>
      <li>ทำ preprocessing ข้อมูลเพื่อให้สอดคล้องกับโมเดล</li>
      <li>เรียกโมเดลเพื่อพยากรณ์ (prediction)</li>
      <li>จัดรูปผลลัพธ์ให้อยู่ในรูปแบบที่ client เข้าใจ</li>
    </ul>

    <h3>เครื่องมือยอดนิยมในการสร้าง Model API</h3>
    <table className="table-auto border w-full text-sm">
      <thead className="bg-gray-700 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">เครื่องมือ</th>
          <th className="border px-4 py-2">จุดเด่น</th>
          <th className="border px-4 py-2">ข้อควรระวัง</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">FastAPI</td>
          <td className="border px-4 py-2">เร็ว ใช้ง่าย รองรับ async</td>
          <td className="border px-4 py-2">ต้องจัดการด้วยตัวเองเรื่อง scaling</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Flask</td>
          <td className="border px-4 py-2">เป็นที่รู้จักกว้าง ใช้งานง่าย</td>
          <td className="border px-4 py-2">ไม่เหมาะกับ concurrent requests จำนวนมาก</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">TorchServe</td>
          <td className="border px-4 py-2">ออกแบบมาเฉพาะสำหรับ PyTorch</td>
          <td className="border px-4 py-2">ซับซ้อนสำหรับผู้เริ่มต้น</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        การเลือกเครื่องมือ serve โมเดลควรขึ้นอยู่กับระดับความซับซ้อนของระบบ เช่น หากเป็น prototype อาจใช้ FastAPI แต่หากต้องการระบบที่ scale ได้อัตโนมัติควรพิจารณาใช้ TensorFlow Serving หรือ BentoML
      </p>
    </div>

    <h3>ตัวอย่างโค้ดการสร้าง API ด้วย FastAPI</h3>
 <pre>
  <code className="language-python">
{`from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"result": prediction.tolist()"}`}
  </code>
</pre>


    <h3>แนวทางการป้องกัน API จากปัญหาความปลอดภัย</h3>
    <ul className="list-disc pl-6">
      <li>ควรใช้ HTTPS สำหรับการสื่อสาร</li>
      <li>จำกัดขนาด payload เพื่อป้องกันการโจมตีแบบ DoS</li>
      <li>ใช้ token authentication เช่น JWT สำหรับ client แต่ละราย</li>
    </ul>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        อย่าลืมทดสอบ API ด้วยข้อมูลที่ผิดรูปแบบ (malformed) เพื่อให้แน่ใจว่าโมเดลสามารถปฏิเสธอินพุตที่ไม่ปลอดภัยได้
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Stanford CS329s: Machine Learning Systems Design</li>
      <li>MIT Lecture: Deploying ML at Scale, 2022</li>
      <li>arXiv:2007.09373 - A Survey on Model Serving Systems</li>
      <li>FastAPI Documentation: https://fastapi.tiangolo.com</li>
      <li>BentoML Whitepaper: https://www.bentoml.com</li>
    </ul>
  </div>
</section>


  <section id="infra" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. การเลือก Infrastructure</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>

  <div className="prose prose-lg max-w-none dark:prose-invert space-y-10 text-base leading-relaxed">
    <h3>บทนำ: ความเชื่อมโยงระหว่างโมเดลกับ Infrastructure</h3>
    <p>
      การเลือก Infrastructure ที่เหมาะสมสำหรับการ Deploy โมเดล Machine Learning มีผลโดยตรงต่อ latency, reliability, scalability และค่าใช้จ่ายของระบบทั้งหมด โดยเฉพาะเมื่อโมเดลต้องให้บริการในสเกลที่ใหญ่ หรือรองรับ real-time inference จึงจำเป็นต้องพิจารณาโครงสร้างพื้นฐานอย่างรอบคอบ
    </p>

    <div className="bg-yellow-700 p-4 rounded border-l-4 border-yellow-500">
      <p className="font-semibold">Insight:</p>
      <p>
        งานวิจัยจาก MIT CSAIL (2023) พบว่าโครงการ ML มากกว่า 50% ที่ล้มเหลวใน production เกิดจากปัญหา latency, cost หรือการเลือก infrastructure ที่ไม่ตรงกับ use case
      </p>
    </div>

    <h3>รูปแบบของ Infrastructure สำหรับ ML Deployment</h3>
    <ul className="list-disc pl-6">
      <li><strong>On-premise:</strong> เหมาะสำหรับองค์กรที่ต้องการควบคุมความปลอดภัยเต็มรูปแบบ เช่น สายการเงินหรือการทหาร</li>
      <li><strong>Cloud-based:</strong> เช่น AWS SageMaker, GCP Vertex AI, Azure ML รองรับการ scale อัตโนมัติ</li>
      <li><strong>Hybrid:</strong> ใช้ cloud เป็น staging/testing และ on-premise สำหรับ production</li>
      <li><strong>Edge Deployment:</strong> สำหรับ inference ที่ต้องการ latency ต่ำ เช่น IoT หรือ Mobile Device</li>
    </ul>

    <h3>การเปรียบเทียบ Platform ชั้นนำ</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-200 dark:bg-gray-800">
          <tr>
            <th className="px-4 py-2 text-left">Provider</th>
            <th className="px-4 py-2 text-left">ข้อดี</th>
            <th className="px-4 py-2 text-left">ข้อจำกัด</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900">
          <tr>
            <td className="border px-4 py-2">AWS SageMaker</td>
            <td className="border px-4 py-2">เครื่องมือครบ ตั้งแต่ train → deploy</td>
            <td className="border px-4 py-2">ค่าใช้จ่ายอาจสูงสำหรับงาน inference ขนาดใหญ่</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">GCP Vertex AI</td>
            <td className="border px-4 py-2">เชื่อมกับ BigQuery และ TFX ได้ดี</td>
            <td className="border px-4 py-2">ยังมีฟีเจอร์น้อยกว่าบางคู่แข่ง</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Azure ML</td>
            <td className="border px-4 py-2">เหมาะกับ enterprise integration และ PowerBI</td>
            <td className="border px-4 py-2">เครื่องมือบางอย่างซับซ้อนในการตั้งค่า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3>แนวทางการเลือก Infrastructure ให้เหมาะกับ Use Case</h3>
    <ul className="list-disc pl-6">
      <li>หากต้องการ real-time → ใช้ FastAPI + Kubernetes บน cloud</li>
      <li>หาก workload สูงแต่ไม่เร่งด่วน → ใช้ batch inference ผ่าน cloud storage</li>
      <li>สำหรับ IoT → ใช้ TensorFlow Lite หรือ ONNX บน edge device</li>
      <li>สำหรับระบบขนาดเล็กหรือ PoC → ใช้ cloud function เช่น AWS Lambda</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded border-l-4 border-blue-500">
      <p className="font-semibold">Highlight:</p>
      <p>
        Infrastructure ที่ดีไม่ใช่เพียงแค่ “เร็ว” หรือ “ถูก” แต่ต้อง align กับ workflow, ข้อจำกัดของทีม และข้อกำหนดขององค์กร เช่น compliance หรือ SLA
      </p>
    </div>

    <h3>ข้อควรระวัง</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ cloud เพียงเพื่อ “ความง่าย” อาจก่อให้เกิด vendor lock-in</li>
      <li>ค่าใช้จ่ายต้องประเมินตาม workload จริง และกำหนด budget monitoring</li>
      <li>ควรมีระบบ autoscaling หรือ fallback สำหรับระบบที่ต้อง always-on</li>
    </ul>

    <h3>แหล่งอ้างอิงทางวิชาการและอุตสาหกรรม</h3>
    <ul className="list-disc pl-6">
      <li>MIT CSAIL (2023). Infrastructure Pitfalls in ML Deployment</li>
      <li>Google Cloud ML Whitepaper, 2022</li>
      <li>“Practical MLOps”, Noah Gift, O’Reilly Media</li>
      <li>Stanford CS329S: Reliable ML Systems</li>
    </ul>
  </div>
</section>


  <section id="cicd" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. CI/CD สำหรับ Machine Learning</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความสำคัญของ CI/CD ในระบบ Machine Learning</h3>
    <p>
      แนวคิดของ Continuous Integration (CI) และ Continuous Deployment (CD) ในระบบ Machine Learning ไม่ใช่แค่การ deploy โมเดลเท่านั้น
      แต่รวมถึงการบริหารจัดการข้อมูล, การเทรนซ้ำ, การ validate และการตรวจสอบ performance อย่างต่อเนื่องเพื่อให้ระบบมีความเสถียรใน production environment
    </p>

    <div className="bg-yellow-700 p-4 rounded border-l-4 border-yellow-500">
      <p className="font-semibold">Insight Box</p>
      <p>
        งานวิจัยจาก Google Research ("Hidden Technical Debt in Machine Learning Systems") ระบุว่า ML systems มีความเสี่ยงในการสะสมความซับซ้อนระยะยาว (technical debt)
        CI/CD จึงเป็นแนวทางหลักในการควบคุมความซับซ้อนนี้ในระดับ production
      </p>
    </div>

    <h3>องค์ประกอบของ ML CI/CD Pipeline</h3>
    <ul className="list-disc pl-6">
      <li><strong>Data Validation:</strong> ตรวจสอบความสมบูรณ์และ drift ของข้อมูล</li>
      <li><strong>Model Training:</strong> รัน pipeline เทรนโมเดลใหม่จาก code/data เวอร์ชันล่าสุด</li>
      <li><strong>Model Evaluation:</strong> ตรวจสอบ performance ก่อน deploy</li>
      <li><strong>Model Registry:</strong> บันทึกเวอร์ชันของโมเดลและ metadata ทั้งหมด</li>
      <li><strong>Automated Deployment:</strong> อัปเดตระบบโดยอัตโนมัติผ่าน container หรือ API</li>
    </ul>

    <h3>เครื่องมือยอดนิยมใน ML CI/CD</h3>
    <table className="table-auto w-full text-sm border">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">เครื่องมือ</th>
          <th className="border px-4 py-2">บทบาท</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">MLflow</td>
          <td className="border px-4 py-2">จัดการ experiment, model versioning และ deployment</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">GitHub Actions</td>
          <td className="border px-4 py-2">ทำ CI pipeline อัตโนมัติหลังมีการ push code</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">DVC</td>
          <td className="border px-4 py-2">จัดการเวอร์ชันของข้อมูลและ pipeline</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Kubeflow</td>
          <td className="border px-4 py-2">จัดการ workflow และ orchestrate ML pipeline บน Kubernetes</td>
        </tr>
      </tbody>
    </table>

    <h3>ตัวอย่าง CI/CD Workflow สำหรับ ML</h3>
    <pre>
      <code className="language-yaml">{`name: ml-ci-pipeline

on:
  push:
    branches: [main]

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repo
        uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Run Model Training
        run: python train.py

      - name: Deploy to API
        run: bash deploy.sh
`}</code>
    </pre>

    <h3>ปัญหาทั่วไปและวิธีรับมือ</h3>
    <ul className="list-disc pl-6">
      <li>Drift ในข้อมูลที่ไม่ถูกตรวจจับ → ใช้ระบบ data validation เช่น Great Expectations</li>
      <li>Performance ลดหลัง deploy → ใช้ shadow testing หรือ canary deployment</li>
      <li>ระบบ training ที่ไม่ reproducible → ใช้ DVC และ environment pinning</li>
    </ul>

    <div className="bg-blue-700 p-4 rounded border-l-4 border-blue-500">
      <p className="font-semibold">Highlight Box</p>
      <p>
        ในระบบ ML ที่มีการ deploy บ่อย เช่น fraud detection หรือ recommendation ควรมีระบบ rollback model อัตโนมัติเมื่อ performance ด้อยกว่าค่า baseline ที่ตั้งไว้
      </p>
    </div>

    <h3>แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc pl-6">
      <li>Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NIPS 2015</li>
      <li>Stanford CS329S, "Reliable Machine Learning Systems", Lecture 7: CI/CD for ML</li>
      <li>MLflow Documentation: https://mlflow.org/docs/latest/</li>
      <li>O'Reilly: "Introducing MLOps" by Mark Treveil et al.</li>
    </ul>
  </div>
</section>


 <section id="monitoring" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. การตั้งระบบ Monitoring</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความสำคัญของ Monitoring ในระบบ Machine Learning</h3>
    <p>
      การ deploy โมเดล Machine Learning ไปยัง production environment ไม่ได้หมายความว่างานเสร็จสมบูรณ์ หากขาดระบบ monitoring ที่มีประสิทธิภาพ ความผิดพลาดเล็กน้อยใน input data, model prediction หรือ infrastructure อาจนำไปสู่ผลกระทบขนาดใหญ่ในระบบ downstream และการตัดสินใจเชิงธุรกิจ
    </p>

    <div className="bg-yellow-700 p-4 rounded border-l-4 border-yellow-500">
      <p className="font-semibold">Insight Box</p>
      <p>
        งานวิจัยจาก Google Research ระบุว่า "90% ของปัญหาที่เกิดกับระบบ AI production ไม่ได้มาจากโมเดลเอง แต่เกิดจากการเปลี่ยนแปลงของข้อมูลและ infrastructure ที่ไม่ได้ถูก monitor" (Sculley et al., NIPS 2015)
      </p>
    </div>

    <h3>องค์ประกอบของระบบ Monitoring ที่มีประสิทธิภาพ</h3>
    <ul className="list-disc pl-6">
      <li><strong>Data Drift Detection:</strong> ตรวจสอบการเปลี่ยนแปลงของ distribution ใน input features</li>
      <li><strong>Prediction Drift:</strong> วิเคราะห์การเปลี่ยนแปลงของ output distribution เช่น confidence score</li>
      <li><strong>Model Performance Monitoring:</strong> ติดตาม metrics เช่น accuracy, precision, recall หากมี label</li>
      <li><strong>Latency & Throughput:</strong> วิเคราะห์ response time และ load ของโมเดล</li>
      <li><strong>Anomaly Detection:</strong> ใช้ statistical test หรือ model-based approach ตรวจจับเหตุการณ์ผิดปกติ</li>
    </ul>

    <h3>เครื่องมือสำหรับ Model Monitoring</h3>
    <table className="table-auto w-full border text-sm">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">เครื่องมือ</th>
          <th className="border px-4 py-2">คุณสมบัติเด่น</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Evidently AI</td>
          <td className="border px-4 py-2">dashboard, drift detection, report generator</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Fiddler AI</td>
          <td className="border px-4 py-2">bias detection, performance & explainability monitoring</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Prometheus + Grafana</td>
          <td className="border px-4 py-2">metrics-level monitoring และ visualization</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Arize AI</td>
          <td className="border px-4 py-2">real-time drift & model observability</td>
        </tr>
      </tbody>
    </table>

    <h3>ตัวอย่าง Code สำหรับ Monitor Drift</h3>
    <pre>
      <code className="language-python">
import evidently
from evidently.test_suite import TestSuite
from evidently.tests import TestDataDrift

test_suite = TestSuite(tests=[TestDataDrift()])
test_suite.run(reference_data=train_df, current_data=prod_df)
test_suite.save_html("drift_report.html")
      </code>
    </pre>

    <div className="bg-blue-700 p-4 rounded border-l-4 border-blue-500">
      <p className="font-semibold">Highlight Box</p>
      <p>
        การมีระบบ monitor ที่ตรวจจับ drift และ anomaly แบบอัตโนมัติจะช่วยให้สามารถ rollback model หรือ retrain ได้ทันท่วงที ลดความเสียหายจากการตัดสินใจผิดพลาดได้อย่างมีนัยสำคัญ
      </p>
    </div>

    <h3>แนวทางจากองค์กรชั้นนำ</h3>
    <ul className="list-disc pl-6">
      <li>Stanford แนะนำให้แยก monitoring layer ออกจาก model layer อย่างชัดเจน</li>
      <li>MIT แนะนำการสร้าง custom metric ที่สอดคล้องกับ business outcome</li>
      <li>Google แนะนำให้มี shadow logging สำหรับวิเคราะห์ performance แบบ non-intrusive</li>
    </ul>

    <h3>แหล่งอ้างอิงทางวิชาการ</h3>
    <ul className="list-disc pl-6">
      <li>Sculley et al. "Hidden Technical Debt in ML Systems", NIPS 2015</li>
      <li>Stanford CS329S Lecture Notes: "Production AI Systems"</li>
      <li>arXiv:2201.12372 — "A Survey on Drift Detection in ML Systems"</li>
      <li>Evidently AI Open Source Docs: https://evidentlyai.com</li>
    </ul>
  </div>
</section>


<section id="versioning" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. การจัดการเวอร์ชันของโมเดล (Model Registry)</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความสำคัญของ Model Versioning ในระบบ Production</h3>
    <p>
      ในระบบ Machine Learning ขนาดใหญ่ที่ต้อง deploy หลายเวอร์ชันของโมเดลพร้อมกัน เช่น A/B testing, rollback หรือ multi-tenant inference — การจัดการเวอร์ชันของโมเดล (Model Versioning) จึงเป็นกระบวนการสำคัญที่ทำให้ระบบสามารถติดตาม ตรวจสอบ และควบคุมได้อย่างมีเสถียรภาพ
    </p>

    <h3>แนวคิดของ Model Registry</h3>
    <p>
      Model Registry คือระบบที่ใช้บันทึกและติดตามเวอร์ชันของโมเดล พร้อม metadata ที่เกี่ยวข้อง เช่น hyperparameters, metrics, dataset ที่ใช้ในการฝึก, เวลาที่สร้าง, และสถานะการใช้งาน (staging, production)
    </p>

    <div className="bg-blue-700 p-4 rounded border-l-4 border-blue-400">
      <p className="font-semibold">Highlight Box:</p>
      <p className="mt-2">ระบบ Model Registry ที่ดีต้องสามารถทำ version promotion, access control และรองรับ integration กับ CI/CD pipeline ได้โดยอัตโนมัติ</p>
    </div>

    <h3>องค์ประกอบหลักของ Model Registry</h3>
    <table className="table-auto w-full border text-sm mt-6">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">องค์ประกอบ</th>
          <th className="border px-4 py-2">คำอธิบาย</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Model Version</td>
          <td className="border px-4 py-2">เลขเวอร์ชันของโมเดล เช่น v1.0, v2.3</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Metadata</td>
          <td className="border px-4 py-2">รายละเอียด เช่น dataset, code commit, training time</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Stage</td>
          <td className="border px-4 py-2">ระบุสถานะ เช่น staging, production, archived</td>
        </tr>
      </tbody>
    </table>

    <h3>ตัวอย่าง Framework ที่รองรับ Model Registry</h3>
    <ul className="list-disc pl-6">
      <li><strong>MLflow Model Registry:</strong> มี API สำหรับการ register, transition และ query โมเดล</li>
      <li><strong>SageMaker Model Registry:</strong> มีการเชื่อมต่อกับ pipeline ของ AWS ได้โดยตรง</li>
      <li><strong>Vertex AI:</strong> รองรับ versioning และ monitoring แบบ integrated</li>
    </ul>

    <div className="bg-yellow-700 p-4 rounded border-l-4 border-yellow-400">
      <p className="font-semibold">Insight Box:</p>
      <p className="mt-2">
        จากงานวิจัยของ MIT CSAIL (2021) พบว่า 72% ของ production bugs ในระบบ ML เกิดจากการ deploy โมเดลผิดเวอร์ชัน หรือไม่ได้ควบคุมการเปลี่ยนแปลงอย่างเหมาะสม
      </p>
    </div>

    <h3>แนวทางปฏิบัติที่แนะนำ</h3>
    <ul className="list-disc pl-6">
      <li>ควรใช้ระบบที่สามารถเชื่อมกับ CI/CD pipeline ได้โดยตรง</li>
      <li>บันทึก metadata ทุกครั้งที่มีการเทรนใหม่</li>
      <li>ตั้ง role-based access control (RBAC) เพื่อความปลอดภัย</li>
      <li>ใช้ระบบ approval flow เมื่อจะ promote จาก staging → production</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>MLflow Documentation: https://mlflow.org/docs/latest/model-registry.html</li>
      <li>MIT CSAIL: "Reliable Machine Learning Systems", 2021</li>
      <li>Google Vertex AI Model Management: Official Docs</li>
      <li>IEEE Software Engineering for Machine Learning, 2022</li>
    </ul>
  </div>
</section>


<section id="best" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Best Practices</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>ความจำเป็นของแนวทางปฏิบัติที่ดีในงาน Deployment</h3>
    <p>
      การนำโมเดล Machine Learning ไปใช้งานจริง (deployment) เป็นขั้นตอนที่มีความซับซ้อน และมีความแตกต่างจากการฝึกโมเดลในสภาพแวดล้อมของนักพัฒนา การมีแนวปฏิบัติที่ดี (Best Practices) จะช่วยลดข้อผิดพลาดในการใช้งานจริง เพิ่มความเสถียร และทำให้ระบบสามารถปรับตัวต่อข้อมูลและสถานการณ์ที่เปลี่ยนแปลงได้
    </p>

    <h3>แนวปฏิบัติในระดับสถาปัตยกรรม</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ containerization (เช่น Docker) เพื่อให้การ deploy โมเดลสามารถทำได้อย่างสม่ำเสมอและควบคุมสภาพแวดล้อมได้</li>
      <li>วางระบบแยกชั้นอย่างชัดเจนระหว่าง data ingestion, preprocessing, inference และ postprocessing</li>
      <li>ใช้ gateway API หรือ proxy layer เพื่อจัดการ traffic และ logging ได้ง่ายขึ้น</li>
    </ul>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        การแยกหน้าที่ของ component อย่างชัดเจน (Separation of Concerns) จะช่วยให้สามารถ debug และปรับเปลี่ยนแต่ละส่วนของระบบได้โดยไม่กระทบส่วนอื่น
      </p>
    </div>

    <h3>แนวปฏิบัติในระดับกระบวนการ</h3>
    <ul className="list-disc pl-6">
      <li>ตั้ง test suite ที่ครอบคลุมทุก layer ของ pipeline ตั้งแต่ data preprocessing ไปจนถึง output validation</li>
      <li>ใช้ CI/CD pipeline เพื่อตรวจสอบการเปลี่ยนแปลงของโค้ดและโมเดลก่อน deploy ทุกครั้ง</li>
      <li>สร้างระบบ monitoring ที่ติดตามทั้ง error rate และ performance metric</li>
    </ul>

    <h3>แนวปฏิบัติด้านความปลอดภัยและการปกป้องโมเดล</h3>
    <ul className="list-disc pl-6">
      <li>ทำ model obfuscation หรือ quantization เพื่อลดโอกาสถูกขโมยโมเดล (model stealing)</li>
      <li>จำกัดการเรียกใช้งาน API ด้วย rate limiting และ authentication</li>
      <li>ตั้งระบบ audit log เพื่อตรวจสอบและย้อนรอยการใช้งานโมเดล</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box</p>
      <p>
        งานวิจัยจาก MIT CSAIL (2022) ระบุว่าโมเดลที่ไม่ถูกป้องกันอย่างเหมาะสมสามารถถูก reverse-engineer ได้ในเวลาไม่ถึง 1 ชั่วโมงผ่าน API ที่เปิดสาธารณะ
      </p>
    </div>

    <h3>ตัวอย่าง Best Practices จากองค์กรระดับโลก</h3>
    <table className="table-auto w-full border border-gray-300 text-sm">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border px-4 py-2">องค์กร</th>
          <th className="border px-4 py-2">แนวทางเด่น</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Google</td>
          <td className="border px-4 py-2">ใช้ TFX สำหรับ pipeline อัตโนมัติและ Kubeflow สำหรับ orchestration</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Netflix</td>
          <td className="border px-4 py-2">ใช้ CI/CD และ Canary Deployments สำหรับ A/B testing อย่างต่อเนื่อง</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Microsoft</td>
          <td className="border px-4 py-2">มี model governance framework ที่ควบคุม version, audit และ compliance</td>
        </tr>
      </tbody>
    </table>

    <h3>สรุป Best Practices สำคัญ</h3>
    <ul className="list-disc pl-6">
      <li>ใช้ระบบเวอร์ชันและ logging เสมอ</li>
      <li>หลีกเลี่ยง hardcoding configuration ในโค้ด</li>
      <li>ใช้ระบบ alert และ dashboard ตรวจสอบ performance</li>
      <li>ประเมิน risk ก่อน deploy เสมอ โดยเฉพาะในระบบที่เกี่ยวข้องกับความปลอดภัยหรือการเงิน</li>
    </ul>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6 text-sm">
      <li>"Reliable Machine Learning" by Chip Huyen, Stanford CS329S, 2023</li>
      <li>"Hidden Technical Debt in Machine Learning Systems" - Google Research, NIPS 2015</li>
      <li>MIT CSAIL - Practical Threats and Security in ML Deployment, 2022</li>
      <li>IEEE Transactions on Software Engineering, 2021: "CI/CD for Machine Learning Applications"</li>
    </ul>
  </div>
</section>


<section id="case" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Case Studies</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-10 text-base leading-relaxed">
    <h3>กรณีศึกษาที่ 1: ระบบ Recommendation ของ Netflix</h3>
    <p>Netflix เป็นหนึ่งในองค์กรที่ประสบความสำเร็จในการ Deploy ระบบ Machine Learning สู่ production ขนาดใหญ่ โดยระบบ recommendation ของ Netflix ต้องรองรับผู้ใช้หลายร้อยล้านรายทั่วโลกพร้อมกัน ทำให้การ deploy ต้องใช้ MLOps infrastructure ที่มีความซับซ้อนสูงและปรับตัวได้แบบ dynamic.</p>
    <ul className="list-disc pl-6">
      <li>ใช้ TensorFlow Extended (TFX) สำหรับจัดการ pipeline</li>
      <li>ใช้ระบบ Feature Store เพื่อให้สามารถ reuse ข้อมูลได้ระหว่างโมเดล</li>
      <li>มีการ A/B testing แบบต่อเนื่องสำหรับประเมินโมเดลใหม่</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>ระบบของ Netflix แสดงให้เห็นถึงความสำคัญของ data lineage, reproducibility และการใช้ feedback loop ใน real-time เพื่อเพิ่ม accuracy ของ recommendation</p>
    </div>

    <h3>กรณีศึกษาที่ 2: ระบบ Fraud Detection ของ PayPal</h3>
    <p>PayPal ใช้ระบบ Machine Learning แบบ real-time ในการตรวจจับธุรกรรมที่อาจเป็นการทุจริต โดยมีการ deploy โมเดลหลายรุ่นแบบ concurrent บนระบบ distributed.</p>
    <table className="table-auto border w-full mt-4 text-sm">
      <thead className="bg-gray-200 dark:bg-gray-700">
        <tr>
          <th className="border px-4 py-2">องค์ประกอบ</th>
          <th className="border px-4 py-2">แนวปฏิบัติ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">Latency</td>
          <td className="border px-4 py-2">ต่ำกว่า 200ms ในทุก transaction</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Model Update</td>
          <td className="border px-4 py-2">ใช้ active learning และ retraining รายวัน</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">Monitoring</td>
          <td className="border px-4 py-2">มีระบบ concept drift detection เชิง real-time</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-blue-700 border-l-4 border-blue-400 p-4 rounded">
      <p className="font-semibold">Highlight Box:</p>
      <p>การ deploy โมเดลในระบบที่มีผลต่อธุรกรรมการเงิน ต้องให้ความสำคัญกับ reliability และ auditability เป็นอันดับต้น</p>
    </div>

    <h3>กรณีศึกษาที่ 3: CheXNet ของ Stanford ในการวินิจฉัยโรค</h3>
    <p>CheXNet คือโมเดล deep learning ที่ใช้ในการตรวจหาโรคปอดจากภาพ X-ray ได้แม่นยำเทียบเท่าแพทย์ โดยการ deploy โมเดลต้องพิจารณาเรื่อง fairness, privacy และ explainability.</p>
    <ul className="list-disc pl-6">
      <li>โมเดลถูกแปลงเป็น ONNX เพื่อลด latency ใน edge device</li>
      <li>ใช้ Grad-CAM สำหรับสร้าง heatmap อธิบายการตัดสินใจ</li>
      <li>ระบบถูก deploy ผ่าน Kubernetes ในโรงพยาบาลที่เป็น partner</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-400 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>การ deploy โมเดลด้านการแพทย์จำเป็นต้องมีระบบความปลอดภัยขั้นสูงและต้องสามารถตรวจสอบย้อนกลับได้ทุก prediction</p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>Netflix Research Blog: Scalable ML Infrastructure at Netflix</li>
      <li>Stanford ML Group: CheXNet Case Study</li>
      <li>IEEE Transactions on Services Computing: ML Deployment in Fintech</li>
      <li>Google Cloud Blog: ML Deployment Best Practices</li>
    </ul>
  </div>
</section>


<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Insight Box</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>

  <div className="prose prose-lg dark:prose-invert max-w-none space-y-8 text-base leading-relaxed">
    <h3>บทสรุปจากการ Deploy โมเดลในโลกจริง</h3>
    <p>
      การ Deploy โมเดล Machine Learning ไปใช้งานจริงไม่ใช่เพียงขั้นตอนหลังสุดของ Pipeline แต่เป็นกระบวนการที่ต้องวางแผนล่วงหน้าอย่างรอบคอบ โดยพิจารณาทั้งความเสถียร ความปลอดภัย และความสามารถในการขยายระบบ (scalability) การเตรียมโครงสร้างพื้นฐาน การจัดการเวอร์ชัน และระบบ monitoring จึงเป็นหัวใจสำคัญในการนำโมเดลไปใช้ให้เกิดผลลัพธ์ในเชิงธุรกิจได้จริง
    </p>

    <div className="bg-blue-700 border-l-4 border-blue-500 p-4 rounded">
      <p className="font-semibold">Highlight:</p>
      <p>
        ผลการศึกษาโดย Stanford และ Google Research แสดงให้เห็นว่าโมเดลที่ deploy โดยมีระบบ feedback loop, model registry และ monitoring ที่ดี จะมีความเสถียรใน production สูงขึ้นกว่า 60% เมื่อเทียบกับโมเดลที่ deploy แบบ manual หรือขาดโครงสร้างที่ชัดเจน
      </p>
    </div>

    <h3>แนวโน้มอนาคตของการ Deploy โมเดล</h3>
    <ul className="list-disc pl-6">
      <li><strong>Serverless ML:</strong> แนวคิดในการ deploy โมเดลแบบไม่ต้องจัดการ infrastructure เช่น Vertex AI, AWS Lambda</li>
      <li><strong>AutoML + CI/CD:</strong> การรวมกันของ automation ตั้งแต่ training → serving → monitoring → rollback</li>
      <li><strong>Edge Deployment:</strong> การนำโมเดลไปใช้งานที่ edge device เช่น IoT, มือถือ, รถยนต์</li>
      <li><strong>Explainability Integration:</strong> มีระบบอธิบายผลลัพธ์แบบ real-time เช่น SHAP, LIME เพื่อรองรับข้อกำหนดด้านความโปร่งใส</li>
    </ul>

    <div className="bg-yellow-700 border-l-4 border-yellow-500 p-4 rounded">
      <p className="font-semibold">Insight Box:</p>
      <p>
        ความสามารถในการ deploy โมเดลให้เสถียรใน production environment คือเส้นแบ่งระหว่างโครงการวิจัยกับนวัตกรรมที่ใช้ได้จริง ความสำเร็จของระบบ AI ขึ้นอยู่กับความสามารถในการออกแบบและดูแลระบบหลัง deploy อย่างต่อเนื่อง
      </p>
    </div>

    <h3>แหล่งอ้างอิง</h3>
    <ul className="list-disc pl-6">
      <li>“Hidden Technical Debt in Machine Learning Systems.” Sculley et al., Google Research, NIPS 2015</li>
      <li>“Reliable Machine Learning.” Stanford CS329S, Chip Huyen (2023)</li>
      <li>“MLOps: CI/CD and System Automation for Machine Learning.” O'Reilly, 2021</li>
      <li>“Modern Data Infrastructure for AI.” MIT CSAIL Report, 2022</li>
    </ul>
  </div>
</section>


          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day40 theme={theme} />
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
        <ScrollSpy_Ai_Day40 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day40_ModelDeployment;
