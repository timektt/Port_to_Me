import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day19 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day19";
import MiniQuiz_Day19 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day19";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day19_GradientDescentVariants = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("gd_variants1").format("auto").quality("auto").resize(scale().width(650));
  const img2 = cld.image("gd_variants2").format("auto").quality("auto").resize(scale().width(600));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">Day 19: Gradient Descent Variants</h1>

        <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img1} />
        </div>

        {/* Sections */}
        <section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: ทำไม Gradient Descent แบบธรรมดาไม่พอ?</h2>
  <div className="flex justify-center my-6">
          <AdvancedImage cldImg={img2} />
        </div>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Gradient Descent เปஈนวୈธีการหลักในการฝึก Neural Networks และโมเดล Machine Learning จำนวนมาก แต่การใช้ Basic Gradient Descent ตรงๆ โดยไม่มีการพัฒนาเพิ่มเติม กลับพบข้อจำกัดสำคัญหลายประการ เพื่อเป็นแรงจูงใจให้เกิดเทคนิค Gradient Descent Variants สมัยให่ในยุคปัจจุบัน
    </p>

    <h3 className="text-xl font-semibold">ปัญหาหลักของ Basic Gradient Descent</h3>
    <ul className="list-disc pl-6">
      <li><strong>Slow Convergence:</strong> การเคลื่อนที่ลงสู่ Minimum ทำได้ช้าโดยเฉพาะในฟังก์ชันที่มีลักษณะ Highly Curved หรือ Narrow Valley</li>
      <li><strong>Local Minima / Saddle Points:</strong> ติดค้างอยู่ที่จุดต่ำสุดท้องถิ่น หรือ Saddle Points ทำให้การฝึกไม่สามารถก้าวหน้าต่อได้</li>
      <li><strong>Oscillation:</strong> การแกว่งไปมาระหว่างแกนในกรณีที่ Surface มีลักษณะยืดยาว (Anisotropic Surface)</li>
      <li><strong>Hyperparameter Sensitivity:</strong> ต้องการการเลือก Learning Rate (η) อย่างละเอียด มิฉะนั้นโมเดลจะไม่เรียนรู้ หรือ Diverge ทันที</li>
    </ul>

    <div className="flex justify-center my-8">
      <img src="/images/gradient_descent_oscillation.png" alt="Oscillation in Gradient Descent" className="rounded-xl shadow-md w-full max-w-2xl" />
    </div>

    <h3 className="text-xl font-semibold">ทำไมการพัฒนา Gradient Descent Variants จึงจำเป็น?</h3>
    <p>
      งานวิจัยของ Yann LeCun (New York University), Geoffrey Hinton (University of Toronto) และ Yoshua Bengio (University of Montreal) 
      ต่างยืนยันว่าการเลือกวิธี Optimization ที่เหมาะสมสามารถเร่งความเร็วในการฝึก Deep Learning Models ได้อย่างมีนัยสำคัญ บางงานเช่น "Efficient BackProp" และ "Understanding the difficulty of training deep feedforward neural networks" ชี้ให้เห็นว่าการพัฒนาวิธีแก้ไขปัญหาของ Basic Gradient Descent เป็นปัจจัยสำคัญที่ทำให้ Deep Learning ประสบความสำเร็จในยุค 2010 เป็นต้นมา
    </p>

    <p>
      เป้าหมายของการพัฒนา Variants ของ Gradient Descent จึงมุ่งไปที่การ:
    </p>
    <ul className="list-disc pl-6">
      <li>เพิ่มความเร็วในการลู่เข้า (Faster Convergence)</li>
      <li>หลีกเลี่ยง Local Minima หรือ Saddle Points</li>
      <li>ลดการแกว่งตัวระหว่างการฝึก</li>
      <li>เพิ่ม Robustness ต่อการตั้งค่า Hyperparameters</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างเชิงเปรียบเทียบ: Gradient Descent vs SGD</h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Gradient Descent</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>ใช้ข้อมูลทั้งหมดในแต่ละรอบอัปเดต (Full Batch)</li>
          <li>เส้นทางการเคลื่อนที่ราบเรียบแต่ช้า</li>
          <li>ต้องการหน่วยความจำขนาดใหญ่</li>
        </ul>
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-semibold mb-2">Stochastic Gradient Descent (SGD)</h4>
        <ul className="list-disc pl-5 text-sm">
          <li>อัปเดตจากตัวอย่างเดียวหรือล็อตเล็กๆ (Mini-batch)</li>
          <li>มี Noise ช่วยให้ข้าม Saddle Points ได้ง่าย</li>
          <li>การลู่เข้ามีความผันผวนสูง แต่ประหยัดหน่วยความจำ</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">ข้อมูลเชิงเทคนิคเพิ่มเติม</h3>
    <p>
      ในทางคณิตศาสตร์ Gradient Descent สามารถอธิบายได้ด้วยสมการพื้นฐาน:
    </p>
    <pre className="bg-gray-800 text-white p-4 rounded-xl overflow-auto text-sm">
θ = θ - η∇J(θ)
    </pre>
    <p>
      โดยที่ θ คือตัวแปรพารามิเตอร์, η คือตัว Learning Rate และ ∇J(θ) คือตัว Gradient ของ Loss Function
    </p>

    <h3 className="text-xl font-semibold">ความเชื่อมโยงกับ Optimization ทฤษฎี</h3>
    <p>
      จากการศึกษาของ Stanford CS231n และ MIT Deep Learning Notes การฝึก Neural Network ถูกมองว่าเป็นกระบวนการ Optimization ใน Non-Convex Landscapes ซึ่งมีลักษณะไม่สม่ำเสมอ เต็มไปด้วยจุดต่ำสุดท้องถิ่นและ Saddle Points 
      ทำให้ต้องการเทคนิคที่สามารถปรับตัวตามรูปร่างของพื้นผิว Loss ได้อย่างมีประสิทธิภาพ เพื่อหลีกเลี่ยงปัญหาการติดขัดและเร่งกระบวนการหาค่า Optimal
    </p>

    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="font-semibold mb-2">Insight Box: บทเรียนจากงานวิจัย</h4>
      <ul className="list-disc pl-6 text-sm">
        <li>การเลือก Optimizer ที่เหมาะสมมีผลต่อเวลาในการฝึกหลายเท่า</li>
        <li>Learning Rate Scheduling มีบทบาทสำคัญในการรักษาเสถียรภาพของการลู่เข้า</li>
        <li>ไม่มี Optimizer ใดที่ดีที่สุดในทุกสถานการณ์ (No Free Lunch Theorem)</li>
        <li>การใช้ Momentum และ Adaptive Techniques ช่วยให้โมเดลหลีกเลี่ยง Plateau ได้ดียิ่งขึ้น</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุปเบื้องต้น</h3>
    <p>
      การพัฒนา Gradient Descent Variants ถือเป็นหนึ่งในเสาหลักที่ทำให้ Deep Learning ก้าวข้ามขีดจำกัดเดิม ๆ ได้ 
      การทำความเข้าใจปัญหาของ Basic Gradient Descent และการพัฒนาต่อยอด เช่น SGD, Momentum, AdaGrad, RMSProp, และ Adam
      จึงเป็นพื้นฐานสำคัญสำหรับนักพัฒนาโมเดลสมัยใหม่ในการออกแบบระบบที่มีความแม่นยำ เสถียร และรวดเร็วในการฝึกฝนบนข้อมูลจริงที่ซับซ้อน
    </p>
  </div>
</section>


<section id="basic-gradient-descent" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Review สั้น: Gradient Descent พื้นฐาน</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Gradient Descent คืออัลกอริทึมพื้นฐานที่สุดใน Optimization สำหรับการเรียนรู้ของโมเดล Machine Learning และ Deep Learning ซึ่งได้รับการอธิบายไว้อย่างกว้างขวางในงานวิจัยของ Stanford University และ MIT AI Lab ว่าเป็นกลไกสำคัญในการหาค่าพารามิเตอร์ที่เหมาะสมเพื่อให้ Loss Function ต่ำที่สุด
    </p>
    
    <h3 className="text-xl font-semibold">นิยาม Loss Function</h3>
    <p>
      Loss Function คือฟังก์ชันที่วัดความคลาดเคลื่อนระหว่างผลลัพธ์ที่โมเดลทำนายได้กับค่าความจริง (Ground Truth) เช่น Mean Squared Error (MSE) สำหรับ Regression หรือ Cross-Entropy Loss สำหรับ Classification จุดประสงค์คือการทำให้ค่าของ Loss ต่ำที่สุดเพื่อให้โมเดลมีความแม่นยำในการพยากรณ์หรือจำแนกข้อมูล
    </p>

    <h3 className="text-xl font-semibold">การคำนวณ Gradient</h3>
    <p>
      Gradient คือเวกเตอร์ของอนุพันธ์ของ Loss Function ตามพารามิเตอร์แต่ละตัว โดยแสดงถึงทิศทางและอัตราการเปลี่ยนแปลงของ Loss เมื่อพารามิเตอร์มีการเปลี่ยนแปลง ความรู้เรื่อง Gradient เป็นพื้นฐานของ Calculus ที่สถาบันเช่น Caltech และ DeepMind เน้นย้ำในการออกแบบระบบเรียนรู้ที่มีประสิทธิภาพ
    </p>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
  <p className="text-sm">
    <strong>นิยามทางคณิตศาสตร์:</strong> <br/>
    ∇J(θ) = [∂J/∂θ₁, ∂J/∂θ₂, ..., ∂J/∂θₙ]
  </p>
</div>

    <h3 className="text-xl font-semibold">Update Rule พื้นฐาน</h3>
    <p>
      หลักการของ Gradient Descent คือการอัปเดตพารามิเตอร์ในทิศทางตรงข้ามกับ Gradient ของ Loss Function เพื่อหาค่า Local Minimum หรือ Global Minimum ของฟังก์ชันเป้าหมาย
    </p>
    <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg border-l-4 border-blue-400 dark:border-blue-600">
      <code className="text-sm block">
        \[ \theta = \theta - \eta \nabla J(\theta) \]
      </code>
    </div>
    <p>
      โดยที่:
    </p>
    <ul className="list-disc list-inside ml-6">
      <li>\( \theta \) คือพารามิเตอร์ของโมเดล</li>
      <li>\( \eta \) คือ Learning Rate หรืออัตราการเรียนรู้</li>
      <li>\( \nabla J(\theta) \) คือ Gradient ของ Loss Function</li>
    </ul>

    <h3 className="text-xl font-semibold">Learning Rate และผลต่อการลู่เข้า (Convergence)</h3>
    <p>
      Learning Rate เป็นตัวกำหนดขนาดก้าวในการอัปเดตพารามิเตอร์ แต่ละค่า Learning Rate มีผลอย่างมากต่อความเร็วและความเสถียรของการลู่เข้า
    </p>

    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">Learning Rate ต่ำเกินไป</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>โมเดลเรียนรู้ช้า</li>
          <li>ต้องใช้จำนวน Epochs สูง</li>
          <li>เสี่ยงต่อการติดอยู่ในค่า Local Minima</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-300 dark:border-gray-700 shadow">
        <h4 className="text-lg font-medium mb-2">Learning Rate สูงเกินไป</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>โมเดล Oscillate หรือแกว่งไปมา</li>
          <li>ไม่สามารถหาค่า Minimum ได้</li>
          <li>เสี่ยงต่อการ Divergence ของ Loss</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-10">ตัวอย่างการใช้ Gradient Descent</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import numpy as np

def loss_function(theta):
    return (theta - 3) ** 2

def gradient(theta):
    return 2 * (theta - 3)

# Initial value
theta = 0.0
learning_rate = 0.1

# Perform iterations
for i in range(20):
    grad = gradient(theta)
    theta = theta - learning_rate * grad
    print(f"Iteration {i+1}: theta = {theta:.4f}, loss = {loss_function(theta):.4f}")`}
    </pre>
    <p>
      ตัวอย่างแสดงการปรับพารามิเตอร์ \( \theta \) ไปยังค่าเป้าหมาย 3 โดยใช้การอัปเดตตาม Gradient Descent ทุก ๆ รอบการเรียนรู้ (epoch)
    </p>

    <h3 className="text-xl font-semibold mt-10">Insight Box</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2">
        <li>Gradient Descent ทำงานได้ดีที่สุดในฟังก์ชันที่มีลักษณะ Smooth และ Convex</li>
        <li>Learning Rate Scheduling เช่น Step Decay, Exponential Decay หรือ Cosine Annealing ช่วยให้การลู่เข้าเร็วขึ้น</li>
        <li>การทำ Mini-Batch หรือ Stochastic Updates มีบทบาทสำคัญในการทำให้โมเดลสามารถหลีกเลี่ยงการติด Local Minima ได้</li>
        <li>การเลือก Initial Parameters ที่ดีช่วยให้โมเดลหาคำตอบได้เร็วและมีเสถียรภาพมากขึ้น</li>
      </ul>
    </div>
  </div>
</section>

<section id="sgd" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Stochastic Gradient Descent (SGD)</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Stochastic Gradient Descent (SGD) เป็นหนึ่งในอัลกอริทึมพื้นฐานที่มีบทบาทสำคัญในกระบวนการฝึกโมเดล Machine Learning และ Deep Learning โดยมีหลักการทำงานที่แตกต่างจาก Gradient Descent แบบดั้งเดิมที่ใช้ข้อมูลทั้งชุด (full batch) ในการคำนวณกราดิเอนต์ในแต่ละรอบการอัปเดตพารามิเตอร์
    </p>

    <p>
      ใน SGD จะมีการเลือกตัวอย่างข้อมูลแบบสุ่มเพียงหนึ่งตัวอย่าง หรือกลุ่มย่อยขนาดเล็ก (mini-batch) เพื่อคำนวณกราดิเอนต์และทำการอัปเดตพารามิเตอร์ทันที ส่งผลให้การอัปเดตเกิดขึ้นบ่อยครั้งและทำให้การฝึกมีความรวดเร็วขึ้น นอกจากนี้การสุ่มยังทำให้เส้นทางการไหลของพารามิเตอร์มีลักษณะเป็น stochastic หรือมีความไม่แน่นอนซึ่งช่วยให้สามารถหลีกเลี่ยง local minima ได้ดีขึ้นในฟังก์ชันที่มีหลาย minima
    </p>

    <div className="flex justify-center my-6">
      <img src="/images/sgd_vs_gd.png" alt="GD vs SGD" className="rounded-lg shadow-lg max-w-full h-auto" />
    </div>

    <h3 className="text-xl font-semibold">สูตรการอัปเดตของ SGD</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
    <p className="text-sm">
  θ = θ − η ∇J(θ; x⁽ᶦ⁾, y⁽ᶦ⁾)
</p>

<p className="text-base leading-relaxed mt-4">
  โดยที่ θ คือพารามิเตอร์ของโมเดล, η คือค่า Learning Rate, และ ∇J(θ; x⁽ᶦ⁾, y⁽ᶦ⁾) คือกราดิเอนต์ของค่าความสูญเสียจากตัวอย่าง (x⁽ᶦ⁾, y⁽ᶦ⁾)
</p>
</div> 



    <h3 className="text-xl font-semibold">ข้อดีของ SGD</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>มีการอัปเดตพารามิเตอร์บ่อย ช่วยให้สามารถเริ่มเห็นผลของการฝึกได้เร็วขึ้น</li>
      <li>ความไม่แน่นอน (Noise) จากการสุ่มช่วยหลีกเลี่ยงการติด local minima หรือ saddle point</li>
      <li>ประหยัดหน่วยความจำ เนื่องจากไม่ต้องโหลดข้อมูลทั้งชุดเข้าหน่วยความจำในแต่ละรอบการฝึก</li>
      <li>เหมาะสำหรับงานที่มีข้อมูลขนาดใหญ่มาก เช่น ImageNet หรือ Big Data</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อเสียของ SGD</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การแกว่ง (Oscillation) รอบ minimum ทำให้การลู่เข้าช้าในช่วงท้ายของการฝึก</li>
      <li>ผลลัพธ์ในแต่ละรอบมีความแปรปรวนสูง ทำให้ Loss มีการกระเพื่อมมากกว่าการใช้ Full Batch GD</li>
      <li>ต้องมีการจูน Hyperparameters เช่น Learning Rate และการใช้เทคนิคเสริมเช่น Learning Rate Decay หรือ Momentum</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างการเปรียบเทียบ GD กับ SGD</h3>
    <div className="flex justify-center my-6">
      <img src="/images/sgd_landscape.gif" alt="SGD Landscape Visualization" className="rounded-lg shadow-lg max-w-full h-auto" />
    </div>

    <p>
      จากภาพจะเห็นได้ว่า Full Batch Gradient Descent เคลื่อนที่เป็นเส้นตรงอย่างมั่นคงสู่ minimum แต่มีความช้า ในขณะที่ SGD กระโดดไปมาแบบ stochastic แต่สามารถไปถึงบริเวณ minimum ได้รวดเร็วกว่ามาก
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดการใช้ SGD ด้วย PyTorch</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-lg overflow-x-auto">
{`import torch
import torch.nn as nn
import torch.optim as optim

# สร้างโมเดลอย่างง่าย
model = nn.Linear(10, 1)

# สร้าง Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# วนเทรนตัวอย่าง
for epoch in range(100):
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)

    outputs = model(inputs)
    loss = ((outputs - targets) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
    </pre>

    <h3 className="text-xl font-semibold">กลยุทธ์ในการใช้ SGD อย่างมีประสิทธิภาพ</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ใช้ Learning Rate Scheduling เช่น Step Decay, Exponential Decay, หรือ Cosine Annealing</li>
      <li>ใช้ Momentum เพื่อลดการแกว่งและเร่งการลู่เข้า</li>
      <li>ใช้ Early Stopping เพื่อลดความเสี่ยงจาก Overfitting</li>
      <li>ใช้ Mini-Batch Gradient Descent เพื่อหาจุดสมดุลระหว่างการอัปเดตเร็วและความเสถียร</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2">
        <li>SGD เป็นตัวเลือกที่เหมาะสมสำหรับการฝึกโมเดล Deep Learning ขนาดใหญ่ โดยเฉพาะเมื่อใช้ร่วมกับ Momentum และเทคนิคปรับ Learning Rate</li>
        <li>การเข้าใจธรรมชาติของความไม่แน่นอนใน SGD ช่วยให้สามารถออกแบบกลยุทธ์การฝึกโมเดลได้ดีกว่าเดิม</li>
        <li>การใช้ Mini-batch เป็นเทคนิคยอดนิยมในงานจริง เพราะลดความผันผวนเกินไปและเพิ่มเสถียรภาพในการฝึก</li>
        <li>ปัจจุบัน Optimizers ชั้นนำเช่น Adam ก็พัฒนาขึ้นจากแนวคิดพื้นฐานของ SGD ผนวกกับการแก้ไขข้อด้อยบางประการ</li>
      </ul>
    </div>
  </div>
</section>

<section id="mini-batch" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. Mini-Batch Gradient Descent</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Mini-Batch Gradient Descent เป็นหนึ่งในเทคนิคการอัปเดตพารามิเตอร์ที่ได้รับความนิยมสูงสุดในวงการ Deep Learning และ Machine Learning สมัยใหม่ โดยมีการผสานข้อดีระหว่าง Gradient Descent แบบดั้งเดิมที่มีเสถียรภาพสูง และ Stochastic Gradient Descent (SGD) ที่มีความเร็วในการอัปเดตสูง ทั้งนี้ Mini-Batch ช่วยให้สามารถปรับตัวได้ดีต่อทั้งปริมาณข้อมูลที่มากและการเรียนรู้ที่มีความผันผวนในระดับที่เหมาะสม
    </p>

    <h3 className="text-xl font-semibold">นิยาม Mini-Batch Gradient Descent</h3>
    <p>
      Mini-Batch Gradient Descent ทำงานโดยการสุ่มเลือกข้อมูลจำนวนหนึ่งที่เรียกว่า Batch ขนาดเล็ก (เช่น 32, 64, 128 ตัวอย่าง) จาก Dataset ทั้งหมดในแต่ละรอบการอัปเดตน้ำหนัก จากนั้นคำนวณ Gradient ของ Loss Function เฉพาะบนตัวอย่างใน Batch นี้ แล้วทำการอัปเดตพารามิเตอร์ การทำเช่นนี้ช่วยลดความซับซ้อนในการคำนวณเมื่อเทียบกับการคำนวณทั้ง Dataset และช่วยลดความผันผวนสูงเกินไปที่พบใน SGD แบบดั้งเดิม
    </p>

    <h3 className="text-xl font-semibold">สูตรการอัปเดตของ Mini-Batch GD</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
    <pre className="text-sm overflow-x-auto">
{`Sample a mini-batch {x¹, ..., xᵐ}
Compute gradient:
  ∇J(θ) = (1/m) Σᵢ₌₁ᵐ ∇θ J(θ; x⁽ᶦ⁾, y⁽ᶦ⁾)
Update parameters:
  θ = θ − η × ∇J(θ)
`}
</pre>

    </div>

    <h3 className="text-xl font-semibold">ข้อดีของ Mini-Batch GD</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>เพิ่มประสิทธิภาพการประมวลผลด้วยการใช้ Hardware Acceleration เช่น GPU และ TPU ได้อย่างมีประสิทธิภาพ</li>
      <li>ลดความผันผวนของการอัปเดตน้ำหนักเมื่อเทียบกับ Stochastic Gradient Descent</li>
      <li>ช่วยให้การเรียนรู้มีความรวดเร็วและมีเสถียรภาพมากขึ้น</li>
      <li>สามารถใช้ Regularization Effect จาก Noise ที่เกิดในแต่ละ Mini-Batch เพื่อช่วยลด Overfitting ได้โดยธรรมชาติ</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดการใช้งาน Mini-Batch GD ใน PyTorch</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <pre className="text-sm overflow-x-auto">
{`import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\n# ข้อมูลตัวอย่าง\nX = torch.randn(1000, 10)\ny = torch.randint(0, 2, (1000,))\n\ndataset = TensorDataset(X, y)\nloader = DataLoader(dataset, batch_size=64, shuffle=True)\n\nmodel = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\n\nfor epoch in range(10):\n    for batch_X, batch_y in loader:\n        optimizer.zero_grad()\n        outputs = model(batch_X)\n        loss = criterion(outputs, batch_y)\n        loss.backward()\n        optimizer.step()`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">การเลือกขนาดของ Batch Size ที่เหมาะสม</h3>
    <p>
      การเลือก Batch Size มีผลโดยตรงต่อความเร็วและคุณภาพของการเรียนรู้ โดยมีหลักการทั่วไปดังนี้
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>Batch Size ขนาดเล็ก</strong> (เช่น 32, 64): ช่วยให้โมเดลสามารถ Escape จาก Local Minima ได้ดี แต่มีความผันผวนสูงขึ้น</li>
      <li><strong>Batch Size ขนาดกลาง</strong> (เช่น 128, 256): สมดุลระหว่างความเร็วและเสถียรภาพ เหมาะสำหรับงานส่วนใหญ่ในปัจจุบัน</li>
      <li><strong>Batch Size ใหญ่มาก</strong> (เช่น 1024, 4096): ใช้ได้เมื่อมี Memory เพียงพอ แต่มีความเสี่ยงต่อการติด Plateau หรือ Saddle Point ได้ง่าย</li>
    </ul>

    <h3 className="text-xl font-semibold">Trade-off: Speed vs Stability</h3>
    <p>
      การเลือก Batch Size ต้องแลกเปลี่ยนระหว่างความเร็วในการฝึกกับเสถียรภาพของการอัปเดตน้ำหนัก โดยทั่วไป Batch ขนาดเล็กจะทำให้ Loss มีการสั่นไหว แต่สามารถช่วยให้โมเดลกระโดดข้ามภูมิประเทศ Optimization ที่มีความซับซ้อน ในขณะที่ Batch ขนาดใหญ่จะช่วยให้การไหลของ Loss เรียบเนียนขึ้น แต่เสี่ยงต่อการติดอยู่ใน Suboptimal Minima
    </p>

    <div className="flex flex-col md:flex-row gap-6">
      <div className="bg-green-100 dark:bg-green-900 p-5 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">ข้อดีของ Batch เล็ก</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>เพิ่มความสามารถในการหลีกเลี่ยง Local Minima</li>
          <li>ทำให้เกิด Regularization Effect</li>
          <li>ประหยัดหน่วยความจำ</li>
        </ul>
      </div>
      <div className="bg-red-100 dark:bg-red-900 p-5 rounded-lg flex-1">
        <h4 className="font-semibold mb-2">ข้อดีของ Batch ใหญ่</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>ไหลของ Loss มีเสถียรภาพสูง</li>
          <li>ประหยัดเวลา Epoch ละรอบเมื่อใช้ Hardware ที่มีขนาดใหญ่</li>
          <li>เหมาะกับ Distributed Training</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold">Insight จากการศึกษาระดับโลก</h3>
    <p>
      งานวิจัยจาก Google Brain และ DeepMind พบว่าการใช้ Learning Rate ที่ปรับตาม Batch Size เช่น Linear Scaling Rule (Goyal et al., 2017) มีผลดีอย่างมากในการเร่งความเร็วการฝึกโมเดลลึก เช่น ResNet, Transformer โดยไม่สูญเสียประสิทธิภาพ
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-4 rounded-lg border-l-4 border-yellow-400 dark:border-yellow-600">
      <h4 className="font-semibold mb-2">Insight Box</h4>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>Mini-Batch Gradient Descent คือรากฐานของการฝึก Neural Networks ขนาดใหญ่</li>
        <li>Batch Size เป็น Hyperparameter ที่ต้องจูนอย่างระมัดระวัง</li>
        <li>Batch ขนาดกลาง (32-256) ให้ผลดีที่สุดในหลายงานจริง</li>
        <li>การเพิ่ม Batch Size จำเป็นต้องเพิ่ม Learning Rate อย่างสอดคล้องกัน</li>
        <li>การใช้ Mixed Precision Training (FP16) ช่วยให้ Batch Size ใหญ่ขึ้นโดยไม่กิน Memory มากนัก</li>
      </ul>
    </div>
  </div>
</section>


<section id="momentum" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Momentum</h2>
  
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <p>
      Momentum เป็นเทคนิคที่ถูกพัฒนาขึ้นเพื่อลดปัญหาการแกว่ง (oscillation) และช่วยเร่งความเร็วในการไหลลงสู่ minimum ของ Loss Function ในระหว่างกระบวนการฝึกโมเดลโดยใช้ Gradient Descent เทคนิคนี้ได้รับการแนะนำครั้งแรกโดย Rumelhart, Hinton และ Williams ในปี 1986 ผ่านงานวิจัยเกี่ยวกับการฝึกโครงข่ายประสาทเทียมเชิงลึก (Deep Neural Networks)
    </p>

    <p>
      แนวคิดพื้นฐานของ Momentum คือการรวมทิศทางของกราดิเอนต์ในอดีตเข้ากับการอัปเดตปัจจุบัน เปรียบเสมือนกับการจำลองการเคลื่อนที่ของวัตถุที่มีแรงเฉื่อยในฟิสิกส์ กล่าวคือ เมื่อวัตถุกำลังเคลื่อนที่ในทิศทางหนึ่ง มันจะมีแนวโน้มรักษาทิศทางนั้นไว้ แม้ว่าจะมีแรงภายนอกกระทำอยู่
    </p>

    <h3 className="text-xl font-semibold">สูตรคำนวณ Momentum</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg border-l-4 border-blue-400 dark:border-blue-600">
      <code className="text-sm block">
        vₜ = βvₜ₋₁ + (1 - β)∇J(θₜ)<br />
        θₜ = θₜ₋₁ - ηvₜ
      </code>
    </div>
    <p>
      โดยที่:
    </p>
    <ul className="list-disc pl-6 space-y-2">
      <li><strong>vₜ</strong> คือค่า velocity หรือ momentum ในรอบที่ t</li>
      <li><strong>β</strong> คือค่าคงที่ Momentum Coefficient (เช่น 0.9)</li>
      <li><strong>∇J(θₜ)</strong> คือ Gradient ของ Loss Function ที่พารามิเตอร์ θ ในรอบที่ t</li>
      <li><strong>η</strong> คือ Learning Rate</li>
      <li><strong>θₜ</strong> คือค่าพารามิเตอร์หลังการอัปเดต</li>
    </ul>

    <h3 className="text-xl font-semibold">หลักการทำงานของ Momentum</h3>
    <p>
      ในช่วงแรกของการฝึก ค่ากราดิเอนต์จะยังเปลี่ยนแปลงสูงเนื่องจากโมเดลยังไม่ได้เข้าใกล้ minimum มากนัก การใช้ Momentum จะทำให้การอัปเดตในทิศทางที่คงที่เกิดขึ้นได้เร็วขึ้น ช่วยลดปัญหาการแกว่งตัวรุนแรงเมื่อเจอพื้นที่ของ Loss Surface ที่มีความโค้งไม่เท่ากัน เช่น ร่องแคบในแนวตั้งและกว้างในแนวนอน
    </p>

    <p>
      การตั้งค่า β มีผลอย่างมากต่อการทำงานของ Momentum ค่า β สูง เช่น 0.9 หรือ 0.99 จะทำให้โมเดล "จดจำ" ทิศทางการไหลได้ยาวนานขึ้น ทำให้การเคลื่อนที่ผ่านพื้นผิว Loss เป็นไปอย่างรวดเร็วและราบรื่น
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งานใน PyTorch</h3>
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <pre className="text-sm overflow-x-auto">
{`import torch
import torch.optim as optim

model = ...  # นิยามโมเดล
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for inputs, labels in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()`}
      </pre>
    </div>

    <h3 className="text-xl font-semibold">ข้อดีของ Momentum</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>ช่วยให้การลู่เข้าของโมเดล (convergence) เร็วขึ้นอย่างมาก</li>
      <li>ลดการแกว่ง (oscillation) รอบ minimum</li>
      <li>ช่วยให้โมเดลสามารถหลีกเลี่ยงการติดอยู่ที่ local minima ตื้น ๆ ได้ดีขึ้น</li>
      <li>เหมาะสำหรับปัญหาที่มี Loss Surface เป็นรูป "หลุมยาวแคบ" (narrow ravine)</li>
    </ul>

    <h3 className="text-xl font-semibold">ข้อควรระวังในการใช้ Momentum</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>การตั้งค่า β ที่สูงเกินไปอาจทำให้การอัปเดตพารามิเตอร์ overshoot ผ่าน minimum ไป</li>
      <li>Momentum ไม่สามารถแก้ปัญหา Learning Rate ที่ตั้งไม่เหมาะสมได้ด้วยตัวเอง</li>
      <li>ต้องมีการทดลองปรับค่า β และ η ควบคู่กันเพื่อให้ได้ผลลัพธ์ที่ดีที่สุด</li>
    </ul>

    <h3 className="text-xl font-semibold">กรณีศึกษา: การใช้ Momentum ใน Deep Network</h3>
    <p>
      งานวิจัยโดย Ilya Sutskever และคณะจาก Google Brain (2013) พบว่าการใช้ Momentum ร่วมกับค่า Learning Rate ที่สูงขึ้น สามารถเร่งความเร็วในการฝึก Deep Neural Networks ได้ถึง 5–10 เท่า เมื่อเทียบกับการใช้ Gradient Descent ธรรมดา ในงานเช่นการจำแนกภาพ (Image Classification) และการแปลภาษา (Machine Translation)
    </p>

    <p>
      การทดลองเปรียบเทียบระหว่าง Gradient Descent ปกติและ Gradient Descent พร้อม Momentum แสดงให้เห็นว่าเส้นทางการลู่เข้า (trajectory) ของโมเดลมีความราบรื่นและรวดเร็วกว่ามาก โดยเฉพาะในช่วงต้นของการฝึก
    </p>

    <h3 className="text-xl font-semibold">การวิเคราะห์เชิงคณิตศาสตร์</h3>
    <p>
      ในเชิงคณิตศาสตร์ Momentum ทำหน้าที่เปลี่ยนการอัปเดตจากรูปแบบการไหลไปตามแนว Gradient แบบจุดต่อจุด มาเป็นการพิจารณาเส้นทางสะสมในอดีต ส่งผลให้การไหลของพารามิเตอร์มีลักษณะ "วิ่งผ่าน" ตลอดแนว slope แทนที่จะไต่ขึ้นและลงไปตามพื้นผิวอย่างช้า ๆ
    </p>

    <div className="bg-blue-50 dark:bg-blue-900 text-blue-900 dark:text-blue-100 p-5 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <h3 className="text-xl font-semibold mb-2">Insight Box: ทำไม Momentum จึงได้ผลดี?</h3>
      <ul className="list-disc pl-6 space-y-2">
        <li>การจำทิศทางเดิมช่วยให้โมเดลหลีกเลี่ยงการแกว่งตัวในพื้นที่ที่มีความโค้งสูง</li>
        <li>การรวมข้อมูลในอดีตทำให้โมเดลมีการไหลที่ราบรื่นกว่าการอัปเดตที่อิงเฉพาะจุดปัจจุบัน</li>
        <li>ช่วยให้การเคลื่อนที่ผ่าน Saddle Points เป็นไปได้รวดเร็วขึ้น ลดเวลาฝึกโมเดล</li>
        <li>สามารถทำงานได้ดียิ่งขึ้นเมื่อใช้ร่วมกับ Learning Rate Scheduling เช่น Cosine Annealing หรือ Step Decay</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวเลือกต่อยอด: Nesterov Momentum</h3>
    <p>
      เพื่อแก้ไขข้อจำกัดของ Momentum แบบเดิม มีการพัฒนา Nesterov Accelerated Gradient (NAG) ซึ่งทำการ "มองล่วงหน้า" ก่อนอัปเดต ทำให้สามารถประเมิน Gradient ในตำแหน่งที่คาดว่าจะอยู่ในอนาคต เพิ่มความแม่นยำในการอัปเดตพารามิเตอร์
    </p>

    <p>
      การใช้ Momentum จึงไม่ใช่เพียงการเร่งกระบวนการฝึก แต่ยังมีบทบาทสำคัญในการปรับพฤติกรรมของโมเดลให้อัปเดตอย่างมีประสิทธิภาพสูงสุด โดยเฉพาะในเครือข่ายที่มีความลึกและซับซ้อนสูง
    </p>

  </div>
</section>


<section id="nesterov" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Nesterov Accelerated Gradient (NAG)</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
    <p>
      Nesterov Accelerated Gradient (NAG) เป็นเทคนิคการเร่งความเร็วของกระบวนการ Optimization ที่ได้รับการพัฒนาขึ้นโดย Yurii Nesterov ในปี 1983 จากงานวิจัยทางคณิตศาสตร์เชิงนูน (Convex Optimization) และต่อมาได้ถูกนำมาปรับใช้ใน Machine Learning และ Deep Learning อย่างแพร่หลาย เพื่อเพิ่มความเร็วในการลู่เข้าสู่ค่าต่ำสุดของ Loss Function และลดปัญหา overshooting ที่พบได้ใน Gradient Descent ทั่วไปและ Momentum ปกติ
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">หลักการเบื้องหลัง Nesterov</h3>
    <p>
      เทคนิค Momentum ช่วยให้การเคลื่อนที่บน Surface ของ Loss Landscape มีลักษณะต่อเนื่องราบรื่นขึ้นโดยการจดจำทิศทางการเคลื่อนที่ในอดีต แต่ Momentum ปกติจะคำนวณ Gradient ณ ตำแหน่งปัจจุบัน ในขณะที่ Nesterov มองล่วงหน้า (Lookahead) เพื่อคำนวณ Gradient ณ ตำแหน่งที่คาดว่าจะไปถึง โดยการก้าวล่วงไปข้างหน้าก่อนแล้วจึงประเมิน Gradient
    </p>

    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
      <p className="text-sm">
        <strong>สูตร Momentum ปกติ:</strong> <br />
        v<sub>t</sub> = \beta v<sub>t-1</sub> + \nabla J(\theta) <br />
        \theta = \theta - \eta v<sub>t</sub>
      </p>
      <p className="text-sm mt-4">
        <strong>สูตร Nesterov Accelerated Gradient (NAG):</strong> <br />
        v<sub>t</sub> = \beta v<sub>t-1</sub> + \nabla J(\theta - \eta \beta v<sub>t-1</sub>) <br />
        \theta = \theta - \eta v<sub>t</sub>
      </p>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-2">ประโยชน์ของการมองล่วงหน้า (Lookahead)</h3>
    <ul className="list-disc pl-6 space-y-2">
      <li>สามารถลดโอกาส overshoot จุดต่ำสุดของ Loss Function</li>
      <li>เพิ่มความแม่นยำในการประเมิน Gradient ที่ตำแหน่งจริง</li>
      <li>ช่วยให้การลู่เข้าเร็วขึ้น โดยเฉพาะในปัญหาที่มี Surface โค้งมาก</li>
      <li>ลดการแกว่ง (oscillation) รอบ Minimum ได้ดีกว่า Momentum ปกติ</li>
    </ul>

    <h3 className="text-xl font-semibold mt-8 mb-2">การเปรียบเทียบ Momentum ปกติและ Nesterov</h3>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
        <h4 className="text-lg font-medium mb-2">Momentum ธรรมดา</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>คำนวณ Gradient ณ ตำแหน่งปัจจุบัน</li>
          <li>อาจ overshoot และต้องกลับทิศทางบ่อยครั้ง</li>
          <li>การลู่เข้าเร็วขึ้นกว่าการใช้ GD ปกติแต่ยังช้ากว่า NAG ในบางกรณี</li>
        </ul>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow border border-green-300 dark:border-green-700">
        <h4 className="text-lg font-medium mb-2">Nesterov Accelerated Gradient</h4>
        <ul className="list-disc pl-5 text-sm space-y-2">
          <li>คำนวณ Gradient ณ ตำแหน่ง Lookahead</li>
          <li>ลดการ overshoot และแกว่งได้ดีขึ้น</li>
          <li>มีแนวโน้มลู่เข้าสู่ Minimum ได้เร็วกว่าหลายกรณี</li>
        </ul>
      </div>
    </div>

    <h3 className="text-xl font-semibold mt-8 mb-2">ตัวอย่างการใช้งาน NAG</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()`}
    </pre>
    <p>
      ในตัวอย่างนี้ มีการเปิดใช้ Nesterov ผ่านการกำหนด <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">nesterov=True</code> ใน Optimizer ของ PyTorch เพื่อช่วยให้การฝึกโมเดลลู่เข้าหา Minimum ได้รวดเร็วและเสถียรขึ้น
    </p>

    <h3 className="text-xl font-semibold mt-8 mb-2">Insight จากงานวิจัย</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>
          งานวิจัยของ Nesterov (1983) และการประยุกต์ใน Deep Learning โดย Sutskever et al. (2013) ชี้ให้เห็นว่า NAG ช่วยเพิ่มประสิทธิภาพการฝึกโมเดลลึกได้อย่างมีนัยสำคัญ
        </li>
        <li>
          การใช้ NAG ร่วมกับ Adaptive Optimizer เช่น Adam สามารถยกระดับการลู่เข้าในงานที่มี Loss Surface ซับซ้อนได้ แต่ต้องปรับ hyperparameter อย่างระมัดระวัง
        </li>
        <li>
          การเลือกใช้ NAG มีประโยชน์มากในกรณีที่ Loss Landscape มีความโค้งหรือมี Saddle Point จำนวนมาก เช่นงาน Vision และ NLP
        </li>
      </ul>
    </div>
  </div>
</section>


<section id="adaptive-methods" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Adaptive Methods (AdaGrad, RMSProp, Adam)</h2>
  <div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

    <p>
      Adaptive Methods ในการ Optimization เป็นวิวัฒนาการสำคัญของอัลกอริทึม Gradient Descent ที่พัฒนาขึ้นมาเพื่อตอบโจทย์ปัญหาความเร็วในการลู่เข้า (Convergence) และความเสถียรในการฝึกโมเดลลึก โดยเฉพาะในกรณีที่ข้อมูลมีความหลากหลายของขนาด (Scale) และลักษณะการกระจาย (Distribution) ของฟีเจอร์แตกต่างกันอย่างมีนัยสำคัญ
    </p>

    <p>
      แนวคิดหลักของ Adaptive Methods คือการปรับ Learning Rate ให้เหมาะสมกับพารามิเตอร์แต่ละตัวตามลักษณะของ Gradient ที่ได้รับระหว่างการฝึก ซึ่งช่วยให้สามารถก้าวไปได้เร็วในมิติที่มี Gradient อ่อน และก้าวเล็กลงในมิติที่ Gradient ผันผวนหรือลาดชันมาก
    </p>

    <h3 className="text-xl font-semibold">AdaGrad (Adaptive Gradient Algorithm)</h3>
    <p>
      AdaGrad เป็นหนึ่งใน Adaptive Methods รุ่นแรกที่นำเสนอโดย John Duchi, Elad Hazan และ Yoram Singer ในปี 2011 แนวคิดหลักของ AdaGrad คือการปรับ Learning Rate แบบลดลงสำหรับพารามิเตอร์ที่มีการอัปเดตบ่อย และรักษา Learning Rate สูงไว้สำหรับพารามิเตอร์ที่อัปเดตน้อย
    </p>

    <p className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      สูตรการอัปเดตของ AdaGrad:
      <br />
      <code className="text-sm">
  gₜ = ∇J(θₜ) <br />
  Gₜ = Gₜ₋₁ + gₜ² <br />
  θₜ₊₁ = θₜ - η * (gₜ / (√Gₜ + ε))
</code>

    </p>

    <p>
      จุดแข็งของ AdaGrad คือเหมาะกับปัญหา Sparse Data เช่น งาน NLP และ Recommendation Systems ที่ฟีเจอร์ส่วนใหญ่มักมีค่าศูนย์และไม่สมมาตร อย่างไรก็ตาม จุดอ่อนสำคัญคือ Learning Rate จะลดลงอย่างรวดเร็วเกินไป ทำให้เมื่อเวลาผ่านไปแล้ว การฝึกหยุดลงก่อนบรรลุ Minimum ที่ดี
    </p>

    <h3 className="text-xl font-semibold">RMSProp (Root Mean Square Propagation)</h3>
    <p>
      RMSProp ถูกพัฒนาขึ้นโดย Geoffrey Hinton เพื่อแก้ไขข้อจำกัดของ AdaGrad โดยใช้ Moving Average ของ Square Gradient แทนการสะสมอย่างไม่มีที่สิ้นสุด ทำให้สามารถควบคุมการลดลงของ Learning Rate ได้อย่างสมเหตุสมผล
    </p>

    <p className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      สูตรการอัปเดตของ RMSProp:
      <br />
      <code className="text-sm">
  gₜ = ∇J(θₜ) <br />
  E[g²]ₜ = γE[g²]ₜ₋₁ + (1−γ)gₜ² <br />
  θₜ₊₁ = θₜ − η * (gₜ / (√E[g²]ₜ + ε))
</code>

    </p>

    <p>
      RMSProp ได้รับความนิยมสูงในการฝึก Recurrent Neural Networks (RNNs) และงานที่มีวัตถุประสงค์ไม่แน่นอน (non-stationary objectives) เช่น Reinforcement Learning เนื่องจากสามารถรักษาความเสถียรของการอัปเดตได้แม้เมื่อข้อมูลมีการเปลี่ยนแปลงตลอดเวลา
    </p>

    <h3 className="text-xl font-semibold">Adam (Adaptive Moment Estimation)</h3>
    <p>
      Adam เป็น Optimizer ที่รวมข้อดีของ Momentum และ RMSProp เข้าด้วยกัน โดยมีการคำนวณค่า Moving Average ทั้งของ Gradient (Momentum) และของ Squared Gradient (Adaptive Scaling) พร้อมกับการทำ Bias Correction เพื่อแก้ปัญหาค่าเฉลี่ยในช่วงต้นของการฝึกที่อาจผิดเพี้ยน
    </p>

    <p className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      สูตรการอัปเดตของ Adam:
      <br />
      <code className="text-sm">
  mₜ = β₁mₜ₋₁ + (1−β₁)gₜ <br />
  vₜ = β₂vₜ₋₁ + (1−β₂)gₜ² <br />
  m̂ₜ = mₜ / (1−β₁ᵗ) <br />
  v̂ₜ = vₜ / (1−β₂ᵗ) <br />
  θₜ₊₁ = θₜ − η * (m̂ₜ / (√v̂ₜ + ε))
</code>

    </p>

    <p>
      Adam ได้รับความนิยมอย่างล้นหลามในงาน Deep Learning โดยเฉพาะงาน Computer Vision และ Natural Language Processing เนื่องจากสามารถฝึกโมเดลที่ซับซ้อนได้รวดเร็ว มีเสถียรภาพสูง และต้องการการจูน Hyperparameter น้อยกว่าวิธีอื่น
    </p>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Adaptive Methods</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border-collapse border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-200 dark:bg-gray-800">
            <th className="border px-4 py-2">Optimizer</th>
            <th className="border px-4 py-2">จุดเด่น</th>
            <th className="border px-4 py-2">จุดอ่อน</th>
            <th className="border px-4 py-2">แนะนำใช้เมื่อ</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">AdaGrad</td>
            <td className="border px-4 py-2">เหมาะกับ sparse data</td>
            <td className="border px-4 py-2">Learning Rate ลดเร็วเกินไป</td>
            <td className="border px-4 py-2">NLP, Sparse Features</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">RMSProp</td>
            <td className="border px-4 py-2">เสถียรในงาน non-stationary</td>
            <td className="border px-4 py-2">มี Hyperparameter เพิ่ม</td>
            <td className="border px-4 py-2">RNNs, Reinforcement Learning</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Adam</td>
            <td className="border px-4 py-2">รวดเร็ว เสถียร เริ่มต้นง่าย</td>
            <td className="border px-4 py-2">อาจพาไป Wrong Minima</td>
            <td className="border px-4 py-2">Deep Learning ทั่วไป</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ดการใช้ Optimizer</h3>
    <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()`}
    </pre>

    <p>
      การเลือก Optimizer ที่เหมาะสมกับลักษณะของปัญหาและข้อมูลมีผลโดยตรงต่อความเร็วในการฝึก และความสามารถของโมเดลในการหา Solution ที่มีประสิทธิภาพ การเข้าใจพื้นฐานของ Adaptive Methods เป็นกุญแจสำคัญในการออกแบบระบบ AI ที่มีเสถียรภาพและประสิทธิภาพสูง
    </p>

  </div>
</section>

<section id="optimizer-comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
<h2 className="text-2xl font-semibold mb-6 text-center">8. ตารางเปรียบเทียบ Optimizer ต่าง ๆ</h2>

<div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
  <p>
    ในการฝึกโมเดล Deep Learning การเลือก Optimizer ที่เหมาะสมถือเป็นหนึ่งในปัจจัยสำคัญที่มีผลต่อความเร็วในการลู่เข้า (convergence speed) ความเสถียร (stability) และประสิทธิภาพโดยรวม (overall performance) ของโมเดล แต่ละ Optimizer มีจุดเด่น จุดด้อย และการใช้งานที่เหมาะสมแตกต่างกันไป การทำความเข้าใจข้อแตกต่างเหล่านี้จึงเป็นกุญแจสำคัญในการออกแบบระบบที่มีประสิทธิภาพสูง
  </p>

  <div className="overflow-x-auto">
    <table className="table-auto w-full text-left border border-gray-300 dark:border-gray-700">
      <thead>
        <tr className="bg-gray-100 dark:bg-gray-800">
          <th className="px-4 py-2 border-b">Optimizer</th>
          <th className="px-4 py-2 border-b">จุดเด่น</th>
          <th className="px-4 py-2 border-b">จุดอ่อน</th>
          <th className="px-4 py-2 border-b">ใช้เมื่อไร</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="px-4 py-2 border-b">SGD</td>
          <td className="px-4 py-2 border-b">Simple, Robust, Low Memory</td>
          <td className="px-4 py-2 border-b">Convergence ช้า, Oscillation สูง</td>
          <td className="px-4 py-2 border-b">เมื่อ dataset ใหญ่มาก ต้องการควบคุมละเอียด</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border-b">Momentum</td>
          <td className="px-4 py-2 border-b">เร่งการลู่เข้า, ลดการแกว่ง</td>
          <td className="px-4 py-2 border-b">ต้องจูนค่า Beta อย่างเหมาะสม</td>
          <td className="px-4 py-2 border-b">Training Deep Networks</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border-b">Nesterov</td>
          <td className="px-4 py-2 border-b">Lookahead การอัปเดต, เสถียรกว่า Momentum</td>
          <td className="px-4 py-2 border-b">ซับซ้อนกว่า ต้องเข้าใจหลักการ</td>
          <td className="px-4 py-2 border-b">โมเดลซับซ้อน หรือต้องการเร่งความเร็ว</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border-b">AdaGrad</td>
          <td className="px-4 py-2 border-b">ปรับ Learning Rate อัตโนมัติ, เหมาะกับ Sparse Data</td>
          <td className="px-4 py-2 border-b">Learning Rate ลดลงเร็วเกินไป</td>
          <td className="px-4 py-2 border-b">งาน NLP หรือ Sparse Feature</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border-b">RMSProp</td>
          <td className="px-4 py-2 border-b">รักษา Learning Rate, เหมาะกับ RNN</td>
          <td className="px-4 py-2 border-b">อาจ Overfit ในบางกรณี</td>
          <td className="px-4 py-2 border-b">Non-stationary Objectives เช่น Time Series</td>
        </tr>
        <tr>
          <td className="px-4 py-2 border-b">Adam</td>
          <td className="px-4 py-2 border-b">Fast Convergence, Adaptive, Default Choice</td>
          <td className="px-4 py-2 border-b">อาจติด Local Minima ที่ไม่ดี</td>
          <td className="px-4 py-2 border-b">Deep Learning ทั่วไป, ข้อมูลขนาดใหญ่</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 className="text-xl font-semibold mt-10">ตัวอย่างการใช้งาน Optimizer จริง</h3>
  <div className="grid md:grid-cols-2 gap-6">
    <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="text-lg font-medium mb-2">Training CNN บน CIFAR-10</h4>
      <ul className="list-disc pl-5 space-y-2 text-sm">
        <li>เริ่มด้วย Adam เพื่อเร่งการลู่เข้า</li>
        <li>เมื่อ Accuracy เริ่มนิ่ง → ปรับเป็น SGD + Momentum เพื่อ Fine-tune</li>
        <li>ใช้ Learning Rate Decay เพื่อลด Learning Rate ตาม Epoch</li>
      </ul>
    </div>
    <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow border border-gray-300 dark:border-gray-700">
      <h4 className="text-lg font-medium mb-2">Training RNN สำหรับ Time-Series</h4>
      <ul className="list-disc pl-5 space-y-2 text-sm">
        <li>เลือกใช้ RMSProp เพราะเหมาะกับ Sequence ที่เปลี่ยนตลอดเวลา</li>
        <li>ถ้า Overfit → ลด Batch Size หรือใช้ Dropout</li>
        <li>ใช้ Gradient Clipping เพื่อป้องกัน Exploding Gradient</li>
      </ul>
    </div>
  </div>

  <h3 className="text-xl font-semibold mt-10">Insight จากงานวิจัย</h3>
  <div className="bg-yellow-100 dark:bg-yellow-900 p-5 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
    <p className="text-sm">
      งานวิจัยของ Kingma และ Ba (2014) ที่นำเสนอ Adam Optimizer แสดงให้เห็นว่า การรวม Momentum และ Adaptive Learning Rate เป็นกุญแจสำคัญที่ทำให้ Adam เหนือกว่าเทคนิคเดิมในงาน Vision และ NLP สมัยใหม่ โดยเฉพาะอย่างยิ่งในโมเดลขนาดใหญ่ เช่น Transformers และ BERT
    </p>
  </div>

  <p>
    แม้ Adam จะได้รับความนิยมสูงสุดในปัจจุบัน แต่ไม่มี Optimizer ใดที่เหมาะกับทุกปัญหา การเลือกควรพิจารณาจากลักษณะของข้อมูล ประเภทของโมเดล และเป้าหมายของการเรียนรู้ การทดสอบและปรับแต่ง (Hyperparameter Tuning) ยังเป็นส่วนสำคัญในการหาแนวทางที่ดีที่สุดสำหรับแต่ละงาน
  </p>
</div>
</section>

<section id="insight" className="mb-16 scroll-mt-32 min-h-[400px]">
<h2 className="text-2xl font-semibold mb-6 text-center">9. Insight เชิงลึก</h2>
<div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">
  <p>
    การพัฒนาเทคนิคการปรับปรุง Gradient Descent ในช่วงหลายทศวรรษที่ผ่านมา เป็นหัวใจสำคัญที่ผลักดันให้การฝึก Neural Networks มีประสิทธิภาพสูงขึ้นในทางปฏิบัติ โดยเฉพาะอย่างยิ่งในงานที่โมเดลมีขนาดใหญ่หรือข้อมูลมีความซับซ้อนสูง แนวคิดเช่น Learning Rate Scheduling, Warm Restarts และ Cyclic Learning Rate ได้รับการยอมรับอย่างกว้างขวางจากสถาบันชั้นนำ เช่น Stanford, MIT, และ Google Brain ว่าสามารถช่วยเร่งการลู่เข้าของโมเดลได้อย่างมีนัยสำคัญ
  </p>

  <h3 className="text-xl font-semibold mt-8">Learning Rate Scheduling</h3>
  <p>
    การเปลี่ยนค่า Learning Rate ตามเวลาฝึกช่วยให้โมเดลสามารถลู่เข้า (converge) ได้เร็วขึ้นและมีโอกาสเข้าสู่ minimum ที่ดีขึ้น ตัวอย่างเทคนิคสำคัญได้แก่:
  </p>
  <ul className="list-disc pl-6 space-y-2">
    <li><strong>Step Decay:</strong> ลดค่า Learning Rate แบบขั้นบันไดหลังผ่านจำนวน epoch ที่กำหนด เช่น ลดลงครึ่งหนึ่งทุก 10 epochs</li>
    <li><strong>Exponential Decay:</strong> ลดค่า Learning Rate ตามสูตรกำลัง เช่น η = η₀ exp(-kt)</li>
    <li><strong>Cosine Annealing:</strong> ลดค่า Learning Rate ตามรูปโคไซน์ เพื่อให้ลดลงอย่างนุ่มนวลสู่ศูนย์เมื่อถึงปลายการฝึก</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8">Warm Restarts</h3>
  <p>
    เทคนิค Warm Restarts หรือ Cosine Annealing with Restarts ได้รับความนิยมจากงานวิจัยของ Loshchilov และ Hutter (2017) ซึ่งนำเสนอแนวทางในการรีเซ็ตค่า Learning Rate กลับไปที่ค่าสูงในบางจุดระหว่างการฝึก ช่วยให้โมเดลหลีกเลี่ยงการติดอยู่ใน local minima และมีโอกาสค้นหา minima ที่ดีกว่า
  </p>
  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-xl border-l-4 border-blue-400 dark:border-blue-600">
    <p className="text-sm">
      <strong>สูตร Warm Restarts:</strong> η(t) = η_min + 0.5 (η_max - η_min)(1 + cos(π t/T))
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8">Cyclic Learning Rate</h3>
  <p>
    Leslie N. Smith แห่ง Johns Hopkins University เสนอแนวคิด Cyclic Learning Rate (CLR) โดยไม่ต้องตั้งค่า Learning Rate แบบตายตัว แต่ใช้การแกว่งขึ้นลงระหว่างค่าต่ำสุดและค่าสูงสุดภายในรอบที่กำหนด ช่วยให้โมเดลสามารถสำรวจพื้นผิวของ Loss Function ได้กว้างขึ้นและมีโอกาสหนีจาก Saddle Points หรือ Sharp Minima
  </p>
  <ul className="list-disc pl-6 space-y-2">
    <li><strong>Triangular Policy:</strong> Learning Rate เพิ่มแล้วลดแบบเส้นตรงภายในรอบ</li>
    <li><strong>Triangular2 Policy:</strong> ลด Learning Rate สูงสุดครึ่งหนึ่งในแต่ละรอบ</li>
    <li><strong>Exp Range:</strong> Learning Rate ลดตามสูตรกำลังพร้อมกับแกว่งขึ้นลง</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8">No Free Lunch Theorem</h3>
  <p>
    งานวิจัยของ David Wolpert และ William Macready (1997) ชี้ให้เห็นว่าไม่มีอัลกอริทึมใดที่ดีที่สุดสำหรับปัญหาทุกประเภทในโลกของ Optimization ทฤษฎีนี้มีผลอย่างมากต่อการเลือก Optimizer ใน Deep Learning เพราะหมายความว่าการเลือก Optimizer จะต้องขึ้นอยู่กับลักษณะเฉพาะของปัญหาแต่ละอย่าง
  </p>
  <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
    <p className="text-sm">
      <strong>Insight:</strong> Adam อาจทำงานได้ดีในปัญหาทั่วไป แต่ SGD + Momentum อาจให้ผลดีกว่าในบางงานที่ต้องการการ generalize สูง เช่น Computer Vision บางประเภท
    </p>
  </div>

  <h3 className="text-xl font-semibold mt-8">แนวทางปฏิบัติที่แนะนำ (Best Practices)</h3>
  <ul className="list-disc pl-6 space-y-2">
    <li>เริ่มต้นด้วย Adam เมื่อยังไม่มีข้อมูลที่ชัดเจนเกี่ยวกับลักษณะของปัญหา</li>
    <li>ปรับ Learning Rate เป็นตัวแรกที่ควรปรับเมื่อการฝึกไม่เป็นไปตามที่คาดหวัง</li>
    <li>ใช้ Learning Rate Warm-up ในโมเดลขนาดใหญ่ เช่น Transformer หรือ ResNet-152</li>
    <li>หากพบว่า Validation Loss แกว่งรุนแรง ให้พิจารณาใช้ Learning Rate Schedulers</li>
    <li>ถ้าโมเดล Overfitting มาก ลองเปลี่ยน Optimizer จาก Adam เป็น SGD + Momentum และใช้ Early Stopping</li>
  </ul>

  <h3 className="text-xl font-semibold mt-8">ตัวอย่างการใช้ Learning Rate Schedulers ใน PyTorch</h3>
  <pre className="bg-gray-800 text-white text-sm p-4 rounded-xl overflow-x-auto">
{`import torch
import torch.optim as optim

model = ...
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(100):
  train(model)
  validate(model)
  scheduler.step()`}
  </pre>

  <h3 className="text-xl font-semibold mt-8">สรุป</h3>
  <p>
    การเลือกใช้กลยุทธ์การอัปเดต Learning Rate และการเลือก Optimizer อย่างชาญฉลาดเป็นกุญแจสำคัญที่ส่งผลโดยตรงต่อความเร็วในการลู่เข้าและคุณภาพของโมเดลขั้นสุดท้าย แนวทางต่าง ๆ ที่กล่าวถึงในหัวข้อนี้ล้วนได้รับการพิสูจน์ทั้งเชิงทฤษฎีและในงานจริงว่ามีผลกระทบอย่างมีนัยสำคัญต่อประสิทธิภาพของ Deep Learning Models
  </p>
</div>
</section>

<section id="special-box" className="mb-16 scroll-mt-32 min-h-[400px]">
<h2 className="text-2xl font-semibold mb-6 text-center">Special Box: Best Practices</h2>
<div className="prose dark:prose-invert max-w-none text-base leading-relaxed space-y-6">

  <div className="bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 p-6 rounded-xl border-l-4 border-yellow-400 dark:border-yellow-600">
    <h3 className="text-xl font-semibold mb-4">แนวทางปฏิบัติที่แนะนำสำหรับการเลือก Optimizer และการตั้งค่า</h3>
    <p>
      จากการรวบรวมข้อมูลจากแหล่งระดับโลก เช่น DeepMind, Google Brain, OpenAI และ MIT CSAIL พบว่าการเลือก Optimizer ที่เหมาะสมและการตั้งค่าที่ถูกต้องมีผลโดยตรงต่อประสิทธิภาพและเสถียรภาพของโมเดล โดยเฉพาะในการฝึกเครือข่ายขนาดใหญ่หรือข้อมูลที่มีความซับซ้อนสูง
    </p>

    <h4 className="text-lg font-semibold mt-6">1. เริ่มต้นด้วย Adam Optimizer</h4>
    <p>
      งานวิจัยของ Kingma และ Ba (2015) แนะนำให้เริ่มต้นด้วย Adam เนื่องจากสามารถรวมข้อดีของ Momentum และ RMSProp ได้อย่างมีประสิทธิภาพ Adam เหมาะกับปัญหาทั่วไป ทั้ง classification และ regression และสามารถทำงานได้ดีแม้ในงานที่มี noise สูงหรือกรณีที่ gradient มีการเปลี่ยนแปลงไม่สม่ำเสมอ
    </p>

    <h4 className="text-lg font-semibold mt-6">2. ปรับ Learning Rate อย่างระมัดระวัง</h4>
    <p>
      Learning Rate เป็นหนึ่งใน Hyperparameter ที่สำคัญที่สุด การตั้งค่า Learning Rate ที่สูงเกินไปจะทำให้โมเดลไม่เสถียรหรือไม่สามารถลู่เข้าได้ ขณะที่ค่าต่ำเกินไปจะทำให้การฝึกช้ามาก แนวทางที่ได้รับการแนะนำจาก Stanford CS231n คือเริ่มจาก 0.001 สำหรับ Adam และทำการปรับลดลงตามความจำเป็น เช่น การใช้ Learning Rate Decay หรือ Scheduler แบบ Cosine Annealing
    </p>

    <h4 className="text-lg font-semibold mt-6">3. ใช้ Mini-Batch ขนาดกลาง</h4>
    <p>
      จากการศึกษาของ Facebook AI Research (FAIR) และ Berkeley AI Research (BAIR) การเลือก batch size ที่พอดีมีผลต่อความเสถียรของการฝึก Mini-Batch Gradient Descent ขนาด 32-256 ตัวอย่างมักให้ผลลัพธ์ที่ดีในทางปฏิบัติ โดย batch size เล็กเพิ่ม stochasticity ซึ่งช่วยให้โมเดลหลีกเลี่ยง local minima แต่ขนาดที่เล็กเกินไปอาจทำให้ convergence ช้า
    </p>

    <h4 className="text-lg font-semibold mt-6">4. พิจารณาใช้ Gradient Clipping</h4>
    <p>
      ในงานที่ต้องฝึกเครือข่ายลึกหรือ RNNs การใช้ Gradient Clipping เป็นเทคนิคที่มีประสิทธิภาพในการป้องกัน Exploding Gradients โดยการจำกัด norm ของ gradient ให้ไม่เกินค่าที่กำหนด เช่น 1.0 การตั้งค่าอย่างเหมาะสมช่วยรักษาเสถียรภาพของการฝึกได้อย่างมาก
    </p>

    <h4 className="text-lg font-semibold mt-6">5. ตรวจสอบและวิเคราะห์ Learning Curve</h4>
    <p>
      การติดตาม Loss และ Accuracy บน Training และ Validation Set เป็นสิ่งสำคัญในการวินิจฉัยปัญหา เช่น Overfitting หรือ Underfitting การสังเกต Learning Curve อย่างสม่ำเสมอสามารถช่วยระบุเวลาที่เหมาะสมในการปรับ Learning Rate หรือเปลี่ยน Optimizer
    </p>

    <h4 className="text-lg font-semibold mt-6">6. ใช้ Warm Restarts และ Learning Rate Scheduling</h4>
    <p>
      เทคนิคอย่าง SGDR (Stochastic Gradient Descent with Warm Restarts) หรือการใช้ Cosine Annealing สามารถช่วยให้โมเดลหลีกเลี่ยง Saddle Points และ Local Minima ได้โดยการเปลี่ยนแปลง Learning Rate แบบเป็นคาบ เพิ่มโอกาสการค้นหา optimal basin ที่ดีกว่า
    </p>

    <h4 className="text-lg font-semibold mt-6">7. หลีกเลี่ยงการใช้ Default Hyperparameters ตลอดเวลา</h4>
    <p>
      แม้ว่าค่าเริ่มต้นในไลบรารี เช่น PyTorch หรือ TensorFlow จะเหมาะสมในหลายกรณี แต่การปรับแต่ง Hyperparameters เฉพาะสำหรับงานที่ทำจริงยังคงมีความสำคัญ เช่น ปรับค่า beta ใน Adam (β₁ = 0.9, β₂ = 0.999) หรือ epsilon เพื่อความเสถียรในการคำนวณ
    </p>

    <h4 className="text-lg font-semibold mt-6">8. เลือก Optimizer ตามลักษณะข้อมูล</h4>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>SGD + Momentum:</strong> เหมาะสำหรับการฝึกเครือข่ายใหญ่ ๆ ที่ต้องการ generalization สูง เช่น Image Classification บน ImageNet</li>
      <li><strong>Adam:</strong> ตัวเลือกที่ดีสำหรับงานทั่วไป, NLP, และข้อมูลที่ noisy</li>
      <li><strong>RMSProp:</strong> นิยมในงานที่ objective เปลี่ยนแปลงตลอดเวลา เช่น reinforcement learning</li>
    </ul>

    <h4 className="text-lg font-semibold mt-6">9. ใช้ Early Stopping ร่วมกับ Optimizer Choice</h4>
    <p>
      การหยุดการฝึกเมื่อ Validation Loss ไม่ลดลงอีกต่อไปช่วยลดโอกาส Overfitting การรวม Early Stopping เข้ากับ Optimizer ที่เลือก เช่น Adam หรือ SGD + Momentum สามารถปรับปรุงประสิทธิภาพโดยรวมของโมเดลได้อย่างมาก
    </p>

    <h4 className="text-lg font-semibold mt-6">10. การเรียนรู้ต่อเนื่องใน Training Strategies</h4>
    <p>
      ในกรณีที่ทำงานกับ Data Streams หรือ Lifelong Learning การเปลี่ยน Optimizer ตามช่วงเวลา เช่น เริ่มจาก Adam แล้วเปลี่ยนไปใช้ SGD หลังจากโมเดลเข้าใกล้ค่าต่ำสุด สามารถช่วยเพิ่มการ generalization ได้
    </p>

  </div>

</div>
</section>

        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
          <MiniQuiz_Day19 theme={theme} />
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
        <ScrollSpy_Ai_Day19 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day19_GradientDescentVariants;
