import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../components/common/Comments";
import SupportMeButton from "../../../support/SupportMeButton";
import ScrollSpy_Ai_Day5 from "./scrollspy/ScrollSpy_Ai_Day5";
import MiniQuiz_Day5 from "./miniquiz/MiniQuiz_Day5";

const Day5_LinearTransform = ({ theme }) => {
  const navigate = useNavigate();

  return (
    <div
      className={`relative min-h-screen ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
      }`}
    >
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <h1 className="text-3xl font-bold mb-6">
          Day 5: Linear Transformation & Feature Extraction
        </h1>

        <section id="what-is-linear-transform" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-6">Linear Transformation คืออะไร?</h2>


  <img
    src="/LinearTransformation.png"
    alt="Linear Transformation "
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />


  <p className="mb-4 text-lg">
    Linear Transformation คือการแปลงเวกเตอร์จากพิกัดหนึ่งไปยังอีกพิกัดหนึ่ง โดยผ่านการคูณกับเมทริกซ์  
    การแปลงนี้จะ <strong>รักษาโครงสร้างเชิงเส้น</strong> ของข้อมูล เช่น จุดที่อยู่บนเส้นตรง จะยังคงอยู่บนเส้นตรง  
    และจุดที่อยู่ใกล้กัน จะยังอยู่ใกล้กันหลังการแปลง
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-200 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mb-6">
    <strong>Insight:</strong> Linear Transformation คือ “พื้นฐาน” ของการเปลี่ยนข้อมูลจากดิบ → เป็น Feature ที่มีความหมาย  
    เป็นแก่นของ Convolution, Embedding, Self-Attention และอีกมากมายครับ
  </div>

  <h3 className="mb-4 text-center">
    ตัวอย่างที่พบได้ทั่วไปของ Linear Transformation 
  </h3>

  <img
    src="/LinearTransformation4.png"
    alt="Linear Transformation "
    className="w-full max-w-lg mx-auto mb-6 rounded-xl shadow-md border border-yellow-400"
  />
  <ul className="list-disc pl-6 space-y-2 mb-6">
    <li><strong>การหมุน (Rotation):</strong> หมุนเวกเตอร์รอบต้นกำเนิด</li>
    <li><strong>การยืด/หด (Scaling):</strong> ยืดเวกเตอร์ในแกนใดแกนหนึ่ง</li>
    <li><strong>การสะท้อน (Reflection):</strong> พลิกเวกเตอร์กับแกน</li>
    <li><strong>การเปลี่ยนมิติ (Projection):</strong> ฉายเวกเตอร์ลงบนแกนที่สนใจ</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">เวกเตอร์ถูกแปลงได้อย่างไร?</h3>
  <p className="mb-4">
    ลองจินตนาการว่าเรามีเวกเตอร์พื้นฐาน [1, 0] และ [0, 1] ซึ่งเป็นแกน X และ Y ตามลำดับ  
    ถ้าเราคูณมันด้วยเมทริกซ์ที่เปลี่ยนขนาดของแกน X เป็น 2 เท่า:
  </p>

  <pre className="bg-gray-800 text-white p-3 text-sm rounded mb-4 overflow-x-auto">import numpy as np

M = np.array([[2, 0], [0, 1]])  # ยืดในแกน X
v1 = np.array([1, 0])
v2 = np.array([0, 1])

print(M @ v1)  # Output: [2 0]
print(M @ v2)  # Output: [0 1]</pre>

  <p className="mb-4">
    จะเห็นได้ว่าเวกเตอร์แกน X ถูกยืดออก 2 เท่า แต่แกน Y ไม่เปลี่ยน — นี่คือลักษณะของ Linear Transformation  
    เราแค่เปลี่ยน "เลนส์การมองข้อมูล" แต่ไม่ได้บิดเบือนโครงสร้างพื้นฐาน
  </p>


  <h3 className="text-xl font-semibold mb-3 text-center">คุณสมบัติสำคัญของ Linear Transformation</h3>
  <img
    src="/LinearTransformation5.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />
  <ul className="list-decimal pl-6 space-y-2 mb-6">
    <li><strong>Preserves Origin:</strong> จุดเริ่มต้น (0,0) จะยังคงอยู่ที่เดิม</li>
    <li><strong>Preserves Linearity:</strong> จุดที่อยู่บนเส้นตรงก่อนแปลง จะยังอยู่บนเส้นตรงหลังแปลง</li>
    <li><strong>Preserves Parallelism:</strong> เส้นที่ขนานกัน จะยังขนานกัน</li>
    <li><strong>Can Change Length & Direction:</strong> ยืดหรือหมุนเวกเตอร์ได้ แต่ไม่บิดโค้ง</li>
  </ul>

  <h3 className="text-xl font-semibold mb-3">ความสำคัญใน AI</h3>
  <p className="mb-4">
    ในโมเดล AI เช่น Neural Network → ทุกเลเยอร์ที่เป็น Linear หรือ Dense จะใช้เมทริกซ์คูณกับข้อมูล  
    ซึ่งก็คือการทำ Linear Transformation ซ้ำแล้วซ้ำอีก จนข้อมูลกลายเป็น Feature ที่ใช้ในการทำนาย
  </p>

  <pre className="bg-gray-800 text-white p-3 text-sm rounded mb-4 overflow-x-auto">import torch
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)
x = torch.tensor([1.0, 0.5, -1.0, 2.0])
output = layer(x)</pre>

  <p className="mb-4">
    โค้ดด้านบนคือ Linear Transformation ในโลกจริง — ข้อมูลขนาด 4 มิติ  
    ถูกแปลงด้วยเมทริกซ์ภายในเลเยอร์ → กลายเป็นเวกเตอร์ 2 มิติใหม่ ที่ถูกใช้ต่อในโมเดล
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    ทุกครั้งที่โมเดลเรียนรู้เวกเตอร์ใหม่ → จริง ๆ แล้วมันเรียนรู้ “การเปลี่ยนมุมมอง” ผ่านเมทริกซ์  
    Linear Transformation ไม่ใช่เพียงการคูณ แต่คือกระบวนการเข้าใจโลกในอีกมิติ
  </div>

  <p className="mt-6">
     ในหัวข้อถัดไป เราจะดูว่า Linear Transformation ช่วยให้เรา "ดึง Feature" จากข้อมูลได้อย่างไร  
    ไม่ว่าจะเป็นขอบของภาพ, ความหมายของคำ, หรือโทนเสียงในเสียงพูด
  </p>
</section>

<section id="matrix-properties" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">คุณสมบัติของเมทริกซ์ และข้อจำกัดของ Linear Transformation</h2>

  <img
    src="/LinearTransformation6.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4">
    แม้ว่า Linear Transformation จะเป็นกลไกหลักในการแปลงข้อมูล แต่ก็ยังมีข้อจำกัด เพราะสามารถสร้างเฉพาะรูปแบบเชิงเส้น เช่น การยืด หมุน หรือสะท้อนเท่านั้น ไม่สามารถเรียนรู้ฟังก์ชันที่ซับซ้อนหรือไม่เป็นเส้นตรงได้
  </p>

  <p className="mb-4">
    นี่คือเหตุผลว่าทำไมใน Neural Network จึงต้องใส่ <strong>Activation Function</strong> เช่น ReLU หรือ Sigmoid หลังจากการคูณเมทริกซ์ เพื่อเพิ่มความสามารถในการเรียนรู้รูปแบบที่ซับซ้อน เช่น ขอบโค้ง, เงา, หรือลักษณะทางไวยากรณ์ที่ซับซ้อน
  </p>

  <div className="bg-yellow-50 dark:bg-yellow-900 text-black dark:text-yellow-200 p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การเรียนรู้ที่แท้จริงเริ่มจากเมทริกซ์ แต่สมบูรณ์ด้วย non-linearity  
    Linear Transformation เปรียบเหมือน "ร่างกาย" ส่วน Activation คือ "จิตวิญญาณ" ของระบบอัจฉริยะ
  </div>
</section>

<section id="layer-wise-transform" className="mb-16 scroll-mt-32 ">
  <h2 className="text-2xl font-semibold mb-3 text-center">การคูณเมทริกซ์ในหลายเลเยอร์ของโมเดล</h2>

  <img
    src="/LinearTransformation7.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4">
    ใน Neural Network ข้อมูลจะถูกแปลงซ้ำหลายรอบด้วยเมทริกซ์ของแต่ละเลเยอร์ โดยข้อมูลจะเปลี่ยนไปในแต่ละรอบ เพื่อเน้น feature ที่สำคัญมากขึ้นเรื่อย ๆ จากดิบ → ขอบ → โครงสร้าง → ความเข้าใจ
  </p>

  <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import torch
import torch.nn as nn

model = nn.Sequential(
  nn.Linear(784, 256),
  nn.ReLU(),
  nn.Linear(256, 128),
  nn.ReLU(),
  nn.Linear(128, 10)
)

x = torch.randn(1, 784)
output = model(x)`}</pre>

  <p className="mb-4">
    ในตัวอย่างด้านบน ข้อมูลภาพถูกแปลงผ่านเมทริกซ์ 3 รอบ → จากขนาด 784 (input) → 256 → 128 → 10  
    ทุกรอบคือการคูณเมทริกซ์และใช้ Activation Function แทรกเพื่อเพิ่มความสามารถในการเรียนรู้
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การคูณเมทริกซ์ในแต่ละเลเยอร์ไม่ใช่การทำซ้ำ แต่คือการก้าวลึกเข้าสู่ "มิติความเข้าใจ" ที่ลึกขึ้นเรื่อย ๆ ในข้อมูล
  </div>
</section>


<section id="compare-raw-feature" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-6 text-center">เปรียบเทียบ Input ดิบ vs. Transformed Feature</h2>
  <img
    src="/LinearTransformation8.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4 text-base leading-relaxed">
    ข้อมูลที่เข้าสู่ระบบ AI มักเริ่มต้นจากรูปแบบดิบ เช่น:
  </p>

  <ul className="list-disc pl-6 mb-6 space-y-2">
    <li><strong>ภาพ:</strong> พิกเซลแบบ RGB ที่มีค่าระหว่าง 0–255</li>
    <li><strong>เสียง:</strong> ความถี่ในรูปคลื่น waveform</li>
    <li><strong>ข้อความ:</strong> One-hot vector ของคำ หรือ token embedding</li>
  </ul>

  <p className="mb-4 text-base leading-relaxed">
    ข้อมูลเหล่านี้ไม่สามารถใช้งานได้โดยตรงสำหรับการตัดสินใจหรือวิเคราะห์ จึงต้องผ่านกระบวนการแปลงด้วยเมทริกซ์เพื่อสร้าง <strong>feature ใหม่</strong> ที่มีความหมายมากขึ้น
  </p>

 
  <p className="mb-4">
    เป้าหมายคือการแยกแยะสิ่งที่สำคัญออกจากข้อมูลที่ไม่จำเป็น เช่น จากภาพทั้งหมด → เน้นที่ <strong>ขอบของวัตถุ</strong>, จากเสียงพูด → เน้นที่ <strong>ความถี่เฉพาะตัว</strong> ของเสียงแต่ละคน
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ภาพ → ขอบ (Edge)</h3>
  <p className="mb-4">
    ในงาน Computer Vision → พิกเซลดิบของภาพจะถูกแปลงด้วยเมทริกซ์ฟิลเตอร์ เช่น Sobel หรือ Convolution Kernel  
    เพื่อดึงข้อมูลเกี่ยวกับขอบ, รูปทรง หรือพื้นผิว
  </p>

  <pre className="bg-gray-800 text-white text-sm rounded-md p-4 overflow-x-auto mb-4"># Example of simple edge filter
import numpy as np
from scipy.ndimage import convolve

image = np.array([[255, 255, 255], [0, 0, 0], [255, 255, 255]])
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

edge = convolve(image, sobel_x)
print(edge)</pre>

  <h3 className="text-lg font-semibold mt-6 mb-2">เสียง → ความถี่ (Frequency)</h3>
  <p className="mb-4">
    สัญญาณเสียง waveform ที่ดูเหมือนไม่มีรูปแบบ จะถูกแปลงด้วยเมทริกซ์ Fourier หรือ Mel filter  
    เพื่อแยกข้อมูลเสียงเป็นลักษณะเฉพาะ เช่น พยางค์, ความสูงของเสียง หรือจังหวะ
  </p>

  <h3 className="text-lg font-semibold mt-6 mb-2">ข้อความ → ความหมาย (Semantic)</h3>
  <p className="mb-4">
    คำที่อยู่ในประโยคจะถูกแปลงเป็นเวกเตอร์ เช่น GloVe, Word2Vec หรือ BERT embeddings  
    เมทริกซ์ weight ใน Neural Network จะช่วยให้เวกเตอร์เหล่านี้สะท้อน <strong>ความหมายเชิงบริบท</strong> มากยิ่งขึ้น
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การแปลงข้อมูลดิบด้วยเมทริกซ์ คือการฉายข้อมูลสู่มุมใหม่ที่ซ่อนความหมายอยู่  
    เป็นจุดเริ่มต้นของความเข้าใจที่ลึกขึ้นใน AI ทุกแขนง ไม่ว่าจะเป็นภาพ, ภาษา, หรือเสียง
  </div>
</section>


<section id="feature-extraction" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3 text-center">การดึง Feature จากภาพ ข้อความ และเสียง</h2>

  <img
    src="/LinearTransformation9.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4">
    Feature Extraction คือกระบวนการดึงคุณลักษณะสำคัญจากข้อมูลดิบ เพื่อให้ระบบสามารถเรียนรู้และตัดสินใจได้ดีขึ้น โดยข้อมูลที่ผ่านการแปลงแล้วจะอยู่ในรูปของเวกเตอร์ที่เก็บข้อมูลเฉพาะทาง ซึ่งสะท้อนความหมายที่ลึกซึ้งกว่าข้อมูลดิบ
  </p>

  <div className="mb-6">
    <h3 className="text-xl font-semibold mt-4 mb-2"> ภาพ (Image)</h3>
    <p className="mb-2">
      ในระบบประมวลผลภาพ เช่น CNN (Convolutional Neural Network) จะมีการดึง feature จาก pixel ดิบ เช่น ขอบ (edge), พื้นผิว (texture), รูปร่างเฉพาะ (pattern) โดยการใช้เมทริกซ์ convolution หรือ filter สไลด์ผ่านภาพ
    </p>
    <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`# ตัวอย่างใน PyTorch
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
input_image = torch.randn(1, 1, 28, 28)  # ขนาดภาพ 28x28
features = conv(input_image)`}</pre>
    <p className="mb-4">
      หลังจากผ่าน convolution → ภาพต้นฉบับจะกลายเป็น feature maps ที่เน้นบางโครงสร้าง เช่น เส้นขอบแนวนอน แนวตั้ง หรือรูปทรงที่ซับซ้อนขึ้นในเลเยอร์ถัดไป
    </p>
 
  </div>

  <div className="mb-6">
    <h3 className="text-xl font-semibold mt-4 mb-2">ข้อความ (Text)</h3>
    <p className="mb-2">
      ในงานด้าน NLP (Natural Language Processing) การแปลงคำหรือประโยคให้เป็นเวกเตอร์ที่มีความหมายเป็นพื้นฐานสำคัญ เช่น Word2Vec, BERT, หรือ GPT จะเปลี่ยนคำดิบให้กลายเป็น embedding ที่แสดงความหมายของคำในบริบท
    </p>
    <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Natural Language", return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state`}</pre>
    <p className="mb-4">
      เวกเตอร์ที่ได้จาก BERT จะแสดงบริบทของคำ เช่น คำว่า “bank” ในบริบทต่าง ๆ จะมี embedding ต่างกัน ขึ้นกับประโยคที่อยู่รอบ ๆ
    </p>

  </div>

  <div className="mb-6">
    <h3 className="text-xl font-semibold mt-4 mb-2"> เสียง (Audio)</h3>
    <p className="mb-2">
      ข้อมูลเสียงสามารถเปลี่ยนให้เป็นเวกเตอร์ของความถี่ผ่านเทคนิคอย่าง FFT หรือ Mel Spectrogram → ช่วยให้โมเดลเข้าใจ pitch, tone, และจังหวะของเสียงที่เกี่ยวข้องกับภาษา พูด หรืออารมณ์
    </p>
    <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import librosa
import numpy as np

signal, sr = librosa.load("voice.wav")
mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
log_mel = librosa.power_to_db(mel, ref=np.max)`}</pre>
    <p className="mb-4">
      ข้อมูลเสียงที่ผ่าน Mel Spectrogram จะกลายเป็นเหมือนภาพ 2D ที่สามารถส่งเข้าโมเดลภาพได้ หรือใช้วิเคราะห์ลักษณะเสียงโดยตรง
    </p>
   
  </div>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    Feature ที่ดีไม่ใช่เพียงค่าที่ผ่านการคำนวณ แต่คือการเปลี่ยนมุมมองของข้อมูลให้เหมาะสมกับการเรียนรู้ในแต่ละโมเดล การดึง feature เปรียบเหมือนการแปลภาษาของข้อมูลให้เข้าใจง่ายขึ้นสำหรับระบบอัจฉริยะ
  </div>
</section>


<section id="vector-change" className="mb-16 scroll-mt-32 ">
  <h2 className="text-2xl font-semibold mb-3 text-center">เวกเตอร์เปลี่ยนอย่างไรหลังจาก Transformation?</h2>
  <img
    src="/LinearTransformation10.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />

  <p className="mb-4">
    Linear Transformation ไม่ได้เปลี่ยนแค่ค่าตัวเลขของเวกเตอร์ แต่เป็นการเปลี่ยน “มุมมอง” หรือ “ทิศทาง” ของข้อมูลโดยตรง → เป็นหัวใจสำคัญของการเรียนรู้ในระบบ AI
  </p>

  <p className="mb-4">
    เวกเตอร์ต้นฉบับสามารถแสดงเป็นลูกศรในพื้นที่ 2D หรือ 3D ซึ่งจะมีความยาวและทิศทางที่แน่นอน การนำเมทริกซ์มาคูณกับเวกเตอร์เหล่านี้ → จะส่งผลต่อรูปร่าง เช่น การหมุน (rotation), การยืด/หด (scaling), หรือการสะท้อน (reflection)
  </p>

  

  <p className="mt-6">
    การทดลองด้านบนสามารถใช้ปรับค่าเมทริกซ์ (Matrix) แบบเรียลไทม์ แล้วดูผลลัพธ์ของเวกเตอร์ที่เปลี่ยนไปอย่างชัดเจน เช่น:
  </p>

  <ul className="list-disc pl-6 space-y-2 mt-2">
    <li>ค่าบนแนวทแยงหลัก → ส่งผลต่อการยืดหดในแนวแกน X/Y</li>
    <li>ค่าที่ไม่เป็นศูนย์นอกแนวทแยง → ส่งผลให้เกิดการหมุน หรือ shearing</li>
    <li>ค่าติดลบ → สร้างการสะท้อนกลับ (reflection) บนแกนใดแกนหนึ่ง</li>
  </ul>

  <div className="mt-6">
    <h3 className="text-lg font-semibold mb-2">ตัวอย่างที่ 1: ยืดเวกเตอร์</h3>
    <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`import numpy as np

v = np.array([2, 1])
M = np.array([[2, 0], [0, 1]])
result = M @ v
print(result)  # Output: [4, 1]`}</pre>
    <p className="mb-4">
      เวกเตอร์ถูกยืดออกในแนวแกน X เท่าตัว → แสดงถึงการเน้นความสำคัญในแกน X
    </p>

    <h3 className="text-lg font-semibold mb-2">ตัวอย่างที่ 2: หมุนเวกเตอร์</h3>
    <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`theta = np.pi / 4  # หมุน 45 องศา
M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
v = np.array([1, 0])
print(M @ v)  # หมุนไปยังทิศทางใหม่`}</pre>
    <p className="mb-4">
      เมทริกซ์นี้ใช้หมุนเวกเตอร์ไปรอบจุดกำเนิดตามมุมที่กำหนด → เป็นหลักการเดียวกับการแปลงภาพ, ทิศทางการมอง, หรือมุมกล้องในโลกจริง
    </p>

    <h3 className="text-lg font-semibold mb-2">ตัวอย่างที่ 3: สะท้อนเวกเตอร์</h3>
    <pre className="bg-gray-800 text-white p-3 rounded-md text-sm overflow-x-auto mb-4">{`M = np.array([[1, 0], [0, -1]])
v = np.array([3, 2])
print(M @ v)  # Output: [3, -2]`}</pre>
    <p className="mb-4">
      สะท้อนเวกเตอร์ในแนวแกน Y → ถูกใช้ในระบบภาพหรือเสียงเพื่อสลับมุมมองซ้ายขวา เช่น การกลับเฟรมของเสียงหรือภาพกระจก
    </p>
  </div>

  <p className="mb-4">
    ความสำคัญของการเปลี่ยนเวกเตอร์ผ่านเมทริกซ์คือสามารถสร้าง feature ใหม่ ๆ ที่มีความหมายมากกว่าเดิม เช่น vector ที่ถูกหมุนอาจทำให้มองเห็นขอบในทิศทางใหม่ หรือการยืดเวกเตอร์ในแกนใดแกนหนึ่งช่วยเพิ่มน้ำหนักให้กับมิติสำคัญ
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    การเปลี่ยนแปลงเวกเตอร์ด้วยเมทริกซ์ไม่ใช่เพียงการคำนวณทางคณิตศาสตร์ แต่คือการจัดโครงสร้างของความเข้าใจใหม่ในพื้นที่ที่เหมาะสมยิ่งขึ้นกับโมเดล AI
  </div>
</section>



<section id="code-example" className="mb-16 scroll-mt-32">
  <h2 className="text-2xl font-semibold mb-3">ตัวอย่างโค้ด: NumPy & PyTorch</h2>

  <p className="mb-4">
    ในการสร้างระบบที่สามารถแปลงข้อมูล input ให้อยู่ในรูปแบบที่เข้าใจง่ายขึ้นสำหรับโมเดล จำเป็นต้องมีการใช้ <strong>เมทริกซ์</strong> เพื่อเปลี่ยนเวกเตอร์จากรูปแบบหนึ่งไปสู่อีกรูปแบบหนึ่ง กระบวนการนี้คือหัวใจของ Linear Transformation และสามารถแสดงออกมาได้ผ่านโค้ดในภาษา Python โดยใช้ไลบรารีอย่าง <strong>NumPy</strong> และ <strong>PyTorch</strong>
  </p>

  <h3 className="text-lg font-semibold mb-2">ตัวอย่างที่ 1: การคูณเมทริกซ์ใน NumPy</h3>
  <p className="mb-4">
    NumPy เป็นไลบรารียอดนิยมสำหรับงานคำนวณเชิงคณิตศาสตร์ สามารถใช้แสดงผลของ Linear Transformation ได้อย่างชัดเจน
  </p>
  <pre className="bg-gray-800 text-white p-4 text-sm rounded-md overflow-x-auto mb-6">
{`# NumPy
import numpy as np

# เมทริกซ์แปลงที่ยืดเวกเตอร์ในแนวแกน X
W = np.array([[2, 0], [0, 1]])

# เวกเตอร์ input
x = np.array([1, 3])

# คูณเมทริกซ์กับเวกเตอร์
result = W @ x
print(result)  # Output: [2 3]`}
  </pre>

  <p className="mb-4">
    จากตัวอย่างจะเห็นว่า vector [1, 3] ถูกแปลงให้แกน X เพิ่มเป็น 2 เท่า ส่วนแกน Y คงเดิม → สิ่งนี้สะท้อนถึงการ <strong>stretch</strong> ของเวกเตอร์ ซึ่งแสดงถึงการแปลงมุมมองของข้อมูลดิบ
  </p>

  <h3 className="text-lg font-semibold mb-2">ตัวอย่างที่ 2: การใช้ Linear Layer ใน PyTorch</h3>
  <p className="mb-4">
    PyTorch เป็นไลบรารี Deep Learning ที่สามารถสร้างเลเยอร์แบบ Linear ได้โดยตรงผ่าน <code>torch.nn.Linear</code> ซึ่งจะสุ่มค่า weight matrix และ bias โดยอัตโนมัติ เพื่อแสดงการคูณและแปลงเวกเตอร์
  </p>
  <pre className="bg-gray-800 text-white p-4 text-sm rounded-md overflow-x-auto mb-6">
{`# PyTorch
import torch

# สร้าง Linear Layer: input dim = 3, output dim = 2
layer = torch.nn.Linear(3, 2)

# สร้าง input vector ขนาด 1x3
x = torch.randn(1, 3)

# คูณเมทริกซ์ผ่านเลเยอร์
output = layer(x)
print(output)`}
  </pre>

  <p className="mb-4">
    การเรียกใช้ <code>layer(x)</code> จะเป็นการคูณเวกเตอร์กับเมทริกซ์ที่ซ่อนอยู่ภายในเลเยอร์ พร้อมทั้งบวก bias ซึ่งเป็นส่วนหนึ่งของกลไกใน Neural Network ทำให้เกิดการแปลงเวกเตอร์แบบ Linear
  </p>

  <h1 className="text-lg font-semibold mb-2 text-center">วิเคราะห์ผลลัพธ์และการใช้งานจริง</h1>
  <img
    src="/LinearTransformation11.png"
    alt="Linear Transformation"
    className="w-full max-w-md mx-auto rounded-xl shadow-lg border border-yellow-400 my-8"
  />
  <p className="mb-4">
    การเปลี่ยนเวกเตอร์จาก input ไปยัง feature vector ผ่านเมทริกซ์ เป็นเหมือนการแปลงมุมมองของข้อมูลให้เข้าสู่ "เลเยอร์ความเข้าใจ" ของโมเดล เช่น จาก pixel → edge หรือจากคำ → ความหมาย
  </p>
  <p className="mb-4">
    ทั้ง NumPy และ PyTorch มีความสามารถในการรองรับการคูณเมทริกซ์ในรูปแบบต่าง ๆ ตั้งแต่ระดับพื้นฐานจนถึงระดับ batch เช่น <code>(batch_size, input_dim) @ (input_dim, output_dim)</code> ซึ่งมีประโยชน์อย่างมากในการเรียนรู้แบบขนาน
  </p>

  <div className="bg-yellow-100 dark:bg-yellow-200 text-black p-4 rounded-xl text-sm border-l-4 border-yellow-500 shadow mt-6">
    <strong>Insight:</strong><br />
    โค้ดคือตัวกลางที่แปลงแนวคิดคณิตศาสตร์ให้กลายเป็นกลไกเรียนรู้ของโมเดล โดยเฉพาะ Linear Layer ที่เปรียบเหมือนประตูแรกที่ตัดสินว่าข้อมูลใดควรถูกเน้น ขยาย หรือกดทับ เพื่อให้สามารถเรียนรู้สิ่งที่สำคัญที่สุดได้
  </div>
</section>


<section id="visual-summary" className="mb-20 scroll-mt-32">
  <h2 className="text-3xl font-bold mb-4 font-orbitron gradient-text"> ภาพรวม: ก่อนและหลังการแปลงด้วยเมทริกซ์</h2>

  <p className="mb-4">
    ก่อนที่ข้อมูลจะถูกนำเข้าโมเดล AI ข้อมูลอยู่ในรูปแบบดิบที่ยังไม่มีความหมายเชิงโครงสร้าง เช่น พิกเซล เสียง waveform หรือข้อความที่ยังไม่เข้าใจความหมาย
  </p>

  <p className="mb-8">
    เมื่อนำข้อมูลเหล่านี้ผ่านกระบวนการคูณเมทริกซ์ (Linear Transformation) จะถูกแปลงเป็น “ฟีเจอร์” ที่เน้นลักษณะสำคัญ เช่น ขอบภาพ เสียงเด่น หรือความหมายในบริบท
  </p>

  {/* Comparison Grid */}
  <div className="grid sm:grid-cols-2 gap-8">
    {/* Raw Input */}
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 p-6 rounded-2xl shadow-md text-white">
      <h3 className="text-xl font-semibold mb-3"> ข้อมูลดิบ (Raw Input)</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ภาพ: พิกเซล RGB (0–255)</li>
        <li>เสียง: คลื่น waveform</li>
        <li>ข้อความ: One-hot vector / Token</li>
      </ul>
      <p className="mt-4 text-xs">
        ข้อมูลเหล่านี้ยังไม่เข้าใจง่าย — ไม่มีบริบท ไม่มีมิติของความหมาย
      </p>
    </div>

    {/* Transformed Feature */}
    <div className="bg-gradient-to-br from-yellow-500/10 to-yellow-800/20 border border-yellow-500 p-6 rounded-2xl shadow-md">
      <h3 className="text-xl font-semibold mb-3"> Feature ที่ถูกแปลง (Transformed)</h3>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>ภาพ: ขอบ, รูปร่าง (via CNN)</li>
        <li>เสียง: ความถี่, โทน (via Mel Filter)</li>
        <li>ข้อความ: บริบท, ความสัมพันธ์ (via Embedding)</li>
      </ul>
      <p className="mt-4 text-xs">
        ข้อมูลผ่านเมทริกซ์แล้วพร้อมต่อการเข้าใจ วิเคราะห์ และทำนาย
      </p>
    </div>
  </div>

  <p className="mt-10 mb-4">
    เมทริกซ์ทำหน้าที่เป็นเหมือน “เลนส์” ที่เปลี่ยนมุมมองของข้อมูล ให้เห็นสิ่งที่สำคัญซ่อนอยู่ ซึ่งเดิมไม่สามารถเข้าใจได้โดยตรง
  </p>

  <p className="mb-6">
    ในโมเดลลึก (เช่น CNN / Transformer) การแปลงข้อมูลผ่านเมทริกซ์หลายชั้นจะทำให้เห็น feature ที่ซับซ้อนมากขึ้น — จากขอบภาพ → วัตถุ หรือจากคำ → ความหมาย/เจตนา
  </p>

  {/* Insight Box */}
  <div className="bg-yellow-100 dark:bg-yellow-300/20 text-black dark:text-yellow-100 p-5 rounded-xl border-l-4 border-yellow-500 shadow-md">
    <strong className="block mb-2"> Insight:</strong>
    การเปรียบเทียบก่อนและหลังผ่านเมทริกซ์ คือหัวใจของ “Feature Extraction” — เมทริกซ์คือประตูที่เปลี่ยนข้อมูลดิบให้กลายเป็นสิ่งที่ AI เข้าใจได้
  </div>
</section>



<section id="insight" className="mb-20 scroll-mt-32">
  <div className="relative bg-gradient-to-br from-yellow-50 to-yellow-200 dark:from-yellow-900 dark:to-yellow-800 text-black dark:text-yellow-100 p-6 rounded-2xl border-l-8 border-yellow-500 shadow-xl">
    <div className="absolute top-4 right-4 opacity-10 text-7xl font-bold pointer-events-none select-none">
      🧠
    </div>
    <h2 className="text-2xl font-bold mb-4">
      Insight: การคูณเมทริกซ์ = ปรับความเข้าใจของโมเดล
    </h2>
    <div className="space-y-4 text-sm leading-relaxed">
      <p>
        การคูณเมทริกซ์ในระบบ AI เปรียบเสมือนการเปลี่ยนเลนส์กล้อง เพื่อให้เห็นข้อมูลในมุมที่คมชัดและมีความหมายต่อการเรียนรู้ ไม่ใช่เพียงการคำนวณ แต่คือการเลือก “มิติที่ควรมองเห็น”
      </p>
      <p>
        ใน Neural Network ทุกเลเยอร์มีเมทริกซ์ของตนเอง → คัดกรอง ปรับทิศทาง และเน้นข้อมูลบางส่วน เช่น เงาในภาพ หรือคำสำคัญในประโยค
      </p>
      <p>
        ใน Layer แรก เมทริกซ์อาจตรวจจับ edge หรือ pattern เบื้องต้น ส่วนเลเยอร์ถัดไปจะสกัดความหมายเชิงลึก เช่น “สิ่งของ”, “อารมณ์”, หรือ “สถานการณ์”
      </p>
      <p>
        เมทริกซ์ embedding ใน NLP แปลงคำให้กลายเป็นเวกเตอร์ที่สะท้อนบริบท เช่น คำว่า "light" อาจหมายถึงแสง หรือความเบา ขึ้นกับคำรอบข้าง
      </p>
      <p>
        ในระบบเสียง เมทริกซ์จะช่วยดึงโทนเสียงออกจาก noise และแปลงเป็นความเข้าใจที่สามารถระบุอารมณ์หรือบุคคล
      </p>
      <p>
        ในระบบแนะนำ (Recommendation) เมทริกซ์จับความสัมพันธ์ระหว่างผู้ใช้และสินค้า → ให้คำแนะนำที่แม่นยำขึ้น
      </p>
      <p>
        ท้ายที่สุด การคูณเมทริกซ์ = การเลือกมุมที่ทำให้ AI “เข้าใจ” โลกใบนี้ในแบบของมัน
      </p>
    </div>
    <p className="italic text-right text-gray-600 dark:text-gray-400 mt-6 text-xs">
      "Matrix คือเลนส์ของระบบอัจฉริยะ ทุกการเปลี่ยนผ่านของข้อมูล ล้วนเกิดขึ้นภายใต้การมองผ่านเลนส์นี้เสมอ"
    </p>
  </div>
</section>





        <section id="quiz" className="mb-16 scroll-mt-32">
          <MiniQuiz_Day5 theme={theme} />
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
        <ScrollSpy_Ai_Day5 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day5_LinearTransform;
