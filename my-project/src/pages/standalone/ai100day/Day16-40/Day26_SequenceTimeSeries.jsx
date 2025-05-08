import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day26 from "../scrollspy/scrollspyDay16-40/ScrollSpy_Ai_Day26";
import MiniQuiz_Day26 from "../miniquiz/miniquizDay16-40/MiniQuiz_Day26";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day26_SequenceTimeSeries = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  const img1 = cld.image("Day26_1").format("auto").quality("auto").resize(scale().width(649));
  const img2 = cld.image("Day26_2").format("auto").quality("auto").resize(scale().width(590));
  const img3 = cld.image("Day26_3").format("auto").quality("auto").resize(scale().width(590));
  const img4 = cld.image("Day26_4").format("auto").quality("auto").resize(scale().width(590));
  const img5 = cld.image("Day26_5").format("auto").quality("auto").resize(scale().width(590));
  const img6 = cld.image("Day26_6").format("auto").quality("auto").resize(scale().width(590));
  const img7 = cld.image("Day26_7").format("auto").quality("auto").resize(scale().width(590));
  const img8 = cld.image("Day26_8").format("auto").quality("auto").resize(scale().width(590));
  const img9 = cld.image("Day26_9").format("auto").quality("auto").resize(scale().width(590));
  const img10 = cld.image("Day26_10").format("auto").quality("auto").resize(scale().width(590));
  const img11 = cld.image("Day26_11").format("auto").quality("auto").resize(scale().width(590));
  const img12 = cld.image("Day26_12").format("auto").quality("auto").resize(scale().width(590));
  const img13 = cld.image("Day26_13").format("auto").quality("auto").resize(scale().width(590));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      <main className="max-w-3xl mx-auto p-6 pt-20"></main>
      <div className="">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Day 26: Sequence Modeling & Time Series</h1>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>

          <div className="w-full flex justify-center my-12">
            <div className="h-1 w-32 bg-gradient-to-r from-yellow-400 via-white to-yellow-400 dark:from-yellow-500 dark:via-gray-900 dark:to-yellow-500 rounded-full shadow-md"></div>
          </div>

          <section id="introduction" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">1. Introduction: ทำไม Sequence Modeling สำคัญมาก?</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img2} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Sequence Modeling คือหัวใจสำคัญของงานวิจัยและการประยุกต์ใช้ AI ที่เกี่ยวข้องกับข้อมูลที่มีลำดับ เช่น เวลา, ภาษา, และคลื่นสัญญาณ เนื่องจากข้อมูลเหล่านี้มีลักษณะพิเศษที่ขึ้นอยู่กับบริบทก่อนหน้า ทำให้การวิเคราะห์ต้องคำนึงถึงลำดับของข้อมูลอย่างหลีกเลี่ยงไม่ได้
    </p>

    <p>
      งานของ Stanford และ MIT แสดงให้เห็นว่า sequence modeling ไม่เพียงแต่ใช้ในภาษาธรรมชาติ แต่ยังมีบทบาทสำคัญในด้านอื่น เช่น การพยากรณ์ราคาหุ้น, การวิเคราะห์ ECG ในการแพทย์, และการจำแนกพฤติกรรมของผู้ใช้งานในแอปพลิเคชันต่าง ๆ
    </p>

    <h3 className="text-xl font-semibold">ตัวอย่างของข้อมูลลำดับ (Sequential Data)</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ข้อมูลราคาหุ้นรายวัน</li>
      <li>ข้อมูลเซนเซอร์จากอุปกรณ์ IoT</li>
      <li>ลำดับของคำในประโยค (Natural Language)</li>
      <li>สัญญาณเสียงหรือ ECG waveform</li>
      <li>การเคลื่อนไหวของร่างกายในคลิปวิดีโอ</li>
    </ul>

    <h3 className="text-xl font-semibold">เหตุผลที่ Sequence Modeling สำคัญ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>ข้อมูลมี temporal dependency: ค่าปัจจุบันขึ้นกับค่าก่อนหน้า</li>
      <li>ลำดับมีบริบท: ความหมายในภาษา หรือพฤติกรรมของข้อมูลเปลี่ยนไปตามลำดับ</li>
      <li>สามารถจับ pattern ที่เกิดซ้ำหรือแนวโน้มได้ (trend & seasonality)</li>
      <li>เหมาะกับการพยากรณ์ในระยะสั้นและระยะยาว</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบข้อมูลแบบ Sequence กับ Non-sequence</h3>
    <div className="overflow-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-100 dark:bg-gray-800">
            <th className="border px-4 py-2">ลักษณะ</th>
            <th className="border px-4 py-2">Sequence Data</th>
            <th className="border px-4 py-2">Non-sequence Data</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">ความสำคัญของลำดับ</td>
            <td className="border px-4 py-2">สูง</td>
            <td className="border px-4 py-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ลักษณะของข้อมูล</td>
            <td className="border px-4 py-2">เปลี่ยนแปลงต่อเนื่อง</td>
            <td className="border px-4 py-2">ค่าคงที่หรืออิสระจากกัน</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">ตัวอย่าง</td>
            <td className="border px-4 py-2">เสียง, ภาษา, Time Series</td>
            <td className="border px-4 py-2">ภาพนิ่ง, ตารางข้อมูล</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">กรณีศึกษาในโลกจริง</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>การพยากรณ์อัตราการเติบโตทางเศรษฐกิจ</li>
      <li>การวิเคราะห์ log การใช้งานเพื่อแนะนำเนื้อหา</li>
      <li>การพยากรณ์ภาวะหัวใจเต้นผิดจังหวะจากสัญญาณ ECG</li>
      <li>การพยากรณ์อุปสงค์สินค้าในอีคอมเมิร์ซ</li>
    </ul>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <h4 className="font-semibold mb-2">Insight Box</h4>
      <ul className="list-disc list-inside space-y-1">
        <li>Sequence Modeling เป็นรากฐานของระบบพยากรณ์สมัยใหม่</li>
        <li>ลำดับของข้อมูลบ่งชี้บริบทเชิงเวลาและโครงสร้างที่ซ่อนอยู่</li>
        <li>เป็นพื้นฐานสำคัญของโมเดลอย่าง RNN, LSTM, GRU และ Transformer</li>
      </ul>
    </div>

    <p>
      งานวิจัยจากมหาวิทยาลัยชั้นนำ เช่น Stanford, MIT และ University of Toronto ต่างเน้นย้ำว่า sequence modeling ไม่เพียงแต่เป็นปัญหาทางเทคนิคที่ท้าทาย แต่ยังเป็นหัวใจของการสร้าง AI ที่เข้าใจข้อมูลโลกจริงได้ลึกซึ้งขึ้น
    </p>
  </div>
</section>


<section id="anatomy" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">2. Anatomy of Time Series Data</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img3} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Time Series คือข้อมูลที่จัดเก็บตามลำดับเวลา โดยค่าของข้อมูลในแต่ละช่วงเวลามีความสัมพันธ์กันตามลำดับ ซึ่งแตกต่างจากข้อมูลทั่วไปที่แต่ละ record มักเป็นอิสระต่อกัน ความเข้าใจโครงสร้างของ Time Series จึงเป็นพื้นฐานสำคัญในการพัฒนาโมเดลที่สามารถจัดการกับข้อมูลประเภทนี้ได้อย่างมีประสิทธิภาพ
    </p>

    <h3 className="text-xl font-semibold">Univariate vs Multivariate Time Series</h3>
    <p>
      Time Series สามารถแบ่งตามจำนวนตัวแปรที่ใช้ในการวิเคราะห์ได้ 2 ประเภทหลัก ๆ:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Univariate Time Series:</strong> เป็น Time Series ที่มีเพียงหนึ่งตัวแปร เช่น ราคาปิดของหุ้นรายวัน โดยขึ้นอยู่กับค่าในอดีตของตัวแปรเดียวเท่านั้น
      </li>
      <li>
        <strong>Multivariate Time Series:</strong> มีหลายตัวแปรประกอบกัน เช่น ราคาหุ้น, ปริมาณการซื้อขาย และดัชนีความผันผวน ซึ่งสัมพันธ์กันตามเวลา
      </li>
    </ul>

    <h3 className="text-xl font-semibold">Stationary vs Non-Stationary</h3>
    <p>
      ข้อมูล Time Series แบ่งตามลักษณะเชิงสถิติได้เป็นสองประเภทคือ:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Stationary Series:</strong> มีค่าเฉลี่ย (mean), ความแปรปรวน (variance), และโครงสร้างการกระจายที่คงที่เมื่อเวลาผ่านไป เช่น สัญญาณเซนเซอร์ที่มีความสม่ำเสมอ
      </li>
      <li>
        <strong>Non-Stationary Series:</strong> มี mean หรือ variance ที่เปลี่ยนแปลงตามเวลา เช่น ราคาอสังหาริมทรัพย์ที่เพิ่มขึ้นต่อเนื่อง
      </li>
    </ul>
    <p>
      โมเดลทางสถิติส่วนใหญ่ เช่น ARIMA จำเป็นต้องใช้ข้อมูลแบบ stationary ซึ่งอาจต้องใช้เทคนิคการเปลี่ยนแปลงข้อมูล เช่น differencing หรือ normalization ก่อนนำไปวิเคราะห์
    </p>

    <h3 className="text-xl font-semibold">การตรวจสอบ Stationarity</h3>
    <p>
      เพื่อประเมินว่าข้อมูลเป็น Stationary หรือไม่ สามารถใช้วิธีต่าง ๆ ได้ เช่น:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        การวาดกราฟ Time Series และตรวจสอบแนวโน้ม (Trend)
      </li>
      <li>
        Augmented Dickey-Fuller Test (ADF Test)
      </li>
      <li>
        การดู Autocorrelation Function (ACF)
      </li>
    </ul>

    <h3 className="text-xl font-semibold">Seasonality, Trend และ Noise</h3>
    <p>
      โครงสร้างพื้นฐานของ Time Series โดยทั่วไปประกอบด้วย 3 องค์ประกอบหลัก:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Trend:</strong> ทิศทางของข้อมูลในระยะยาว เช่น การเพิ่มขึ้นของรายได้ปีต่อปี
      </li>
      <li>
        <strong>Seasonality:</strong> รูปแบบที่เกิดซ้ำในช่วงเวลาหนึ่ง เช่น การใช้ไฟฟ้าเพิ่มขึ้นช่วงหน้าร้อน
      </li>
      <li>
        <strong>Noise:</strong> ความแปรปรวนแบบสุ่มที่ไม่สามารถอธิบายด้วยโมเดลได้
      </li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>ข้อมูล Time Series มีโครงสร้างที่เฉพาะเจาะจง และต้องการการเตรียมข้อมูลที่แตกต่างจากข้อมูลตารางทั่วไป</li>
        <li>การรู้จักแยกแยะข้อมูลระหว่าง Stationary กับ Non-Stationary ช่วยให้เลือกโมเดลได้เหมาะสมยิ่งขึ้น</li>
        <li>ในงานจริง ข้อมูลมักไม่เป็น Stationary ต้องใช้ preprocessing เสมอ</li>
        <li>Multivariate Time Series มีศักยภาพสูง แต่ต้องการการวิเคราะห์ความสัมพันธ์เชิงเวลาและระหว่างตัวแปร</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่าง Visualization</h3>
    <p>
      ภาพด้านล่างแสดงให้เห็นถึงตัวอย่างของข้อมูลที่มี trend, seasonality และ noise ซึ่งต้องแยกองค์ประกอบเหล่านี้ก่อนสร้างโมเดลพยากรณ์:
    </p>
    <div className="flex justify-center my-6">
      <AdvancedImage cldImg={img4} />
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      การเข้าใจ anatomy ของ Time Series เป็นพื้นฐานสำคัญในการประมวลผลลำดับข้อมูลอย่างมีประสิทธิภาพ โดยต้องแยกความแตกต่างระหว่างรูปแบบของข้อมูล (Univariate/Multivariate), ลักษณะเชิงสถิติ (Stationary/Non-Stationary) และโครงสร้างเชิงเวลา (Trend/Seasonality/Noise) เพื่อออกแบบ preprocessing pipeline และเลือกโมเดลที่เหมาะสมกับลักษณะของปัญหา
    </p>
  </div>
</section>


<section id="task-types" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">3. Sequence Modeling Task Types</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img4} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การทำความเข้าใจประเภทของงาน (task types) ที่เกี่ยวข้องกับ Sequence Modeling เป็นพื้นฐานสำคัญในการออกแบบโมเดลและเลือกกลยุทธ์การพัฒนาให้เหมาะสมกับบริบทของข้อมูล
      โดยในงานวิจัยจาก Stanford และ MIT ได้จัดประเภทของ sequence tasks ออกเป็น 4 ประเภทหลัก ซึ่งแต่ละประเภทมีลักษณะการป้อนข้อมูล (input) และผลลัพธ์ที่แตกต่างกัน
    </p>

    <h3 className="text-xl font-semibold">ประเภทของงานตามโครงสร้าง Input → Output</h3>
    <div className="overflow-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Task Type</th>
            <th className="border px-4 py-2">Input</th>
            <th className="border px-4 py-2">Output</th>
            <th className="border px-4 py-2">ตัวอย่างการใช้งาน</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Sequence → Value</td>
            <td className="border px-4 py-2">ชุดข้อมูลต่อเนื่อง (30 วัน)</td>
            <td className="border px-4 py-2">ค่าหนึ่งค่าในอนาคต (วันที่ 31)</td>
            <td className="border px-4 py-2">พยากรณ์ราคาหุ้น, คาดการณ์อุณหภูมิ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Sequence → Sequence</td>
            <td className="border px-4 py-2">ลำดับเฟรมของวิดีโอ</td>
            <td className="border px-4 py-2">ลำดับเฟรมในอนาคต</td>
            <td className="border px-4 py-2">Video Prediction, Machine Translation</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Sequence → Label</td>
            <td className="border px-4 py-2">สัญญาณ ECG แบบต่อเนื่อง</td>
            <td className="border px-4 py-2">Label 0/1 ผิดปกติหรือไม่</td>
            <td className="border px-4 py-2">Anomaly Detection, Medical Signal Classification</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Sequence → Category</td>
            <td className="border px-4 py-2">ประโยค (sequence ของคำ)</td>
            <td className="border px-4 py-2">คลาสของอารมณ์</td>
            <td className="border px-4 py-2">Sentiment Analysis, Intent Detection</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">การวิเคราะห์การเลือก Task ให้เหมาะสม</h3>
    <p>
      การเลือกประเภทของ task มีผลอย่างมากต่อการออกแบบสถาปัตยกรรมของโมเดล เช่น หากต้องการคาดการณ์ค่าเดียวล่วงหน้า อาจเลือกโมเดล regression เช่น LSTM; แต่หากต้องการ output เป็นลำดับต่อเนื่อง จะต้องใช้โมเดลแบบ encoder-decoder เช่น Seq2Seq หรือ Transformer
    </p>

    <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg">
      <p className="font-medium">Insight Box:</p>
      <ul className="list-disc list-inside space-y-1">
        <li>Sequence → Value เหมาะกับงานพยากรณ์ตัวเลข เช่น Demand, Stock</li>
        <li>Sequence → Sequence พบใน NLP, Translation และ Speech Generation</li>
        <li>Sequence → Label เหมาะกับงานตรวจจับพฤติกรรมหรือสัญญาณผิดปกติ</li>
        <li>Sequence → Category นิยมใช้ใน Text Classification, Chatbot, Emotion</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการนำไปใช้จริงจาก Google และ Meta</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>Google ใช้ Sequence → Sequence ใน Google Translate ด้วย Transformer</li>
      <li>Meta ใช้ Sequence → Category เพื่อจำแนกกลุ่มเป้าหมายจาก user behavior</li>
      <li>Apple ใช้ Sequence → Label ในการประเมินความเสี่ยงของอัตราการเต้นหัวใจผ่าน Apple Watch</li>
    </ul>

    <h3 className="text-xl font-semibold">โค้ดตัวอย่าง: Sequence → Value ด้วย PyTorch</h3>
    <pre><code className="language-python">
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # ใช้ output สุดท้ายใน sequence
    </code></pre>

    <p>
      ในตัวอย่างนี้ โมเดลรับ input เป็นลำดับ (เช่น ราคาหุ้น 30 วัน) และพยากรณ์ค่าของวันถัดไป ซึ่งเหมาะกับ task แบบ Sequence → Value
    </p>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      Sequence Modeling สามารถใช้ได้หลากหลายประเภทขึ้นอยู่กับ nature ของข้อมูลและเป้าหมายของโมเดล การเข้าใจ task type อย่างลึกซึ้งเป็นขั้นตอนแรกที่นำไปสู่การออกแบบระบบ Machine Learning ที่มีประสิทธิภาพในงานด้านลำดับเวลาและภาษา
    </p>
  </div>
</section>


<section id="models" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">4. โมเดลสำหรับ Time Series</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img5} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      การสร้างแบบจำลองสำหรับข้อมูล Time Series จำเป็นต้องใช้โมเดลที่สามารถเรียนรู้ลำดับและความสัมพันธ์เชิงเวลาได้อย่างมีประสิทธิภาพ โดยเฉพาะในกรณีที่ข้อมูลมีการเปลี่ยนแปลงต่อเนื่องและมีแนวโน้มที่ไม่คงที่ งานวิจัยจาก Stanford, MIT และ Oxford ชี้ให้เห็นว่าโมเดลที่ออกแบบมาเฉพาะสำหรับ Time Series มีศักยภาพในการคาดการณ์สูงกว่าโมเดลทั่วไป
    </p>

    <h3 className="text-xl font-semibold">Recurrent Neural Network (RNN)</h3>
    <p>
      RNN เป็นโมเดลพื้นฐานที่ถูกออกแบบมาเพื่อจัดการกับข้อมูลลำดับ โดยการส่งสถานะภายใน (hidden state) จาก timestep ก่อนหน้าไปยัง timestep ถัดไป ทำให้สามารถจดจำลำดับข้อมูลได้ระดับหนึ่ง
    </p>
    <pre><code className="language-python">
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
    </code></pre>

    <h3 className="text-xl font-semibold">Long Short-Term Memory (LSTM)</h3>
    <p>
      LSTM ถูกพัฒนาขึ้นเพื่อแก้ปัญหา vanishing gradient ใน RNN โดยมีโครงสร้างหน่วยความจำภายใน (memory cell) และ gate ที่ควบคุมการเก็บ ลืม และดึงข้อมูลที่สำคัญจากลำดับก่อนหน้า
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เหมาะสำหรับข้อมูลที่มี long-term dependencies</li>
      <li>นิยมใช้ในงานด้าน NLP, การพยากรณ์อากาศ, การวิเคราะห์หุ้น</li>
    </ul>

    <h3 className="text-xl font-semibold">Gated Recurrent Unit (GRU)</h3>
    <p>
      GRU เป็นอีกทางเลือกหนึ่งที่มีโครงสร้างง่ายกว่า LSTM แต่ยังสามารถจับข้อมูลระยะยาวได้ดี มีเพียง 2 gates (update และ reset) ทำให้ฝึกได้เร็วขึ้นและมีพารามิเตอร์น้อยลง
    </p>

    <h3 className="text-xl font-semibold">1D Convolutional Neural Network (1D-CNN)</h3>
    <p>
      1D-CNN เหมาะสำหรับการดึง pattern ระยะสั้นจากข้อมูล Time Series โดยใช้ kernel เลื่อนผ่านข้อมูลในแกนเวลา ช่วยลด noise และจับสัญญาณเฉพาะช่วงได้ดี
    </p>
    <pre><code className="language-python">
import torch.nn as nn

class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.mean(dim=2)
        return self.fc(x)
    </code></pre>

    <h3 className="text-xl font-semibold">Transformer for Time Series</h3>
    <p>
      Transformer ซึ่งประสบความสำเร็จใน NLP ได้ถูกนำมาใช้ใน Time Series โดยใช้ Attention Mechanism เพื่อเรียนรู้ความสัมพันธ์เชิงเวลาระหว่าง timestep ได้โดยตรง โดยไม่ต้องพึ่ง sequential computation
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>เหมาะกับ sequence ยาว ๆ และข้อมูล multivariate</li>
      <li>มี scalability สูง และฝึกได้แบบ parallel</li>
    </ul>

    <h3 className="text-xl font-semibold">เปรียบเทียบโมเดล Time Series</h3>
    <table className="w-full text-sm border border-gray-300 dark:border-gray-700 text-left">
      <thead className="bg-gray-100 dark:bg-gray-800">
        <tr>
          <th className="py-2 px-4 border-r">โมเดล</th>
          <th className="py-2 px-4 border-r">จุดเด่น</th>
          <th className="py-2 px-4">ข้อควรระวัง</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">RNN</td>
          <td className="py-2 px-4 border-r">เข้าใจลำดับได้ดี</td>
          <td className="py-2 px-4">Vanishing Gradient</td>
        </tr>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">LSTM</td>
          <td className="py-2 px-4 border-r">ระยะยาวดี, ใช้งานกว้าง</td>
          <td className="py-2 px-4">ฝึกช้า, ใช้พลังงานสูง</td>
        </tr>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">GRU</td>
          <td className="py-2 px-4 border-r">เร็วกว่า LSTM</td>
          <td className="py-2 px-4">อาจแม่นยำน้อยกว่าเล็กน้อย</td>
        </tr>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">1D-CNN</td>
          <td className="py-2 px-4 border-r">ดีสำหรับ pattern ระยะสั้น</td>
          <td className="py-2 px-4">ไม่เก่งกับ dependency ระยะยาว</td>
        </tr>
        <tr className="border-t">
          <td className="py-2 px-4 border-r">Transformer</td>
          <td className="py-2 px-4 border-r">Scalable, multi-step</td>
          <td className="py-2 px-4">ต้องการ data & compute มาก</td>
        </tr>
      </tbody>
    </table>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg mt-6">
      <h4 className="text-lg font-semibold mb-2">Insight Box</h4>
      <ul className="list-disc list-inside space-y-1">
        <li>ไม่มีโมเดลใดที่ดีที่สุดในทุกสถานการณ์</li>
        <li>การเลือกโมเดลขึ้นกับความยาวของลำดับ, จำนวน feature, และทรัพยากรในการฝึก</li>
        <li>ควรเริ่มจาก LSTM หรือ GRU ก่อนพิจารณา Transformer</li>
      </ul>
    </div>

  </div>
</section>


<section id="sliding-vs-seq2seq" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">5. Sliding Window vs Sequence-to-Sequence</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img6} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Sliding Window และ Sequence-to-Sequence (Seq2Seq) เป็นสองกลยุทธ์หลักในการประมวลผลลำดับข้อมูล (sequential data) โดยเฉพาะในบริบทของ Time Series และ Natural Language Processing (NLP) ทั้งสองแนวทางนี้มีจุดเด่นและข้อจำกัดต่างกันขึ้นอยู่กับวัตถุประสงค์ของการพยากรณ์ลำดับ
    </p>

    <h3 className="text-xl font-semibold">Sliding Window Approach</h3>
    <p>
      เทคนิค Sliding Window ใช้ข้อมูลในอดีตภายในหน้าต่างขนาดคงที่ (window size) เพื่อพยากรณ์ค่าถัดไปในลำดับ เทคนิคนี้เป็นแนวทางพื้นฐานแต่ทรงพลัง โดยใช้กันอย่างแพร่หลายในงาน Time Series Forecasting เช่นราคาหุ้น การใช้พลังงาน หรือข้อมูลเซ็นเซอร์
    </p>

    <pre><code className="language-python">
# ตัวอย่างการสร้าง window
import numpy as np

def create_windowed_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)
    </code></pre>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>ข้อดี:</strong> เข้าใจง่าย ใช้งานได้เร็ว เหมาะกับโมเดลเชิงเส้นหรือ shallow neural networks</li>
      <li><strong>ข้อจำกัด:</strong> จับ context ระยะยาวไม่ได้ดีนัก หาก window size เล็ก</li>
      <li><strong>กรณีใช้งาน:</strong> การพยากรณ์จุดถัดไปจากข้อมูลก่อนหน้า เช่น การคาดการณ์ราคาวันพรุ่งนี้จากข้อมูล 7 วันล่าสุด</li>
    </ul>

    <h3 className="text-xl font-semibold">Sequence-to-Sequence (Seq2Seq)</h3>
    <p>
      โมเดล Seq2Seq ได้รับการพัฒนาโดยทีมวิจัยจาก Google สำหรับงานแปลภาษา (Machine Translation) โดยใช้ Encoder-Decoder Architecture เพื่อ map input sequence หนึ่งไปยัง output sequence ที่มีความยาวต่างกันได้ มีการนำมาใช้ใน Time Series, Speech Recognition, Text Generation และ Conversational AI
    </p>

    <pre><code className="language-python">
# โครงสร้างพื้นฐานของ Seq2Seq
encoder_input = Input(shape=(timesteps, features))
encoder = LSTM(128, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_input)

decoder_input = Input(shape=(None, features))
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=[state_h, state_c])
    </code></pre>

    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>ข้อดี:</strong> รองรับการพยากรณ์ลำดับหลายจุดพร้อมกัน (multi-step forecasting)</li>
      <li><strong>ความสามารถ:</strong> เรียนรู้ dependencies ที่ซับซ้อนและยาวนานภายใน sequence</li>
      <li><strong>กรณีใช้งาน:</strong> การแปลภาษา, วิดีโอ captioning, การพยากรณ์ชุดข้อมูลช่วงเวลาหลายวัน</li>
    </ul>

    <h3 className="text-xl font-semibold">การเปรียบเทียบเชิงโครงสร้าง</h3>
    <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
      <thead>
        <tr className="bg-gray-100 dark:bg-gray-800">
          <th className="px-4 py-2 border">ลักษณะ</th>
          <th className="px-4 py-2 border">Sliding Window</th>
          <th className="px-4 py-2 border">Seq2Seq</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border px-4 py-2">รูปแบบ Input</td>
          <td className="border px-4 py-2">Fixed Window</td>
          <td className="border px-4 py-2">Full Sequence</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">รูปแบบ Output</td>
          <td className="border px-4 py-2">1 จุดหรือไม่กี่จุด</td>
          <td className="border px-4 py-2">ทั้งลำดับ</td>
        </tr>
        <tr>
          <td className="border px-4 py-2">เหมาะกับข้อมูล</td>
          <td className="border px-4 py-2">มีโครงสร้างระยะสั้น</td>
          <td className="border px-4 py-2">ต้องเข้าใจบริบททั้งลำดับ</td>
        </tr>
      </tbody>
    </table>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-800 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Sliding Window ดีสำหรับ baseline และโมเดลที่ไม่ซับซ้อน</li>
        <li>Seq2Seq เปิดประตูสู่การพยากรณ์ที่ละเอียดและต่อเนื่อง</li>
        <li>การเลือกแนวทางขึ้นกับประเภทข้อมูลและวัตถุประสงค์ของ task</li>
        <li>สามารถผสมผสานทั้งสองเทคนิคในงาน real-world เพื่อเพิ่มประสิทธิภาพ</li>
      </ul>
    </div>

    <p>
      งานของ Cho et al. (2014) ที่แนะนำ GRU และ Sutskever et al. (2014) ที่ริเริ่ม Seq2Seq แสดงให้เห็นว่าการใช้ RNN-based Encoder-Decoder ช่วยให้การพยากรณ์ข้อมูลลำดับซับซ้อนได้อย่างมีประสิทธิภาพ ทั้งสองแนวทางนี้ยังคงเป็นรากฐานของการเรียนรู้ลำดับในยุคปัจจุบัน แม้จะมีการพัฒนา Transformer ที่เหนือชั้นกว่าในบางกรณี
    </p>
  </div>
</section>


<section id="metrics" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">6. Evaluation Metrics for Sequence Forecasting</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img7} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การประเมินความแม่นยำของโมเดลพยากรณ์ลำดับเวลา (Sequence Forecasting) มีความสำคัญอย่างยิ่งในการใช้งานจริง โดยเฉพาะเมื่อโมเดลถูกนำไปใช้ในระบบที่มีผลกระทบต่อการตัดสินใจ เช่น การเงิน พลังงาน และการวางแผนอุปสงค์ ค่าชี้วัดที่เลือกใช้ต้องสอดคล้องกับลักษณะของข้อมูลและเป้าหมายของงานพยากรณ์
    </p>

    <h3 className="text-xl font-semibold">1. Mean Absolute Error (MAE)</h3>
    <p>
      MAE เป็น metric ที่วัดค่าความคลาดเคลื่อนเฉลี่ยในเชิงสัมบูรณ์ระหว่างค่าพยากรณ์กับค่าจริง โดยไม่สนใจทิศทางของความคลาดเคลื่อน จึงเหมาะกับงานที่ไม่ต้องการ penalize ความผิดพลาดขนาดใหญ่เป็นพิเศษ
    </p>
    <pre><code className="language-python">
import numpy as np
mae = np.mean(np.abs(y_true - y_pred))
    </code></pre>

    <h3 className="text-xl font-semibold">2. Mean Squared Error (MSE) และ Root Mean Squared Error (RMSE)</h3>
    <p>
      MSE จะให้ค่าความผิดพลาดในรูปของกำลังสอง ทำให้ penalize ข้อผิดพลาดที่มีค่าสูงมากกว่า MAE ส่วน RMSE คือการถอดรากที่สองของ MSE ซึ่งทำให้มีหน่วยตรงกับค่าจริง
    </p>
    <pre><code className="language-python">
mse = np.mean((y_true - y_pred)**2)
rmse = np.sqrt(mse)
    </code></pre>
    <p>
      เหมาะกับงานที่ความคลาดเคลื่อนขนาดใหญ่มีผลกระทบสูง เช่น ระบบวางแผนทรัพยากรหรือการควบคุมอัตโนมัติ
    </p>

    <h3 className="text-xl font-semibold">3. Mean Absolute Percentage Error (MAPE)</h3>
    <p>
      วัดความคลาดเคลื่อนในรูปเปอร์เซ็นต์ โดยเทียบกับค่าจริง เหมาะกับการเปรียบเทียบโมเดลข้ามช่วงเวลาหรือ dataset ที่มีหน่วยต่างกัน
    </p>
    <pre><code className="language-python">
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    </code></pre>
    <p>
      ข้อควรระวัง: MAPE ใช้ไม่ได้กับค่าจริงที่เท่ากับศูนย์
    </p>

    <h3 className="text-xl font-semibold">4. Symmetric Mean Absolute Percentage Error (sMAPE)</h3>
    <p>
      เป็นเวอร์ชันของ MAPE ที่แก้ปัญหาค่าศูนย์ในตัวหาร โดยทำให้ numerator และ denominator เป็นค่ารวมแบบสมมาตร
    </p>
    <pre><code className="language-python">
smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    </code></pre>

    <h3 className="text-xl font-semibold">5. Dynamic Time Warping (DTW)</h3>
    <p>
      DTW วัดความแตกต่างระหว่างลำดับสองชุดที่อาจมีความยาวไม่เท่ากัน โดยพิจารณาถึงการจัดแนว (alignment) ระหว่างจุดในลำดับ เหมาะกับข้อมูลที่มีการ shift หรือ stretch ในเวลา เช่น สัญญาณ ECG หรือการเคลื่อนไหว
    </p>
    <pre><code className="language-python">
from dtaidistance import dtw
distance = dtw.distance(y_true, y_pred)
    </code></pre>

    <h3 className="text-xl font-semibold">ตารางเปรียบเทียบ Metrics</h3>
    <div className="overflow-auto">
      <table className="table-auto w-full border border-gray-300 dark:border-gray-600 text-sm">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            <th className="border px-4 py-2">Metric</th>
            <th className="border px-4 py-2">เหมาะกับ</th>
            <th className="border px-4 py-2">จุดเด่น</th>
            <th className="border px-4 py-2">ข้อควรระวัง</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">MAE</td>
            <td className="border px-4 py-2">ทั่วไป</td>
            <td className="border px-4 py-2">ง่ายต่อการตีความ</td>
            <td className="border px-4 py-2">ไม่ penalize ความผิดพลาดใหญ่</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">RMSE</td>
            <td className="border px-4 py-2">เมื่อ error ขนาดใหญ่สำคัญ</td>
            <td className="border px-4 py-2">เน้น penalty ความคลาดเคลื่อนสูง</td>
            <td className="border px-4 py-2">ไวต่อ outlier</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">MAPE</td>
            <td className="border px-4 py-2">การเปรียบเทียบเป็นเปอร์เซ็นต์</td>
            <td className="border px-4 py-2">สื่อสารง่ายในภาคธุรกิจ</td>
            <td className="border px-4 py-2">ไม่เหมาะกับค่าจริงใกล้ 0</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">sMAPE</td>
            <td className="border px-4 py-2">ข้อมูลมีค่า 0</td>
            <td className="border px-4 py-2">คำนวณได้แม้ค่าจริง = 0</td>
            <td className="border px-4 py-2">ตีความยากกว่าปกติ</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">DTW</td>
            <td className="border px-4 py-2">ลำดับไม่เท่ากัน / time shift</td>
            <td className="border px-4 py-2">จับ alignment ได้ดี</td>
            <td className="border px-4 py-2">คำนวณช้า</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>ไม่มี metric ใดเหมาะกับทุกงาน ควรเลือกตามลักษณะของ data</li>
        <li>ควรใช้หลาย metric ร่วมกันเพื่อประเมินภาพรวมของโมเดล</li>
        <li>อย่าลืมวิเคราะห์ error ด้วย visualization เช่น error plot หรือ residual histogram</li>
      </ul>
    </div>
  </div>
</section>


<section id="feature-engineering" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">7. Feature Engineering for Time Series</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img8} />
  </div>

  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">

    <p>
      Feature Engineering คือกระบวนการดัดแปลงหรือสร้างข้อมูลใหม่จากข้อมูลดิบเพื่อเพิ่มประสิทธิภาพการเรียนรู้ของโมเดล ในงาน Time Series การแยก feature ที่สะท้อน pattern ตามเวลาเป็นขั้นตอนที่สำคัญมาก โดยเฉพาะเมื่อใช้โมเดลแบบ Machine Learning ที่ไม่สามารถเรียนรู้ temporal dependency ได้โดยตรงเหมือน Deep Sequence Models
    </p>

    <h3 className="text-xl font-semibold">Lag Features</h3>
    <p>
      Lag Features คือการสร้าง feature ใหม่จากค่าก่อนหน้าของตัวแปร เช่น <code>x(t-1)</code>, <code>x(t-2)</code> เป็นต้น แนวทางนี้ช่วยให้โมเดลเข้าใจว่า "อดีต" มีผลต่อ "ปัจจุบัน" อย่างไร โดยไม่ต้องใช้ลูปหรือ sequential layer
    </p>

    <pre><code className="language-python">
import pandas as pd

df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["lag_3"] = df["value"].shift(3)
    </code></pre>

    <h3 className="text-xl font-semibold">Rolling Statistics</h3>
    <p>
      การใช้ค่าเฉลี่ย (mean), ส่วนเบี่ยงเบนมาตรฐาน (std), หรือ quantile ภายในช่วงเวลาหนึ่ง (rolling window) เป็นอีกวิธีหนึ่งที่ช่วยให้โมเดลเข้าใจแนวโน้มระยะสั้น
    </p>

    <pre><code className="language-python">
df["rolling_mean_7"] = df["value"].rolling(window=7).mean()
df["rolling_std_7"] = df["value"].rolling(window=7).std()
    </code></pre>

    <h3 className="text-xl font-semibold">Time-Based Features</h3>
    <p>
      เวลามีบทบาทสำคัญใน Time Series เช่น วันในสัปดาห์, เดือน, หรือชั่วโมง ซึ่งสามารถ encode เป็น feature ได้ เช่น:
    </p>

    <pre><code className="language-python">
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["hour"] = df.index.hour
    </code></pre>

    <p>
      เพื่อเพิ่ม non-linearity อาจใช้ <strong>sin/cos transformation</strong> เพื่อรักษาลักษณะของความต่อเนื่องเชิงมุม:
    </p>

    <pre><code className="language-python">
import numpy as np

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    </code></pre>

    <h3 className="text-xl font-semibold">Fourier Transform</h3>
    <p>
      เทคนิคนี้มาจากสาขา signal processing ใช้เพื่อแยกองค์ประกอบความถี่ของ Time Series เช่น ฤดูกาลหรือคลื่นซ้ำ ๆ การใช้ FFT สามารถสร้าง feature frequency ได้
    </p>

    <pre><code className="language-python">
from numpy.fft import fft

fft_result = fft(df["value"].values)
df["fft_real"] = fft_result.real
df["fft_imag"] = fft_result.imag
    </code></pre>

    <h3 className="text-xl font-semibold">Seasonal Decomposition</h3>
    <p>
      เทคนิคจาก StatsModels ที่แยก Time Series ออกเป็น 3 ส่วนคือ trend, seasonal และ residual ซึ่งสามารถใช้เป็น feature ได้โดยตรง
    </p>

    <pre><code className="language-python">
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df["value"], model="additive", period=12)
df["trend"] = result.trend
df["seasonal"] = result.seasonal
df["resid"] = result.resid
    </code></pre>

    <h3 className="text-xl font-semibold">Interaction Features</h3>
    <p>
      การสร้าง interaction ระหว่าง feature ต่าง ๆ เช่น lag กับ rolling หรือ seasonal กับ time-based features อาจช่วยให้โมเดลเห็น pattern ที่ซับซ้อนขึ้น
    </p>

    <pre><code className="language-python">
df["lag_1_x_trend"] = df["lag_1"] * df["trend"]
df["weekday_x_seasonal"] = df["day_of_week"] * df["seasonal"]
    </code></pre>

    <h3 className="text-xl font-semibold">Autocorrelation Features</h3>
    <p>
      การวิเคราะห์ว่าค่าที่ผ่านมามีผลกับค่าปัจจุบันแค่ไหนในรูปแบบของ ACF/PACF ก็สามารถนำมาใช้สร้าง feature ได้เช่นกัน
    </p>

    <pre><code className="language-python">
from statsmodels.tsa.stattools import acf

acf_vals = acf(df["value"].dropna(), nlags=10)
for i, val in enumerate(acf_vals[1:], 1):
    df[f"acf_lag_i"] = val
    </code></pre>

    <h3 className="text-xl font-semibold">Scaling & Normalization</h3>
    <p>
      เนื่องจาก Time Series มักมี magnitude แตกต่างกันมาก การ scale ค่าด้วย MinMax, Z-score หรือ RobustScaler เป็นสิ่งจำเป็นก่อนนำไปเรียนรู้
    </p>

    <pre><code className="language-python">
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["value_scaled"]] = scaler.fit_transform(df[["value"]])
    </code></pre>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Feature Engineering คือจุดแตกต่างระหว่าง baseline model กับ state-of-the-art ในงาน Time Series</li>
        <li>การรวม lag + rolling + seasonal + interaction ทำให้โมเดลมีความสามารถสูงขึ้นอย่างชัดเจน</li>
        <li>เทคนิคพื้นฐานสามารถ outperform deep learning ได้ในบางกรณีถ้าทำ feature ได้ดี</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      การเตรียม feature สำหรับ Time Series ต้องใช้ทั้งความเข้าใจเชิงสถิติและความเข้าใจในโดเมนของข้อมูลจริง การผสมผสานระหว่าง time-based, lag-based, seasonality และ frequency domain feature ช่วยยกระดับความแม่นยำของโมเดลได้อย่างชัดเจน โดยเฉพาะเมื่อใช้กับโมเดล linear หรือ tree-based ที่ไม่สามารถจับ temporal dependency ได้โดยตรง
    </p>
  </div>
</section>

<section id="challenges" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">8. Handling Time Series Challenges</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img9} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Time series data often present unique challenges that differ significantly from traditional tabular data. These challenges arise due to the inherent temporal structure, potential irregularities in data collection, and the presence of underlying trends or seasonality. Researchers from institutions like Stanford, MIT, and Carnegie Mellon have extensively studied techniques to manage these issues effectively.
    </p>

    <h3 className="text-xl font-semibold">1. Handling Missing Values</h3>
    <p>
      Missing values in time series can distort analysis and reduce model accuracy. It's crucial to handle these gaps appropriately depending on the nature of the data and the intended application.
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Forward Fill (Last Observation Carried Forward):</strong> Assumes the last value persists until updated.</li>
      <li><strong>Interpolation:</strong> Estimates missing values using linear or spline interpolation.</li>
      <li><strong>Model-based Imputation:</strong> Uses statistical models like ARIMA or Kalman Filters for imputation.</li>
    </ul>

    <h3 className="text-xl font-semibold">2. Handling Outliers</h3>
    <p>
      Outliers can significantly skew predictions in time series models, especially those sensitive to extreme values. Methods for outlier detection and handling include:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Z-score Method:</strong> Flags points with Z-scores above a threshold (e.g., &gt;3 or &lt;-3).</li>
      <li><strong>Interquartile Range (IQR):</strong> Detects values outside the range of Q1−1.5×IQR to Q3+1.5×IQR.</li>
      <li><strong>Isolation Forest:</strong> A machine learning approach for high-dimensional anomaly detection.</li>
    </ul>

    <h3 className="text-xl font-semibold">3. Managing Trend and Seasonality</h3>
    <p>
      Trends and seasonal patterns introduce non-stationarity into time series data, making it harder for models to generalize. These patterns must be extracted and addressed explicitly.
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Seasonal Decomposition of Time Series (STL):</strong> Separates data into trend, seasonality, and residual.</li>
      <li><strong>Differencing:</strong> Removes trend by subtracting previous values (e.g., y<sub>t</sub> - y<sub>t-1</sub>).</li>
      <li><strong>Fourier Transforms:</strong> Identifies cyclical components using frequency analysis.</li>
    </ul>

    <h3 className="text-xl font-semibold">4. Irregular Sampling and Uneven Intervals</h3>
    <p>
      Not all time series are recorded at regular intervals. Dealing with irregular time steps involves:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Resampling:</strong> Aggregates or interpolates data to a regular frequency (e.g., hourly, daily).</li>
      <li><strong>Temporal Embedding:</strong> Includes time-delta as a feature for neural networks.</li>
    </ul>

    <h3 className="text-xl font-semibold">5. Concept Drift and Regime Change</h3>
    <p>
      In non-stationary environments, the underlying data distribution can change over time. This phenomenon, known as concept drift, necessitates adaptive modeling strategies.
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Windowing Techniques:</strong> Use only recent data in training to adapt to new trends.</li>
      <li><strong>Online Learning:</strong> Continuously updates the model as new data arrives.</li>
      <li><strong>Ensemble Models:</strong> Combine models trained on different time periods.</li>
    </ul>

    <h3 className="text-xl font-semibold">6. Noise and Signal-to-Noise Ratio</h3>
    <p>
      Many time series datasets contain noise that obscures meaningful signals. Techniques to address noise include:
    </p>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Moving Average Smoothing:</strong> Reduces short-term fluctuations.</li>
      <li><strong>Low-pass Filtering:</strong> Filters out high-frequency noise.</li>
      <li><strong>Empirical Mode Decomposition (EMD):</strong> Separates data into intrinsic mode functions.</li>
    </ul>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>Time series modeling requires tailored preprocessing steps for robust predictions.</li>
        <li>Failing to manage anomalies and non-stationarity can result in poor model generalization.</li>
        <li>State-of-the-art solutions often combine statistical and deep learning methods.</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างใน Python: การจัดการ Outliers ด้วย Z-Score</h3>
    <pre><code className="language-python">
import numpy as np
import pandas as pd

# สมมติว่ามี Series ของราคาหุ้น
prices = pd.Series([100, 102, 105, 110, 250, 108, 107])

# คำนวณ Z-score
z_scores = (prices - prices.mean()) / prices.std()
outliers = prices[np.abs(z_scores) &gt; 2.5]
print("Outliers Detected:\n", outliers)
    </code></pre>

    <p>
      ในตัวอย่างนี้ การคำนวณ Z-score ถูกใช้ในการตรวจจับค่าผิดปกติในข้อมูลราคาหุ้นอย่างง่าย
    </p>

    <h3 className="text-xl font-semibold">สรุป</h3>
    <p>
      การจัดการปัญหาเฉพาะของ Time Series อย่างเหมาะสมมีผลอย่างยิ่งต่อความสำเร็จของโมเดล ไม่ว่าจะเป็นปัญหา missing values, outliers, non-stationarity หรือ noise วิธีการที่เป็นระบบและเหมาะสมกับบริบทของข้อมูลจะเป็นรากฐานสำคัญของการพัฒนาโมเดลที่มีประสิทธิภาพ
    </p>
  </div>
</section>


<section id="case-study" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">9. Case Study: Stock Price Forecasting with LSTM</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img10} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การพยากรณ์ราคาหุ้นด้วย LSTM เป็นหนึ่งในกรณีศึกษาคลาสสิกของการนำ Deep Learning มาประยุกต์กับ Time Series ซึ่งมีการเปลี่ยนแปลงต่อเนื่องและแฝง noise อยู่มาก แนวทางนี้ถูกใช้ในงานวิจัยและอุตสาหกรรมทางการเงินทั่วโลก เช่น Bloomberg และ Quant Hedge Funds
    </p>

    <h3 className="text-xl font-semibold">วัตถุประสงค์ของกรณีศึกษา</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>นำข้อมูลราคาหุ้นย้อนหลังมาใช้สร้างโมเดลทำนายราคาวันถัดไป</li>
      <li>ใช้ LSTM เพื่อเรียนรู้ลำดับของข้อมูลที่มี dependency ระยะยาว</li>
      <li>เปรียบเทียบผลกับ baseline แบบ naive และ moving average</li>
    </ul>

    <h3 className="text-xl font-semibold">ขั้นตอนการพัฒนาโมเดล</h3>
    <ol className="list-decimal list-inside ml-6 space-y-3">
      <li><strong>รวบรวมข้อมูล:</strong> ใช้ข้อมูลราคาปิดรายวันจาก Yahoo Finance หรือ Alpha Vantage</li>
      <li><strong>Normalize ข้อมูล:</strong> ใช้ MinMaxScaler เพื่อให้ค่าระหว่าง 0–1</li>
      <li><strong>สร้าง Sliding Window:</strong> ข้อมูลย้อนหลัง 60 วันเป็น input → ทำนายราคาวันที่ 61</li>
      <li><strong>สร้างโมเดล LSTM:</strong> ใช้ 2 ชั้น LSTM ตามด้วย Dense Layer</li>
      <li><strong>Train และ Validate:</strong> ใช้ loss function MSE และ optimizer Adam</li>
      <li><strong>Evaluate:</strong> ด้วย RMSE, MAE และ visualization</li>
    </ol>

    <h3 className="text-xl font-semibold">ตัวอย่างโค้ด PyTorch (ย่อ)</h3>
    <pre><code className="language-python">
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = StockLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    </code></pre>

    <h3 className="text-xl font-semibold">ภาพการเปรียบเทียบผลลัพธ์</h3>
    <p>
      กราฟเปรียบเทียบราคาจริงและราคาที่โมเดลพยากรณ์แสดงให้เห็นว่า LSTM สามารถจับแนวโน้มระยะกลางได้ดี แต่ยังมีความผันผวนสูงในช่วงข้อมูลสั้น ทำให้ควรเสริมด้วยเทคนิคเช่น Regularization และ Ensemble
    </p>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>LSTM เหมาะกับข้อมูลที่มี dependency ระยะยาว เช่น ราคาหุ้น</li>
        <li>การ normalize มีผลสำคัญต่อ convergence ของโมเดล</li>
        <li>การใช้ window size ที่เหมาะสมช่วยให้โมเดลเรียนรู้ pattern ได้ดีขึ้น</li>
        <li>ควรเปรียบเทียบกับ baseline เช่น naive forecast เสมอ</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      กรณีศึกษานี้แสดงให้เห็นถึงความสามารถของ LSTM ในการทำ Sequence Forecasting กับข้อมูล Time Series จริง การออกแบบโมเดล การเตรียมข้อมูล และการประเมินผลที่เหมาะสมคือหัวใจของการนำ Deep Learning ไปใช้ในตลาดการเงินและการพยากรณ์ที่ซับซ้อน
    </p>
  </div>
</section>


<section id="visualization" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">10. Visualization Example</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img11} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      การแสดงผลข้อมูล (Visualization) เป็นเครื่องมือสำคัญในการทำความเข้าใจ Time Series ทั้งในด้านลักษณะของแนวโน้ม (trend), ความผันผวน, ความผิดปกติ, และผลการพยากรณ์ของโมเดล โดยการมองเห็นรูปแบบผ่านกราฟเป็นวิธีที่มนุษย์เข้าใจได้ง่ายที่สุด และเป็นขั้นตอนสำคัญที่ไม่ควรมองข้ามในการพัฒนาและตรวจสอบโมเดล
    </p>

    <h3 className="text-xl font-semibold">กราฟ Time Series พื้นฐาน</h3>
    <p>
      การสร้างกราฟแบบเส้น (line plot) เป็นวิธีมาตรฐานสำหรับการแสดง Time Series โดยใช้แกน x แสดงเวลา และแกน y แสดงค่าของตัวแปร เช่น ราคา, ความดัน, หรือความถี่ของการใช้งาน
    </p>
    <pre><code className="language-python">
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(time, value)
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()
    </code></pre>

    <h3 className="text-xl font-semibold">Highlight จุด Overfitting และความเสถียร</h3>
    <p>
      การเปรียบเทียบระหว่างผลลัพธ์ของโมเดลกับค่าจริงใน Validation Set สามารถเปิดเผยการ overfit หรือ underfit ได้อย่างชัดเจน โดยการใช้สีหรือลักษณะเส้นที่แตกต่างกันแสดง prediction กับ ground truth
    </p>
    <pre><code className="language-python">
plt.plot(dates, y_true, label='Ground Truth')
plt.plot(dates, y_pred, label='Model Prediction')
plt.legend()
plt.title('Model Prediction vs Ground Truth')
plt.show()
    </code></pre>

    <h3 className="text-xl font-semibold">การแสดงผล Multi-series</h3>
    <p>
      หากโมเดลมีหลาย feature เช่น input หลาย channel การสร้าง subplot หรือใช้สีแตกต่างสามารถช่วยให้แยกแยะผลกระทบของแต่ละ feature ต่อการพยากรณ์ได้
    </p>
    <pre><code className="language-python">
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(time, feature_1)
ax[0].set_title('Feature 1')
ax[1].plot(time, feature_2)
ax[1].set_title('Feature 2')
plt.tight_layout()
plt.show()
    </code></pre>

    <h3 className="text-xl font-semibold">การแสดงผลด้วย Plotly เพื่อ Interactive Visualization</h3>
    <p>
      Plotly เป็นเครื่องมือที่รองรับ interactive visualization ซึ่งสามารถใช้งานได้ดีใน notebook และ web application
    </p>
    <pre><code className="language-python">
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='True'))
fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted'))
fig.update_layout(title='Interactive Time Series')
fig.show()
    </code></pre>

    <h3 className="text-xl font-semibold">กราฟ Error Trend</h3>
    <p>
      การพล็อต error ในแต่ละ timestep เป็นอีกวิธีที่ช่วยระบุช่วงเวลาที่โมเดลทำงานไม่ดี และสามารถใช้เป็นแนวทางในการวิเคราะห์สาเหตุของความล้มเหลวใน prediction
    </p>
    <pre><code className="language-python">
error = y_true - y_pred
plt.plot(dates, error)
plt.title('Prediction Error Over Time')
plt.axhline(0, color='gray', linestyle='--')
plt.show()
    </code></pre>

    <h3 className="text-xl font-semibold">Insight Box</h3>
    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <ul className="list-disc list-inside space-y-2">
        <li>การแสดงภาพช่วยเปิดเผย pattern ที่ซ่อนอยู่ในข้อมูลได้อย่างรวดเร็ว</li>
        <li>เป็นเครื่องมือสำคัญในการ debug โมเดล โดยเฉพาะช่วงที่โมเดล overfit</li>
        <li>ช่วยในการสื่อสารผลการวิเคราะห์ให้ผู้ใช้หรือทีมที่ไม่ใช่สายเทคนิคเข้าใจ</li>
        <li>เมื่อใช้ควบคู่กับ metric อย่าง RMSE/MAPE จะให้มุมมองรอบด้านยิ่งขึ้น</li>
      </ul>
    </div>
  </div>
</section>


<section id="tools-libraries" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">11. Tools & Libraries</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img12} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      ในการทำงานด้าน Sequence Modeling และ Time Series การเลือกใช้เครื่องมือและไลบรารีที่เหมาะสมเป็นปัจจัยสำคัญที่ช่วยให้สามารถพัฒนาโมเดลที่มีประสิทธิภาพได้รวดเร็วและยืดหยุ่นมากขึ้น ไลบรารีเหล่านี้ครอบคลุมทั้งการจัดการข้อมูลเบื้องต้น การสร้างโมเดล ไปจนถึงการประเมินผลและ deployment
    </p>

    <h3 className="text-xl font-semibold">ไลบรารียอดนิยมสำหรับ Time Series</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Pandas:</strong> ใช้สำหรับจัดการข้อมูลชนิด Series และตารางเวลา (time-indexed data)
      </li>
      <li>
        <strong>NumPy:</strong> ใช้สำหรับคำนวณและจัดการ array ที่มีลำดับเวลา
      </li>
      <li>
        <strong>Scikit-learn:</strong> ใช้สำหรับ preprocessing และการเลือก feature ในข้อมูล Time Series
      </li>
      <li>
        <strong>TensorFlow / Keras:</strong> รองรับการสร้างโมเดล RNN, LSTM, GRU, Transformer ที่เหมาะกับข้อมูลลำดับ
      </li>
      <li>
        <strong>PyTorch:</strong> ใช้สร้างโมเดลลำดับด้วยความยืดหยุ่นสูงและ custom training loop ได้
      </li>
    </ul>

    <h3 className="text-xl font-semibold">ไลบรารีเฉพาะสำหรับ Time Series Forecasting</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li>
        <strong>Darts:</strong> ไลบรารี open-source โดย Unit8 รองรับ forecasting ทั้ง ARIMA, Prophet, RNN, Transformer พร้อม metric และ backtesting
      </li>
      <li>
        <strong>Facebook Prophet:</strong> พัฒนาโดย Meta สำหรับงาน time series forecasting ที่ไม่ต้องการ preprocessing มาก รองรับ holiday effects และ seasonality
      </li>
      <li>
        <strong>GluonTS:</strong> ไลบรารีของ Amazon สำหรับ probabilistic forecasting บน MXNet/PyTorch
      </li>
      <li>
        <strong>tslearn:</strong> ใช้สำหรับ clustering และ classification ของ time series โดยเฉพาะ
      </li>
    </ul>

    <h3 className="text-xl font-semibold">Visualization Libraries</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>Matplotlib / Seaborn:</strong> ใช้ plot trend และ visualize pattern ของข้อมูลเวลา</li>
      <li><strong>Plotly / Bokeh:</strong> สำหรับการสร้าง interactive visualization ของ series และการเปรียบเทียบ forecast</li>
    </ul>

    <h3 className="text-xl font-semibold">การเลือกเครื่องมือให้เหมาะสม</h3>
    <p>
      การเลือกไลบรารีควรขึ้นอยู่กับความซับซ้อนของโมเดล ขนาดข้อมูล และลักษณะของปัญหา เช่น งาน simple forecasting อาจใช้ Prophet หรือ ARIMA ได้ทันที ในขณะที่ deep learning จำเป็นต้องใช้ TensorFlow หรือ PyTorch และอาจรวมกับ Darts เพื่อความสะดวก
    </p>

    <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-lg">
      <h4 className="font-semibold mb-2">Insight Box:</h4>
      <ul className="list-disc list-inside space-y-2">
        <li>Prophet เหมาะกับผู้เริ่มต้น เนื่องจากใช้งานง่ายและตีความได้ดี</li>
        <li>Darts รองรับทุกประเภทโมเดลใน workflow เดียว ตั้งแต่ ARIMA ถึง Transformer</li>
        <li>การผสม Pandas + PyTorch + Darts ช่วยให้ pipeline มีความยืดหยุ่นสูง</li>
      </ul>
    </div>

    <h3 className="text-xl font-semibold">ตัวอย่างการใช้งานร่วมกัน</h3>
    <pre><code className="language-python">
from darts.models import RNNModel
from darts.datasets import AirPassengersDataset
from darts.metrics import mape

# Load dataset
series = AirPassengersDataset().load()

# Split
train, val = series.split_after(0.8)

# Build Model
model = RNNModel(model='LSTM', input_chunk_length=12, output_chunk_length=6, n_epochs=50)

# Fit
model.fit(train)

# Predict
forecast = model.predict(n=6)

# Evaluate
print("MAPE:", mape(val, forecast))
    </code></pre>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      เครื่องมือและไลบรารีเป็นองค์ประกอบสำคัญของความสำเร็จในงาน Time Series การเลือกใช้ให้เหมาะสมตามลักษณะของปัญหาจะช่วยเพิ่มความเร็ว ความแม่นยำ และความสามารถในการปรับใช้โมเดลใน production ได้อย่างมั่นคง
    </p>
  </div>
</section>

<section id="summary" className="mb-16 scroll-mt-32 min-h-[400px]">
  <h2 className="text-2xl font-semibold mb-6 text-center">12. Summary & Real-World Use Cases</h2>
  <div className="flex justify-center my-6">
    <AdvancedImage cldImg={img13} />
  </div>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-6">
    <p>
      Sequence modeling เป็นหัวใจสำคัญของงานวิจัยและการประยุกต์ในหลายอุตสาหกรรม โดยเฉพาะในบริบทของข้อมูลที่มีลำดับเวลาและมี dependency ระหว่างตัวแปรในอดีตกับอนาคต การเข้าใจประเภทของข้อมูลลำดับ, เทคนิคการประมวลผล, และการเลือกโมเดลที่เหมาะสม คือรากฐานสำคัญของการสร้างระบบที่แม่นยำและยืดหยุ่นในโลกจริง
    </p>

    <h3 className="text-xl font-semibold">ประเด็นสำคัญที่ควรจดจำ</h3>
    <ul className="list-disc list-inside space-y-2">
      <li>ข้อมูล Time Series มีโครงสร้างเฉพาะ: ลำดับของข้อมูลส่งผลต่อความหมาย</li>
      <li>ประเภทของ Task มีผลต่อการเลือกโมเดล เช่น Sequence→Value หรือ Sequence→Label</li>
      <li>Time Series อาจมีลักษณะ non-stationary, missing values หรือ outliers ซึ่งต้องมีการจัดการอย่างระมัดระวัง</li>
      <li>โมเดล RNN, LSTM, GRU, และ Transformer ล้วนมีข้อดีเฉพาะในแต่ละบริบท</li>
      <li>Evaluation metric เช่น MAE, RMSE, MAPE ต้องเลือกให้เหมาะกับลักษณะปัญหา</li>
    </ul>

    <h3 className="text-xl font-semibold">ตัวอย่างการประยุกต์ในโลกความจริง</h3>
    <div className="overflow-x-auto">
      <table className="table-auto w-full text-sm border border-gray-300 dark:border-gray-700">
        <thead>
          <tr className="bg-gray-100 dark:bg-gray-800">
            <th className="border px-4 py-2">Use Case</th>
            <th className="border px-4 py-2">รายละเอียด</th>
            <th className="border px-4 py-2">องค์กรตัวอย่าง</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2">Financial Forecasting</td>
            <td className="border px-4 py-2">การคาดการณ์ราคาหุ้น, อัตราแลกเปลี่ยน, หรือดัชนีทางเศรษฐกิจ</td>
            <td className="border px-4 py-2">Bloomberg, Quant Hedge Funds</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Demand Prediction</td>
            <td className="border px-4 py-2">คาดการณ์ความต้องการของสินค้าเพื่อปรับคลังสินค้าและการผลิต</td>
            <td className="border px-4 py-2">Amazon, Walmart</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Health Monitoring</td>
            <td className="border px-4 py-2">วิเคราะห์ข้อมูลชีวภาพ เช่น ECG, heart rate, หรือ oxygen level</td>
            <td className="border px-4 py-2">Fitbit, Apple Health</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Energy Consumption Forecasting</td>
            <td className="border px-4 py-2">คาดการณ์การใช้พลังงานสำหรับการจัดการสมาร์ทกริด</td>
            <td className="border px-4 py-2">Smart Grid Providers, Siemens</td>
          </tr>
          <tr>
            <td className="border px-4 py-2">Language Modeling</td>
            <td className="border px-4 py-2">การทำนายลำดับของคำถัดไปในประโยค</td>
            <td className="border px-4 py-2">OpenAI (GPT), Google (BERT)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold">คำแนะนำจากมหาวิทยาลัยชั้นนำ</h3>
    <ul className="list-disc list-inside ml-6 space-y-2">
      <li><strong>MIT (6.S191):</strong> ย้ำถึงความสำคัญของการใช้ Recurrent Architecture สำหรับ sequence ที่มี dependency ในระยะยาว เช่น LSTM และ GRU</li>
      <li><strong>Stanford CS224n:</strong> เสนอการใช้ Transformer แทน RNN ใน NLP ด้วยข้อดีด้าน parallelization และ long-range dependency</li>
      <li><strong>Carnegie Mellon:</strong> แนะนำเทคนิค preprocessing ข้อมูล time series เช่น Seasonal Adjustment และ Fourier Decomposition</li>
    </ul>

    <h3 className="text-xl font-semibold">บทสรุป</h3>
    <p>
      Sequence modeling ไม่ใช่เพียงแค่เครื่องมือในการพยากรณ์ แต่เป็นกุญแจสำคัญของการเข้าใจโครงสร้างข้อมูลที่มีลำดับ ซึ่งปรากฏในทุกมิติของโลกดิจิทัล ตั้งแต่การเงิน พลังงาน จนถึงสุขภาพ การนำเทคนิคเหล่านี้ไปใช้อย่างเหมาะสมสามารถสร้างมูลค่าและ insight ได้อย่างมหาศาล
    </p>
  </div>
</section>


<section id="references" className="mb-16 scroll-mt-32 min-h-[300px]">
  <h2 className="text-2xl font-semibold mb-6 text-center"> แหล่งอ้างอิงข้อมูล (References)</h2>
  <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 prose dark:prose-invert text-base leading-relaxed space-y-4">
    <ul className="list-disc list-inside space-y-2">
      <li>
        <strong>MIT 6.S191: Introduction to Deep Learning</strong> - Lecture Notes & Labs (https://introtodeeplearning.mit.edu)
      </li>
      <li>
        <strong>Stanford CS224n: Natural Language Processing with Deep Learning</strong> - Sequence Modeling Lectures (https://web.stanford.edu/class/cs224n/)
      </li>
      <li>
        <strong>DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks</strong>, Amazon Research – Salinas et al. (2019)
      </li>
      <li>
        <strong>Facebook Prophet</strong> – Open-source time series forecasting library (https://facebook.github.io/prophet/)
      </li>
      <li>
        <strong>Unit8 Darts: Time Series Made Easy in Python</strong> (https://github.com/unit8co/darts)
      </li>
      <li>
        <strong>Harvard University: Data Science for Time Series</strong> – Department of Statistics course materials
      </li>
      <li>
        <strong>Time Series Forecasting Principles with Python</strong> – Jason Brownlee, Machine Learning Mastery
      </li>
      <li>
        <strong>Deep Learning for Time Series Forecasting</strong> – Google Cloud AI Research Whitepapers
      </li>
      <li>
        <strong>PyTorch Forecasting Documentation</strong> (https://pytorch-forecasting.readthedocs.io/)
      </li>
      <li>
        <strong>Carnegie Mellon University: Advanced Machine Learning for Time Series</strong> – Research Reports and Lecture Notes
      </li>
      <li>
        <strong>Transformer Models for Time Series Forecasting: Papers with Code</strong> (https://paperswithcode.com/task/time-series-forecasting)
      </li>
    </ul>
    <div className="mt-6 bg-blue-50 dark:bg-gray-800 border-l-4 border-blue-400 dark:border-blue-600 p-4 rounded">
      <p className="font-medium">หมายเหตุ:</p>
      <p>
        ข้อมูลทั้งหมดในบทนี้ถูกรวบรวมจากงานวิจัยที่ผ่านการตีพิมพ์ การบรรยายจากมหาวิทยาลัยชั้นนำระดับโลก และคู่มือจากเครื่องมือจริงในอุตสาหกรรม เพื่อให้ผู้อ่านสามารถเรียนรู้ได้อย่างถูกต้องตามหลักวิชาการและใช้งานได้จริงในระดับ production
      </p>
    </div>
  </div>
</section>



          <section id="quiz" className="mb-16 scroll-mt-32 min-h-[500px]">
            <MiniQuiz_Day26 theme={theme} />
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
        <ScrollSpy_Ai_Day26 />
      </div>

      <SupportMeButton />
    </div>
  );
};

export default Day26_SequenceTimeSeries;
