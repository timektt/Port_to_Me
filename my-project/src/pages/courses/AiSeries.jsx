import React, { useState, useEffect } from "react";
import { useNavigate, useParams, Outlet } from "react-router-dom";
import Navbar from "../../components/common/Navbar";
import AiSidebar from "../../components/common/sidebar/AiSidebar";
import SupportMeButton from "../../support/SupportMeButton";
import Comments from "../../components/common/Comments";
import AiMobileMenu from "../../components/common/sidebar/MobileMenus/AiMobileMenu";
import Breadcrumb from "../../components/common/Breadcrumb";
import { FaPlay } from "react-icons/fa";

const AiSeries = ({ theme, setTheme }) => {
  const { "*": subPage } = useParams();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setSidebarOpen(false);
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <div className="fixed w-full z-50 top-0">
        <Navbar theme={theme} setTheme={setTheme} onMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen)} />
      </div>

      <div className="hidden md:block fixed top-16 left-0 h-[calc(100vh-4rem)] w-64 z-40">
        <AiSidebar theme={theme} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      </div>

      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          <AiMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
        </div>
      )}

      <main className="flex-1 md:ml-64 mt-16 p-4 md:p-6 relative z-10">
        <div className="max-w-5xl mx-auto">
          <Breadcrumb courseName="AISeries" theme={theme} />
          {subPage ? <Outlet /> : (
            <>
              <h1 className="text-3xl md:text-4xl font-bold mt-4">AI Series</h1>
              <div className={`p-4 mt-4 rounded-md shadow-md ${theme === "dark" ? "bg-yellow-700 text-white" : "bg-yellow-300 text-black"}`}>
                <strong className="text-lg">⚠ WARNING</strong>
                <p>Series นี้อยู่ระหว่างการพัฒนา เนื้อหาอาจมีการปรับปรุง</p>
              </div>

              {/* ✅ ส่วนแนะนำภาพรวมของซีรีส์ AI */}
              <div className={`mt-6 p-6  text-center ${theme === "dark" ? "" : "bg-white border-gray-300"}`}>
              <h2 className="text-2xl font-bold mb-4">เริ่มต้นสู่โลกแห่ง AI อย่างมั่นใจ</h2>
              <img src="/ai_series.png" alt="AI Series" className="w-full  rounded-lg object-cover mb-4" />
<p className="mb-4 text-base leading-relaxed">
  หากเคยสงสัยว่า AI คืออะไร ทำงานอย่างไร และจะเปลี่ยนโลกไปยังไง — ท่านมาถูกที่แล้ว! คอร์สนี้จะพาคุณเริ่มต้นจากศูนย์
  ไปจนถึงเข้าใจแนวคิดสำคัญ เช่น Vector, Matrix, Neural Network, Optimization และการทำให้ AI ฉลาดขึ้นจริงในโลกแห่งการใช้งาน
</p>
<p className="mb-6 text-base leading-relaxed">
  ไม่ว่าคุณจะไม่มีพื้นฐานมาก่อนหรืออยากทบทวนแบบเป็นระบบ ซีรีส์นี้จะช่วยให้คุณวางรากฐานการเป็นนักพัฒนา AI ได้อย่างมั่นคง
  พร้อมตัวอย่างโค้ดและภารกิจให้ฝึกมือจริง
</p>
<p className="mb-4 text-base leading-relaxed">
  AI ไม่ใช่เพียงแค่เครื่องมือหรือเทคโนโลยีใหม่ล่าสุด แต่คือพลังที่จะเปลี่ยนแปลงทุกอุตสาหกรรม ตั้งแต่การแพทย์ การเงิน พลังงาน ไปจนถึงศิลปะ การศึกษา และแม้แต่ชีวิตประจำวันของคุณเอง
  มันคือการเรียนรู้ของเครื่องจากข้อมูลมหาศาล เพื่อให้สามารถตัดสินใจ วิเคราะห์ และคาดการณ์ได้ราวกับมนุษย์ (หรือดียิ่งกว่า)
</p>
<p className="mb-4 text-base leading-relaxed">
  ในแต่ละวันของซีรีส์นี้ คุณจะได้สัมผัสกับคอนเซ็ปต์ที่เคยดูซับซ้อนให้กลายเป็นเรื่องที่เข้าใจง่าย ผ่านภาษาธรรมชาติ การเปรียบเทียบแบบเห็นภาพ และโค้ด Python ที่คุณสามารถรันและทดลองได้ด้วยตัวเอง
  ความเข้าใจที่แท้จริงจะเกิดขึ้นเมื่อคุณได้ลงมือทำ ไม่ใช่แค่ฟังหรือดูเท่านั้น
</p>
<p className="mb-4 text-base leading-relaxed">
  เราจะเริ่มต้นจากพื้นฐานของข้อมูล คือเวกเตอร์และเมทริกซ์ ที่อยู่เบื้องหลังทุกการคำนวณใน AI ไม่ว่าจะเป็นการวิเคราะห์ภาพ เสียง ข้อความ หรือแม้กระทั่งพฤติกรรมของผู้ใช้
  จากนั้นคุณจะเข้าใจการแปลงข้อมูล (Transformation) เพื่อให้ AI เรียนรู้ได้ดีขึ้น และเข้าใจความสำคัญของ Activation Function ที่ช่วยให้โมเดลแสดงพฤติกรรมซับซ้อนได้
</p>
<p className="mb-4 text-base leading-relaxed">
  ในบทต่อๆ ไป คุณจะได้เรียนรู้เรื่อง Optimization ที่ทำให้โมเดลเก่งขึ้น Loss Function ที่เปรียบเหมือนเข็มทิศ และ Backpropagation ที่ช่วยให้ AI ปรับตัวดีขึ้น
  พร้อมทั้งเทคนิค Regularization เพื่อป้องกันการจำแบบผิดๆ (Overfitting) ซึ่งเป็นปัญหาหลักของหลายโปรเจกต์ AI
</p>
<p className="mb-4 text-base leading-relaxed">
  ไม่ใช่แค่ทฤษฎีเท่านั้น คุณจะได้ฝึกโค้ดจริงด้วย PyTorch และ Keras พร้อมทำ Quiz สั้นๆ ทดสอบความเข้าใจในแต่ละบท พร้อม insight box และ interactive mini tasks ให้ได้ทดลองทำทันที
</p>
<p className="mb-4 text-base leading-relaxed">
  เมื่อคุณเรียนจบซีรีส์นี้ คุณจะไม่ใช่แค่ "เข้าใจ AI" แต่จะสามารถอธิบายได้ สาธิตได้ และเริ่มต้นพัฒนาโปรเจกต์จริงของตัวเองได้เลย
  ไม่ว่าคุณจะอยากเป็น AI Developer, Data Scientist, หรือผู้ประกอบการที่ใช้ AI เพิ่มศักยภาพให้ธุรกิจ — จุดเริ่มต้นคือที่นี่
</p>
<p className="mb-6 text-base leading-relaxed">
  ยินดีต้อนรับสู่การเดินทางที่จะเปลี่ยนมุมมองของคุณต่อโลก เทคโนโลยี และตัวคุณเอง เตรียมตัวให้พร้อม แล้วคลิกปุ่มด้านล่างเพื่อเริ่มบทแรกได้เลย!
</p>
                <button
                onClick={() => navigate("/courses/ai/intro-to-vectors-matrices")}
                className="inline-flex items-center justify-center gap-3 px-6 py-3 text-lg font-semibold bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-indigo-500 hover:to-blue-600 text-white rounded-xl shadow-lg transform hover:scale-105 transition-all duration-300"
                >
                <FaPlay className="text-white text-base" />
                เริ่มเรียนเลย
                </button>
              </div>
            </>
          )}
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

          {/* ปุ่ม Previous & Next (ยังเก็บไว้หากเข้าหัวข้อย่อย) */}
          <div className="flex justify-between items-center max-w-5xl mx-auto px-4 mt-4">
            {/* prevTopic และ nextTopic ใช้ได้เฉพาะตอนดู subPage */}
          </div>
        </div>
      </main>
      <SupportMeButton />
    </div>
  );
};

export default AiSeries;
