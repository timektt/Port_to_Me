import React from "react";
import { useNavigate } from "react-router-dom";
import Comments from "../../../../components/common/Comments";
import SupportMeButton from "../../../../support/SupportMeButton";
import ScrollSpy_Ai_Day61 from "../scrollspy/scrollspyDay61-80/ScrollSpy_Ai_Day61";
import MiniQuiz_Day61 from "../miniquiz/miniquizDay61-80/MiniQuiz_Day61";
import { Cloudinary } from "@cloudinary/url-gen";
import { AdvancedImage } from "@cloudinary/react";
import { scale } from "@cloudinary/url-gen/actions/resize";
import AiSidebar from "../../../../components/common/sidebar/AiSidebar";

const Day61_IntroRL = ({ theme }) => {
  const navigate = useNavigate();
  const cld = new Cloudinary({ cloud: { cloudName: "dxtnq9fxw" } });

  // Image assets (ตามลำดับหัวข้อ)
  const img1 = cld.image("Day61_1").format("auto").quality("auto").resize(scale().width(660));
  const img2 = cld.image("Day61_2").format("auto").quality("auto").resize(scale().width(660));
  const img3 = cld.image("Day61_3").format("auto").quality("auto").resize(scale().width(660));
  const img4 = cld.image("Day61_4").format("auto").quality("auto").resize(scale().width(660));
  const img5 = cld.image("Day61_5").format("auto").quality("auto").resize(scale().width(660));
  const img6 = cld.image("Day61_6").format("auto").quality("auto").resize(scale().width(660));
  const img7 = cld.image("Day61_7").format("auto").quality("auto").resize(scale().width(660));
  const img8 = cld.image("Day61_8").format("auto").quality("auto").resize(scale().width(660));
  const img9 = cld.image("Day61_9").format("auto").quality("auto").resize(scale().width(660));
  const img10 = cld.image("Day61_10").format("auto").quality("auto").resize(scale().width(660));
  const img11 = cld.image("Day61_11").format("auto").quality("auto").resize(scale().width(660));
  const img12 = cld.image("Day61_12").format("auto").quality("auto").resize(scale().width(660));

  return (
    <div className={`relative min-h-screen ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      {/* Sidebar */}
      <div className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 z-40">
        <AiSidebar theme={theme} />
      </div>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto p-6 pt-20">
        <ScrollSpy_Ai_Day61 />

        <section id="intro" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">1. บทนำ: Reinforcement Learning คืออะไร?</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img1} />
          </div>
        </section>

        <section id="comparison" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">2. เปรียบเทียบกับ Learning แบบอื่น</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img2} />
          </div>
        </section>

        <section id="components" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">3. ส่วนประกอบพื้นฐานของ RL</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img3} />
          </div>
        </section>

        <section id="loop" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">4. Loop การทำงานของ RL (Agent–Environment Interaction)</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img4} />
          </div>
        </section>

        <section id="types" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">5. Types of Reinforcement Learning</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img5} />
          </div>
        </section>

        <section id="examples" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">6. ตัวอย่างปัญหา RL ที่เป็นมาตรฐาน</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img6} />
          </div>
        </section>

        <section id="advantages" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">7. จุดแข็งของ RL</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img7} />
          </div>
        </section>

        <section id="limitations" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">8. ข้อจำกัดของ RL</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img8} />
          </div>
        </section>

        <section id="real-world" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">9. ใช้งานในโลกจริง</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img9} />
          </div>
        </section>

        <section id="insight-box" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">10. Insight Box</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img10} />
          </div>
        </section>

        <section id="references" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">11. Academic References</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img11} />
          </div>
        </section>

        <section id="quiz" className="mb-16 scroll-mt-32 min-h-[400px]">
          <h2 className="text-2xl font-semibold mb-6 text-center">12. Mini Quiz</h2>
          <div className="flex justify-center my-6">
            <AdvancedImage cldImg={img12} />
          </div>
        </section>

        <MiniQuiz_Day61 />
        <SupportMeButton theme={theme} />
        <Comments theme={theme} />
      </main>
    </div>
  );
};

export default Day61_IntroRL;
