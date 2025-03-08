import React, { useState } from "react";
import SupportMeModal from "./SupportMeModal";

const SupportMeButton = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      {/* ปุ่ม Support Me */}
      <div className="fixed bottom-6 left-6 z-50">
        <button
          onClick={() => setIsModalOpen(true)}
          className="bg-green-500 text-white px-4 py-2 rounded-full flex items-center gap-3 shadow-lg hover:bg-green-800 transition"
        >
          {/* รูปโปรไฟล์วงกลม */}
          <img
            src="/spm2.jpg"
            alt="Support Me"
            className="w-9 h-9 rounded-full border-2 border-white object-cover"
          />
          Support me
        </button>
      </div>

      {/* แสดง Modal ถ้า isModalOpen = true */}
      {isModalOpen && <SupportMeModal closeModal={() => setIsModalOpen(false)} />}
    </>
  );
};

export default SupportMeButton;
