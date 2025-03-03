import React from "react";

const SupportMeButton = () => {
  return (
    <div className="fixed bottom-6 left-6 z-50">
      <a
        href="#"
        className="bg-green-500 text-white px-5 py-2 rounded-full flex items-center gap-3 shadow-lg hover:bg-green-800 transition"
      >
        {/* **📌 รูปภาพมีกรอบวงกลม และขนาดเหมาะสม** */}
        <img
          src="/spm2.jpg"
          alt="Support Me"
          className="w-10 h-10 rounded-full border-2 border-white object-cover"
        />
        Support me
      </a>
    </div>
  );
};

export default SupportMeButton;
