import React from "react";

const SupportMeButton = () => {
  return (
    <div className="fixed bottom-6 left-6 z-50">
      <a
        href="https://buymeacoffee.com/superbear"
        target="_blank"
        rel="noopener noreferrer"
        className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-full flex items-center gap-3 shadow-lg transition"
      >
        <img
          src="/spm2.jpg"
          alt="Support Me"
          className="w-9 h-9 rounded-full border-2 border-white object-cover"
        />
        <span className="font-semibold">Support me</span>
      </a>
    </div>
  );
};

export default SupportMeButton;
