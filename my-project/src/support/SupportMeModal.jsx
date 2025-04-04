import React, { useState } from "react";
import { FaTimes, FaCoffee, FaPlus, FaMinus } from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";

const SupportMeModal = ({ closeModal }) => {
  const [quantity, setQuantity] = useState(1);
  const pricePerCoffee = 40;
  const total = quantity * pricePerCoffee;

  // ✅ เพิ่มฟังก์ชันเปิด BuyMeACoffee
  const handleDonate = () => {
    const url = `https://buymeacoffee.com/superbear?amount=${total}`;
    window.open(url, "_blank");
  };

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 flex items-end justify-start z-50 px-4 pb-20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={closeModal}
      >
        <motion.div
          className="bg-white p-3 rounded-lg shadow-lg w-64 relative"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 50, opacity: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            className="absolute top-2 right-2 text-gray-600 hover:text-gray-800"
            onClick={closeModal}
          >
            <FaTimes size={16} />
          </button>

          <div className="flex justify-center">
            <img
              src="/spm2.jpg"
              alt="Profile"
              className="w-14 h-14 rounded-full border-2 border-gray-300 object-cover"
            />
          </div>

          <h2 className="text-sm font-semibold text-center mt-3 text-gray-700">
            Buy a Coffee for Superbear
          </h2>

          <div className="flex items-center justify-center mt-3">
            <FaCoffee size={22} className="text-gray-700 mr-3" />
            <button
              className="w-8 h-8 flex items-center justify-center bg-gray-200 text-gray-700 rounded-full border border-gray-400 hover:bg-gray-300 transition"
              onClick={() => setQuantity(Math.max(1, quantity - 1))}
            >
              <FaMinus size={14} />
            </button>
            <span className="w-10 h-8 flex items-center justify-center text-sm font-semibold bg-white border border-gray-400 rounded-full mx-2 text-black">
              {quantity}
            </span>
            <button
              className="w-8 h-8 flex items-center justify-center bg-gray-200 text-gray-700 rounded-full border border-gray-400 hover:bg-gray-300 transition"
              onClick={() => setQuantity(quantity + 1)}
            >
              <FaPlus size={14} />
            </button>
          </div>

          <div className="bg-gray-300 p-3 mt-3 rounded-lg text-center text-sm font-semibold text-black">
            ฿ {total}
          </div>

          <input
            type="text"
            placeholder="Your name or nickname"
            className="w-full p-2 mt-3 border border-gray-300 rounded-md text-sm placeholder-gray-600 text-black"
          />

          <textarea
            placeholder="Your message"
            className="w-full p-2 mt-2 border border-gray-300 rounded-md text-sm placeholder-gray-600 text-black"
            rows="3"
          ></textarea>

          {/* ✅ ปุ่ม Donate แบบใช้งานจริง */}
          <button
            onClick={handleDonate}
            className="w-full bg-blue-600 text-white p-3 rounded-lg mt-3 text-sm hover:bg-blue-700 transition"
          >
            Donate ฿{total}
          </button>

          <p className="text-xs text-gray-500 text-center mt-2">
            Payments go directly to Superbear
          </p>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default SupportMeModal;
