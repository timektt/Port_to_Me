import React, { useState } from "react";
import { FaTimes, FaCoffee, FaPlus, FaMinus } from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";

const SupportMeModal = ({ closeModal }) => {
  const [quantity, setQuantity] = useState(1);
  const pricePerCoffee = 40;
  const total = quantity * pricePerCoffee;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 flex items-end justify-start z-50 px-4 pb-20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={closeModal} // ปิด Modal เมื่อคลิกด้านนอก
      >
        {/* ✅ ปรับความกว้างเล็กลง และเพิ่มความสูงขององค์ประกอบ */}
        <motion.div
          className="bg-white p-3 rounded-lg shadow-lg w-64 relative"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 50, opacity: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()} // ป้องกันการปิด Modal เมื่อคลิกด้านใน
        >
          {/* ปุ่มปิด */}
          <button
            className="absolute top-2 right-2 text-gray-600 hover:text-gray-800"
            onClick={closeModal}
          >
            <FaTimes size={16} />
          </button>

          {/* รูปโปรไฟล์ */}
          <div className="flex justify-center">
            <img
              src="/spm2.jpg"
              alt="Profile"
              className="w-14 h-14 rounded-full border-2 border-gray-300 object-cover"
            />
          </div>

          {/* หัวข้อ */}
          <h2 className="text-sm font-semibold text-center mt-3 text-gray-700">
            Buy a Coffee for Supermhee
          </h2>

          {/* ไอคอนกาแฟ & จำนวน */}
          <div className="flex items-center justify-center mt-3">
            <FaCoffee size={30} className="text-gray-700" />
            <button
              className="bg-gray-500 px-3 py-1 mx-2 rounded-md hover:bg-gray-800"
              onClick={() => setQuantity(Math.max(1, quantity - 1))}
            >
              <FaMinus size={16} />
            </button>
            <span className="text-sm font-semibold">{quantity}</span>
            <button
              className="bg-gray-500 px-3 py-1 mx-2 rounded-md hover:bg-gray-800"
              onClick={() => setQuantity(quantity + 1)}
            >
              <FaPlus size={16} />
            </button>
          </div>

          {/* ราคาทั้งหมด */}
          <div className="bg-gray-300 p-3 mt-3 rounded-lg text-center text-sm font-semibold text-gray-800">
            ฿ {total}
          </div>

          {/* Input ช่องชื่อและข้อความ */}
          <input
            type="text"
            placeholder="Your name"
            className="w-full p-2 mt-3 border border-gray-300 rounded-md text-xs"
          />
          <textarea
            placeholder="Your message"
            className="w-full p-2 mt-2 border border-gray-300 rounded-md text-xs"
            rows="3"
          ></textarea>

          {/* ปุ่ม Donate */}
          <button className="w-full bg-blue-600 text-white p-3 rounded-lg mt-3 text-sm hover:bg-blue-700 transition">
            Donate ฿{total}
          </button>

          {/* คำอธิบาย */}
          <p className="text-xs text-gray-500 text-center mt-2">
            Payments go directly to Supermhee
          </p>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default SupportMeModal;
