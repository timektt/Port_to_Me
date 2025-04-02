import React, { useEffect, useState } from "react";
import { AiOutlinePlus, AiOutlineDelete } from "react-icons/ai";

const AdminUpdates = () => {
  const [updates, setUpdates] = useState([]);
  const [form, setForm] = useState({
    category: "",
    level: "",
    title: "",
    path: "",
    date: "",
  });

  useEffect(() => {
    fetch("/data/updates.json")
      .then((res) => res.json())
      .then((data) => setUpdates(data))
      .catch((err) => console.error("❌ Error loading updates.json:", err));
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleAdd = () => {
    if (!form.title || !form.path || !form.date)
      return alert("กรุณากรอกข้อมูลที่จำเป็นให้ครบ");
    const newUpdates = [...updates, form];
    setUpdates(newUpdates);
    downloadJSON(newUpdates);
    setForm({ category: "", level: "", title: "", path: "", date: "" });
  };

  const handleDelete = (index) => {
    const updated = updates.filter((_, i) => i !== index);
    setUpdates(updated);
    downloadJSON(updated);
  };

  const downloadJSON = (data) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "updates.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 mb-40 text-white bg-black/80 border border-yellow-500 rounded-2xl shadow-xl">
      <h1 className="text-3xl font-bold mb-8 text-yellow-400">จัดการรายการอัปเดต</h1>

      {/* ✅ ฟอร์มเพิ่มข้อมูล */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        <input
          name="category"
          value={form.category}
          onChange={handleChange}
          placeholder="หมวดหมู่ (เช่น python-series)"
          className="p-2 rounded-lg bg-white text-black placeholder-gray-500"
        />
        <input
          name="level"
          value={form.level}
          onChange={handleChange}
          placeholder="ระดับ (เช่น 101: Basic Python)"
          className="p-2 rounded-lg bg-white text-black placeholder-gray-500"
        />
        <input
          name="title"
          value={form.title}
          onChange={handleChange}
          placeholder="ชื่อเรื่อง"
          className="p-2 rounded-lg bg-white text-black placeholder-gray-500"
        />
        <input
          name="path"
          value={form.path}
          onChange={handleChange}
          placeholder="เส้นทางลิงก์ (เช่น /courses/python-series/intro)"
          className="p-2 rounded-lg bg-white text-black placeholder-gray-500"
        />
        <input
          name="date"
          value={form.date}
          onChange={handleChange}
          placeholder="วันที่ (เช่น 01/04/2568)"
          className="p-2 rounded-lg bg-white text-black placeholder-gray-500"
        />

        <button
          onClick={handleAdd}
          className="bg-gradient-to-r from-gray-700 to-gray-900 hover:from-gray-800 hover:to-black text-white py-2 rounded-xl font-semibold transition flex items-center justify-center gap-2 col-span-full"
        >
          <AiOutlinePlus className="text-xl" /> เพิ่มอัปเดต
        </button>
      </div>

      {/* ✅ รายการอัปเดต */}
      <ul className="space-y-5">
        {updates.map((item, index) => (
          <li
            key={index}
            className="bg-zinc-900 border border-yellow-400 rounded-xl p-4 shadow-md flex justify-between items-center flex-col sm:flex-row gap-3"
          >
            <div className="text-left w-full sm:w-auto">
              <h2 className="font-bold text-lg text-yellow-300">{item.title}</h2>
              <p className="text-sm text-gray-400">
                {item.category} / {item.level}
              </p>
              <p className="text-sm text-gray-500 mt-1">{item.date}</p>
            </div>
            <button
              onClick={() => handleDelete(index)}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-1 rounded-full font-semibold flex items-center gap-2"
            >
              <AiOutlineDelete className="text-lg" /> ลบ
            </button>
          </li>
        ))}
      </ul>

      <p className="mt-12 text-sm text-gray-400 text-center">
        หลังจากเพิ่มหรือลบข้อมูล ไฟล์ <code>updates.json</code> จะถูกดาวน์โหลดอัตโนมัติ<br />
        กรุณานำไฟล์นี้ไปวางที่ <code>/public/data/updates.json</code>
      </p>
    </div>
  );
};

export default AdminUpdates;
