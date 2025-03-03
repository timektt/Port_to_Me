import React from "react";
import LatestUpdates from "./LatestUpdates";
import PopularTags from "./PopularTags"; // ✅ นำเข้า PopularTags

const videos = [
  { id: "dQw4w9WgXcQ", title: "Python Series", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
  { id: "3JZ_D3ELwOQ", title: "C++ Data Structure & Algorithm", description: "คอร์สนี้เหมาะทุกคนที่อยากเรียนพื้นฐานการเขียนโปรแกรมและ Algorithm" },
  { id: "tgbNymZ7vqY", title: "GoAPI Essential", description: "คอร์สนี้เหมาะทุกคนที่อยากเข้าใจ API และ Backend ผ่าน Go" },
  { id: "M7lc1UVf-VE", title: "Vue Firebase Masterclass", description: "คอร์สสอนสร้างโปรเจกต์ด้วย Vue และ Firebase" },
  { id: "Zi_XLOBDo_Y", title: "Web Development 101", description: "คอร์สเรียนพื้นฐานสำหรับเริ่มต้นสร้างเว็บไซต์" },
  { id: "sBws8MSXN7A", title: "Basic Programming", description: "คอร์สเรียนพื้นฐานที่ Programmer ทุกคนควรรู้" },
];

const VideoGrid = () => {
  return (
    <div className="p-8 bg-gray-900 max-w-screen-lg mx-auto w-full">
      <h2 className="text-2xl md:text-3xl font-bold text-white text-left mb-6">🎬 Latest Courses</h2>
      <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {videos.map((video) => (
          <div key={video.id} className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <iframe
              className="w-full h-[180px] md:h-[220px] rounded-lg"
              src={`https://www.youtube.com/embed/${video.id}`}
              title={video.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
            <h3 className="text-white font-semibold text-md md:text-lg mt-3">{video.title}</h3>
            <p className="text-gray-400 text-sm md:text-base mt-1">{video.description}</p>
            <a href="#" className="text-green-400 text-sm md:text-base mt-2 inline-block">อ่าน documents</a>
          </div>
        ))}
      </div>

      {/* ✅ แสดง `LatestUpdates` ถัดจากวิดีโอ */}
      <div className="w-full">
        <LatestUpdates />
      </div>

      {/* ✅ เพิ่ม `PopularTags` ถัดจาก `LatestUpdates` */}
      <div className="w-full">
        <PopularTags />
      </div>
    </div>
  );
};

export default VideoGrid;
