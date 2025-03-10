import React from "react";

const ProfileInfo = ({ navigate }) => {
  return (
    <div className="flex items-center gap-2">
      {/* ✅ คลิกที่โลโก้เพื่อกลับหน้า Home */}
      <img 
        src="/spm2.jpg" 
        alt="Profile" 
        className="w-30 h-12 rounded-full cursor-pointer" 
        onClick={() => navigate("/")} // ✅ ย้ายการคลิกกลับหน้า Home มาไว้ที่โลโก้
      />

      {/* ✅ แสดงชื่อเว็บ */}
      <h1 className="text-3xl font-bold flex items-center gap-5 relative -top-1.5"> 
        Supermhee 

        {/* ❌ เอาฟังก์ชัน navigate ออกจาก "Courses" (ทำให้เป็นข้อความธรรมดา) */}
        <span className="text-xl text-gray-400 hidden md:inline relative top-1">Courses</span>

        {/* ❌ เอาฟังก์ชัน navigate ออกจาก "| Post" (ทำให้เป็นข้อความธรรมดา) */}
        <span className="text-xl text-gray-400 hidden md:inline relative top-1"> |  Post</span>
      </h1>
    </div>
  );
};

export default ProfileInfo;
