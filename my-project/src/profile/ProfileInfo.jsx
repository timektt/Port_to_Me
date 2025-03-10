import React from "react";

const ProfileInfo = ({ navigate }) => {
  return (
    <div className="flex items-center gap-2">
      {/* ✅ คลิกที่โลโก้เพื่อกลับหน้า Home */}
      <img 
        src="/spm2.jpg" 
        alt="Profile" 
        className="w-30 h-12 rounded-full cursor-pointer" 
        onClick={() => navigate("/")} // ✅ กลับหน้า Home
      />

      {/* ✅ ชื่อเว็บ & Navigation */}
      <h1 className="text-3xl font-bold flex items-center gap-5 relative -top-1.5"> 
        Supermhee 

        {/* ✅ กด "Courses" จะไปที่หน้ารวมคอร์ส */}
        <span 
          className="text-xl text-gray-400 hidden md:inline relative top-1 cursor-pointer hover:text-gray-600"
          onClick={() => navigate("/courses")}
        >
          Courses
        </span>

        {/* ❌ "Post" เป็นข้อความธรรมดา (ไม่สามารถกดได้) */}
        <span className="text-xl text-gray-400 hidden md:inline relative top-1"> |  Post</span>
      </h1>
    </div>
  );
};

export default ProfileInfo;
