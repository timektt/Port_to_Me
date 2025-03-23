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

<h1 className="hidden sm:flex flex-col md:flex-row md:items-center text-xl sm:text-2xl md:text-3xl font-bold gap-1 md:gap-6 relative -top-1.5">
  <span>Superbear</span>

  <div className="flex items-center gap-4 text-xl text-gray-400">
    {/* ✅ กด "Courses" จะไปที่หน้ารวมคอร์ส */}
    <span
      className="cursor-pointer hover:text-gray-600 hidden md:inline"
      onClick={() => navigate("/courses")}
    >
      Courses
    </span>

    {/* ❌ "Post" เป็นข้อความธรรมดา */}
    <span className="hidden md:inline">| Post</span>
  </div>
</h1>
    </div>
  );
};

export default ProfileInfo;
