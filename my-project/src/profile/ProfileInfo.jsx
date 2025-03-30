import React from "react";

const ProfileInfo = ({ navigate }) => {
  return (
    <div className="flex items-center gap-2">
      {/* ✅ โลโก้ responsive ป้องกันบีบใน mobile */}
      <img 
        src="/spm2.jpg" 
        alt="Profile" 
        onClick={() => navigate("/")} // ✅ กลับหน้า Home
        className="h-10 md:h-12 w-auto max-w-[120px] rounded-full object-cover cursor-pointer"
      />

      <h1 className="hidden sm:flex flex-col md:flex-row md:items-center text-xl sm:text-2xl md:text-3xl font-bold gap-1 md:gap-6 relative -top-1.5">
        <span>Superbear</span>

        <div className="flex items-center gap-3 text-base text-gray-400">
          {/* Courses */}
          <span
            className="cursor-pointer hover:text-gray-600 hidden lg:inline"
            onClick={() => navigate("/courses")}
          >
            Courses
          </span>

          {/* Post */}
          <span
            className="cursor-pointer hover:text-gray-600 hidden lg:inline"
            onClick={() => navigate("/posts")}
          >
            | Post
          </span>
        </div>
      </h1>
    </div>
  );
};

export default ProfileInfo;
