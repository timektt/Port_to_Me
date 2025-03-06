import React from "react";

const ProfileInfo = ({ navigate }) => {
  return (
    <div className="flex items-center gap-2">
      <img src="/spm2.jpg" alt="Profile" className="w-30 h-12 rounded-full" />
      <h1 className="text-3xl font-bold flex items-center gap-5 relative -top-1.5"> 
        Supermhee 
        <button 
          onClick={() => navigate("/")} // ✅ คลิกแล้วกลับไป Home
          className="text-xl text-dark hidden md:inline relative top-1 hover:text-white transition"
        >
           Courses
        </button>
        <span className="text-xl text- hidden md:inline relative top-1"> |  Post</span>
      </h1>
    </div>
  );
};

export default ProfileInfo;
