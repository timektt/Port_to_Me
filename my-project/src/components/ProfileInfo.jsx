import React from "react";

const ProfileInfo = () => {
  return (
    <div className="flex items-center gap-2">
      <img src="/spm2.jpg" alt="Profile" className="w-30 h-12 rounded-full" />
      <h1 className="text-3xl font-bold flex items-center gap-5">
        Supermhee 
        <span className="text-xl text-gray-300 hidden md:inline">Course | Post</span>
      </h1>
    </div>
  );
};

export default ProfileInfo;
