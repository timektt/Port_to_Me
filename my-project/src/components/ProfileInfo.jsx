import React from "react";

const ProfileInfo = () => {
  return (
    <div className="flex items-center gap-2">
      <img src="/spm2.jpg" alt="Profile" className="w-10 md:w-12 h-auto rounded-full" />
      <h1 className="text-xl md:text-2xl font-bold flex items-center gap-5">
        Supermhee 
        <span className="text-md md:text-lg text-gray-300 hidden md:inline">Course | Post</span>
      </h1>
    </div>
  );
};

export default ProfileInfo;
