import React from "react";

const ProfileInfo = () => {
  return (
    <div className="flex items-center gap-3">
      <img src="/spm.png" alt="Profile" className="w-40 h-15 rounded-full" />
      <h1 className="text-2xl font-bold">
        Supermhee <span className="text-xl font-normal text-gray-300">| Your Course | Your Post</span>
      </h1>
    </div>
  );
};

export default ProfileInfo;
