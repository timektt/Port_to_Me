import React from "react";
import ProfileInfo from "./ProfileInfo";
import SocialLinks from "./SocialLinks";

const Navbar = () => {
  return (
    <nav className="bg-gray-800 text-white p-4  h-20 flex justify-between items-center">
      <ProfileInfo />
      <SocialLinks />
    </nav>
  );
};

export default Navbar;
