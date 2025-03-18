import React from "react";

const OAuthApiKeys = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">OAuth & API Keys</h1>
      <p className="mb-4">
        <strong>OAuth</strong> เป็นมาตรฐานสำหรับการให้สิทธิ์การเข้าถึง API โดยไม่ต้องใช้รหัสผ่านของผู้ใช้โดยตรง.
      </p>
      <h2 className="text-xl font-semibold mt-4">ตัวอย่างการใช้ OAuth กับ Google</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`const passport = require("passport");
const GoogleStrategy = require("passport-google-oauth20").Strategy;

passport.use(new GoogleStrategy({
  clientID: "GOOGLE_CLIENT_ID",
  clientSecret: "GOOGLE_CLIENT_SECRET",
  callbackURL: "/auth/google/callback"
}, (token, tokenSecret, profile, done) => {
  return done(null, profile);
}));`}
      </pre>
    </div>
  );
};

export default OAuthApiKeys;
