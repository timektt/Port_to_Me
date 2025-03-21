import React from "react";

const OAuthApiKeys = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4 text-gray-900 dark:text-white">🔑 OAuth & API Keys</h1>

      <p className="mb-4 text-gray-700 dark:text-gray-300">
        <strong>OAuth</strong> เป็นมาตรฐานสำหรับการให้สิทธิ์เข้าถึงทรัพยากร โดยไม่ต้องเปิดเผยรหัสผ่านของผู้ใช้ โดยเฉพาะเมื่อใช้บริการจาก Third-party อย่างเช่น Google, GitHub, Facebook เป็นต้น
      </p>

      <h2 className="text-xl font-semibold mt-6 text-gray-800 dark:text-gray-200">📌 ตัวอย่างการใช้ OAuth กับ Google</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <code>{`const passport = require("passport");
const GoogleStrategy = require("passport-google-oauth20").Strategy;

passport.use(new GoogleStrategy(
  {
    clientID: "GOOGLE_CLIENT_ID",
    clientSecret: "GOOGLE_CLIENT_SECRET",
    callbackURL: "/auth/google/callback"
  },
  (accessToken, refreshToken, profile, done) => {
    // ดำเนินการเก็บข้อมูลผู้ใช้หรือลงทะเบียน
    return done(null, profile);
  }
));`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 text-gray-800 dark:text-gray-200">📌 API Key คืออะไร?</h2>
      <p className="mt-2 text-gray-700 dark:text-gray-300">
        <strong>API Key</strong> คือรหัสที่ใช้ยืนยันตัวตนของแอปพลิเคชันเพื่อเข้าถึง API โดยไม่ผูกกับผู้ใช้
        มักใช้กับ Public API เช่น Google Maps, OpenWeather เป็นต้น
      </p>

      <h2 className="text-xl font-semibold mt-6 text-gray-800 dark:text-gray-200">🔍 เปรียบเทียบ OAuth กับ API Key</h2>
      <table className="w-full border-collapse border border-gray-700 mt-2 text-sm">
        <thead>
          <tr className="bg-gray-700 text-white">
            <th className="p-2 border">Feature</th>
            <th className="p-2 border">OAuth</th>
            <th className="p-2 border">API Key</th>
          </tr>
        </thead>
        <tbody className="text-center">
          <tr className="bg-gray-800 text-white">
            <td className="border p-2">ระดับความปลอดภัย</td>
            <td className="border p-2">สูง</td>
            <td className="border p-2">ต่ำ</td>
          </tr>
          <tr>
            <td className="border p-2">ระบุผู้ใช้</td>
            <td className="border p-2">✅</td>
            <td className="border p-2">❌</td>
          </tr>
          <tr className="bg-gray-800 text-white">
            <td className="border p-2">เหมาะกับ</td>
            <td className="border p-2">Login, OAuth Apps</td>
            <td className="border p-2">Public APIs</td>
          </tr>
        </tbody>
      </table>

      <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-200 rounded-lg">
        💡 <strong>Tip:</strong> เก็บ API Key และ OAuth Credentials ไว้ใน <code>.env</code> file และอย่า push ขึ้น GitHub!
      </div>
    </div>
  );
};

export default OAuthApiKeys;
