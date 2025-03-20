import React from "react";

const NodeSetup = () => {
  return (
    <div className="p-4 sm:p-6 md:p-8 max-w-screen-lg mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">🚀 ติดตั้ง Node.js</h1>
      <p className="mt-4">
        เราสามารถติดตั้ง Node.js ได้จากเว็บไซต์ทางการ:
        <a href="https://nodejs.org" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline ml-1">
          nodejs.org
        </a>
      </p>
      
      <h2 className="text-xl font-semibold mt-6">📌 1. ติดตั้งบน Windows</h2>
      <p className="mt-2">ดาวน์โหลดและติดตั้งไฟล์ `.msi` จากเว็บไซต์ Node.js จากนั้นติดตั้งตามขั้นตอน</p>
      
      <h2 className="text-xl font-semibold mt-6">📌 2. ติดตั้งบน macOS และ Linux</h2>
      <p className="mt-2">ใช้คำสั่งต่อไปนี้เพื่อติดตั้งผ่าน Homebrew หรือ apt:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`# ติดตั้งผ่าน Homebrew (macOS)
brew install node

# ติดตั้งผ่าน apt (Linux)
sudo apt update
sudo apt install nodejs npm`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">🔹 ตรวจสอบเวอร์ชัน Node.js</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`node -v`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6">📌 3. ใช้ Node Version Manager (NVM)</h2>
      <p className="mt-2">NVM ช่วยให้สามารถสลับเวอร์ชันของ Node.js ได้ง่าย</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        {`# ติดตั้ง NVM
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash

# ติดตั้ง Node.js เวอร์ชันล่าสุด
nvm install node

# ตรวจสอบเวอร์ชัน
nvm list`}
      </pre>
    </div>
  );
};

export default NodeSetup;
