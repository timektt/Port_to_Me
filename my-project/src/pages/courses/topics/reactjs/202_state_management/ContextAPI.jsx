import React, { createContext, useContext, useState } from "react";

// สร้าง Context
const ThemeContext = createContext();

// Provider สำหรับแชร์ Theme
const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState("light");
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Component ลูกที่ใช้ Context
const ChildComponent = () => {
  const { theme, setTheme } = useContext(ThemeContext);
  const isDark = theme === "dark";

  return (
    <div className={`p-6 border rounded-lg shadow-md transition-all duration-300 ${isDark ? "bg-gray-800 text-white" : "bg-white text-gray-900"}`}>
      <p className="text-lg font-medium">โหมดปัจจุบัน: <span className="font-bold">{theme}</span></p>
      <button
        className="mt-4 bg-green-600 hover:bg-green-700 text-white font-semibold px-5 py-2 rounded-lg"
        onClick={() => setTheme(isDark ? "light" : "dark")}
      >
        เปลี่ยนโหมดเป็น {isDark ? "Light" : "Dark"}
      </button>
    </div>
  );
};

// Component หลักที่ใช้ ThemeProvider
const ContextAPI = () => {
  return (
    <ThemeProvider>
      <div className="max-w-3xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-4">React Context API</h1>
        <p>
          Context API ช่วยให้สามารถแชร์ข้อมูลระหว่าง Components โดยไม่ต้องส่งผ่าน Props ทุกระดับ
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">ตัวอย่าง Theme Provider</h2>
        <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
{`const ThemeContext = createContext();

const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState("light");
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};`}
        </pre>

        <h2 className="text-2xl font-semibold mt-6 mb-2">ใช้ useContext เพื่อเข้าถึง Theme</h2>
        <pre className="bg-gray-800 text-white p-4 rounded-md text-sm overflow-x-auto">
{`const { theme, setTheme } = useContext(ThemeContext);

return (
  <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
    เปลี่ยนโหมด
  </button>
);`}
        </pre>

        <div className="mt-8">
          <ChildComponent />
        </div>
      </div>
    </ThemeProvider>
  );
};

export default ContextAPI;
