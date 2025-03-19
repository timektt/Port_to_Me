import React, { createContext, useContext, useState } from "react";

const ThemeContext = createContext();

const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState("light");
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

const ChildComponent = () => {
  const { theme, setTheme } = useContext(ThemeContext);
  return (
    <div className="p-4 border rounded-lg">
      <p>โหมดปัจจุบัน: {theme}</p>
      <button 
        className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition mt-2"
        onClick={() => setTheme(theme === "light" ? "dark" : "light")}
      >
        เปลี่ยนโหมด
      </button>
    </div>
  );
};

const ContextAPI = () => {
  return (
    <ThemeProvider>
      <div className="max-w-3xl mx-auto p-6 shadow-lg rounded-lg border">
        <h1 className="text-2xl font-bold">React Context API</h1>
        <p className="mt-4">
          <strong>Context API</strong> ใช้สำหรับแชร์ข้อมูลระหว่าง Components โดยไม่ต้องส่ง Props ผ่านทุกระดับ
        </p>

        <h2 className="text-xl font-semibold mt-6">📌 ตัวอย่าง</h2>
        <pre className="p-4 rounded-md text-sm overflow-x-auto border bg-gray-200 dark:bg-gray-800">
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

        <div className="mt-6">
          <ChildComponent />
        </div>
      </div>
    </ThemeProvider>
  );
};

export default ContextAPI;
