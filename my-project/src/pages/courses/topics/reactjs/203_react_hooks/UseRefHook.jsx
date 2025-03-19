import React, { useRef } from "react";

const UseRefHook = () => {
  const inputRef = useRef(null);

  const focusInput = () => {
    inputRef.current.focus();
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">useRef & Manipulating DOM</h1>
      <input ref={inputRef} className="p-2 border rounded mt-4 block text-black" placeholder="พิมพ์อะไรสักอย่าง..." />
      <button className="mt-4 p-2 border rounded bg-green-500 text-gray-800" onClick={focusInput}>
        โฟกัส Input
      </button>
    </div>
  );
};

export default UseRefHook;