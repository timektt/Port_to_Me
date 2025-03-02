import React from "react";
import Navbar from "./components/Navbar";

function App() {
  return (
    <div className="bg-gray-900 min-h-screen text-white">
      <Navbar />
      <main className="p-4">
        <h1 className="text-2xl">Welcome to My Website</h1>
      </main>
    </div>
  );
}

export default App;
