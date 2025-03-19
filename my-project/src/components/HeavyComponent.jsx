import React from "react";

const HeavyComponent = () => {
  return (
    <div className="p-6 border rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold">Heavy Component</h2>
      <p className="mt-2">นี่คือตัวอย่างของ Component ที่โหลดแบบ Lazy Loading</p>
    </div>
  );
};

export default HeavyComponent;
