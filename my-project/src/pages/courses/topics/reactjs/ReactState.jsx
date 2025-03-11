import React, { useState } from "react";

const ReactState = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Managing State in React</h1>
      <p>State allows React components to store and update data dynamically.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`const Toggle = () => {
  const [isOn, setIsOn] = useState(false);
  return <button onClick={() => setIsOn(!isOn)}>{isOn ? "ON" : "OFF"}</button>;
};

export default Toggle;`}
      </pre>
    </div>
  );
};

export default ReactState;
