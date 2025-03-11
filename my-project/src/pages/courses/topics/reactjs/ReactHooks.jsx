import React, { useState } from "react";

const ReactHooks = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">React Hooks</h1>
      <p>React hooks allow function components to use state and effects.</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`const Counter = () => {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>Count: {count}</button>;
};

export default Counter;`}
      </pre>
    </div>
  );
};

export default ReactHooks;
