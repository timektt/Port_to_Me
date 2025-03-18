import React from "react";

const ApiTypes = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">Types of APIs</h1>
      <p className="text-lg">
        APIs can be classified into different types based on their accessibility and usage:
      </p>
      <ul className="list-disc ml-6 mt-4 space-y-2">
        <li><strong>Public APIs:</strong> Open to external developers and users.</li>
        <li><strong>Private APIs:</strong> Used within an organization for internal communication.</li>
        <li><strong>Partner APIs:</strong> Shared between business partners for collaboration.</li>
        <li><strong>Composite APIs:</strong> Combine multiple API calls into a single request.</li>
      </ul>
    </div>
  );
};

export default ApiTypes;
