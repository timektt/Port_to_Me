import React from "react";

const ApiDocumentation = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">API Documentation & Tools</h1>
      <p className="text-lg">
        API documentation is essential for developers to understand how to use an API effectively.
        Good documentation includes clear instructions, example requests, and error handling guidelines.
      </p>
      <p className="mt-4">
        Popular tools for API documentation include:
      </p>
      <ul className="list-disc ml-6 mt-4 space-y-2">
        <li><strong>Swagger:</strong> Used for designing, building, and documenting APIs.</li>
        <li><strong>Postman:</strong> Helps developers test API endpoints easily.</li>
        <li><strong>OpenAPI Specification:</strong> A standard for API documentation.</li>
      </ul>
    </div>
  );
};

export default ApiDocumentation;
