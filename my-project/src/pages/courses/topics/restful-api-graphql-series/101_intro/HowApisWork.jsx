import React from "react";

const HowApisWork = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">How APIs Work</h1>
      <p className="text-lg">
        APIs work by defining a set of endpoints that clients can use to send requests and receive responses.
        These requests typically follow HTTP methods like GET, POST, PUT, and DELETE.
      </p>
      <p className="mt-4">
        A client makes a request to an API endpoint, and the API responds with the requested data or an appropriate message.
      </p>
    </div>
  );
};

export default HowApisWork;
