import React from "react";

const ApiDeploymentScaling = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">Deploying & Scaling APIs</h1>
      <p className="mb-4">
        การนำ API ขึ้นใช้งานจริง (Deploy) และการปรับขนาด (Scaling) เป็นสิ่งสำคัญเมื่อมีผู้ใช้จำนวนมากขึ้น.
      </p>
      <h2 className="text-xl font-semibold mt-4">การ Deploy ด้วย Docker</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`FROM node:14
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "server.js"]`}
      </pre>
    </div>
  );
};

export default ApiDeploymentScaling;
