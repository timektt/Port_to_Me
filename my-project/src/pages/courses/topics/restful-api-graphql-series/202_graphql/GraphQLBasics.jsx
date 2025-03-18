import React from "react";

const GraphQLBasics = () => {
  return (
    <div className="max-w-4xl mx-auto p-6 text-left">
      <h1 className="text-3xl font-bold mb-4">GraphQL Basics</h1>
      <p className="mb-4">
        GraphQL เป็นภาษาสำหรับการ Query ข้อมูลที่พัฒนาโดย Facebook ซึ่งช่วยให้การดึงข้อมูลจาก API มีประสิทธิภาพและยืดหยุ่นมากขึ้น
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`{
  user(id: "1") {
    name
    email
  }
}`}
      </pre>
      <p className="mt-4">GraphQL ทำให้สามารถเลือกข้อมูลที่ต้องการดึงมาได้โดยไม่ต้องโหลดข้อมูลที่ไม่จำเป็น</p>
    </div>
  );
};

export default GraphQLBasics;
