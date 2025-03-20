import React from "react";

const PythonControlStructure = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">โครงสร้างควบคุมใน Python</h1>
      <p>
        โครงสร้างควบคุม (Control Structures) เป็นองค์ประกอบสำคัญของการเขียนโปรแกรมที่ใช้กำหนดลำดับการทำงานของโค้ด
        ใน Python มีโครงสร้างควบคุมที่สำคัญ ได้แก่ คำสั่งเงื่อนไข (Conditional Statements), ลูป (Loops) และการจัดการข้อผิดพลาด (Exception Handling)
      </p>
      
      <h2 className="text-2xl font-semibold mt-4">1. คำสั่งเงื่อนไข (Conditional Statements)</h2>
      <p>
        คำสั่งเงื่อนไขช่วยให้โปรแกรมสามารถตัดสินใจเลือกทำงานตามเงื่อนไขที่กำหนด
      </p>
      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: คำสั่ง If-Else</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`x = 10
if x > 0:
    print("เป็นจำนวนบวก")
elif x == 0:
    print("เป็นศูนย์")
else:
    print("เป็นจำนวนลบ")`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-4">2. ลูป (Looping Structures)</h2>
      <p>
        ลูปใช้ในการทำซ้ำโค้ดหลายครั้งตามเงื่อนไขที่กำหนด
      </p>
      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ลูป For</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`for i in range(5):
    print("รอบที่", i)`}
      </pre>
      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ลูป While</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`x = 0
while x < 5:
    print("ค่าของ x:", x)
    x += 1`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-4">3. การจัดการข้อผิดพลาด (Exception Handling)</h2>
      <p>
        การจัดการข้อผิดพลาดช่วยให้โปรแกรมสามารถรับมือกับข้อผิดพลาดที่เกิดขึ้นระหว่างการทำงานได้อย่างมีประสิทธิภาพ
      </p>
      <h3 className="text-xl font-medium mt-3">ตัวอย่าง: ใช้ Try-Except</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md">
        {`try:
    num = int(input("ป้อนตัวเลข: "))
    print("คุณป้อน:", num)
except ValueError:
    print("ข้อมูลไม่ถูกต้อง! กรุณาป้อนตัวเลขเท่านั้น")`}
      </pre>
    </div>
  );
};

export default PythonControlStructure;
