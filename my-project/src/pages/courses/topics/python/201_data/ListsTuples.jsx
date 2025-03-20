import React from "react";

const ListsTuples = () => {
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold">Lists & Tuples ใน Python</h1>
      <p className="mt-4">
        Lists และ Tuples เป็นโครงสร้างข้อมูลพื้นฐานที่ใช้เก็บค่าหลายค่าภายในตัวแปรเดียวใน Python
      </p>
      
      <h2 className="text-2xl font-semibold mt-6">1. ความแตกต่างระหว่าง Lists และ Tuples</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Lists:</strong> เปลี่ยนแปลงค่าได้ (Mutable), มีลำดับแน่นอน และสามารถเพิ่ม, ลบ หรือแก้ไขค่าได้</li>
        <li><strong>Tuples:</strong> เปลี่ยนแปลงค่าไม่ได้ (Immutable), มีลำดับแน่นอน และมีประสิทธิภาพสูงกว่า Lists เมื่อใช้กับข้อมูลที่ไม่ต้องเปลี่ยนค่า</li>
      </ul>
      
      <h2 className="text-2xl font-semibold mt-6">2. ตัวอย่างการใช้งาน Lists และ Tuples</h2>
      <p className="mt-4">การสร้าง Lists และ Tuples:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`my_list = [1, 2, 3, 4]
my_tuple = (1, 2, 3, 4)
print(my_list)
print(my_tuple)`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">3. การเข้าถึงค่าภายใน Lists และ Tuples</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`print(my_list[0])  # Output: 1
print(my_tuple[1])  # Output: 2`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">4. การแก้ไข Lists และข้อจำกัดของ Tuples</h2>
      <p className="mt-4">Lists สามารถเปลี่ยนแปลงค่าได้ แต่ Tuples ไม่สามารถเปลี่ยนแปลงได้:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`my_list[0] = 10  # เปลี่ยนค่าใน List ได้
print(my_list)  # Output: [10, 2, 3, 4]

my_tuple[0] = 10  # ❌ Error! ไม่สามารถเปลี่ยนค่าใน Tuple ได้`}
      </pre>
      
      <h2 className="text-2xl font-semibold mt-6">5. การวนลูปผ่าน Lists และ Tuples</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg">
        {`for item in my_list:
    print(item)  # แสดงค่าทั้งหมดใน List

for item in my_tuple:
    print(item)  # แสดงค่าทั้งหมดใน Tuple`}
      </pre>
    </div>
  );
};

export default ListsTuples;