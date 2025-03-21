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
        <li><strong>Lists:</strong> เปลี่ยนแปลงค่าได้ (Mutable)</li>
        <li><strong>Tuples:</strong> เปลี่ยนแปลงค่าไม่ได้ (Immutable)</li>
      </ul>

      <h2 className="text-2xl font-semibold mt-6">2. ตัวอย่างการสร้าง List และ Tuple</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm w-full">
{`my_list = [1, 2, 3, 4]
my_tuple = (1, 2, 3, 4)

print(my_list)
print(my_tuple)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. การเข้าถึงข้อมูล</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm w-full">
{`print(my_list[0])   # 1
print(my_tuple[2])  # 3`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">4. การเปลี่ยนค่า (List เท่านั้น)</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm w-full">
{`my_list[1] = 10
print(my_list)  # [1, 10, 3, 4]

my_tuple[1] = 10  # ❌ Error: Tuples are immutable`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">5. การวนลูป</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm w-full">
{`for item in my_list:
    print(item)

for item in my_tuple:
    print(item)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">6. เมธอดที่ใช้กับ List</h2>
      <p className="mt-2">List มีเมธอดหลายแบบ เช่น append(), remove(), sort()</p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm w-full">
{`my_list.append(5)
my_list.remove(2)
my_list.sort()
print(my_list)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">7. การแปลงระหว่าง List และ Tuple</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto text-sm w-full">
{`my_tuple = tuple(my_list)
my_list = list(my_tuple)`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">8. กรณีการใช้งาน</h2>
      <ul className="list-disc ml-5 mt-2">
        <li>ใช้ <strong>List</strong> เมื่อข้อมูลต้องมีการเปลี่ยนแปลง</li>
        <li>ใช้ <strong>Tuple</strong> เมื่อข้อมูลเป็นค่าคงที่ หรือเพื่อประสิทธิภาพที่ดีกว่า</li>
      </ul>
    </div>
  );
};

export default ListsTuples;
