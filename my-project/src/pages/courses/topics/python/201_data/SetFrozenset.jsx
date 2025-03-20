import React from "react";

const SetFrozenset = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        Set & Frozenset ใน Python
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        <strong>Set</strong> เป็นคอลเลกชันที่ไม่มีลำดับแน่นอนและไม่สามารถมีค่าที่ซ้ำกันได้
      </p>
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        <strong>Frozenset</strong> เป็น Set ที่ไม่สามารถเปลี่ยนแปลงค่าได้ (Immutable)
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การสร้าง Set และ Frozenset</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`my_set = {1, 2, 3}
my_frozenset = frozenset([1, 2, 3])
print(my_set)
print(my_frozenset)`}
        </pre>
      </div>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การเพิ่มและลบค่าจาก Set</h2>
      <p className="mt-2">สามารถเพิ่มหรือลบค่าจาก Set ได้ แต่ Frozenset ไม่สามารถเปลี่ยนแปลงได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`my_set.add(4)  # เพิ่มค่าเข้าไปใน Set
my_set.remove(2)  # ลบค่าออกจาก Set
print(my_set)

# my_frozenset.add(4) ❌ จะเกิด Error เพราะ Frozenset เปลี่ยนค่าไม่ได้`}
        </pre>
      </div>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การดำเนินการทางเซต</h2>
      <p className="mt-2">Set รองรับการดำเนินการทางเซต เช่น Union, Intersection, Difference</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`set_a = {1, 2, 3}
set_b = {3, 4, 5}

print(set_a | set_b)  # Union: {1, 2, 3, 4, 5}
print(set_a & set_b)  # Intersection: {3}
print(set_a - set_b)  # Difference: {1, 2}`}
        </pre>
      </div>
    </div>
  );
};

export default SetFrozenset;
