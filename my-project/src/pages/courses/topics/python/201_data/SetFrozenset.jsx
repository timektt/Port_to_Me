import React from "react";

const SetFrozenset = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        Set & Frozenset ใน Python
      </h1>

      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        <strong>Set</strong> เป็นคอลเลกชันที่ไม่มีลำดับแน่นอนและไม่สามารถมีค่าซ้ำกันได้
      </p>
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        <strong>Frozenset</strong> เป็นเวอร์ชันที่ไม่สามารถแก้ไขได้ของ Set (Immutable)
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การสร้าง Set และ Frozenset</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`my_set = {1, 2, 3}
my_frozenset = frozenset([1, 2, 3])
print(my_set)         # Output: {1, 2, 3}
print(my_frozenset)   # Output: frozenset({1, 2, 3})`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การเพิ่มและลบค่าจาก Set</h2>
      <p className="mt-2">สามารถเพิ่มหรือลบค่าใน Set ได้ แต่ Frozenset ไม่สามารถเปลี่ยนแปลงได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`my_set.add(4)       # เพิ่มค่าเข้าไป
my_set.discard(2)    # ลบค่าออก (ถ้ามี)
print(my_set)        # Output: {1, 3, 4}

# my_frozenset.add(4) ❌ จะเกิด Error เพราะแก้ไขค่าไม่ได้`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การดำเนินการทางเซต</h2>
      <p className="mt-2">Set และ Frozenset รองรับการรวมเซต ตัดกัน และหาค่าที่ต่างกัน</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`a = {1, 2, 3}
b = frozenset([2, 3, 4])

print(a | b)  # Union: {1, 2, 3, 4}
print(a & b)  # Intersection: {2, 3}
print(a - b)  # Difference: {1}`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. ตรวจสอบสมาชิกในเซต</h2>
      <p className="mt-2">สามารถตรวจสอบค่าด้วย `in` และใช้เปรียบเทียบ Subset/Superset ได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`s = {1, 2, 3}
print(2 in s)            # Output: True
print({1, 2}.issubset(s))  # Output: True
print(s.issuperset({2}))  # Output: True`}
        </pre>
      </div>
    </div>
  );
};

export default SetFrozenset;
