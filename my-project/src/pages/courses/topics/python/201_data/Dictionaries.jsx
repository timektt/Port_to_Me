import React from "react";

const Dictionaries = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        ดิกชันนารี (Dictionaries) ใน Python
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        ดิกชันนารี (Dictionary) เป็นโครงสร้างข้อมูลแบบ Key-Value ที่ใช้จัดเก็บข้อมูลใน Python
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การสร้างดิกชันนารี</h2>
      <p className="mt-2">เราสามารถสร้างดิกชันนารีโดยใช้เครื่องหมาย `{}` และกำหนด key-value pairs</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`my_dict = {"name": "John", "age": 25}
print(my_dict["name"])  # Output: John`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การเพิ่มและแก้ไขค่าภายในดิกชันนารี</h2>
      <p className="mt-2">สามารถเพิ่ม key ใหม่ หรือเปลี่ยนค่าของ key ที่มีอยู่ได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`my_dict["city"] = "Bangkok"
my_dict["age"] = 30
print(my_dict)  # Output: {'name': 'John', 'age': 30, 'city': 'Bangkok'}`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การลบค่าในดิกชันนารี</h2>
      <p className="mt-2">สามารถใช้ `del` เพื่อลบค่าออกจากดิกชันนารี</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`del my_dict["age"]
print(my_dict)  # Output: {'name': 'John', 'city': 'Bangkok'}`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การวนลูปผ่านดิกชันนารี</h2>
      <p className="mt-2">สามารถใช้ `for` loop เพื่อวนลูปผ่าน key และ value</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`for key, value in my_dict.items():
    print(f"{key}: {value}")`}
        </pre>
      </div>
    </div>
  );
};

export default Dictionaries;
