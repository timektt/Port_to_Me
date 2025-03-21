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
        สามารถเพิ่ม ลบ แก้ไข และเข้าถึงข้อมูลได้อย่างยืดหยุ่น
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การสร้างดิกชันนารี</h2>
      <p className="mt-2">สามารถสร้างโดยใช้เครื่องหมาย <code>{`{}`}</code> และกำหนด key-value</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`my_dict = {"name": "John", "age": 25}
print(my_dict["name"])  # Output: John`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การเพิ่มและแก้ไขค่า</h2>
      <p className="mt-2">เพิ่ม key ใหม่ หรือแก้ไขค่าที่มีอยู่ได้โดยตรง</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`my_dict["city"] = "Bangkok"
my_dict["age"] = 30
print(my_dict)`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การลบค่าภายในดิกชันนารี</h2>
      <p className="mt-2">ใช้ <code>del</code> หรือ <code>.pop()</code> เพื่อลบ key ออก</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`del my_dict["age"]
# หรือ
# my_dict.pop("age")
print(my_dict)`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การตรวจสอบว่า key มีอยู่หรือไม่</h2>
      <p className="mt-2">ใช้ <code>in</code> เพื่อตรวจสอบว่ามี key หรือไม่</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`if "name" in my_dict:
    print("พบ key 'name'")`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">5. การเข้าถึงค่าด้วย .get()</h2>
      <p className="mt-2">ช่วยให้เข้าถึงค่าได้ปลอดภัย หาก key ไม่พบจะไม่เกิด error</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`print(my_dict.get("name"))     # Output: John
print(my_dict.get("gender"))   # Output: None`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">6. การวนลูปผ่านดิกชันนารี</h2>
      <p className="mt-2">ใช้ <code>.items()</code> เพื่อดึงทั้ง key และ value</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`for key, value in my_dict.items():
    print(f"{key}: {value}")`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">7. ดิกชันนารีซ้อน (Nested Dictionary)</h2>
      <p className="mt-2">สามารถซ้อนดิกชันนารีภายในอีกดิกชันนารีได้</p>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4">
        <pre>
{`students = {
  "001": {"name": "Alice", "age": 20},
  "002": {"name": "Bob", "age": 22}
}

print(students["001"]["name"])  # Output: Alice`}
        </pre>
      </div>
    </div>
  );
};

export default Dictionaries;
