import React from "react";

const LoopsForWhile = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">
        🔁 การใช้ลูป for และ while ใน Python
      </h1>

      <p className="mt-2">
        ในภาษา Python การใช้ลูปช่วยให้เราสามารถทำซ้ำคำสั่งหรือโค้ดบล็อกได้หลายครั้ง โดยไม่ต้องเขียนซ้ำ
        ลูปที่นิยมใช้มีสองประเภทหลักคือ <strong>for loop</strong> และ <strong>while loop</strong>
      </p>

      <h2 className="text-xl font-semibold mt-6">1. ลูป for</h2>
      <p className="mt-2">
        ใช้สำหรับการวนลูปที่มีจำนวนรอบแน่นอน เช่น การวนผ่านลิสต์ หรือการใช้งานร่วมกับ <code>range()</code>
      </p>
      <div className="bg-gray-800 text-white p-4 rounded-md mt-2 text-sm overflow-x-auto">
        <pre>
{`# วนลูปผ่านลิสต์
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# วนลูป 5 ครั้งด้วย range
for i in range(5):
    print("รอบที่", i)`}
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">2. ลูป while</h2>
      <p className="mt-2">
        ใช้เมื่อยังไม่รู้จำนวนรอบแน่นอน แต่ให้ลูปทำงานจนกว่าเงื่อนไขจะเป็นเท็จ
      </p>
      <div className="bg-gray-800 text-white p-4 rounded-md mt-2 text-sm overflow-x-auto">
        <pre>
{`x = 0
while x < 5:
    print("ค่าของ x:", x)
    x += 1`}
        </pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">📌 ความแตกต่าง</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>for</strong> ใช้เมื่อต้องการวนลูปตามลำดับของข้อมูล</li>
        <li><strong>while</strong> ใช้เมื่อต้องการให้ลูปทำงานตามเงื่อนไขบางอย่าง</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">⚠️ ข้อควรระวัง</h2>
      <p className="mt-2">
        ต้องระวังไม่ให้ <code>while</code> ลูปทำงานแบบไม่มีที่สิ้นสุด หากลืมอัปเดตค่าที่ใช้ในเงื่อนไข
      </p>
    </div>
  );
};

export default LoopsForWhile;
