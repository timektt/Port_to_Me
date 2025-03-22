import React from "react";

const Constants = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">🔒 ค่าคงที่ (Constants) และข้อมูลที่ไม่เปลี่ยนแปลง</h1>

      <p className="mt-4">
        ค่าคงที่ (Constants) หมายถึงค่าที่ไม่เปลี่ยนแปลงตลอดอายุการทำงานของโปรแกรม โดยปกติแล้วในบางภาษาเช่น C หรือ Java จะมีคีย์เวิร์ดอย่าง
        <code className="bg-gray-200 px-1 mx-1 rounded text-sm">const</code> หรือ <code className="bg-gray-200 px-1 mx-1 rounded text-sm">final</code> เพื่อกำหนดค่าคงที่
        แต่ใน Python แม้จะไม่มีคีย์เวิร์ดเฉพาะสำหรับ constants แต่เราก็สามารถใช้แนวทางปฏิบัติ (convention) แทนได้
      </p>

      <h2 className="text-xl font-semibold mt-6">📌 แนวทางการตั้งค่าคงที่ใน Python</h2>
      <p className="mt-2">
        ใน Python เราจะใช้ตัวพิมพ์ใหญ่ทั้งหมดเพื่อสื่อว่าเป็นค่าคงที่ และมักจะประกาศไว้ตอนต้นไฟล์ของโปรแกรม เช่น:
      </p>

      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm">
        <pre>{`PI = 3.14159
GRAVITY = 9.8
MAX_CONNECTIONS = 100`}</pre>
      </div>

      <p className="mt-4">
        โดย convention แล้ว ตัวแปรเหล่านี้ไม่ควรถูกเปลี่ยนค่าในโปรแกรม แม้ว่าทางเทคนิคจะสามารถเปลี่ยนค่าได้
      </p>

      <h2 className="text-xl font-semibold mt-6">🔐 ข้อมูลที่ไม่เปลี่ยนแปลง (Immutable)</h2>
      <p className="mt-2">
        ค่าที่ไม่สามารถเปลี่ยนแปลงได้หลังจากที่ถูกสร้างเรียกว่า <strong>Immutable</strong> เช่น:
      </p>
      <ul className="list-disc ml-6 mt-2">
        <li><strong>int</strong>, <strong>float</strong>, <strong>str</strong></li>
        <li><strong>tuple</strong>, <strong>frozenset</strong></li>
      </ul>

      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm">
        <pre>{`x = 10
x = 20  # ได้ (สร้าง int ตัวใหม่แทนของเดิม)

t = (1, 2, 3)
t[0] = 5  # ❌ Error - Tuple เป็น immutable`}</pre>
      </div>

      <h2 className="text-xl font-semibold mt-6">🧠 ทำไมค่าคงที่จึงสำคัญ?</h2>
      <ul className="list-disc ml-6 mt-2">
        <li>เพิ่มความปลอดภัยในโปรแกรม ลดโอกาสเกิดบั๊ก</li>
        <li>ทำให้โค้ดอ่านง่าย และแยกการตั้งค่าจาก logic ได้ดีขึ้น</li>
        <li>สามารถใช้กับการคอนฟิก เช่น API_KEY, BASE_URL ฯลฯ</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📦 ตัวอย่างการใช้ค่าคงที่ในโปรเจค</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-4 text-sm">
        <pre>{`# constants.py
PI = 3.14
SITE_NAME = "MyWebsite"

# main.py
from constants import PI
print(f"ค่าของ PI คือ: {PI}")`}</pre>
      </div>
    </div>
  );
};

export default Constants;