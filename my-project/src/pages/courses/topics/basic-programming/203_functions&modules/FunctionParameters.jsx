import React from "react";

const FunctionParameters = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">
        🧩 พารามิเตอร์และอาร์กิวเมนต์ในฟังก์ชัน (Function Parameters & Arguments)
      </h1>

      <p className="mb-4">
        ในการเขียนฟังก์ชันในภาษา Python เราสามารถกำหนดพารามิเตอร์เพื่อรับค่าจากภายนอก และส่งค่าเหล่านั้นเข้ามาใช้งานภายในฟังก์ชันได้ โดยค่าที่ส่งเข้ามาเรียกว่า <strong>arguments</strong>
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">1. พารามิเตอร์ (Parameters)</h2>
      <p>
        คือ ตัวแปรที่ระบุไว้ตอนนิยามฟังก์ชัน เพื่อรับค่าในภายหลังจาก arguments
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        <code>{`def greet(name):
    print(f"สวัสดี {name}")

greet("ซาร่า")  # ส่งค่า 'ซาร่า' เป็น argument ไปยัง parameter 'name'`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">2. อาร์กิวเมนต์ (Arguments)</h2>
      <p>
        คือ ค่าที่ส่งเข้าไปตอนเรียกใช้ฟังก์ชัน ซึ่งจะถูกจับคู่กับพารามิเตอร์ที่ระบุไว้
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">3. พารามิเตอร์หลายตัว</h2>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        <code>{`def add(a, b):
    return a + b

print(add(3, 4))  # Output: 7`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">4. Default Parameter</h2>
      <p>สามารถกำหนดค่าเริ่มต้นให้พารามิเตอร์ได้</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        <code>{`def greet(name="ผู้ใช้"):
    print(f"สวัสดี {name}")

greet()          # สวัสดี ผู้ใช้
greet("บ็อบ")    # สวัสดี บ็อบ`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">5. Keyword Arguments</h2>
      <p>สามารถส่ง arguments โดยระบุชื่อพารามิเตอร์</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        <code>{`def introduce(name, age):
    print(f"{name} อายุ {age} ปี")

introduce(age=25, name="ก้อง")`}</code>
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">6. *args และ **kwargs</h2>
      <p>
        ใช้เมื่อต้องการส่งค่าหลายค่าเข้าไปแบบไม่จำกัดจำนวน
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2">
        <code>{`def show_info(*args, **kwargs):
    for arg in args:
        print("arg:", arg)
    for key, value in kwargs.items():
        print(f"{key} = {value}")

show_info("ข้อมูล1", "ข้อมูล2", name="พีท", age=22)`}</code>
      </pre>
    </div>
  );
};

export default FunctionParameters;
