import React from "react";

const CommonProgrammingErrors = () => {
  return (
    <div className="min-h-screen p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">🛠️ ข้อผิดพลาดที่พบบ่อยในการเขียนโปรแกรม</h1>
      <p className="mb-4">
        การเขียนโปรแกรมมักจะเกิดข้อผิดพลาดขึ้นได้บ่อย ไม่ว่าจะเป็นมือใหม่หรือมืออาชีพ การเข้าใจและรู้วิธีแก้ไขข้อผิดพลาดจะช่วยให้พัฒนาโปรแกรมได้รวดเร็วและมีประสิทธิภาพมากยิ่งขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. Syntax Errors (ข้อผิดพลาดทางไวยากรณ์)</h2>
      <p className="mt-2">
        เป็นข้อผิดพลาดที่เกิดจากการเขียนโค้ดไม่ถูกต้องตามไวยากรณ์ของภาษา เช่น ลืมปิดวงเล็บ ลืมเครื่องหมาย : หรือสะกดคำสั่งผิด
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded mt-2 overflow-x-auto">
{`# ❌ Syntax Error
if x > 10
    print("มากกว่า 10")`}      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. Runtime Errors (ข้อผิดพลาดขณะทำงาน)</h2>
      <p className="mt-2">
        เกิดขึ้นระหว่างที่โปรแกรมกำลังทำงาน เช่น การหารด้วยศูนย์ หรือพยายามเข้าถึงตัวแปรที่ไม่มีอยู่
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded mt-2 overflow-x-auto">
{`# ❌ Runtime Error
x = 10 / 0`}      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. Logic Errors (ข้อผิดพลาดทางตรรกะ)</h2>
      <p className="mt-2">
        โปรแกรมสามารถทำงานได้โดยไม่มี error แต่ผลลัพธ์ไม่ถูกต้องตามที่คาดไว้ เช่น การใช้เครื่องหมายเปรียบเทียบผิด
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded mt-2 overflow-x-auto">
{`# ❌ Logic Error
x = 5
y = 10
if x > y:
    print("x มากกว่า y")  # ผลลัพธ์ผิด`}      </pre>

      <h2 className="text-2xl font-semibold mt-6">4. Indentation Errors (ข้อผิดพลาดจากการเยื้องบรรทัด)</h2>
      <p className="mt-2">
        Python ใช้การเยื้องบรรทัด (indentation) แทนวงเล็บ การจัดรูปแบบไม่ถูกต้องจะทำให้เกิด error
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded mt-2 overflow-x-auto">
{`# ❌ Indentation Error
def say_hello():
print("Hello")`}      </pre>

      <h2 className="text-2xl font-semibold mt-6">5. Name Errors (การอ้างถึงตัวแปรที่ไม่ถูกต้อง)</h2>
      <p className="mt-2">
        เกิดเมื่อเรียกใช้ตัวแปรที่ยังไม่ถูกกำหนดหรือสะกดชื่อผิด
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded mt-2 overflow-x-auto">
{`# ❌ Name Error
print(namme)`}      </pre>

      <p className="mt-6">
        ✅ การฝึกหัดอ่าน error message และใช้เครื่องมือช่วย debug จะช่วยให้คุณสามารถหาสาเหตุและแก้ไขข้อผิดพลาดได้อย่างรวดเร็ว
      </p>
    </div>
  );
};

export default CommonProgrammingErrors;
