import React from "react";

const ErrorTypes = () => {
  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 md:px-8 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">🛠️ ประเภทของข้อผิดพลาด (Types of Errors)</h1>

      <p className="mb-4">
        ในการเขียนโปรแกรม ข้อผิดพลาดเป็นสิ่งที่เลี่ยงไม่ได้ นักพัฒนาควรเข้าใจประเภทของข้อผิดพลาดต่าง ๆ เพื่อสามารถตรวจจับ แก้ไข และป้องกันได้อย่างมีประสิทธิภาพ โดยทั่วไปแบ่งออกเป็น 3 ประเภทหลัก ๆ:
      </p>

      <h2 className="text-2xl font-semibold mt-6">1. ข้อผิดพลาดทางไวยากรณ์ (Syntax Errors)</h2>
      <p className="mt-2">
        ข้อผิดพลาดที่เกิดจากการเขียนโค้ดไม่ถูกต้องตามกฎของภาษา เช่น ลืมวงเล็บ ปิดเครื่องหมายคำพูดไม่ครบ หรือใช้คีย์เวิร์ดผิด
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2 text-sm">
{`# SyntaxError: expected ':'
if x > 5
    print("x มากกว่า 5")`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">2. ข้อผิดพลาดขณะรันโปรแกรม (Runtime Errors)</h2>
      <p className="mt-2">
        เป็นข้อผิดพลาดที่เกิดขึ้นระหว่างการทำงานของโปรแกรม แม้โค้ดจะเขียนถูกต้องตามไวยากรณ์ แต่โปรแกรมเกิดปัญหา เช่น หารด้วยศูนย์ หรือเปิดไฟล์ที่ไม่มีอยู่
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2 text-sm">
{`# ZeroDivisionError
x = 10 / 0

# FileNotFoundError
with open("notfound.txt") as f:
    content = f.read()`}
      </pre>

      <h2 className="text-2xl font-semibold mt-6">3. ข้อผิดพลาดเชิงตรรกะ (Logic Errors)</h2>
      <p className="mt-2">
        ข้อผิดพลาดที่โค้ดรันได้ตามปกติแต่ผลลัพธ์ไม่ถูกต้องตามที่คาดหวัง เช่น การใช้เครื่องหมายผิด หรือสูตรผิด ซึ่งยากที่สุดในการตรวจจับ
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto mt-2 text-sm">
{`# ตั้งใจจะตรวจสอบว่า x เป็นเลขคู่
x = 7
if x % 2 == 1:
    print("x เป็นเลขคู่")  # ❌ จริง ๆ แล้ว x เป็นเลขคี่`}
      </pre>

      <div className="mt-6 p-4 bg-yellow-100 text-yellow-800 rounded-lg dark:bg-yellow-900 dark:text-yellow-200">
        💡 <strong>Tip:</strong> การใช้เครื่องมือ debug และการเขียน test cases จะช่วยในการค้นหา logic error ได้ดีมาก
      </div>
    </div>
  );
};

export default ErrorTypes;
