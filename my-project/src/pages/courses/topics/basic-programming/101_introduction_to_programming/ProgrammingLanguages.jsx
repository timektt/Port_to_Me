import React from "react";

const ProgrammingLanguages = () => {
  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">ประเภทของภาษาโปรแกรม (Types of Programming Languages)</h1>
      <p className="text-base sm:text-lg leading-relaxed">
        ภาษาโปรแกรมคือเครื่องมือที่มนุษย์ใช้ในการสื่อสารกับคอมพิวเตอร์ โดยมีหลากหลายประเภทที่ออกแบบมาให้เหมาะกับงานที่แตกต่างกันออกไป ซึ่งสามารถแบ่งได้เป็นหลายกลุ่มตามลักษณะการทำงานของภาษา เช่น
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">1. ภาษาในระดับสูง (High-Level Languages)</h2>
      <p>
        เป็นภาษาที่เข้าใจง่าย ใกล้เคียงภาษามนุษย์ เช่น Python, Java, C#, JavaScript ใช้งานง่าย มีไลบรารีช่วยเหลือมากมาย เหมาะสำหรับผู้เริ่มต้นและงานพัฒนาทั่วไป
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-3 overflow-x-auto text-sm">
{`print("Hello, World!")  # ตัวอย่างภาษา Python`}
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">2. ภาษาในระดับต่ำ (Low-Level Languages)</h2>
      <p>
        เช่น Assembly หรือภาษาเครื่อง (Machine Code) ซึ่งใกล้เคียงกับการทำงานของฮาร์ดแวร์จริง ๆ ใช้สำหรับงานที่ต้องการประสิทธิภาพสูง เช่น ระบบปฏิบัติการหรือ Embedded Systems
      </p>
      <pre className="bg-gray-800 text-white p-4 rounded-lg mt-3 overflow-x-auto text-sm">
{`; Assembly
MOV AX, 01
MOV BX, 02
ADD AX, BX`}
      </pre>

      <h2 className="text-xl font-semibold mt-6 mb-2">3. ภาษาเชิงวัตถุ (Object-Oriented Languages)</h2>
      <p>
        เช่น Java, C++, Python (รองรับ) เหมาะสำหรับการพัฒนาซอฟต์แวร์ขนาดใหญ่ โดยเน้นการจัดกลุ่มของข้อมูลและฟังก์ชันไว้ในรูปของ &quot;อ็อบเจกต์&quot;
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">4. ภาษาเชิงฟังก์ชัน (Functional Languages)</h2>
      <p>
        เช่น Haskell, Lisp หรือบางฟีเจอร์ใน JavaScript ที่เน้นการทำงานแบบไม่มีสถานะ (stateless) และหลีกเลี่ยงผลข้างเคียง (side effects)
      </p>

      <h2 className="text-xl font-semibold mt-6 mb-2">5. ภาษาเชิงสคริปต์ (Scripting Languages)</h2>
      <p>
        ใช้สำหรับเขียนสคริปต์เล็ก ๆ เพื่อควบคุมการทำงาน เช่น Bash, Python, JavaScript ในการทำเว็บไซต์
      </p>

      <div className="mt-6 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 p-4 rounded-lg">
        💡 <strong>สรุป:</strong> การเลือกใช้ภาษาโปรแกรมควรพิจารณาจากเป้าหมายของโปรเจกต์ ความถนัด และทรัพยากรที่มีอยู่ เช่น สำหรับเว็บมักใช้ JavaScript / สำหรับ Data Science มักใช้ Python
      </div>
    </div>
  );
};

export default ProgrammingLanguages;