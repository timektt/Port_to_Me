import React from "react";

const PythonLeetcode = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      {/* ✅ Title */}
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        แบบฝึกหัด Leetcode ด้วย Python
      </h1>

      {/* ✅ Description */}
      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        มาทดสอบความสามารถของคุณด้วยโจทย์โปรแกรมมิ่งพื้นฐาน!
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ปัญหา: Two Sum</h2>
      <p className="mt-2">
        ให้ลิสต์ของตัวเลข และค่าหนึ่งค่า ให้หาตำแหน่งของตัวเลขสองตัวที่เมื่อรวมกันแล้วได้ค่าที่กำหนด
      </p>
      
      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

nums = [2, 7, 11, 15]
target = 9
print(twoSum(nums, target))  # Output: [0, 1]`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. ปัญหา: Reverse a String</h2>
      <p className="mt-2">
        จงเขียนฟังก์ชันที่รับสตริง และคืนค่าสตริงที่ถูกกลับด้าน
      </p>
      
      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`def reverseString(s):
    return s[::-1]

print(reverseString("hello"))  # Output: "olleh"`}
        </pre>
      </div>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. ปัญหา: FizzBuzz</h2>
      <p className="mt-2">
        ให้เขียนโปรแกรมที่แสดงตัวเลข 1 ถึง n แต่ถ้าตัวเลขหาร 3 ลงตัวให้แสดง "Fizz" ถ้าหาร 5 ลงตัวให้แสดง "Buzz" ถ้าหารทั้ง 3 และ 5 ลงตัวให้แสดง "FizzBuzz"
      </p>
      
      {/* ✅ Code Block */}
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-4 text-sm sm:text-base">
        <pre className="whitespace-pre-wrap sm:whitespace-pre">
{`def fizzBuzz(n):
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

fizzBuzz(15)`}
        </pre>
      </div>
    </div>
  );
};

export default PythonLeetcode;