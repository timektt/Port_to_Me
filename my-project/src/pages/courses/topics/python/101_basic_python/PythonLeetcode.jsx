import React from "react";

const PythonLeetcode = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center sm:text-left">
        แบบฝึกหัด Leetcode ด้วย Python
      </h1>

      <p className="mt-2 text-center sm:text-left text-gray-700 dark:text-gray-300">
        มาทดสอบความสามารถของคุณด้วยโจทย์โปรแกรมมิ่งพื้นฐาน!
      </p>

      {/* ✅ Two Sum */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ปัญหา: Two Sum</h2>
      <p className="mt-2">
        ให้ลิสต์ของตัวเลข และค่าหนึ่งค่า ให้หาตำแหน่งของตัวเลขสองตัวที่เมื่อรวมกันแล้วได้ค่าที่กำหนด
      </p>
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm sm:text-base">
        <pre>
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

      {/* ✅ Reverse String */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. ปัญหา: Reverse a String</h2>
      <p className="mt-2">จงเขียนฟังก์ชันที่รับสตริง และคืนค่าสตริงที่ถูกกลับด้าน</p>
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm sm:text-base">
        <pre>
{`def reverseString(s):
    return s[::-1]

print(reverseString("hello"))  # Output: "olleh"`}
        </pre>
      </div>

      {/* ✅ FizzBuzz */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. ปัญหา: FizzBuzz</h2>
      <p className="mt-2">
        แสดงตัวเลข 1 ถึง n ถ้าหาร 3 ลงตัวให้แสดง "Fizz", หาร 5 ลงตัวให้แสดง "Buzz", หารทั้ง 3 และ 5 ให้แสดง "FizzBuzz"
      </p>
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm sm:text-base">
        <pre>
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

      {/* ✅ Palindrome Check */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. ปัญหา: เช็คว่าข้อความเป็น Palindrome หรือไม่</h2>
      <p className="mt-2">Palindrome คือข้อความที่อ่านจากหน้าไปหลังหรือหลังไปหน้าก็เหมือนกัน เช่น "madam"</p>
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm sm:text-base">
        <pre>
{`def isPalindrome(s):
    return s == s[::-1]

print(isPalindrome("madam"))  # Output: True
print(isPalindrome("hello"))  # Output: False`}
        </pre>
      </div>

      {/* ✅ Count Vowels */}
      <h2 className="text-lg sm:text-xl font-semibold mt-6">5. ปัญหา: นับจำนวนสระในสตริง</h2>
      <p className="mt-2">ให้นับจำนวนตัวอักษร a, e, i, o, u (ไม่แยกตัวพิมพ์เล็ก-ใหญ่) ที่อยู่ในข้อความ</p>
      <div className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mt-2 text-sm sm:text-base">
        <pre>
{`def countVowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')

print(countVowels("Leetcode is awesome"))  # Output: 9`}
        </pre>
      </div>
    </div>
  );
};

export default PythonLeetcode;
