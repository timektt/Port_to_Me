import React from "react";

const FsPromises = () => {
  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">📂 Using fs.promises for File System</h1>
      <p className="mb-4">
        ใน Node.js สามารถใช้ <code>fs.promises</code> เพื่อทำงานกับไฟล์แบบ Asynchronous ได้โดยไม่ต้องใช้ Callback
      </p>
      
      <h2 className="text-xl font-semibold mt-6">📌 อ่านไฟล์แบบ Asynchronous</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const fs = require('fs').promises;

async function readFile() {
  try {
    const data = await fs.readFile('example.txt', 'utf8');
    console.log(data);
  } catch (err) {
    console.error('Error reading file:', err);
  }
}

readFile();`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 เขียนไฟล์แบบ Asynchronous</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const fs = require('fs').promises;

async function writeFile() {
  try {
    await fs.writeFile('example.txt', 'Hello, Node.js!');
    console.log('File written successfully');
  } catch (err) {
    console.error('Error writing file:', err);
  }
}

writeFile();`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 ลบไฟล์แบบ Asynchronous</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const fs = require('fs').promises;

async function deleteFile() {
  try {
    await fs.unlink('example.txt');
    console.log('File deleted successfully');
  } catch (err) {
    console.error('Error deleting file:', err);
  }
}

deleteFile();`}</code>
        </pre>
      </div>
    </div>
  );
};

export default FsPromises;
