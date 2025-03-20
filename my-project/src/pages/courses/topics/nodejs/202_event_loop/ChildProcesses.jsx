import React from "react";

const ChildProcesses = () => {
  return (
    <div className="max-w-3xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">⚡ การใช้ Child Processes ใน Node.js</h1>
      <p>
        ใน Node.js เราสามารถสร้างและจัดการ <code>Child Processes</code> เพื่อรันงานแบบขนานได้ โดยใช้ <code>child_process</code> module
      </p>
      
      <h2 className="text-xl font-semibold mt-6">📌 การใช้ exec() รันคำสั่ง Shell</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const { exec } = require('child_process');

exec('ls', (error, stdout, stderr) => {
  if (error) {
    console.error('Error:', error.message);
    return;
  }
  console.log('Output:', stdout);
});`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 การใช้ spawn() เพื่อรันโปรเซสแยก</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const { spawn } = require('child_process');

const child = spawn('node', ['-v']);

child.stdout.on('data', (data) => {
  console.log('Output:', data.toString());
});

child.stderr.on('data', (data) => {
  console.error('Error:', data.toString());
});`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 การใช้ fork() เพื่อแยกโปรเซสแบบอิสระ</h2>
      <div className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
        <pre>
          <code>{`const { fork } = require('child_process');

const child = fork('child_script.js');

child.on('message', (message) => {
  console.log('Message from child:', message);
});

child.send({ text: 'Hello from parent' });`}</code>
        </pre>
      </div>
      
      <h2 className="text-xl font-semibold mt-6">📌 สรุป</h2>
      <p>
        ✅ <code>exec()</code>: ใช้รันคำสั่ง shell และรับ output ทันที<br/>
        ✅ <code>spawn()</code>: ใช้สร้างโปรเซสแยกและสื่อสารผ่าน <code>stdout</code>/<code>stderr</code><br/>
        ✅ <code>fork()</code>: ใช้สำหรับแยกโปรเซส Node.js และสื่อสารผ่าน <code>message</code>
      </p>
    </div>
  );
};

export default ChildProcesses;
