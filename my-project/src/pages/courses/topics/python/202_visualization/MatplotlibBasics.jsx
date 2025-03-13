import React from "react";

const MatplotlibBasics = () => {
  return (
    <div>
      <h1>Matplotlib Basics</h1>
      <p>Matplotlib เป็นไลบรารีสำหรับการสร้างกราฟใน Python</p>
      <pre>
        <code>{`import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Matplotlib Graph')
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default MatplotlibBasics;
