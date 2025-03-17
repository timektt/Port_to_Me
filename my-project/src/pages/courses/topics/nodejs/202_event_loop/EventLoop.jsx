import React from "react";

const EventLoop = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-gray-100 dark:bg-gray-900">
      <h1 className="text-2xl md:text-4xl font-bold text-gray-800 dark:text-white text-center">
        Understanding the Event Loop
      </h1>
      <p className="mt-4 text-gray-600 dark:text-gray-300 text-center max-w-2xl">
        The Event Loop in JavaScript allows asynchronous programming by handling tasks in a non-blocking manner.
        It continuously checks the call stack, microtask queue, and event queue to manage execution.
      </p>
      <div className="mt-6 p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg w-full max-w-2xl">
        <h2 className="text-lg font-semibold text-gray-700 dark:text-white">Key Concepts:</h2>
        <ul className="list-disc mt-2 pl-5 text-gray-600 dark:text-gray-300">
          <li>Call Stack</li>
          <li>Web APIs</li>
          <li>Callback Queue</li>
          <li>Microtasks & Macrotasks</li>
          <li>Execution Order</li>
        </ul>
      </div>
    </div>
  );
};

export default EventLoop;
