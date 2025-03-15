import React from "react";

const DeepLearningIntro = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Deep Learning Basics</h1>
      <p>พื้นฐานของ Deep Learning และการสร้างโมเดลเครือข่ายประสาทเทียม</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import tensorflow as tf
from tensorflow import keras

# สร้างโมเดลเครือข่ายประสาทเทียมแบบง่าย
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("โมเดลถูกสร้างเรียบร้อย!")`}</code>
      </pre>
    </div>
  );
};

export default DeepLearningIntro;
