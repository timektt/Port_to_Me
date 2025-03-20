import React from "react";

const DeepLearningIntro = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">🧠 พื้นฐาน Deep Learning (Deep Learning Basics)</h1>
      <p className="mt-4">
        Deep Learning เป็นส่วนหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทเทียม (Neural Networks) ในการเรียนรู้จากข้อมูลจำนวนมาก
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ส่วนประกอบหลักของ Deep Learning</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Neuron</strong>: หน่วยพื้นฐานของโครงข่ายประสาทเทียม</li>
        <li><strong>Layers</strong>: ประกอบไปด้วยชั้นนำเข้า (Input Layer), ชั้นซ่อน (Hidden Layers), และชั้นส่งออก (Output Layer)</li>
        <li><strong>Activation Functions</strong>: ฟังก์ชันที่ช่วยให้เครือข่ายเรียนรู้ค่าที่ซับซ้อนขึ้น เช่น ReLU และ Softmax</li>
      </ul>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การสร้างโมเดลเครือข่ายประสาทเทียมแบบง่าย</h2>
      <p className="mt-2">ใช้ TensorFlow และ Keras เพื่อสร้างโมเดลที่มี 2 ชั้นหลัก</p>
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
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การฝึกโมเดล (Training the Model)</h2>
      <p className="mt-2">โมเดลต้องถูกฝึกด้วยข้อมูลและปรับค่าพารามิเตอร์ให้เหมาะสม</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`# โหลดชุดข้อมูล MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# แปลงข้อมูลเป็นช่วง 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# ฝึกโมเดล
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การประเมินผลและทำนายค่า</h2>
      <p className="mt-2">โมเดลสามารถถูกนำไปใช้ทดสอบกับข้อมูลใหม่</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`# ทดสอบโมเดล
loss, acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {acc}")

# ทำนายค่าบนตัวอย่างแรก
predictions = model.predict(x_test[:1])
print("Prediction:", predictions.argmax())`}</code>
      </pre>
    </div>
  );
};

export default DeepLearningIntro;
