import React from "react";

const DeepLearningIntro = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">🧠 พื้นฐาน Deep Learning (Deep Learning Basics)</h1>
      <p className="mt-4">
        Deep Learning เป็นแขนงหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทเทียม (Neural Networks)
        ในการเรียนรู้จากข้อมูลจำนวนมาก โดยเฉพาะข้อมูลที่ซับซ้อน เช่น รูปภาพ เสียง หรือข้อความ
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. ส่วนประกอบหลักของ Neural Network</h2>
      <ul className="list-disc ml-5 mt-2">
        <li><strong>Neuron:</strong> หน่วยพื้นฐานที่ประมวลผลข้อมูล</li>
        <li><strong>Layers:</strong> ประกอบด้วย Input Layer, Hidden Layers และ Output Layer</li>
        <li><strong>Activation Function:</strong> เช่น ReLU, Sigmoid, Softmax ใช้เพิ่มความไม่เชิงเส้น</li>
      </ul>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การสร้างโมเดลด้วย Keras</h2>
      <p className="mt-2">โมเดล Sequential เป็นวิธีสร้างโครงข่ายแบบง่ายที่สุดใน Keras</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import tensorflow as tf
from tensorflow import keras

# สร้างโมเดลแบบ Sequential
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),      # แปลงภาพ 2D เป็นเวกเตอร์ 1D
    keras.layers.Dense(128, activation='relu'),      # Hidden layer
    keras.layers.Dense(10, activation='softmax')     # Output layer สำหรับคลาส 0-9
])

# คอมไพล์โมเดล
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ โมเดลถูกสร้างเรียบร้อย")`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การฝึกโมเดลด้วยข้อมูล MNIST</h2>
      <p className="mt-2">MNIST เป็นชุดข้อมูลตัวเลข 0-9 ที่ใช้บ่อยในการทดลอง Deep Learning</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`# โหลดชุดข้อมูล
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ปรับค่า pixel ให้อยู่ในช่วง 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# ฝึกโมเดล
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การประเมินผลและการทำนาย</h2>
      <p className="mt-2">เราสามารถทดสอบโมเดลด้วยข้อมูลใหม่ และใช้ในการทำนายผล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`# ประเมินโมเดล
loss, acc = model.evaluate(x_test, y_test)
print(f"🎯 Accuracy: {acc:.2f}")

# ทำนายค่าตัวเลขจากภาพ
pred = model.predict(x_test[:1])
print("Prediction:", pred.argmax())`}</code>
      </pre>
    </div>
  );
};

export default DeepLearningIntro;
