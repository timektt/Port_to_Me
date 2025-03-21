import React from "react";

const PlotlyInteractiveGraphs = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">📊 การสร้างกราฟอินเทอร์แอคทีฟด้วย Plotly</h1>
      <p className="mt-4">
        Plotly เป็นไลบรารีสำหรับสร้างกราฟแบบอินเทอร์แอคทีฟ รองรับกราฟหลายรูปแบบ เช่น เส้น, แท่ง, Scatter, Heatmap และสามารถซูม, เลื่อน, โต้ตอบกับข้อมูลได้
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การติดตั้ง Plotly</h2>
      <p className="mt-2">ติดตั้งด้วยคำสั่ง:</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`pip install plotly`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. Scatter Plot (กราฟจุด)</h2>
      <p className="mt-2">แสดงความสัมพันธ์ระหว่างความกว้างและความยาวของกลีบจากชุดข้อมูล Iris</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. Bar Chart (กราฟแท่ง)</h2>
      <p className="mt-2">เปรียบเทียบค่า Total Bill แยกตามวันและเพศจากชุดข้อมูล Tips</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.tips()
fig = px.bar(df, x="day", y="total_bill", color="sex", barmode="group")
fig.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. Line Chart (กราฟเส้น)</h2>
      <p className="mt-2">แสดงการเปลี่ยนแปลงของ GDP ต่อคนในประเทศไทยตามช่วงปี</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.gapminder()
fig = px.line(df[df["country"] == "Thailand"], x="year", y="gdpPercap", title="GDP ของประเทศไทย")
fig.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">5. Heatmap (กราฟความร้อน)</h2>
      <p className="mt-2">แสดงความสัมพันธ์แบบเมทริกซ์ (Matrix) ของค่า Z</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px
import numpy as np

z = [[1, 20, 30],
     [20, 1, 60],
     [30, 60, 1]]

fig = px.imshow(z, text_auto=True)
fig.show()`}</code>
      </pre>
    </div>
  );
};

export default PlotlyInteractiveGraphs;
