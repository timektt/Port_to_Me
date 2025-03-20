import React from "react";

const PlotlyInteractiveGraphs = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 การสร้างกราฟอินเทอร์แอคทีฟด้วย Plotly</h1>
      <p className="mt-4">
        Plotly เป็นไลบรารีสำหรับการสร้างกราฟแบบอินเทอร์แอคทีฟที่สามารถโต้ตอบได้ ใช้งานง่าย และรองรับหลากหลายรูปแบบของกราฟ เช่น กราฟเส้น, กราฟแท่ง, Scatter plot และ Heatmap
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การติดตั้ง Plotly</h2>
      <p className="mt-2">ก่อนใช้งาน Plotly ต้องติดตั้งไลบรารีก่อน โดยใช้คำสั่ง:</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`pip install plotly`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. ตัวอย่างการสร้าง Scatter Plot</h2>
      <p className="mt-2">ใช้ <code>plotly.express</code> ในการสร้างกราฟ Scatter Plot จากข้อมูลชุด Iris</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การสร้างกราฟแท่ง (Bar Chart)</h2>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.tips()
fig = px.bar(df, x="day", y="total_bill", color="sex", barmode="group")
fig.show()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">4. การสร้างกราฟเส้น (Line Chart)</h2>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.gapminder()
fig = px.line(df[df["country"] == "Thailand"], x="year", y="gdpPercap", title="GDP ของประเทศไทย")
fig.show()`}</code>
      </pre>
    </div>
  );
};

export default PlotlyInteractiveGraphs;
