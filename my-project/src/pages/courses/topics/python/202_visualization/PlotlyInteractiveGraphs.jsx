import React from "react";

const PlotlyInteractiveGraphs = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Plotly: Interactive Graphs</h1>
      <p>Plotly เป็นไลบรารีสำหรับการสร้างกราฟแบบอินเทอร์แอคทีฟ</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()`}</code>
      </pre>
    </div>
  );
};

export default PlotlyInteractiveGraphs;
