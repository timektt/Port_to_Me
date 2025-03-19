import React, { useState, useEffect } from "react";
import axios from "axios";

const UsingAxiosForHttpRequests = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get("https://jsonplaceholder.typicode.com/posts/1")
      .then((response) => {
        setData(response.data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Using Axios for HTTP Requests</h1>
      <p className="mt-4 text-lg">
        Axios เป็นไลบรารีที่ช่วยให้การทำ HTTP Requests สะดวกขึ้น
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการใช้ Axios</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
{`axios.get("https://jsonplaceholder.typicode.com/posts/1")
  .then(response => console.log(response.data))
  .catch(error => console.error(error));`}
      </pre>

      {loading ? <p>Loading...</p> : <p className="mt-4">{data?.title}</p>}
    </div>
  );
};

export default UsingAxiosForHttpRequests;
