import React, { useState, useEffect } from "react";

const HandlingLoadingAndErrors = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("https://jsonplaceholder.typicode.com/posts/9999")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Data not found");
        }
        return response.json();
      })
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold">Handling Loading & Errors</h1>
      <p className="mt-4 text-lg">
        เราสามารถจัดการสถานะ Loading และข้อผิดพลาดเมื่อทำการ Fetch Data ได้
      </p>

      <h2 className="text-2xl font-semibold mt-6">📌 ตัวอย่างการจัดการ Loading & Error</h2>
      <pre className="p-4 rounded-md overflow-x-auto border">
{`useEffect(() => {
  fetch("https://jsonplaceholder.typicode.com/posts/9999")
    .then(response => {
      if (!response.ok) {
        throw new Error("Data not found");
      }
      return response.json();
    })
    .then(data => setData(data))
    .catch(error => setError(error.message));
}, []);`}
      </pre>

      {loading && <p>Loading...</p>}
      {error && <p className="text-red-600">{error}</p>}
      {data && <p className="mt-4">{data?.title}</p>}
    </div>
  );
};

export default HandlingLoadingAndErrors;
