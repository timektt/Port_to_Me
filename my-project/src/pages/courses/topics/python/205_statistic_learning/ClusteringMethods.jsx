import React from "react";

const ClusteringMethods = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">Clustering Methods</h1>
      <p>การจัดกลุ่มข้อมูลเป็นเทคนิคสำคัญใน Machine Learning</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# สร้างข้อมูลตัวอย่าง
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# ใช้ K-Means แบ่งกลุ่มข้อมูล
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# แสดงผลลัพธ์
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X')
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default ClusteringMethods;
