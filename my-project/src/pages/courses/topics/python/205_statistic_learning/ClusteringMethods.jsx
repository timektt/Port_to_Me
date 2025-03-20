import React from "react";

const ClusteringMethods = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 วิธีการจัดกลุ่มข้อมูล (Clustering Methods)</h1>
      <p className="mt-4">
        การจัดกลุ่มข้อมูลเป็นเทคนิคสำคัญใน Machine Learning ที่ใช้ในการแบ่งกลุ่มข้อมูลโดยอัตโนมัติโดยพิจารณาจากลักษณะของข้อมูล
      </p>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การจัดกลุ่มข้อมูลด้วย K-Means</h2>
      <p className="mt-2">K-Means เป็นวิธีการจัดกลุ่มข้อมูลโดยใช้ระยะห่างจากจุดศูนย์กลางของแต่ละกลุ่ม</p>
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
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การจัดกลุ่มข้อมูลด้วย Hierarchical Clustering</h2>
      <p className="mt-2">ใช้วิธีการแบ่งกลุ่มแบบเป็นลำดับชั้น ซึ่งสามารถใช้ดูความสัมพันธ์ระหว่างข้อมูล</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# สร้างโครงสร้างการจัดกลุ่มแบบ Hierarchical
linked = linkage(X, 'ward')

plt.figure(figsize=(6,4))
dendrogram(linked)
plt.show()`}</code>
      </pre>
      
      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การจัดกลุ่มด้วย DBSCAN</h2>
      <p className="mt-2">DBSCAN เป็นอัลกอริธึมที่ใช้ในการจัดกลุ่มข้อมูลที่ไม่มีจำนวนกลุ่มที่แน่นอน</p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from sklearn.cluster import DBSCAN

# ใช้ DBSCAN จัดกลุ่มข้อมูล
dbscan = DBSCAN(eps=1.5, min_samples=2)
dbscan.fit(X)

# แสดงผลลัพธ์
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap="plasma")
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default ClusteringMethods;
