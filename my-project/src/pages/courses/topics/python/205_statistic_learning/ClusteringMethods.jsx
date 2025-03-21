import React from "react";

const ClusteringMethods = () => {
  return (
    <div className="min-h-screen flex flex-col justify-start p-6">
      <h1 className="text-3xl font-bold">📊 วิธีการจัดกลุ่มข้อมูล (Clustering Methods)</h1>
      <p className="mt-4">
        การจัดกลุ่มข้อมูล (Clustering) เป็นเทคนิคแบบไม่ต้องมีคำตอบล่วงหน้า (Unsupervised Learning)
        ที่ใช้สำหรับแบ่งข้อมูลออกเป็นกลุ่มย่อยตามความคล้ายกันของลักษณะข้อมูล โดยไม่รู้ล่วงหน้าว่ามีกี่กลุ่ม
      </p>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">1. การจัดกลุ่มข้อมูลด้วย K-Means</h2>
      <p className="mt-2">
        K-Means จะสุ่มจุดศูนย์กลาง (Centroid) แล้วทำการแบ่งกลุ่มโดยยึดจากระยะห่างน้อยที่สุด และปรับศูนย์กลางซ้ำจนกว่าจะนิ่ง
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=100)
plt.title("K-Means Clustering")
plt.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">2. การจัดกลุ่มแบบ Hierarchical</h2>
      <p className="mt-2">
        Hierarchical Clustering สร้างโครงสร้างต้นไม้ (Dendrogram) เพื่อแสดงระดับของการรวมกลุ่ม เหมาะสำหรับการวิเคราะห์ความสัมพันธ์เชิงลำดับ
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linked = linkage(X, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("ตัวอย่างข้อมูล")
plt.ylabel("ระยะทาง")
plt.show()`}</code>
      </pre>

      <h2 className="text-lg sm:text-xl font-semibold mt-6">3. การจัดกลุ่มด้วย DBSCAN</h2>
      <p className="mt-2">
        DBSCAN ใช้ระยะใกล้ (eps) และจำนวนเพื่อนบ้านขั้นต่ำ (min_samples) เพื่อจัดกลุ่ม เหมาะกับข้อมูลที่มีรูปทรงไม่แน่นอนหรือมี outlier
      </p>
      <pre className="overflow-x-auto bg-gray-800 text-white p-4 rounded-lg">
        <code>{`from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

dbscan = DBSCAN(eps=1.5, min_samples=2)
dbscan.fit(X)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap="plasma")
plt.title("DBSCAN Clustering")
plt.show()`}</code>
      </pre>
    </div>
  );
};

export default ClusteringMethods;
