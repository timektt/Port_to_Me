// File: LoggingMonitoring.jsx

import React from "react";

const LoggingMonitoring = () => {
  return (
    <div className="p-4 sm:p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl sm:text-3xl font-bold">📋 การบันทึก Log และการตรวจสอบประสิทธิภาพ (Logging & Monitoring)</h1>

      <p className="mt-4 text-base sm:text-lg">
        ในการพัฒนาแอปพลิเคชัน การบันทึก Log และการตรวจสอบประสิทธิภาพ (Performance Monitoring)
        มีบทบาทสำคัญอย่างยิ่งในการแก้ไขปัญหา ตรวจสอบข้อผิดพลาด และปรับปรุงประสบการณ์ผู้ใช้ให้ดีขึ้น
      </p>

      <h2 className="text-xl font-semibold mt-6">📝 การใช้ Logging</h2>
      <p className="mt-2">Python มีไลบรารี <code>logging</code> ที่ใช้บันทึกเหตุการณ์ต่าง ๆ ในระบบ เช่น:</p>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-x-auto text-sm">
{`import logging

# ตั้งค่าระดับของ log
logging.basicConfig(level=logging.INFO)

logging.debug("ข้อความสำหรับ debugging")
logging.info("เริ่มต้นโปรแกรม")
logging.warning("คำเตือนบางอย่าง")
logging.error("เกิดข้อผิดพลาดบางอย่าง")
logging.critical("เกิดปัญหาร้ายแรง")`}
      </pre>
      <ul className="list-disc ml-6 mt-2 text-sm sm:text-base">
        <li><code>DEBUG</code>: รายละเอียดสำหรับนักพัฒนา</li>
        <li><code>INFO</code>: ข้อมูลทั่วไปเกี่ยวกับการทำงาน</li>
        <li><code>WARNING</code>: คำเตือนที่อาจต้องตรวจสอบ</li>
        <li><code>ERROR</code>: ข้อผิดพลาดที่เกิดขึ้น</li>
        <li><code>CRITICAL</code>: ข้อผิดพลาดร้ายแรง</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">📊 การตรวจสอบประสิทธิภาพ (Monitoring)</h2>
      <p className="mt-2">Monitoring ช่วยให้เราทราบว่าแอปพลิเคชันทำงานช้าตรงไหน ใช้ทรัพยากรเท่าไร หรือเกิดปัญหาบ่อยแค่ไหน</p>
      <p className="mt-2">
        เครื่องมือยอดนิยม เช่น:
      </p>
      <ul className="list-disc ml-6 mt-2 text-sm sm:text-base">
        <li><strong>Prometheus</strong>: เก็บข้อมูล Metric</li>
        <li><strong>Grafana</strong>: แสดงข้อมูลเป็น Dashboard</li>
        <li><strong>Sentry</strong>: ตรวจจับและรายงาน Error ของโปรแกรม</li>
        <li><strong>New Relic / Datadog</strong>: ครบวงจรทั้ง Monitoring และ Tracing</li>
      </ul>

      <h2 className="text-xl font-semibold mt-6">✅ สรุป</h2>
      <p className="mt-2">
        การบันทึก Log และการ Monitoring เป็นเครื่องมือที่จำเป็นสำหรับการดูแลระบบให้เสถียร
        ลดเวลาในการแก้ไขปัญหา และเข้าใจพฤติกรรมของแอปพลิเคชันในโลกจริง
      </p>
    </div>
  );
};

export default LoggingMonitoring;
