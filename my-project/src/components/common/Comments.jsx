import React, { useEffect } from "react";

const Comments = ({ theme = "dark" }) => {
  useEffect(() => {
    const commentsContainer = document.getElementById("giscus-container");

    if (!commentsContainer) return;

    // เคลียร์ของเก่าเผื่อมี script ซ้ำ
    commentsContainer.innerHTML = "";

    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    script.setAttribute("data-repo", "timektt/Port_to_Me");
    script.setAttribute("data-repo-id", "R_kgDOKmd8jQ");
    script.setAttribute("data-category", "Announcements");
    script.setAttribute("data-category-id", "DIC_kwDOKmd8jc4Ccin1");
    script.setAttribute("data-mapping", "pathname");
    script.setAttribute("data-strict", "0");
    script.setAttribute("data-reactions-enabled", "1");
    script.setAttribute("data-emit-metadata", "0");
    script.setAttribute("data-input-position", "bottom");
    script.setAttribute("data-theme", theme);
    script.setAttribute("data-lang", "en");
    script.crossOrigin = "anonymous";
    script.async = true;

    commentsContainer.appendChild(script);

    return () => {
      // ❌ ล้าง Giscus ตอนเปลี่ยนหน้า ป้องกัน memory leak หรือ duplication
      commentsContainer.innerHTML = "";
    };
  }, [theme]);

  return <div id="giscus-container" className="mt-8" />;
};

export default Comments;
