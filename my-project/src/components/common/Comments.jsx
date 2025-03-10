import React, { useEffect } from 'react';

const Comments = ({ theme = 'dark' }) => {
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.setAttribute('data-repo', 'timektt/Port_to_Me');
    script.setAttribute('data-repo-id', 'R_kgDOKmd8jQ'); // ✅ เปลี่ยนเป็น repo ID แบบถูกต้อง
    script.setAttribute('data-category', 'Announcements');
    script.setAttribute('data-category-id', 'DIC_kwDOKmd8jc4Ccin1'); // ✅ เปลี่ยนเป็น Category ID แบบถูกต้อง
    script.setAttribute('data-mapping', 'pathname');
    script.setAttribute('data-strict', '0');
    script.setAttribute('data-reactions-enabled', '1');
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', 'bottom');
    script.setAttribute('data-theme', theme); // ✅ ปรับเปลี่ยนธีมอัตโนมัติ (dark/light)
    script.setAttribute('data-lang', 'en');
    script.crossOrigin = 'anonymous';
    script.async = true;

    const commentsContainer = document.getElementById('giscus-container');
    commentsContainer.innerHTML = '';
    commentsContainer.appendChild(script);
  }, [theme]); // ✅ ปรับ theme ตาม state ได้

  return <div id="giscus-container" className="mt-8" />;
};

export default Comments;
