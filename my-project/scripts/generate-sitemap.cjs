const fs = require("fs");
const path = require("path");
const { routePaths } = require("../src/data/staticRoutes.cjs"); // ✅ import จาก array จริง

const baseUrl = "https://www.superbear.dev";

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${routePaths.map((route) => `<url><loc>${baseUrl}${route}</loc></url>`).join("\n")}
</urlset>
`;

fs.writeFileSync(path.resolve(__dirname, "../public/sitemap(1).xml"), xml);
console.log("✅ sitemap.xml generated from staticRoutes.js");
