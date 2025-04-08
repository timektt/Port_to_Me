// tailwind.config.js (ESM)
import typography from '@tailwindcss/typography'

export default {
  darkMode: "class", // ✅ เพิ่มตรงนี้
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      typography: (theme) => ({
        DEFAULT: {
          css: {
            pre: {
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              overflowX: "auto",
              maxWidth: "100%",
              fontSize: theme("fontSize.sm"),
              backgroundColor: theme("colors.gray.800"),
              color: theme("colors.white"),
              borderRadius: theme("borderRadius.lg"),
              padding: theme("padding.4"),
            },
            code: {
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              overflowX: "auto",
            },
          },
        },
      }),
    },
  },
  plugins: [typography],
}
