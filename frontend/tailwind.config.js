/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"] },
      colors: { ink: "#17243b", muted: "#6d7890", canvas: "#f5f7fb" },
      boxShadow: { card: "0 8px 30px rgba(30, 48, 85, 0.045)" },
    },
  },
  plugins: [],
};
