/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#070B16',
          900: '#0B1020',
          800: '#131B2F',
          700: '#1B2642',
        },
        teal: {
          accent: '#22D3C5',
        },
        purple: {
          soft: '#A78BFA',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
};

