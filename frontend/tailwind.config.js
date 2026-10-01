/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#F7F2EB',
        surface: {
          DEFAULT: '#FFFFFF',
          secondary: '#EAE2D6',
          recessed: '#F0EAE1',
        },
        debtox: {
          primary: '#8B9A6E',
          'primary-hover': '#728056',
          'primary-light': '#F1F4EB',
          border: '#E3DDD3',
          'border-subtle': '#EEEEEE',
          ink: '#242720',
          'ink-muted': '#6C7065',
          'ink-subtle': '#8F9288',
        },
        risk: {
          high: '#B85A5A',
          'high-bg': '#FAF0F0',
          medium: '#C89255',
          'medium-bg': '#FAF4ED',
          low: '#8B9A6E',
          'low-bg': '#F2F5ED',
        }
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        subtle: '0 1px 3px rgba(36, 39, 32, 0.04), 0 1px 2px rgba(36, 39, 32, 0.02)',
        card: '0 2px 8px -2px rgba(36, 39, 32, 0.05), 0 1px 3px 0 rgba(36, 39, 32, 0.03)',
        modal: '0 8px 30px -4px rgba(36, 39, 32, 0.12), 0 2px 8px -2px rgba(36, 39, 32, 0.04)',
      }
    },
  },
  plugins: [],
}
