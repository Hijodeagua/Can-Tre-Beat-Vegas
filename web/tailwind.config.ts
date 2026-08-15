import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        fav: '#16a34a',
        dog: '#dc2626',
        // Sport accents. Also declared as CSS variables in globals.css, which
        // is what the components actually read; these let a utility class
        // reach the same value where that is simpler.
        accent: '#3ddc84',
        highlight: '#ffd23f',
        magenta: '#ff2e88',
      },
      // The Slate layer's radius scale: tiles/chips 6, rows/tabs 8, cards and
      // tables 12. Overrides Tailwind's defaults so `rounded-lg` means the
      // spec's 12px everywhere rather than 8.
      borderRadius: {
        sm: '6px',
        md: '8px',
        lg: '12px',
      },
    },
  },
  plugins: [],
};

export default config;
