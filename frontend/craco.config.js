module.exports = {
  style: {
    postcssOptions: {
      plugins: [
        require('tailwindcss/nesting'),
        require('tailwindcss'),
        require('autoprefixer'),
      ],
    },
  },
}