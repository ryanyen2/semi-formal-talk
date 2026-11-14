import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// BASE is the path the site is served under: /semi-formal-talk/ on GitHub Pages, / elsewhere
export default defineConfig({ base: process.env.BASE ?? '/', plugins: [react()], worker: { format: 'es' } })
