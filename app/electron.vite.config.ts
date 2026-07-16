import { resolve } from 'path'
import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  main: {},
  preload: {},
  renderer: {
    resolve: {
      alias: {
        '@renderer': resolve('src/renderer/src')
      }
    },
    build: {
      rollupOptions: {
        input: {
          index: resolve('src/renderer/index.html'),
          notif: resolve('src/renderer/notif.html')
        }
      }
    },
    plugins: [react(), tailwindcss()],
    server: {
      port: Number(process.env.PORT) || 5174,
      strictPort: true
    }
  }
})
