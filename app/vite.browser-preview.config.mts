// Temporary browser-only preview of the renderer (not part of the build).
// Proxies sidecar REST + websockets through the dev-server origin so the
// preview browser (which can only reach this origin) can talk to the demo
// sidecar on 8932. Run with VITE_SIDECAR_PORT=5321.
import { resolve } from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const SIDECAR = 'http://127.0.0.1:8932'
const SIDECAR_WS = 'ws://127.0.0.1:8932'

export default defineConfig({
  root: resolve(import.meta.dirname, 'src/renderer'),
  resolve: {
    alias: {
      '@renderer': resolve(import.meta.dirname, 'src/renderer/src')
    }
  },
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5322,
    strictPort: true,
    proxy: {
      '/startup': SIDECAR,
      '/projects': SIDECAR,
      '/graph': SIDECAR,
      '/facts': SIDECAR,
      '/notes': SIDECAR,
      '/profile': SIDECAR,
      '/summary': SIDECAR,
      '/chats': SIDECAR,
      '/connectors': SIDECAR,
      '/index': SIDECAR,
      '/sessions': SIDECAR,
      '/updates': { target: SIDECAR_WS, ws: true },
      '/session': { target: SIDECAR_WS, ws: true },
      '/chat': { target: SIDECAR_WS, ws: true }
    }
  }
})
