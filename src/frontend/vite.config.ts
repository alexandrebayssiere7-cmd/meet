import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tsconfigPaths from 'vite-tsconfig-paths'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())
  return {
    plugins: [react(), tsconfigPaths()],
    build: {
      sourcemap: env.VITE_BUILD_SOURCEMAP === 'true',
    },
    optimizeDeps: {
      // onnxruntime-web ships pre-bundled WASM workers that Vite's optimizer
      // breaks — exclude it so it's loaded as-is from node_modules.
      exclude: ['onnxruntime-web'],
    },
    server: {
      port: parseInt(env.VITE_PORT) || 3000,
      host: env.VITE_HOST ?? 'localhost',
      allowedHosts: ['.nip.io'],
      // In a local dev setup, we proxy the media server ourselves to avoid CORS issues
      proxy: {
        '/media': {
          target: 'http://localhost:8083',
          changeOrigin: true,
          secure: false
        }
      }
    },
  }
})
