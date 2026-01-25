import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  envPrefix: "CTRON_",
  test: {
    environment: "jsdom",
    setupFiles: "vitest.setup.ts",
  },
  server: {
    port: 3000,
    host: true, // needed for the Docker Container port mapping to work
    proxy: {
      // Proxy API requests to the arena backend
      '/api/arena': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy WebSocket connections
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
