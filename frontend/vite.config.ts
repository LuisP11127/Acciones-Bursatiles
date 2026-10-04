import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base relativa: funciona en https://<usuario>.github.io/<repositorio>/ sin configurar nada
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1200,
  },
});
