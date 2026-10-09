import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/styles/base.css";
import { OverlayApp } from "./OverlayApp";

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <OverlayApp />
    </StrictMode>,
  );
}
