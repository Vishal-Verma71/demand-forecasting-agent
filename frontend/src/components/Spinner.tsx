"use client";

import React from "react";

export function Spinner({ size = 18 }: { size?: number }) {
  return (
    <span
      aria-hidden="true"
      style={{
        display: "inline-block",
        width: size,
        height: size,
        border: "2px solid rgba(148,163,184,0.35)",
        borderTopColor: "#6366f1",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
      }}
    />
  );
}

export function LoadingStatus({ visible, message }: { visible: boolean; message: string }) {
  if (!visible) return null;
  return (
    <div role="status" aria-live="polite" style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", color: "#cbd5e1", fontSize: 13 }}>
      <Spinner />
      <span>{message}</span>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}