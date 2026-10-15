"use client";

import React from "react";

export type TableColumn<T> = {
  key: string;
  header: string;
  width?: string;
  align?: "left" | "right" | "center";
  render?: (row: T) => React.ReactNode;
};

export function Table<T extends Record<string, any>>({
  columns,
  rows,
  rowKey,
  loading = false,
  loadingMessage = "Loading...",
  emptyMessage = "No data available.",
}: {
  columns: TableColumn<T>[];
  rows: T[];
  rowKey?: (row: T, index: number) => string;
  loading?: boolean;
  loadingMessage?: string;
  emptyMessage?: string;
}) {
  const gridTemplate = columns.map((c) => c.width || "1fr").join(" ");
  const cellBase: React.CSSProperties = {
    padding: "10px 12px",
    borderBottom: "1px solid #1f2a3d",
    overflowWrap: "break-word", // never split short words such as WARNING or HOUSEHOLD
    verticalAlign: "top",
    fontSize: 13,
  };

  return (
    <div style={{ width: "100%", overflowX: "auto", border: "1px solid #1f2a3d", borderRadius: 12, background: "#0b1220" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: gridTemplate,
          background: "#1e293b",
          borderBottom: "1px solid #334155",
          position: "sticky",
          top: 0,
        }}
      >
        {columns.map((c) => (
          <div
            key={c.key}
            style={{ ...cellBase, fontWeight: 700, color: "#94a3b8", textAlign: c.align || "left", borderTop: "none" }}
          >
            {c.header}
          </div>
        ))}
      </div>
      {loading ? (
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: 16, color: "#cbd5e1" }}>
          <span className="spinner" style={{ width: 16, height: 16, border: "2px solid rgba(148,163,184,.35)", borderTopColor: "#6366f1", borderRadius: "50%", animation: "spin .8s linear infinite", display: "inline-block" }} />
          {loadingMessage}
        </div>
      ) : rows.length === 0 ? (
        <div style={{ padding: 16, color: "#94a3b8" }}>{emptyMessage}</div>
      ) : (
        rows.map((row, index) => (
          <div
            key={rowKey ? rowKey(row, index) : `${index}`}
            style={{ display: "grid", gridTemplateColumns: gridTemplate, background: index % 2 === 0 ? "#0b1220" : "#0f172a" }}
          >
            {columns.map((c) => (
              <div key={c.key} style={{ ...cellBase, textAlign: c.align || "left", color: "#e2e8f0" }}>
                {c.render ? c.render(row) : row[c.key]}
              </div>
            ))}
          </div>
        ))
      )}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}