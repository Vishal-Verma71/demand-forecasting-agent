"use client";

import React from "react";

type Props = {
  stores: string[];
  departments: string[];
  model: string;
  rows: number;
  horizon: number;
  dataThrough?: string;
  modelReady: boolean;
  planner: string;
  llm: boolean;
};

function Icon({ d, color }: { d: string; color: string }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

const ICONS = {
  store: "M3 9l1-5h16l1 5M4 9v11h16V9M9 20v-6h6v6",
  dept: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  model: "M12 2l3 6 6 1-4.5 4.5L18 20l-6-3-6 3 1.5-6.5L3 9l6-1z",
  rows: "M4 6h16M4 12h16M4 18h16",
  horizon: "M3 12h14M13 6l6 6-6 6",
  agent: "M12 3a4 4 0 014 4v2a4 4 0 01-8 0V7a4 4 0 014-4zM5 21v-1a7 7 0 0114 0v1",
};

function Chip({ icon, color, label, value }: { icon: string; color: string; label: string; value: React.ReactNode }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 8, padding: "7px 12px", borderRadius: 12,
      background: "rgba(15, 23, 42, 0.55)", border: "1px solid rgba(148, 163, 184, 0.18)",
      backdropFilter: "blur(6px)", fontSize: 12.5, lineHeight: 1,
    }}>
      <span style={{ display: "grid", placeItems: "center", width: 24, height: 24, borderRadius: 8, background: `${color}22`, border: `1px solid ${color}55` }}>
        <Icon d={icon} color={color} />
      </span>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function StatusRow({ ok, label, value }: { ok: boolean; label: string; value: string }) {
  const color = ok ? "#34d399" : "#fbbf24";
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, fontSize: 13 }}>
      <span style={{ display: "flex", alignItems: "center", gap: 8, color: "#cbd5e1" }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, boxShadow: `0 0 10px ${color}` }} />
        {label}
      </span>
      <span style={{ color: "#f8fafc", fontWeight: 600 }}>{value}</span>
    </div>
  );
}

export function HeaderBanner(p: Props) {
  return (
    <header style={{
      position: "relative", overflow: "hidden", marginBottom: 18, borderRadius: 20, padding: "26px 28px 22px",
      background: "linear-gradient(120deg, #3b0764 0%, #312e81 38%, #1e1b4b 68%, #0f172a 100%)",
      border: "1px solid rgba(167, 139, 250, 0.28)",
      boxShadow: "0 20px 50px rgba(15, 23, 42, 0.55), inset 0 1px 0 rgba(255,255,255,0.06)",
    }}>
      {/* Decorative glows and grid texture */}
      <div aria-hidden="true" style={{ position: "absolute", width: 420, height: 420, top: -220, right: -80, borderRadius: "50%", background: "radial-gradient(circle, rgba(34,211,238,0.28), transparent 65%)" }} />
      <div aria-hidden="true" style={{ position: "absolute", width: 360, height: 360, bottom: -240, left: 120, borderRadius: "50%", background: "radial-gradient(circle, rgba(236,72,153,0.20), transparent 65%)" }} />
      <div aria-hidden="true" style={{
        position: "absolute", inset: 0, opacity: 0.35,
        backgroundImage: "linear-gradient(rgba(148,163,184,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.07) 1px, transparent 1px)",
        backgroundSize: "32px 32px", maskImage: "linear-gradient(90deg, transparent, black 55%)", WebkitMaskImage: "linear-gradient(90deg, transparent, black 55%)",
      }} />
      <div aria-hidden="true" style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 3, background: "linear-gradient(90deg, #8b5cf6, #22d3ee, #ec4899)" }} />

      <div style={{ position: "relative", display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 18, alignItems: "center", minWidth: 0, flex: "1 1 520px" }}>
          <div style={{
            flexShrink: 0, width: 64, height: 64, borderRadius: 18, display: "grid", placeItems: "center",
            background: "linear-gradient(135deg, #8b5cf6, #06b6d4)", boxShadow: "0 10px 30px rgba(139, 92, 246, 0.45)",
          }}>
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 3v18h18" />
              <path d="M7 15l4-4 3 3 6-7" />
              <path d="M16 7h4v4" />
            </svg>
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: ".14em", textTransform: "uppercase", color: "#c4b5fd" }}>
                Supply chain copilot
              </span>
              <span style={{
                display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 9px", borderRadius: 999, fontSize: 11,
                color: "#a7f3d0", background: "rgba(16,185,129,0.12)", border: "1px solid rgba(52,211,153,0.35)",
              }}>
                <span className="hb-pulse" style={{ width: 7, height: 7, borderRadius: "50%", background: "#34d399" }} />
                Live{p.dataThrough ? ` · data through ${p.dataThrough}` : ""}
              </span>
            </div>
            <h1 style={{
              margin: 0, fontSize: 34, lineHeight: 1.1, fontWeight: 800, letterSpacing: "-0.02em",
              background: "linear-gradient(90deg, #ffffff 0%, #c4b5fd 45%, #67e8f9 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              Demand Forecasting Agent
            </h1>
            <p style={{ margin: "8px 0 0", color: "#cbd5e1", fontSize: 14.5, maxWidth: 640 }}>
              Forecasts Walmart M5 demand, spots stockouts before they happen, and drafts supplier-balanced
              purchase orders for your approval.
            </p>
          </div>
        </div>

        <div style={{
          flex: "0 1 280px", display: "grid", gap: 10, padding: "14px 16px", borderRadius: 14,
          background: "rgba(15, 23, 42, 0.55)", border: "1px solid rgba(148, 163, 184, 0.2)", backdropFilter: "blur(8px)",
        }}>
          <StatusRow ok={p.modelReady} label="Forecast model" value={p.modelReady ? "Ready" : "Not trained"} />
          <StatusRow ok={p.llm} label="Agent planner" value={p.planner} />
          <StatusRow ok label="Forecast horizon" value={`${p.horizon} days`} />
        </div>
      </div>

      <div style={{ position: "relative", display: "flex", gap: 8, flexWrap: "wrap", marginTop: 20 }}>
        <Chip icon={ICONS.store} color="#a78bfa" label="Stores" value={p.stores.join(" · ")} />
        <Chip icon={ICONS.dept} color="#22d3ee" label="Departments" value={p.departments.join(" · ")} />
        <Chip icon={ICONS.model} color="#f472b6" label="Model" value={p.model} />
        <Chip icon={ICONS.rows} color="#34d399" label="History" value={p.rows ? `${p.rows.toLocaleString()} daily rows` : "loading…"} />
      </div>

      <style>{`
        @keyframes hb-pulse { 0% { box-shadow: 0 0 0 0 rgba(52,211,153,.7); } 70% { box-shadow: 0 0 0 7px rgba(52,211,153,0); } 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); } }
        .hb-pulse { animation: hb-pulse 1.8s ease-out infinite; }
      `}</style>
    </header>
  );
}
