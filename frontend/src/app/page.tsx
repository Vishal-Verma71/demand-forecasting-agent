"use client";

import { useEffect, useState } from "react";
import { Table } from "@/components/Table";
import { LoadingStatus } from "@/components/Spinner";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Kpi = {
  rows: number;
  stores: number;
  departments: number;
  on_hand_units: number;
  at_risk_skus: number;
  draft_pos: number;
};

type RiskItem = {
  store: string;
  dept: string;
  days_of_supply: number;
  risk_level: "OK" | "WARNING" | "CRITICAL";
  on_hand: number;
  avg_daily_demand: number;
};

const STORES = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"];
const DEPTS = ["FOODS", "HOBBIES", "HOUSEHOLD"];

export default function Home() {
  const [tab, setTab] = useState("dashboard");
  const [store, setStore] = useState("CA_1");
  const [dept, setDept] = useState("FOODS");
  const [horizon, setHorizon] = useState(28);
  const [health, setHealth] = useState<any>(null);
  const [kpi, setKpi] = useState<Kpi | null>(null);
  const [riskItems, setRiskItems] = useState<RiskItem[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [series, setSeries] = useState<any[]>([]);
  const [forecast, setForecast] = useState<any>(null);
  const [trendCat, setTrendCat] = useState("FOODS");
  const [trend, setTrend] = useState<any>(null);
  const [reorders, setReorders] = useState<any[]>([]);
  const [pos, setPos] = useState<any[]>([]);
  const [reportMd, setReportMd] = useState("");
  const [agentQ, setAgentQ] = useState("");
  const [agentHistory, setAgentHistory] = useState<any[]>([]);
  const [agentLoading, setAgentLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [reordersLoading, setReordersLoading] = useState(false);
  const [poCreating, setPoCreating] = useState(false);
  const [riskScanLoading, setRiskScanLoading] = useState(false);
  const [riskScanMessage, setRiskScanMessage] = useState("");
  const [trendLoading, setTrendLoading] = useState(false);

  async function fetchAll() {
    setDashboardLoading(true);
    try {
      const [h, k, r, a] = await Promise.all([
        fetch(`${API}/health`).then((x) => x.json()),
        fetch(`${API}/kpi?horizon=${horizon}`).then((x) => x.json()),
        fetch(`${API}/dashboard/risk?horizon=${horizon}`).then((x) => x.json()),
        fetch(`${API}/alerts`).then((x) => x.json()),
      ]);
      setHealth(h);
      setKpi(k);
      setRiskItems(r.items || []);
      setAlerts(a.alerts || []);
    } finally {
      setDashboardLoading(false);
    }
  }

  async function loadSeries() {
    setForecastLoading(true);
    try {
      const res = await fetch(`${API}/series?store=${store}&dept=${dept}&limit=90`);
      const j = await res.json();
      setSeries(j.records || []);
    } finally {
      setForecastLoading(false);
    }
  }

  async function loadForecast() {
    setForecastLoading(true);
    try {
      const res = await fetch(`${API}/forecast?store=${store}&dept=${dept}&horizon=${horizon}`);
      const j = await res.json();
      setForecast(j);
    } finally {
      setForecastLoading(false);
    }
  }

  async function loadReorders() {
    setReordersLoading(true);
    try {
      const calls = STORES.flatMap((s) => DEPTS.map((d) => ({ s, d })));
      const res = await Promise.all(
        calls.map(({ s, d }) =>
          fetch(`${API}/reorder?store=${s}&dept=${d}&horizon=${horizon}`).then((x) => x.json())
        )
      );
      setReorders(res.map((x, i) => ({ ...x.recommendation, key: i })));
      const p = await fetch(`${API}/pos`).then((x) => x.json());
      setPos(p.pos || []);
    } finally {
      setReordersLoading(false);
    }
  }

  async function loadTrend(c = trendCat) {
    setTrendLoading(true);
    try {
      const res = await fetch(`${API}/trend?category=${c}`).then((x) => x.json());
      setTrend(res.trend);
    } finally {
      setTrendLoading(false);
    }
  }

  async function loadReport() {
    const res = await fetch(`${API}/report`).then((x) => x.json());
    setReportMd(res.markdown || "");
  }

  useEffect(() => {
    fetchAll();
  }, []);

  useEffect(() => {
    fetchAll();
  }, [horizon]);

  useEffect(() => {
    loadSeries();
    loadForecast();
  }, [store, dept, horizon]);

  useEffect(() => {
    if (tab === "reorders") loadReorders();
    if (tab === "report") loadReport();
    if (tab === "dashboard") fetchAll();
  }, [tab]);

  useEffect(() => {
    if (tab === "dashboard") loadTrend(trendCat);
  }, [trendCat, tab]);

  async function sendAgent() {
    if (!agentQ.trim()) return;
    const q = agentQ;
    setAgentQ("");
    setAgentLoading(true);
    setAgentHistory((prev) => [...prev, { role: "user", content: q }]);
    try {
      const res = await fetch(`${API}/agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const j = await res.json();
      setAgentHistory((prev) => [...prev, { role: "assistant", content: j.answer, steps: j.steps || [] }]);
    } catch (e: any) {
      setAgentHistory((prev) => [...prev, { role: "assistant", content: "Error: " + String(e) }]);
    } finally {
      setAgentLoading(false);
    }
  }

  async function runTrain() {
    setTraining(true);
    try {
      await fetch(`${API}/train`, { method: "POST" });
      await fetchAll();
    } finally {
      setTraining(false);
    }
  }

  async function createPo(s: string, d: string) {
    setPoCreating(true);
    try {
      await fetch(`${API}/reorder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ store: s, dept: d, horizon }),
      });
      await loadReorders();
    } finally {
      setPoCreating(false);
    }
  }

  async function riskScan() {
    setRiskScanLoading(true);
    setRiskScanMessage("");
    try {
      const res = await fetch(`${API}/risk-scan`, { method: "POST" });
      const j = await res.json();
      await fetchAll();
      setRiskScanMessage(`Scanned inventory risks: ${j.flagged ?? 0} risks detected.`);
    } catch (e) {
      setRiskScanMessage("Risk scan failed. Please retry.");
    } finally {
      setRiskScanLoading(false);
    }
  }

  const riskColumns = [
    { key: "label", header: "Store / Department", width: "2fr", render: (r: any) => `${r.store} / ${r.dept}` },
    { key: "days_of_supply", header: "Days of supply", width: "1fr", align: "center" as const },
    { key: "on_hand", header: "On hand", width: "1fr", align: "right" as const, render: (r: any) => Number(r.on_hand).toLocaleString() },
    { key: "risk_level", header: "Risk", width: "1fr", render: (r: any) => <span style={{ color: r.risk_level === "CRITICAL" ? "#ef4444" : r.risk_level === "WARNING" ? "#f59e0b" : "#22c55e", fontWeight: 700 }}>{r.risk_level}</span> },
  ];

  const forecastColumns = [
    { key: "Date", header: "Date", width: "2fr", render: (r: any) => String(r.Date).slice(0, 10) },
    { key: "yhat", header: "Forecast", width: "1fr", align: "right" as const, render: (r: any) => Number(r.yhat).toFixed(1) },
    { key: "yhat_lower", header: "Lower bound", width: "1fr", align: "right" as const, render: (r: any) => Number(r.yhat_lower).toFixed(1) },
    { key: "yhat_upper", header: "Upper bound", width: "1fr", align: "right" as const, render: (r: any) => Number(r.yhat_upper).toFixed(1) },
  ];

  const reorderColumns = [
    { key: "store", header: "Store", width: "1fr" },
    { key: "dept", header: "Department", width: "1fr" },
    { key: "recommended_qty", header: "Qty", width: "1fr", align: "right" as const, render: (r: any) => Number(r.recommended_qty).toLocaleString() },
    { key: "supplier", header: "Supplier", width: "1.5fr" },
    { key: "lead_time_days", header: "Lead days", width: "1fr", align: "center" as const },
    { key: "risk_level", header: "Risk", width: "1fr", render: (r: any) => <span style={{ color: r.risk_level === "CRITICAL" ? "#ef4444" : r.risk_level === "WARNING" ? "#f59e0b" : "#22c55e", fontWeight: 700 }}>{r.risk_level}</span> },
  ];

  const poColumns = [
    { key: "id", header: "ID", width: "0.6fr", align: "center" as const, render: (r: any) => `#${r.id}` },
    { key: "store", header: "Store", width: "1fr" },
    { key: "dept", header: "Dept", width: "1fr" },
    { key: "qty", header: "Qty", width: "1fr", align: "right" as const, render: (r: any) => Number(r.qty).toLocaleString() },
    { key: "supplier", header: "Supplier", width: "1.5fr" },
    { key: "lead_time_days", header: "Lead days", width: "1fr", align: "center" as const },
    { key: "status", header: "Status", width: "1fr" },
  ];

  const page: React.CSSProperties = { fontFamily: "Inter, system-ui, sans-serif", background: "#0b1220", color: "#e2e8f0", minHeight: "100vh", padding: 24 };
  const card: React.CSSProperties = { background: "#101826", border: "1px solid #1f2a3d", borderRadius: 16, padding: 18, boxShadow: "0 8px 24px rgba(0,0,0,.25)" };
  const btn: React.CSSProperties = { background: "#6366f1", color: "#fff", border: "none", borderRadius: 10, padding: "8px 14px", cursor: "pointer", fontWeight: 600 };
  const input: React.CSSProperties = { background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 10, padding: "8px 12px" };
  const thStyle: React.CSSProperties = { padding: 8, borderBottom: "1px solid #334155", color: "#94a3b8", fontWeight: 600, textAlign: "left", whiteSpace: "nowrap" };
  const tdStyle: React.CSSProperties = { padding: 8, borderTop: "1px solid #1f2a3d", verticalAlign: "top" };

  function renderMarkdown(md: string) {
    const lines = md.split("\\n");
    const out: React.ReactNode[] = [];
    let i = 0;
    let key = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.startsWith("# ")) {
        out.push(<h2 key={key++} style={{ color: "#e2e8f0", marginTop: 8 }}>{line.replace(/^#\\s*/, "")}</h2>);
        i++;
        continue;
      }
      if (line.startsWith("## ")) {
        out.push(<h3 key={key++} style={{ color: "#cbd5e1", marginTop: 14 }}>{line.replace(/^##\\s*/, "")}</h3>);
        i++;
        continue;
      }
      if (line.trim().startsWith("|")) {
        const rows: string[][] = [];
        while (i < lines.length && lines[i].trim().startsWith("|")) {
          rows.push(lines[i].trim().split("|").slice(1, -1).map((x) => x.trim()));
          i++;
        }
        // remove separator row
        const bodyRows = rows.filter((r) => !r.every((c) => /^[-:]+$/.test(c)));
        const header = bodyRows.length ? bodyRows[0] : [];
        const body = bodyRows.slice(1);
        out.push(
          <table key={key++} style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, margin: "8px 0", tableLayout: "fixed" }}>
            <thead>
              <tr style={{ background: "#1e293b" }}>
                {header.map((h, hi) => <th key={hi} style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #334155" }}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {body.map((row, ri) => (
                <tr key={ri} style={{ borderTop: "1px solid #1f2a3d" }}>
                  {row.map((cell, ci) => <td key={ci} style={{ padding: 8 }}>{cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        );
        continue;
      }
      if (line.trim().startsWith("- ")) {
        out.push(<div key={key++} style={{ marginLeft: 18, listStyle: "disc" }}>• {line.trim().slice(2)}</div>);
        i++;
        continue;
      }
      if (line.trim() === "") {
        i++;
        continue;
      }
      out.push(<p key={key++} style={{ color: "#cbd5e1", margin: "6px 0" }}>{line}</p>);
      i++;
    }
    return <div>{out}</div>;
  }

  return (
    <div style={page}>
      <div style={{ ...card, background: "radial-gradient(1200px 300px at 10% -20%, #312e81 0%, #0f172a 60%)", marginBottom: 18 }}>
        <h1 style={{ margin: 0, fontSize: 30, fontWeight: 800, background: "linear-gradient(90deg,#6366f1,#22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Demand Forecasting Agent
        </h1>
        <p style={{ color: "#cbd5e1", margin: "6px 0 12px" }}>
          Logistics supply chain intelligence · gradient boosting · ReAct agent · stockout risk & auto-reorder
        </p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span style={{ padding: "3px 10px", borderRadius: 999, background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}>Stores: {health?.stores ?? 5}</span>
          <span style={{ padding: "3px 10px", borderRadius: 999, background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}>Departments: {health?.departments ?? 3}</span>
          <span style={{ padding: "3px 10px", borderRadius: 999, background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}>Rows: {kpi?.rows ?? 0}</span>
          <span style={{ padding: "3px 10px", borderRadius: 999, background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}>Horizon: {horizon}d</span>
          <span style={{ padding: "3px 10px", borderRadius: 999, background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}>LLM: {health?.llm_backend ?? "none"} {health?.llm_available ? "✅" : "(offline)"}</span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 18 }}>
        <div style={{ ...card, height: "fit-content" }}>
          <h3 style={{ marginTop: 0 }}>Controls</h3>
          <div style={{ display: "grid", gap: 12 }}>
            <label>Store
              <select style={{ ...input, width: "100%", marginTop: 4 }} value={store} onChange={(e) => setStore(e.target.value)}>
                {STORES.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label>Department
              <select style={{ ...input, width: "100%", marginTop: 4 }} value={dept} onChange={(e) => setDept(e.target.value)}>
                {DEPTS.map((d) => <option key={d}>{d}</option>)}
              </select>
            </label>
            <label>Horizon (days): {horizon}
              <input type="range" min="7" max="56" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} style={{ width: "100%" }} />
            </label>
            {!health?.models_ready && (
              <button style={btn} onClick={runTrain} disabled={training}>{training ? "Training…" : "🚀 Train model"}</button>
            )}
            {health?.models_ready && <div style={{ color: "#22c55e", fontSize: 13 }}>Models ready ✅</div>}
            <div style={{ fontSize: 12, color: "#94a3b8" }}>Model: {health?.models_ready ? "" : "not trained"} {health?.llm_available ? "" : "offline router active"}</div>
          </div>
          <h3 style={{ marginTop: 20 }}>Tabs</h3>
          <div style={{ display: "grid", gap: 6 }}>
            {["dashboard", "agent", "forecast", "reorders", "report"].map((t) => (
              <button key={t} onClick={() => setTab(t)} style={{ ...btn, background: tab === t ? "#6366f1" : "#1f2937", textAlign: "left", textTransform: "capitalize" }}>{t}</button>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gap: 18 }}>
          {tab === "dashboard" && (
            <>
              {kpi && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
                  {[
                    { l: "Data rows", v: kpi.rows.toLocaleString(), s: `${kpi.stores} stores × ${kpi.departments} depts` },
                    { l: "On-hand units", v: kpi.on_hand_units.toLocaleString(undefined, { maximumFractionDigits: 0 }), s: "across all SKUs" },
                    { l: "At-risk SKUs", v: String(kpi.at_risk_skus), s: `horizon ${horizon}d` },
                    { l: "Draft POs", v: String(kpi.draft_pos), s: "auto-generated" },
                  ].map((c, i) => (
                    <div key={i} style={card}>
                      <div style={{ color: "#94a3b8", fontSize: 11, textTransform: "uppercase", letterSpacing: ".08em" }}>{c.l}</div>
                      <div style={{ fontSize: 26, fontWeight: 700, margin: "4px 0" }}>{c.v}</div>
                      <div style={{ color: "#64748b", fontSize: 12 }}>{c.s}</div>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 18 }}>
                <div style={card}>
                  <h3 style={{ marginTop: 0 }}>Inventory risk map</h3>
                  <LoadingStatus visible={dashboardLoading || riskScanLoading} message={riskScanLoading ? "Scanning inventory risks..." : "Loading inventory risk..."} />
                  <Table columns={riskColumns as any} rows={riskItems} loading={dashboardLoading || riskScanLoading} loadingMessage="Scanning inventory risks..." emptyMessage="No inventory data available." />
                  <button style={{ ...btn, marginTop: 10 }} onClick={riskScan} disabled={riskScanLoading}>{riskScanLoading ? "Scanning..." : "🔎 Run full risk scan"}</button>
                  {!riskScanLoading && riskScanMessage && <div style={{ color: riskScanMessage.includes("failed") ? "#ef4444" : "#22c55e", marginTop: 8 }}>{riskScanMessage}</div>}
                </div>
                <div style={card}>
                  <h3 style={{ marginTop: 0 }}>Risk feed</h3>
                  {riskItems.filter((r) => r.risk_level !== "OK").length === 0 && <div style={{ color: "#22c55e" }}>No at-risk SKUs 🎉</div>}
                  {riskItems.filter((r) => r.risk_level !== "OK").map((r) => (
                    <div key={r.store + r.dept} style={{ marginBottom: 8 }}>
                      {r.risk_level === "CRITICAL" ? "🔴" : "🟡"} <b>{r.store}/{r.dept}</b> — {r.days_of_supply} days supply
                    </div>
                  ))}
                  <h4 style={{ marginBottom: 6 }}>External trend</h4>
                  <select style={{ ...input, width: "100%", marginBottom: 8 }} value={trendCat} onChange={(e) => { setTrendCat(e.target.value); loadTrend(e.target.value); }} disabled={trendLoading}>
                    {DEPTS.map((d) => <option key={d}>{d}</option>)}
                  </select>
                  {trendLoading && <LoadingStatus visible message="Fetching external trend data..." />}
                  {trend && !trendLoading && (
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700 }}>{trend.direction} ({trend.delta_pct}%)</div>
                      <div style={{ color: "#94a3b8", fontSize: 12 }}>latest {trend.latest} · baseline {trend.baseline} · {trend.source}</div>
                      {String(trend.source).includes("fallback") && (
                        <div style={{ color: "#f59e0b", fontSize: 12, marginTop: 6 }}>Live external data unavailable; showing fallback. {String(trend.source)}</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              <div style={card}>
                <h3 style={{ marginTop: 0 }}>Recent demand — {store} / {dept}</h3>
                <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 8px" }}>Sales units over the last {series.length} days</p>
                <LoadingStatus visible={forecastLoading} message="Loading recent demand..." />
                {series.length === 0 ? (
                  <div style={{ color: "#64748b" }}>No series data available.</div>
                ) : (
                  <svg width="100%" height="260" viewBox="0 0 640 260" style={{ background: "#0b1220", borderRadius: 12, border: "1px solid #1f2a3d" }}>
                    {(() => {
                      const max = Math.max(...series.map((x: any) => Number(x.Sales) || 0), 1);
                      const left = 48, top = 18, bottom = 228, width = 560, height = bottom - top;
                      const barWidth = width / series.length;
                      const ticks = [0, max / 2, max];
                      return (
                        <>
                          {/* Y axis */}
                          <line x1={left} y1={top} x2={left} y2={bottom} stroke="#334155" />
                          {ticks.map((t, i) => (
                            <g key={i}>
                              <line x1={left - 4} y1={bottom - (t / max) * height} x2={left} y2={bottom - (t / max) * height} stroke="#334155" />
                              <text x={left - 8} y={bottom - (t / max) * height + 4} fill="#94a3b8" fontSize="10" textAnchor="end">{Math.round(t)}</text>
                            </g>
                          ))}
                          <text x="12" y={top + height / 2} fill="#94a3b8" fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${top + height / 2})`}>Sales units</text>
                          {/* Bars */}
                          {series.map((s: any, i: number) => {
                            const h = ((Number(s.Sales) || 0) / max) * height;
                            return <rect key={i} x={left + i * barWidth + 1} y={bottom - h} width={Math.max(1, barWidth - 2)} height={h} fill="#6366f1" rx="1"><title>{`${s.Date}: ${s.Sales}`}</title></rect>;
                          })}
                          {/* X axis */}
                          <line x1={left} y1={bottom} x2={left + width} y2={bottom} stroke="#334155" />
                          <text x={left} y={bottom + 16} fill="#94a3b8" fontSize="10">{String(series[0].Date).slice(0, 10)}</text>
                          <text x={left + width} y={bottom + 16} fill="#94a3b8" fontSize="10" textAnchor="end">{String(series[series.length - 1].Date).slice(0, 10)}</text>
                          <text x={left + width / 2} y={252} fill="#94a3b8" fontSize="11" textAnchor="middle">Recent daily demand</text>
                        </>
                      );
                    })()}
                  </svg>
                )}
              </div>
            </>
          )}

          {tab === "agent" && (
            <div style={card}>
              <h3 style={{ marginTop: 0 }}>Ask the agent</h3>
              <div style={{ display: "grid", gap: 10, marginBottom: 12 }}>
                {agentHistory.map((m, i) => (
                  <div key={i} style={{ background: m.role === "user" ? "#1e293b" : "#0b1220", padding: 12, borderRadius: 12, border: "1px solid #1f2a3d" }}>
                    <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 4 }}>{m.role}</div>
                    <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
                    {m.steps && (
                      <details style={{ marginTop: 8 }}>
                        <summary style={{ color: "#94a3b8", cursor: "pointer" }}>Reasoning trace ({m.steps.length} tool calls)</summary>
                        {m.steps.map((s: any, j: number) => (
                          <div key={j} style={{ fontFamily: "monospace", fontSize: 12, marginTop: 6, padding: 8, background: "#0f172a", borderLeft: "3px solid #6366f1", borderRadius: 6 }}>
                            <b>{s.tool}</b> {JSON.stringify(s.args)}<br />
                            <span style={{ color: "#94a3b8" }}>{String(s.observation).slice(0, 700)}</span>
                          </div>
                        ))}
                      </details>
                    )}
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <input style={{ ...input, flex: 1 }} placeholder="e.g. Forecast 28 days for TX_1 HOUSEHOLD and check risk" value={agentQ} onChange={(e) => setAgentQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendAgent()} disabled={agentLoading} />
                <button style={btn} onClick={sendAgent} disabled={agentLoading}>{agentLoading ? "Thinking…" : "Send"}</button>
              </div>
              <LoadingStatus visible={agentLoading} message="Running agent tools..." />
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                {["Forecast CA_1 FOODS 28 days", "Stockout risk for TX_2 HOBBIES", "Reorder WI_1 HOUSEHOLD", "Run a risk scan across all", "Detect demand spike CA_1 FOODS"].map((q) => (
                  <button key={q} onClick={() => { setAgentQ(q); sendAgent(); }} style={{ ...btn, background: "#1f2937", fontSize: 12 }}>{q}</button>
                ))}
              </div>
            </div>
          )}

          {tab === "forecast" && forecast && (
            <div style={card}>
              <h3 style={{ marginTop: 0 }}>Forecast · {store} / {dept} · {horizon} days</h3>
              {forecast.summary && (
                <div style={{ display: "flex", gap: 24, marginBottom: 12, color: "#cbd5e1" }}>
                  <span>Total: <b>{forecast.summary.total_demand}</b></span>
                  <span>Avg daily: <b>{forecast.summary.avg_daily}</b></span>
                  <span>Range: {forecast.summary.first_day} → {forecast.summary.last_day}</span>
                  {forecast.metrics && <span>MAPE {forecast.metrics.MAPE}% · MAE {forecast.metrics.MAE}</span>}
                </div>
              )}
              <svg width="100%" height="280" viewBox="0 0 640 280" style={{ background: "#0b1220", borderRadius: 12, border: "1px solid #1f2a3d", marginBottom: 12 }}>
                {(() => {
                  const recs = forecast.records || [];
                  const max = Math.max(...recs.map((x: any) => Math.max(Number(x.yhat) || 0, Number(x.yhat_upper) || 0)), 1);
                  const left = 48, top = 18, bottom = 238, width = 560, height = bottom - top;
                  const barWidth = width / (recs.length || 1);
                  const ticks = [0, max / 2, max];
                  return (
                    <>
                      {/* Y axis */}
                      <line x1={left} y1={top} x2={left} y2={bottom} stroke="#334155" />
                      {ticks.map((t, i) => (
                        <g key={i}>
                          <line x1={left - 4} y1={bottom - (t / max) * height} x2={left} y2={bottom - (t / max) * height} stroke="#334155" />
                          <text x={left - 8} y={bottom - (t / max) * height + 4} fill="#94a3b8" fontSize="10" textAnchor="end">{Math.round(t)}</text>
                        </g>
                      ))}
                      <text x="12" y={top + height / 2} fill="#94a3b8" fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${top + height / 2})`}>Forecast units</text>
                      {/* Forecast bars */}
                      {recs.map((r: any, i: number) => {
                        const h = ((Number(r.yhat) || 0) / max) * height;
                        const uh = ((Number(r.yhat_upper) || 0) / max) * height;
                        return (
                          <g key={i}>
                            <rect x={left + i * barWidth + 1} y={bottom - uh} width={Math.max(1, barWidth - 2)} height={Math.max(0, uh - h)} fill="rgba(99,102,241,.18)" rx="1" />
                            <rect x={left + i * barWidth + 1} y={bottom - h} width={Math.max(1, barWidth - 2)} height={h} fill="#6366f1" rx="1"><title>{`${r.Date}: forecast ${r.yhat}, lower ${r.yhat_lower}, upper ${r.yhat_upper}`}</title></rect>
                          </g>
                        );
                      })}
                      {/* X axis */}
                      <line x1={left} y1={bottom} x2={left + width} y2={bottom} stroke="#334155" />
                      <text x={left} y={bottom + 16} fill="#94a3b8" fontSize="10">{String(recs[0]?.Date).slice(0, 10)}</text>
                      <text x={left + width} y={bottom + 16} fill="#94a3b8" fontSize="10" textAnchor="end">{String(recs[recs.length - 1]?.Date).slice(0, 10)}</text>
                      <text x={left + width / 2} y={268} fill="#94a3b8" fontSize="11" textAnchor="middle">Forecast horizon</text>
                    </>
                  );
                })()}
              </svg>
              <LoadingStatus visible={forecastLoading} message="Generating demand forecast..." />
              <Table columns={forecastColumns as any} rows={forecast.records || []} loading={forecastLoading} loadingMessage="Generating demand forecast..." emptyMessage="No forecast available." />
            </div>
          )}

          {tab === "reorders" && (
            <div style={card}>
              <h3 style={{ marginTop: 0 }}>Reorder recommendations</h3>
              <LoadingStatus visible={reordersLoading} message="Calculating reorder recommendations..." />
              <Table columns={reorderColumns as any} rows={reorders} loading={reordersLoading} loadingMessage="Calculating reorder recommendations..." emptyMessage="No reorder data available." />
              <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                <select style={input as any} value={store} onChange={(e) => setStore(e.target.value)}>
                  {STORES.map((s) => <option key={s}>{s}</option>)}
                </select>
                <select style={input as any} value={dept} onChange={(e) => setDept(e.target.value)}>
                  {DEPTS.map((d) => <option key={d}>{d}</option>)}
                </select>
                <button style={{ ...btn, opacity: poCreating ? 0.6 : 1 }} onClick={() => createPo(store, dept)} disabled={poCreating}>{poCreating ? "Creating..." : "✅ Create draft PO"}</button>
              </div>
              <h4>Draft POs</h4>
              <LoadingStatus visible={poCreating} message="Creating draft purchase order..." />
              <Table columns={poColumns as any} rows={pos} loading={poCreating} loadingMessage="Loading draft purchase orders..." emptyMessage="No draft purchase orders yet." />
            </div>
          )}

          {tab === "report" && (
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <h3 style={{ margin: 0 }}>Auto-generated report</h3>
                <button style={btn} onClick={() => {
                  const blob = new Blob([reportMd], { type: "text/markdown" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url; a.download = "supply_chain_report.md"; a.click();
                  URL.revokeObjectURL(url);
                }}>⬇ Download Markdown</button>
              </div>
              <div style={{ background: "#0b1220", padding: 14, borderRadius: 12, border: "1px solid #1f2a3d", fontSize: 13, maxHeight: 600, overflow: "auto" }}>
                {renderMarkdown(reportMd)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}