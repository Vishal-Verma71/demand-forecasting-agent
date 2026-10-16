"use client";

import { useEffect, useRef, useState } from "react";
import { Table } from "@/components/Table";
import { LoadingStatus, Spinner } from "@/components/Spinner";
import { HeaderBanner } from "@/components/HeaderBanner";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Kpi = {
  rows: number;
  stores: number;
  departments: number;
  on_hand_units: number;
  inbound_units: number;
  at_risk_combinations: number;
  critical_combinations: number;
  draft_pos: number;
};

type RiskItem = {
  store: string;
  dept: string;
  days_of_supply: number;
  risk_level: "OK" | "WARNING" | "CRITICAL" | "UNKNOWN";
  on_hand: number;
  on_order: number;
  reorder_point: number;
  projected_stockout_date: string | null;
  protection_days: number;
  avg_daily_demand: number;
};

const riskColor = (level: string) =>
  level === "CRITICAL" ? "#ef4444" : level === "WARNING" ? "#f59e0b" : level === "OK" ? "#22c55e" : "#94a3b8";

const fmt = (n: any) => Number(n ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 });

const STORES = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"];
const DEPTS = ["FOODS", "HOBBIES", "HOUSEHOLD"];

export default function Home() {
  const [tab, setTab] = useState("dashboard");
  const [store, setStore] = useState("CA_1");
  const [dept, setDept] = useState("FOODS");
  const [horizon, setHorizon] = useState(28);
  // The slider moves horizonDraft instantly; horizon (which triggers requests) follows once the user pauses.
  const [horizonDraft, setHorizonDraft] = useState(28);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const forecastRequestRef = useRef(0);
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
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");
  const [poStoreFilter, setPoStoreFilter] = useState("ALL");
  const [poDeptFilter, setPoDeptFilter] = useState("ALL");
  const [poStatusFilter, setPoStatusFilter] = useState("ALL");
  const [evalData, setEvalData] = useState<any>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [selectedRecentSales, setSelectedRecentSales] = useState<any>(null);
  const [selectedRecentDetailPos, setSelectedRecentDetailPos] = useState({ left: 12, top: 12 });
  const [selectedForecastPoint, setSelectedForecastPoint] = useState<any>(null);
  const [selectedForecastDetailPos, setSelectedForecastDetailPos] = useState({ left: 12, top: 12 });
  const [evalRunning, setEvalRunning] = useState(false);
  const [agentQ, setAgentQ] = useState("");
  const [agentHistory, setAgentHistory] = useState<any[]>([]);
  const [agentLoading, setAgentLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [reordersLoading, setReordersLoading] = useState(false);
  const [poCreating, setPoCreating] = useState(false);
  const [poMessage, setPoMessage] = useState("");
  const [riskScanLoading, setRiskScanLoading] = useState(false);
  const [riskScanMessage, setRiskScanMessage] = useState("");
  const [trendLoading, setTrendLoading] = useState(false);
  const dashboardRequestRef = useRef(0);

  async function fetchStatic() {
    try {
      const [h, a] = await Promise.all([
        fetch(`${API}/health`).then((x) => x.json()),
        fetch(`${API}/alerts`).then((x) => x.json()),
      ]);
      setHealth(h);
      setAlerts(a.alerts || []);
    } catch {
      /* Static dashboard data is optional after the first render. */
    }
  }

  async function fetchDashboard(horizonValue = horizon) {
    const runId = ++dashboardRequestRef.current;
    setDashboardLoading(true);
    try {
      const [k, r] = await Promise.all([
        fetch(`${API}/kpi?horizon=${horizonValue}`).then((x) => x.json()),
        fetch(`${API}/dashboard/risk?horizon=${horizonValue}`).then((x) => x.json()),
      ]);
      if (runId !== dashboardRequestRef.current) return;
      setKpi(k);
      setRiskItems(r.items || []);
    } catch {
      if (runId === dashboardRequestRef.current) setDashboardLoading(false);
    } finally {
      if (runId === dashboardRequestRef.current) setDashboardLoading(false);
    }
  }

  async function loadSeries() {
    setSeriesLoading(true);
    try {
      const res = await fetch(`${API}/series?store=${store}&dept=${dept}&limit=90`);
      const j = await res.json();
      setSeries(j.records || []);
      setSelectedRecentSales(null);
    } finally {
      setSeriesLoading(false);
    }
  }

  async function loadForecast() {
    // Only the newest request may update the chart; slower, older responses are dropped.
    const runId = ++forecastRequestRef.current;
    setForecastLoading(true);
    try {
      const res = await fetch(`${API}/forecast?store=${store}&dept=${dept}&horizon=${horizon}`);
      const j = await res.json();
      if (runId !== forecastRequestRef.current) return;
      setForecast(j);
      setSelectedForecastPoint(null);
    } finally {
      if (runId === forecastRequestRef.current) setForecastLoading(false);
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
    setReportLoading(true);
    setReportError("");
    try {
      const res = await fetch(`${API}/report`);
      if (!res.ok) throw new Error(`report request failed (${res.status})`);
      const j = await res.json();
      setReportMd(j.markdown || "");
    } catch (e: any) {
      setReportError(String(e.message || e));
    } finally {
      setReportLoading(false);
    }
  }

  async function loadEval() {
    setEvalLoading(true);
    try {
      const res = await fetch(`${API}/evaluation`).then((x) => x.json());
      setEvalData(res?.metrics ?? null);
    } finally {
      setEvalLoading(false);
    }
  }

  async function runEvaluation() {
    setEvalRunning(true);
    try {
      const res = await fetch(`${API}/evaluation/run`, { method: "POST" });
      const j = await res.json();
      setEvalData(j?.metrics ?? null);
    } catch {
      setEvalData(null);
    } finally {
      setEvalRunning(false);
    }
  }

  useEffect(() => {
    fetchStatic();
  }, []);

  useEffect(() => {
    if (horizonDraft === horizon) return;
    const t = setTimeout(() => setHorizon(horizonDraft), 350);
    return () => clearTimeout(t);
  }, [horizonDraft]);

  useEffect(() => {
    fetchDashboard(horizon);
  }, [horizon]);

  useEffect(() => {
    loadSeries();
  }, [store, dept]);

  useEffect(() => {
    loadForecast();
  }, [store, dept, horizon]);

  useEffect(() => {
    if (tab === "reorders") loadReorders();
    if (tab === "report") {
      loadReport();
      loadEval();
    }
    if (tab === "dashboard") fetchDashboard();
  }, [tab]);

  useEffect(() => {
    if (tab === "dashboard") loadTrend(trendCat);
  }, [trendCat, tab]);

  async function sendAgent(text?: string) {
    // Suggestion chips pass their text directly: reading agentQ right after setAgentQ would see the old value.
    const q = (text ?? agentQ).trim();
    if (!q) return;
    setAgentQ("");
    setAgentLoading(true);
    setAgentHistory((prev) => [...prev, { role: "user", content: q }]);
    try {
      const res = await fetch(`${API}/agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, store, dept }),
      });
      const j = await res.json();
      setAgentHistory((prev) => [...prev, { role: "assistant", content: j.answer, steps: j.steps || [], backend: j.backend }]);
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
      await fetchStatic();
      await fetchDashboard(horizon);
    } finally {
      setTraining(false);
    }
  }

  async function createPo(s: string, d: string) {
    setPoCreating(true);
    setPoMessage("");
    try {
      const res = await fetch(`${API}/reorder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ store: s, dept: d, horizon }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || "Failed to create purchase order");
      const r = j.result;
      setPoMessage(r.po_ids?.length
        ? `${s}/${d}: ${r.status} — PO #${r.po_ids.join(", #")}`
        : `${s}/${d}: no order needed (${r.risk_level}).`);
      await loadReorders();
    } catch (e: any) {
      setPoMessage(String(e.message || e));
    } finally {
      setPoCreating(false);
    }
  }

  async function decidePo(id: number, action: "approve" | "reject") {
    const res = await fetch(`${API}/pos/${id}/${action}`, { method: "POST" });
    const j = await res.json();
    setPoMessage(res.ok ? `PO #${id} ${j.po.status}.` : String(j.detail));
    await loadReorders();
    fetchDashboard(horizon);
  }

  async function riskScan() {
    setRiskScanLoading(true);
    setRiskScanMessage("");
    try {
      const res = await fetch(`${API}/risk-scan?horizon=${horizon}`, { method: "POST" });
      const j = await res.json();
      await fetchDashboard(horizon);
      setRiskScanMessage(`Scanned inventory risks: ${j.flagged ?? 0} risks detected.`);
    } catch (e) {
      setRiskScanMessage("Risk scan failed. Please retry.");
    } finally {
      setRiskScanLoading(false);
    }
  }

  const forecastPending = forecastLoading || horizonDraft !== horizon;

  const riskBadge = (r: any) => <span style={{ color: riskColor(r.risk_level), fontWeight: 700 }}>{r.risk_level}</span>;

  const riskColumns = [
    { key: "label", header: "Store / Dept", width: "1.6fr", render: (r: any) => `${r.store} / ${r.dept}` },
    { key: "on_hand", header: "On hand", width: "1fr", align: "right" as const, render: (r: any) => fmt(r.on_hand) },
    { key: "on_order", header: "Inbound", width: "1fr", align: "right" as const, render: (r: any) => fmt(r.on_order) },
    { key: "reorder_point", header: "Reorder pt", width: "1fr", align: "right" as const, render: (r: any) => fmt(r.reorder_point) },
    { key: "days_of_supply", header: "Cover / needed (d)", width: "1.1fr", align: "center" as const, render: (r: any) => `${r.days_of_supply} / ${r.protection_days}` },
    { key: "projected_stockout_date", header: "Runs out if no delivery", width: "1.2fr", align: "center" as const, render: (r: any) => r.projected_stockout_date ?? "—" },
    { key: "risk_level", header: "Risk", width: "1fr", render: riskBadge },
  ];

  const forecastColumns = [
    { key: "Date", header: "Date", width: "2fr", render: (r: any) => String(r.Date).slice(0, 10) },
    { key: "yhat", header: "Forecast", width: "1fr", align: "right" as const, render: (r: any) => Number(r.yhat).toFixed(1) },
    { key: "yhat_lower", header: "Lower bound", width: "1fr", align: "right" as const, render: (r: any) => Number(r.yhat_lower).toFixed(1) },
    { key: "yhat_upper", header: "Upper bound", width: "1fr", align: "right" as const, render: (r: any) => Number(r.yhat_upper).toFixed(1) },
  ];

  const reorderColumns = [
    { key: "label", header: "Store / Dept", width: "1.4fr", render: (r: any) => `${r.store} / ${r.dept}` },
    { key: "risk_level", header: "Risk", width: "0.9fr", render: riskBadge },
    { key: "recommended_qty", header: "Order qty", width: "0.9fr", align: "right" as const, render: (r: any) => fmt(r.recommended_qty) },
    {
      key: "allocations", header: "Supplier split", width: "2fr",
      render: (r: any) => r.allocations?.length
        ? r.allocations.map((a: any) => <div key={a.supplier}>{fmt(a.qty)} × {a.supplier} <span style={{ color: "#94a3b8" }}>({a.lead_time_days}d)</span></div>)
        : <span style={{ color: "#64748b" }}>—</span>,
    },
    { key: "rationale", header: "Why", width: "3fr", render: (r: any) => <span style={{ color: "#94a3b8", fontSize: 12 }}>{r.rationale}</span> },
    {
      key: "action", header: "", width: "1fr",
      render: (r: any) => r.recommended_qty > 0
        ? <button style={{ ...btn, padding: "6px 10px", fontSize: 12 }} disabled={poCreating} onClick={() => createPo(r.store, r.dept)}>Draft PO</button>
        : null,
    },
  ];

  const poColumns = [
    { key: "id", header: "ID", width: "0.6fr", align: "center" as const, render: (r: any) => `#${r.id}` },
    { key: "label", header: "Store / Dept", width: "1.4fr", render: (r: any) => `${r.store} / ${r.dept}` },
    { key: "qty", header: "Qty", width: "0.8fr", align: "right" as const, render: (r: any) => fmt(r.qty) },
    { key: "supplier", header: "Supplier", width: "1.2fr" },
    { key: "lead_time_days", header: "Lead", width: "0.6fr", align: "center" as const, render: (r: any) => `${r.lead_time_days}d` },
    { key: "reason", header: "Reason", width: "2fr", render: (r: any) => <span style={{ color: "#94a3b8", fontSize: 12 }}>{r.reason ?? ""}</span> },
    { key: "status", header: "Status", width: "1fr", render: (r: any) => <span style={{ color: r.status === "APPROVED" ? "#22c55e" : r.status === "REJECTED" ? "#ef4444" : "#f59e0b", fontWeight: 700 }}>{r.status}</span> },
    {
      key: "decide", header: "Decision", width: "1.6fr",
      render: (r: any) => r.status === "DRAFT" ? (
        <div style={{ display: "flex", gap: 6 }}>
          <button style={{ ...btn, padding: "5px 9px", fontSize: 12, background: "#15803d" }} onClick={() => decidePo(r.id, "approve")}>Approve</button>
          <button style={{ ...btn, padding: "5px 9px", fontSize: 12, background: "#3f3f46" }} onClick={() => decidePo(r.id, "reject")}>Reject</button>
        </div>
      ) : null,
    },
  ];

  const page: React.CSSProperties = { fontFamily: "Inter, system-ui, sans-serif", background: "radial-gradient(circle at top left, #312e81 0%, #111827 28%, #0f172a 58%, #0b1020 100%)", color: "#e2e8f0", minHeight: "100vh", padding: 24 };
  const card: React.CSSProperties = { background: "linear-gradient(180deg, rgba(17,24,39,.95), rgba(15,23,42,.95))", border: "1px solid #334155", borderRadius: 16, padding: 18, boxShadow: "0 10px 28px rgba(0,0,0,.35)" };
  const btn: React.CSSProperties = { background: "linear-gradient(135deg,#7C3AED,#06B6D4)", color: "#fff", border: "none", borderRadius: 10, padding: "8px 14px", cursor: "pointer", fontWeight: 600 };
  const input: React.CSSProperties = { background: "#111827", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 10, padding: "8px 12px" };
  const thStyle: React.CSSProperties = { padding: 8, borderBottom: "1px solid #334155", color: "#94a3b8", fontWeight: 600, textAlign: "left", whiteSpace: "nowrap" };
  const tdStyle: React.CSSProperties = { padding: 8, borderTop: "1px solid #1f2a3d", verticalAlign: "top" };

  function renderMarkdown(md: string) {
    const lines = md.split("\n");
    const out: React.ReactNode[] = [];
    let i = 0;
    let key = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.startsWith("# ")) {
        out.push(<h2 key={key++} style={{ color: "#e2e8f0", marginTop: 8 }}>{line.replace(/^#\s*/, "")}</h2>);
        i++;
        continue;
      }
      if (line.startsWith("## ")) {
        out.push(<h3 key={key++} style={{ color: "#cbd5e1", marginTop: 14 }}>{line.replace(/^##\s*/, "")}</h3>);
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
      out.push(<p key={key++} style={{ color: "#cbd5e1", margin: "6px 0" }}>{line.replace(/\*\*/g, "")}</p>);
      i++;
    }
    return <div>{out}</div>;
  }

  return (
    <div style={page}>
      <HeaderBanner
        stores={health?.stores ?? STORES}
        departments={health?.departments ?? DEPTS}
        model={health?.model ?? "LightGBM"}
        rows={kpi?.rows ?? 0}
        horizon={horizon}
        dataThrough={series.length ? String(series[series.length - 1].Date).slice(0, 10) : undefined}
        modelReady={!!health?.models_ready}
        llm={!!health?.llm_available}
        planner={health?.llm_available ? `LLM · ${health.llm_backend}` : "Offline"}
      />

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
            <label>Horizon (days): {horizonDraft}
              {(horizonDraft !== horizon || forecastLoading) && <span style={{ color: "#a5b4fc", fontSize: 12, marginLeft: 6 }}>updating…</span>}
              <input type="range" min="7" max="56" value={horizonDraft} onChange={(e) => setHorizonDraft(Number(e.target.value))} style={{ width: "100%" }} />
            </label>
            {!health?.models_ready && (
              <button style={btn} onClick={runTrain} disabled={training}>{training ? "Training…" : "🚀 Train model"}</button>
            )}
            {health?.models_ready && <div style={{ color: "#22c55e", fontSize: 13 }}>Model ready ✅</div>}
            {!health?.llm_available && <div style={{ fontSize: 12, color: "#94a3b8" }}>No LLM key set: the offline planner runs the same tools. Add GROQ_API_KEY to .env for LLM planning.</div>}
          </div>
          <h3 style={{ marginTop: 20 }}>Tabs</h3>
          <div style={{ display: "grid", gap: 6 }}>
            {["dashboard", "agent", "forecast", "reorders", "report"].map((t) => (
              <button key={t} onClick={() => setTab(t)} style={{ ...btn, background: tab === t ? "linear-gradient(135deg,#7C3AED,#A78BFA)" : "#18181B", textAlign: "left", textTransform: "capitalize", border: tab === t ? "1px solid #8B5CF6" : "1px solid #3F3F46" }}>{t}</button>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gap: 18 }}>
          {tab === "dashboard" && (
            <>
              {kpi && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
                  {[
                    { l: "Sales history", v: kpi.rows.toLocaleString(), s: `daily rows · ${kpi.stores} stores × ${kpi.departments} depts`, accent: "#8B5CF6" },
                    { l: "Stock on hand", v: fmt(kpi.on_hand_units), s: `+ ${fmt(kpi.inbound_units)} units inbound`, accent: "#14B8A6" },
                    { l: "At-risk series", v: `${kpi.at_risk_combinations} / 15`, s: `${kpi.critical_combinations} critical · store/department level`, accent: "#F59E0B" },
                    { l: "POs awaiting approval", v: String(kpi.draft_pos), s: "drafted by the agent", accent: "#F43F5E" },
                  ].map((c, i) => (
                      <div key={i} style={{ ...card, borderTop: `4px solid ${c.accent}`, background: `linear-gradient(180deg, #111827, #0f172a)` }}>
                        <div style={{ color: c.accent, fontSize: 11, textTransform: "uppercase", letterSpacing: ".08em", fontWeight: 700 }}>{c.l}</div>
                        <div style={{ fontSize: 28, fontWeight: 800, margin: "6px 0", color: "#E2E8F0" }}>{c.v}</div>
                        <div style={{ color: "#94A3B8", fontSize: 12 }}>{c.s}</div>
                      </div>
                    ))}
                </div>
              )}
              <div style={card}>
                  <h3 style={{ marginTop: 0 }}>Inventory risk map</h3>
                  <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 10px" }}>
                    Stock on hand + inbound vs LightGBM-forecast demand until the next scheduled delivery (days to next weekly review + supplier lead time) plus safety stock.
                  </p>
                  <LoadingStatus visible={dashboardLoading || riskScanLoading} message={riskScanLoading ? "Scanning inventory risks..." : "Loading inventory risk..."} />
                  <Table columns={riskColumns as any} rows={riskItems} loading={dashboardLoading || riskScanLoading} loadingMessage="Scanning inventory risks..." emptyMessage="No inventory data available." />
                  <button style={{ ...btn, marginTop: 10 }} onClick={riskScan} disabled={riskScanLoading}>{riskScanLoading ? "Scanning..." : "🔎 Run full risk scan"}</button>
                  {!riskScanLoading && riskScanMessage && <div style={{ color: riskScanMessage.includes("failed") ? "#ef4444" : "#22c55e", marginTop: 8 }}>{riskScanMessage}</div>}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 18 }}>
                <div style={card}>
                  <h3 style={{ marginTop: 0 }}>Risk feed</h3>
                  {riskItems.filter((r) => r.risk_level !== "OK").length === 0 && <div style={{ color: "#22c55e" }}>No series at risk 🎉</div>}
                  {[...riskItems].filter((r) => r.risk_level !== "OK")
                    .sort((a, b) => (a.risk_level === "CRITICAL" ? 0 : 1) - (b.risk_level === "CRITICAL" ? 0 : 1))
                    .map((r) => (
                    <div key={r.store + r.dept} style={{ marginBottom: 8, fontSize: 14 }}>
                      {r.risk_level === "CRITICAL" ? "🔴" : "🟡"} <b>{r.store}/{r.dept}</b> — {r.days_of_supply} days cover, needs {r.protection_days}
                      {r.projected_stockout_date && <span style={{ color: "#94a3b8" }}> · runs out {r.projected_stockout_date} without a delivery</span>}
                    </div>
                  ))}
                  {riskItems.some((r) => r.risk_level !== "OK") && (
                    <button style={{ ...btn, fontSize: 12, marginTop: 4 }} onClick={() => { setTab("agent"); sendAgent("Plan replenishment"); }}>
                      🤖 Ask the agent to plan replenishment
                    </button>
                  )}
                </div>
                <div style={card}>
                  <h3 style={{ marginTop: 0 }}>External demand signal</h3>
                  <select style={{ ...input, width: "100%", marginBottom: 8 }} value={trendCat} onChange={(e) => { setTrendCat(e.target.value); loadTrend(e.target.value); }} disabled={trendLoading}>
                    {DEPTS.map((d) => <option key={d}>{d}</option>)}
                  </select>
                  {trendLoading && <LoadingStatus visible message="Fetching external trend data..." />}
                  {trend && !trendLoading && (trend.delta_pct === null ? (
                    <div style={{ color: "#f59e0b", fontSize: 13 }}>{trend.source}</div>
                  ) : (
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700 }}>
                        {trend.direction === "up" ? "▲" : trend.direction === "down" ? "▼" : "▬"} {trend.direction} ({trend.delta_pct > 0 ? "+" : ""}{trend.delta_pct}%)
                      </div>
                      <div style={{ color: "#94a3b8", fontSize: 12 }}>“{trend.query}”: last 4 weeks {trend.latest} vs 12-month avg {trend.baseline} · {trend.source}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div style={card}>
                <h3 style={{ marginTop: 0 }}>Recent demand — {store} / {dept}</h3>
                <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 8px" }}>Sales units over the last {series.length} days</p>
                                <LoadingStatus visible={seriesLoading} message="Loading recent demand..." />
                {series.length === 0 ? (
                  <div style={{ color: "#64748b" }}>No series data available.</div>
                ) : (
                  <div style={{ position: "relative" }}>
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
                            const selected = selectedRecentSales?.Date === s.Date;
                            const recentColor = Number(s.Sales || 0) > max * .75 ? "#22C55E" : Number(s.Sales || 0) > max * .4 ? "#06B6D4" : "#F59E0B";
                            const detailLeft = Math.max(4, Math.min(68, (((48 + i * barWidth + barWidth / 2) / 640) * 100) - 16));
                            const detailTop = Math.max(8, Math.min(62, (((bottom - h) / 260) * 100) - 18));
                            const handleRecentClick = () => {
                              setSelectedRecentSales(s);
                              setSelectedRecentDetailPos({ left: detailLeft, top: detailTop });
                            };
                            return <rect key={i} x={left + i * barWidth + 1} y={bottom - h} width={Math.max(1, barWidth - 2)} height={h} fill={selected ? "#EC4899" : recentColor} rx="2" style={{ cursor: "pointer", stroke: selected ? "#F8FAFC" : "none", strokeWidth: 2 }} onClick={handleRecentClick}><title>{`${s.Date}: sales ${s.Sales}`}</title></rect>;
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
                {selectedRecentSales && (
                  <div style={{ position: "absolute", top: selectedRecentDetailPos.top + "%", left: selectedRecentDetailPos.left + "%", width: 220, padding: 10, borderRadius: 10, border: "1px solid #A78BFA", background: "rgba(15,23,42,.96)", color: "#E2E8F0", fontSize: 12, boxShadow: "0 12px 32px rgba(0,0,0,.45)", zIndex: 20, backdropFilter: "blur(8px)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
                      <b style={{ color: "#06B6D4" }}>Selected daily demand</b>
                      <button style={{ border: "none", background: "transparent", color: "#94A3B8", cursor: "pointer" }} onClick={() => setSelectedRecentSales(null)}>×</button>
                    </div>
                    <div><b>Date:</b> {String(selectedRecentSales.Date).slice(0, 10)}</div>
                    <div><b>Actual sales:</b> {Number(selectedRecentSales.Sales).toLocaleString()}</div>
                    <div style={{ color: "#94A3B8", marginTop: 6 }}>Click another bar to update details.</div>
                  </div>
                )}
                  </div>
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
                    <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 4 }}>{m.role === "user" ? "you" : `agent · ${m.backend ?? ""}`}</div>
                    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{m.content}</div>
                    {m.steps && m.steps.length > 0 && (
                      <details style={{ marginTop: 8 }} open={m.steps.length > 3 && i === agentHistory.length - 1}>
                        <summary style={{ color: "#94a3b8", cursor: "pointer" }}>Reasoning trace ({m.steps.length} tool calls)</summary>
                        {m.steps.map((s: any, j: number) => (
                          <div key={j} style={{ fontSize: 12, marginTop: 6, padding: 8, background: "#0f172a", borderLeft: `3px solid ${s.observation?.error ? "#ef4444" : "#6366f1"}`, borderRadius: 6 }}>
                            <div style={{ color: "#cbd5e1" }}>{j + 1}. {s.thought}</div>
                            <div style={{ fontFamily: "monospace", marginTop: 2 }}><b style={{ color: "#a78bfa" }}>{s.tool}</b>({Object.entries(s.args || {}).map(([k, v]) => `${k}=${v}`).join(", ")})</div>
                            <details>
                              <summary style={{ color: "#64748b", cursor: "pointer" }}>observation</summary>
                              <pre style={{ color: "#94a3b8", whiteSpace: "pre-wrap", maxHeight: 220, overflow: "auto", margin: 0 }}>{JSON.stringify(s.observation, null, 1).slice(0, 3000)}</pre>
                            </details>
                          </div>
                        ))}
                      </details>
                    )}
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <input style={{ ...input, flex: 1 }} placeholder="e.g. Plan replenishment, or: forecast TX_1 HOUSEHOLD for the next 2 weeks" value={agentQ} onChange={(e) => setAgentQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendAgent()} disabled={agentLoading} />
                <button style={btn} onClick={() => sendAgent()} disabled={agentLoading}>{agentLoading ? "Thinking…" : "Send"}</button>
              </div>
              <LoadingStatus visible={agentLoading} message="Running agent tools..." />
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                {["Plan replenishment", "Which series are at risk?", "Forecast CA_1 FOODS for 14 days", "Stockout risk for CA_1 HOBBIES", "Reorder WI_1 HOBBIES", "Upcoming promotions for TX_2", "External trends for HOBBIES", "How many critical items?"].map((q) => (
                  <button key={q} onClick={() => sendAgent(q)} disabled={agentLoading} style={{ ...btn, background: "#1f2937", fontSize: 12 }}>{q}</button>
                ))}
              </div>
            </div>
          )}

          {tab === "forecast" && forecast && (
            <div style={card}>
              <h3 style={{ marginTop: 0, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                Forecast · {store} / {dept} · {forecastPending ? horizonDraft : horizon} days
                {forecastPending && (
                  <span role="status" style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "4px 10px", borderRadius: 999, fontSize: 12, fontWeight: 600, color: "#c7d2fe", background: "rgba(99,102,241,.15)", border: "1px solid rgba(129,140,248,.45)" }}>
                    <Spinner size={12} /> Updating forecast…
                  </span>
                )}
              </h3>
              {forecast.summary && (
                <div style={{ display: "flex", gap: 24, marginBottom: 12, color: "#cbd5e1" }}>
                  <span>Total: <b>{forecast.summary.total_demand}</b></span>
                  <span>Avg daily: <b>{forecast.summary.avg_daily}</b></span>
                  <span>Range: {forecast.summary.first_day} → {forecast.summary.last_day}</span>
                  {forecast.metrics?.MAE !== undefined && (
                    <span title={forecast.metrics.source}>
                      Backtest MAPE {Number(forecast.metrics.MAPE).toFixed(1)}% · MAE {Number(forecast.metrics.MAE).toFixed(0)}
                      {forecast.metrics.seasonal_naive_MAE !== undefined && <span style={{ color: "#94a3b8" }}> (seasonal-naive MAE {Number(forecast.metrics.seasonal_naive_MAE).toFixed(0)})</span>}
                    </span>
                  )}
                </div>
              )}
              <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 8 }}>Shaded band: {forecast.summary?.interval}. SNAP promotion days and M5 events are model inputs.</div>
              <div style={{ position: "relative" }}>
              {forecastPending && (
                <div role="status" aria-live="polite" style={{ position: "absolute", inset: 0, marginBottom: 12, zIndex: 30, borderRadius: 12, display: "grid", placeItems: "center", background: "rgba(11,18,32,.72)", backdropFilter: "blur(2px)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 18px", borderRadius: 12, background: "#111827", border: "1px solid #4f46e5", boxShadow: "0 10px 30px rgba(0,0,0,.5)", color: "#e2e8f0", fontSize: 14 }}>
                    <Spinner size={20} />
                    <div>
                      <div style={{ fontWeight: 700 }}>Forecasting {horizonDraft} days ahead…</div>
                      <div style={{ color: "#94a3b8", fontSize: 12 }}>Running the LightGBM model for {store} / {dept}</div>
                    </div>
                  </div>
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
                         const forecastColor = Number(r.yhat) > max * .8 ? "#8B5CF6" : Number(r.yhat) > max * .5 ? "#06B6D4" : "#14B8A6";
                         const detailLeft = Math.max(4, Math.min(68, (((48 + i * barWidth + barWidth / 2) / 640) * 100) - 16));
                         const detailTop = Math.max(8, Math.min(56, (((bottom - h) / 280) * 100) - 18));
                         const handleForecastClick = () => {
                           setSelectedForecastPoint(r);
                           setSelectedForecastDetailPos({ left: detailLeft, top: detailTop });
                         };
                         return (
                           <g key={i}>
                             <rect x={left + i * barWidth + 1} y={bottom - uh} width={Math.max(1, barWidth - 2)} height={Math.max(0, uh - h)} fill="rgba(168,85,247,.14)" rx="1" />
                             <rect x={left + i * barWidth + 1} y={bottom - h} width={Math.max(1, barWidth - 2)} height={h} fill={selectedForecastPoint?.Date === r.Date ? "#EC4899" : forecastColor} rx="2" style={{ cursor: "pointer", stroke: selectedForecastPoint?.Date === r.Date ? "#F8FAFC" : "none", strokeWidth: 2 }} onClick={handleForecastClick}><title>{`Forecast ${r.Date}: forecast ${r.yhat}, lower ${r.yhat_lower}, upper ${r.yhat_upper}`}</title></rect>
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
              {selectedForecastPoint && (
                <div style={{ position: "absolute", top: selectedForecastDetailPos.top + "%", left: selectedForecastDetailPos.left + "%", width: 230, padding: 10, borderRadius: 10, border: "1px solid #A78BFA", background: "rgba(15,23,42,.96)", color: "#E2E8F0", fontSize: 12, boxShadow: "0 12px 32px rgba(0,0,0,.45)", zIndex: 20, backdropFilter: "blur(8px)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
                    <b style={{ color: "#8B5CF6" }}>Selected forecast date</b>
                    <button style={{ border: "none", background: "transparent", color: "#94A3B8", cursor: "pointer" }} onClick={() => setSelectedForecastPoint(null)}>×</button>
                  </div>
                  <div><b>Date:</b> {String(selectedForecastPoint.Date).slice(0, 10)}</div>
                  <div><b>Forecast:</b> {Number(selectedForecastPoint.yhat).toLocaleString(undefined, { maximumFractionDigits: 1 })}</div>
                  <div><b>Lower:</b> {Number(selectedForecastPoint.yhat_lower).toLocaleString(undefined, { maximumFractionDigits: 1 })}</div>
                  <div><b>Upper:</b> {Number(selectedForecastPoint.yhat_upper).toLocaleString(undefined, { maximumFractionDigits: 1 })}</div>
                  <div style={{ color: "#94A3B8", marginTop: 6 }}>Click another bar to update details.</div>
                </div>
              )}
              </div>
              <Table columns={forecastColumns as any} rows={forecast.records || []} loading={forecastPending} loadingMessage="Generating demand forecast..." emptyMessage="No forecast available." />
            </div>
          )}

          {tab === "reorders" && (
            <div style={card}>
              <h3 style={{ marginTop: 0 }}>Reorder recommendations</h3>
              <p style={{ color: "#94a3b8", fontSize: 13, marginTop: 0 }}>
                Expedite orders for series the weekly static order will not protect. Urgent units go to the fastest supplier,
                the rest to the best cost/reliability supplier that still delivers in time.
              </p>
              <LoadingStatus visible={reordersLoading} message="Calculating reorder recommendations..." />
              <Table columns={reorderColumns as any}
                rows={[...reorders].sort((a: any, b: any) => (b.recommended_qty > 0 ? 1 : 0) - (a.recommended_qty > 0 ? 1 : 0))}
                loading={reordersLoading} loadingMessage="Calculating reorder recommendations..." emptyMessage="No reorder data available." />
              {poMessage && <div style={{ color: "#a5b4fc", margin: "10px 0" }}>{poMessage}</div>}
              <h4>Purchase orders</h4>
              <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
                <select style={input as any} value={poStoreFilter} onChange={(e) => setPoStoreFilter(e.target.value)}>
                  <option value="ALL">All stores</option>
                  {STORES.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
                <select style={input as any} value={poDeptFilter} onChange={(e) => setPoDeptFilter(e.target.value)}>
                  <option value="ALL">All departments</option>
                  {DEPTS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
                <select style={input as any} value={poStatusFilter} onChange={(e) => setPoStatusFilter(e.target.value)}>
                  <option value="ALL">All statuses</option>
                  <option value="DRAFT">DRAFT</option>
                  <option value="APPROVED">APPROVED</option>
                  <option value="REJECTED">REJECTED</option>
                </select>
              </div>
              <LoadingStatus visible={poCreating} message="Creating draft purchase order..." />
              <Table columns={poColumns as any} rows={pos.filter((r: any) => (poStoreFilter === "ALL" || r.store === poStoreFilter) && (poDeptFilter === "ALL" || r.dept === poDeptFilter) && (poStatusFilter === "ALL" || r.status === poStatusFilter))} loading={poCreating} loadingMessage="Loading draft purchase orders..." emptyMessage="No draft purchase orders yet." />
            </div>
          )}

          {tab === "report" && (
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <h3 style={{ margin: 0 }}>Auto-generated report</h3>
                <button style={{ ...btn, opacity: reportLoading || !reportMd ? 0.5 : 1 }} disabled={reportLoading || !reportMd} onClick={() => {
                  const blob = new Blob([reportMd], { type: "text/markdown" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url; a.download = "supply_chain_report.md"; a.click();
                  URL.revokeObjectURL(url);
                }}>⬇ Download Markdown</button>
              </div>
              <div style={{ background: "#0b1220", padding: 14, borderRadius: 12, border: "1px solid #1f2a3d", marginBottom: 12 }}>
                <h4 style={{ marginTop: 0 }}>Chronological forecast evaluation</h4>
                <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 10px" }}>
                  Real M5 holdout evaluation with naive, seasonal-naive, and trained-model comparison.
                </p>
                <button style={{ ...btn, marginBottom: 10, opacity: evalRunning ? 0.6 : 1 }} onClick={runEvaluation} disabled={evalRunning}>
                  {evalRunning ? "Running evaluation..." : "Run real evaluation"}
                </button>
                <LoadingStatus visible={evalLoading || evalRunning} message={evalRunning ? "Running chronological backtest..." : "Loading evaluation..." } />
                {evalData ? (
                  <>
                    <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 10 }}>
                      Holdout: {evalData.primary_test_period?.from} to {evalData.primary_test_period?.to}; observations: {evalData.primary_test_observations}; windows: {evalData.windows}
                    </div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                      <thead>
                        <tr style={{ background: "#1e293b" }}>
                          <th style={{ padding: 8, textAlign: "left" }}>Model</th>
                          <th style={{ padding: 8, textAlign: "right" }}>MAE</th>
                          <th style={{ padding: 8, textAlign: "right" }}>RMSE</th>
                          <th style={{ padding: 8, textAlign: "right" }}>WAPE (%)</th>
                          <th style={{ padding: 8, textAlign: "right" }}>MAPE (%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {["naive", "seasonal_naive", "model"].map((name) => (
                          <tr key={name} style={{ borderTop: "1px solid #1f2a3d" }}>
                            <td style={{ padding: 8 }}>{name === "seasonal_naive" ? "Seasonal naive (same day last week)" : name === "naive" ? "Naive (yesterday repeated)" : "LightGBM model"}</td>
                            {["MAE", "RMSE", "WAPE", "MAPE_nonzero"].map((metric) => (
                              <td key={metric} style={{ padding: 8, textAlign: "right" }}>
                                {Number(evalData.primary_overall_macro?.[name]?.[metric] || 0).toFixed(1)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ color: "#94a3b8", fontSize: 12, marginTop: 8 }}>
                      Model beats naive: {String(evalData.primary_overall_macro?.model?.MAE < evalData.primary_overall_macro?.naive?.MAE)}.
                      Model beats seasonal naive: {String(evalData.primary_overall_macro?.model?.MAE < evalData.primary_overall_macro?.seasonal_naive?.MAE)}.
                    </div>
                  </>
                ) : (
                  <div style={{ color: "#64748b" }}>No evaluation results yet. Run the real evaluation to compare models.</div>
                )}
              </div>
              <div style={{ background: "#0b1220", padding: 14, borderRadius: 12, border: "1px solid #1f2a3d", fontSize: 13, maxHeight: 600, overflow: "auto" }}>
                {reportLoading ? (
                  <LoadingStatus visible message="Generating the supply-chain report (risk snapshot, reorders, POs, evaluation)…" />
                ) : reportError ? (
                  <div style={{ color: "#f87171" }}>Could not generate the report: {reportError}. <button style={{ ...btn, padding: "4px 10px", fontSize: 12, marginLeft: 8 }} onClick={loadReport}>Retry</button></div>
                ) : (
                  renderMarkdown(reportMd)
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}