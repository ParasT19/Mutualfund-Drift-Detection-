"""MutualFundDrift — Streamlit Dashboard (pure Streamlit, no HTML/CSS)."""

import time
from datetime import date, timedelta
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE        = "http://localhost:8000"
DRIFT_THRESHOLD = 0.25

st.set_page_config(
    page_title="MutualFundDrift",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API helpers ───────────────────────────────────────────────────────────────
def _call(method, endpoint, timeout=30, **kwargs):
    try:
        r = getattr(requests, method)(f"{API_BASE}{endpoint}", timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[API ERROR] {method.upper()} {endpoint}: {e}")
        return None

api_get  = lambda ep, **kw: _call("get",  ep, params=kw.get("params"))
api_post = lambda ep, data: _call("post", ep, json=data)
api_put  = lambda ep, data: _call("put",  ep, json=data)
api_delete = lambda ep: _call("delete", ep)

# ── Mock data ─────────────────────────────────────────────────────────────────
def _snaps(drift_list, size_list, style_list, large_list, mid_list, small_list, corr_list):
    today = date.today().replace(day=1)
    return [
        {
            "snapshot_date": (today - timedelta(days=30*i)).replace(day=1).isoformat(),
            "drift_score":   drift_list[i],  "size_score":   size_list[i],
            "style_score":   style_list[i],  "large_cap_pct": large_list[i],
            "mid_cap_pct":   mid_list[i],    "small_cap_pct": small_list[i],
            "rolling_corr":  corr_list[i],   "active_share":  0.75,
        }
        for i in range(len(drift_list))
    ]

MOCK_FUNDS = [
    {
        "scheme_code": "120503", "amc_name": "HDFC AMC",
        "scheme_name": "HDFC Mid-Cap Opportunities Fund",
        "category": "Mid Cap Fund", "benchmark_index": "Nifty Midcap 150 TRI",
        "current_drift_score": 0.31, "severity": "amber",
        "mandate_size_score": 0.50,  "mandate_style_score": 0.55,
        "latest_alert_message": "Portfolio shifted 24% toward large-caps. Drift: 0.31 (AMBER). Review allocation.",
        "snapshots": _snaps(
            [0.31,0.28,0.24,0.19,0.15,0.10,0.07,0.05],
            [0.74,0.70,0.65,0.60,0.56,0.53,0.51,0.50],
            [0.61,0.60,0.59,0.58,0.57,0.56,0.56,0.55],
            [42,38,33,28,23,19,16,13], [48,51,55,60,65,69,72,75],
            [10,11,12,12,12,12,12,12], [0.73,0.77,0.81,0.85,0.88,0.90,0.91,0.92],
        ),
    },
    {
        "scheme_code": "125354", "amc_name": "Axis AMC",
        "scheme_name": "Axis Bluechip Fund",
        "category": "Large Cap Fund", "benchmark_index": "Nifty 50 TRI",
        "current_drift_score": 0.08, "severity": "normal",
        "mandate_size_score": 0.85, "mandate_style_score": 0.70,
        "latest_alert_message": None,
        "snapshots": _snaps(
            [0.08,0.08,0.07,0.08,0.08,0.07,0.07,0.06],
            [0.84,0.85,0.86,0.85,0.85,0.84,0.85,0.84],
            [0.70,0.70,0.71,0.70,0.70,0.69,0.69,0.68],
            [80,81,82,81,81,80,81,80], [15,14,13,14,14,15,14,15],
            [5,5,5,5,5,5,5,5], [0.95,0.95,0.94,0.95,0.95,0.94,0.95,0.96],
        ),
    },
    {
        "scheme_code": "119551", "amc_name": "Nippon India AMC",
        "scheme_name": "Nippon India Small Cap Fund",
        "category": "Small Cap Fund", "benchmark_index": "Nifty Smallcap 250 TRI",
        "current_drift_score": 0.41, "severity": "red",
        "mandate_size_score": 0.15, "mandate_style_score": 0.60,
        "latest_alert_message": "Drift 0.41 (RED) — shifted toward mid-cap over 8 quarters. Benchmark corr: 0.68. Review urgently.",
        "snapshots": _snaps(
            [0.41,0.37,0.32,0.27,0.23,0.18,0.14,0.10],
            [0.40,0.36,0.32,0.28,0.25,0.22,0.19,0.17],
            [0.53,0.52,0.51,0.50,0.49,0.47,0.46,0.45],
            [8,7,6,5,5,4,4,3], [35,30,26,23,20,17,14,12],
            [57,63,68,72,75,79,82,85], [0.68,0.72,0.76,0.80,0.84,0.87,0.90,0.92],
        ),
    },
]

def get_funds():
    return api_get("/api/drift/leaderboard") or MOCK_FUNDS

def get_snapshots(code):
    data = api_get(f"/api/funds/{code}/snapshots")
    if data: return data
    return next((f["snapshots"] for f in MOCK_FUNDS if f["scheme_code"] == code), [])

def get_fund(code):
    return api_get(f"/api/funds/{code}") or next(
        (f for f in MOCK_FUNDS if f["scheme_code"] == code), None
    )

def quarter_label(sd):
    try:
        p = str(sd).split("-")
        return f"Q{(int(p[1])-1)//3+1} {p[0]}"
    except Exception:
        return str(sd)

# ── Charts ────────────────────────────────────────────────────────────────────
def chart_style_box(snaps, ms, mst, name):
    if not snaps:
        return st.info("No style box data.")
    xs = [s["style_score"] for s in snaps]
    ys = [s["size_score"]  for s in snaps]
    ds = [s["drift_score"] for s in snaps]
    labels = [quarter_label(s["snapshot_date"]) for s in snaps]
    n = len(snaps)

    fig = go.Figure()
    for v in [0.33, 0.67]:
        for kw in [dict(x0=v,x1=v,y0=0,y1=1), dict(x0=0,x1=1,y0=v,y1=v)]:
            fig.add_shape(type="line", line=dict(color="lightgrey", dash="dot"), **kw)

    def cell(s): return (0.67,1.0) if s>0.67 else (0.33,0.67) if s>=0.33 else (0.0,0.33)
    mx0,mx1 = cell(mst); my0,my1 = cell(ms)
    fig.add_shape(type="rect", x0=mx0,x1=mx1,y0=my0,y1=my1, fillcolor="green", opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=[mst],y=[ms],mode="markers",marker=dict(symbol="star",size=18,color="green"),name="Mandate"))
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(color="grey",dash="dot"),showlegend=False,hoverinfo="skip"))

    for i,(x,y,d,lbl) in enumerate(zip(xs,ys,ds,labels)):
        latest = (i==n-1)
        opacity = 0.3 + 0.7*i/max(n-1,1)
        fig.add_trace(go.Scatter(
            x=[x],y=[y],mode="markers",
            marker=dict(size=18 if latest else 10,color=f"rgba(52,152,219,{opacity})",line=dict(color="white",width=2)),
            name=lbl if latest else "",showlegend=latest,
            hovertemplate=f"<b>{lbl}</b><br>Style={x:.3f}<br>Size={y:.3f}<br>Drift={d:.3f}<extra></extra>",
        ))

    for lbl,(x,y) in zip(
        ["Large Value","Large Blend","Large Growth","Mid Value","Mid Blend","Mid Growth","Small Value","Small Blend","Small Growth"],
        [(0.17,0.83),(0.50,0.83),(0.83,0.83),(0.17,0.50),(0.50,0.50),(0.83,0.50),(0.17,0.17),(0.50,0.17),(0.83,0.17)],
    ):
        fig.add_annotation(x=x,y=y,text=lbl,showarrow=False,font=dict(size=8,color="grey"),opacity=0.7)

    fig.update_layout(
        title=f"{name} — Style Box Journey",
        xaxis=dict(title="Value → Growth", range=[0,1], showgrid=False),
        yaxis=dict(title="Large → Small",  range=[0,1], showgrid=False),
        height=400, margin=dict(l=50,r=20,t=50,b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_composition(snaps, name):
    if not snaps: return st.info("No composition data.")
    qs = [quarter_label(s["snapshot_date"]) for s in snaps]
    fig = go.Figure([
        go.Bar(name="Large", x=qs, y=[s["large_cap_pct"] for s in snaps], marker_color="#3498db"),
        go.Bar(name="Mid",   x=qs, y=[s["mid_cap_pct"]   for s in snaps], marker_color="#e67e22"),
        go.Bar(name="Small", x=qs, y=[s["small_cap_pct"] for s in snaps], marker_color="#e74c3c"),
    ])
    fig.update_layout(barmode="stack", title=f"{name} — Cap Composition",
                      xaxis=dict(tickangle=-30), yaxis=dict(title="% NAV", range=[0,105]),
                      height=340, margin=dict(l=40,r=20,t=50,b=70))
    st.plotly_chart(fig, use_container_width=True)


def chart_drift_timeline(snaps, name):
    if not snaps: return st.info("No drift data.")
    dates  = [s["snapshot_date"] for s in snaps]
    drifts = [s["drift_score"]   for s in snaps]
    fig = go.Figure()
    for y0,y1,col,lbl in [(0,0.15,"green","Normal"),(0.15,0.25,"yellow","Watch"),(0.25,0.35,"orange","Amber"),(0.35,1.5,"red","Red")]:
        fig.add_hrect(y0=y0,y1=y1,fillcolor=col,opacity=0.07,line_width=0,annotation_text=lbl,annotation_position="left")
    fig.add_hline(y=DRIFT_THRESHOLD, line_dash="dot", line_color="red", annotation_text=f"Threshold {DRIFT_THRESHOLD}")
    fig.add_trace(go.Scatter(x=dates,y=drifts,mode="lines+markers",line=dict(color="#2980b9",width=2),
                             marker=dict(size=7),name="Drift",
                             hovertemplate="<b>%{x}</b><br>Drift=%{y:.3f}<extra></extra>"))
    fig.update_layout(title=f"{name} — Drift Timeline",
                      xaxis=dict(tickangle=-30),
                      yaxis=dict(title="Drift Score",range=[0,max(0.6,max(drifts)+0.1)]),
                      height=340,margin=dict(l=50,r=80,t=50,b=70))
    st.plotly_chart(fig, use_container_width=True)


def chart_corr_heatmap(snaps, name):
    rows = []
    for s in snaps:
        if s.get("rolling_corr") is None: continue
        try:
            p = str(s["snapshot_date"]).split("-")
            rows.append({"Year": int(p[0]), "Q": f"Q{(int(p[1])-1)//3+1}", "Corr": s["rolling_corr"]})
        except Exception: pass
    if not rows: return st.info("No correlation data.")
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Q", columns="Year", values="Corr", aggfunc="mean")
    pivot = pivot.reindex([q for q in ["Q1","Q2","Q3","Q4"] if q in pivot.index])
    fig, ax = plt.subplots(figsize=(7,2.5))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0.5, vmax=1.0, annot=True, fmt=".3f", linewidths=0.5)
    ax.set_title(f"{name} — Rolling Correlation")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Pages ─────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.title("📊 MutualFundDrift")

    st.divider()

    st.sidebar.header("Filters")
    cat_filter = st.sidebar.selectbox("Category", ["All","Large Cap Fund","Mid Cap Fund","Small Cap Fund","Flexi Cap Fund"])
    sev_filter = st.sidebar.multiselect("Severity", ["normal","watch","amber","red"], default=["normal","watch","amber","red"])
    max_drift  = st.sidebar.slider("Max Drift", 0.0, 1.5, 1.5, 0.05)

    funds = get_funds()
    if cat_filter != "All":      funds = [f for f in funds if f.get("category") == cat_filter]
    funds = [f for f in funds if f.get("severity","normal") in sev_filter]
    funds = [f for f in funds if f.get("current_drift_score",0) <= max_drift]

    avg_drift = np.mean([f.get("current_drift_score",0) for f in funds]) if funds else 0.0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Funds Tracked",  len(funds))
    c2.metric("Active Alerts",  sum(1 for f in funds if f.get("severity") in ("amber","red")))
    c3.metric("Avg Drift",      f"{avg_drift:.3f}")
    c4.metric("Red Alerts",     sum(1 for f in funds if f.get("severity")=="red"))

    st.divider()
    st.subheader("Fund Leaderboard")
    rows = [{"Fund": f.get("scheme_name","")[:45], "AMC": f.get("amc_name","")[:20],
             "Category": f.get("category",""), "Drift": round(f.get("current_drift_score",0),3),
             "Severity": f.get("severity","normal").upper()}
            for f in sorted(funds, key=lambda x: x.get("current_drift_score",0), reverse=True)]
    if rows:
        df = pd.DataFrame(rows)
        df.insert(0, "#", range(1, len(df) + 1))
        bg = {"NORMAL":"#d5f5e3","WATCH":"#fef9e7","AMBER":"#fdebd0","RED":"#fadbd8"}
        st.dataframe(df.style.format({"Drift": "{:.3f}"}).apply(lambda row: [f"background-color:{bg.get(row['Severity'],'')}"] * len(row), axis=1),
                     use_container_width=True, height=350, hide_index=True)


def page_analyser():
    st.title("🔬 Fund Analyser")
    st.divider()

    funds_list = api_get("/api/funds") or MOCK_FUNDS
    options    = {f["scheme_name"]: f["scheme_code"] for f in funds_list}
    
    st.sidebar.markdown("**Select Duration**")
    duration = st.sidebar.selectbox("Select Duration", ["Last 6 Months", "Last 1 Year", "Last 2 Years", "All Time"], index=3, label_visibility="collapsed")

    st.sidebar.markdown("**Select Fund**")
    with st.sidebar.container(height=350):
        name = st.radio("Select Fund", list(options.keys()), label_visibility="collapsed")
    code = options[name]

    with st.sidebar.expander("📁 Upload Portfolio (CSV / Excel)"):
        st.caption("Upload custom holdings or snapshot file")
        up_file = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"], key=f"up_{code}")
        if up_file is not None:
            if st.button("Ingest File Data", key=f"btn_up_{code}", use_container_width=True):
                with st.spinner("Processing file..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/funds/{code}/upload",
                            files={"file": (up_file.name, up_file.getvalue(), up_file.type or "application/octet-stream")},
                            timeout=15,
                        )
                        res = resp.json()
                        if res.get("status") == "success":
                            st.success(res.get("message"))
                            st.rerun()
                        else:
                            st.error(res.get("message", "Upload failed."))
                    except Exception as e:
                        st.error(f"Error uploading: {str(e)}")

    fund  = get_fund(code)
    snaps = get_snapshots(code)

    # get_fund returns DriftSummary which lacks amc_name & benchmark_index —
    # look them up from the already-loaded funds_list which has both fields
    extra = next((f for f in funds_list if f.get("scheme_code") == code), {})
    if fund:
        fund = {**fund, **{k: extra[k] for k in ("amc_name", "benchmark_index") if k in extra}}

    if snaps:
        # API returns newest-first — sort ascending so [-N:] gives the most recent N months
        snaps = sorted(snaps, key=lambda s: s.get("snapshot_date", ""))
        if duration == "Last 6 Months":
            snaps = snaps[-6:]
        elif duration == "Last 1 Year":
            snaps = snaps[-12:]
        elif duration == "Last 2 Years":
            snaps = snaps[-24:]


    if not fund: return st.error("Fund data unavailable.")

    c1,c2,c3 = st.columns(3)
    c1.info(f"**AMC:** {fund.get('amc_name','—')}")
    c2.info(f"**Category:** {fund.get('category','—')}")
    c3.info(f"**Benchmark:** {fund.get('benchmark_index','—')}")

    latest    = snaps[-1] if snaps else {}
    severity  = fund.get("severity","normal")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Drift Score",    f"{fund.get('current_drift_score',0):.3f}")
    m2.metric("Severity",       severity.upper())
    m3.metric("Rolling Corr",   f"{latest.get('rolling_corr',0):.3f}" if latest.get("rolling_corr") else "N/A")
    m4.metric("Active Share",   f"{latest.get('active_share',0):.0%}" if latest.get("active_share") else "N/A")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Style Box Journey**")
        chart_style_box(snaps, fund.get("mandate_size_score",0.5), fund.get("mandate_style_score",0.5), name)
    with col2:
        st.markdown("**Cap Composition**")
        chart_composition(snaps, name)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Drift Timeline**")
        chart_drift_timeline(snaps, name)
    with col4:
        st.markdown("**Correlation Heatmap**")
        chart_corr_heatmap(snaps, name)

    # ── AI Prediction Panel ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 Drift Prediction")
    pred = api_get(f"/api/drift/{code}/predict")
    if pred and pred.get("status") == "ok":
        prob  = pred.get("drift_probability", 0)
        wdrift = pred.get("will_drift", False)
        shap_f = pred.get("top_shap_features", {})

        risk_label = "🔴 HIGH RISK — Likely to drift" if wdrift else "🟢 LOW RISK — Likely to stay on mandate"
        risk_color = "#e74c3c" if wdrift else "#2ecc71"

        pa, pb, pc = st.columns([1, 1, 2])
        pa.metric("Drift Probability (Next 2 Months)", f"{prob:.1%}")
        pb.metric("Verdict", "Will Drift ⚠️" if wdrift else "Stable ✅")
        pc.markdown(f"<div style='padding:12px;border-radius:8px;background:{risk_color}22;border:1.5px solid {risk_color};color:{risk_color};font-weight:600;font-size:1rem;'>{risk_label}</div>", unsafe_allow_html=True)

        if shap_f:
            st.markdown("**Top Factors Driving This Prediction** *(SHAP feature importance)*")
            shap_df = pd.DataFrame(
                {"Feature": [k.replace("_", " ").title() for k in shap_f.keys()], "Impact": list(shap_f.values())}
            ).sort_values("Impact", key=abs, ascending=True)
            colours = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["Impact"]]
            fig = go.Figure(go.Bar(
                x=shap_df["Impact"], y=shap_df["Feature"],
                orientation="h", marker_color=colours,
                text=[f"{v:+.4f}" for v in shap_df["Impact"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="🔴 = pushes toward drift  |  🟢 = pushes toward stability",
                height=260, margin=dict(l=10, r=60, t=40, b=20),
                xaxis=dict(title="SHAP Value"), yaxis=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True)
    elif pred and pred.get("status") == "no_model":
        st.info("🤖 AI model not trained yet. Run `python engine/train_model.py` to enable predictions.")
    else:
        st.warning("AI prediction unavailable — ensure the backend is online.")

    st.divider()
    st.subheader("Actions")

    btn_alert, btn_stop = st.columns(2)

    with btn_alert:
        if st.button("🔔 Generate Alert", use_container_width=True, type="primary"):
            with st.spinner("Analysing drift and generating alert..."):
                resp = api_post(f"/api/alerts/{code}", {})
            if resp and resp.get("id"):
                sev = resp.get("severity", "watch")
                sev_fn = {"red": st.error, "amber": st.warning, "watch": st.info}.get(sev, st.info)
                sev_fn(f"✅ Alert generated! [{sev.upper()}] {resp.get('alert_message','')}")
                st.caption(f"Alert #{resp['id']} saved — now visible in Alerts Centre.")
            else:
                st.error("Failed to generate alert. Ensure backend is running.")

    with btn_stop:
        if st.button("❌ Stop Tracking Fund", use_container_width=True):
            api_delete(f"/api/funds/{code}")
            st.success("Fund removed from watch list. Please refresh the page.")



def page_alerts():
    st.title("🔔 Alerts Centre")
    st.divider()

    all_alerts = api_get("/api/alerts") or []

    d1,d2 = st.columns(2)
    start = d1.date_input("From", date.today() - timedelta(days=90))
    end   = d2.date_input("To",   date.today())

    filtered = [a for a in all_alerts if a.get("alert_message") and
                start <= date.fromisoformat(str(a.get("alert_date",date.today()))[:10]) <= end]

    tabs = st.tabs([f"All ({len(filtered)})","Watch","Amber","Red"])
    filters = [None,"watch","amber","red"]

    for tab, sev in zip(tabs, filters):
        tab_name = sev or "all"
        with tab:
            shown = [a for a in filtered if sev is None or a.get("severity")==sev]
            if not shown:
                st.info("No alerts.")
                continue
            for a in shown:
                with st.container(border=True):
                    c1,c2 = st.columns([3,1])
                    c1.markdown(f"**{a.get('scheme_name',a.get('scheme_code'))}** — `{a.get('severity','').upper()}`")
                    c2.caption(str(a.get("alert_date","")))
                    sev_fn = {"red":st.error,"amber":st.warning,"watch":st.warning}.get(a.get("severity"),st.info)
                    sev_fn(a.get("alert_message",""))
                    st.caption(f"Drift: {a.get('drift_score',0):.3f} | #{a.get('id','')}")
                    if not a.get("acknowledged") and a.get("id"):
                        if st.button(f"Acknowledge #{a['id']}", key=f"ack_{tab_name}_{a['id']}"):
                            api_put(f"/api/alerts/{a['id']}/ack", {"acknowledged": True})
                            st.rerun()


def page_register_fund():
    st.title("➕ Register New Fund")
    st.divider()
    st.markdown("Add a new mutual fund scheme to track and monitor style drift.")

    with st.form("register_fund_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        code = c1.text_input("Scheme Code (AMFI)", placeholder="e.g. 120503")
        name = c2.text_input("Scheme Name", placeholder="e.g. HDFC Mid-Cap Opportunities Fund")

        c3, c4 = st.columns(2)
        amc = c3.text_input("AMC Name", placeholder="e.g. HDFC AMC")
        category = c4.selectbox("SEBI Category", ["Large Cap Fund", "Mid Cap Fund", "Small Cap Fund", "Flexi Cap Fund"])

        c5 = st.text_input("Benchmark Index", placeholder="e.g. Nifty Midcap 150 TRI")

        c6, c7 = st.columns(2)
        ms = c6.slider("Mandate Size Score", 0.0, 1.0, 0.5, 0.05, help="0.0 = Small Cap, 0.5 = Mid Cap, 1.0 = Large Cap")
        mst = c7.slider("Mandate Style Score", 0.0, 1.0, 0.5, 0.05, help="0.0 = Value, 1.0 = Growth")

        submitted = st.form_submit_button("Register Fund")
        if submitted:
            if not code or not name or not amc:
                st.error("Please fill in all required fields (Scheme Code, Scheme Name, AMC Name).")
            else:
                payload = {
                    "scheme_code": code,
                    "scheme_name": name,
                    "amc_name": amc,
                    "category": category,
                    "benchmark_index": c5,
                    "mandate_size_score": ms,
                    "mandate_style_score": mst
                }
                resp = api_post("/api/funds/", payload)
                if resp:
                    st.success(f"Fund '{name}' registered successfully! The system will now begin tracking it.")
                else:
                    st.error("Failed to register fund. Please ensure the backend is online.")


# ── Sidebar nav & status ──────────────────────────────────────────────────────
def main():
    st.sidebar.title("📊 MutualFundDrift")
    st.sidebar.divider()
    page = st.sidebar.radio("Navigate", ["Dashboard","Fund Analyser","Alerts Centre","Register Fund"])

    st.sidebar.divider()
    st.sidebar.subheader("Status")
    health = api_get("/health")
    if health:
        st.sidebar.success(f"API v{health.get('version','1.0')} Online")
        st.sidebar.caption("DB: " + ("Connected" if health.get("db_connected") else "Disconnected"))
    else:
        st.sidebar.error("Backend offline — mock data")
        st.sidebar.caption("`uvicorn backend.main:app --reload`")

    {"Dashboard": page_dashboard, "Fund Analyser": page_analyser,
     "Alerts Centre": page_alerts,
     "Register Fund": page_register_fund}[page]()


if __name__ == "__main__":
    main()
