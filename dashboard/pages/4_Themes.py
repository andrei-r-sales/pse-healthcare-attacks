import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_data, sidebar_filters

st.set_page_config(page_title="Themes (NLP)", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f7f5f2; }
    header[data-testid="stHeader"] { background-color: #f7f5f2; }
    [data-testid="stToolbar"] { background-color: #f7f5f2; }
    </style>
""", unsafe_allow_html=True)

# ── Load the labeled topic file ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TOPIC_PATH = BASE_DIR / "data" / "pse_topics_labeled.csv"
# Fallback to flat project dir if not in data/
if not TOPIC_PATH.exists():
    TOPIC_PATH = Path("/Users/andreisales/Desktop/document idf/pse_topics_labeled.csv")

@st.cache_data
def load_topics():
    return pd.read_csv(TOPIC_PATH, parse_dates=["Date"])

df = load_topics()
df = df[df["theme"] != "Unclassified"].copy()

st.title("Discovered Themes — NLP Topic Modeling")
st.markdown(
    "<p style='font-size:13px; color:#888888;'>"
    "Themes surfaced by unsupervised topic modeling (BERTopic) on incident descriptions — "
    "no keywords supplied. Each incident is assigned to its dominant theme."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ── Mode-of-attack metric cards ───────────────────────────────────────────────
mode_counts = df["mode"].value_counts()
total = len(df)

c1, c2, c3 = st.columns(3)
with c1:
    n = int(mode_counts.get("Direct Violence", 0))
    st.metric("Direct Violence", f"{n:,}")
    st.caption(f"{n/total*100:.0f}% — airstrikes, shootings, shelling, residential strikes")
with c2:
    n = int(mode_counts.get("Access Denial", 0))
    st.metric("Access Denial", f"{n:,}")
    st.caption(f"{n/total*100:.0f}% — blocked ambulances, fuel deprivation, checkpoint obstruction")
with c3:
    n = int(mode_counts.get("Detention", 0))
    st.metric("Detention", f"{n:,}")
    st.caption(f"{n/total*100:.0f}% — arrests of health workers")

st.info(
    "**Key finding:** roughly 1 in 5 attacks on healthcare were non-kinetic — "
    "denying access, fuel, or movement rather than striking directly. This systemic "
    "obstruction pattern was surfaced by topic modeling and is not captured by "
    "casualty counts alone."
)

st.divider()

# ── Theme distribution ────────────────────────────────────────────────────────
st.subheader("Incidents by theme")

dist = (
    df.groupby(["theme", "mode"])
    .size()
    .reset_index(name="Incidents")
    .sort_values("Incidents", ascending=True)
)

mode_colors = {
    "Direct Violence": "#c0392b",
    "Access Denial":   "#e8916a",
    "Detention":       "#7fb3d3",
    "Other":           "#aaaaaa",
}

fig_dist = px.bar(
    dist, x="Incidents", y="theme", orientation="h",
    color="mode", color_discrete_map=mode_colors,
    labels={"theme": "", "mode": ""}, text="Incidents",
)
fig_dist.update_traces(
    textposition="outside", cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>%{x} incidents<extra></extra>"
)
fig_dist.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10, r=80), height=420,
)
st.plotly_chart(fig_dist, width='stretch')

st.divider()

# ── Theme x Region ────────────────────────────────────────────────────────────
st.subheader("Where each theme occurs")
st.caption("Regional split per theme · Detention and obstruction concentrate in the West Bank; airstrikes in Gaza")

region_df = (
    df.groupby(["theme", "Region"])
    .size()
    .reset_index(name="Incidents")
)
# Keep the two main regions; fold anything else into the order by total size
theme_order = (
    df.groupby("theme").size().sort_values(ascending=True).index.tolist()
)

region_colors = {
    "Gaza Strip": "#c0392b",
    "West Bank":  "#7fb3d3",
}

fig_region = px.bar(
    region_df, x="Incidents", y="theme", color="Region",
    orientation="h", barmode="stack",
    color_discrete_map=region_colors,
    category_orders={"theme": theme_order},
    labels={"theme": "", "Region": ""},
)
fig_region.update_traces(
    hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x}<extra></extra>"
)
fig_region.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=420,
)
st.plotly_chart(fig_region, width='stretch')

st.divider()

# ── Average lethality per theme ───────────────────────────────────────────────
st.subheader("Average lethality by theme")
st.caption("Mean health workers killed per incident · Quantifies the kinetic vs. non-kinetic divide")

leth = (
    df.groupby(["theme", "mode"])["Health Workers Killed"]
    .mean()
    .reset_index(name="Avg_Killed")
    .sort_values("Avg_Killed", ascending=True)
)
leth["Avg_Killed"] = leth["Avg_Killed"].round(2)

fig_leth = px.bar(
    leth, x="Avg_Killed", y="theme", orientation="h",
    color="mode", color_discrete_map=mode_colors,
    labels={"theme": "", "Avg_Killed": "Avg. killed / incident", "mode": ""},
    text="Avg_Killed",
)
fig_leth.update_traces(
    textposition="outside", cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>Avg killed/incident: %{x:.2f}<extra></extra>"
)
fig_leth.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10, r=50), height=420,
)
st.plotly_chart(fig_leth, width='stretch')

st.divider()

# ── Cross-validation: unsupervised topics vs. hand-coded flags ────────────────
st.subheader("Model validation — discovered topics vs. hand-coded flags")
st.caption(
    "Each topic was found by the model with no keywords. The table shows how often "
    "incidents in each theme also triggered an independent, hand-written keyword flag. "
    "High overlap = two independent methods agreeing on the same incidents."
)

flag_cols = ["protected_entity", "residential_strike", "children_affected", "repeat_target_text"]
flag_labels = {
    "protected_entity":   "Red Cross / UN flag",
    "residential_strike": "Residential flag",
    "children_affected":  "Children flag",
    "repeat_target_text": "Repeat-target flag",
}

overlap = (
    df.groupby("theme")[flag_cols]
    .mean()
    .round(2)
    .rename(columns=flag_labels)
)
# Order by theme size, largest first
size_order = df.groupby("theme").size().sort_values(ascending=False).index
overlap = overlap.loc[size_order]

# Show as a percentage-styled table
overlap_pct = (overlap * 100).round(0).astype(int).astype(str) + "%"
st.dataframe(overlap_pct, use_container_width=True)

st.caption(
    "Read across a row: e.g. if the Red Cross / Crescent theme shows a high value under the "
    "Red Cross / UN flag, the unsupervised model independently recovered the same incidents the "
    "keyword flag was written to catch — cross-validating both methods."
)

st.divider()

# ── Themes over time (normalized share per month) ────────────────────────────
st.subheader("How themes evolved")
st.caption("Share of each month's incidents by mode of attack · Oct 2023 onward")

df["Year_Month"] = df["Date"].dt.strftime("%Y-%m")

tot = (
    df[df["Year_Month"] >= "2023-10"]
    .groupby(["Year_Month", "mode"])
    .size()
    .reset_index(name="Incidents")
)
tot_total = tot.groupby("Year_Month")["Incidents"].transform("sum")
tot["Share"] = (tot["Incidents"] / tot_total * 100).round(1)

# Force chronological categorical order so the x-axis is discrete, not a 1970+ timeline
month_order = sorted(tot["Year_Month"].unique())
tot["Year_Month"] = pd.Categorical(tot["Year_Month"], categories=month_order, ordered=True)
tot = tot.sort_values("Year_Month")

fig_tot = px.area(
    tot, x="Year_Month", y="Share", color="mode",
    color_discrete_map=mode_colors,
    labels={"Year_Month": "Month", "Share": "% of month", "mode": ""},
    category_orders={"Year_Month": month_order},
)
fig_tot.update_traces(
    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<extra></extra>"
)
fig_tot.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(type="category", tickangle=45, tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#eeeeee", ticksuffix="%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=340,
)
st.plotly_chart(fig_tot, width='stretch')

st.divider()

# ── Theme explorer ────────────────────────────────────────────────────────────
st.subheader("Explore incidents by theme")

theme_pick = st.selectbox(
    "Select a theme",
    sorted(df["theme"].unique()),
)

sub = df[df["theme"] == theme_pick].copy()
sub = sub.sort_values("Health Workers Killed", ascending=False)

st.caption(f"{len(sub):,} incidents in this theme")

for _, row in sub.head(8).iterrows():
    date   = row["Date"].date() if pd.notna(row["Date"]) else "Unknown"
    region = row.get("Region", "Unknown")
    killed = int(row.get("Health Workers Killed", 0) or 0)
    injured = int(row.get("Health Workers Injured", 0) or 0)
    desc   = str(row.get("description_clean", ""))

    cas = []
    if killed:  cas.append(f"<span style='color:#c0392b;font-weight:600'>{killed} killed</span>")
    if injured: cas.append(f"<span style='color:#e8916a;font-weight:600'>{injured} injured</span>")
    cas_str = " · ".join(cas) if cas else "<span style='color:#888'>No casualties recorded</span>"

    st.markdown(
        f"""
        <div style='background:#ffffff;border-left:4px solid #c0392b;
                    border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
            <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
                <span style='font-weight:600;color:#333;font-size:13px'>{date} · {region}</span>
                <span style='font-size:12px'>{cas_str}</span>
            </div>
            <div style='color:#555;font-size:13px;line-height:1.6'>{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption(
    "Method: BERTopic with all-MiniLM-L6-v2 sentence embeddings on incident descriptions. "
    "Themes are unsupervised — discovered from the text, not predefined. Topics were grouped "
    "into three modes of attack for interpretability. Incidents below the model's confidence "
    "threshold (Unclassified) are excluded from this view."
)
