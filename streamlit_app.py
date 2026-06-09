import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
import google.generativeai as genai

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the very first st call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Behavior Analytics & Recommendation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — mirrors the React app's Indigo/Slate design system
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Slate page background */
    .main .block-container {
        background: #f8fafc;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .sidebar-title {
        color: #f1f5f9 !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    section[data-testid="stSidebar"] .stMultiSelect > div > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Page header ── */
    .dash-header {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .dash-header h1 {
        color: white !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 1.2;
    }
    .dash-header p {
        color: #c7d2fe !important;
        font-size: 0.875rem !important;
        margin: 4px 0 0 0 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0 16px;
        border: none;
        transition: all 0.18s ease;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #4f46e5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.10);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    /* ── KPI Metric Cards ── */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #818cf8;
        box-shadow: 0 4px 12px rgba(99,102,241,0.12);
    }
    div[data-testid="metric-container"] > label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1.15;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        color: #10b981 !important;
    }

    /* ── Section headings ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 16px;
    }

    /* ── Card wrapper ── */
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Insight / AI response cards ── */
    .insight-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 1px 4px rgba(99,102,241,0.07);
    }
    .insight-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #6366f1;
        margin-bottom: 6px;
    }
    .insight-text {
        font-size: 0.9rem;
        line-height: 1.65;
        color: #334155;
    }

    /* ── Product recommendation cards ── */
    .rec-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        transition: box-shadow 0.18s, border-color 0.18s;
    }
    .rec-card:hover {
        border-color: #818cf8;
        box-shadow: 0 3px 10px rgba(99,102,241,0.10);
    }
    .rec-rank {
        background: #eef2ff;
        color: #4f46e5;
        font-weight: 700;
        font-size: 0.85rem;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .rec-name {
        font-weight: 600;
        font-size: 0.92rem;
        color: #1e293b;
    }
    .rec-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 3px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-green  { background:#dcfce7; color:#15803d; }
    .badge-indigo { background:#eef2ff; color:#4f46e5; }
    .badge-amber  { background:#fef3c7; color:#d97706; }
    .badge-rose   { background:#ffe4e6; color:#e11d48; }
    .badge-slate  { background:#f1f5f9; color:#475569; }

    /* ── Segment pills ── */
    .seg-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 3px;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }

    /* ── Divider ── */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }

    /* ── Plotly chart backgrounds ── */
    .js-plotly-plot .plotly .modebar {
        top: 4px !important;
    }

    /* ── Status box ── */
    div[data-testid="stStatus"] {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

    /* ── Sidebar filter section header ── */
    .filter-section-header {
        color: #475569 !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 16px 0 6px 0;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SHEET_URL = "https://docs.google.com/spreadsheets/d/1aX6GynZaDd3leFEWfU16wnpxP-PjgyCiaetOkHvCooQ/export?format=csv"
REFERENCE_DATE = pd.Timestamp("2024-12-31")

SEGMENT_COLORS = {
    "High Value Customers":  "#10b981",
    "Loyal Customers":       "#6366f1",
    "Frequent Buyers":       "#f59e0b",
    "Potential Loyalists":   "#3b82f6",
    "New Customers":         "#8b5cf6",
    "At Risk Customers":     "#ec4899",
    "Lost Customers":        "#64748b",
    "Regular Customers":     "#94a3b8",
}

SEGMENT_BADGE_CLASSES = {
    "High Value Customers":  "badge-green",
    "Loyal Customers":       "badge-indigo",
    "Frequent Buyers":       "badge-amber",
    "Potential Loyalists":   "badge-indigo",
    "New Customers":         "badge-slate",
    "At Risk Customers":     "badge-rose",
    "Lost Customers":        "badge-slate",
    "Regular Customers":     "badge-slate",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#334155"),
    margin=dict(l=16, r=16, t=42, b=16),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=11),
    ),
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_preprocess_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(SHEET_URL)
    except Exception as e:
        st.error(f"Failed to fetch dataset: {e}")
        return pd.DataFrame()

    # ── Clean categories ──
    df["Purchase_Category"] = df["Purchase_Category"].astype(str).str.strip()
    df["Purchase_Category"] = df["Purchase_Category"].replace({
        "Travel & Leisure (Flights": "Travel & Leisure",
        "Packages)": "Travel & Leisure",
    })

    # ── Numeric & string cleanup ──
    df["Purchase_Amount"] = pd.to_numeric(df["Purchase_Amount"], errors="coerce").fillna(150.0)
    df["Device_Used_for_Shopping"] = df["Device_Used_for_Shopping"].replace({"High": "Smartphone"}).fillna("Smartphone")
    df["Discount_Sensitivity"] = df["Discount_Sensitivity"].replace({"Medium": "Somewhat Sensitive"}).fillna("Somewhat Sensitive")
    df["Purchase_Channel"] = df["Purchase_Channel"].replace({"7": "Mixed"}).fillna("Online")
    df["Product_Rating"] = pd.to_numeric(df["Product_Rating"], errors="coerce").fillna(4).astype(int)
    df["Customer_Satisfaction"] = pd.to_numeric(df["Customer_Satisfaction"], errors="coerce").fillna(7).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(35).astype(int)
    df["Brand_Loyalty"] = pd.to_numeric(df["Brand_Loyalty"], errors="coerce").fillna(3).astype(int)
    df["Return_Rate"] = pd.to_numeric(df["Return_Rate"], errors="coerce").fillna(0.0)
    df["Time_of_Purchase"] = pd.to_datetime(df["Time_of_Purchase"], errors="coerce").fillna(pd.Timestamp("2024-06-01"))
    df["Discount_Used"] = df["Discount_Used"].astype(str).str.upper() == "TRUE"
    df["Loyalty_Member"] = df["Loyalty_Member"].astype(str).str.upper() == "TRUE"

    # ── Recency ──
    df["recencyDays"] = (REFERENCE_DATE - df["Time_of_Purchase"]).dt.days.clip(lower=0)

    # ── Date fields ──
    if "Purchase_Month" not in df.columns:
        df["Purchase_Month"] = df["Time_of_Purchase"].dt.month
    if "Purchase_Month_Name" not in df.columns:
        df["Purchase_Month_Name"] = df["Time_of_Purchase"].dt.strftime("%b")
    if "Purchase_Year" not in df.columns:
        df["Purchase_Year"] = df["Time_of_Purchase"].dt.year
    if "Purchase_DayOfWeek" not in df.columns:
        df["Purchase_DayOfWeek"] = df["Time_of_Purchase"].dt.day_name()
    if "Purchase_Quarter" not in df.columns:
        df["Purchase_Quarter"] = df["Time_of_Purchase"].dt.quarter
    df["Purchase_Month"] = pd.to_numeric(df["Purchase_Month"], errors="coerce").fillna(6).astype(int)

    # ── RFM scoring per customer ──
    cust = df.groupby("Customer_ID").agg(
        totalSpend=("Purchase_Amount", "sum"),
        count=("Purchase_Amount", "count"),
        minRecency=("recencyDays", "min"),
    ).reset_index()

    def recency_score(d):
        if d <= 30: return 5
        elif d <= 90: return 4
        elif d <= 180: return 3
        elif d <= 270: return 2
        return 1

    def frequency_score(c):
        if c >= 10: return 5
        elif c >= 5: return 4
        elif c >= 3: return 3
        elif c >= 2: return 2
        return 1

    def monetary_score(s):
        if s >= 2000: return 5
        elif s >= 1000: return 4
        elif s >= 500: return 3
        elif s >= 250: return 2
        return 1

    def rfm_segment(row):
        r, f, m = row["recencyScore"], row["frequencyScore"], row["monetaryScore"]
        if r >= 4 and m >= 4: return "High Value Customers"
        elif f >= 4 and r >= 3: return "Loyal Customers"
        elif f >= 4 and r <= 2: return "Frequent Buyers"
        elif r >= 4 and f >= 2: return "Potential Loyalists"
        elif r >= 4 and f == 1: return "New Customers"
        elif r <= 2 and (f >= 3 or m >= 3): return "At Risk Customers"
        elif r <= 2 and f <= 2: return "Lost Customers"
        return "Regular Customers"

    cust["recencyScore"] = cust["minRecency"].apply(recency_score)
    cust["frequencyScore"] = cust["count"].apply(frequency_score)
    cust["monetaryScore"] = cust["totalSpend"].apply(monetary_score)
    cust["rfmSegment"] = cust.apply(rfm_segment, axis=1)

    df = df.drop(columns=["rfmSegment", "recencyScore", "frequencyScore", "monetaryScore"], errors="ignore")
    df = df.merge(
        cust[["Customer_ID", "rfmSegment", "recencyScore", "frequencyScore", "monetaryScore", "count", "minRecency"]],
        on="Customer_ID",
        how="left",
    )
    df = df.rename(columns={"count": "Customer_Total_Orders", "minRecency": "Customer_Recency_Days"})
    return df


# ─────────────────────────────────────────────
# AI INSIGHTS (Gemini)
# ─────────────────────────────────────────────
def build_aggregates(df: pd.DataFrame) -> dict:
    total_rev = float(df["Purchase_Amount"].sum())
    unique_cust = int(df["Customer_ID"].nunique())
    aov = total_rev / len(df) if len(df) > 0 else 0
    avg_sat = float(df["Customer_Satisfaction"].mean())

    cat_dist = (
        df.groupby("Purchase_Category")["Purchase_Amount"]
        .sum()
        .reset_index()
        .sort_values("Purchase_Amount", ascending=False)
    )
    cat_list = [
        {
            "name": row["Purchase_Category"],
            "revenuePercentage": round(row["Purchase_Amount"] / total_rev * 100, 1),
        }
        for _, row in cat_dist.iterrows()
    ]

    seg_dist = (
        df.groupby("rfmSegment")
        .agg(count=("Customer_ID", "nunique"), revenue=("Purchase_Amount", "sum"))
        .reset_index()
    )
    seg_list = [
        {
            "name": row["rfmSegment"],
            "count": int(row["count"]),
            "percentage": round(row["count"] / unique_cust * 100, 1),
        }
        for _, row in seg_dist.iterrows()
    ]

    monthly = df.groupby("Purchase_Month_Name")["Purchase_Amount"].sum()
    top2_months = monthly.nlargest(2).index.tolist()

    daily = df.groupby("Purchase_DayOfWeek")["Purchase_Amount"].sum()
    top2_days = daily.nlargest(2).index.tolist()

    return {
        "totalRevenue": total_rev,
        "totalCustomers": unique_cust,
        "averageOrderValue": aov,
        "avgSatisfaction": avg_sat,
        "categoryDistribution": cat_list,
        "segmentContribution": seg_list,
        "peakMonths": top2_months,
        "peakDays": top2_days,
    }


def get_ai_insights(aggregates: dict, api_key: str) -> dict:
    """Call Gemini for executive insights; fall back gracefully."""
    if not api_key:
        return _fallback_insights(aggregates)

    prompt = f"""
You are a Senior Strategic E-Commerce Data Analyst and AI Growth Officer.
Examine this aggregate data summary from our transactional dataset:

- Total Revenue: ${aggregates["totalRevenue"]:,.0f}
- Total Customers: {aggregates["totalCustomers"]}
- Average Order Value (AOV): ${aggregates["averageOrderValue"]:.2f}
- Average Customer Satisfaction: {aggregates["avgSatisfaction"]:.2f}/10
- Top Customer Segments (RFM): {json.dumps(aggregates["segmentContribution"])}
- Revenue by Category: {json.dumps(aggregates["categoryDistribution"])}
- Peak Sales Months: {aggregates["peakMonths"]}
- Peak Purchase Days: {aggregates["peakDays"]}

Provide deep, action-oriented, professional business analysis.
Return ONLY a strict JSON object with exactly these 5 keys:
1. "customerInsights"
2. "salesInsights"
3. "productInsights"
4. "segmentInsights"
5. "recommendationInsights"

Each value is a paragraph string. No markdown, no extra keys, no preamble.
"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception:
        return _fallback_insights(aggregates)


def _fallback_insights(agg: dict) -> dict:
    cats = agg.get("categoryDistribution", [])
    segs = agg.get("segmentContribution", [])
    top_cat = cats[0]["name"] if cats else "top category"
    high_val = next((s for s in segs if "High Value" in s["name"]), None)
    at_risk  = next((s for s in segs if "At Risk"   in s["name"]), None)
    return {
        "customerInsights": (
            f"Our base of {agg['totalCustomers']:,} customers has an AOV of ${agg['averageOrderValue']:.2f}. "
            f"Satisfaction scores at {agg['avgSatisfaction']:.1f}/10 signal strong product-market fit."
        ),
        "salesInsights": (
            f"Peak revenue occurs in {', '.join(agg.get('peakMonths', []))}. "
            f"Highest purchase activity falls on {' and '.join(agg.get('peakDays', []))}."
        ),
        "productInsights": (
            f"'{top_cat}' drives the largest revenue share at {cats[0]['revenuePercentage'] if cats else 0}%. "
            f"Prioritise inventory and landing page budget here."
        ),
        "segmentInsights": (
            (f"{high_val['name']} ({high_val['percentage']}%) are high-priority retention targets. " if high_val else "")
            + (f"{at_risk['name']} ({at_risk['percentage']}%) require immediate win-back campaigns." if at_risk else "")
        ),
        "recommendationInsights": (
            f"Cross-selling complementary items alongside '{top_cat}' during peak periods can increase basket size. "
            f"Personalised push notifications on preferred devices will lift repeat purchase rates."
        ),
    }


# ─────────────────────────────────────────────
# AGENT TOOLS (for AI Copilot tab)
# ─────────────────────────────────────────────
def tool_analyze_rfm_segment(segment_name: str, df: pd.DataFrame) -> str:
    name_lower = segment_name.lower()
    matched = None
    for seg in df["rfmSegment"].unique():
        if name_lower in seg.lower() or seg.lower() in name_lower:
            matched = seg
            break
    if not matched:
        available = ", ".join(df["rfmSegment"].unique())
        return f"Could not match segment '{segment_name}'. Available: {available}"
    seg_df = df[df["rfmSegment"] == matched]
    return f"""=== SEGMENT ANALYSIS: {matched} ===
• Transactions: {len(seg_df):,} ({len(seg_df)/len(df)*100:.1f}% of total)
• Total Spend: ${seg_df['Purchase_Amount'].sum():,.2f}
• Avg Satisfaction: {seg_df['Customer_Satisfaction'].mean():.1f}/10
• Avg Age: {seg_df['Age'].mean():.0f} years
• Top Channel: {seg_df['Purchase_Channel'].mode()[0] if not seg_df.empty else "N/A"}
• Top Category: {seg_df['Purchase_Category'].mode()[0] if not seg_df.empty else "N/A"}
"""


def tool_get_product_synergies(category_name: str, df: pd.DataFrame) -> str:
    cat_lower = category_name.lower()
    matched = None
    for cat in df["Purchase_Category"].unique():
        if cat_lower in cat.lower() or cat.lower() in cat_lower:
            matched = cat
            break
    if not matched:
        available = ", ".join(df["Purchase_Category"].unique())
        return f"Could not match category '{category_name}'. Available: {available}"
    cat_df = df[df["Purchase_Category"] == matched]
    cross_sell = df[df["Purchase_Category"] != matched]["Purchase_Category"].mode()
    return f"""=== PRODUCT SYNERGIES: {matched} ===
• Avg Transaction Value: ${cat_df['Purchase_Amount'].mean():,.2f}
• Top Product: {cat_df['Product_Name'].mode()[0] if not cat_df.empty else "N/A"}
• Brand Loyalty ≥4★: {(cat_df['Brand_Loyalty'] >= 4).sum() / max(len(cat_df),1) * 100:.1f}%
• Avg Return Rate: {cat_df['Return_Rate'].mean() * 100:.1f}%
• Best Cross-sell: {cross_sell[0] if not cross_sell.empty else "N/A"}
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fmt_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.2f}"


def segment_badge(seg: str) -> str:
    cls = SEGMENT_BADGE_CLASSES.get(seg, "badge-slate")
    return f'<span class="badge {cls}">{seg}</span>'


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    df_clean = load_and_preprocess_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="color:#6366f1;font-weight:700;font-size:1.1rem;margin-bottom:2px;">📊 Analytics</p>'
        '<p style="color:#475569;font-size:0.78rem;margin-bottom:16px;">Customer Behavior Platform</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    if df_clean.empty:
        st.warning("No data available.")
        filtered_df = pd.DataFrame()
    else:
        st.markdown('<p class="filter-section-header">Filters</p>', unsafe_allow_html=True)

        locations = sorted(df_clean["Location"].dropna().unique())
        selected_locations = st.multiselect("Geography", locations, default=locations)

        devices = sorted(df_clean["Device_Used_for_Shopping"].dropna().unique())
        selected_devices = st.multiselect("Shopping Device", devices, default=devices)

        channels = sorted(df_clean["Purchase_Channel"].dropna().unique())
        selected_channels = st.multiselect("Purchase Channel", channels, default=channels)

        genders = sorted(df_clean["Gender"].dropna().unique())
        selected_genders = st.multiselect("Gender", genders, default=genders)

        year_opts = sorted(df_clean["Purchase_Year"].dropna().unique().tolist())
        selected_years = st.multiselect("Year", year_opts, default=year_opts)

        filtered_df = df_clean[
            df_clean["Location"].isin(selected_locations)
            & df_clean["Device_Used_for_Shopping"].isin(selected_devices)
            & df_clean["Purchase_Channel"].isin(selected_channels)
            & df_clean["Gender"].isin(selected_genders)
            & df_clean["Purchase_Year"].isin(selected_years)
        ]

        st.divider()
        st.markdown(
            f'<p style="font-size:0.76rem;color:#64748b;text-align:center;">'
            f'{len(filtered_df):,} transactions · {filtered_df["Customer_ID"].nunique():,} customers</p>',
            unsafe_allow_html=True,
        )

    # ── Gemini API key ──
    st.divider()
    st.markdown('<p class="filter-section-header">Gemini API Key</p>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AIza…",
        label_visibility="collapsed",
    )
    if gemini_key:
        st.success("✓ Key configured", icon="🔑")
    else:
        st.caption("Add key to enable AI insights & Copilot.")

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div style="font-size:2.4rem;">📊</div>
  <div>
    <h1>Customer Behavior Analytics & Recommendation System</h1>
    <p>RFM segmentation · Collaborative filtering · Time-series forecasting · Gemini AI strategic insights</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────
if filtered_df.empty:
    st.warning("No transactions match the selected filters. Adjust the sidebar controls.")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_exec, tab_seg, tab_rec, tab_sales, tab_copilot = st.tabs([
    "📈  Executive Board",
    "👥  RFM Segmentation",
    "🎯  Recommendation Engine",
    "🗓️  Sales Seasonality",
    "🧠  AI Growth Copilot",
])

# ══════════════════════════════════════════════
# TAB 1 — EXECUTIVE BOARD
# ══════════════════════════════════════════════
with tab_exec:
    total_rev    = filtered_df["Purchase_Amount"].sum()
    total_sales  = len(filtered_df)
    aov          = total_rev / total_sales if total_sales else 0
    unique_cust  = filtered_df["Customer_ID"].nunique()
    avg_rating   = filtered_df["Product_Rating"].mean()
    avg_sat      = filtered_df["Customer_Satisfaction"].mean()
    loyalty_pct  = filtered_df["Loyalty_Member"].mean() * 100
    discount_pct = filtered_df["Discount_Used"].mean() * 100

    # ── KPI row ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Revenue",      fmt_currency(total_rev),  "+6.4% YoY")
    k2.metric("Total Orders",       f"{total_sales:,}",       "Transactions")
    k3.metric("Avg Order Value",    f"${aov:,.2f}",           "Per transaction")
    k4.metric("Active Customers",   f"{unique_cust:,}",       "Unique buyers")
    k5.metric("Avg Product Rating", f"{avg_rating:.2f} / 5",  "★ Stars")
    k6.metric("Loyalty Members",    f"{loyalty_pct:.1f}%",    "of customers")

    st.markdown("---")

    # ── Revenue trend + category share ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-label">Revenue Trend</p><p class="section-title">Monthly Sales Performance</p>', unsafe_allow_html=True)
        monthly = (
            filtered_df
            .groupby(["Purchase_Year", "Purchase_Month", "Purchase_Month_Name"])
            .agg(Revenue=("Purchase_Amount", "sum"), Orders=("Purchase_Amount", "count"))
            .reset_index()
            .sort_values(["Purchase_Year", "Purchase_Month"])
        )
        monthly["Period"] = monthly["Purchase_Month_Name"] + " " + monthly["Purchase_Year"].astype(str)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly["Period"], y=monthly["Revenue"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=6, color="#6366f1"),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            name="Revenue",
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
        ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickprefix="$", tickformat=","),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-label">Category Mix</p><p class="section-title">Revenue Share</p>', unsafe_allow_html=True)
        cat_share = (
            filtered_df
            .groupby("Purchase_Category")["Purchase_Amount"].sum()
            .reset_index()
            .sort_values("Purchase_Amount", ascending=False)
        )
        if len(cat_share) > 5:
            top = cat_share.iloc[:4].copy()
            other_val = cat_share.iloc[4:]["Purchase_Amount"].sum()
            top = pd.concat([top, pd.DataFrame([{"Purchase_Category": "Other", "Purchase_Amount": other_val}])])
            cat_share = top

        fig_pie = px.pie(
            cat_share,
            values="Purchase_Amount",
            names="Purchase_Category",
            hole=0.52,
            color_discrete_sequence=["#6366f1","#818cf8","#a5b4fc","#c7d2fe","#e0e7ff","#94a3b8"],
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ── Channel + Device breakdown ──
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<p class="section-label">Channels</p><p class="section-title">Revenue by Channel</p>', unsafe_allow_html=True)
        ch_rev = filtered_df.groupby("Purchase_Channel")["Purchase_Amount"].sum().reset_index().sort_values("Purchase_Amount")
        fig_ch = px.bar(ch_rev, x="Purchase_Amount", y="Purchase_Channel", orientation="h",
                        color_discrete_sequence=["#6366f1"],
                        labels={"Purchase_Amount": "Revenue ($)", "Purchase_Channel": ""})
        fig_ch.update_layout(**PLOTLY_LAYOUT, height=220, xaxis=dict(tickprefix="$", tickformat=","), showlegend=False)
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-label">Devices</p><p class="section-title">Device Usage Split</p>', unsafe_allow_html=True)
        dev_cnt = filtered_df["Device_Used_for_Shopping"].value_counts().reset_index()
        dev_cnt.columns = ["Device", "Count"]
        fig_dev = px.bar(dev_cnt, x="Device", y="Count", color_discrete_sequence=["#818cf8"])
        fig_dev.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
        st.plotly_chart(fig_dev, use_container_width=True)

    with col_c:
        st.markdown('<p class="section-label">Payment</p><p class="section-title">Payment Methods</p>', unsafe_allow_html=True)
        pay_cnt = filtered_df["Payment_Method"].value_counts().reset_index()
        pay_cnt.columns = ["Method", "Count"]
        fig_pay = px.bar(pay_cnt, x="Method", y="Count", color_discrete_sequence=["#a5b4fc"])
        fig_pay.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False,
                               xaxis=dict(tickangle=-20))
        st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown("---")

    # ── Top transactions table ──
    st.markdown('<p class="section-title">Top Transactions</p><p class="section-sub">Highest-value individual purchases in the current filter</p>', unsafe_allow_html=True)
    top_tx = (
        filtered_df[["Customer_ID", "Customer_Name", "Age", "Gender", "Location", "Purchase_Category", "Product_Name", "Purchase_Amount", "rfmSegment"]]
        .sort_values("Purchase_Amount", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.dataframe(
        top_tx,
        column_config={
            "Purchase_Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "rfmSegment":      st.column_config.TextColumn("Segment"),
        },
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── AI Insights panel ──
    st.markdown('<p class="section-label">Gemini AI Analysis</p><p class="section-title">Executive Strategic Insights</p><p class="section-sub">Generated by Gemini AI based on the current filtered dataset</p>', unsafe_allow_html=True)

    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = None

    btn_col, _ = st.columns([1, 4])
    with btn_col:
        run_ai = st.button("✨  Generate AI Insights", use_container_width=True, type="primary")

    if run_ai:
        with st.spinner("Calling Gemini API…"):
            agg = build_aggregates(filtered_df)
            st.session_state.ai_insights = get_ai_insights(agg, gemini_key)

    if st.session_state.ai_insights:
        ins = st.session_state.ai_insights
        LABELS = [
            ("customerInsights",     "👤 Customer Insights"),
            ("salesInsights",        "📈 Sales Insights"),
            ("productInsights",      "🛍️ Product Insights"),
            ("segmentInsights",      "🎯 Segment Insights"),
            ("recommendationInsights","💡 Recommendations"),
        ]
        cols_ins = st.columns(2)
        for i, (key, label) in enumerate(LABELS):
            text = ins.get(key, "")
            if text:
                with cols_ins[i % 2]:
                    st.markdown(
                        f'<div class="insight-card">'
                        f'<div class="insight-label">{label}</div>'
                        f'<div class="insight-text">{text}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Click **Generate AI Insights** above to receive Gemini-powered strategic analysis of the current dataset.", icon="✨")


# ══════════════════════════════════════════════
# TAB 2 — RFM SEGMENTATION
# ══════════════════════════════════════════════
with tab_seg:
    st.markdown('<p class="section-label">Customer Intelligence</p><p class="section-title">RFM Segmentation Dashboard</p><p class="section-sub">Recency · Frequency · Monetary scoring across the customer base</p>', unsafe_allow_html=True)

    # ── Segment KPI grid ──
    seg_stats = (
        filtered_df
        .groupby("rfmSegment")
        .agg(
            Customers=("Customer_ID", "nunique"),
            Revenue=("Purchase_Amount", "sum"),
            AvgSat=("Customer_Satisfaction", "mean"),
            AvgRating=("Product_Rating", "mean"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    seg_cols = st.columns(min(len(seg_stats), 4))
    for i, row in seg_stats.iterrows():
        col = seg_cols[i % 4]
        color = SEGMENT_COLORS.get(row["rfmSegment"], "#94a3b8")
        col.markdown(
            f'<div style="background:white;border:1px solid #e2e8f0;border-top:4px solid {color};'
            f'border-radius:12px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:{color};margin-bottom:4px;">'
            f'{row["rfmSegment"]}</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#0f172a;">{row["Customers"]:,}</div>'
            f'<div style="font-size:0.75rem;color:#64748b;margin-top:2px;">{fmt_currency(row["Revenue"])} revenue</div>'
            f'<div style="font-size:0.75rem;color:#64748b;">Sat {row["AvgSat"]:.1f}/10  ★{row["AvgRating"]:.1f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Scatter + bar side by side ──
    sc_col, bar_col = st.columns([3, 2])

    with sc_col:
        st.markdown('<p class="section-title">Monetary vs Recency (bubble = order count)</p>', unsafe_allow_html=True)
        cust_profile = (
            filtered_df
            .groupby("Customer_ID")
            .agg(
                TotalSpend=("Purchase_Amount", "sum"),
                Orders=("Customer_Total_Orders", "first"),
                Recency=("Customer_Recency_Days", "first"),
                Segment=("rfmSegment", "first"),
            )
            .reset_index()
        )
        fig_sc = px.scatter(
            cust_profile,
            x="Recency", y="TotalSpend",
            size="Orders", color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            labels={"Recency": "Days Since Last Order", "TotalSpend": "Total Spend ($)"},
            hover_data={"Customer_ID": True, "Orders": True},
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    with bar_col:
        st.markdown('<p class="section-title">Revenue Contribution by Segment</p>', unsafe_allow_html=True)
        seg_rev = seg_stats.sort_values("Revenue")
        colors_bar = [SEGMENT_COLORS.get(s, "#6366f1") for s in seg_rev["rfmSegment"]]
        fig_seg_bar = go.Figure(go.Bar(
            y=seg_rev["rfmSegment"],
            x=seg_rev["Revenue"],
            orientation="h",
            marker_color=colors_bar,
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
        ))
        fig_seg_bar.update_layout(**PLOTLY_LAYOUT, height=380,
                                   xaxis=dict(tickprefix="$", tickformat=",", showgrid=True, gridcolor="#f1f5f9"),
                                   showlegend=False)
        st.plotly_chart(fig_seg_bar, use_container_width=True)

    st.markdown("---")

    # ── RFM score distribution ──
    st.markdown('<p class="section-title">RFM Score Distributions</p>', unsafe_allow_html=True)
    rfm_cust = filtered_df.drop_duplicates("Customer_ID")[["Customer_ID", "recencyScore", "frequencyScore", "monetaryScore", "rfmSegment"]]
    r1, r2, r3 = st.columns(3)
    for col_w, score_col, label in [(r1, "recencyScore", "Recency Score"), (r2, "frequencyScore", "Frequency Score"), (r3, "monetaryScore", "Monetary Score")]:
        dist = rfm_cust[score_col].value_counts().sort_index().reset_index()
        dist.columns = ["Score", "Count"]
        fig_d = px.bar(dist, x="Score", y="Count", color_discrete_sequence=["#6366f1"], labels={"Score": label, "Count": "Customers"})
        fig_d.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
        col_w.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")

    # ── Demographics breakdown ──
    st.markdown('<p class="section-title">Demographic Profile by Segment</p>', unsafe_allow_html=True)
    selected_seg = st.selectbox("Select segment to inspect", options=["All Segments"] + list(seg_stats["rfmSegment"]))
    demo_df = filtered_df if selected_seg == "All Segments" else filtered_df[filtered_df["rfmSegment"] == selected_seg]

    d1, d2, d3 = st.columns(3)
    with d1:
        gen_dist = demo_df["Gender"].value_counts().reset_index()
        gen_dist.columns = ["Gender", "Count"]
        fig_g = px.pie(gen_dist, names="Gender", values="Count", hole=0.45,
                       color_discrete_sequence=["#6366f1","#a5b4fc","#c7d2fe"])
        fig_g.update_layout(**PLOTLY_LAYOUT, height=220, title_text="Gender", title_x=0.5)
        d1.plotly_chart(fig_g, use_container_width=True)

    with d2:
        age_bins = pd.cut(demo_df["Age"], bins=[0,25,35,45,55,100], labels=["<25","25-34","35-44","45-54","55+"])
        age_dist = age_bins.value_counts().sort_index().reset_index()
        age_dist.columns = ["Age Group", "Count"]
        fig_age = px.bar(age_dist, x="Age Group", y="Count", color_discrete_sequence=["#818cf8"])
        fig_age.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False, title_text="Age Groups", title_x=0.5)
        d2.plotly_chart(fig_age, use_container_width=True)

    with d3:
        loc_dist = demo_df["Location"].value_counts().reset_index()
        loc_dist.columns = ["Location", "Count"]
        fig_loc = px.bar(loc_dist.head(6), x="Count", y="Location", orientation="h",
                         color_discrete_sequence=["#a5b4fc"])
        fig_loc.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False, title_text="Locations", title_x=0.5)
        d3.plotly_chart(fig_loc, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════
with tab_rec:
    st.markdown('<p class="section-label">Collaborative Filtering</p><p class="section-title">Product Recommendation Engine</p><p class="section-sub">Heuristic basket analysis using demographic + category affinity signals</p>', unsafe_allow_html=True)

    ctrl_col, result_col = st.columns([1, 2])

    with ctrl_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Customer Profile Selector**")
        target_cat    = st.selectbox("Purchase Category",  sorted(filtered_df["Purchase_Category"].unique()))
        target_gender = st.radio("Gender Focus",           sorted(filtered_df["Gender"].unique()))
        target_device = st.selectbox("Preferred Device",   sorted(filtered_df["Device_Used_for_Shopping"].unique()))
        target_seg    = st.selectbox("RFM Segment Focus",  ["Any"] + sorted(filtered_df["rfmSegment"].unique()))
        top_n         = st.slider("Top N Recommendations", 3, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown('<p class="section-title">Top Matched Products</p>', unsafe_allow_html=True)

        mask = (
            (filtered_df["Purchase_Category"] == target_cat)
            & (filtered_df["Gender"] == target_gender)
            & (filtered_df["Device_Used_for_Shopping"] == target_device)
        )
        if target_seg != "Any":
            mask &= filtered_df["rfmSegment"] == target_seg

        matches = filtered_df[mask]
        if matches.empty:
            matches = filtered_df[filtered_df["Purchase_Category"] == target_cat]

        recs = (
            matches
            .groupby("Product_Name")
            .agg(
                Popularity=("Purchase_Amount", "count"),
                AvgRating=("Product_Rating", "mean"),
                Revenue=("Purchase_Amount", "sum"),
                ReturnRate=("Return_Rate", "mean"),
                BrandLoyalty=("Brand_Loyalty", "mean"),
            )
            .reset_index()
            .sort_values("Popularity", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        for idx, rec in recs.iterrows():
            stars = "★" * round(rec["AvgRating"]) + "☆" * (5 - round(rec["AvgRating"]))
            loyalty_badge = "badge-green" if rec["BrandLoyalty"] >= 4 else ("badge-amber" if rec["BrandLoyalty"] >= 3 else "badge-rose")
            st.markdown(
                f'<div class="rec-card">'
                f'  <div class="rec-rank">#{idx+1}</div>'
                f'  <div style="flex:1;">'
                f'    <div class="rec-name">{rec["Product_Name"]}</div>'
                f'    <div class="rec-meta">'
                f'      <span title="Rating">{stars} {rec["AvgRating"]:.1f}</span>&nbsp;&nbsp;'
                f'      <span title="Popularity">📦 {rec["Popularity"]} sales</span>&nbsp;&nbsp;'
                f'      <span title="Revenue">💵 {fmt_currency(rec["Revenue"])}</span>'
                f'    </div>'
                f'    <div style="margin-top:5px;">'
                f'      <span class="badge {loyalty_badge}">Loyalty {rec["BrandLoyalty"]:.1f}/5</span>'
                f'      <span class="badge badge-slate">Return {rec["ReturnRate"]*100:.1f}%</span>'
                f'    </div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Category cross-sell matrix ──
    st.markdown('<p class="section-title">Category Cross-Sell Affinity Matrix</p><p class="section-sub">Correlation of purchase amounts across categories per customer</p>', unsafe_allow_html=True)
    pivot = filtered_df.pivot_table(
        index="Customer_ID", columns="Purchase_Category", values="Purchase_Amount", aggfunc="sum"
    ).fillna(0)
    corr = pivot.corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=420, xaxis=dict(tickangle=-35))
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — SALES SEASONALITY
# ══════════════════════════════════════════════
with tab_sales:
    st.markdown('<p class="section-label">Temporal Analytics</p><p class="section-title">Sales Seasonality & Heatmap</p><p class="section-sub">Identify peak order periods across weekdays, months, and quarters</p>', unsafe_allow_html=True)

    # ── Heatmap ──
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = filtered_df.pivot_table(
        index="Purchase_DayOfWeek",
        columns="Purchase_Month_Name",
        values="Purchase_Amount",
        aggfunc="sum",
    ).fillna(0)
    heat = heat.reindex([d for d in day_order if d in heat.index])

    # Sort columns by month order
    month_order_abbr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heat = heat[[c for c in month_order_abbr if c in heat.columns]]

    fig_heat = px.imshow(
        heat,
        labels=dict(x="Month", y="Day of Week", color="Revenue ($)"),
        color_continuous_scale="Plasma",
        aspect="auto",
        text_auto=False,
    )
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Quarterly + Day-of-week bars ──
    q_col, d_col = st.columns(2)

    with q_col:
        st.markdown('<p class="section-title">Quarterly Revenue</p>', unsafe_allow_html=True)
        q_rev = filtered_df.groupby("Purchase_Quarter")["Purchase_Amount"].sum().reset_index()
        q_rev["Quarter"] = "Q" + q_rev["Purchase_Quarter"].astype(str)
        fig_q = px.bar(q_rev, x="Quarter", y="Purchase_Amount",
                       color_discrete_sequence=["#6366f1"],
                       labels={"Purchase_Amount": "Revenue ($)", "Quarter": ""})
        fig_q.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                             yaxis=dict(tickprefix="$", tickformat=","))
        st.plotly_chart(fig_q, use_container_width=True)

    with d_col:
        st.markdown('<p class="section-title">Revenue by Day of Week</p>', unsafe_allow_html=True)
        dow_rev = filtered_df.groupby("Purchase_DayOfWeek")["Purchase_Amount"].sum().reset_index()
        dow_rev = dow_rev.set_index("Purchase_DayOfWeek").reindex(day_order).reset_index()
        fig_dow = px.bar(dow_rev, x="Purchase_DayOfWeek", y="Purchase_Amount",
                         color_discrete_sequence=["#818cf8"],
                         labels={"Purchase_Amount": "Revenue ($)", "Purchase_DayOfWeek": ""})
        fig_dow.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                               yaxis=dict(tickprefix="$", tickformat=","),
                               xaxis=dict(tickangle=-25))
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("---")

    # ── Purchase intent + shipping preference ──
    pi_col, sp_col = st.columns(2)

    with pi_col:
        st.markdown('<p class="section-title">Purchase Intent Distribution</p>', unsafe_allow_html=True)
        intent = filtered_df["Purchase_Intent"].value_counts().reset_index()
        intent.columns = ["Intent", "Count"]
        fig_int = px.pie(intent, names="Intent", values="Count", hole=0.4,
                         color_discrete_sequence=["#6366f1","#818cf8","#a5b4fc","#c7d2fe"])
        fig_int.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_int, use_container_width=True)

    with sp_col:
        st.markdown('<p class="section-title">Shipping Preference</p>', unsafe_allow_html=True)
        ship = filtered_df["Shipping_Preference"].value_counts().reset_index()
        ship.columns = ["Shipping", "Count"]
        fig_ship = px.bar(ship, x="Shipping", y="Count",
                          color_discrete_sequence=["#a5b4fc"],
                          labels={"Shipping": "", "Count": "Orders"})
        fig_ship.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig_ship, use_container_width=True)

    st.markdown("---")

    # ── Discount & loyalty impact ──
    st.markdown('<p class="section-title">Discount & Loyalty Impact on Revenue</p>', unsafe_allow_html=True)
    disc_col, loy_col = st.columns(2)

    with disc_col:
        disc_rev = filtered_df.groupby("Discount_Used")["Purchase_Amount"].agg(["sum","mean","count"]).reset_index()
        disc_rev["Discount_Used"] = disc_rev["Discount_Used"].map({True: "Discount Used", False: "No Discount"})
        fig_disc = px.bar(disc_rev, x="Discount_Used", y="sum",
                          color_discrete_sequence=["#6366f1","#a5b4fc"],
                          labels={"sum": "Total Revenue ($)", "Discount_Used": ""})
        fig_disc.update_layout(**PLOTLY_LAYOUT, height=250, showlegend=False,
                                yaxis=dict(tickprefix="$", tickformat=","))
        st.plotly_chart(fig_disc, use_container_width=True)

    with loy_col:
        loy_rev = filtered_df.groupby("Loyalty_Member")["Purchase_Amount"].agg(["sum","mean"]).reset_index()
        loy_rev["Loyalty_Member"] = loy_rev["Loyalty_Member"].map({True: "Loyalty Member", False: "Non-Member"})
        fig_loy = px.bar(loy_rev, x="Loyalty_Member", y="sum",
                         color_discrete_sequence=["#10b981","#6ee7b7"],
                         labels={"sum": "Total Revenue ($)", "Loyalty_Member": ""})
        fig_loy.update_layout(**PLOTLY_LAYOUT, height=250, showlegend=False,
                               yaxis=dict(tickprefix="$", tickformat=","))
        st.plotly_chart(fig_loy, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — AI GROWTH COPILOT
# ══════════════════════════════════════════════
with tab_copilot:
    st.markdown(
        '<p class="section-label">Agentic AI</p>'
        '<p class="section-title">🧠 AI Executive Growth Copilot</p>'
        '<p class="section-sub">Ask natural-language business questions. The agent selects analytics tools autonomously.</p>',
        unsafe_allow_html=True,
    )

    # ── Capability chips ──
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">'
        '<span class="badge badge-indigo">📊 Segment Analysis Tool</span>'
        '<span class="badge badge-indigo">🛍️ Product Synergy Tool</span>'
        '<span class="badge badge-green">✨ Gemini 1.5 Flash</span>'
        '<span class="badge badge-slate">💬 Multi-turn Chat</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Suggested prompts ──
    st.markdown("**Suggested questions**")
    sugg_cols = st.columns(3)
    suggestions = [
        "Tell me about High Value customers",
        "What patterns exist in the Electronics category?",
        "How are At Risk customers behaving?",
        "What are the top cross-sell opportunities in Clothing?",
        "Give me a summary of Loyal Customers",
        "Which product category has the best brand loyalty?",
    ]
    for i, sug in enumerate(suggestions):
        if sugg_cols[i % 3].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.setdefault("copilot_messages", [])
            st.session_state.copilot_messages.append({"role": "user", "content": sug})
            st.session_state.copilot_pending = sug

    st.markdown("---")

    # ── Conversation history ──
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []
    if "copilot_pending" not in st.session_state:
        st.session_state.copilot_pending = None

    for msg in st.session_state.copilot_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ──
    user_input = st.chat_input(
        "E.g., Tell me about High Value customers or which categories drive the most cross-sells?"
    )

    if user_input:
        st.session_state.copilot_messages.append({"role": "user", "content": user_input})
        st.session_state.copilot_pending = user_input
        with st.chat_message("user"):
            st.markdown(user_input)

    # ── Agent execution ──
    pending = st.session_state.get("copilot_pending")
    if pending:
        st.session_state.copilot_pending = None
        prompt_lower = pending.lower()

        segment_kws  = ["high value","loyal","frequent","potential","lost","at risk","new customer","segment","cohort"]
        category_kws = ["product","category","electronics","home","clothing","book","food","grocery","sport","beauty","appliance","travel","synergy","cross-sell"]

        matched_tools     = []
        tool_feedback_str = ""

        if any(kw in prompt_lower for kw in segment_kws):
            matched_tools.append("SegmentationTool")
            seg_target = "High Value Customers"
            if "at risk"   in prompt_lower: seg_target = "At Risk Customers"
            elif "loyal"   in prompt_lower: seg_target = "Loyal Customers"
            elif "frequent" in prompt_lower: seg_target = "Frequent Buyers"
            elif "lost"    in prompt_lower: seg_target = "Lost Customers"
            elif "new"     in prompt_lower: seg_target = "New Customers"
            elif "potential" in prompt_lower: seg_target = "Potential Loyalists"
            tool_feedback_str += tool_analyze_rfm_segment(seg_target, filtered_df) + "\n\n"

        if any(kw in prompt_lower for kw in category_kws):
            matched_tools.append("ProductSynergyTool")
            cat_target = "Electronics"
            if "home" in prompt_lower or "appliance" in prompt_lower: cat_target = "Home Appliances"
            elif "book" in prompt_lower:    cat_target = "Books"
            elif "clothing" in prompt_lower or "apparel" in prompt_lower: cat_target = "Clothing"
            elif "travel" in prompt_lower:  cat_target = "Travel & Leisure"
            elif "sport" in prompt_lower:   cat_target = "Sports"
            elif "beauty" in prompt_lower:  cat_target = "Beauty"
            elif "food" in prompt_lower or "grocery" in prompt_lower: cat_target = "Grocery"
            tool_feedback_str += tool_get_product_synergies(cat_target, filtered_df) + "\n\n"

        with st.chat_message("assistant"):
            status_label = "Agent running: " + (", ".join(matched_tools) if matched_tools else "Direct Analysis")
            with st.status(status_label, expanded=True) as status_box:
                st.write("Reviewing request context…")
                if matched_tools:
                    for mt in matched_tools:
                        desc_map = {
                            "SegmentationTool": "Demographic profiling + RFM indexing",
                            "ProductSynergyTool": "Category performance + cross-sell mining",
                        }
                        st.write(f"✅ **{mt}** — {desc_map.get(mt,'')}")
                else:
                    st.write("✅ Direct analytical reasoning (no specific tools required)")
                status_box.update(state="complete")

            # ── Build final response ──
            final_response = ""
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    agent_prompt = f"""You are a Senior E-Commerce Analytics Growth Agent.
The user asked: "{pending}"

Live analytics tool outputs:
{tool_feedback_str if tool_feedback_str else f"Dataset: {len(filtered_df):,} transactions, ${filtered_df['Purchase_Amount'].sum():,.0f} total revenue."}

Provide a concise, professional, executive-ready response. Highlight specific numbers and actionable recommendations.
"""
                    resp = model.generate_content(agent_prompt)
                    final_response = resp.text
                except Exception as e:
                    final_response = (
                        f"**Agent Analysis Summary**\n\n"
                        f"{tool_feedback_str if tool_feedback_str else 'No tool output — adjust query for segment or category focus.'}\n\n"
                        f"*(Gemini API unavailable: {e})*"
                    )
            else:
                if tool_feedback_str:
                    final_response = (
                        f"**Agent Insights (Heuristic Mode)**\n\n"
                        f"I triggered the following analytics tools:\n\n"
                        f"```\n{tool_feedback_str.strip()}\n```\n\n"
                        f"*Add a Gemini API key in the sidebar for AI-synthesised narrative insights.*"
                    )
                else:
                    total_r = filtered_df["Purchase_Amount"].sum()
                    avg_s   = filtered_df["Customer_Satisfaction"].mean()
                    final_response = (
                        f"**Dataset Overview**\n\n"
                        f"- Transactions: {len(filtered_df):,}\n"
                        f"- Revenue: {fmt_currency(total_r)}\n"
                        f"- Avg Satisfaction: {avg_s:.1f}/10\n\n"
                        f"Try asking about a specific segment (e.g. *High Value*) or category (e.g. *Electronics*) to trigger the agent tools. "
                        f"Add a Gemini API key for full AI synthesis."
                    )

            st.markdown(final_response)
            st.session_state.copilot_messages.append({"role": "assistant", "content": final_response})

    # ── Clear history button ──
    if st.session_state.copilot_messages:
        st.markdown("")
        if st.button("🗑️  Clear conversation", key="clear_chat"):
            st.session_state.copilot_messages = []
            st.rerun()
