# ───────── VaporIQ Dashboard  • v8.1  ─────────
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
from pathlib import Path
import base64, textwrap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score,
                             roc_curve, auc, r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ───── Page config & base CSS ─────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ───── Galaxy star-field (body::before) ─────
star_path = Path(__file__).with_name("starfield.png")
if star_path.exists():
    star_b64 = base64.b64encode(star_path.read_bytes()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
        body::before {{
            content:""; position:fixed; inset:0; z-index:-4; pointer-events:none;
            background:url("data:image/png;base64,{star_b64}") repeat; background-size:600px;
            opacity:.35; animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{
          0% {{transform:translate3d(0,0,0)}} 100% {{transform:translate3d(-2000px,1500px,0)}}
        }}
        .smoke-layer {{animation:smokeFlow 210s linear infinite;  opacity:.25}}
        .smoke-layer-2{{animation:smokeFlowR 280s linear infinite; opacity:.15}}
        @keyframes smokeFlow  {{0%{{background-position:0 0}}100%{{background-position:1600px 0}}}}
        @keyframes smokeFlowR {{0%{{background-position:0 0}}100%{{background-position:-1600px 0}}}}
        </style>"""), unsafe_allow_html=True)
else:
    st.sidebar.error("⚠️ starfield.png missing — galaxy layer disabled")

st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)

# ───── Watermark ─────
with open("vape_watermark.png","rb") as f:
    wm_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"<img src='data:image/png;base64,{wm_b64}' style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>", unsafe_allow_html=True)

# ───── Data load ─────
@st.cache_data
def load_data():
    df_u = pd.read_csv("users_synthetic.csv")
    df_t = pd.read_csv("flavor_trends.csv")
    df_t["Date"] = pd.to_datetime(df_t["Date"])
    return df_u, df_t

users_df, trends_df = load_data()
core = ["Age","SweetLike","MentholLike","PodsPerWeek"]

# Ensure Cluster exists for visuals
if "Cluster" not in users_df.columns:
    users_df["Cluster"] = KMeans(4, random_state=42, n_init="auto") \
        .fit_predict(MinMaxScaler().fit_transform(users_df[core]))

# ───── Tabs ─────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# ========== 1. Data-Viz TAB ==========
with viz:
    st.header("📊 Data Visualization Explorer")

    genders = st.sidebar.multiselect("Gender filter",
        users_df["Gender"].unique().tolist(),
        default=users_df["Gender"].unique().tolist())
    channels = st.sidebar.multiselect("Purchase Channel filter",
        users_df["PurchaseChannel"].unique().tolist(),
        default=users_df["PurchaseChannel"].unique().tolist())

    df = users_df[users_df["Gender"].isin(genders) &
                  users_df["PurchaseChannel"].isin(channels)]

    if df.empty:
        st.warning("No rows match current filters — adjust sidebar selections."); st.stop()

    # 1. Hexbin density
    st.plotly_chart(
        px.density_heatmap(df, x="Age", y="PodsPerWeek", nbinsx=30, nbinsy=15,
                           color_continuous_scale="magma",
                           title="Density of Consumption by Age"),
        use_container_width=True)
    st.caption("Hot-spots reveal which age bands are heavy users.")

    # 2. Parallel-coordinates
    st.plotly_chart(
        px.parallel_coordinates(users_df[core + ["Cluster"]],
                                color="Cluster",
                                title="TasteDNA Fingerprint by Cluster",
                                color_continuous_scale=px.colors.diverging.Portland),
        use_container_width=True)
    st.caption("Visual fingerprint of clusters across sensory & usage traits.")

    # 3. Radar chart
    cent = users_df.groupby("Cluster")[core].mean().reset_index().melt(
        id_vars="Cluster", var_name="Metric", value_name="Value")
    st.plotly_chart(
        px.line_polar(cent, r="Value", theta="Metric", color="Cluster",
                      line_close=True, title="Cluster Centroids – Radar View"),
        use_container_width=True)
    st.caption("Compare clusters at a glance; spot high-sweet vs high-menthol groups.")

    # 4. Cumulative adoption curve
    cum = users_df.sort_values("UserID").SubscribeIntent.cumsum()/np.arange(1,len(users_df)+1)
    fig, ax = plt.subplots(); ax.plot(cum.index, cum.values)
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_xlabel("User join order"); ax.set_ylabel("Cumulative % subscribed")
    ax.set_title("Cumulative Subscribe Intent")
    st.pyplot(fig); plt.close(fig)
    st.caption("Shows adoption saturation and inflection points.")

    # 5. Monthly flavour heat-map
    month = trends_df.set_index("Date").resample("M").mean().reset_index()
    month["Month"] = month["Date"].dt.to_period("M").astype(str)
    heat = month.drop(columns="Date").set_index("Month")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(heat.T, cmap="rocket_r", ax=ax)
    ax.set_title("Monthly Flavour Intensity"); st.pyplot(fig); plt.close(fig)
    st.caption("Seasonality and monthly spikes become obvious.")

    with st.expander("Key Insights"):
        dom = df["Gender"].value_counts(normalize=True).idxmax()
        rising = trends_df.drop(columns="Date").mean().idxmax()
        st.markdown(f"- Dominant gender in current view: **{dom}**")
        st.markdown(f"- Fastest rising flavour overall: **{rising}**")

# ========== 2. TasteDNA TAB (unchanged) ==========
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    # ... (same classification/clustering logic from previous v7) ...
    # For brevity: identical to earlier snippet – no changes needed.

# ========== 3. Forecast TAB (unchanged) ==========
with forecast_tab:
    st.header("📈 Forecasting")
    # ... (same regressor selector and plot) ...

# ========== 4. Apriori TAB (rules.empty fixed) ==========
with rules_tab:
    st.header("🧩 Apriori Explorer")
    sup=st.slider("Support",0.01,0.4,0.05,0.01); conf=st.slider("Confidence",0.05,1.0,0.3,0.05)
    basket=users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket=pd.concat([basket,
        pd.get_dummies(users_df["PurchaseChannel"],prefix="Chan").astype(bool)],axis=1)
    rules=association_rules(apriori(basket,min_support=sup,use_colnames=True),
                            metric="confidence",min_threshold=conf)

    if rules.empty:
        st.warning("No rules under thresholds."); best=None
    else:
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules); best = rules.iloc[0]

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- Best rule: {best['antecedents']} → {best['consequents']} (lift {best['lift']:.2f})")
        st.markdown(f"- Support ≥ {sup:.2f} • Confidence ≥ {conf:.2f}")
