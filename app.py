# ───────── VaporIQ Dashboard  • v8  ─────────
# full app.py with:
#  • Flicker-proof galaxy star-field & slow smoke overlay
#  • New Data-Viz charts (hexbin, parallel, radar, adoption curve, monthly heat)
#  • Extra metrics, regressor selector, Apriori tab, etc.
#  • Works with starfield.png, style.css, watermark, synthetic CSVs
# -------------------------------------------------------

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

# ─────────────────────  Page & base CSS  ─────────────────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ─────────  Galaxy star-field via body::before  ─────────
star_path = Path(__file__).with_name("starfield.png")
if star_path.exists():
    star_b64 = base64.b64encode(star_path.read_bytes()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
        body::before {{
            content:"";
            position:fixed; inset:0; z-index:-4;
            pointer-events:none;
            background:url("data:image/png;base64,{star_b64}") repeat;
            background-size:600px;
            opacity:.35;
            animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{
            0%   {{ transform:translate3d(0,0,0); }}
            100% {{ transform:translate3d(-2000px,1500px,0); }}
        }}
        /* slow smoke */
        .smoke-layer    {{animation:smokeFlow 210s linear infinite;  opacity:.25;}}
        .smoke-layer-2  {{animation:smokeFlowR 280s linear infinite; opacity:.15;}}
        @keyframes smokeFlow  {{0%{{background-position:0 0}} 100%{{background-position:1600px 0}}}}
        @keyframes smokeFlowR {{0%{{background-position:0 0}} 100%{{background-position:-1600px 0}}}}
        </style>
    """), unsafe_allow_html=True)
else:
    st.sidebar.error("⚠️ `starfield.png` missing – galaxy backdrop disabled.")

# smoke divs
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)

# watermark
with open("vape_watermark.png","rb") as f:
    wm_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"<img src='data:image/png;base64,{wm_b64}' style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>", unsafe_allow_html=True)

# ─────────────────────  Load data  ─────────────────────
@st.cache_data
def load_data():
    df_users  = pd.read_csv("users_synthetic.csv")
    df_trends = pd.read_csv("flavor_trends.csv")
    df_trends["Date"] = pd.to_datetime(df_trends["Date"])
    return df_users, df_trends

users_df, trends_df = load_data()
core = ["Age","SweetLike","MentholLike","PodsPerWeek"]

# ─────────────────────  Tabs  ─────────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# =================== 1. Data-Viz TAB ===================
with viz:
    st.header("📊 Data Visualization Explorer")

    # sidebar filters
    genders  = st.sidebar.multiselect("Gender filter",
        users_df["Gender"].unique().tolist(),
        default=users_df["Gender"].unique().tolist())
    channels = st.sidebar.multiselect("Purchase Channel filter",
        users_df["PurchaseChannel"].unique().tolist(),
        default=users_df["PurchaseChannel"].unique().tolist())

    df = users_df[users_df["Gender"].isin(genders) &
                  users_df["PurchaseChannel"].isin(channels)]

    if df.empty:
        st.warning("No rows match current filters — adjust sidebar selections.")
        st.stop()

    # ⭐ 1. Hexbin density
    hex_fig = px.density_heatmap(
        df, x="Age", y="PodsPerWeek",
        nbinsx=30, nbinsy=15,
        color_continuous_scale="magma",
        title="Density of Consumption by Age"
    )
    st.plotly_chart(hex_fig, use_container_width=True)
    st.caption("Hot-spots reveal which age bands are heavy users.\nDense orange cells = many vapers.")

    # ⭐ 2. Parallel-coordinates
    par_df  = users_df[core + ["Cluster"]]
    par_fig = px.parallel_coordinates(
        par_df, color="Cluster",
        title="TasteDNA Fingerprint by Cluster",
        color_continuous_scale=px.colors.diverging.Portland
    )
    st.plotly_chart(par_fig, use_container_width=True)
    st.caption("Visual fingerprint of clusters across sensory & usage traits.")

    # ⭐ 3. Radar / spider chart
    cent = users_df.groupby("Cluster")[core].mean().reset_index().melt(
        id_vars="Cluster", var_name="Metric", value_name="Value")
    radar_fig = px.line_polar(
        cent, r="Value", theta="Metric", color="Cluster",
        line_close=True, title="Cluster Centroids – Radar View")
    st.plotly_chart(radar_fig, use_container_width=True)
    st.caption("Compare clusters at a glance; spot high-sweet vs high-menthol groups.")

    # ⭐ 4. Cumulative adoption curve
    cum = users_df.sort_values("UserID").SubscribeIntent.cumsum() / np.arange(1,len(users_df)+1)
    fig, ax = plt.subplots()
    ax.plot(cum.index, cum.values)
    ax.axhline(0.5, color="gray", ls="--")
    ax.set_xlabel("User join order"); ax.set_ylabel("Cumulative % subscribed")
    ax.set_title("Cumulative Subscribe Intent")
    st.pyplot(fig); plt.close(fig)
    st.caption("Shows adoption saturation and where inflection points occur.")

    # ⭐ 5. Monthly flavour heat-map
    month = trends_df.set_index("Date").resample("M").mean().reset_index()
    month["Month"] = month["Date"].dt.to_period("M").astype(str)
    heat = month.drop(columns="Date").set_index("Month")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(heat.T, cmap="rocket_r", ax=ax)
    ax.set_title("Monthly Flavour Intensity")
    st.pyplot(fig); plt.close(fig)
    st.caption("Seasonality and monthly spikes become obvious.")

    # insights block (unchanged)
    with st.expander("Key Insights"):
        dom_gender = df["Gender"].value_counts(normalize=True).idxmax()
        fast_flav  = trends_df.drop(columns="Date").mean().idxmax()
        st.markdown(f"- Dominant gender in current filters: **{dom_gender}**")
        st.markdown(f"- Fastest rising flavour overall: **{fast_flav}**")
        st.markdown("- Data: 1 200 synthetic survey rows + 120-week trend crawl.")

# =================== 2. TasteDNA TAB ===================
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    mode = st.radio("Mode",["Classification","Clustering"],horizontal=True)

    if mode == "Classification":
        algo = st.selectbox("Classifier",["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        tune = st.checkbox("GridSearch (5-fold F1)",False)

        X,y = users_df[core], users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.25,stratify=y,random_state=42)

        base = {"KNN":KNeighborsClassifier(),
                "Decision Tree":DecisionTreeClassifier(random_state=42),
                "Random Forest":RandomForestClassifier(random_state=42),
                "Gradient Boosting":GradientBoostingClassifier(random_state=42)}[algo]
        grid = {"KNN":{"n_neighbors":[3,5,7],"weights":["uniform","distance"]},
                "Decision Tree":{"max_depth":[None,3,5],"min_samples_split":[2,5]},
                "Random Forest":{"n_estimators":[100,200,300],"max_depth":[None,10]},
                "Gradient Boosting":{"n_estimators":[100,200],"learning_rate":[0.05,0.1]}}[algo]

        if tune:
            gs = GridSearchCV(base,grid,scoring="f1",cv=5,n_jobs=-1).fit(X_tr,y_tr)
            model,best = gs.best_estimator_, gs.best_params_
        else:
            model,best = base.fit(X_tr,y_tr), None

        y_pred = model.predict(X_te); prob = model.predict_proba(X_te)[:,1]
        prec, rec = precision_score(y_te,y_pred), recall_score(y_te,y_pred)
        acc, f1  = accuracy_score(y_te,y_pred), f1_score(y_te,y_pred)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Precision",f"{prec:.2f}")
        m2.metric("Recall",   f"{rec:.2f}")
        m3.metric("Accuracy", f"{acc:.2f}")
        m4.metric("F1",       f"{f1:.2f}")

        fig,ax=plt.subplots()
        sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt="d",cmap="Blues",ax=ax)
        st.pyplot(fig); plt.close(fig)

        with st.expander("Key Insights"):
            if best: st.markdown(f"- Best params: `{best}`")
            st.markdown(f"- Prec {prec:.2f}, Rec {rec:.2f}, F1 {f1:.2f}")

    else:  # Clustering
        k=st.slider("k clusters",2,10,4)
        X_scaled = MinMaxScaler().fit_transform(users_df[core])
        km = KMeans(k,random_state=42,n_init="auto").fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        users_df["Cluster"] = km.labels_
        st.metric("Silhouette", f"{sil:.3f}")
        st.dataframe(users_df.groupby("Cluster")[core].mean().round(2))

# =================== 3. Forecast TAB ===================
with forecast_tab:
    st.header("📈 Forecasting")
    flavour = st.selectbox("Flavour signal", trends_df.columns[1:])
    reg_name = st.selectbox("Regressor", ["Linear","Ridge","Lasso","Decision Tree"])
    reg_map = {"Linear":LinearRegression(),
               "Ridge":Ridge(alpha=1.0),
               "Lasso":Lasso(alpha=0.01),
               "Decision Tree":DecisionTreeRegressor(max_depth=5,random_state=42)}
    reg = reg_map[reg_name]

    X = np.arange(len(trends_df)).reshape(-1,1); y = trends_df[flavour].values
    split = int(.8*len(X))
    reg.fit(X[:split],y[:split]); y_pred = reg.predict(X[split:])
    r2=r2_score(y[split:],y_pred); rmse=np.sqrt(mean_squared_error(y[split:],y_pred))

    st.metric("R²", f"{r2:.3f}"); st.metric("RMSE", f"{rmse:.2f}")

    fig,ax=plt.subplots()
    ax.scatter(y[split:],y_pred,alpha=.6)
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
    ax.set_xlabel("Actual Mentions"); ax.set_ylabel("Predicted Mentions")
    st.pyplot(fig); plt.close(fig)

    with st.expander("Key Insights"):
        slopes = {c: np.polyfit(np.arange(len(trends_df)), trends_df[c], 1)[0]
                  for c in trends_df.columns[1:]}
        st.markdown(f"- Regressor **{reg_name}** → R² {r2:.2f}, RMSE {rmse:.2f}")
        st.markdown(f"- Steepest flavour slope: **{max(slopes, key=slopes.get)}**")

# =================== 4. Apriori TAB ===================
with rules_tab:
    st.header("🧩 Apriori Explorer")
    sup = st.slider("Support",0.01,0.4,0.05,0.01)
    conf= st.slider("Confidence",0.05,1.0,0.3,0.05)
    basket = users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket = pd.concat([basket,
                        pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool)],
                        axis=1)
    freq  = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)

    if rules.empty:
        st.warning("No rules under these thresholds."); best=None
    else:
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules); best = rules.iloc[0]

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- Best rule: {best['antecedents']} → {best['consequents']} (lift {best['lift']:.2f})")
        st.markdown(f"- Support ≥ {sup:.2f}, Confidence ≥ {conf:.2f}")
