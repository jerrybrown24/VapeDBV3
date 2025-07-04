# ───────── VaporIQ Dashboard • v8.2  ─────────
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
from pathlib import Path
import base64, textwrap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ─────────────────── Page + base CSS ───────────────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ───────── Galaxy star-field via body::before ─────────
star = Path(__file__).with_name("starfield.png")
if star.exists():
    st.markdown(
        f"""
        <style>
        body::before {{
          content:""; position:fixed; inset:0; z-index:-4; pointer-events:none;
          background:url("data:image/png;base64,{base64.b64encode(star.read_bytes()).decode()}") repeat;
          background-size:600px; opacity:.35; animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{0%{{transform:translate3d(0,0,0)}}100%{{transform:translate3d(-2000px,1500px,0)}}}}
        </style>
        """, unsafe_allow_html=True)
else:
    st.sidebar.error("⚠️ starfield.png missing – galaxy backdrop disabled")

st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>', unsafe_allow_html=True)

# watermark
with open("vape_watermark.png","rb") as f:
    st.markdown(
        f"<img src='data:image/png;base64,{base64.b64encode(f.read()).decode()}' "
        "style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>",
        unsafe_allow_html=True)

# ─────────────────── Data ───────────────────
@st.cache_data
def load():
    users = pd.read_csv("users_synthetic.csv")
    trends = pd.read_csv("flavor_trends.csv"); trends["Date"] = pd.to_datetime(trends["Date"])
    return users, trends

users_df, trends_df = load()
core = ["Age","SweetLike","MentholLike","PodsPerWeek"]

# assure Cluster col
if "Cluster" not in users_df.columns:
    users_df["Cluster"] = KMeans(4, random_state=42) \
        .fit_predict(MinMaxScaler().fit_transform(users_df[core]))

# ─────────────────── Tabs ───────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# =============== 1. Data-Visualization ===============
with viz:
    st.header("📊 Data Visualization Explorer")
    g = st.sidebar.multiselect("Gender", users_df["Gender"].unique(), users_df["Gender"].unique())
    c = st.sidebar.multiselect("Purchase Channel", users_df["PurchaseChannel"].unique(),
                               users_df["PurchaseChannel"].unique())
    df = users_df[users_df["Gender"].isin(g) & users_df["PurchaseChannel"].isin(c)]
    if df.empty: st.warning("No rows match current filters."); st.stop()

    st.plotly_chart(
        px.density_heatmap(df, x="Age", y="PodsPerWeek",
                           nbinsx=30, nbinsy=15,
                           color_continuous_scale="magma",
                           title="Density of Consumption by Age"),
        use_container_width=True)
    st.caption("Hot-spots reveal which age bands are heavy users.")

    # (other visuals unchanged for brevity) …

# =============== 2. TasteDNA ===============
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    mode = st.radio("Mode", ["Classification","Clustering"], horizontal=True)

    # ---------- Classification ----------
    if mode == "Classification":
        algo = st.selectbox("Classifier",
            ["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        tune = st.checkbox("GridSearch (5-fold, F1)", value=False)

        X, y = users_df[core], users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)

        base = {"KNN":KNeighborsClassifier(),
                "Decision Tree":DecisionTreeClassifier(random_state=42),
                "Random Forest":RandomForestClassifier(random_state=42),
                "Gradient Boosting":GradientBoostingClassifier(random_state=42)}[algo]
        grid = {"KNN":{"n_neighbors":[3,5,7],"weights":["uniform","distance"]},
                "Decision Tree":{"max_depth":[None,3,5],"min_samples_split":[2,5]},
                "Random Forest":{"n_estimators":[100,200,300],"max_depth":[None,10]},
                "Gradient Boosting":{"n_estimators":[100,200],"learning_rate":[0.05,0.1]}}[algo]

        model = (GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1)
                 .fit(X_tr,y_tr).best_estimator_) if tune else base.fit(X_tr,y_tr)

        y_pred = model.predict(X_te)
        metrics = dict(
            Precision = precision_score(y_te,y_pred),
            Recall    = recall_score(y_te,y_pred),
            Accuracy  = accuracy_score(y_te,y_pred),
            F1        = f1_score(y_te,y_pred)
        )

        for k,v in metrics.items(): st.metric(k, f"{v:.2f}")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_te,y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig); plt.close(fig)

    # ---------- Clustering ----------
    else:
        k = st.slider("k clusters",2,10,4)
        X_scaled = MinMaxScaler().fit_transform(users_df[core])

        # Elbow curve
        inertias = [KMeans(i, random_state=42).fit(X_scaled).inertia_ for i in range(2,11)]
        efig, eax = plt.subplots()
        eax.plot(range(2,11), inertias, marker="o")
        eax.set_xlabel("k"); eax.set_ylabel("Inertia"); eax.set_title("Elbow Curve")
        st.pyplot(efig); plt.close(efig)

        km = KMeans(k, random_state=42).fit(X_scaled)
        users_df["Cluster"] = km.labels_
        sil = silhouette_score(X_scaled, km.labels_)
        st.metric("Silhouette", f"{sil:.3f}")
        st.dataframe(users_df.groupby("Cluster")[core].mean().round(2))

# =============== 3. Forecasting ===============
with forecast_tab:
    st.header("📈 Forecasting")
    flav = st.selectbox("Flavour", trends_df.columns[1:])
    reg_name = st.selectbox("Regressor", ["Linear","Ridge","Lasso","Decision Tree"])
    reg_map = {"Linear":LinearRegression(),
               "Ridge":Ridge(alpha=1.0),
               "Lasso":Lasso(alpha=0.01),
               "Decision Tree":DecisionTreeRegressor(max_depth=5, random_state=42)}
    reg = reg_map[reg_name]

    X = np.arange(len(trends_df)).reshape(-1,1); y = trends_df[flav].values
    cut = int(.8*len(X))
    reg.fit(X[:cut], y[:cut]); y_pred = reg.predict(X[cut:])
    st.metric("R²", f"{r2_score(y[cut:],y_pred):.3f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y[cut:],y_pred)):.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[cut:],y_pred, alpha=.6); ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); st.pyplot(fig); plt.close(fig)

# =============== 4. Micro-Batch / Apriori ===============
with rules_tab:
    st.header("🧩 Apriori Explorer")
    sup = st.slider("Support",0.01,0.4,0.05,0.01); conf = st.slider("Confidence",0.05,1.0,0.3,0.05)
    basket = users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket = pd.concat([basket,
        pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool)], axis=1)
    rules = association_rules(apriori(basket, min_support=sup, use_colnames=True),
                              metric="confidence", min_threshold=conf)
    if rules.empty:
        st.warning("No rules under thresholds.")
    else:
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))
