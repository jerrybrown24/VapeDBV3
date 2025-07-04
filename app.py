# ───────── VaporIQ Dashboard • v8 (hex + elbow) ─────────
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

# ─────────────────── Page & base CSS ───────────────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ───────── Galaxy star-field (body::before) ─────────
starfile = Path(__file__).with_name("starfield.png")
if starfile.exists():
    star_b64 = base64.b64encode(starfile.read_bytes()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
        body::before {{
          content:""; position:fixed; inset:0; z-index:-4; pointer-events:none;
          background:url("data:image/png;base64,{star_b64}") repeat; background-size:600px;
          opacity:.35; animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{
          0%   {{transform:translate3d(0,0,0)}} 
          100% {{transform:translate3d(-2000px,1500px,0)}}
        }}
        .smoke-layer {{animation:smokeFlow 210s linear infinite;  opacity:.25}}
        .smoke-layer-2{{animation:smokeFlowR 280s linear infinite; opacity:.15}}
        @keyframes smokeFlow  {{0%{{background-position:0 0}}100%{{background-position:1600px 0}}}}
        @keyframes smokeFlowR {{0%{{background-position:0 0}}100%{{background-position:-1600px 0}}}}
        </style>"""), unsafe_allow_html=True)
else:
    st.sidebar.warning("⚠️ starfield.png missing – galaxy background disabled")

st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>', unsafe_allow_html=True)

# Watermark
with open("vape_watermark.png","rb") as f:
    wm_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"<img src='data:image/png;base64,{wm_b64}' style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>", unsafe_allow_html=True)

# ─────────────────── Data ───────────────────
@st.cache_data
def load_data():
    users = pd.read_csv("users_synthetic.csv")
    trends = pd.read_csv("flavor_trends.csv")
    trends["Date"] = pd.to_datetime(trends["Date"])
    return users, trends

users_df, trends_df = load_data()
core = ["Age", "SweetLike", "MentholLike", "PodsPerWeek"]

# Ensure a Cluster column exists (for visuals)
if "Cluster" not in users_df.columns:
    users_df["Cluster"] = KMeans(4, random_state=42, n_init="auto") \
        .fit_predict(MinMaxScaler().fit_transform(users_df[core]))

# ─────────────────── Tabs ───────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# =============== 1. Data-Visualization TAB ===============
with viz:
    st.header("📊 Data Visualization Explorer")

    genders = st.sidebar.multiselect("Gender", users_df["Gender"].unique(), users_df["Gender"].unique())
    channels = st.sidebar.multiselect("Purchase Channel", users_df["PurchaseChannel"].unique(), users_df["PurchaseChannel"].unique())
    df = users_df[users_df["Gender"].isin(genders) & users_df["PurchaseChannel"].isin(channels)]
    if df.empty:
        st.warning("No rows match current filters."); st.stop()

    # ⭐ Hex-density (Age × PodsPerWeek) — replaces violin
    st.plotly_chart(
        px.density_heatmap(df, x="Age", y="PodsPerWeek",
                           nbinsx=30, nbinsy=15,
                           color_continuous_scale="magma",
                           title="Density of Consumption by Age"),
        use_container_width=True)
    st.caption("Hot-spots reveal which age bands are heavy users.\nDarker cells = more vapers.")

    # Scatter Pods vs Age
    fig, ax = plt.subplots(); sns.scatterplot(data=df, x="Age", y="PodsPerWeek", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Relationship between age and weekly pod usage.")

    # Correlation heat-map
    fig, ax = plt.subplots()
    sns.heatmap(df[["Age","SweetLike","MentholLike","PodsPerWeek"]].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Pair-wise numeric correlations drive feature engineering choices.")

    # Top flavour families
    flat = df["FlavourFamilies"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
    st.bar_chart(flat)
    st.caption("Which flavour families dominate under current filters.")

    # Trend lines (top-3)
    top3 = trends_df.drop(columns="Date").mean().nlargest(3).index
    st.plotly_chart(px.line(trends_df, x="Date", y=top3,
                            title="Top-3 Flavour Trends"), use_container_width=True)
    st.caption("Comparative momentum of leading flavours.")

    # Boxplot PodsPerWeek by channel
    fig, ax = plt.subplots(); sns.boxplot(data=df, x="PurchaseChannel", y="PodsPerWeek", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Usage distribution across sales channels.")

    # Stacked bar SubscribeIntent × Gender
    stack = pd.crosstab(df["Gender"], df["SubscribeIntent"])
    st.plotly_chart(px.bar(stack, barmode="stack",
                           title="Subscribe Intent vs Gender"), use_container_width=True)

    # Rug plots Sweet & Menthol likes
    fig, ax = plt.subplots()
    sns.rugplot(df["SweetLike"], height=.1, color="g", ax=ax, label="Sweet")
    sns.rugplot(df["MentholLike"], height=.1, color="r", ax=ax, label="Menthol")
    ax.legend(); st.pyplot(fig); plt.close(fig)

    # Treemap cluster × flavour
    tdf = users_df.assign(lead_flav=users_df["FlavourFamilies"].str.split(",").str[0])
    st.plotly_chart(px.treemap(tdf, path=["Cluster","lead_flav"], values="PodsPerWeek"),
                    use_container_width=True)

    # Cumulative Custard Kunafa
    ck = trends_df.assign(cum=trends_df["Custard Kunafa"].cumsum())
    st.plotly_chart(px.area(ck, x="Date", y="cum",
                    title="Cumulative Mentions – Custard Kunafa"), use_container_width=True)

# =============== 2. TasteDNA TAB ===============
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    mode = st.radio("Mode", ["Classification", "Clustering"], horizontal=True)

    if mode == "Classification":
        algo = st.selectbox("Classifier",
            ["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        tune = st.checkbox("GridSearch (5-fold F1)", value=False)

        X, y = users_df[core], users_df["SubscribeIntent"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                                  test_size=0.25,
                                                  stratify=y, random_state=42)

        base = {"KNN":KNeighborsClassifier(),
                "Decision Tree":DecisionTreeClassifier(random_state=42),
                "Random Forest":RandomForestClassifier(random_state=42),
                "Gradient Boosting":GradientBoostingClassifier(random_state=42)}[algo]
        grid = {"KNN":{"n_neighbors":[3,5,7],"weights":["uniform","distance"]},
                "Decision Tree":{"max_depth":[None,3,5],"min_samples_split":[2,5]},
                "Random Forest":{"n_estimators":[100,200,300],"max_depth":[None,10]},
                "Gradient Boosting":{"n_estimators":[100,200],"learning_rate":[0.05,0.1]}}[algo]

        if tune:
            gs = GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1).fit(X_tr,y_tr)
            model, best = gs.best_estimator_, gs.best_params_
        else:
            model, best = base.fit(X_tr,y_tr), None

        y_pred = model.predict(X_te)
        prec, rec = precision_score(y_te,y_pred), recall_score(y_te,y_pred)
        acc, f1 = accuracy_score(y_te,y_pred), f1_score(y_te,y_pred)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Precision", f"{prec:.2f}")
        c2.metric("Recall",    f"{rec:.2f}")
        c3.metric("Accuracy",  f"{acc:.2f}")
        c4.metric("F1",        f"{f1:.2f}")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_te,y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig); plt.close(fig)

    else:  # ─── Clustering ───
        k = st.slider("k clusters", 2, 10, 4)
        X_scaled = MinMaxScaler().fit_transform(users_df[core])

        # Elbow curve
        inertias = [KMeans(i, random_state=42, n_init="auto").fit(X_scaled).inertia_
                    for i in range(2, 11)]
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), inertias, "o-")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Curve")
        st.pyplot(fig); plt.close(fig)

        km = KMeans(k, random_state=42, n_init="auto").fit(X_scaled)
        users_df["Cluster"] = km.labels_
        sil = silhouette_score(X_scaled, km.labels_)
        st.metric("Silhouette", f"{sil:.3f}")
        st.dataframe(users_df.groupby("Cluster")[core].mean().round(2))

# =============== 3. Forecast TAB ===============
with forecast_tab:
    st.header("📈 Forecasting")

    flavour = st.selectbox("Flavour signal", trends_df.columns[1:])
    reg_name = st.selectbox("Regressor", ["Linear","Ridge","Lasso","Decision Tree"])
    reg_map = {"Linear":LinearRegression(),
               "Ridge":Ridge(alpha=1.0),
               "Lasso":Lasso(alpha=0.01),
               "Decision Tree":DecisionTreeRegressor(max_depth=5, random_state=42)}
    reg = reg_map[reg_name]

    X = np.arange(len(trends_df)).reshape(-1,1); y = trends_df[flavour].values
    split = int(.8 * len(X))
    reg.fit(X[:split], y[:split])
    y_pred = reg.predict(X[split:])
    r2 = r2_score(y[split:], y_pred)
    rmse = np.sqrt(mean_squared_error(y[split:], y_pred))

    st.metric("R²", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[split:], y_pred, alpha=.6)
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
    ax.set_xlabel("Actual Mentions"); ax.set_ylabel("Predicted Mentions")
    st.pyplot(fig); plt.close(fig)

# =============== 4. Micro-Batch / Apriori TAB ===============
with rules_tab:
    st.header("🧩 Apriori Explorer")

    sup  = st.slider("Support", 0.01, 0.4, 0.05, 0.01)
    conf = st.slider("Confidence", 0.05, 1.0, 0.3, 0.05)

    basket = users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket = pd.concat(
        [basket,
         pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool)],
        axis=1)

    rules = association_rules(apriori(basket, min_support=sup, use_colnames=True),
                              metric="confidence", min_threshold=conf)

    if rules.empty:
        st.warning("No rules under these thresholds."); best = None
    else:
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules); best = rules.iloc[0]

    if best is not None:
        st.markdown(f"*Top rule:* **{best['antecedents']} → {best['consequents']}** "
                    f"(lift {best['lift']:.2f})")
    st.markdown(f"*Thresholds:* support ≥ {sup:.2f}, confidence ≥ {conf:.2f}")
