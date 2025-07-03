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

# ─────────────────────────── Theme & layers ───────────────────────────
st.set_page_config(page_title="VaporIQ v7", layout="wide")

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# --- Star-field  (PNG must sit next to app.py) ---
star_path = Path(__file__).with_name("starfield.png")
if star_path.exists():
    star_b64 = base64.b64encode(star_path.read_bytes()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
        .starfield {{
            position:fixed; inset:0; pointer-events:none; z-index:-5;
            background:url("data:image/png;base64,{star_b64}") repeat;
            background-size:600px;
            opacity:.30;
            animation:driftStars 200s linear infinite;
        }}
        @keyframes driftStars {{
          0% {{background-position:0 0;}}
          100%{{background-position:-2000px 1500px;}}
        }}
        </style>
        <div class="starfield"></div>
    """), unsafe_allow_html=True)
else:
    st.sidebar.error("⚠️ `starfield.png` not found — galaxy layer disabled.")

# Smoke layers (keep original order)
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)

# Watermark bottle
with open("vape_watermark.png","rb") as f:
    wm_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"<img src='data:image/png;base64,{wm_b64}' "
            "style='position:fixed;bottom:15px;right:15px;width:110px;"
            "opacity:0.8;z-index:1;'/>", unsafe_allow_html=True)

# ─────────────────────────── Load data ───────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("users_synthetic.csv"), pd.read_csv("flavor_trends.csv")

users_df, trends_df = load_data()
core = ["Age", "SweetLike", "MentholLike", "PodsPerWeek"]

# ─────────────────────────── Tabs ───────────────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch & Drops"]
)

# ===================== 1. DATA-VIS TAB =====================
with viz:
    st.header("📊 Data Visualization Explorer")

    gender_sel = st.sidebar.multiselect(
        "Gender filter",
        users_df["Gender"].unique().tolist(),
        default=users_df["Gender"].unique().tolist()
    )
    chan_sel = st.sidebar.multiselect(
        "Purchase Channel filter",
        users_df["PurchaseChannel"].unique().tolist(),
        default=users_df["PurchaseChannel"].unique().tolist()
    )

    df = users_df[
        users_df["Gender"].isin(gender_sel) &
        users_df["PurchaseChannel"].isin(chan_sel)
    ]

    if df.empty:
        st.warning("No data under current filters — change selections in the sidebar.")
        st.stop()

    # 1 Violin (Age)
    fig, ax = plt.subplots()
    sns.violinplot(data=df, y="Age", inner="box", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Shows full age spread; detects multi-modal distributions useful for age-based segmentation.")

    # 2 Scatter (Pods vs Age)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="PodsPerWeek", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Reveals relationship between age and consumption volume—helps identify heavy-usage age groups.")

    # 3 Correlation heat-map
    fig, ax = plt.subplots()
    sns.heatmap(df[["Age","SweetLike","MentholLike","PodsPerWeek"]].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Highlights strongest numeric correlations—guides which features combine well in models.")

    # 4 Bar chart top flavour families
    flat = df["FlavourFamilies"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
    st.bar_chart(flat)
    st.caption("Ranks flavour popularity under current filters—steers limited-drop flavour decisions.")

    # 5 Trend lines top-3 flavours
    top3 = trends_df.drop(columns="Date").mean().nlargest(3).index
    st.plotly_chart(px.line(trends_df, x="Date", y=top3,
                            title="Top-3 Flavour Trends"), use_container_width=True)
    st.caption("Compares growth trajectories of leading flavours—feeds forecasting and supply planning.")

    # 6 Boxplot PodsPerWeek by channel
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="PurchaseChannel", y="PodsPerWeek", ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Shows distribution & outliers by channel—optimise channel-specific promotions.")

    # 7 Stacked bar Intent vs Gender
    sb = pd.crosstab(df["Gender"], df["SubscribeIntent"])
    st.plotly_chart(px.bar(sb, barmode="stack",
                           title="Subscribe Intent vs Gender"), use_container_width=True)
    st.caption("Visualises gender skew in subscription intent—tailor messaging accordingly.")

    # 8 Rug plots Sweet vs Menthol preference
    fig, ax = plt.subplots()
    sns.rugplot(df["SweetLike"], height=.1, color="g", ax=ax, label="Sweet")
    sns.rugplot(df["MentholLike"], height=.1, color="r", ax=ax, label="Menthol")
    ax.legend(); st.pyplot(fig); plt.close(fig)
    st.caption("Density of sweetness vs menthol preference—identify flavour gaps.")

    # 9 Treemap cluster × flavour
    if "Cluster" not in users_df.columns:
        users_df["Cluster"] = KMeans(4, random_state=42, n_init="auto") \
            .fit_predict(MinMaxScaler().fit_transform(users_df[core]))
    tdf = users_df.assign(lead_flavour=users_df["FlavourFamilies"].str.split(",").str[0])
    st.plotly_chart(px.treemap(tdf, path=["Cluster","lead_flavour"], values="PodsPerWeek"),
                    use_container_width=True)
    st.caption("Maps clusters to leading flavour families—quickly shows whitespace flavour opportunities.")

    # 10 Cumulative Custard Kunafa
    ck = trends_df.assign(cum=trends_df["Custard Kunafa"].cumsum())
    st.plotly_chart(px.area(ck, x="Date", y="cum", title="Cumulative Mentions – Custard Kunafa"),
                    use_container_width=True)
    st.caption("Quantifies total buzz; informs batch size for limited Custard Kunafa drops.")

    with st.expander("Key Insights"):
        skew_demo = df["Gender"].value_counts(normalize=True).idxmax()
        fastest = trends_df.drop(columns="Date").mean().idxmax()
        st.markdown(f"- **Demographic skew:** {skew_demo} dominates current filter.")
        st.markdown(f"- **Fastest rising flavour overall:** {fastest}.")
        st.markdown("- Data: 1 200 synthetic survey rows + 120-week trend scrape.")

# ===================== 2. TASTEDNA TAB =====================
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    mode = st.radio("Mode", ["Classification","Clustering"], horizontal=True)

    if mode == "Classification":
        algo = st.selectbox("Classifier",
                            ["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        tune = st.checkbox("Run GridSearch (5-fold F1)", value=False)

        X, y = users_df[core], users_df["SubscribeIntent"]
        X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=0.25,
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
            model = gs.best_estimator_; best = gs.best_params_
        else:
            model = base.fit(X_tr,y_tr); best = None

        y_pred = model.predict(X_te); prob = model.predict_proba(X_te)[:,1]
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

        with st.expander("Key Insights"):
            if best:
                st.markdown(f"- Best hyper-params: `{best}`.")
            st.markdown(f"- Metrics ⇒ Prec {prec:.2f}, Rec {rec:.2f}, F1 {f1:.2f}.")

    else:  # Clustering
        k = st.slider("Number of clusters (k)", 2, 10, 4)
        X_scaled = MinMaxScaler().fit_transform(users_df[core])
        km = KMeans(k, random_state=42, n_init="auto").fit(X_scaled)
        users_df["Cluster"] = km.labels_; sil = silhouette_score(X_scaled, km.labels_)
        st.metric("Silhouette", f"{sil:.3f}")
        st.dataframe(users_df.groupby("Cluster")[core].mean().round(2))

# ===================== 3. FORECAST TAB =====================
with forecast_tab:
    st.header("📈 Forecasting")
    flav = st.selectbox("Flavour signal", trends_df.columns[1:])
    reg_name = st.selectbox("Regressor", ["Linear","Ridge","Lasso","Decision Tree"])
    reg_map = {"Linear":LinearRegression(),
               "Ridge":Ridge(alpha=1.0),
               "Lasso":Lasso(alpha=0.01),
               "Decision Tree":DecisionTreeRegressor(max_depth=5, random_state=42)}
    reg = reg_map[reg_name]

    X = np.arange(len(trends_df)).reshape(-1,1); y = trends_df[flav].values
    split = int(0.8 * len(X))
    reg.fit(X[:split], y[:split]); y_pred = reg.predict(X[split:])
    r2 = r2_score(y[split:], y_pred)
    rmse = np.sqrt(mean_squared_error(y[split:], y_pred))

    st.metric("R²", f"{r2:.3f}"); st.metric("RMSE", f"{rmse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[split:], y_pred, alpha=.6)
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
    ax.set_xlabel("Actual Mentions"); ax.set_ylabel("Predicted Mentions")
    st.pyplot(fig); plt.close(fig)

    with st.expander("Key Insights"):
        slopes = {c: np.polyfit(np.arange(len(trends_df)), trends_df[c], 1)[0]
                  for c in trends_df.columns[1:]}
        st.markdown(f"- **Regressor:** {reg_name} → R² {r2:.2f}.")
        st.markdown(f"- Steepest trend flavour: **{max(slopes, key=slopes.get)}**.")

# ===================== 4. APRIORI TAB =====================
with rules_tab:
    st.header("🧩 Micro-Batch Apriori Explorer")
    sup = st.slider("Support", 0.01, 0.4, 0.05, 0.01)
    conf = st.slider("Confidence", 0.05, 1.0, 0.3, 0.05)
    basket = users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket = pd.concat([basket,
                        pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool)],
                        axis=1)
    freq = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)

    if rules.empty:
        st.warning("No rules under these thresholds."); best = None
    else:
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules); best = rules.iloc[0]

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- Best rule: {best['antecedents']} → {best['consequents']} (lift {best['lift']:.2f}).")
        st.markdown(f"- Support ≥ {sup:.2f}, Confidence ≥ {conf:.2f}.")
