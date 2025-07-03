import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, confusion_matrix, f1_score,
                             roc_curve, auc, r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules
import base64, matplotlib

# ─────────────────────── Theme & watermark ───────────────────────
st.set_page_config(page_title="VaporIQ v5", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
st.markdown('<div class="smoke-layer"></div>', unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>', unsafe_allow_html=True)

with open("vape_watermark.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.markdown(
    f"<img src='data:image/png;base64,{b64}' "
    "style='position:fixed;bottom:15px;right:15px;width:110px;"
    "opacity:0.8;z-index:1;'/>",
    unsafe_allow_html=True
)

# ─────────────────────── Data ───────────────────────
@st.cache_data
def load_data():
    return (
        pd.read_csv("users_synthetic.csv"),
        pd.read_csv("flavor_trends.csv")
    )

users_df, trends_df = load_data()
core_cols = ["Age", "SweetLike", "MentholLike", "PodsPerWeek"]

# ─────────────────────── Tabs ───────────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch & Drops"]
)

# ==================== 1. Data-Viz Tab ====================
with viz:
    st.header("📊 Data Visualization Explorer")

    # Sidebar filters
    g_sel = st.sidebar.multiselect(
        "Gender", users_df["Gender"].unique(), users_df["Gender"].unique())
    c_sel = st.sidebar.multiselect(
        "Purchase Channel", users_df["PurchaseChannel"].unique(),
        users_df["PurchaseChannel"].unique())

    df = users_df[
        users_df["Gender"].isin(g_sel) &
        users_df["PurchaseChannel"].isin(c_sel)
    ]

    # 1-2  Violin & Scatter
    fig, ax = plt.subplots()
    sns.violinplot(data=df, y="Age", inner="box", ax=ax)
    st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="PodsPerWeek", ax=ax)
    st.pyplot(fig); plt.close(fig)

    # 3  Correlation heat-map
    fig, ax = plt.subplots()
    sns.heatmap(df[["Age","SweetLike","MentholLike","PodsPerWeek"]]
                .corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig); plt.close(fig)

    # 4  Top flavour families
    flat = df["FlavourFamilies"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
    st.bar_chart(flat)

    # 5  Trend lines (top-3)
    top3 = trends_df.drop(columns=["Date"]).mean().nlargest(3).index
    st.plotly_chart(px.line(trends_df, x="Date", y=top3,
                            title="Top-3 Flavour Trends"), use_container_width=True)

    # 6  Boxplot PodsPerWeek by channel
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="PurchaseChannel", y="PodsPerWeek", ax=ax)
    st.pyplot(fig); plt.close(fig)

    # 7  Stacked bar SubscribeIntent vs Gender
    stacked = pd.crosstab(df["Gender"], df["SubscribeIntent"])
    fig = px.bar(stacked, barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    # 8  Rug plots SweetLike & MentholLike
    fig, ax = plt.subplots()
    sns.rugplot(df["SweetLike"], height=.1, color="g", ax=ax, label="SweetLike")
    sns.rugplot(df["MentholLike"], height=.1, color="r", ax=ax, label="MentholLike")
    ax.legend(); st.pyplot(fig); plt.close(fig)

    # 9  Treemap flavour families + cluster
    if "Cluster" not in users_df.columns:
        users_df["Cluster"] = KMeans(4, random_state=42, n_init='auto') \
            .fit_predict(MinMaxScaler().fit_transform(users_df[core_cols]))
    tmap_df = users_df.assign(main_flavour=users_df["FlavourFamilies"].str.split(",").str[0])
    st.plotly_chart(px.treemap(tmap_df, path=["Cluster","main_flavour"], values="PodsPerWeek"),
                    use_container_width=True)

    # 10  Cumulative Custard Kunafa
    ck = trends_df.copy(); ck["cum"] = ck["Custard Kunafa"].cumsum()
    st.plotly_chart(px.area(ck, x="Date", y="cum",
                            title="Cumulative Custard Kunafa Mentions"),
                    use_container_width=True)

    # ── Insight block ──
    with st.expander("Key Insights"):
        retail_mean = df[df["PurchaseChannel"] == "Retail"]["PodsPerWeek"].mean()
        online_mean = df[df["PurchaseChannel"] == "Online"]["PodsPerWeek"].mean()
        st.markdown(f"- Retail users avg **{retail_mean:.1f}** pods/week vs **{online_mean:.1f}** online.")
        st.markdown("- Data from synthetic survey (1 200 rows) and simulated trend scrape (120 weeks).")
        st.markdown("- Core viz metrics: means, correlations, slopes, cumulative sums.")

# ==================== 2. TasteDNA Tab ====================
with taste_tab:
    st.header("🔮 TasteDNA Analysis")
    mode = st.radio("Mode", ["Classification", "Clustering"], horizontal=True)

    if mode == "Classification":
        clf_name = st.selectbox(
            "Classifier",
            ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
        )
        run_gs = st.checkbox("Run Grid Search (5-fold CV)", value=False)

        X, y = users_df[core_cols], users_df["SubscribeIntent"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        base_est = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }[clf_name]

        param_grid = {
            "KNN": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"]
            },
            "Decision Tree": {
                "max_depth": [None, 3, 5, 7, 9],
                "min_samples_split": [2, 5, 10]
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3]
            }
        }[clf_name]

        if run_gs:
            with st.spinner("Running GridSearch…"):
                gs = GridSearchCV(base_est, param_grid,
                                  scoring="f1", cv=5, n_jobs=-1)
                gs.fit(X_tr, y_tr)
                model = gs.best_estimator_
                best_params = gs.best_params_
                cv_f1 = gs.best_score_
        else:
            model = base_est.fit(X_tr, y_tr)
            best_params, cv_f1 = "—", None

        y_pred = model.predict(X_te)
        f1 = f1_score(y_te, y_pred)
        st.metric("F1-Score", f"{f1:.3f}")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig); plt.close(fig)

        prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
        ax.plot([0,1],[0,1],"k--"); ax.legend()
        st.pyplot(fig); plt.close(fig)

        with st.expander("Key Insights"):
            if run_gs:
                st.markdown(f"- **Grid Search** CV-F1 = **{cv_f1:.2f}**; best params `{best_params}`.")
            churn = (prob > 0.5).mean() * 100
            st.markdown(f"- Hold-out F1 **{f1:.2f}**; churn-risk (prob > 0.5) ≈ **{churn:.1f}%**.")
            st.markdown("- Metrics: F1, ROC-AUC; data: synthetic survey.")

    else:  # ─── Clustering ───
        k = st.slider("Number of clusters (k)", 2, 10, 4)
        X_scaled = MinMaxScaler().fit_transform(users_df[core_cols])

        km = KMeans(k, random_state=42, n_init="auto").fit(X_scaled)
        users_df["Cluster"] = km.labels_
        sil = silhouette_score(X_scaled, km.labels_)
        inertias = [
            KMeans(i, random_state=42, n_init="auto").fit(X_scaled).inertia_
            for i in range(2, 11)
        ]

        fig, ax = plt.subplots()
        ax.plot(range(2, 11), inertias, "o-"); ax.set_title("Elbow Curve")
        st.pyplot(fig); plt.close(fig)
        st.metric("Silhouette", f"{sil:.3f}")
        st.dataframe(users_df.groupby("Cluster")[core_cols].mean().round(2))

        with st.expander("Key Insights"):
            st.markdown(f"- Silhouette **{sil:.2f}** at k={k}.")
            st.markdown("- Clusters saved for Apriori analysis; metrics: inertia, silhouette.")

# ================== 3. Forecasting Tab ==================
with forecast_tab:
    st.header("📈 Forecasting")
    flav = st.selectbox("Flavour signal", trends_df.columns[1:])
    X = np.arange(len(trends_df)).reshape(-1, 1); y = trends_df[flav].values
    split = int(0.8*len(X))
    model = LinearRegression().fit(X[:split], y[:split])
    y_pred = model.predict(X[split:])
    r2 = r2_score(y[split:], y_pred)
    rmse = np.sqrt(mean_squared_error(y[split:], y_pred))
    st.metric("R²", f"{r2:.3f}"); st.metric("RMSE", f"{rmse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[split:], y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    st.pyplot(fig); plt.close(fig)

    slopes = {c: np.polyfit(np.arange(len(trends_df)), trends_df[c], 1)[0]
              for c in trends_df.columns[1:]}
    top_flav = max(slopes, key=slopes.get)

    with st.expander("Key Insights"):
        st.markdown(f"- Steepest upward trend: **{top_flav}**.")
        st.markdown(f"- Current model R² **{r2:.2f}**, RMSE **{rmse:.1f}**.")
        st.markdown("- Data: simulated weekly flavour mentions.")

# ================== 4. Apriori Tab ==================
with rules_tab:
    st.header("🧩 Micro-Batch & Drops — Apriori")
    sup = st.slider("Min support", 0.01, 0.4, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.05, 1.0, 0.3, 0.05)
    metric = st.selectbox("Sort metric", ["confidence", "lift", "leverage"])

    basket = users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket = pd.concat(
        [basket,
         pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool)],
        axis=1
    )
    freq = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)

    if not rules.empty:
        rules = rules.sort_values(metric, ascending=False).head(10)
        st.dataframe(rules)
        best = rules.iloc[0]
    else:
        st.warning("No rules for chosen thresholds.")
        best = None

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- Top rule: **{best['antecedents']} → {best['consequents']}** "
                        f"(confidence {best['confidence']:.2f}).")
        st.markdown("- Basket: flavour families + channel; metrics: confidence / lift.")
