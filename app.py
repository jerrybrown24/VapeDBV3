# ───────── VaporIQ Dashboard  • v7  ─────────
# Full app.py with:
#  • Flicker-proof galaxy star-field behind a slow smoke overlay
#  • Data-Viz, TasteDNA (extra metrics), Forecast (regressor selector) & Apriori tabs
#  • All earlier captions / key-insights
# -------------------------------------------

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

# ─────────────────────  Page & Gradient  ─────────────────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ─────────  Galaxy star-field (body::before)  ─────────
star_path = Path(__file__).with_name("starfield.png")
if star_path.exists():
    star_b64 = base64.b64encode(star_path.read_bytes()).decode()
    st.markdown(textwrap.dedent(f"""
        <style>
        /* Deep-space gradient already in style.css; now add drifting stars */
        body::before {{
            content:"";
            position:fixed; inset:0; z-index:-4;       /* below smoke, above gradient */
            pointer-events:none;
            background:url("data:image/png;base64,{star_b64}") repeat;
            background-size:600px;
            opacity:.35;
            animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{
          0%   {{transform:translate3d(0,0,0);}}
          100% {{transform:translate3d(-2000px,1500px,0);}}
        }}

        /* Slow, opposing smoke flow */
        .smoke-layer    {{animation:smokeFlow 210s linear infinite;  opacity:.25;}}
        .smoke-layer-2  {{animation:smokeFlowR 280s linear infinite; opacity:.15;}}
        @keyframes smokeFlow  {{0%{{background-position:0 0}} 100%{{background-position:1600px 0}}}}
        @keyframes smokeFlowR {{0%{{background-position:0 0}} 100%{{background-position:-1600px 0}}}}
        </style>
    """), unsafe_allow_html=True)
else:
    st.sidebar.error("⚠️ `starfield.png` not found – galaxy backdrop disabled.")

# Inject smoke divs (they rely on CSS above)
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)

# ─────────  Watermark bottle  ─────────
with open("vape_watermark.png","rb") as f:
    wm_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"<img src='data:image/png;base64,{wm_b64}' style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>",
            unsafe_allow_html=True)

# ─────────────────────  Data  ─────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("users_synthetic.csv"), pd.read_csv("flavor_trends.csv")
users_df, trends_df = load_data()
core = ["Age","SweetLike","MentholLike","PodsPerWeek"]

# ─────────────────────  Tabs  ─────────────────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# =================== 1. Data-Viz TAB ===================
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
        st.warning("No rows match current filters — tweak sidebar options.")
        st.stop()

    ## 1 Violin (Age)
    hex_fig = px.density_heatmap(
        df, x="Age", y="PodsPerWeek",
        nbinsx=30, nbinsy=15,
        color_continuous_scale="magma",
        title="Density of Consumption by Age")
    st.plotly_chart(hex_fig, use_container_width=True)
    st.caption("Hot-spots reveal which age bands are heavy users.\n"
               "Darker cells indicate more vapers in that age-volume bin.")

    ## 2 Scatter Pods vs Age
    fig, ax = plt.subplots(); sns.scatterplot(data=df,x="Age",y="PodsPerWeek",ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Do older vapers consume more or fewer pods? Helps with age-wise inventory planning.")

    ## 3 Correlation heat-map
    fig, ax = plt.subplots(); sns.heatmap(df[["Age","SweetLike","MentholLike","PodsPerWeek"]].corr(),
                                          annot=True,cmap="coolwarm",ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Quickly spots numeric relationships—strong pairs become candidate features.")

    ## 4 Bar: flavour family counts
    flat = df["FlavourFamilies"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
    st.bar_chart(flat)
    st.caption("Which flavour families dominate under current filters? Guides micro-batch choices.")

    ## 5 Top-3 flavour trends
    top3 = trends_df.drop(columns="Date").mean().nlargest(3).index
    st.plotly_chart(px.line(trends_df, x="Date", y=top3, title="Top-3 Flavour Trends"),
                    use_container_width=True)
    st.caption("Comparative momentum of leading flavours shapes forward-looking production.")

    ## 6 Boxplot PodsPerWeek by channel
    fig, ax = plt.subplots(); sns.boxplot(data=df,x="PurchaseChannel",y="PodsPerWeek",ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.caption("Distribution & outliers across channels—helps craft channel-specific offers.")

    ## 7 Stacked bar Intent vs Gender
    sb = pd.crosstab(df["Gender"], df["SubscribeIntent"])
    st.plotly_chart(px.bar(sb, barmode="stack",title="Subscribe Intent × Gender"),
                    use_container_width=True)
    st.caption("Visualises intent skew by gender—inform copywriting & A/B tests.")

    ## 8 Rugplots Sweet & Menthol liking
    fig, ax = plt.subplots()
    sns.rugplot(df["SweetLike"],height=.1,color="g",ax=ax,label="Sweet")
    sns.rugplot(df["MentholLike"],height=.1,color="r",ax=ax,label="Menthol")
    ax.legend(); st.pyplot(fig); plt.close(fig)
    st.caption("Preference density overlay hints at flavour gaps to exploit.")

    ## 9 Treemap cluster × flavour
    if "Cluster" not in users_df.columns:
        users_df["Cluster"] = KMeans(4,random_state=42,n_init="auto") \
            .fit_predict(MinMaxScaler().fit_transform(users_df[core]))
    tdf = users_df.assign(lead_flav=users_df["FlavourFamilies"].str.split(",").str[0])
    st.plotly_chart(px.treemap(tdf, path=["Cluster","lead_flav"], values="PodsPerWeek"),
                    use_container_width=True)
    st.caption("Maps behavioural clusters to lead flavours—find underserved combos.")

    ## 10 Cum Custard Kunafa
    ck = trends_df.assign(cum=trends_df["Custard Kunafa"].cumsum())
    st.plotly_chart(px.area(ck, x="Date", y="cum", title="Cumulative Mentions – Custard Kunafa"),
                    use_container_width=True)
    st.caption("Total buzz size—helps size micro-batch production for trending flavour.")

    with st.expander("Key Insights"):
        dom_gender = df["Gender"].value_counts(normalize=True).idxmax()
        fast_flav  = trends_df.drop(columns="Date").mean().idxmax()
        st.markdown(f"- Dominant gender in current filters: **{dom_gender}**")
        st.markdown(f"- Fastest rising flavour overall: **{fast_flav}**")
        st.markdown("- Data derived from 1 200 synthetic survey rows + 120-week trend crawl.")

# =================== 2. TasteDNA TAB ===================
with taste_tab:
    st.header("🔮 TasteDNA Engine")
    m_mode = st.radio("Mode",["Classification","Clustering"],horizontal=True)

    if m_mode == "Classification":
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

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt="d",cmap="Blues",ax=ax)
        st.pyplot(fig); plt.close(fig)

        with st.expander("Key Insights"):
            if best: st.markdown(f"- GridSearch best params: `{best}`")
            st.markdown(f"- Precision {prec:.2f}, Recall {rec:.2f}, F1 {f1:.2f}")

    else:  # Clustering
        k = st.slider("k clusters",2,10,4)
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
    reg_name = st.selectbox("Regressor",["Linear","Ridge","Lasso","Decision Tree"])
    reg_map = {"Linear":LinearRegression(),
               "Ridge":Ridge(alpha=1.0),
               "Lasso":Lasso(alpha=0.01),
               "Decision Tree":DecisionTreeRegressor(max_depth=5,random_state=42)}
    reg = reg_map[reg_name]

    X = np.arange(len(trends_df)).reshape(-1,1); y = trends_df[flavour].values
    split=int(.8*len(X)); reg.fit(X[:split],y[:split]); y_pred=reg.predict(X[split:])
    r2=r2_score(y[split:],y_pred); rmse=np.sqrt(mean_squared_error(y[split:],y_pred))

    st.metric("R²",f"{r2:.3f}"); st.metric("RMSE",f"{rmse:.2f}")

    fig,ax=plt.subplots()
    ax.scatter(y[split:],y_pred,alpha=.6)
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
    ax.set_xlabel("Actual Mentions"); ax.set_ylabel("Predicted Mentions")
    st.pyplot(fig); plt.close(fig)

    with st.expander("Key Insights"):
        slopes={c:np.polyfit(np.arange(len(trends_df)), trends_df[c],1)[0] for c in trends_df.columns[1:]}
        st.markdown(f"- Regressor **{reg_name}** → R² {r2:.2f}, RMSE {rmse:.2f}")
        st.markdown(f"- Steepest flavour slope: **{max(slopes,key=slopes.get)}**")

# =================== 4. Apriori TAB ===================
with rules_tab:
    st.header("🧩 Apriori Explorer")
    sup=st.slider("Support",0.01,0.4,0.05,0.01); conf=st.slider("Confidence",0.05,1.0,0.3,0.05)
    basket=users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket=pd.concat([basket,pd.get_dummies(users_df["PurchaseChannel"],prefix="Chan").astype(bool)],axis=1)
    freq=apriori(basket,min_support=sup,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=conf)

    if rules.empty:
        st.warning("No rules under thresholds.")
        best=None
    else:
        rules=rules.sort_values("confidence",ascending=False).head(10)
        st.dataframe(rules); best=rules.iloc[0]

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- Best rule: {best['antecedents']} → {best['consequents']} (lift {best['lift']:.2f})")
        st.markdown(f"- Support ≥ {sup:.2f}, Confidence ≥ {conf:.2f}")
