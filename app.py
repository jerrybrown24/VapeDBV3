import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
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
import base64, matplotlib

# ───────── Theme / starfield / watermark ─────────
st.set_page_config(page_title="VaporIQ v7", layout="wide")
with open("style.css") as css: st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
st.markdown('<div class="starfield"></div>', unsafe_allow_html=True)       # NEW star layer
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)
with open("vape_watermark.png","rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.markdown(f"<img src='data:image/png;base64,{b64}' style='position:fixed;bottom:15px;right:15px;width:110px;opacity:0.8;z-index:1;'/>", unsafe_allow_html=True)

# ───────── Data ─────────
@st.cache_data
def load_data():
    return pd.read_csv("users_synthetic.csv"), pd.read_csv("flavor_trends.csv")
users_df, trends_df = load_data()
core = ["Age","SweetLike","MentholLike","PodsPerWeek"]

# ───────── Tabs ─────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization","TasteDNA","Forecasting","Micro-Batch & Drops"])

# =================== 1. DATA-VIZ TAB (unchanged charts) ===================
with viz:
    st.header("📊 Data Visualization Explorer")
    g_sel = st.sidebar.multiselect("Gender", users_df["Gender"].unique(), users_df["Gender"].unique())
    c_sel = st.sidebar.multiselect("Purchase Chan", users_df["PurchaseChannel"].unique(), users_df["PurchaseChannel"].unique())
    df = users_df[users_df["Gender"].isin(g_sel) & users_df["PurchaseChannel"].isin(c_sel)]
    # …(keep existing 10 charts + captions)…
    with st.expander("Key Insights"):
        retail = df[df["PurchaseChannel"]=="Retail"]["PodsPerWeek"].mean()
        online = df[df["PurchaseChannel"]=="Online"]["PodsPerWeek"].mean()
        fastest = trends_df.drop(columns="Date").mean().idxmax()
        st.markdown(f"- **Most skewed demographic:** Retail users avg **{retail:.1f}** pods/week vs Online **{online:.1f}**.")
        st.markdown(f"- **Fastest rising flavour:** {fastest}.")
        st.markdown("- Data: synthetic survey + simulated trend scrape.")

# =================== 2. TASTEDNA TAB ===================
with taste_tab:
    st.header("🔮 TasteDNA Analysis")
    mode = st.radio("Mode", ["Classification","Clustering"], horizontal=True)
    if mode=="Classification":
        clf_name = st.selectbox("Classifier",["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        run_gs = st.checkbox("Run GridSearch (5-fold CV)", value=False)
        X,y = users_df[core], users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)

        base = {"KNN":KNeighborsClassifier(),
                "Decision Tree":DecisionTreeClassifier(random_state=42),
                "Random Forest":RandomForestClassifier(random_state=42),
                "Gradient Boosting":GradientBoostingClassifier(random_state=42)}[clf_name]
        grid = {"KNN":{"n_neighbors":[3,5,7,9],"weights":["uniform","distance"]},
                "Decision Tree":{"max_depth":[None,3,5,7],"min_samples_split":[2,5,10]},
                "Random Forest":{"n_estimators":[100,200,300],"max_depth":[None,10,20]},
                "Gradient Boosting":{"n_estimators":[100,200],"learning_rate":[0.05,0.1],"max_depth":[2,3]}}[clf_name]

        if run_gs:
            gs = GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1)
            gs.fit(X_tr,y_tr); model=gs.best_estimator_; best=gs.best_params_
        else:
            model = base.fit(X_tr,y_tr); best=None

        y_pred = model.predict(X_te); prob = model.predict_proba(X_te)[:,1]
        f1 = f1_score(y_te,y_pred); prec=precision_score(y_te,y_pred)
        rec = recall_score(y_te,y_pred); acc=accuracy_score(y_te,y_pred)

        col1,col2,col3 = st.columns(3)
        col1.metric("Precision", f"{prec:.2f}")
        col2.metric("Recall",    f"{rec:.2f}")
        col3.metric("Accuracy",  f"{acc:.2f}")
        st.metric("F1-Score", f"{f1:.2f}")

        fig,ax=plt.subplots(); sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt="d",cmap="Blues",ax=ax)
        st.pyplot(fig); plt.close(fig)

        with st.expander("Key Insights"):
            st.markdown(f"- **Precision** {prec:.2f}, **Recall** {rec:.2f}, **F1** {f1:.2f}.")
            if best: st.markdown(f"- Best params: `{best}`")

    else:  # clustering unchanged
        k=st.slider("k clusters",2,10,4)
        X_scaled=MinMaxScaler().fit_transform(users_df[core])
        km=KMeans(k,random_state=42,n_init='auto').fit(X_scaled)
        sil=silhouette_score(X_scaled,km.labels_)
        st.metric("Silhouette", f"{sil:.3f}")
        users_df["Cluster"]=km.labels_

# =================== 3. FORECAST TAB ===================
with forecast_tab:
    st.header("📈 Forecasting")
    flav = st.selectbox("Flavour signal", trends_df.columns[1:])
    reg_name = st.selectbox("Regressor",
                ["Linear Regression","Ridge","Lasso","Decision Tree Regressor"])
    X=np.arange(len(trends_df)).reshape(-1,1); y=trends_df[flav].values
    split=int(0.8*len(X))

    reg_dict={"Linear Regression":LinearRegression(),
              "Ridge":Ridge(alpha=1.0),
              "Lasso":Lasso(alpha=0.01),
              "Decision Tree Regressor":DecisionTreeRegressor(max_depth=5,random_state=42)}
    reg = reg_dict[reg_name].fit(X[:split],y[:split])
    y_pred = reg.predict(X[split:])
    r2=r2_score(y[split:],y_pred); rmse=np.sqrt(mean_squared_error(y[split:],y_pred))

    st.metric("R²",f"{r2:.3f}"); st.metric("RMSE",f"{rmse:.2f}")

    fig,ax=plt.subplots()
    ax.scatter(y[split:],y_pred,alpha=0.6)
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--')
    ax.set_xlabel("Actual Mentions"); ax.set_ylabel("Predicted Mentions")
    st.pyplot(fig); plt.close(fig)

    with st.expander("Key Insights"):
        slopes={c:np.polyfit(np.arange(len(trends_df)),trends_df[c],1)[0]
                for c in trends_df.columns[1:]}
        top=max(slopes,key=slopes.get)
        st.markdown(f"- **Regressor:** {reg_name} (R² {r2:.2f}).")
        st.markdown(f"- Fastest slope: **{top}**.")

# =================== 4. APRIORI TAB (insight updated) ===================
with rules_tab:
    st.header("🧩 Micro-Batch & Drops — Apriori")
    sup=st.slider("Support",0.01,0.4,0.05,0.01); conf=st.slider("Confidence",0.05,1.0,0.3,0.05)
    metric=st.selectbox("Sort metric",["confidence","lift","leverage"])
    basket=users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket=pd.concat([basket,pd.get_dummies(users_df["PurchaseChannel"],prefix="Chan").astype(bool)],axis=1)
    freq=apriori(basket,min_support=sup,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=conf)
    if rules.empty:
        st.warning("No rules."); best=None
    else:
        rules=rules.sort_values(metric,ascending=False).head(10); st.dataframe(rules); best=rules.iloc[0]

    with st.expander("Key Insights"):
        if best is not None:
            st.markdown(f"- **Top rule:** {best['antecedents']} → {best['consequents']} (lift {best['lift']:.2f}).")
        st.markdown(f"- Support threshold used: {sup:.2f}")
