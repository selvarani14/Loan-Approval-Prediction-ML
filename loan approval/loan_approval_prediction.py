# loan_long_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, time, joblib, traceback

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# --- Page config & theme (like flight app) ---
st.set_page_config(page_title="🏦 Loan Status Prediction (Long)", layout="wide", page_icon="💳")
sns.set_style("whitegrid")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
    color: #0f111a;
    font-family: 'Segoe UI', sans-serif;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #1f3c88;
    text-align: center;
    margin-bottom: 10px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.12);
}
.stButton>button {
    background-color: #1f3c88;
    color: white;
    font-weight: 700;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🏦 Loan Status Prediction</h1>', unsafe_allow_html=True)

# ---------------- Sidebar configuration (similar to flight app) ----------------
st.sidebar.header("Configuration")
sample_limit = st.sidebar.slider("Max rows to use (sample)", min_value=500, max_value=200000, value=50000, step=500)
generate_extra = st.sidebar.checkbox("Generate extra engineered features", value=True)
n_extra = st.sidebar.slider("SelectKBest — keep top K features", min_value=5, max_value=500, value=60, step=5)
train_all = st.sidebar.checkbox("Train ALL models? (ignore ≡ train 3 models only)", value=False)  # we'll still train only 3 models
sample_for_curve = st.sidebar.slider("Samples used for learning curve", min_value=200, max_value=10000, value=2000, step=200)

st.sidebar.markdown("---")
st.sidebar.write("Models: Logistic Regression, Random Forest, SVM (default).")
st.sidebar.write("Feature engineering: polynomial (deg2), interaction cat*num, freq/text features.")

# ---------------- Helpers (adapted from flight code) ----------------
def basic_encoding(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str).fillna("___missing___"))
        X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str).fillna("___missing___"))
    return X_train, X_test, {"ordinal_encoder": encoder, "cat_cols": cat_cols}, cat_cols

def add_freq_encoding(X_train, X_test, cat_cols):
    for col in cat_cols:
        vc = X_train[col].value_counts(normalize=True)
        X_train[col + "_freq"] = X_train[col].map(vc).fillna(0)
        X_test[col + "_freq"] = X_test[col].map(vc).fillna(0)
    return X_train, X_test

def add_text_features(X_train, X_test, cols):
    for col in cols:
        X_train[col + "_len"] = X_train[col].fillna("").astype(str).apply(len)
        X_test[col + "_len"] = X_test[col].fillna("").astype(str).apply(len)
        X_train[col + "_words"] = X_train[col].fillna("").astype(str).apply(lambda s: len(s.split()))
        X_test[col + "_words"] = X_test[col].fillna("").astype(str).apply(lambda s: len(s.split()))
    return X_train, X_test

def gen_poly_features(X_train_num, X_test_num, max_out=200):
    if X_train_num.shape[1] == 0:
        return pd.DataFrame(index=X_train_num.index), pd.DataFrame(index=X_test_num.index), []
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    Xt_train_poly = poly.fit_transform(X_train_num.fillna(0))
    Xt_test_poly = poly.transform(X_test_num.fillna(0))
    try:
        names = poly.get_feature_names_out(X_train_num.columns)
    except Exception:
        names = []
        cols = X_train_num.columns.tolist()
        m = len(cols)
        for i in range(m):
            names.append(cols[i])
        for i in range(m):
            for j in range(i, m):
                names.append(f"{cols[i]}__x__{cols[j]}")
    poly_train = pd.DataFrame(Xt_train_poly, columns=names, index=X_train_num.index)
    poly_test  = pd.DataFrame(Xt_test_poly, columns=names, index=X_test_num.index)
    if poly_train.shape[1] > max_out:
        variances = poly_train.var().sort_values(ascending=False)
        keep = variances.index[:max_out].tolist()
        poly_train = poly_train[keep]
        poly_test = poly_test[keep]
        names = keep
    return poly_train, poly_test, names

def gen_interaction_features(X_train, X_test, cat_label_cols, num_cols, max_pairs=300):
    frames_train, frames_test, pairs = [], [], []
    count = 0
    for c in cat_label_cols:
        if c not in X_train.columns: continue
        for n in num_cols:
            if n not in X_train.columns: continue
            frames_train.append((X_train[c] * X_train[n]).rename(f"{c}*{n}"))
            frames_test.append((X_test[c] * X_test[n]).rename(f"{c}*{n}"))
            pairs.append(f"{c}*{n}")
            count += 1
            if count >= max_pairs: break
        if count >= max_pairs: break
    if frames_train:
        return pd.concat(frames_train, axis=1), pd.concat(frames_test, axis=1), pairs
    else:
        return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index), []

def select_top_k(X_train_cand, y_train, X_test_cand, k):
    imputer = SimpleImputer(strategy='median')
    Xtr = imputer.fit_transform(X_train_cand)
    Xte = imputer.transform(X_test_cand)
    selector = SelectKBest(score_func=f_classif, k=min(k, Xtr.shape[1]))
    selector.fit(Xtr, y_train)
    mask = selector.get_support()
    cols = X_train_cand.columns[mask]
    return pd.DataFrame(Xtr[:, mask], columns=cols, index=X_train_cand.index), pd.DataFrame(Xte[:, mask], columns=cols, index=X_test_cand.index), cols.tolist()

def train_and_eval(model, X_train, y_train, X_test, y_test, name, show_learning_curve=True, sample_for_curve=2000):
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    lc_fig = None
    try:
        if show_learning_curve and len(X_train) > 20:
            n_samples = min(len(X_train), sample_for_curve)
            if n_samples < len(X_train):
                ix = np.random.choice(len(X_train), n_samples, replace=False)
                X_lc = X_train.iloc[ix]
                y_lc = np.array(y_train)[ix]
            else:
                X_lc, y_lc = X_train, y_train
            train_sizes, train_scores, test_scores = learning_curve(model, X_lc, y_lc, cv=3,
                                                                    train_sizes=np.linspace(0.2,1.0,5), scoring='accuracy', n_jobs=1)
            train_scores_mean = train_scores.mean(axis=1)
            test_scores_mean = test_scores.mean(axis=1)
            fig, ax = plt.subplots()
            ax.plot(train_sizes, train_scores_mean, 'o-', label='Train score')
            ax.plot(train_sizes, test_scores_mean, 'o-', label='CV score')
            ax.set_xlabel("Training examples")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Learning curve - {name}")
            ax.legend(loc='best')
            lc_fig = fig
    except Exception:
        lc_fig = None

    return {
        "model": model,
        "time": t1 - t0,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
        "y_pred": y_pred,
        "learning_curve_fig": lc_fig
    }

# ---------------- Upload dataset (like flight app) ----------------
uploaded_file = st.file_uploader("Upload CSV dataset (loan dataset)", type=["csv"])
if uploaded_file is None:
    st.info("Upload your CSV file (columns like loan_id, income_annum, loan_amount, cibil_score, loan_status etc.).")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Data shape:", df.shape)

if len(df) > sample_limit:
    df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
    st.warning(f"Data sampled to {sample_limit} rows to speed up processing.")

# ---------------------- INTRODUCTION ----------------------
st.markdown(f"""
### 📝 Introduction  
This Loan Status Prediction dashboard is built using a structured loan dataset that contains key financial, demographic, and credit-related information of customers who applied for a loan.  
The dataset typically includes attributes such as:

- Applicant Income  
- Co-applicant Income  
- Loan Amount  
- Credit History  
- Education Level  
- Employment Type  
- Property Area  
- Gender and Marital Status  

Using these features, the dashboard aims to analyze patterns that influence loan approval and apply machine-learning models to predict whether an application is likely to be **Approved** or **Rejected**.

This dashboard helps users to:  
- Understand the dataset with quick visual insights  
- Explore important customer attributes  
- Apply feature engineering and SelectKBest filtering  
- Train multiple machine-learning models  
- Compare model accuracy using visual charts  

By transforming raw loan data into meaningful insights and predictions, this project demonstrates how machine learning can support risk assessment, automate decision-making, and improve efficiency in the loan approval process.

---

### 📦 Dataset Information  
- **Rows:** {df.shape[0]}  
- **Columns:** {df.shape[1]}  
- **First 12 Columns:** {', '.join(df.columns[:12])}{'...' if len(df.columns) > 12 else ''}

""")


st.subheader("Step 1 — Quick EDA")
with st.expander("Show dataset head"):
    st.write(df.head())

# ---------------- Visualizations — ONLY 1 (categorical count plot) ----------------
st.markdown("---")
st.subheader("📊 Visualization")

# Select categorical column (first object column is auto-selected)
cat_columns = df.select_dtypes(include='object').columns

if len(cat_columns) > 0:
    cat_col = st.selectbox("Choose a categorical column for bar chart", options=cat_columns)

    fig1, ax1 = plt.subplots()
    sns.countplot(x=cat_col, data=df, palette='pastel', ax=ax1)
    ax1.set_title(f"Count of {cat_col}")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

else:
    st.info("No categorical columns detected for bar chart visualization.")

# ---------------- Allow dropping columns & selecting features/target ----------------
st.markdown("---")

st.subheader("Step 2 — Features & Target")

drop_default = [c for c in df.columns if any(x in c.lower() for x in ["id","name","loan_id"])]
drop_cols = st.multiselect("Columns to drop (irrelevant):", options=df.columns.tolist(), default=drop_default)
if drop_cols:
    df = df.drop(columns=drop_cols, errors='ignore')
    st.write("Dropped columns. New shape:", df.shape)

cols = df.columns.tolist()
default_target = "loan_status" if "loan_status" in cols else cols[-1]
target = st.selectbox("Target column (loan_status)", options=cols, index=cols.index(default_target))
features = st.multiselect("Input features (choose at least 1)", options=[c for c in cols if c != target], default=[c for c in cols if c != target][:6])

if not features:
    st.warning("Please select at least one feature.")
    st.stop()

X = df[features].copy()
y = df[target].copy()

# Split early to fit encoders only on train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=None)

# ---------------- Preprocessing & feature engineering (step-by-step with progress) ----------------
st.subheader("Step 3 — Preprocessing & Feature Engineering")
progress = st.progress(0)
step = 0
total_steps = 5

st.write("3.1 Label-encoding categorical columns..."); step += 1; progress.progress(step/total_steps)
X_train_encoded, X_test_encoded, enc_maps, cat_label_cols = basic_encoding(X_train.copy(), X_test.copy())
st.write("Encoded categorical columns:", cat_label_cols)

if generate_extra:
    st.write("3.2 Frequency & text features..."); step += 1; progress.progress(step/total_steps)
    orig_cat_cols = [c for c in features if df[c].dtype == 'object' or df[c].dtype.name == 'category']
    if orig_cat_cols:
        X_train_encoded, X_test_encoded = add_freq_encoding(X_train_encoded, X_test_encoded, orig_cat_cols)
        X_train_encoded, X_test_encoded = add_text_features(X_train_encoded, X_test_encoded, orig_cat_cols)

# numeric cols after encoding
num_cols = X_train_encoded.select_dtypes(include=[np.number]).columns.tolist()
st.write("Numeric columns available:", len(num_cols))

poly_train = pd.DataFrame(index=X_train_encoded.index)
poly_test  = pd.DataFrame(index=X_test_encoded.index)
poly_names = []
if generate_extra and len(num_cols) > 0:
    st.write("3.3 Generating polynomial features (degree=2)..."); step += 1; progress.progress(step/total_steps)
    max_poly_cols = 50
    use_num_cols = num_cols if len(num_cols) <= max_poly_cols else num_cols[:max_poly_cols]
    poly_train, poly_test, poly_names = gen_poly_features(X_train_encoded[use_num_cols], X_test_encoded[use_num_cols], max_out=200)
    st.write("Polynomial features (created):", len(poly_names))

inter_train = pd.DataFrame(index=X_train_encoded.index)
inter_test  = pd.DataFrame(index=X_test_encoded.index)
inter_names = []
if generate_extra and cat_label_cols and num_cols:
    st.write("3.4 Generating interaction features (cat_label * numeric) ..."); step += 1; progress.progress(step/total_steps)
    inter_train, inter_test, inter_names = gen_interaction_features(X_train_encoded, X_test_encoded, cat_label_cols, num_cols, max_pairs=300)
    st.write("Interaction features (created):", len(inter_names))

st.write("3.5 Assembling candidate features..."); step += 1; progress.progress(step/total_steps)
candidates_train = pd.concat([X_train_encoded.select_dtypes(include=[np.number]), poly_train, inter_train], axis=1).fillna(0)
candidates_test  = pd.concat([X_test_encoded.select_dtypes(include=[np.number]), poly_test, inter_test], axis=1).fillna(0)
st.write("Candidate feature matrix shape (train):", candidates_train.shape)

# ---------------- Feature selection ---------------
st.markdown("---")
st.subheader("Step 4 — Feature Selection (SelectKBest)")
k = min(n_extra, candidates_train.shape[1])
if k < 1:
    st.error("No candidate features available — reduce 'n_extra' or enable feature generation.")
    st.stop()

X_tr_sel, X_te_sel, selected_cols = select_top_k(candidates_train, np.array(LabelEncoder().fit_transform(y_train.astype(str))), candidates_test, k=k)
st.write(f"Selected top {len(selected_cols)} features.")

# ---------------- Model training ----------------
st.markdown("---")
st.subheader("Step 5 — Train (3 models)")

models_map = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train only the 3 selected models
to_train = list(models_map.keys())
results = {}

if st.button("🚀 Start training"):
    overall_progress = st.progress(0)
    for i, mname in enumerate(to_train):
        st.write(f"Training model: **{mname}** ({i+1}/{len(to_train)})")
        model = models_map[mname]
        try:
            res = train_and_eval(
                model,
                X_tr_sel,
                np.array(LabelEncoder().fit_transform(y_train.astype(str))),
                X_te_sel,
                np.array(LabelEncoder().fit_transform(y_test.astype(str))),
                name=mname,
                show_learning_curve=True,
                sample_for_curve=sample_for_curve
            )

            results[mname] = res
            st.success(f"{mname} finished — Accuracy: {res['accuracy']:.3f}  Time: {res['time']:.1f}s")

            # Metric cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{res['accuracy']:.3f}")
            c2.metric("Precision", f"{res['precision']:.3f}")
            c3.metric("Recall", f"{res['recall']:.3f}")
            c4.metric("F1-Score", f"{res['f1']:.3f}")

            # Classification report
            st.write("Classification report (per class):")
            report_df = pd.DataFrame(res['report']).transpose()
            st.dataframe(report_df, use_container_width=True)

            # Confusion matrix
            cm = confusion_matrix(
                LabelEncoder().fit_transform(y_test.astype(str)),
                res['y_pred']
            )
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_title(f"Confusion Matrix - {mname}")
            st.pyplot(fig_cm)

            # Learning curve
            if res['learning_curve_fig'] is not None:
                st.write("Learning curve:")
                st.pyplot(res['learning_curve_fig'])

            # Download model
            buf = io.BytesIO()
            joblib.dump(res['model'], buf)
            buf.seek(0)
            st.download_button(
                label=f"Download model: {mname}.joblib",
                data=buf,
                file_name=f"{mname}_model.joblib"
            )

        except Exception as e:
            st.error(f"Model {mname} failed: {e}")
            st.text(traceback.format_exc())

        overall_progress.progress((i + 1) / len(to_train))

    st.success("All requested models finished training.")

# ---------------- Model comparison (bar chart + grouped bars for metrics) ----------------
if results:
    st.markdown("---")
    st.subheader("Model Comparison")

    # Safely build comp table
    comp_rows = []
    for k, v in results.items():
        if all(m in v for m in ["accuracy", "precision", "recall", "f1", "time"]):
            comp_rows.append({
                "model": k,
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1"],
                "time_s": v["time"]
            })

    if len(comp_rows) == 0:
        st.warning("⚠️ No valid model results found. Please train models again.")
    else:
        comp = pd.DataFrame(comp_rows)

        st.write("Summary table:")
        st.dataframe(comp.sort_values("accuracy", ascending=False), use_container_width=True)

        # ---------------- Accuracy Bar Chart ----------------
        fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=comp.sort_values("accuracy", ascending=False),
            x="model", y="accuracy", palette="pastel", ax=ax_acc
        )
        ax_acc.set_ylim(0, 1)
        ax_acc.set_title("Model Accuracy Comparison")
        st.pyplot(fig_acc)

        # ---------------- Grouped Metrics Chart ----------------
        metrics = ["accuracy", "precision", "recall", "f1"]
        fig_metrics, axm = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comp))
        width = 0.18

        for idx, metric in enumerate(metrics):
            axm.bar(x + (idx - 1.5) * width, comp[metric].values,
                    width=width, label=metric.capitalize())

        axm.set_xticks(x)
        axm.set_xticklabels(comp["model"].values)
        axm.set_ylim(0, 1)
        axm.set_title("Model Metrics Comparison")
        axm.legend()
        st.pyplot(fig_metrics)

        # ---------------- Conclusion ----------------
        st.markdown("---")
        st.subheader("📌 Conclusion")

        best_model = comp.loc[comp['accuracy'].idxmax()]

        model_name = str(best_model['model'])
        acc = float(best_model['accuracy'])
        prec = float(best_model['precision'])
        rec = float(best_model['recall'])
        f1 = float(best_model['f1'])

        text = (
            "We trained **" + str(len(results)) + "** ML models on your dataset.\n\n"
            "- **Best Model:** " + model_name + "\n"
            "- **Accuracy:** " + str(round(acc, 3)) + "\n"
            "- **Precision:** " + str(round(prec, 3)) + "\n"
            "- **Recall:** " + str(round(rec, 3)) + "\n"
            "- **F1 Score:** " + str(round(f1, 3))
        )

        st.markdown(text)

else:
    st.warning("⚠️ No model results found. Train at least one model.")
