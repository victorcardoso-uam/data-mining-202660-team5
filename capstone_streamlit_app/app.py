import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Session 24: Enterprise AI Decision Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8F9FA;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #ADB5BD;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E2125;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #343A40;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. CACHED DATASETS & MODEL TRAINING
# =============================================================================
@st.cache_data
def load_data():
    """Loads unified enterprise demonstration dataset."""
    try:
        df = pd.read_csv("session24_unified_enterprise_demo.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/session24_unified_enterprise_demo.csv")
    return df

@st.cache_resource
def train_models(df):
    """Trains baseline classification models on preprocessed features."""
    X = df[['operating_temperature_c', 'vibration_amplitude_mm', 'power_consumption_kw', 'pressure_psi', 'operating_hours']]
    y = df['system_failure_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    models = {
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Decision Tree (CART)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42)
    }
    
    fitted_models = {}
    performance = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        fitted_models[name] = model
        performance.append({
            "Model Algorithm": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 3)
        })
        
    return fitted_models, pd.DataFrame(performance), X_test, y_test

# Load data and train models
df_raw = load_data()
fitted_models, performance_df, X_test, y_test = train_models(df_raw)


# =============================================================================
# 3. SIDEBAR CONTROLS & NAVIGATION
# =============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=64)
st.sidebar.title("Enterprise AI Suite")
st.sidebar.caption("Data Mining & Modern AI Systems (IIND4417)")

st.sidebar.divider()
st.sidebar.subheader("🎛️ Global Inference Settings")
selected_model_name = st.sidebar.selectbox("Select Active Model:", list(fitted_models.keys()))
model_threshold = st.sidebar.slider("Risk Decision Cutoff (Threshold)", 0.05, 0.95, 0.50, 0.05)

st.sidebar.divider()
st.sidebar.info("📌 Master Blueprint Notice: Use this unified dashboard in Session 24 as your template. Adapt this architecture for your Capstone Project in Milestone 3/3 (Session 27 / Session 28 — Thursday, November 12, 2026)!")


# =============================================================================
# 4. MAIN EXECUTIVE DASHBOARD HEADER
# =============================================================================
st.markdown('<div class="main-header">⚡ Session 24: Unified Enterprise AI Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industrial Telemetry & Risk Prediction | Master Blueprint for Capstone Milestone 3/3</div>', unsafe_allow_html=True)

# Top Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Ingested Telemetry Records", f"{len(df_raw):,}", delta="100% Ingested")
with kpi2:
    st.metric("Feature Dimensions", f"{df_raw.shape[1] - 1} Features", delta="Engineered")
with kpi3:
    failure_rate = (df_raw['system_failure_flag'].mean()) * 100
    st.metric("System Failure Rate", f"{failure_rate:.1f}%", delta="Target: system_failure_flag")
with kpi4:
    best_auc = performance_df[performance_df['Model Algorithm'] == selected_model_name]['ROC-AUC'].values[0]
    st.metric("Active Model ROC-AUC", f"{best_auc:.3f}", delta=f"Model: {selected_model_name.split()[0]}")

st.divider()


# =============================================================================
# 5. WORKSHOP TABS: EXPLORATION, PIPELINE, BENCHMARK, & INFERENCE
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer & EDA", 
    "⚙️ Preprocessing & Correlation", 
    "🤖 Multi-Model Benchmark", 
    "🔮 Real-Time What-If Inference"
])

# -----------------------------------------------------------------------------
# TAB 1: DATA EXPLORER & EDA
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Raw Industrial Telemetry Ingestion & Statistical Profile")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.dataframe(df_raw.head(100), use_container_width=True, height=320)
    
    with col_b:
        st.write("**Dataset Summary Statistics:**")
        st.dataframe(df_raw.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)
    
    st.divider()
    st.subheader("Interactive Feature Distribution Analysis")
    selected_feature = st.selectbox("Select Feature to Visualize:", [c for c in df_raw.columns if c != 'system_failure_flag'])
    
    fig_hist = px.histogram(
        df_raw, 
        x=selected_feature, 
        color="system_failure_flag", 
        barmode="overlay",
        title=f"Distribution of {selected_feature} by System Failure Status",
        color_discrete_map={0: "#10B981", 1: "#EF4444"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: PREPROCESSING & CORRELATION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Feature Correlation & Preprocessing Matrix")
    num_df = df_raw.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Pearson Feature Correlation Matrix",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: MULTI-MODEL BENCHMARK
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Empirical Model Performance Benchmark Suite")
    st.dataframe(performance_df, use_container_width=True)
    
    col_m1, col_m2 = st.columns(2)
    active_model = fitted_models[selected_model_name]
    y_prob_test = active_model.predict_proba(X_test)[:, 1] if hasattr(active_model, "predict_proba") else active_model.predict(X_test)
    y_pred_custom = (y_prob_test >= model_threshold).astype(int)
    
    with col_m1:
        st.subheader(f"Confusion Matrix ({selected_model_name})")
        cm = confusion_matrix(y_test, y_pred_custom)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Predicted Safe (0)', 'Predicted Failure (1)'],
            y=['Actual Safe (0)', 'Actual Failure (1)'],
            color_continuous_scale="Reds",
            title=f"Threshold = {model_threshold:.2f}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_m2:
        st.subheader("ROC-AUC Curve Comparison")
        fig_roc = go.Figure()
        for name, m in fitted_models.items():
            prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = roc_auc_score(y_test, prob)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc_val:.2f})"))
            
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Chance'))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves across Algorithms")
        st.plotly_chart(fig_roc, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: REAL-TIME WHAT-IF SCENARIO INFERENCE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Real-Time What-If Scenario Prediction Engine")
    st.write("Adjust industrial telemetry sliders to test real-time risk predictions:")
    
    col_i1, col_i2 = st.columns([1, 1])
    
    with col_i1:
        st.write("### 🎛️ Telemetry Input Parameters")
        input_temp = st.slider("Operating Temperature (°C):", 30.0, 110.0, 65.0)
        input_vib = st.slider("Vibration Amplitude (mm):", 0.1, 8.0, 1.5)
        input_pwr = st.slider("Power Consumption (kW):", 10.0, 60.0, 35.0)
        input_press = st.slider("Pressure (PSI):", 20.0, 150.0, 75.0)
        input_hrs = st.slider("Operating Hours:", 50, 6000, 1500)
        
    with col_i2:
        st.write("### 🎯 Live Model Inference Output")
        input_df = pd.DataFrame([{
            'operating_temperature_c': input_temp,
            'vibration_amplitude_mm': input_vib,
            'power_consumption_kw': input_pwr,
            'pressure_psi': input_press,
            'operating_hours': input_hrs
        }])
        
        simulated_prob = active_model.predict_proba(input_df)[0, 1]
        predicted_class = 1 if simulated_prob >= model_threshold else 0
        
        st.markdown(f"#### Predicted Failure Probability: **{simulated_prob*100:.1f}%**")
        st.progress(float(simulated_prob))
        
        if predicted_class == 1:
            st.error(f"⚠️ CRITICAL ALERT: System Failure Risk Detected! (Cutoff = {model_threshold:.2f})")
            st.write("<b>Recommended Client Operational Policy:</b> Trigger immediate preventative maintenance lockout and dispatch field engineers.")
        else:
            st.success(f"✅ NORMAL OPERATION: Safe Telemetry State (Cutoff = {model_threshold:.2f})")
            st.write("<b>Recommended Client Operational Policy:</b> Continue standard telemetry monitoring schedule.")
            
        st.divider()
        st.write("**Scenario Features Transmitted:**")
        st.json(input_df.to_dict(orient='records')[0])
