import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%) !important;
        min-height: 100vh;
        font-family: 'Segoe UI', Trebuchet MS, sans-serif;
    }
    
    .stApp {
        background: transparent !important;
    }
    
    .main {
        background: transparent !important;
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 20px rgba(139, 92, 246, 0.5); }
        50% { text-shadow: 0 0 40px rgba(139, 92, 246, 0.8); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .header-title {
        font-size: 3em;
        font-weight: 900;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        letter-spacing: -1px;
        animation: glow 3s ease-in-out infinite;
    }
    
    .header-subtitle {
        font-size: 1.1em;
        color: #cbd5e1;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        backdrop-filter: blur(10px);
        animation: slideIn 0.6s ease-out;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 800;
        color: #8b5cf6;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9em;
    }
    
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        margin: 5px;
    }
    
    .loser-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        margin: 5px;
    }
    
    .neutral-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        margin: 5px;
    }
    
    .significant-result {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #cbd5e1;
    }
    
    .insignificant-result {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #cbd5e1;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.5) !important;
    }
    
    .section-title {
        font-size: 1.5em;
        font-weight: 800;
        color: #e2e8f0;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(139, 92, 246, 0.3);
    }
    
    .info-box {
        background: rgba(139, 92, 246, 0.05);
        border-left: 4px solid #8b5cf6;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        color: #cbd5e1;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "test_results" not in st.session_state:
    st.session_state.test_results = None
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

# Helper functions
def calculate_statistical_significance(control_conversions, control_total, variant_conversions, variant_total, test_type='chi2'):
    """Calculate p-value and statistical significance"""
    control_rate = control_conversions / control_total
    variant_rate = variant_conversions / variant_total
    
    if test_type == 'chi2':
        # Chi-square test for categorical data
        contingency_table = np.array([
            [control_conversions, control_total - control_conversions],
            [variant_conversions, variant_total - variant_conversions]
        ])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    else:
        # T-test for continuous data
        control_data = np.random.binomial(1, control_rate, control_total)
        variant_data = np.random.binomial(1, variant_rate, variant_total)
        t_stat, p_value = ttest_ind(control_data, variant_data)
    
    return p_value, control_rate, variant_rate

def calculate_lift(control_rate, variant_rate):
    """Calculate percentage lift"""
    return ((variant_rate - control_rate) / control_rate * 100) if control_rate != 0 else 0

def calculate_sample_size(baseline_rate, mde, alpha=0.05, beta=0.20):
    """Calculate required sample size for statistical power"""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(1 - beta)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    
    pooled_p = (p1 + p2) / 2
    n = ((z_alpha + z_beta) ** 2 * (pooled_p * (1 - pooled_p) * 2)) / ((p2 - p1) ** 2)
    return int(np.ceil(n))

# Header
st.markdown("""
<div style="margin-bottom: 20px;">
    <div class="header-title">⚖️ A/B Testing Dashboard</div>
    <div class="header-subtitle">Analyze experiments with statistical significance testing</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Upload & Analyze", "📈 Results", "🔬 Sample Size Calculator", "📋 Guide"])

with tab1:
    st.markdown("### 📤 Upload Test Data")
    st.markdown("<div class='info-box'>Upload a CSV file with columns: group (A/B), conversions, total_visitors. Or use manual input below.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload experiment data (CSV)", type=["csv"], key="ab_file")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        
        with st.expander("👁️ Preview Data"):
            st.dataframe(df, use_container_width=True)
    
    st.markdown("### 📝 Or Enter Data Manually")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Control (A)**")
        control_conversions = st.number_input("Conversions (Control)", min_value=0, value=100, key="ctrl_conv")
        control_total = st.number_input("Total Visitors (Control)", min_value=1, value=1000, key="ctrl_total")
    
    with col2:
        st.markdown("**Variant (B)**")
        variant_conversions = st.number_input("Conversions (Variant)", min_value=0, value=120, key="var_conv")
        variant_total = st.number_input("Total Visitors (Variant)", min_value=1, value=1000, key="var_total")
    
    # Significance level
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01, help="Standard is 0.05 (95% confidence)")
    
    with col2:
        test_type = st.selectbox("Statistical Test", ["Chi-Square Test", "T-Test"], help="Chi-Square for proportions, T-Test for means")
    
    if st.button("🔍 Analyze A/B Test", use_container_width=True, key="analyze_ab"):
        test_type_param = 'chi2' if test_type == 'Chi-Square Test' else 't'
        p_value, control_rate, variant_rate = calculate_statistical_significance(
            control_conversions, control_total, variant_conversions, variant_total, test_type_param
        )
        
        lift = calculate_lift(control_rate, variant_rate)
        is_significant = p_value < alpha
        
        st.session_state.test_results = {
            'control_conversions': control_conversions,
            'control_total': control_total,
            'variant_conversions': variant_conversions,
            'variant_total': variant_total,
            'control_rate': control_rate,
            'variant_rate': variant_rate,
            'p_value': p_value,
            'lift': lift,
            'is_significant': is_significant,
            'alpha': alpha,
            'test_type': test_type
        }
        
        st.success("✅ Analysis complete!")
        st.rerun()

with tab2:
    if st.session_state.test_results:
        results = st.session_state.test_results
        
        st.markdown("### 📊 Test Results")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Control Conversion Rate</div>
                <div class="metric-value">{results['control_rate']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Variant Conversion Rate</div>
                <div class="metric-value">{results['variant_rate']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Lift</div>
                <div class="metric-value">{results['lift']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">P-Value</div>
                <div class="metric-value">{results['p_value']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistical significance
        st.markdown("### 🎯 Statistical Significance")
        
        if results['is_significant']:
            st.markdown(f"""
            <div class="significant-result">
                <strong>✅ STATISTICALLY SIGNIFICANT</strong><br>
                The difference between Control and Variant is statistically significant at α={results['alpha']}.
                P-value ({results['p_value']:.4f}) < α ({results['alpha']})<br><br>
                We have sufficient evidence to reject the null hypothesis.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insignificant-result">
                <strong>❌ NOT STATISTICALLY SIGNIFICANT</strong><br>
                The difference between Control and Variant is NOT statistically significant at α={results['alpha']}.
                P-value ({results['p_value']:.4f}) ≥ α ({results['alpha']})<br><br>
                We cannot reject the null hypothesis. The difference may be due to random chance.
            </div>
            """, unsafe_allow_html=True)
        
        # Winner declaration
        st.markdown("### 🏆 Winner Declaration")
        
        if results['is_significant']:
            if results['lift'] > 0:
                st.markdown(f'<div class="winner-badge">🎉 VARIANT WINS!</div>', unsafe_allow_html=True)
                st.markdown(f"Variant B outperforms Control A by **{results['lift']:.2f}%** with statistical significance.")
            else:
                st.markdown(f'<div class="winner-badge">🎉 CONTROL WINS!</div>', unsafe_allow_html=True)
                st.markdown(f"Control A outperforms Variant B by **{abs(results['lift']):.2f}%** with statistical significance.")
        else:
            st.markdown(f'<div class="neutral-badge">➡️ NO CLEAR WINNER</div>', unsafe_allow_html=True)
            st.markdown(f"The test is inconclusive. Continue collecting data or run the experiment longer.")
        
        # Visualization
        st.markdown("### 📈 Conversion Rate Comparison")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        groups = ['Control (A)', 'Variant (B)']
        rates = [results['control_rate']*100, results['variant_rate']*100]
        colors = ['#6366f1', '#8b5cf6']
        
        bars = ax1.bar(groups, rates, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Conversion Rate (%)', fontsize=12)
        ax1.set_title('Conversion Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(rates) * 1.2)
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Samples sizes
        sizes = [results['control_total'], results['variant_total']]
        ax2.bar(groups, sizes, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax2.set_ylabel('Sample Size', fontsize=12)
        ax2.set_title('Sample Size Distribution', fontsize=14, fontweight='bold')
        
        for i, (bar, size) in enumerate(zip(ax2.patches, sizes)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        fig.patch.set_facecolor('transparent')
        st.pyplot(fig)
        
        # Detailed statistics
        st.markdown("### 📋 Detailed Statistics")
        
        stats_data = {
            'Metric': ['Conversions', 'Total Visitors', 'Conversion Rate', 'Sample Proportion'],
            'Control (A)': [
                f"{results['control_conversions']:,}",
                f"{results['control_total']:,}",
                f"{results['control_rate']*100:.4f}%",
                f"{results['control_rate']:.4f}"
            ],
            'Variant (B)': [
                f"{results['variant_conversions']:,}",
                f"{results['variant_total']:,}",
                f"{results['variant_rate']*100:.4f}%",
                f"{results['variant_rate']:.4f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Confidence intervals
        st.markdown("### 📊 Confidence Intervals (95%)")
        
        z_score = stats.norm.ppf(0.975)  # 95% CI
        
        control_ci = (
            results['control_rate'] - z_score * np.sqrt(results['control_rate'] * (1 - results['control_rate']) / results['control_total']),
            results['control_rate'] + z_score * np.sqrt(results['control_rate'] * (1 - results['control_rate']) / results['control_total'])
        )
        
        variant_ci = (
            results['variant_rate'] - z_score * np.sqrt(results['variant_rate'] * (1 - results['variant_rate']) / results['variant_total']),
            results['variant_rate'] + z_score * np.sqrt(results['variant_rate'] * (1 - results['variant_rate']) / results['variant_total'])
        )
        
        ci_data = {
            'Group': ['Control (A)', 'Variant (B)'],
            'Lower Bound': [f"{control_ci[0]*100:.2f}%", f"{variant_ci[0]*100:.2f}%"],
            'Point Estimate': [f"{results['control_rate']*100:.2f}%", f"{results['variant_rate']*100:.2f}%"],
            'Upper Bound': [f"{control_ci[1]*100:.2f}%", f"{variant_ci[1]*100:.2f}%"]
        }
        
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df, use_container_width=True, hide_index=True)
    else:
        st.info("👈 Run an analysis first to see results")

with tab3:
    st.markdown("### 🔬 Sample Size Calculator")
    st.markdown("<div class='info-box'>Calculate the required sample size for your A/B test to achieve statistical power.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        baseline_rate = st.number_input("Baseline Conversion Rate (%)", min_value=0.1, value=5.0, step=0.1) / 100
    
    with col2:
        mde = st.number_input("Minimum Detectable Effect (%)", min_value=1, value=20, step=1) / 100
    
    with col3:
        power = st.number_input("Desired Power (1-β)", min_value=0.70, value=0.80, step=0.05)
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)
    
    with col2:
        tails = st.selectbox("Test Type", ["Two-Tailed (Standard)", "One-Tailed"])
    
    if st.button("📐 Calculate Sample Size", use_container_width=True, key="calc_sample"):
        beta = 1 - power
        alpha_adjusted = alpha if tails == "Two-Tailed (Standard)" else alpha * 2
        
        n = calculate_sample_size(baseline_rate, mde, alpha_adjusted, beta)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Per Group</div>
                <div class="metric-value">{n:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Needed</div>
                <div class="metric-value">{n*2:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated MDE</div>
                <div class="metric-value">{mde*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Sample Size Recommendation:</strong><br>
            You need <strong>{n:,} visitors per group</strong> (total {n*2:,}) to detect a {mde*100:.1f}% difference
            with {power*100:.0f}% statistical power and {(1-alpha)*100:.0f}% confidence level.
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### 📖 A/B Testing Guide")
    
    st.markdown("""
    #### What is A/B Testing?
    A/B testing is a method of comparing two versions of a webpage or app to determine which one performs better.
    
    #### Key Concepts:
    
    **Control (A):** The original/baseline version  
    **Variant (B):** The new version with changes  
    **Conversion:** A desired user action (purchase, signup, click, etc.)  
    **Conversion Rate:** (Conversions / Total Visitors) × 100%  
    **Lift:** Percentage improvement of Variant over Control  
    
    #### Statistical Significance:
    
    - **P-Value:** Probability that results occurred by random chance
    - **Alpha (α):** Significance level (typically 0.05 = 95% confidence)
    - **Significant Result:** p-value < α means the result is statistically significant
    - **Power (1-β):** Probability of detecting a real effect (typically 80%)
    
    #### Interpretation:
    
    **✅ Significant & Positive Lift:** Variant B wins - implement it!  
    **❌ Significant & Negative Lift:** Control A wins - stick with original  
    **➡️ Not Significant:** Results are inconclusive - collect more data  
    
    #### Best Practices:
    
    1. **Determine sample size** before running the test
    2. **Run for sufficient duration** to capture user behavior patterns
    3. **Randomize users** into Control and Variant groups
    4. **Track one primary metric** to avoid false positives
    5. **Don't peek** at results until test is complete
    6. **Document everything** for reproducibility
    
    #### Common Mistakes:
    
    - ❌ Stopping test too early (underpowered)
    - ❌ Peeking at results repeatedly (inflates false positives)
    - ❌ Running too many simultaneous tests (multiple comparison problem)
    - ❌ Not controlling for external factors
    - ❌ Ignoring practical significance vs statistical significance
    """)

st.markdown("""
<div class="footer">
    <p>🚀 A/B Testing Dashboard | Powered by SciPy & Streamlit</p>
    <p style="font-size: 0.9em; color: #64748b; margin-top: 10px;">
        💡 Use this tool to analyze experiments, calculate required sample sizes, and make data-driven decisions.
    </p>
</div>
""", unsafe_allow_html=True)