# Professional Hypothesis Testing Platform - Minimal Version
# Maximum compatibility with Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, levene, normaltest, jarque_bera, 
                        wilcoxon, mannwhitneyu, kruskal, friedmanchisquare,
                        fisher_exact, chi2_contingency)
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Professional Hypothesis Testing Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .test-result-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(40,167,69,0.2);
    }
    
    .test-result-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220,53,69,0.2);
    }
    
    .professional-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'step_completed' not in st.session_state:
    st.session_state['step_completed'] = {
        'Data Input': False,
        'Data Analysis': False,
        'Assumption Checking': False,
        'Test Selection': False,
        'Statistical Testing': False
    }

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'data_type' not in st.session_state:
    st.session_state['data_type'] = None
if 'parametric' not in st.session_state:
    st.session_state['parametric'] = False
if 'paired' not in st.session_state:
    st.session_state['paired'] = None

# --- Helper Functions ---
def create_progress_indicator():
    """Create a visual progress indicator"""
    completed_steps = sum(st.session_state['step_completed'].values())
    total_steps = len(st.session_state['step_completed'])
    progress = completed_steps / total_steps
    
    st.progress(progress)
    st.write(f"Progress: {int(progress*100)}% Complete")

def perform_assumption_tests(data):
    """Comprehensive assumption testing"""
    results = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        values = data[col].dropna()
        if len(values) < 3:
            continue
            
        col_results = {}
        
        # Normality tests
        try:
            stat, p = shapiro(values)
            col_results['shapiro'] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
        except:
            col_results['shapiro'] = None
        
        # Outlier detection using IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
        col_results['outliers'] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(values) * 100
        }
        
        results[col] = col_results
    
    return results

def recommend_test(data_type, n_groups, is_paired, is_parametric):
    """Smart test recommendation system"""
    if data_type == 'continuous':
        if n_groups == 1:
            return 'One-Sample t-test' if is_parametric else 'Wilcoxon Signed-Rank test'
        elif n_groups == 2:
            if is_paired:
                return 'Paired t-test' if is_parametric else 'Wilcoxon Signed-Rank test'
            else:
                return 'Independent t-test' if is_parametric else 'Mann-Whitney U test'
        else:  # n_groups > 2
            if is_paired:
                return 'Repeated Measures ANOVA' if is_parametric else 'Friedman test'
            else:
                return 'One-Way ANOVA' if is_parametric else 'Kruskal-Wallis test'
    
    return "No recommendation available"

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("## 📊 Hypothesis Testing Platform")
    st.markdown("Professional Statistical Analysis Tool")
    st.markdown("---")
    
    # Progress Tracker
    st.markdown("### 📈 Progress Tracker")
    create_progress_indicator()
    
    for step, completed in st.session_state['step_completed'].items():
        status_icon = "✅" if completed else "⏳"
        st.write(f"{status_icon} {step}")
    
    st.markdown("---")
    
    # Quick Actions
    if st.button("🔄 Reset Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main Application ---
st.markdown('<h1 class="main-header">Professional Hypothesis Testing Platform</h1>', unsafe_allow_html=True)

# Create tabs for the main workflow
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Input", 
    "📊 Data Analysis", 
    "🔍 Assumption Checking", 
    "🧪 Test Selection", 
    "📈 Results & Interpretation"
])

# --- Tab 1: Data Input ---
with tab1:
    st.markdown('<h2 class="section-header">📂 Data Input & Upload</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                st.session_state['data'] = pd.read_csv(uploaded_file)
                st.session_state['step_completed']['Data Input'] = True
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.markdown("### ✏️ Manual Data Entry")
        
        manual_input_type = st.selectbox("Select input method", ["Single Column", "Multiple Columns"])
        
        if manual_input_type == "Single Column":
            values = st.text_area("Enter values (comma separated)", placeholder="1.2, 2.3, 3.4, 4.5, 5.6")
            
            if values:
                try:
                    data_list = [float(x.strip()) for x in values.split(',')]
                    st.session_state['data'] = pd.DataFrame({'Values': data_list})
                    st.session_state['step_completed']['Data Input'] = True
                    st.success("✅ Data entered successfully!")
                except:
                    st.error("❌ Invalid format. Please use comma-separated numbers.")
        
        elif manual_input_type == "Multiple Columns":
            n_groups = st.number_input("Number of groups", min_value=2, max_value=5, value=2)
            
            group_data = {}
            for i in range(n_groups):
                values = st.text_area(f"Group {i+1} values", placeholder="1.2, 2.3, 3.4", key=f"group_{i}")
                if values:
                    try:
                        group_data[f'Group_{i+1}'] = [float(x.strip()) for x in values.split(',')]
                    except:
                        st.error(f"❌ Invalid format in Group {i+1}")
            
            if len(group_data) >= 2:
                max_len = max(len(v) for v in group_data.values())
                for k, v in group_data.items():
                    group_data[k] = v + [np.nan] * (max_len - len(v))
                
                st.session_state['data'] = pd.DataFrame(group_data)
                st.session_state['step_completed']['Data Input'] = True
                st.success("✅ Multiple groups data entered successfully!")
    
    # Display data preview
    if st.session_state.get('data') is not None:
        st.markdown("---")
        st.markdown("### 📊 Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", st.session_state['data'].shape[0])
        with col2:
            st.metric("📈 Columns", st.session_state['data'].shape[1])
        with col3:
            st.metric("🔢 Numeric Cols", st.session_state['data'].select_dtypes(include=[np.number]).shape[1])
        with col4:
            st.metric("🅰️ Text Cols", st.session_state['data'].select_dtypes(include=['object']).shape[1])
        
        st.dataframe(st.session_state['data'], use_container_width=True)

# --- Tab 2: Data Analysis ---
with tab2:
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('data') is not None:
        data = st.session_state['data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_type_auto = st.selectbox("Select data type for analysis", ["Continuous", "Discrete/Categorical"])
            st.session_state['data_type'] = data_type_auto.lower()
        
        with col2:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                n_groups = len(numeric_cols)
                st.metric("🧑‍🤝‍🧑 Number of Groups Detected", n_groups)
                
                if n_groups >= 2:
                    is_paired = st.radio("Are the groups paired/related?", ["Independent", "Paired/Related"])
                    st.session_state['paired'] = (is_paired == "Paired/Related")
        
        # Visualization
        if data_type_auto == "Continuous" and len(numeric_cols) > 0:
            st.markdown("---")
            st.markdown("### 📈 Data Visualization")
            
            selected_column = st.selectbox("Select column for detailed analysis", numeric_cols)
            
            # Create matplotlib plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Distribution Analysis: {selected_column}', fontsize=16)
            
            values = data[selected_column].dropna()
            
            # Histogram
            axes[0, 0].hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Histogram')
            
            # Box plot
            axes[0, 1].boxplot(values)
            axes[0, 1].set_title('Box Plot')
            
            # Q-Q plot
            stats.probplot(values, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            
            # Density plot
            axes[1, 1].hist(values, bins=30, density=True, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('Density Plot')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.session_state['step_completed']['Data Analysis'] = True
        
    else:
        st.warning("⚠️ Please upload or enter data in the **Data Input** tab first.")

# --- Tab 3: Assumption Checking ---
with tab3:
    st.markdown('<h2 class="section-header">🔍 Statistical Assumptions Testing</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('data') is not None and st.session_state.get('data_type') == 'continuous':
        data = st.session_state['data']
        
        with st.spinner("🔍 Running assumption tests..."):
            assumption_results = perform_assumption_tests(data)
        
        if assumption_results:
            st.markdown("### 📊 Assumption Test Results")
            
            for col, results in assumption_results.items():
                with st.expander(f"📈 Analysis for {col}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 🔔 Normality Tests")
                        if results.get('shapiro'):
                            sw_result = results['shapiro']
                            status = "✅ Normal" if sw_result['normal'] else "❌ Non-normal"
                            st.metric("Shapiro-Wilk Test", f"p = {sw_result['p_value']:.4f}", delta=status)
                    
                    with col2:
                        st.markdown("#### 🎯 Outlier Detection")
                        outlier_info = results['outliers']
                        st.metric("Outlier Count", outlier_info['count'], delta=f"{outlier_info['percentage']:.1f}%")
                    
                    with col3:
                        st.markdown("#### 💡 Recommendations")
                        is_normal = results.get('shapiro', {}).get('normal', False)
                        has_outliers = results['outliers']['count'] > 0
                        
                        if is_normal and not has_outliers:
                            st.success("✅ **Parametric tests recommended**")
                            st.session_state['parametric'] = True
                        else:
                            st.info("🔄 **Non-parametric tests recommended**")
                            st.session_state['parametric'] = False
        
        st.session_state['step_completed']['Assumption Checking'] = True
        
    elif st.session_state.get('data') is not None:
        st.info("📋 **Discrete/Categorical data** does not require assumption testing.")
        st.session_state['step_completed']['Assumption Checking'] = True
    else:
        st.warning("⚠️ Please complete the **Data Analysis** tab first.")

# --- Tab 4: Test Selection ---
with tab4:
    st.markdown('<h2 class="section-header">🧪 Statistical Test Selection</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('data') is not None and st.session_state.get('data_type') is not None:
        data = st.session_state['data']
        data_type = st.session_state['data_type']
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        n_groups = len(numeric_cols)
        is_paired = st.session_state.get('paired', False)
        is_parametric = st.session_state.get('parametric', False)
        
        recommended_test = recommend_test(data_type, n_groups, is_paired, is_parametric)
        
        st.markdown("### 🎯 Recommended Statistical Test")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="test-result-positive">
                <h4>📊 {recommended_test}</h4>
                <p><strong>Based on your data characteristics:</strong></p>
                <ul>
                    <li>Data type: {data_type.title()}</li>
                    <li>Number of groups: {n_groups}</li>
                    <li>Paired/Independent: {"Paired" if is_paired else "Independent"}</li>
                    <li>Parametric assumptions: {"Met" if is_parametric else "Not met"}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            alpha = st.selectbox("Significance level (α)", [0.05, 0.01, 0.10], index=0)
            alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"])
        
        st.session_state['selected_test'] = recommended_test
        st.session_state['alpha'] = alpha
        st.session_state['alternative'] = alternative
        st.session_state['step_completed']['Test Selection'] = True
        
    else:
        st.warning("⚠️ Please complete the previous steps first.")

# --- Tab 5: Results & Interpretation ---
with tab5:
    st.markdown('<h2 class="section-header">📈 Statistical Analysis Results</h2>', unsafe_allow_html=True)
    
    if (st.session_state.get('data') is not None and st.session_state.get('selected_test') is not None):
        data = st.session_state['data']
        test_name = st.session_state['selected_test']
        alpha = st.session_state.get('alpha', 0.05)
        alternative = st.session_state.get('alternative', 'two-sided')
        
        st.markdown(f"### 🧪 Running: {test_name}")
        
        try:
            statistic = None
            p_value = None
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Run appropriate test
            if "One-Sample t-test" in test_name:
                test_value = st.number_input("Population mean to test against", value=0.0)
                statistic, p_value = stats.ttest_1samp(data[numeric_cols[0]].dropna(), test_value)
                
            elif "Independent t-test" in test_name:
                statistic, p_value = stats.ttest_ind(data[numeric_cols[0]].dropna(), data[numeric_cols[1]].dropna())
                
            elif "Paired t-test" in test_name:
                statistic, p_value = stats.ttest_rel(data[numeric_cols[0]].dropna(), data[numeric_cols[1]].dropna())
                
            elif "One-Way ANOVA" in test_name:
                groups = [data[col].dropna() for col in numeric_cols]
                statistic, p_value = stats.f_oneway(*groups)
                
            elif "Wilcoxon" in test_name:
                if len(numeric_cols) == 1:
                    test_value = st.number_input("Median to test against", value=0.0)
                    statistic, p_value = stats.wilcoxon(data[numeric_cols[0]].dropna() - test_value)
                else:
                    statistic, p_value = stats.wilcoxon(data[numeric_cols[0]].dropna(), data[numeric_cols[1]].dropna())
                    
            elif "Mann-Whitney" in test_name:
                statistic, p_value = stats.mannwhitneyu(data[numeric_cols[0]].dropna(), data[numeric_cols[1]].dropna(), alternative=alternative)
                
            elif "Kruskal-Wallis" in test_name:
                groups = [data[col].dropna() for col in numeric_cols]
                statistic, p_value = stats.kruskal(*groups)
                
            elif "Friedman" in test_name:
                groups = [data[col].dropna() for col in numeric_cols]
                statistic, p_value = stats.friedmanchisquare(*groups)
            
            # Display results
            if statistic is not None and p_value is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Statistic", f"{statistic:.4f}")
                with col2:
                    st.metric("P-value", f"{p_value:.6f}")
                with col3:
                    significance = "Significant" if p_value < alpha else "Not Significant"
                    st.metric("Result", significance)
                
                # Create simple bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(['P-value', 'Alpha (α)'], [p_value, alpha], 
                             color=['green' if p_value < alpha else 'red', 'blue'])
                ax.set_title(f"Statistical Test Results: {test_name}")
                ax.set_ylabel("Value")
                
                # Add value labels on bars
                for bar, value in zip(bars, [p_value, alpha]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.6f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                plt.close()
                
                # Interpretation
                st.markdown("---")
                st.markdown("### 🎯 Statistical Interpretation")
                
                if p_value < alpha:
                    st.markdown(f"""
                    <div class="test-result-positive">
                        <h4>✅ Statistically Significant Result</h4>
                        <p><strong>Conclusion:</strong> Reject the null hypothesis (H₀)</p>
                        <p><strong>Interpretation:</strong> There is sufficient evidence to support the alternative hypothesis at α = {alpha} level.</p>
                        <p><strong>P-value ({p_value:.6f}) < α ({alpha})</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="test-result-negative">
                        <h4>❌ Not Statistically Significant</h4>
                        <p><strong>Conclusion:</strong> Fail to reject the null hypothesis (H₀)</p>
                        <p><strong>Interpretation:</strong> There is insufficient evidence to support the alternative hypothesis at α = {alpha} level.</p>
                        <p><strong>P-value ({p_value:.6f}) ≥ α ({alpha})</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state['step_completed']['Statistical Testing'] = True
                
        except Exception as e:
            st.error(f"❌ Error running statistical test: {str(e)}")
            
    else:
        st.warning("⚠️ Please complete all previous steps first.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;">
    <h4>📊 Professional Hypothesis Testing Platform</h4>
    <p style="color: #6c757d;">Built with Streamlit • Powered by SciPy & Matplotlib</p>
    <p style="color: #6c757d; font-size: 0.9rem;">
        🔬 Advanced Statistical Analysis • 📈 Interactive Visualizations • 🎯 Automated Test Selection
    </p>
</div>
""", unsafe_allow_html=True)
