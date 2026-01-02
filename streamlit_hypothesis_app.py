# Professional Hypothesis Testing Platform - Cloud Compatible Version
# Simplified for Streamlit Cloud deployment

import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, levene, normaltest, jarque_bera, 
                        wilcoxon, mannwhitneyu, kruskal, friedmanchisquare,
                        fisher_exact, chi2_contingency)
try:
    import statsmodels.api as sm
    from statsmodels.stats.power import ttest_power
    from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    
import ast
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Professional Hypothesis Testing Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
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
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3498db;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
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
if 'group_selection' not in st.session_state:
    st.session_state['group_selection'] = None
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
    
    st.markdown(f"""
    <div style="background-color: #e9ecef; border-radius: 10px; padding: 3px; margin: 1rem 0;">
        <div style="background: linear-gradient(90deg, #28a745 0%, #20c997 100%); 
                    width: {progress*100}%; height: 20px; border-radius: 8px; 
                    display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
            {int(progress*100)}% Complete
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_data_summary_card(data):
    """Create a comprehensive data summary card"""
    if data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", data.shape[0])
        with col2:
            st.metric("📈 Columns", data.shape[1])
        with col3:
            st.metric("🔢 Numeric Cols", data.select_dtypes(include=[np.number]).shape[1])
        with col4:
            st.metric("🅰️ Text Cols", data.select_dtypes(include=['object']).shape[1])

def create_matplotlib_plots(data, column):
    """Create distribution plots using matplotlib as fallback"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Distribution Analysis: {column}', fontsize=16)
    
    values = data[column].dropna()
    
    # Histogram
    axes[0, 0].hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Histogram')
    axes[0, 0].set_xlabel('Values')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(values)
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Values')
    
    # Q-Q plot
    stats.probplot(values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Density plot
    axes[1, 1].hist(values, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Density Plot')
    axes[1, 1].set_xlabel('Values')
    axes[1, 1].set_ylabel('Density')
    
    plt.tight_layout()
    return fig

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
        if len(values) >= 3:
            # Shapiro-Wilk test
            try:
                stat, p = shapiro(values)
                col_results['shapiro'] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
            except:
                col_results['shapiro'] = None
                
            # Jarque-Bera test  
            try:
                stat, p = jarque_bera(values)
                col_results['jarque_bera'] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
            except:
                col_results['jarque_bera'] = None
        
        # Outlier detection using IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
        col_results['outliers'] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(values) * 100,
            'values': outliers.tolist() if len(outliers) < 10 else f"{len(outliers)} outliers detected"
        }
        
        results[col] = col_results
    
    return results

def recommend_test(data_type, n_groups, is_paired, is_parametric):
    """Smart test recommendation system"""
    recommendations = {
        'continuous': {
            1: {
                'parametric': {'paired': None, 'unpaired': 'One-Sample t-test'},
                'non_parametric': {'paired': None, 'unpaired': 'Wilcoxon Signed-Rank test'}
            },
            2: {
                'parametric': {
                    'paired': 'Paired t-test', 
                    'unpaired': 'Independent t-test'
                },
                'non_parametric': {
                    'paired': 'Wilcoxon Signed-Rank test',
                    'unpaired': 'Mann-Whitney U test'
                }
            },
            'multiple': {
                'parametric': {
                    'paired': 'Repeated Measures ANOVA',
                    'unpaired': 'One-Way ANOVA'
                },
                'non_parametric': {
                    'paired': 'Friedman test',
                    'unpaired': 'Kruskal-Wallis test'
                }
            }
        },
        'discrete': {
            1: {'test': 'Binomial test'},
            2: {
                'paired': 'McNemar test',
                'unpaired': 'Fisher\'s Exact test / Chi-Square test'
            },
            'multiple': {
                'paired': 'Cochran\'s Q test',
                'unpaired': 'Chi-Square test'
            }
        }
    }
    
    if data_type == 'continuous':
        group_key = n_groups if n_groups <= 2 else 'multiple'
        param_key = 'parametric' if is_parametric else 'non_parametric'
        
        if n_groups == 1:
            return recommendations[data_type][group_key][param_key]['unpaired']
        else:
            pair_key = 'paired' if is_paired else 'unpaired'
            return recommendations[data_type][group_key][param_key][pair_key]
    
    elif data_type == 'discrete':
        if n_groups == 1:
            return recommendations[data_type][n_groups]['test']
        else:
            group_key = n_groups if n_groups <= 2 else 'multiple'
            pair_key = 'paired' if is_paired else 'unpaired'
            return recommendations[data_type][group_key][pair_key]
    
    return "No recommendation available"

# --- Sidebar Configuration ---
with st.sidebar:
    # Application Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #2c3e50; font-family: 'Inter', sans-serif;">📊 Hypothesis Testing Platform</h2>
        <p style="color: #7f8c8d; font-size: 0.9rem;">Professional Statistical Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Progress Tracker
    st.markdown("### 📈 Progress Tracker")
    create_progress_indicator()
    
    for step, completed in st.session_state['step_completed'].items():
        status_icon = "✅" if completed else "⏳"
        st.write(f"{status_icon} {step}")
    
    st.markdown("---")
    
    # Statistical Tests Reference
    with st.expander("📚 Statistical Tests Guide"):
        st.markdown("""
        **Parametric Tests (Normal Distribution Required)**
        - One-Sample t-test
        - Independent t-test  
        - Paired t-test
        - One-Way ANOVA
        - Repeated Measures ANOVA
        
        **Non-Parametric Tests (No Distribution Assumption)**
        - Wilcoxon Signed-Rank test
        - Mann-Whitney U test
        - Kruskal-Wallis test
        - Friedman test
        
        **Discrete/Categorical Tests**
        - Binomial test
        - Chi-Square test
        - Fisher's Exact test
        - McNemar test
        """)
    
    # Quick Actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Analysis", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main Application ---
st.markdown('<h1 class="main-header">Professional Hypothesis Testing Platform</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px;">
    <p style="font-size: 1.1rem; color: #495057; margin: 0;">
        🎯 <strong>Smart Statistical Analysis</strong> | 📊 <strong>Automated Test Selection</strong> | 📈 <strong>Professional Visualizations</strong>
    </p>
</div>
""", unsafe_allow_html=True)

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
    
    # Data input method selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h4>📤 Upload CSV File</h4>
            <p>Upload your dataset in CSV format for automatic processing and analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Supported formats: CSV"
        )
        
        if uploaded_file:
            try:
                st.session_state['data'] = pd.read_csv(uploaded_file)
                st.session_state['step_completed']['Data Input'] = True
                st.success("✅ File uploaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <h4>✏️ Manual Data Entry</h4>
            <p>Enter your data manually for quick analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        manual_input_type = st.selectbox(
            "Select input method",
            ["Single Column", "Multiple Columns"]
        )
        
        if manual_input_type == "Single Column":
            values = st.text_area(
                "Enter values (comma separated)",
                placeholder="1.2, 2.3, 3.4, 4.5, 5.6",
                help="Enter numeric values separated by commas"
            )
            
            if values:
                try:
                    data_list = [float(x.strip()) for x in values.split(',')]
                    st.session_state['data'] = pd.DataFrame({'Values': data_list})
                    st.session_state['step_completed']['Data Input'] = True
                    st.success("✅ Data entered successfully!")
                except:
                    st.error("❌ Invalid format. Please use comma-separated numbers.")
        
        elif manual_input_type == "Multiple Columns":
            st.markdown("**Enter data for multiple groups:**")
            n_groups = st.number_input("Number of groups", min_value=2, max_value=5, value=2)
            
            group_data = {}
            for i in range(n_groups):
                values = st.text_area(
                    f"Group {i+1} values",
                    placeholder="1.2, 2.3, 3.4",
                    key=f"group_{i}"
                )
                if values:
                    try:
                        group_data[f'Group_{i+1}'] = [float(x.strip()) for x in values.split(',')]
                    except:
                        st.error(f"❌ Invalid format in Group {i+1}")
            
            if len(group_data) >= 2:
                # Pad with NaN to make equal length
                max_len = max(len(v) for v in group_data.values())
                for k, v in group_data.items():
                    group_data[k] = v + [np.nan] * (max_len - len(v))
                
                st.session_state['data'] = pd.DataFrame(group_data)
                st.session_state['step_completed']['Data Input'] = True
                st.success("✅ Multiple groups data entered successfully!")
    
    # Display data preview if available
    if st.session_state.get('data') is not None:
        st.markdown("---")
        st.markdown("### 📊 Data Preview")
        
        # Data summary metrics
        create_data_summary_card(st.session_state['data'])
        
        # Data preview table
        st.dataframe(
            st.session_state['data'], 
            use_container_width=True,
            height=300
        )
        
        # Basic data info
        with st.expander("🔍 Detailed Data Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Types**")
                st.dataframe(pd.DataFrame({
                    'Column': st.session_state['data'].columns,
                    'Type': st.session_state['data'].dtypes,
                    'Non-Null': st.session_state['data'].count()
                }))
            
            with col2:
                if st.session_state['data'].select_dtypes(include=[np.number]).shape[1] > 0:
                    st.markdown("**Descriptive Statistics**")
                    st.dataframe(st.session_state['data'].describe())

# --- Tab 2: Data Analysis ---
with tab2:
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('data') is not None:
        data = st.session_state['data']
        
        # Data type classification
        st.markdown("### 🏷️ Data Type Classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_type_auto = st.selectbox(
                "Select data type for analysis",
                ["Continuous", "Discrete/Categorical"],
                help="This will determine which statistical tests are appropriate"
            )
            
            st.session_state['data_type'] = data_type_auto.lower()
        
        with col2:
            # Group configuration
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                n_groups = len(numeric_cols)
                st.metric("🧑‍🤝‍🧑 Number of Groups Detected", n_groups)
                
                if n_groups >= 2:
                    is_paired = st.radio(
                        "Are the groups paired/related?",
                        ["Independent", "Paired/Related"],
                        help="Paired: Same subjects measured multiple times. Independent: Different subjects"
                    )
                    st.session_state['paired'] = (is_paired == "Paired/Related")
        
        # Visualization section
        if data_type_auto == "Continuous" and len(numeric_cols) > 0:
            st.markdown("---")
            st.markdown("### 📈 Data Visualization")
            
            selected_column = st.selectbox("Select column for detailed analysis", numeric_cols)
            
            # Choose plotting library based on availability
            if PLOTLY_AVAILABLE:
                # Create plotly visualization (simplified)
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=data[selected_column].dropna(), name="Distribution"))
                fig.update_layout(title=f"Distribution: {selected_column}", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to matplotlib
                fig = create_matplotlib_plots(data, selected_column)
                st.pyplot(fig)
                plt.close()
            
            # Additional visualizations for multiple columns
            if len(numeric_cols) > 1:
                st.markdown("#### 📊 Group Comparison")
                
                if PLOTLY_AVAILABLE:
                    fig_box = go.Figure()
                    for col in numeric_cols:
                        fig_box.add_trace(go.Box(
                            y=data[col].dropna(),
                            name=col,
                            boxpoints='outliers'
                        ))
                    
                    fig_box.update_layout(
                        title="Box Plot Comparison",
                        yaxis_title="Values",
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    # Matplotlib box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data[numeric_cols].boxplot(ax=ax)
                    ax.set_title('Box Plot Comparison')
                    st.pyplot(fig)
                    plt.close()
        
        # Mark step as completed
        st.session_state['step_completed']['Data Analysis'] = True
        
    else:
        st.warning("⚠️ Please upload or enter data in the **Data Input** tab first.")

# --- Tab 3: Assumption Checking ---
with tab3:
    st.markdown('<h2 class="section-header">🔍 Statistical Assumptions Testing</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('data') is not None and st.session_state.get('data_type') == 'continuous':
        data = st.session_state['data']
        
        st.markdown("""
        <div class="professional-card">
            <h4>🎯 Why Check Assumptions?</h4>
            <p>Statistical assumptions ensure the validity of your test results. Violating assumptions can lead to incorrect conclusions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Perform comprehensive assumption testing
        with st.spinner("🔍 Running assumption tests..."):
            assumption_results = perform_assumption_tests(data)
        
        if assumption_results:
            # Create assumption testing dashboard
            st.markdown("### 📊 Assumption Test Results")
            
            for col, results in assumption_results.items():
                with st.expander(f"📈 Analysis for {col}"):
                    col1, col2, col3 = st.columns(3)
                    
                    # Normality Tests
                    with col1:
                        st.markdown("#### 🔔 Normality Tests")
                        
                        if results.get('shapiro'):
                            sw_result = results['shapiro']
                            status = "✅ Normal" if sw_result['normal'] else "❌ Non-normal"
                            st.metric(
                                "Shapiro-Wilk Test",
                                f"p = {sw_result['p_value']:.4f}",
                                delta=status
                            )
                        
                        if results.get('jarque_bera'):
                            jb_result = results['jarque_bera']
                            status = "✅ Normal" if jb_result['normal'] else "❌ Non-normal"
                            st.metric(
                                "Jarque-Bera Test",
                                f"p = {jb_result['p_value']:.4f}",
                                delta=status
                            )
                    
                    # Outlier Analysis
                    with col2:
                        st.markdown("#### 🎯 Outlier Detection")
                        outlier_info = results['outliers']
                        st.metric(
                            "Outlier Count",
                            outlier_info['count'],
                            delta=f"{outlier_info['percentage']:.1f}%"
                        )
                        
                        if outlier_info['count'] > 0:
                            st.warning(f"⚠️ {outlier_info['count']} outliers detected")
                    
                    # Recommendations
                    with col3:
                        st.markdown("#### 💡 Recommendations")
                        
                        # Determine if parametric tests are appropriate
                        is_normal = (results.get('shapiro', {}).get('normal', False) and 
                                   results.get('jarque_bera', {}).get('normal', False))
                        has_outliers = results['outliers']['count'] > 0
                        
                        if is_normal and not has_outliers:
                            st.success("✅ **Parametric tests recommended**")
                            st.session_state['parametric'] = True
                        elif is_normal and has_outliers:
                            st.warning("⚠️ **Consider outlier treatment**")
                            st.session_state['parametric'] = False
                        else:
                            st.info("🔄 **Non-parametric tests recommended**")
                            st.session_state['parametric'] = False
            
            # Overall recommendation
            st.markdown("---")
            st.markdown("### 🎯 Overall Recommendation")
            
            if st.session_state.get('parametric'):
                st.success("✅ **Parametric Analysis Recommended** - Assumptions are satisfied")
            else:
                st.info("🔄 **Non-Parametric Analysis Recommended** - Some assumptions are violated")
            
        # Mark step as completed
        st.session_state['step_completed']['Assumption Checking'] = True
        
    elif st.session_state.get('data') is not None and st.session_state.get('data_type') == 'discrete':
        st.info("📋 **Discrete/Categorical data** does not require assumption testing. Proceeding to test selection.")
        st.session_state['step_completed']['Assumption Checking'] = True
        
    else:
        st.warning("⚠️ Please complete the **Data Analysis** tab first.")

# --- Tab 4: Test Selection ---
with tab4:
    st.markdown('<h2 class="section-header">🧪 Statistical Test Selection</h2>', unsafe_allow_html=True)
    
    if (st.session_state.get('data') is not None and 
        st.session_state.get('data_type') is not None):
        
        data = st.session_state['data']
        data_type = st.session_state['data_type']
        
        # Automatic test recommendation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        n_groups = len(numeric_cols)
        is_paired = st.session_state.get('paired', False)
        is_parametric = st.session_state.get('parametric', False)
        
        recommended_test = recommend_test(data_type, n_groups, is_paired, is_parametric)
        
        # Display recommendation
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
            st.markdown("**Test Parameters**")
            
            alpha = st.selectbox(
                "Significance level (α)",
                [0.05, 0.01, 0.10],
                index=0,
                help="Probability of Type I error"
            )
            
            alternative = st.selectbox(
                "Alternative hypothesis",
                ["two-sided", "less", "greater"],
                help="Direction of the alternative hypothesis"
            )
        
        # Store test selection
        st.session_state['selected_test'] = recommended_test
        st.session_state['alpha'] = alpha
        st.session_state['alternative'] = alternative
        st.session_state['step_completed']['Test Selection'] = True
        
    else:
        st.warning("⚠️ Please complete the previous steps first.")

# --- Tab 5: Results & Interpretation ---  
with tab5:
    st.markdown('<h2 class="section-header">📈 Statistical Analysis Results</h2>', unsafe_allow_html=True)
    
    if (st.session_state.get('data') is not None and 
        st.session_state.get('selected_test') is not None):
        
        data = st.session_state['data']
        test_name = st.session_state['selected_test']
        alpha = st.session_state.get('alpha', 0.05)
        alternative = st.session_state.get('alternative', 'two-sided')
        
        st.markdown(f"### 🧪 Running: {test_name}")
        
        try:
            # Initialize results
            statistic = None
            p_value = None
            effect_size = None
            
            # Run the appropriate test
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if "t-test" in test_name.lower():
                if "one-sample" in test_name.lower():
                    # One-sample t-test
                    test_value = st.number_input("Population mean to test against", value=0.0)
                    statistic, p_value = stats.ttest_1samp(data[numeric_cols[0]].dropna(), test_value)
                    effect_size = (data[numeric_cols[0]].mean() - test_value) / data[numeric_cols[0]].std()
                    
                elif "independent" in test_name.lower():
                    # Independent t-test
                    statistic, p_value = stats.ttest_ind(
                        data[numeric_cols[0]].dropna(), 
                        data[numeric_cols[1]].dropna()
                    )
                    
                elif "paired" in test_name.lower():
                    # Paired t-test
                    statistic, p_value = stats.ttest_rel(
                        data[numeric_cols[0]].dropna(), 
                        data[numeric_cols[1]].dropna()
                    )
            
            elif "anova" in test_name.lower():
                if "one-way" in test_name.lower():
                    # One-way ANOVA
                    groups = [data[col].dropna() for col in numeric_cols]
                    statistic, p_value = stats.f_oneway(*groups)
                    
            elif "wilcoxon" in test_name.lower():
                if len(numeric_cols) == 1:
                    # One-sample Wilcoxon
                    test_value = st.number_input("Median to test against", value=0.0)
                    statistic, p_value = stats.wilcoxon(
                        data[numeric_cols[0]].dropna() - test_value
                    )
                else:
                    # Paired Wilcoxon
                    statistic, p_value = stats.wilcoxon(
                        data[numeric_cols[0]].dropna(),
                        data[numeric_cols[1]].dropna()
                    )
            
            elif "mann-whitney" in test_name.lower():
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(
                    data[numeric_cols[0]].dropna(),
                    data[numeric_cols[1]].dropna(),
                    alternative=alternative
                )
                
            elif "kruskal" in test_name.lower():
                # Kruskal-Wallis test
                groups = [data[col].dropna() for col in numeric_cols]
                statistic, p_value = stats.kruskal(*groups)
                
            elif "friedman" in test_name.lower():
                # Friedman test
                groups = [data[col].dropna() for col in numeric_cols]
                statistic, p_value = stats.friedmanchisquare(*groups)
            
            # Display results
            if statistic is not None and p_value is not None:
                
                # Results metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Statistic", f"{statistic:.4f}")
                
                with col2:
                    st.metric("P-value", f"{p_value:.6f}")
                
                with col3:
                    significance = "Significant" if p_value < alpha else "Not Significant"
                    st.metric("Result", significance)
                
                # Results visualization
                if PLOTLY_AVAILABLE:
                    # Create beautiful bar chart for p-value vs alpha
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['P-value', 'Alpha (α)'], 
                            y=[p_value, alpha],
                            marker_color=['green' if p_value < alpha else 'red', 'blue'],
                            text=[f'{p_value:.6f}', f'{alpha:.3f}'],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title=f"Statistical Test Results: {test_name}",
                        yaxis_title="Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
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
                
                # Effect size and practical significance
                if effect_size is not None:
                    st.markdown("### 📏 Effect Size")
                    
                    effect_interpretation = ""
                    if abs(effect_size) < 0.2:
                        effect_interpretation = "Small effect"
                    elif abs(effect_size) < 0.5:
                        effect_interpretation = "Medium effect"
                    else:
                        effect_interpretation = "Large effect"
                    
                    st.metric("Cohen's d", f"{effect_size:.3f}", delta=effect_interpretation)
                
                # Mark as completed
                st.session_state['step_completed']['Statistical Testing'] = True
                
        except Exception as e:
            st.error(f"❌ Error running statistical test: {str(e)}")
            st.info("Please check your data and test selection.")
    
    else:
        st.warning("⚠️ Please complete all previous steps first.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;">
    <h4>📊 Professional Hypothesis Testing Platform</h4>
    <p style="color: #6c757d;">Built with Streamlit • Powered by SciPy & Plotly</p>
    <p style="color: #6c757d; font-size: 0.9rem;">
        🔬 Advanced Statistical Analysis • 📈 Interactive Visualizations • 🎯 Automated Test Selection
    </p>
</div>
""", unsafe_allow_html=True)
