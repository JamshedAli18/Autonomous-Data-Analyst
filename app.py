import streamlit as st
import pandas as pd
from PIL import Image
import os
from backend import DataAnalyzer

# Page configuration
st.set_page_config(
    page_title="Autonomous Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme professional styling
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background-color: #0e1117;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(31, 110, 251, 0.3);
    }
    
    .main-header h1 {
        font-size: 32px;
        margin-bottom: 10px;
        font-weight: 600;
        color: #ffffff;
    }
    
    .main-header p {
        font-size: 16px;
        opacity: 0.95;
        color: #e6edf3;
    }
    
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #c9d1d9;
        margin: 30px 0 20px 0;
        border-bottom: 3px solid #1f6feb;
        padding-bottom: 10px;
    }
    
    .info-box {
        background-color: #161b22;
        padding: 20px;
        border-left: 4px solid #1f6feb;
        border-radius: 6px;
        margin: 15px 0;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    .insight-box {
        background-color: #0d3d56;
        padding: 25px;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        color: #79c0ff;
        margin: 15px 0;
        border: 1px solid #1f6feb;
        line-height: 1.8;
    }
    
    .insight-box ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .insight-box li {
        margin: 12px 0;
        padding-left: 20px;
        position: relative;
    }
    
    .insight-box li:before {
        content: "•";
        position: absolute;
        left: 0;
        color: #58a6ff;
        font-weight: bold;
        font-size: 18px;
    }
    
    .chart-container {
        background-color: #161b22;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        border: 1px solid #30363d;
    }
    
    .chart-title {
        font-size: 18px;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 2px solid #30363d;
    }
    
    .chart-explanation {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 15px;
        font-style: italic;
        line-height: 1.6;
        padding: 12px;
        background-color: #0d1117;
        border-left: 3px solid #58a6ff;
        border-radius: 4px;
    }
    
    .upload-section {
        background-color: #161b22;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid #30363d;
    }
    
    .metrics-row {
        display: flex;
        gap: 20px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .metric-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        flex: 1;
        min-width: 200px;
        border: 1px solid #30363d;
    }
    
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #58a6ff;
    }
    
    footer {
        text-align: center;
        padding: 20px;
        color: #8b949e;
        font-size: 12px;
        margin-top: 50px;
        border-top: 1px solid #30363d;
    }
    
    .model-info {
        background-color: #161b22;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border-left: 3px solid #58a6ff;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    .model-info strong {
        color: #79c0ff;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Session state initialization
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
    <div class="main-header">
        <h1>Autonomous Data Analyst</h1>
        <p>Professional AI-powered data analysis. Upload CSV, get insights instantly.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### Settings & Information")
    
    st.markdown("**Status**: Ready")
    
    st.markdown("---")
    
    st.markdown("**AI Models Used**")
    st.markdown("""
    <div class="model-info">
    <strong>Planner</strong><br>
    llama-3.3-70b-versatile
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-info">
    <strong>Code Generator</strong><br>
    openai/gpt-oss-120b
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-info">
    <strong>Explainer</strong><br>
    groq/compound-mini
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("**How to Use**")
    st.markdown("""
    1. Upload your CSV file
    2. Click Analyze Data
    3. View results on Home tab
    4. Check code on Code tab
    5. Explore charts on Charts tab
    """)
    
    st.markdown("---")
    
    st.markdown("**About**")
    st.markdown("""
    Version: 2.3
    
    Autonomous Data Analyst uses AI to automatically analyze CSV files and generate insights, code, and visualizations.
    """)

# ============================================
# MAIN CONTENT - TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["Home", "Generated Code", "Smart Charts"])

# ============================================
# TAB 1: HOME PAGE
# ============================================
with tab1:
    st.markdown("### Upload and Analyze Data")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select CSV file", type=["csv"], key="csv_uploader")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show file preview if uploaded
    if uploaded_file is not None:
        # Save temp file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.markdown("---")
        st.markdown("### Data Preview")
        
        df = pd.read_csv(temp_path)
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{df.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">File Name</div>
                <div class="metric-value" style="font-size: 14px;">{uploaded_file.name}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Table
        st.markdown("**Dataset Preview (First 10 rows)**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data Info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Column Information**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Missing Values**")
            missing = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum().values,
                'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("Analyze Data", use_container_width=True, type="primary")
        
        # Analysis Process
        if analyze_button:
            with st.spinner("Analyzing your data..."):
                try:
                    analyzer = DataAnalyzer(temp_path)
                    result = analyzer.analyze()
                    
                    st.session_state.analysis_result = result
                    st.success("Analysis complete! Check other tabs for results.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                finally:
                    # Clean temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        # Display Results
        if st.session_state.analysis_result:
            st.markdown("---")
            st.markdown(f"<h3 class='section-header'>Key Insights</h3>", unsafe_allow_html=True)
            
            result = st.session_state.analysis_result
            
            # Format insights as proper bullet points
            insights_text = result["insights"]
            insights_lines = [line.strip() for line in insights_text.split('\n') if line.strip()]
            
            # Build HTML for bullet points
            insights_html = '<div class="insight-box"><ul>'
            for insight in insights_lines:
                insight_clean = insight.lstrip('- ').strip()
                if insight_clean:
                    insights_html += f'<li>{insight_clean}</li>'
            insights_html += '</ul></div>'
            
            st.markdown(insights_html, unsafe_allow_html=True)
            
            # Summary Metrics
            st.markdown("---")
            st.markdown("**Dataset Summary**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Shape", result["data_shape"])
            
            with col2:
                st.metric("Charts Generated", result["charts_generated"])
            
            with col3:
                st.metric("Columns Analyzed", len(result["columns"]))

# ============================================
# TAB 2: GENERATED CODE
# ============================================
with tab2:
    if st.session_state.analysis_result:
        st.markdown(f"<h3 class='section-header'>Generated Analysis Code</h3>", unsafe_allow_html=True)
        
        result = st.session_state.analysis_result
        
        st.markdown("**Python Code for Data Analysis**")
        st.markdown("This code was automatically generated by the AI Code Generator model.")
        
        st.code(result["generated_code"], language="python")
        
        st.markdown("---")
        
        st.markdown("**How to Use This Code**")
        st.markdown("""
        1. Install required packages: `pip install pandas numpy matplotlib seaborn`
        2. Load your CSV file into the script
        3. Run the analysis functions
        4. Generate visualizations
        5. Extract insights from the results
        """)
    
    else:
        st.info("Upload and analyze a CSV file first to see the generated code.")

# ============================================
# TAB 3: SMART CHARTS
# ============================================
with tab3:
    if st.session_state.analysis_result:
        st.markdown(f"<h3 class='section-header'>Smart Generated Charts</h3>", unsafe_allow_html=True)
        
        result = st.session_state.analysis_result
        
        st.markdown(f"**Total Charts Generated: {result['charts_generated']}**")
        st.markdown("The system intelligently selected the most relevant charts based on your data characteristics.")
        
        st.markdown("---")
        
        # Display Charts
        if result['charts_generated'] > 0:
            for idx, (chart_file, title, explanation) in enumerate(
                zip(result['chart_files'], result['chart_titles'], result['chart_explanations']), 1
            ):
                if os.path.exists(chart_file):
                    st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="chart-title">Chart {idx}: {title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chart-explanation">{explanation}</div>', unsafe_allow_html=True)
                    
                    image = Image.open(chart_file)
                    st.image(image, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.warning("No charts were generated for this dataset.")
    
    else:
        st.info("Upload and analyze a CSV file first to see the generated charts.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<footer>
    Autonomous Data Analyst v2.3 | Professional Data Analysis Platform
</footer>
""", unsafe_allow_html=True)
