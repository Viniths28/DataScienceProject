import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="DataAnalytics Pro - Advanced Data Analytics & Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
}
.sub-header {
    font-size: 1.8rem;
    color: #4a5568;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: 600;
}
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.analytics-container {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.insight-container {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.info-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 1rem 0;
    border-left: 5px solid #ff6b6b;
}
.success-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 1rem 0;
    border-left: 5px solid #00d4ff;
}
.feature-card {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    border: none;
}
.stDataFrame {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Beautiful color palettes
COLOR_PALETTES = {
    'sunset': ['#ff6b6b', '#ee5a24', '#ff9ff3', '#54a0ff', '#5f27cd'],
    'ocean': ['#0abde3', '#006ba6', '#74b9ff', '#00d2d3', '#55efc4'],
    'forest': ['#00b894', '#00cec9', '#6c5ce7', '#a29bfe', '#fd79a8'],
    'cosmic': ['#e84393', '#fd79a8', '#fdcb6e', '#e17055', '#74b9ff'],
    'tropical': ['#ff7675', '#fd79a8', '#fdcb6e', '#55efc4', '#74b9ff'],
    'rainbow': ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#1dd1a1']
}

def main():
    st.markdown('<h1 class="main-header">📊 DataAnalytics Pro - Advanced Data Analytics & Visualization</h1>', unsafe_allow_html=True)
    st.markdown("### 🔍 Transform your data into powerful insights and stunning visual stories!")
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation Hub")
    page = st.sidebar.selectbox(
        "Choose your analytics journey:",
        ["🏠 Home", "📊 Data Explorer", "📈 Data Analytics", "🎨 Beautiful Visualizations", "📋 Insights & Reports", "📋 Sample Datasets"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "📈 Data Analytics":
        show_data_analytics()
    elif page == "🎨 Beautiful Visualizations":
        show_visualizations()
    elif page == "📋 Insights & Reports":
        show_insights_reports()
    elif page == "📋 Sample Datasets":
        show_sample_data()

def show_home():
    st.markdown("## 🚀 Welcome to Your Complete Data Analytics Journey!")
    
    # Create beautiful feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Explorer</h3>
            <p>Upload CSV/Excel files and get instant comprehensive data profiling with quality metrics and statistical insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Data Analytics</h3>
            <p>Advanced statistical analysis, trend detection, correlation studies, and data profiling for deep insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Visualizations</h3>
            <p>Create stunning charts with rainbow colors, gradients, and professional styling with multiple themes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📋 Smart Insights</h3>
            <p>Automated data insights, trend analysis, and intelligent recommendations based on your data patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Analytics Reports</h3>
            <p>Generate comprehensive analytics reports with statistical summaries and data quality assessments.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Sample Data</h3>
            <p>Try our beautifully crafted sample datasets to explore all analytics and visualization features immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Perfect For Data Analytics:")
        st.markdown("""
        - **📚 College Analytics Projects** 
        - **📊 Business Intelligence Reports**
        - **📈 Data Science Presentations** 
        - **🔍 Exploratory Data Analysis**
        - **📋 Statistical Analysis Studies**
        - **🎭 Data Storytelling**
        """)
    
    with col2:
        st.markdown("### ✨ Advanced Features:")
        st.markdown("""
        - **🔍 Automated Data Profiling**
        - **📊 Statistical Analysis Suite**
        - **🎨 Multiple Color Palettes**
        - **📈 Trend & Pattern Detection**
        - **⚡ Interactive Dashboards**
        - **📋 Smart Insights Generation**
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Ready to Analyze Your Data?</h3>
        <p>👈 Use the sidebar to start your analytics journey. Begin with <strong>Data Explorer</strong> for data profiling, then explore <strong>Data Analytics</strong> for insights, and <strong>Visualizations</strong> for stunning charts!</p>
    </div>
    """, unsafe_allow_html=True)

def show_data_explorer():
    st.markdown("## 📊 Data Explorer - Discover Your Data's Beauty")
    
    # File upload with beautiful styling
    st.markdown("""
    <div class="info-box">
        <h3>📁 Upload Your Data</h3>
        <p>Drag and drop your CSV or Excel files below to begin the magic!</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file to explore your data"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.df = df
            
            st.markdown("""
            <div class="success-box">
                <h3>✅ Data Loaded Successfully!</h3>
                <p>Your data is ready for visualization magic!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Beautiful metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h2>{df.shape[0]}</h2>
                    <p>Total Rows</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h2>{df.shape[1]}</h2>
                    <p>Total Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing_values = df.isnull().sum().sum()
                st.markdown(f"""
                <div class="metric-container">
                    <h2>{missing_values}</h2>
                    <p>Missing Values</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.markdown(f"""
                <div class="metric-container">
                    <h2>{numeric_cols}</h2>
                    <p>Numeric Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview with beautiful styling
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Colorful data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Column Information")
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(info_df, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Statistical Summary")
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    st.dataframe(df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for statistical summary")
            
            # Beautiful missing values analysis
            if missing_values > 0:
                st.markdown("### 🚨 Missing Values Analysis")
                missing_data = df.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    color=missing_data.values,
                    color_continuous_scale='Reds',
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    title_font_size=16
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>👆 Upload Your Data Above!</h3>
            <p>Or try our beautiful sample datasets from the sidebar menu!</p>
        </div>
        """, unsafe_allow_html=True)

def show_visualizations():
    st.markdown("## 🎨 Beautiful Data Visualizations")
    
    if 'df' not in st.session_state:
        st.markdown("""
        <div class="info-box">
            <h3>⚠️ No Data Found!</h3>
            <p>Please upload a dataset first in the Data Explorer section or try our Sample Datasets!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Color palette selector
    st.sidebar.markdown("### 🎨 Choose Your Color Palette")
    selected_palette = st.sidebar.selectbox(
        "Select a color theme:",
        options=list(COLOR_PALETTES.keys()),
        format_func=lambda x: f"🎨 {x.title()}"
    )
    
    colors = COLOR_PALETTES[selected_palette]
    
    # Visualization options
    viz_type = st.selectbox(
        "🎯 Choose your visualization magic:",
        ["🔗 Correlation Heatmap", "📊 Distribution Gallery", "📦 Box Plot Showcase", 
         "🎯 Scatter Plot Matrix", "📈 Count Plot Rainbow", "🌈 Multi-Chart Dashboard"]
    )
    
    if viz_type == "🔗 Correlation Heatmap":
        st.markdown("### 🔗 Beautiful Correlation Heatmap")
        st.markdown("*Discover hidden relationships in your data with stunning colors*")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="🔥 Correlation Matrix - Data Relationships Revealed",
                color_continuous_scale='RdYlBu',
                labels=dict(color="Correlation")
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=20,
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>💡 Reading the Heatmap</h4>
                <p>🔴 Red = Strong Negative Correlation | 🟡 Yellow = Weak Correlation | 🔵 Blue = Strong Positive Correlation</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🤔 Need at least 2 numerical columns for correlation magic!")
    
    elif viz_type == "📊 Distribution Gallery":
        st.markdown("### 📊 Beautiful Distribution Gallery")
        st.markdown("*Explore how your data is distributed with colorful histograms*")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("🎯 Select your column:", numeric_cols)
            
            # Create beautiful histogram
            fig = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                title=f"🌈 Distribution of {selected_col}",
                color_discrete_sequence=[colors[0]],
                marginal="box"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=18,
                title_x=0.5
            )
            fig.update_traces(marker_line_width=2, marker_line_color="white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Beautiful stats display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{df[selected_col].mean():.2f}</h3>
                    <p>Mean</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{df[selected_col].median():.2f}</h3>
                    <p>Median</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{df[selected_col].std():.2f}</h3>
                    <p>Std Dev</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{df[selected_col].skew():.2f}</h3>
                    <p>Skewness</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("🤔 No numerical columns found for distribution magic!")
    
    elif viz_type == "📦 Box Plot Showcase":
        st.markdown("### 📦 Stunning Box Plot Showcase")
        st.markdown("*Spot outliers and quartiles with beautiful box plots*")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("🎯 Select your column:", numeric_cols)
            
            fig = px.box(
                df,
                y=selected_col,
                title=f"📦 Box Plot Magic - {selected_col}",
                color_discrete_sequence=[colors[1]]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=18,
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>💡 Box Plot Guide</h4>
                <p>📊 The box shows the middle 50% of your data | 🔴 Dots are outliers | 📏 Lines show the data range</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🤔 No numerical columns found for box plot magic!")
    
    elif viz_type == "🎯 Scatter Plot Matrix":
        st.markdown("### 🎯 Scatter Plot Matrix Adventure")
        st.markdown("*Explore relationships between variables with colorful scatter plots*")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("🎯 Select X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("🎯 Select Y-axis:", numeric_cols)
            
            # Add color grouping option
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                color_col = st.selectbox("🎨 Color by category (optional):", 
                                       ['None'] + list(categorical_cols))
            else:
                color_col = 'None'
            
            if color_col != 'None':
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"🎯 {x_col} vs {y_col} (Colored by {color_col})",
                    color_discrete_sequence=colors
                )
            else:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"🎯 {x_col} vs {y_col}",
                    color_discrete_sequence=[colors[2]]
                )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=18,
                title_x=0.5
            )
            fig.update_traces(marker_size=8, marker_line_width=1, marker_line_color="white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            corr = df[x_col].corr(df[y_col])
            st.markdown(f"""
            <div class="metric-container">
                <h2>{corr:.3f}</h2>
                <p>Correlation Coefficient</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🤔 Need at least 2 numerical columns for scatter plot magic!")
    
    elif viz_type == "📈 Count Plot Rainbow":
        st.markdown("### 📈 Rainbow Count Plot Spectacular")
        st.markdown("*Beautiful bar charts with rainbow colors for categorical data*")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            selected_col = st.selectbox("🎯 Select your category:", categorical_cols)
            
            # Limit to top categories for better visualization
            top_categories = df[selected_col].value_counts().head(15)
            
            fig = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title=f"🌈 Rainbow Count Plot - {selected_col}",
                labels={'x': selected_col, 'y': 'Count'},
                color=top_categories.values,
                color_continuous_scale='Rainbow'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=18,
                title_x=0.5,
                xaxis_tickangle=-45
            )
            fig.update_traces(marker_line_width=2, marker_line_color="white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="success-box">
                <h4>📊 Category Insights</h4>
                <p>Showing top 15 categories out of {df[selected_col].nunique()} unique values</p>
                <p>Most common: <strong>{top_categories.index[0]}</strong> ({top_categories.values[0]} occurrences)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🤔 No categorical columns found for count plot magic!")
    
    elif viz_type == "🌈 Multi-Chart Dashboard":
        st.markdown("### 🌈 Multi-Chart Dashboard Spectacular")
        st.markdown("*Multiple beautiful charts in one amazing dashboard*")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 2:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('📊 Distribution', '📦 Box Plot', '🔗 Correlation', '🎯 Scatter Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Distribution plot
            fig.add_trace(
                go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0], 
                           marker_color=colors[0], opacity=0.7),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=df[numeric_cols[1]], name=numeric_cols[1], 
                      marker_color=colors[1]),
                row=1, col=2
            )
            
            # Correlation heatmap (simplified)
            if len(numeric_cols) >= 3:
                corr_subset = df[numeric_cols[:3]].corr()
                fig.add_trace(
                    go.Heatmap(z=corr_subset.values, 
                              x=corr_subset.columns, 
                              y=corr_subset.columns,
                              colorscale='RdYlBu'),
                    row=2, col=1
                )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]], 
                          mode='markers', name='Data Points',
                          marker=dict(color=colors[3], size=8)),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="🌈 Multi-Chart Dashboard - Your Data in All Its Glory",
                title_x=0.5,
                title_font_size=20,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="success-box">
                <h4>🎉 Dashboard Complete!</h4>
                <p>Your data is now displayed in multiple beautiful visualizations! Each chart tells a different story about your data.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🤔 Need at least 2 numerical columns for dashboard magic!")

def show_sample_data():
    st.markdown("## 📋 Beautiful Sample Datasets")
    st.markdown("*Try these gorgeous datasets to see the magic in action!*")
    
    # Create sample datasets
    sample_data = {
        "🏠 House Prices": create_house_price_data(),
        "🎓 Student Performance": create_student_data(),
        "🛒 Sales Analytics": create_sales_data()
    }
    
    # Beautiful dataset selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🏠 House Prices</h3>
            <p>Real estate data with prices, locations, and features. Perfect for exploring correlations!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎓 Student Performance</h3>
            <p>Academic data with grades, study hours, and demographics. Great for educational insights!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🛒 Sales Analytics</h3>
            <p>Business data with sales, products, and regions. Ideal for business visualizations!</p>
        </div>
        """, unsafe_allow_html=True)
    
    selected_dataset = st.selectbox("🎯 Choose your dataset adventure:", list(sample_data.keys()))
    
    if st.button("🚀 Load Sample Dataset", type="primary"):
        st.session_state.df = sample_data[selected_dataset]
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ {selected_dataset} Loaded Successfully!</h3>
            <p>Your sample data is ready for visualization magic! Go to the Beautiful Visualizations section to explore.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show preview
        st.markdown("### 👀 Data Preview")
        st.dataframe(sample_data[selected_dataset].head(10), use_container_width=True)
        
        # Quick stats
        df_sample = sample_data[selected_dataset]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h2>{df_sample.shape[0]}</h2>
                <p>Rows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h2>{df_sample.shape[1]}</h2>
                <p>Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = len(df_sample.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div class="metric-container">
                <h2>{numeric_cols}</h2>
                <p>Numeric</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            categorical_cols = len(df_sample.select_dtypes(include=['object']).columns)
            st.markdown(f"""
            <div class="metric-container">
                <h2>{categorical_cols}</h2>
                <p>Categorical</p>
            </div>
            """, unsafe_allow_html=True)

def create_house_price_data():
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic house data
    size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    property_type = np.random.choice(['Apartment', 'House', 'Condo', 'Villa'], n_samples)
    
    # Create price based on features with some noise
    price = (size * 100 + bedrooms * 15000 + bathrooms * 8000 - age * 1000 + 
             np.where(location == 'Urban', 50000, 
                     np.where(location == 'Suburban', 20000, 0)) + 
             np.random.normal(0, 20000, n_samples))
    
    return pd.DataFrame({
        'Property_ID': [f'PROP_{i+1000:04d}' for i in range(n_samples)],
        'Size_sqft': size.astype(int),
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_years': age,
        'Location': location,
        'Property_Type': property_type,
        'Price': price.astype(int),
        'Price_per_sqft': (price / size).round(2)
    })

def create_student_data():
    np.random.seed(42)
    n_samples = 150
    
    study_hours = np.random.normal(5, 2, n_samples)
    attendance = np.random.normal(85, 10, n_samples)
    previous_grade = np.random.normal(75, 15, n_samples)
    extracurricular = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    major = np.random.choice(['Engineering', 'Business', 'Arts', 'Science'], n_samples)
    
    # Create final grade based on features
    final_grade = (study_hours * 8 + attendance * 0.3 + previous_grade * 0.4 + 
                   np.where(extracurricular == 'Yes', 5, 0) +
                   np.random.normal(0, 5, n_samples))
    
    # Create pass/fail based on grade
    pass_fail = np.where(final_grade >= 70, 'Pass', 'Fail')
    
    return pd.DataFrame({
        'Student_ID': [f'STU_{i+3000:04d}' for i in range(n_samples)],
        'Study_Hours': np.clip(study_hours, 0, 12).round(1),
        'Attendance_Percent': np.clip(attendance, 0, 100).round(1),
        'Previous_Grade': np.clip(previous_grade, 0, 100).round(1),
        'Final_Grade': np.clip(final_grade, 0, 100).round(1),
        'Major': major,
        'Extracurricular': extracurricular,
        'Pass_Fail': pass_fail,
        'GPA': (np.clip(final_grade, 0, 100) / 25).round(2)
    })

def create_sales_data():
    np.random.seed(42)
    n_samples = 300
    
    # Generate sales data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch']
    regions = ['North', 'South', 'East', 'West', 'Central']
    sales_channels = ['Online', 'Retail', 'Wholesale']
    
    data = []
    for i in range(n_samples):
        month = np.random.choice(months)
        product = np.random.choice(products)
        region = np.random.choice(regions)
        channel = np.random.choice(sales_channels)
        
        # Base sales with seasonal and product effects
        base_sales = np.random.normal(1000, 200)
        if month in ['Nov', 'Dec']:
            base_sales *= 1.5  # Holiday boost
        if product == 'Laptop':
            base_sales *= 1.2
        
        sales = max(int(base_sales), 0)
        price = np.random.normal(300, 50) if product != 'Laptop' else np.random.normal(800, 100)
        price = max(price, 100)
        
        data.append({
            'Sale_ID': f'SALE_{i+4000:04d}',
            'Month': month,
            'Product': product,
            'Region': region,
            'Sales_Channel': channel,
            'Units_Sold': sales,
            'Unit_Price': round(price, 2),
            'Total_Revenue': round(sales * price, 2),
            'Profit_Margin': round(np.random.normal(0.25, 0.05), 2)
        })
    
    return pd.DataFrame(data)

def show_data_analytics():
    st.markdown("## 📈 Advanced Data Analytics Suite")
    
    if 'df' not in st.session_state:
        st.markdown("""
        <div class="info-box">
            <h3>⚠️ No Data Found!</h3>
            <p>Please upload a dataset first in the Data Explorer section or try our Sample Datasets!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Analytics navigation
    analytics_tab = st.selectbox(
        "🔍 Choose your analytics approach:",
        ["📊 Data Profiling", "📈 Statistical Analysis", "🔍 Correlation Analysis", "📋 Data Quality Assessment", "🎯 Trend Analysis"]
    )
    
    if analytics_tab == "📊 Data Profiling":
        show_data_profiling(df)
    elif analytics_tab == "📈 Statistical Analysis":
        show_statistical_analysis(df)
    elif analytics_tab == "🔍 Correlation Analysis":
        show_correlation_analysis(df)
    elif analytics_tab == "📋 Data Quality Assessment":
        show_data_quality(df)
    elif analytics_tab == "🎯 Trend Analysis":
        show_trend_analysis(df)

def show_data_profiling(df):
    st.markdown("### 📊 Comprehensive Data Profiling")
    st.markdown("*Get a complete overview of your dataset structure and characteristics*")
    
    # Basic dataset info
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="analytics-container">
            <h2>{df.shape[0]:,}</h2>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="analytics-container">
            <h2>{df.shape[1]}</h2>
            <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f"""
        <div class="analytics-container">
            <h2>{memory_usage:.1f} MB</h2>
            <p>Memory Usage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="analytics-container">
            <h2>{numeric_cols}</h2>
            <p>Numeric Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.markdown(f"""
        <div class="analytics-container">
            <h2>{categorical_cols}</h2>
            <p>Text Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Column-wise analysis
    st.markdown("### 📋 Column-wise Data Profile")
    
    profile_data = []
    for col in df.columns:
        col_data = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null Percentage': f"{(df[col].isnull().sum() / len(df) * 100):.1f}%",
            'Unique Values': df[col].nunique(),
            'Uniqueness': f"{(df[col].nunique() / len(df) * 100):.1f}%"
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_data.update({
                'Mean': f"{df[col].mean():.2f}" if pd.notnull(df[col].mean()) else "N/A",
                'Std Dev': f"{df[col].std():.2f}" if pd.notnull(df[col].std()) else "N/A",
                'Min': f"{df[col].min():.2f}" if pd.notnull(df[col].min()) else "N/A",
                'Max': f"{df[col].max():.2f}" if pd.notnull(df[col].max()) else "N/A"
            })
        else:
            col_data.update({
                'Mean': "N/A",
                'Std Dev': "N/A", 
                'Min': "N/A",
                'Max': "N/A"
            })
        
        profile_data.append(col_data)
    
    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df, use_container_width=True)

def show_statistical_analysis(df):
    st.markdown("### 📈 Statistical Analysis Suite")
    st.markdown("*Comprehensive statistical insights and descriptive analytics*")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("🤔 No numerical columns found for statistical analysis!")
        return
    
    # Overall statistics
    st.markdown("#### 📊 Descriptive Statistics")
    desc_stats = df[numeric_cols].describe()
    st.dataframe(desc_stats, use_container_width=True)
    
    # Advanced statistics
    st.markdown("#### 🎯 Advanced Statistical Measures")
    
    advanced_stats = []
    for col in numeric_cols:
        stats = {
            'Column': col,
            'Skewness': f"{df[col].skew():.3f}",
            'Kurtosis': f"{df[col].kurtosis():.3f}",
            'Coefficient of Variation': f"{(df[col].std() / df[col].mean() * 100):.2f}%" if df[col].mean() != 0 else "N/A",
            'Range': f"{df[col].max() - df[col].min():.2f}",
            'IQR': f"{df[col].quantile(0.75) - df[col].quantile(0.25):.2f}",
            'Outliers (IQR method)': count_outliers_iqr(df[col])
        }
        advanced_stats.append(stats)
    
    advanced_df = pd.DataFrame(advanced_stats)
    st.dataframe(advanced_df, use_container_width=True)
    
    # Statistical interpretations
    st.markdown("#### 💡 Statistical Insights")
    
    interpretations = []
    for col in numeric_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        
        skew_interpretation = "Normally distributed" if abs(skew) < 0.5 else \
                            "Slightly skewed" if abs(skew) < 1 else "Highly skewed"
        
        kurt_interpretation = "Normal distribution" if abs(kurt) < 0.5 else \
                            "Heavy-tailed" if kurt > 0.5 else "Light-tailed"
        
        interpretations.append({
            'Column': col,
            'Distribution Shape': f"{skew_interpretation} ({skew:.2f})",
            'Tail Behavior': f"{kurt_interpretation} ({kurt:.2f})",
            'Data Spread': "High variance" if df[col].std() / df[col].mean() > 0.5 else "Low variance"
        })
    
    insights_df = pd.DataFrame(interpretations)
    st.dataframe(insights_df, use_container_width=True)

def count_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)

def show_correlation_analysis(df):
    st.markdown("### 🔍 Advanced Correlation Analysis")
    st.markdown("*Discover relationships and dependencies between variables*")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("🤔 Need at least 2 numerical columns for correlation analysis!")
        return
    
    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Heatmap visualization
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="🔥 Correlation Matrix - Relationship Strength Map",
        color_continuous_scale='RdYlBu',
        labels=dict(color="Correlation Strength")
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=18,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Strong correlations analysis
    st.markdown("#### 🎯 Strong Correlations Detected")
    
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Only show correlations > 0.5
                strength = "Very Strong" if abs(corr_val) > 0.8 else "Strong"
                direction = "Positive" if corr_val > 0 else "Negative"
                
                strong_correlations.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': f"{corr_val:.3f}",
                    'Strength': strength,
                    'Direction': direction,
                    'Interpretation': get_correlation_interpretation(corr_val)
                })
    
    if strong_correlations:
        corr_df = pd.DataFrame(strong_correlations)
        st.dataframe(corr_df, use_container_width=True)
        
        # Top correlations insights
        st.markdown("""
        <div class="insight-container">
            <h4>🔍 Correlation Insights</h4>
            <p>Strong correlations (>0.5) indicate variables that move together. This can help identify patterns, redundancies, or potential causal relationships in your data.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No strong correlations (>0.5) found between variables.")

def get_correlation_interpretation(corr_val):
    abs_corr = abs(corr_val)
    if abs_corr > 0.8:
        return "Very strong relationship"
    elif abs_corr > 0.6:
        return "Strong relationship"
    elif abs_corr > 0.4:
        return "Moderate relationship"
    elif abs_corr > 0.2:
        return "Weak relationship"
    else:
        return "Very weak relationship"

def show_data_quality(df):
    st.markdown("### 📋 Comprehensive Data Quality Assessment")
    st.markdown("*Evaluate data completeness, consistency, and reliability*")
    
    # Overall data quality score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness_score = ((total_cells - missing_cells) / total_cells) * 100
    
    # Data quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="insight-container">
            <h2>{completeness_score:.1f}%</h2>
            <p>Data Completeness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = ((len(df) - duplicate_rows) / len(df)) * 100
        st.markdown(f"""
        <div class="insight-container">
            <h2>{uniqueness_score:.1f}%</h2>
            <p>Data Uniqueness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = sum([count_outliers_iqr(df[col]) for col in numeric_cols])
        st.markdown(f"""
        <div class="insight-container">
            <h2>{outlier_count}</h2>
            <p>Outliers Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        overall_quality = (completeness_score + uniqueness_score) / 2
        quality_grade = "Excellent" if overall_quality > 90 else \
                       "Good" if overall_quality > 75 else \
                       "Fair" if overall_quality > 60 else "Poor"
        st.markdown(f"""
        <div class="insight-container">
            <h2>{quality_grade}</h2>
            <p>Overall Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed quality assessment
    st.markdown("#### 🔍 Column-wise Quality Assessment")
    
    quality_data = []
    for col in df.columns:
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        unique_pct = (df[col].nunique() / len(df)) * 100
        
        quality_score = 100 - null_pct  # Simple quality score based on completeness
        quality_status = "Excellent" if quality_score > 95 else \
                        "Good" if quality_score > 85 else \
                        "Fair" if quality_score > 70 else "Poor"
        
        quality_data.append({
            'Column': col,
            'Completeness': f"{100 - null_pct:.1f}%",
            'Missing Values': df[col].isnull().sum(),
            'Uniqueness': f"{unique_pct:.1f}%",
            'Quality Score': f"{quality_score:.1f}%",
            'Status': quality_status,
            'Recommendations': get_quality_recommendations(df[col], null_pct, unique_pct)
        })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)

def get_quality_recommendations(series, null_pct, unique_pct):
    recommendations = []
    
    if null_pct > 20:
        recommendations.append("High missing values - consider imputation")
    elif null_pct > 5:
        recommendations.append("Some missing values - review data collection")
    
    if unique_pct < 1:
        recommendations.append("Low variability - check for constant values")
    elif unique_pct > 95 and series.dtype == 'object':
        recommendations.append("High uniqueness - potential identifier column")
    
    return "; ".join(recommendations) if recommendations else "Good quality"

def show_trend_analysis(df):
    st.markdown("### 🎯 Trend and Pattern Analysis")
    st.markdown("*Identify trends, patterns, and anomalies in your data*")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("🤔 No numerical columns found for trend analysis!")
        return
    
    # Time-based analysis if applicable
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        st.markdown("#### 📅 Time Series Analysis")
        
        date_col = st.selectbox("Select date column:", date_cols)
        value_col = st.selectbox("Select value column for trend analysis:", numeric_cols)
        
        # Simple trend visualization
        temp_df = df.dropna(subset=[date_col, value_col])
        temp_df = temp_df.sort_values(date_col)
        
        fig = px.line(temp_df, x=date_col, y=value_col, 
                     title=f"📈 Trend Analysis: {value_col} over time")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pattern detection
    st.markdown("#### 🔍 Pattern Detection")
    
    selected_col = st.selectbox("Select column for pattern analysis:", numeric_cols)
    
    # Basic pattern analysis
    col_data = df[selected_col].dropna()
    
    # Calculate moving averages if enough data
    if len(col_data) > 10:
        rolling_mean = col_data.rolling(window=min(5, len(col_data)//4)).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=col_data.values, mode='lines', name='Original Data', 
                                line=dict(color='lightblue', width=1)))
        fig.add_trace(go.Scatter(y=rolling_mean.values, mode='lines', name='Trend Line',
                                line=dict(color='red', width=3)))
        
        fig.update_layout(
            title=f"📊 Pattern Analysis: {selected_col}",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern insights
        trend_direction = "Increasing" if rolling_mean.iloc[-1] > rolling_mean.iloc[0] else \
                         "Decreasing" if rolling_mean.iloc[-1] < rolling_mean.iloc[0] else "Stable"
        
        volatility = col_data.std() / col_data.mean() * 100 if col_data.mean() != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="insight-container">
                <h3>{trend_direction}</h3>
                <p>Overall Trend</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-container">
                <h3>{volatility:.1f}%</h3>
                <p>Volatility</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            stability = "High" if volatility < 10 else "Medium" if volatility < 30 else "Low"
            st.markdown(f"""
            <div class="insight-container">
                <h3>{stability}</h3>
                <p>Stability</p>
            </div>
            """, unsafe_allow_html=True)

def show_insights_reports():
    st.markdown("## 📋 Smart Insights & Analytics Reports")
    
    if 'df' not in st.session_state:
        st.markdown("""
        <div class="info-box">
            <h3>⚠️ No Data Found!</h3>
            <p>Please upload a dataset first in the Data Explorer section or try our Sample Datasets!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Report type selection
    report_type = st.selectbox(
        "🎯 Choose your report type:",
        ["📊 Executive Summary", "🔍 Detailed Analytics Report", "📈 Data Quality Report", "🎯 Key Insights Dashboard"]
    )
    
    if report_type == "📊 Executive Summary":
        show_executive_summary(df)
    elif report_type == "🔍 Detailed Analytics Report":
        show_detailed_analytics_report(df)
    elif report_type == "📈 Data Quality Report":
        show_data_quality_report(df)
    elif report_type == "🎯 Key Insights Dashboard":
        show_key_insights_dashboard(df)

def show_executive_summary(df):
    st.markdown("### 📊 Executive Summary - Data Overview")
    st.markdown("*High-level insights for decision makers*")
    
    # Key metrics overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{df.shape[0]:,}</h2>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{df.shape[1]}</h2>
            <p>Data Points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        completeness = ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1])) * 100
        st.markdown(f"""
        <div class="metric-container">
            <h2>{completeness:.1f}%</h2>
            <p>Data Completeness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="metric-container">
            <h2>{numeric_cols}</h2>
            <p>Numeric Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.markdown(f"""
        <div class="metric-container">
            <h2>{categorical_cols}</h2>
            <p>Categorical Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key findings
    st.markdown("### 🎯 Key Findings")
    
    findings = generate_key_findings(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Data Characteristics")
        for finding in findings['characteristics']:
            st.markdown(f"• {finding}")
    
    with col2:
        st.markdown("#### 💡 Recommendations")
        for rec in findings['recommendations']:
            st.markdown(f"• {rec}")
    
    # Summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.markdown("### 📊 Quick Statistics Summary")
        
        summary_stats = []
        for col in numeric_cols[:5]:  # Show top 5 numeric columns
            stats = {
                'Variable': col,
                'Average': f"{df[col].mean():.2f}",
                'Median': f"{df[col].median():.2f}",
                'Std Dev': f"{df[col].std():.2f}",
                'Min': f"{df[col].min():.2f}",
                'Max': f"{df[col].max():.2f}"
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        st.dataframe(summary_df, use_container_width=True)

def generate_key_findings(df):
    findings = {
        'characteristics': [],
        'recommendations': []
    }
    
    # Data size assessment
    if df.shape[0] > 10000:
        findings['characteristics'].append("Large dataset with extensive data points")
    elif df.shape[0] > 1000:
        findings['characteristics'].append("Medium-sized dataset suitable for analysis")
    else:
        findings['characteristics'].append("Small dataset - consider collecting more data")
    
    # Missing data assessment
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 20:
        findings['characteristics'].append("High missing data detected")
        findings['recommendations'].append("Consider data cleaning and imputation strategies")
    elif missing_pct > 5:
        findings['characteristics'].append("Some missing values present")
        findings['recommendations'].append("Review data collection processes")
    else:
        findings['characteristics'].append("High data completeness - excellent quality")
    
    # Column distribution
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=['object']).columns)
    
    if numeric_cols > categorical_cols:
        findings['characteristics'].append("Numeric-heavy dataset - suitable for statistical analysis")
        findings['recommendations'].append("Focus on correlation and regression analysis")
    elif categorical_cols > numeric_cols:
        findings['characteristics'].append("Category-rich dataset - ideal for classification")
        findings['recommendations'].append("Consider category analysis and cross-tabulation")
    else:
        findings['characteristics'].append("Balanced mix of numeric and categorical data")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        findings['characteristics'].append(f"{duplicates} duplicate records found")
        findings['recommendations'].append("Consider removing duplicate entries")
    
    return findings

def show_detailed_analytics_report(df):
    st.markdown("### 🔍 Comprehensive Analytics Report")
    st.markdown("*In-depth analysis of your dataset*")
    
    # Generate comprehensive report
    report = generate_analytics_report(df)
    
    # Display report sections
    st.markdown("#### 📋 Dataset Overview")
    st.markdown(report['overview'])
    
    st.markdown("#### 📊 Statistical Summary")
    st.markdown(report['statistical_summary'])
    
    st.markdown("#### 🔍 Data Quality Analysis")
    st.markdown(report['quality_analysis'])
    
    st.markdown("#### 📈 Pattern Analysis")
    st.markdown(report['pattern_analysis'])
    
    st.markdown("#### 💡 Insights & Recommendations")
    st.markdown(report['insights_recommendations'])

def generate_analytics_report(df):
    report = {}
    
    # Overview section
    report['overview'] = f"""
    **Dataset Dimensions**: {df.shape[0]:,} rows × {df.shape[1]} columns
    
    **Data Types Distribution**:
    - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}
    - Text/Categorical columns: {len(df.select_dtypes(include=['object']).columns)}
    - Date columns: {len(df.select_dtypes(include=['datetime64']).columns)}
    
    **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
    """
    
    # Statistical summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['statistical_summary'] = f"""
        **Numeric Variables Analysis**:
        - Variables analyzed: {len(numeric_cols)}
        - Average correlation strength: {abs(df[numeric_cols].corr()).mean().mean():.3f}
        - Highest variance column: {df[numeric_cols].var().idxmax()}
        - Most skewed distribution: {abs(df[numeric_cols].skew()).idxmax()}
        """
    else:
        report['statistical_summary'] = "No numeric variables found for statistical analysis."
    
    # Quality analysis
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    duplicates = df.duplicated().sum()
    
    report['quality_analysis'] = f"""
    **Data Quality Metrics**:
    - Overall completeness: {100 - missing_pct:.1f}%
    - Missing values: {df.isnull().sum().sum():,} cells
    - Duplicate records: {duplicates:,}
    - Data quality grade: {"Excellent" if missing_pct < 5 else "Good" if missing_pct < 15 else "Fair"}
    """
    
    # Pattern analysis
    patterns = []
    for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
        col_data = df[col].dropna()
        if len(col_data) > 1:
            trend = "Increasing" if col_data.iloc[-1] > col_data.iloc[0] else "Decreasing"
            patterns.append(f"- {col}: {trend} trend detected")
    
    report['pattern_analysis'] = f"""
    **Pattern Detection Results**:
    {chr(10).join(patterns) if patterns else "No significant patterns detected in numeric variables."}
    """
    
    # Insights and recommendations
    insights = []
    if missing_pct > 10:
        insights.append("- Consider data imputation strategies for missing values")
    if duplicates > 0:
        insights.append("- Remove duplicate records to improve data quality")
    if len(numeric_cols) > 5:
        insights.append("- Perform dimensionality reduction for high-dimensional data")
    
    report['insights_recommendations'] = f"""
    **Key Recommendations**:
    {chr(10).join(insights) if insights else "- Dataset is in good condition for analysis"}
    
    **Next Steps**:
    - Explore correlations between variables
    - Create visualizations for key insights
    - Consider advanced analytics techniques
    """
    
    return report

def show_data_quality_report(df):
    st.markdown("### 📈 Data Quality Assessment Report")
    st.markdown("*Comprehensive evaluation of data reliability and usability*")
    
    # Quality score calculation
    quality_metrics = calculate_quality_metrics(df)
    
    # Display quality score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-container">
            <h2>{quality_metrics['overall_score']:.1f}%</h2>
            <p>Overall Quality Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-container">
            <h2>{quality_metrics['grade']}</h2>
            <p>Quality Grade</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-container">
            <h2>{quality_metrics['usability']}</h2>
            <p>Data Usability</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed quality breakdown
    st.markdown("#### 📊 Quality Breakdown")
    
    quality_breakdown = pd.DataFrame([
        {'Metric': 'Completeness', 'Score': f"{quality_metrics['completeness']:.1f}%", 'Status': get_quality_status(quality_metrics['completeness'])},
        {'Metric': 'Uniqueness', 'Score': f"{quality_metrics['uniqueness']:.1f}%", 'Status': get_quality_status(quality_metrics['uniqueness'])},
        {'Metric': 'Consistency', 'Score': f"{quality_metrics['consistency']:.1f}%", 'Status': get_quality_status(quality_metrics['consistency'])},
        {'Metric': 'Validity', 'Score': f"{quality_metrics['validity']:.1f}%", 'Status': get_quality_status(quality_metrics['validity'])}
    ])
    
    st.dataframe(quality_breakdown, use_container_width=True)
    
    # Quality recommendations
    st.markdown("#### 💡 Quality Improvement Recommendations")
    
    for rec in quality_metrics['recommendations']:
        st.markdown(f"• {rec}")

def calculate_quality_metrics(df):
    metrics = {}
    
    # Completeness
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    metrics['completeness'] = ((total_cells - missing_cells) / total_cells) * 100
    
    # Uniqueness  
    duplicate_rows = df.duplicated().sum()
    metrics['uniqueness'] = ((len(df) - duplicate_rows) / len(df)) * 100
    
    # Consistency (simplified - based on data type consistency)
    metrics['consistency'] = 90  # Simplified for demo
    
    # Validity (simplified - based on non-null numeric values in numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        valid_numeric = sum([df[col].notna().sum() for col in numeric_cols])
        total_numeric = len(numeric_cols) * len(df)
        metrics['validity'] = (valid_numeric / total_numeric) * 100
    else:
        metrics['validity'] = 100
    
    # Overall score
    metrics['overall_score'] = (metrics['completeness'] + metrics['uniqueness'] + 
                               metrics['consistency'] + metrics['validity']) / 4
    
    # Grade
    if metrics['overall_score'] >= 90:
        metrics['grade'] = "Excellent"
        metrics['usability'] = "Ready for Analysis"
    elif metrics['overall_score'] >= 75:
        metrics['grade'] = "Good"
        metrics['usability'] = "Suitable for Use"
    elif metrics['overall_score'] >= 60:
        metrics['grade'] = "Fair"
        metrics['usability'] = "Needs Improvement"
    else:
        metrics['grade'] = "Poor"
        metrics['usability'] = "Requires Cleaning"
    
    # Recommendations
    recommendations = []
    if metrics['completeness'] < 90:
        recommendations.append("Address missing values through imputation or data collection")
    if metrics['uniqueness'] < 95:
        recommendations.append("Remove duplicate records to improve uniqueness")
    if metrics['overall_score'] < 80:
        recommendations.append("Implement data validation rules for future data collection")
    
    metrics['recommendations'] = recommendations if recommendations else ["Data quality is excellent - no immediate actions needed"]
    
    return metrics

def get_quality_status(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Poor"

def show_key_insights_dashboard(df):
    st.markdown("### 🎯 Key Insights Dashboard")
    st.markdown("*Automated insights and patterns discovered in your data*")
    
    # Generate insights
    insights = generate_automated_insights(df)
    
    # Display insights in cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Data Insights")
        for insight in insights['data_insights']:
            st.markdown(f"""
            <div class="success-box">
                <h4>🔍 {insight['title']}</h4>
                <p>{insight['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Actionable Recommendations")
        for action in insights['actionable_insights']:
            st.markdown(f"""
            <div class="info-box">
                <h4>💡 {action['title']}</h4>
                <p>{action['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistical insights
    if insights['statistical_insights']:
        st.markdown("#### 📈 Statistical Discoveries")
        for stat in insights['statistical_insights']:
            st.markdown(f"• **{stat['metric']}**: {stat['value']} - {stat['interpretation']}")

def generate_automated_insights(df):
    insights = {
        'data_insights': [],
        'actionable_insights': [],
        'statistical_insights': []
    }
    
    # Data insights
    if df.shape[0] > 5000:
        insights['data_insights'].append({
            'title': 'Large Dataset Detected',
            'description': f'With {df.shape[0]:,} records, this dataset provides substantial statistical power for analysis.'
        })
    
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct < 5:
        insights['data_insights'].append({
            'title': 'High Data Quality',
            'description': f'Only {missing_pct:.1f}% missing values detected - excellent data completeness.'
        })
    
    # Actionable insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 3:
        insights['actionable_insights'].append({
            'title': 'Rich Numeric Data',
            'description': f'With {len(numeric_cols)} numeric variables, consider correlation analysis and predictive modeling.'
        })
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 2:
        insights['actionable_insights'].append({
            'title': 'Category Analysis Opportunity',
            'description': f'{len(categorical_cols)} categorical variables available for segmentation analysis.'
        })
    
    # Statistical insights
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        max_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
        max_corr = max_corr[max_corr < 1.0].iloc[0]  # Exclude self-correlation
        
        if max_corr > 0.7:
            insights['statistical_insights'].append({
                'metric': 'Highest Correlation',
                'value': f'{max_corr:.3f}',
                'interpretation': 'Strong relationship detected between variables'
            })
    
    return insights

if __name__ == "__main__":
    main() 