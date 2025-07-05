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
    page_title="DataViz Pro - Data Visualization Dashboard",
    page_icon="🎨",
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
    st.markdown('<h1 class="main-header">🎨 DataViz Pro - Beautiful Data Visualization</h1>', unsafe_allow_html=True)
    st.markdown("### 🌟 Transform your data into stunning visual stories!")
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose your adventure:",
        ["🏠 Home", "📊 Data Explorer", "🎨 Beautiful Visualizations", "📋 Sample Datasets"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🎨 Beautiful Visualizations":
        show_visualizations()
    elif page == "📋 Sample Datasets":
        show_sample_data()

def show_home():
    st.markdown("## 🚀 Welcome to Your Data Visualization Journey!")
    
    # Create beautiful feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Explorer</h3>
            <p>Upload your CSV or Excel files and instantly see beautiful data summaries with colorful metrics and insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Visual Magic</h3>
            <p>Create stunning charts with rainbow colors, gradients, and professional styling that will wow your audience.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📋 Sample Data</h3>
            <p>Try our beautifully crafted sample datasets to explore all features immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Perfect For:")
        st.markdown("""
        - **College Projects** 📚
        - **Data Presentations** 📈
        - **Visual Storytelling** 🎭
        - **Interactive Dashboards** 💻
        - **Colorful Reports** 🌈
        """)
    
    with col2:
        st.markdown("### ✨ What Makes It Special:")
        st.markdown("""
        - **Beautiful Color Schemes** 🎨
        - **Interactive Charts** 🖱️
        - **Professional Styling** 💎
        - **Easy to Use** 🎯
        - **Instantly Impressive** ⚡
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Ready to Create Magic?</h3>
        <p>👈 Use the sidebar to start your data visualization journey. Begin with <strong>Data Explorer</strong> to upload your data or try our <strong>Sample Datasets</strong>!</p>
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

if __name__ == "__main__":
    main() 