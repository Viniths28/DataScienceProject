import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="DataInsight - Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #333;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📊 DataInsight - Data Science Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Upload your dataset and explore data science concepts!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Data Explorer", "📈 Visualizations", "🤖 Machine Learning", "📋 Sample Data"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "📈 Visualizations":
        show_visualizations()
    elif page == "🤖 Machine Learning":
        show_machine_learning()
    elif page == "📋 Sample Data":
        show_sample_data()

def show_home():
    st.markdown("## Welcome to DataInsight! 🚀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What You Can Do:")
        st.markdown("""
        - **Upload Data**: CSV or Excel files
        - **Explore Data**: Statistics and data quality
        - **Visualize**: Interactive charts and plots
        - **ML Models**: Linear Regression, Decision Trees, K-Means
        - **Learn**: Understand data science concepts
        """)
    
    with col2:
        st.markdown("### 🧠 Learning Objectives:")
        st.markdown("""
        - Data preprocessing and cleaning
        - Exploratory data analysis
        - Data visualization techniques
        - Basic machine learning algorithms
        - Model evaluation metrics
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("👈 Use the sidebar to navigate to different sections. Start with **Data Explorer** to upload your dataset!")

def show_data_explorer():
    st.markdown("## 📊 Data Explorer")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file to get started"
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
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Data info
            st.markdown("### 📊 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Column Information:**")
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(info_df)
            
            with col2:
                st.markdown("**Statistical Summary:**")
                st.dataframe(df.describe())
            
            # Missing values analysis
            if df.isnull().sum().sum() > 0:
                st.markdown("### 🚨 Missing Values Analysis")
                missing_data = df.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a dataset to get started!")

def show_visualizations():
    st.markdown("## 📈 Data Visualizations")
    
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset first in the Data Explorer section!")
        return
    
    df = st.session_state.df
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["Correlation Heatmap", "Distribution Plots", "Box Plots", "Scatter Plots", "Count Plots"]
    )
    
    if viz_type == "Correlation Heatmap":
        st.markdown("### 🔗 Correlation Heatmap")
        st.markdown("Shows relationships between numerical variables (-1 to 1)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdYlBu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Values close to 1 or -1 indicate strong relationships")
        else:
            st.warning("Need at least 2 numerical columns for correlation analysis")
    
    elif viz_type == "Distribution Plots":
        st.markdown("### 📊 Distribution Plots")
        st.markdown("Shows the distribution of values in numerical columns")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column:", numeric_cols)
            
            fig = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{df[selected_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[selected_col].std():.2f}")
        else:
            st.warning("No numerical columns found for distribution analysis")
    
    elif viz_type == "Box Plots":
        st.markdown("### 📦 Box Plots")
        st.markdown("Shows quartiles and outliers in your data")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column:", numeric_cols)
            
            fig = px.box(
                df,
                y=selected_col,
                title=f"Box Plot of {selected_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Dots outside the whiskers are potential outliers")
        else:
            st.warning("No numerical columns found for box plot analysis")
    
    elif viz_type == "Scatter Plots":
        st.markdown("### 🎯 Scatter Plots")
        st.markdown("Explore relationships between two numerical variables")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Select Y-axis:", numeric_cols)
            
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            corr = df[x_col].corr(df[y_col])
            st.metric("Correlation Coefficient", f"{corr:.3f}")
        else:
            st.warning("Need at least 2 numerical columns for scatter plot")
    
    elif viz_type == "Count Plots":
        st.markdown("### 📊 Count Plots")
        st.markdown("Shows frequency of categorical variables")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            selected_col = st.selectbox("Select column:", categorical_cols)
            
            # Limit to top 10 categories to avoid cluttered plots
            top_categories = df[selected_col].value_counts().head(10)
            
            fig = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title=f"Count of {selected_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 Showing top 10 categories out of {df[selected_col].nunique()} unique values")
        else:
            st.warning("No categorical columns found for count plot analysis")

def show_machine_learning():
    st.markdown("## 🤖 Machine Learning")
    
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset first in the Data Explorer section!")
        return
    
    df = st.session_state.df
    
    # ML algorithm selection
    ml_type = st.selectbox(
        "Choose ML algorithm:",
        ["Linear Regression", "Decision Tree Classification", "K-Means Clustering"]
    )
    
    if ml_type == "Linear Regression":
        show_linear_regression(df)
    elif ml_type == "Decision Tree Classification":
        show_decision_tree(df)
    elif ml_type == "K-Means Clustering":
        show_kmeans_clustering(df)

def show_linear_regression(df):
    st.markdown("### 📈 Linear Regression")
    st.markdown("Predict a continuous target variable using linear relationships")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numerical columns for linear regression")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select target variable (Y):", numeric_cols)
    with col2:
        feature_cols = st.multiselect("Select features (X):", 
                                     [col for col in numeric_cols if col != target_col])
    
    if len(feature_cols) > 0:
        # Prepare data
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()
        
        # Align X and y indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:
            st.warning("Not enough data points for reliable analysis")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.3f}")
        
        # Prediction vs Actual plot
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            title="Actual vs Predicted Values",
            labels={'x': 'Actual', 'y': 'Predicted'}
        )
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                     x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 R² closer to 1 means better model performance. The red line shows perfect predictions.")

def show_decision_tree(df):
    st.markdown("### 🌳 Decision Tree Classification")
    st.markdown("Classify data using a series of yes/no questions")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        st.warning("Need at least one categorical column as target for classification")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select target variable (Y):", categorical_cols)
    with col2:
        feature_cols = st.multiselect("Select features (X):", 
                                     [col for col in numeric_cols if col != target_col])
    
    if len(feature_cols) > 0:
        # Prepare data
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()
        
        # Align X and y indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:
            st.warning("Not enough data points for reliable analysis")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth for simplicity
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display results
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Accuracy shows the percentage of correct predictions. Diagonal values in confusion matrix are correct predictions.")

def show_kmeans_clustering(df):
    st.markdown("### 🎯 K-Means Clustering")
    st.markdown("Group similar data points together")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numerical columns for clustering")
        return
    
    feature_cols = st.multiselect("Select features for clustering:", numeric_cols)
    
    if len(feature_cols) >= 2:
        # Prepare data
        X = df[feature_cols].dropna()
        
        if len(X) < 10:
            st.warning("Not enough data points for reliable clustering")
            return
        
        # Choose number of clusters
        n_clusters = st.slider("Number of clusters:", 2, 8, 3)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Visualize clusters (use first two features)
        fig = px.scatter(
            x=X.iloc[:, 0],
            y=X.iloc[:, 1],
            color=clusters,
            title=f"K-Means Clustering (k={n_clusters})",
            labels={'x': feature_cols[0], 'y': feature_cols[1]}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster centers
        centers = kmeans.cluster_centers_
        st.markdown("### 🎯 Cluster Centers")
        center_df = pd.DataFrame(centers, columns=feature_cols)
        center_df.index = [f"Cluster {i+1}" for i in range(n_clusters)]
        st.dataframe(center_df)
        
        st.info("💡 Each color represents a different cluster. Points of the same color are similar to each other.")

def show_sample_data():
    st.markdown("## 📋 Sample Datasets")
    st.markdown("Try these sample datasets to explore the dashboard features:")
    
    # Create sample datasets
    sample_data = {
        "🏠 House Prices": create_house_price_data(),
        "🎓 Student Performance": create_student_data(),
        "🛒 Sales Data": create_sales_data()
    }
    
    selected_dataset = st.selectbox("Choose a sample dataset:", list(sample_data.keys()))
    
    if st.button("Load Sample Dataset"):
        st.session_state.df = sample_data[selected_dataset]
        st.success(f"✅ {selected_dataset} loaded successfully!")
        st.dataframe(sample_data[selected_dataset].head())

def create_house_price_data():
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic house data
    size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    
    # Create price based on features with some noise
    price = (size * 100 + bedrooms * 15000 - age * 1000 + 
             np.where(location == 'Urban', 50000, 
                     np.where(location == 'Suburban', 20000, 0)) + 
             np.random.normal(0, 20000, n_samples))
    
    return pd.DataFrame({
        'Size_sqft': size.astype(int),
        'Bedrooms': bedrooms,
        'Age_years': age,
        'Location': location,
        'Price': price.astype(int)
    })

def create_student_data():
    np.random.seed(42)
    n_samples = 150
    
    study_hours = np.random.normal(5, 2, n_samples)
    attendance = np.random.normal(85, 10, n_samples)
    previous_grade = np.random.normal(75, 15, n_samples)
    
    # Create final grade based on features
    final_grade = (study_hours * 8 + attendance * 0.3 + previous_grade * 0.4 + 
                   np.random.normal(0, 5, n_samples))
    
    # Create pass/fail based on grade
    pass_fail = np.where(final_grade >= 70, 'Pass', 'Fail')
    
    return pd.DataFrame({
        'Study_Hours': np.clip(study_hours, 0, 12).round(1),
        'Attendance_Percent': np.clip(attendance, 0, 100).round(1),
        'Previous_Grade': np.clip(previous_grade, 0, 100).round(1),
        'Final_Grade': np.clip(final_grade, 0, 100).round(1),
        'Pass_Fail': pass_fail
    })

def create_sales_data():
    np.random.seed(42)
    n_samples = 300
    
    # Generate sales data
    months = pd.date_range('2023-01-01', periods=12, freq='M')
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for month in months:
        for product in products:
            for region in regions:
                sales = np.random.normal(1000, 200)
                price = np.random.normal(50, 10)
                data.append({
                    'Month': month.strftime('%Y-%m'),
                    'Product': product,
                    'Region': region,
                    'Sales_Units': max(int(sales), 0),
                    'Price_per_Unit': round(max(price, 10), 2),
                    'Revenue': round(max(sales, 0) * max(price, 10), 2)
                })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main() 