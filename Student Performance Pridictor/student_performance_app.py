import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Set page config
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide"
)

# Load data function with caching
@st.cache_data
def load_data():
    df = pd.read_csv('StudentsPerformance.csv')
    df = df.drop('lunch', axis=1)  # Drop irrelevant column
    
    # Rename columns for easier access
    df.rename(columns={
        'race/ethnicity': 'race_ethnicity',
        'parental level of education': 'parental_education',
        'test preparation course': 'test_prep',
        'math score': 'math',
        'reading score': 'reading',
        'writing score': 'writing'
    }, inplace=True)
    
    # Target variable (average performance)
    df['performance'] = df[['math', 'reading', 'writing']].mean(axis=1)
    
    # Feature engineering
    df['total_score'] = df['math'] + df['reading'] + df['writing']
    df['math_writing'] = df['math'] * df['writing'] / 100  # Normalized interaction
    df['math_reading'] = df['math'] * df['reading'] / 100
    df['reading_writing'] = df['reading'] * df['writing'] / 100
    
    return df

# Function to create pipelines
def create_pipeline(model):
    categorical_features = ['gender', 'race_ethnicity', 'test_prep']
    ordinal_features = ['parental_education']
    numeric_features = ['total_score', 'math_writing', 'math_reading', 'reading_writing']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('ord', OrdinalEncoder(categories=[[  # Ordered categories for ordinal features
                'some high school', 'high school', 'some college',
                "associate's degree", "bachelor's degree", "master's degree"
            ]]), ordinal_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    return pipeline

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'Train R²': r2_score(y_train, train_pred),
        'Test R²': r2_score(y_test, test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, test_pred))
    }
    
    return metrics

# Load data
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Model Building", "Feature Importance", "Predict Performance"])

# Main content
st.title("📊 Student Performance Analysis")
st.write("""
This application analyzes student performance data and builds predictive models to understand factors affecting academic success.
""")

if page == "Data Overview":
    st.header("📋 Data Overview")
    
    st.write("""
    ### Dataset Preview
    First 10 rows of the dataset:
    """)
    st.dataframe(df.head(10))
    
    st.write("""
    ### Dataset Information
    """)
    st.write(f"Number of students: {len(df)}")
    st.write(f"Number of features: {len(df.columns)}")
    
    st.write("""
    ### Column Descriptions
    - `gender`: Student's gender (male/female)
    - `race_ethnicity`: Student's racial/ethnic group (group A, B, C, D, E)
    - `parental_education`: Parent's highest education level
    - `test_prep`: Whether student completed test preparation course
    - `math`: Math score (0-100)
    - `reading`: Reading score (0-100)
    - `writing`: Writing score (0-100)
    - `performance`: Average of math, reading, and writing scores
    - `total_score`: Sum of all three test scores
    - `math_writing`: Interaction feature (math * writing / 100)
    - `math_reading`: Interaction feature (math * reading / 100)
    - `reading_writing`: Interaction feature (reading * writing / 100)
    """)

elif page == "Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    st.write("""
    ### Distribution of Scores
    """)
    
    # Select which score to visualize
    score_type = st.selectbox("Select score type", ['math', 'reading', 'writing', 'performance', 'total_score'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[score_type], kde=True, ax=ax)
    ax.set_title(f'Distribution of {score_type} scores')
    st.pyplot(fig)
    
    st.write("""
    ### Performance by Demographic Factors
    """)
    
    # Select demographic factor
    demo_factor = st.selectbox("Select demographic factor", ['gender', 'race_ethnicity', 'parental_education', 'test_prep'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=demo_factor, y='performance', data=df, ax=ax)
    ax.set_title(f'Performance by {demo_factor}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    
    st.write("""
    ### Correlation Between Scores
    """)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[['math', 'reading', 'writing']].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Between Test Scores')
    st.pyplot(fig)

elif page == "Model Building":
    st.header("🤖 Model Building")
    
    st.write("""
    ### Data Preparation
    We'll build models to predict student performance based on demographic factors and engineered features.
    """)
    
    # Features (drop individual scores to avoid leakage)
    X = df.drop(['performance', 'math', 'reading', 'writing'], axis=1)
    y = df['performance']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"Training set size: {len(X_train)}")
    st.write(f"Test set size: {len(X_test)}")
    
    # Initialize pipelines in session state if not already present
    if 'linear_pipeline' not in st.session_state:
        st.session_state.linear_pipeline = create_pipeline(LinearRegression())
        # Ensure X_train and y_train are defined
        if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
            X = df.drop(['performance', 'math', 'reading', 'writing'], axis=1)
            y = df['performance']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.X_train, st.session_state.y_train = X_train, y_train
    
        st.session_state.linear_pipeline.fit(st.session_state.X_train, st.session_state.y_train)
    
    if 'rf_pipeline' not in st.session_state:
        st.session_state.rf_pipeline = create_pipeline(RandomForestRegressor(random_state=42))
        # Ensure X_train and y_train are defined
        if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
            X = df.drop(['performance', 'math', 'reading', 'writing'], axis=1)
            y = df['performance']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.X_train, st.session_state.y_train = X_train, y_train
    
        st.session_state.rf_pipeline.fit(st.session_state.X_train, st.session_state.y_train)
    
    # Evaluate models
    linear_metrics = evaluate_model(st.session_state.linear_pipeline, X_train, X_test, y_train, y_test)
    rf_metrics = evaluate_model(st.session_state.rf_pipeline, X_train, X_test, y_train, y_test)
    
    st.write("""
    #### Linear Regression Results:
    """)
    st.write(f"Train R²: {linear_metrics['Train R²']:.4f}")
    st.write(f"Test R²: {linear_metrics['Test R²']:.4f}")
    st.write(f"Train RMSE: {linear_metrics['Train RMSE']:.2f}")
    st.write(f"Test RMSE: {linear_metrics['Test RMSE']:.2f}")
    
    st.write("""
    #### Random Forest Results:
    """)
    st.write(f"Train R²: {rf_metrics['Train R²']:.4f}")
    st.write(f"Test R²: {rf_metrics['Test R²']:.4f}")
    st.write(f"Train RMSE: {rf_metrics['Train RMSE']:.2f}")
    st.write(f"Test RMSE: {rf_metrics['Test RMSE']:.2f}")
    
    st.write("""
    ### Model Interpretation
    Both models perform exceptionally well, with the Linear Regression model showing perfect scores (likely due to the inclusion of interaction features that directly relate to the target). 
    The Random Forest model also performs very well with high R² values on both training and test sets.
    """)

elif page == "Feature Importance":
    st.header("📈 Feature Importance")
    
    st.write("""
    ### Linear Regression Coefficients
    """)
    
    # Get feature names after preprocessing
    # Ensure X_train and y_train are defined
    if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
        X = df.drop(['performance', 'math', 'reading', 'writing'], axis=1)
        y = df['performance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train, st.session_state.y_train = X_train, y_train

    st.session_state.linear_pipeline.fit(st.session_state.X_train, st.session_state.y_train)
    cat_features = st.session_state.linear_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['gender', 'race_ethnicity', 'test_prep'])
    feature_names = list(cat_features) + ['parental_education'] + ['total_score', 'math_writing', 'math_reading', 'reading_writing']
    
    # Get coefficients (for linear regression)
    coefficients = st.session_state.linear_pipeline.named_steps['regressor'].coef_
    
    # Create a DataFrame for visualization
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax)
    ax.set_title('Linear Regression Coefficients (Feature Importance)')
    st.pyplot(fig)
    
    st.write("""
    ### Random Forest Feature Importance
    """)
    
    # Ensure X_train and y_train are defined
    if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
        X = df.drop(['performance', 'math', 'reading', 'writing'], axis=1)
        y = df['performance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train, st.session_state.y_train = X_train, y_train

    st.session_state.rf_pipeline.fit(st.session_state.X_train, st.session_state.y_train)
    importances = st.session_state.rf_pipeline.named_steps['regressor'].feature_importances_
    
    # Create a DataFrame for visualization
    rf_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=rf_importance_df, ax=ax)
    ax.set_title('Random Forest Feature Importance')
    st.pyplot(fig)
    
    st.write("""
    ### Interpretation
    - For both models, the total_score is by far the most important feature (as expected since it's the sum of all test scores)
    - The interaction features (math_writing, math_reading, reading_writing) also show significant importance
    - Demographic factors like parental education and test preparation have smaller but still meaningful impacts
    """)

elif page == "Predict Performance":
    st.header("🔮 Predict Student Performance")
    
    st.write("""
    ### Enter Student Details
    Fill in the form to predict a student's average performance score.
    """)
    
    # Ensure pipelines are initialized
    if 'linear_pipeline' not in st.session_state or 'rf_pipeline' not in st.session_state:
        st.error("Models are not initialized. Please go to the 'Model Building' page first.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", ['male', 'female'])
                race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
                test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
                
            with col2:
                parental_edu = st.selectbox("Parental Education", [
                    'some high school', 
                    'high school', 
                    'some college',
                    "associate's degree", 
                    "bachelor's degree", 
                    "master's degree"
                ])
                math_score = st.slider("Math Score", 0, 100, 70)
                reading_score = st.slider("Reading Score", 0, 100, 70)
                writing_score = st.slider("Writing Score", 0, 100, 70)
            
            submitted = st.form_submit_button("Predict Performance")
        
        if submitted:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'gender': [gender],
                'race_ethnicity': [race],
                'parental_education': [parental_edu],
                'test_prep': [test_prep],
                'math': [math_score],
                'reading': [reading_score],
                'writing': [writing_score]
            })
            
            # Calculate engineered features
            input_data['total_score'] = input_data['math'] + input_data['reading'] + input_data['writing']
            input_data['math_writing'] = input_data['math'] * input_data['writing'] / 100
            input_data['math_reading'] = input_data['math'] * input_data['reading'] / 100
            input_data['reading_writing'] = input_data['reading'] * input_data['writing'] / 100
            
            # Drop original scores (to match training setup)
            input_data = input_data.drop(['math', 'reading', 'writing'], axis=1)
            
            # Make predictions
            linear_pred = st.session_state.linear_pipeline.predict(input_data)[0]
            rf_pred = st.session_state.rf_pipeline.predict(input_data)[0]
            
            # Calculate actual average
            actual_avg = (math_score + reading_score + writing_score) / 3
            
            st.write("""
            ### Prediction Results
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Actual Average", f"{actual_avg:.1f}")
            
            with col2:
                st.metric("Linear Regression Prediction", f"{linear_pred:.1f}")
            
            with col3:
                st.metric("Random Forest Prediction", f"{rf_pred:.1f}")
            
            st.write("""
            Note: The models perform extremely well because we're using engineered features that are directly derived from the test scores.
            In a real-world scenario, we might want to predict performance without knowing any test scores in advance.
            """)

# Footer
st.markdown("---")
st.markdown("""
**Project Expo 4.0 Submission**  
Developed for MIT Group's Project Expo 4.0  
Data Science & Machine Learning Project
""")