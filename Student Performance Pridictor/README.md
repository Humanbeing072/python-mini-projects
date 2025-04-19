Student Performance Analysis
This is a Streamlit web application that analyzes student performance data and builds predictive models to understand the factors affecting academic success. The application allows users to visualize data, explore relationships, build machine learning models, evaluate their performance, and make predictions based on student details.

Features:
Data Overview: Provides a snapshot of the dataset, with descriptions and statistics of the columns.

Exploratory Data Analysis (EDA): Visualizes the distribution of test scores and explores the performance of students based on demographic factors such as gender, race/ethnicity, and parental education.

Model Building: Builds two predictive models (Linear Regression and Random Forest Regressor) to predict student performance, based on various demographic and test score features.

Feature Importance: Visualizes the importance of different features in the models using coefficients for Linear Regression and feature importance for Random Forest.

Predict Student Performance: A form where users can input student details (e.g., gender, race/ethnicity, test scores) to predict the average performance score.

Requirements:
To run the application, ensure the following Python packages are installed:

Streamlit

Pandas

Numpy

Matplotlib

Seaborn

scikit-learn

You can install the required packages using the following command:

bash
Copy
Edit
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
Setup Instructions:
Clone this repository to your local machine.

Ensure you have the required packages installed.

Run the Streamlit application:

bash
Copy
Edit
streamlit run app.py
Open the browser and navigate to the local server address (e.g., http://localhost:8501).

Data Overview:
The dataset includes the following columns:

gender: Gender of the student (male/female)

race_ethnicity: Student's racial/ethnic group (group A, B, C, D, E)

parental_education: Parent's highest level of education

test_prep: Whether the student completed a test preparation course

math: Math score (0-100)

reading: Reading score (0-100)

writing: Writing score (0-100)

performance: Average of math, reading, and writing scores

total_score: Sum of all three test scores

math_writing: Interaction feature (math * writing / 100)

math_reading: Interaction feature (math * reading / 100)

reading_writing: Interaction feature (reading * writing / 100)

Model Building:
Two machine learning models are built:

Linear Regression: A basic linear model to predict student performance.

Random Forest Regressor: A more complex ensemble model to predict student performance.

Evaluation Metrics:
R² (Coefficient of Determination): Indicates how well the model fits the data. R² values closer to 1 are better.

RMSE (Root Mean Squared Error): A measure of the differences between predicted and actual values. Lower values are better.

Model Interpretation:
The Linear Regression model uses coefficients to determine the importance of each feature.

The Random Forest model provides feature importance based on how useful each feature is in making predictions.

Predict Performance:
Users can input student details to predict their performance. After filling in the form, the app uses the trained models to predict the average performance score.

License:
This project is open-source and available under the MIT License.
