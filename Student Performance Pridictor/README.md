# Student Performance Analysis

This is a Streamlit web application that analyzes student performance data and builds predictive models to understand the factors affecting academic success. The application allows users to visualize data, explore relationships, build machine learning models, evaluate their performance, and make predictions based on student details.

## Features:
- **Data Overview**: View basic statistics and summaries of the dataset.
- **Exploratory Data Analysis (EDA)**: Visualize the distribution of test scores and explore the performance of students based on demographic factors such as gender, race/ethnicity, and parental education.
- **Model Building**: Train two machine learning models (Linear Regression and Random Forest Regressor) to predict student performance based on various factors.
- **Feature Importance**: Visualize the importance of different features in the models, such as coefficients in Linear Regression and feature importance in Random Forest.
- **Student Performance Prediction**: Input student details to predict their average performance score.

## Requirements:
To run the application locally, you need to install the following Python packages:
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Seaborn
- scikit-learn

### Installation:
You can install all the necessary dependencies by running:
```bash
pip install -r requirements.txt

License:
This project is open-source and available under the MIT License.


Setup Instructions:
Clone this repository to your local machine:

bash
Copy
Edit
git clone https://github.com/<your-username>/student-performance-analysis.git
Navigate to the project directory:

bash
Copy
Edit
cd student-performance-analysis
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit application:

bash
Copy
Edit
streamlit run app.py
Open the browser and visit the local server (usually http://localhost:8501).

Data Overview:
The dataset contains the following columns:

gender: Gender of the student (male/female)

race_ethnicity: Student's racial/ethnic group (group A, B, C, D, E)

parental_education: Highest education level of the student's parents

test_prep: Whether the student completed a test preparation course

math: Math score (0-100)

reading: Reading score (0-100)

writing: Writing score (0-100)

performance: Average score of math, reading, and writing

total_score: Sum of math, reading, and writing scores

math_writing: Interaction feature (math * writing / 100)

math_reading: Interaction feature (math * reading / 100)

reading_writing: Interaction feature (reading * writing / 100)

Model Building:
This project uses two machine learning models for prediction:

Linear Regression: A basic linear model to predict student performance based on input features.

Random Forest Regressor: An ensemble model that improves prediction accuracy.

Model Evaluation:
R² (Coefficient of Determination): A measure of how well the model fits the data. A value close to 1 means the model is a good fit.

RMSE (Root Mean Squared Error): A measure of prediction error. Lower values indicate better model performance.

Predict Student Performance:
The app allows users to input details like gender, race/ethnicity, and test scores (math, reading, writing) to predict their average performance.

License:
This project is open-source and available under the MIT License. See the LICENSE file for more details.

Contribution:
Feel free to fork this repository and submit pull requests with improvements or new features. If you have suggestions or bugs, please open an issue.

go
Copy
Edit

### `requirements.txt` (For dependencies)

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
