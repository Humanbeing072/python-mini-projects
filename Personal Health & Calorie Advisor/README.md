# Personal Health & Calorie Advisor

This is a Streamlit web app designed to help users track their daily calorie intake, calculate their basal metabolic rate (BMR), and suggest personalized meal plans based on their health and fitness goals. The app also predicts weight changes based on calorie differences and allows users to track their meals.

## Features
- **User Profile Setup**: Enter personal information like age, gender, height, weight, and activity level.
- **BMR and TDEE Calculation**: Automatically calculates Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE).
- **Daily Calorie Goals**: Adjusts daily calorie needs based on user goals (weight loss, weight maintenance, or weight gain).
- **Weight Change Prediction**: Estimates weight change over the next 30 days based on daily calorie surplus or deficit.
- **Personalized Meal Plans**: Suggests meals based on daily calorie needs and body type (lean, muscular, or average).
- **Food and Recipe Suggestions**: Displays food items rich in protein and recipes based on your goal.
- **Meal Tracker**: Track the meals you eat each day and compare total calories consumed to your daily goal.

## Requirements

Before running the app, make sure you have the following libraries installed:

- Python 3.7+
- Streamlit
- Pandas
- NumPy

You can install the required dependencies using:

```bash
pip install streamlit pandas numpy
How to Run the App
Clone the repository or download the code to your local machine.

Install the necessary libraries using the command above.

Run the Streamlit app using the following command:

bash
Copy
Edit
streamlit run app.py
Open the app in your browser (typically at http://localhost:8501).

How It Works
1. BMR and TDEE Calculation
BMR is calculated using the Mifflin-St Jeor equation and is adjusted based on activity levels to get TDEE (Total Daily Energy Expenditure).

2. Daily Calorie Goals
Depending on the user's goal (Lose weight, Maintain weight, or Gain weight), the app adjusts daily calorie recommendations.

3. Meal Plan Suggestions
The app provides a personalized meal plan with macronutrient breakdown (Protein, Carbs, Fat) based on the user’s body type and goal.

4. Food and Recipe Suggestions
The app suggests high-protein foods and recipes that align with the user’s goal, such as weight loss or muscle gain.

5. Daily Meal Tracker
Users can track their meals throughout the day, input food items, and calories consumed. The app compares total calories consumed against the user's target and provides feedback.

Example Output
The app displays the following:

Your BMR, TDEE, and Recommended Daily Calories.

Predicted weight change for the next 30 days.

Meal plan with detailed macronutrient breakdown.

High-protein foods and recipes based on your goal.

Meal tracker to log meals and monitor total calories.

License
This app is open-source and available under the MIT License.

