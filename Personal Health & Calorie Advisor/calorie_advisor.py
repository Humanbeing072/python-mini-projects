import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants
ACTIVITY_LEVELS = {
    'Sedentary (little or no exercise)': 1.2,
    'Lightly active (light exercise/sports 1-3 days/week)': 1.375,
    'Moderately active (moderate exercise/sports 3-5 days/week)': 1.55,
    'Very active (hard exercise/sports 6-7 days a week)': 1.725,
    'Extra active (very hard exercise & physical job)': 1.9
}

GOAL_RATES = {
    'Lose weight': -0.85,
    'Maintain weight': 0,
    'Gain weight': 0.85
}

# Sample food database (in a real app, this would be more comprehensive)
FOOD_DB = pd.DataFrame({
    'Food': ['Chicken Breast', 'Salmon', 'Brown Rice', 'Broccoli', 'Eggs', 'Oatmeal', 'Almonds', 'Greek Yogurt'],
    'Calories': [165, 208, 215, 55, 70, 150, 170, 100],
    'Protein (g)': [31, 20, 5, 3.7, 6, 5, 6, 10],
    'Carbs (g)': [0, 0, 45, 11, 0.6, 27, 6, 7],
    'Fat (g)': [3.6, 13, 1.8, 0.6, 5, 3, 15, 0.7]
})

# Sample recipes (in a real app, this would be more comprehensive)
RECIPES = pd.DataFrame({
    'Recipe': ['Grilled Chicken with Vegetables', 'Salmon with Quinoa', 'Vegetable Stir Fry', 'Oatmeal with Fruits'],
    'Calories': [400, 450, 350, 300],
    'Protein (g)': [35, 30, 15, 10],
    'Carbs (g)': [25, 40, 45, 50],
    'Fat (g)': [15, 20, 10, 5],
    'Ingredients': [
        'Chicken breast, broccoli, carrots, olive oil',
        'Salmon fillet, quinoa, lemon, asparagus',
        'Mixed vegetables, tofu, soy sauce, sesame oil',
        'Oats, almond milk, banana, berries, honey'
    ]
})

def calculate_bmr(weight, height, age, gender):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure"""
    return bmr * activity_level

def calculate_daily_calories(tdee, goal):
    """Adjust calories based on goal"""
    return tdee + (goal * 500)  # 500 calorie deficit/surplus per day

def predict_weight_change(current_weight, calorie_difference, days=30):
    """Predict weight change based on calorie difference (3500 calories ≈ 1 lb)"""
    weight_change = (calorie_difference * days) / 3500
    return weight_change

def suggest_meal_plan(daily_calories, body_type='average', weight=70):
    """Suggest a realistic meal plan based on daily calorie needs and body type"""
    # Set protein requirement based on body type and weight
    if body_type == 'muscular':
        protein_grams = weight * 2.0
    elif body_type == 'lean':
        protein_grams = weight * 1.5
    else:
        protein_grams = weight * 1.2

    protein_cals = protein_grams * 4
    remaining_cals = daily_calories - protein_cals
    
    # Distribute remaining calories between carbs and fat
    fat_cals = remaining_cals * 0.3
    carb_cals = remaining_cals * 0.7

    return {
        'Protein (g)': round(protein_grams),
        'Carbs (g)': round(carb_cals / 4),
        'Fat (g)': round(fat_cals / 9),
        'Breakfast': f"{carb_cals/3:.0f} calories from carbs, {protein_cals/3:.0f} from protein",
        'Lunch': f"{carb_cals/3:.0f} calories from carbs, {protein_cals/3:.0f} from protein",
        'Dinner': f"{carb_cals/3:.0f} calories from carbs, {protein_cals/3:.0f} from protein",
        'Snacks': f"Remaining {daily_calories * 0.1:.0f} calories"
    }

def main():
    st.title("Personal Health & Calorie Advisor")
    
    # User inputs
    st.sidebar.header("User Profile")
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.sidebar.radio("Gender", ['Male', 'Female'])
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.sidebar.number_input("Current Weight (kg)", min_value=30, max_value=200, value=70)
    goal = st.sidebar.selectbox("Goal", list(GOAL_RATES.keys()))
    activity_level = st.sidebar.selectbox("Activity Level", list(ACTIVITY_LEVELS.keys()))
    body_type = st.sidebar.selectbox("Body Type", ['Average', 'Lean', 'Muscular'])
    
    # Calculations
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, ACTIVITY_LEVELS[activity_level])
    daily_calories = calculate_daily_calories(tdee, GOAL_RATES[goal])
    weight_change = predict_weight_change(weight, GOAL_RATES[goal] * 500)
    
    # Display results
    st.header("Your Health Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("BMR (Basal Metabolic Rate)", f"{bmr:.0f} kcal")
    col2.metric("TDEE (Total Daily Energy Expenditure)", f"{tdee:.0f} kcal")
    col3.metric("Recommended Daily Calories", f"{daily_calories:.0f} kcal")
    
    st.subheader(f"Predicted Weight Change in 30 Days: {weight_change:.1f} kg")
    
    # Meal plan
    st.header("Personalized Meal Plan")
    meal_plan = suggest_meal_plan(daily_calories, body_type.lower(), weight)
    st.json(meal_plan)
    
    # Food and recipe suggestions
    st.header("Food & Recipe Suggestions")
    
    st.subheader("High Protein Foods")
    st.dataframe(FOOD_DB.sort_values('Protein (g)', ascending=False).head(5))
    
    st.subheader("Recipes Matching Your Goals")
    if goal == 'Lose weight':
        st.dataframe(RECIPES.sort_values('Calories').head(3))
    elif goal == 'Gain weight':
        st.dataframe(RECIPES.sort_values('Calories', ascending=False).head(3))
    else:
        st.dataframe(RECIPES.sample(3))
    
    # Daily meal tracker (simplified)
    st.header("Daily Meal Tracker")
    today = datetime.now().strftime("%Y-%m-%d")
    if 'meals' not in st.session_state:
        st.session_state.meals = {}
    
    meal_type = st.selectbox("Meal Type", ['Breakfast', 'Lunch', 'Dinner', 'Snack'])
    food_item = st.text_input("What did you eat?")
    calories = st.number_input("Calories", min_value=0)
    
    if st.button("Add Meal"):
        if today not in st.session_state.meals:
            st.session_state.meals[today] = []
        st.session_state.meals[today].append({
            'meal_type': meal_type,
            'food_item': food_item,
            'calories': calories
        })
        st.success("Meal added!")
    
    if today in st.session_state.meals:
        st.subheader(f"Today's Meals ({today})")
        today_meals = pd.DataFrame(st.session_state.meals[today])
        st.dataframe(today_meals)
        total_calories = today_meals['calories'].sum()
        st.metric("Total Calories Today", total_calories)
        
        # Compare to goal
        if total_calories > daily_calories:
            st.warning(f"You're {total_calories - daily_calories:.0f} calories over your goal!")
        elif total_calories < daily_calories:
            st.info(f"You're {daily_calories - total_calories:.0f} calories under your goal.")
        else:
            st.success("Perfect! You've met your calorie goal exactly.")

if __name__ == "__main__":
    main()