#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def predict_popular_ev():
    # Read the cleaned data from CleanEV.csv
    df_cleaned = pd.read_csv('CleanEV.csv')

    # Select relevant features for prediction (e.g., EV_Model, Year, Sales)
    df_selected = df_cleaned[['EV_Model', 'Year', 'Sales']]

    # Convert the 'Sales' column to categorical labels
    df_selected['Sales'] = pd.Categorical(df_selected['Sales']).codes

    # Create dummy variables for the EV_Model feature
    df_encoded = pd.get_dummies(df_selected, columns=['EV_Model'], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop('Sales', axis=1), df_encoded['Sales'], test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict the most popular EV in 2025
    prediction_2025 = rf_classifier.predict([[2025]])

    # Visualize the feature importances
    feature_importances = rf_classifier.feature_importances_
    feature_names = df_encoded.drop('Sales', axis=1).columns
    plt.bar(feature_names, feature_importances, color='blue')
    plt.xlabel('EV Models')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance for Predicting EV Sales')
    plt.xticks(rotation=90)
    plt.show()

    # Convert the predicted label back to the original category
    predicted_ev = pd.Categorical(df_selected['Sales']).categories[prediction_2025[0]]

    print("Predicted most popular EV in 2025:", predicted_ev)

if __name__ == '__main__':
    predict_popular_ev()

