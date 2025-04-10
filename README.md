# AI-Group-2
# File Upload

uploaded_file = st.file_uploader("Upload your dataset", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    # Feature Importance Visualization
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = importances.argsort()
    fig, ax = plt.subplots()
    ax.barh(feature_names[sorted_idx], importances[sorted_idx], color='orange')
    ax.set_xlabel("Feature Importance")
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to begin analysis.")
import joblib  # For saving the model

# Step 10: Compare Actual vs Predicted values in a DataFrame

comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nComparison of Actual and Predicted Final Scores:")
print(comparison_df)

# Step 11: Save the trained model and scaler

joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler have been saved.")

# Step 12: Predict new data

new_data = pd.DataFrame({'study_hours': [6], 'attendance_percentage': [85]})
new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)
print(f"\nPredicted final score for 6 study hours and 85% attendance: {new_prediction[0]:.2f}")

# Step 13: Enhanced Visualization: Plot Actual vs Predicted with a line of perfect prediction

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Actual vs Predicted Final Scores')
plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 14 : Residuals plot

residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Final Score')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()
