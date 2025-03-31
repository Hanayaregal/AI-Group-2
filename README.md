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
