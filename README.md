# AI-Group-2
# File Upload
uploaded_file = st.file_uploader("Upload your dataset", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
