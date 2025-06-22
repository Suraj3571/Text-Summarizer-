import streamlit as st
from src.TextSummarizer.pipeline.prediction import PredictionPipeline

# Set Streamlit page configuration
st.set_page_config(page_title="Text Summarization", layout="centered")

# Title of the app
st.title("📝 Text Summarization App")

# Text input for summarization
text_input = st.text_area("Enter text to summarize:", height=200)

# Handling the Summarization button
if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Summarizing..."):
            try:
                # Create pipeline object and generate summary
                obj = PredictionPipeline()
                summary = obj.predict(text_input)

                # Display the summary
                st.subheader("Summary:")
                st.success(summary)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
