import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Text Summarization", layout="centered")
st.title("📝 Text Summarization App")

text_input = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Summarizing..."):
            try:
                # Use the default T5 model for testing
                pipe = pipeline("summarization", model="t5-small")
                output = pipe(text_input, max_length=150, num_beams=4, early_stopping=True)[0]["summary_text"]
                st.subheader("Summary:")
                st.success(output)
            except Exception as e:
                st.error(f"Error: {e}")
