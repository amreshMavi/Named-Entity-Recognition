import streamlit as st
from ner_model import predict_entities, load_model_and_maps


st.set_page_config(page_title="NER Demo")

st.title("Named Entity Recognition")

text = st.text_area("Enter text")

if st.button("Analyze"):
    results = predict_entities(text)
    for word, tag in results:
        st.write(f"**{word}** -> {tag}")
