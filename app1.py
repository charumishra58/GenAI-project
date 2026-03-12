import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.title("AI Language Translator")

# Language models
model_names = {
    "English → Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "English → French": "Helsinki-NLP/opus-mt-en-fr",
    "English → Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English → German": "Helsinki-NLP/opus-mt-en-de"
}

translation_type = st.selectbox(
    "Select Translation",
    list(model_names.keys())
)

text = st.text_area("Enter text to translate")

# Load model
@st.cache_resource
def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

if st.button("Translate"):

    if text.strip() == "":
        st.warning("Please enter text")

    else:
        model_name = model_names[translation_type]
        tokenizer, model = load_model(model_name)

        tokens = tokenizer(text, return_tensors="pt", padding=True)

        translated = model.generate(**tokens)

        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        st.subheader("Translation")
        st.text_area("Result", value=result, height=200)