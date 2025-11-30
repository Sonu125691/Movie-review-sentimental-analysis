from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st


with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

@st.cache_resource
def select():
    from tensorflow.keras.models import load_model
    return load_model("model.keras")

model = select()


st.title("🎞️ Movie Review Emotion Checker")
st.write('''
         Please write the movie review below. 
         This application will predict whether the review is **Positive** or **Negative**.
         ''')


uploading_text = st.text_area("Enter the review")

if st.button("Predict"):
    if uploading_text.strip() != "":
        #Preprocessing
        uploading_text = uploading_text.lower()

        text = tokenizer.texts_to_sequences([uploading_text])
        text = pad_sequences(text, maxlen = 200)

        #Predict
        predict = model.predict(text)
        predict_class = (predict > 0.50).astype(int)
        predict_percentage = predict[0][0]
        
        if 0.3 < predict_percentage < 0.7:
            st.warning("Low confidence rate. Please write a clear movie review")
        else:
            if predict_class[0][0] == 1:
                st.info("Positive Review")
            else:
                st.info("Negative Review")


    else:
        st.warning("Please write the movie review first")