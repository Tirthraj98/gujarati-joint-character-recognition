import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from keras.models import load_model
import joblib
# import cv2
# from googletrans import Translator
import pandas as pd

# def translate_to_gujarati(text):
#     translator = Translator()
#     translated_text = translator.translate(text, src='en', dest='gu').text
#     return translated_text

gujarati_consonants_dict = {
    'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ', 'ng': 'ઙ',
    'ch': 'ચ', 'chh': 'છ', 'j': 'જ', 'z': 'ઝ',
    'at': 'ટ', 'ath': 'ઠ', 'ad': 'ડ', 'adh': 'ઢ', 'an': 'ણ',
    't': 'ત', 'th': 'થ', 'd': 'દ', 'dh': 'ધ', 'n': 'ન',
    'p': 'પ', 'f': 'ફ', 'b': 'બ', 'bh': 'ભ', 'm': 'મ',
    'y': 'ય', 'r': 'ર', 'l': 'લ', 'v': 'વ', 'sh': 'શ',
    'shh': 'ષ', 's': 'સ', 'h': 'હ', 'al': 'ળ', 'ks': 'ક્ષ',
    'gn': 'જ્ઞ'
}

df = pd.read_csv("Guj_joint_char.csv", index_col=0)

def main():
    # Placeholder for heading
    st.title("Gujarati Handwritten Joint Character Recognizer By Rachana Chaudhari_v2!!!")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Load models
    pre_joint_model, pre_joint_label_decoder = load_model_and_decoder("pre_joint_model_gray_v5.h5",
                                                                       "pre_joint_label_encoder_gray_v5.joblib")
    post_joint_model, post_joint_label_decoder = load_model_and_decoder("post_joint_model_gray_v5.h5",
                                                                         "post_joint_label_encoder_gray_v5.joblib")

    # Make predictions
    if st.button("Predict"):
        predict_image(uploaded_file, pre_joint_model, post_joint_model, pre_joint_label_decoder, post_joint_label_decoder)

def load_model_and_decoder(model_path, label_encoder_path):
    model = load_model(model_path)
    label_decoder = joblib.load(label_encoder_path)
    return model, label_decoder

def predict_image(uploaded_file, pre_joint_model, post_joint_model, pre_joint_label_decoder, post_joint_label_decoder):
    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((50, 50))
        image = image.convert('L')
        image = image.filter(ImageFilter.GaussianBlur(radius=0.9))
        image = np.array(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        image_array = image / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions
        pre_joint_prediction = pre_joint_model.predict(image_array)
        pre_joint_predicted_class = np.argmax(pre_joint_prediction)
        pre_joint_class_label = pre_joint_label_decoder.inverse_transform([pre_joint_predicted_class])[0]

        post_joint_prediction = post_joint_model.predict(image_array)
        post_joint_predicted_class = np.argmax(post_joint_prediction)
        post_joint_class_label = post_joint_label_decoder.inverse_transform([post_joint_predicted_class])[0]

        # Decode the predicted classes using label encoder
        pre_joint_guj_class_label = get_gujarati_label(pre_joint_class_label)
        post_joint_guj_class_label = get_gujarati_label(post_joint_class_label)

        # Concatenate predicted strings
        concatenated_prediction = pre_joint_class_label[:-1] + post_joint_class_label
        # concatenated_prediction_guj = pre_joint_guj_class_label + post_joint_guj_class_label
        # Retrieve the corresponding element from the DataFrame
        concatenated_prediction_guj = df.loc[pre_joint_guj_class_label.strip(), post_joint_guj_class_label.strip()]
        # concatenated_prediction_guj = translate_to_gujarati(concatenated_prediction)

        # Display results
        st.subheader("Prediction Results")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("First Model Predicted Class:", f"{pre_joint_class_label}")
        st.write("In Gujarati:", f"{pre_joint_guj_class_label}")
        st.write("Second Model Predicted Class:", f"{post_joint_class_label}")
        st.write("In Gujarati:", f"{post_joint_guj_class_label}")
        st.write("Joint Character Prediction:", f"{concatenated_prediction}")
        st.write("Joint Character Prediction in Gujarati:", f"{concatenated_prediction_guj}")
    else:
        st.warning("Please upload an image before predicting.")

def get_gujarati_label(class_label):
    guj_class_label = ""
    if class_label.lower()[:-1] in gujarati_consonants_dict.keys():
        guj_class_label = gujarati_consonants_dict[class_label.lower()[:-1]]
    elif class_label.lower()[:-1] in gujarati_consonants_dict.keys():
        guj_class_label = gujarati_consonants_dict[class_label.lower()[:-1]]

    return guj_class_label

if __name__ == "__main__":
    main()
