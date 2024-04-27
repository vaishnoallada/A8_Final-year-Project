import cv2
import matplotlib.cm as cm
#from IPython.display import Image, display
import numpy as np
import streamlit as st
import tensorflow as tf
#from PIL import Image
from tensorflow.keras.preprocessing import image as siva
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
model = tf.keras.models.load_model("cnn_final_.h5")
model_res = tf.keras.models.load_model("resnet50_res_2.h5")
model_vgg=tf.keras.models.load_model("VGG16.h5")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.theculturetrip.com/wp-content/uploads/2018/05/shutterstock_1097144396.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



new_title = '<p style="font-family:sans-serif; color:white; font-size: 42px;">Detecting Fake Currency using Deep Learning Technique CNN</p>'
st.markdown(new_title, unsafe_allow_html=True)
### load file
def display_content():
    st.markdown("<span style='font-size:25px;color:white;'>Counterfeiting in the currency market has intensified due to advancements in printing and scanning technology, posing a significant threat to the economy by devaluing genuine currency so deep convolution neural networks are proposed to spot fake bills by learning from lots of real and fake money pictures.Our program efficiently detects fake 100 and 500 bills, improving security and accuracy to protect the economy. </span>", unsafe_allow_html=True)

# Add button to display content
if st.button("Abstract"):
    display_content()
st.markdown("<span style='font-size:25px;color:white;'>We Designed our project using CNN,Resnet50 and VGG16 models For the detection of Fake and Real currency.</span>",unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a image file", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file)
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        img = tf.keras.utils.load_img(uploaded_file,target_size=(224, 224, 3))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        img1 = tf.keras.utils.load_img(uploaded_file,target_size=(120, 120, 3))
        input_arr1 = tf.keras.preprocessing.image.img_to_array(img1)
        input_arr1 = np.array([input_arr1])
        res = model.predict(input_arr1)
        res_resnet = model_res.predict(input_arr1)
        res_vgg    =model_vgg.predict(input_arr)
        #plt.imshow(img)
        fr=res_resnet[0][0]
        rr=res_resnet[0][1]
        fv=res_vgg[0][0]
        rv=res_vgg[0][1]
        resf=(fr+fv)/2
        resr=(rr+rv)/2
        resf1=res[0][0]
        resr1=res[0][1]
        average_percentage_real = (resr + resr1) / 2
        average_percentage_fake = (resf + resf1) / 2
        if average_percentage_real>=0.5:st.markdown("<span style='font-size:30px;color:white;'>The given currency is Real !!! </span>",unsafe_allow_html=True)
            
        else:
            st.markdown("<span style='font-size:30px;color:white;'>The given currency is Fake !!! </span>",unsafe_allow_html=True)
        st.markdown("<span style='font-size:30px;color:white;'>The given currency is {:.2f}% Real ! </span>".format(average_percentage_real * 100), unsafe_allow_html=True)
        st.markdown("<span style='font-size:30px;color:white;'>The given currency is {:.2f}% Fake ! </span>".format(average_percentage_fake * 100), unsafe_allow_html=True)
        
        st.markdown("<span style='font-size:30px;color:white;'>Thank you for using our currency detection system. We have successfully predicted whether the given currency is real or fake based on our analysis.</span>", unsafe_allow_html=True)

        #st.markdown("<span style='font-size:30px;color:white;'>average percentage {} Real ! </span>".format(average_percentage_real),unsafe_allow_html=True)
        # st.markdown("<span style='font-size:30px;color:white;'>The given currency is {:.2f}% Real ! </span>".format(resr1 * 100), unsafe_allow_html=True)
        # st.markdown("<span style='font-size:30px;color:white;'>The given currency is {:.2f}% Fake ! </span>".format(resf1 * 100), unsafe_allow_html=True)
           