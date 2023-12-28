# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the Model
model = load_model('crop_disease_detection_model.h5')

# Name of Classes
CLASS_NAMES = ['Apple-Apple_scab', 'Blueberry-healthy', 'Cherry-Powdery_mildew', 'Corn-Common_rust', 'Grape-Black_rot', 'Raspberry-healthy',
              'Orange-Citrus_greening' , 'Pepper_bell-Bacterial_spot', 'Potato-Early_blight', 'Peach-Bacterial_spot', 'Soybean-healthy',
              'Squash-Powdery_mildew', 'Strawberry-Leaf_scorch', 'Tomato-Bacterial_spot']

# Dictionary containing information about diseases and their treatments
treatment_info = {
    'Apple-Apple_scab': (
        "Apple scab is a fungal disease that commonly affects apple trees, causing dark scaly lesions on leaves and fruit. "
        "To manage Apple scab, use fungicides such as copper-based sprays (e.g., Copper Fungicide). Additionally, maintaining "
        "good orchard hygiene, including proper pruning and removal of infected leaves, helps prevent the spread of the disease."
    ),
    'Blueberry-healthy': (
        "No disease detected. The plant is healthy. Blueberries are susceptible to various diseases, so it is crucial to "
        "monitor the plants regularly for any signs of infection. Implementing proper care and hygiene practices contributes "
        "to overall plant health."
    ),
    'Cherry-Powdery_mildew': (
        "Powdery mildew is a common fungal disease affecting cherries. It appears as a white powdery substance on leaves and "
        "can impact fruit development. Control measures include using fungicides specifically designed for powdery mildew (e.g., "
        "Funginex). Pruning affected branches and improving air circulation around the tree also help manage the disease."
    ),
    'Corn-Common_rust': (
        "Common rust is a fungal disease that affects corn plants. It presents as orange-brown pustules on leaves. Fungicides "
        "can be used to manage common rust, and planting resistant corn varieties is recommended. Remove and destroy infected "
        "plants to prevent the disease from spreading."
    ),
    'Grape-Black_rot': (
        "Black rot is a fungal disease affecting grapevines. It causes dark lesions on leaves and can lead to fruit rot. Manage "
        "black rot by applying fungicides during the growing season (e.g., Captan). Practice good vineyard management, including "
        "pruning and proper spacing, to reduce humidity and minimize disease development."
    ),
    'Raspberry-healthy': (
        "No disease detected. The plant is healthy. Raspberry plants are susceptible to various diseases, including fungal "
        "infections and viruses. Regular monitoring and proper care, such as adequate spacing and removal of infected canes, "
        "help maintain plant health."
    ),
    'Orange-Citrus_greening': (
        "Citrus greening, also known as Huanglongbing, is a bacterial disease affecting citrus trees. It is transmitted by the "
        "Asian citrus psyllid. Control measures include managing psyllid populations, removing and destroying infected trees, "
        "and applying appropriate bactericides (e.g., Streptomycin). Early detection and intervention are crucial for disease "
        "management."
    ),
    'Pepper_bell-Bacterial_spot': (
        "Bacterial spot is a common bacterial disease affecting bell peppers. Copper-based sprays are effective in managing "
        "bacterial spot (e.g., Kocide). Planting disease-resistant pepper varieties and providing proper spacing for the air "
        "circulation can help prevent the disease."
    ),
    'Potato-Early_blight': (
        "Early blight is a fungal disease that affects potato plants. It causes dark lesions on leaves and can reduce yields. "
        "Control measures include applying fungicides (e.g., Mancozeb), practicing crop rotation, and removing infected plant  "
        "material. Proper irrigation management also helps prevent the development of early blight."
    ),
    'Peach-Bacterial_spot': (
        "Bacterial spot is a bacterial disease affecting peaches. It causes dark lesions on leaves and fruit. Manage bacterial "
        "spot by applying copper sprays during the growing season (e.g., Cuprofix). Proper pruning and air circulation help reduce "
        "disease incidence."
    ),
    'Soybean-healthy' : (
        "No disease detected. The plant is healthy. Soybean plants are susceptible to various diseases, mainly pests destroy the "
        "crops. Maintain soybean health by monitoring for pests and diseases regularly. Ensure well-drained soil with appropriate "
        "fertility levels for optimal growth and use balanced fertilizers rich in nitrogen to support lush foliage. "   
    ),
    'Squash-Powdery_mildew': (
        "Powdery mildew is a fungal disease that commonly affects squash plants. It appears as a white powdery substance on leaves. "
        "Control measures include using fungicides labeled for powdery mildew (e.g., Sulfur) and planting disease-resistant varieties. "
        "Proper spacing and airflow also help prevent the disease."
    ),
    'Strawberry-Leaf_scorch': (
        "Leaf scorch is a disease affecting strawberries, causing browning and necrosis of leaf edges. Control measures include "
        "applying fungicides (e.g., Propiconazole), removing and destroying infected plants, and maintaining proper irrigation. "
        "Good air circulation is essential to reduce humidity and prevent leaf scorch."
    ),
    'Tomato-Bacterial_spot': (
        "Bacterial spot is a bacterial disease affecting tomatoes. It causes dark lesions on leaves and fruit. Control measures include "
        "applying copper-based sprays (e.g., Bordeaux mixture), avoiding overhead irrigation, and practicing crop rotation. Early detection "
        "and removal of infected plants help manage bacterial spot."
    ),
}

# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Title and description
st.markdown("<h1>🌱 Crop Disease Detection</h1><p>Upload an image of the crop leaf for disease prediction.</p>", unsafe_allow_html=True)

# Image upload section
st.markdown("---", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col2:
    crop_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction button
submit = st.button('Detect Disease', key="detect_button")

# On predict button click
if submit:
    # Check if a plant image has been uploaded
    if crop_image is None:
        # If no image is uploaded, display a warning message
        st.write("<div style='font-size: 1.2em;'>⚠️ Please upload a crop leaf image for detection. </div>", unsafe_allow_html=True)
    else:
        # If an image is uploaded, proceed with the following steps:
        
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(crop_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image with styling
        st.markdown("---")
        st.image(opencv_image, channels="BGR", width=256, use_column_width=False)

        # Displaying the image shape next to the image
        st.write(f"Image Shape: {opencv_image.shape[1]} x {opencv_image.shape[0]} pixels")

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4 Dimension
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        # Displaying the result with a styled message
        if "healthy" in result:
            st.write(f"<h3>Result: {result.split('-')[0]} leaf is {result.split('-')[1]}</h3>", unsafe_allow_html=True)
        else:
            st.write(f"<h3>Result: {result.split('-')[0]} leaf infected with {result.split('-')[1]}</h3>", unsafe_allow_html=True)
        disease_name = result.split('-')[1]
        medication_info = treatment_info.get(result)
        st.write(f"<p><strong>Treatment:</strong> {medication_info}</p>", unsafe_allow_html=True)

        