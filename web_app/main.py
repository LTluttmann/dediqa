import os
import cv2
import streamlit as st
from transfer_learning.utils import (
    change_res_quality, resize_if_too_small, change_res,
    downsample_tf, quality_downsample_tf
)
from cv_preprocessing.main import get_corrected_img
import tensorflow as tf
import numpy as np
from streamlit_cropper import st_cropper
import glob
from PIL import Image
import pytesseract
# relative imoports
from utils import (
    model_load,
    get_prediction,
    get_model_info,
    make_kde_plot, 
    make_pred_plot,
    make_feature_maps_plot,
    get_ocr,
    make_activation_plot,
    make_sensitivity_map,
    vis_dense_layers
)


if os.name == "nt":
    # when running on windows, the path to the tesseract executable must be explicitly specified
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\lluttmann\Miniconda3\envs\diqa\Library\bin\tesseract.exe'

# get available models in .model subdirectory
models = glob.glob("./models/*.pickle")
models = [os.path.split(x)[-1].split(".")[0] for x in models]

predict_methods = {
    "Random": "random_weighted_patches",
    "Convolution": "patches_by_convolution",
}

big_font_stlye_definition = """
    <style>
    .big-font {
        font-size:250px !important;
    }
    </style>
"""

# @st.cache(allow_output_mutation=True)
def get_model_layers(model_path):
    model = model_load(model_path)
    # get conv layers of base model
    conv_layers = [
        layer 
        for layer in model.base_model.layers 
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    return conv_layers, model

def main():
    # define container
    st.set_page_config(layout="wide")
    main_container = st.container()
    main_container.markdown("<h1 style='text-align: center;'>Estimate the quality of images</h1>", unsafe_allow_html=True)
    image = main_container.file_uploader('Upload your image here',type=['jpg','jpeg','png'])
    image_container = main_container.expander("Display Images")
    model_container = main_container.expander("Model Information")
    prediction_container = main_container.expander("Prediction")
    main_container.markdown("<h2 style='text-align: center;'>Model Interpretability Features</h2>", unsafe_allow_html=True)
    model_interpret = main_container.container()
    feature_container = model_interpret.expander("View Feature Maps")
    feat_map_layer_type = feature_container.radio("choose layer type", options=["convolutional layers", "dense layers"])
    grad_cam_container = model_interpret.expander("View  Model Sensitivity")
    activation_container = model_interpret.expander("View what makes the model activate")   

    # define sidebar
    st.sidebar.header('Model Selection')
    model_choice = st.sidebar.selectbox('Select a Model', models, index=models.index("resnet_4096"))
    model_path = os.path.join('models', f'{model_choice}.pickle')
    # read model infor
    get_model_info(model_container, model_path)
    
    predict_choice = st.sidebar.selectbox('Choose how to crop patches', list(predict_methods.keys()))
    pred_method = predict_methods[predict_choice]

    conv_layers, model = get_model_layers(model_path)

    st.sidebar.header('Configuration')
    scale = st.sidebar.slider('Downsample by scale', min_value=0.1, max_value=1.0, step=0.05, value=1.0)
    antialias = st.sidebar.checkbox("Antialiasing (for Downsampling)", value=False)
    autocrop = st.sidebar.checkbox('Auto Crop Image',value=False)
    exclude = st.sidebar.checkbox('Exclude bad quality images',value=False)
    st.sidebar.header('Explainablility Features')
    vis_feat_map = st.sidebar.checkbox('Visualize Feature Maps',value=False)
    vis_grad_cam = st.sidebar.checkbox('Visualize Model Sensitivity',value=False)
    vis_activations = st.sidebar.checkbox('Visualize Activations',value=False)
    conv_layer_idx: int = st.sidebar.slider(
        "Convolutional Layer to make visualizations for",
        min_value=1, max_value=len(conv_layers)    
    )
    st.sidebar.header('Get OCR results')
    ocr_crop = st.sidebar.checkbox('Perform OCR on crop',value=False)

    preprocessed_img = None
    if image is not None:
        img_name = image.name
        prediction_placeholder = prediction_container.empty()
        prediction_placeholder.warning("Making the prediciton...")

        # read in image
        img = image.read()
        img = tf.image.decode_image(img, channels=3).numpy()                  
        # Image preprocessing
        if autocrop:
            # add homography here
            preprocessed_img = get_corrected_img(img)
        else:
            preprocessed_img = img.copy() 
        # images can be downsample on the fly
        if scale < 1:
            size = preprocessed_img.shape
            if antialias:
                preprocessed_img = quality_downsample_tf(preprocessed_img, scale)
            else:
                preprocessed_img = downsample_tf(preprocessed_img, scale)
            preprocessed_img = cv2.resize(preprocessed_img, (size[1], size[0]))

        # display images
        col1, col2 = image_container.columns(2)
        with col1:
            # show the UNpreprocessed image in the first column
            st.text('Input image:') 
            st.image(img)
        with col2:
            # show the preprocessed image in the second column
            st.text('Preprocessed/downsampled image:') 
            st.image(preprocessed_img)
        
        # use the pred
        prediction = get_prediction(
            model_path, preprocessed_img, img_name, scale, autocrop, pred_method, antialias)
        prediction_placeholder.info("Prediction is ready")
        mean_pred = np.mean(prediction)
        fig_res = make_pred_plot(prediction)
        fig_kde = make_kde_plot(prediction)
        col1, col2, col3 = prediction_container.columns(3)
        with col1:
            st.markdown(big_font_stlye_definition, unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: left;'>The estimated DPI is:</h2>", unsafe_allow_html=True)
            st.markdown(f'<p class="big-font">{round(mean_pred)}</p>', unsafe_allow_html=True)
        with col2:
            st.pyplot(fig_res)
        with col3:
            st.pyplot(fig_kde)

        # ---------- model interpretability part -----------
        # visualize conolutional kernels
        img_crop_seed = feature_container.slider("Change Patch of Image", 1, 20)
        if vis_feat_map:
            if feat_map_layer_type == "convolutional layers":
                fig_maps = make_feature_maps_plot(
                    model_path, preprocessed_img, conv_layers[conv_layer_idx-1], seed=img_crop_seed
                )
            else:
                fig_maps = vis_dense_layers(
                    model_path, preprocessed_img, seed=img_crop_seed
                )
            col1, col2, col3 = feature_container.columns([1,4, 1])
            col2.pyplot(fig_maps)
        else:
            feature_container.warning("Tick the 'Visualize Feature Maps' checkbox to see results")
        
        # visualize gradient camera
        if vis_grad_cam:
            fig_grad_cam = make_sensitivity_map(
                model, preprocessed_img
            )
            col1, col2, col3 = grad_cam_container.columns([1,4, 1])
            col2.pyplot(fig_grad_cam)
        else:
            grad_cam_container.warning("Tick the 'Visualize  Model Sensitivity' checkbox to see results")
        
        # visualize what lead to high activations
        feat_map = activation_container.slider("Feature Map Index", 0, conv_layers[conv_layer_idx-1].get_output_at(0).get_shape().as_list()[-1]-1)
        if vis_activations:
            fig_act = make_activation_plot(model, conv_layers[conv_layer_idx-1], feat_map)
            col1, col2, col3 = activation_container.columns([1, 4, 1])
            col2.pyplot(fig_act)
        else:
            activation_container.warning("Tick the 'Visualize Activations' checkbox to see results")
    else:
        image_container.info('Awaiting the upload of an image.')
        prediction_container.info('Awaiting the upload of an image.')
        feature_container.info('Awaiting the upload of an image.')
        grad_cam_container.info('Awaiting the upload of an image.')
        activation_container.info('Awaiting the upload of an image.')
    # ---------- OCR PART -----------
    ocr_container = main_container.container()
    if ocr_crop:
        ocr_container.markdown("<h2 style='text-align: center;'>Crop part from Image for OCR</h2>", unsafe_allow_html=True)
        if preprocessed_img is not None:
            col1, col2 = ocr_container.columns(2)
            pil_img = Image.fromarray(preprocessed_img)
            with col1:
                cropped_img = st_cropper(pil_img, realtime_update=True, box_color='#0000FF', aspect_ratio=None, max_size=(1500,1500))
            with col2:
                cropped_img = np.asarray(cropped_img)
                crop_img_res = resize_if_too_small(cropped_img, 700)
                st.image(crop_img_res)

    ocr_container.markdown("<h2 style='text-align: center;'>Optical Character Recognition (OCR)</h2>", unsafe_allow_html=True)
    ocr_text_expander = ocr_container.expander("Extracted Text")
    if st.sidebar.button("Perform OCR"):
        ocr_warning = st.sidebar.empty()
        try:
            if mean_pred < 70 and exclude:
                ocr_warning.warning("Oops. The quality of your image seems to be too bad!")
            else:
                if ocr_crop:
                    string = get_ocr(crop_img_res)
                else:
                    string = get_ocr(preprocessed_img)
                ocr_text_expander.write(string)
        except UnboundLocalError:
            ocr_warning.warning("Upload and crop image first")
    else:
        ocr_text_expander.warning("Press 'Perform OCR' button to get results")
            

if __name__ == '__main__':
    main()