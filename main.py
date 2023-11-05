import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    st.image(cam_path)

def app(title=None)-> None:
    """Creates the streamlit app

    Args:
        title (string, optional): The App name. Defaults to None.
    """
    st.title(title)
    col1, col2, col3 = st.columns([0.1,.015,.1])
    col1.markdown("### Developer: Mike Salem")
    col2.image("LI-In-Bug.jpg", width=48)
    col3.markdown("### [LinkedIn](https://www.linkedin.com/in/mike-salem/)")    
    st.write("The following is an implementation of GradCam from the Keras's Library")
    st.write("The implementation has been wrapped into Streamlit for the audiance to explore. To use, copy a URL of an image (not the image itself) and watch the algorithm tell you the prediction and what pixels it used to make that prediction. Note this is a general example so please use general images (e.g. footballs, airplanes, food, etc.)")
    model_builder = keras.applications.xception.Xception
    img_size = (299, 299)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    last_conv_layer_name = "block14_sepconv2_act"

    st.markdown("### Enter an image URL")

    url_text = None
    url_text = st.text_input("Enter a URL Here")

    if url_text != "":
        # The local path to our target image 
        img_path = keras.utils.get_file(
            f"{hash(url_text)}.jpeg", url_text
        )

        st.write(url_text)
        # Prepare image
        img_array = preprocess_input(get_img_array(img_path, size=img_size))

        # Make model
        model = model_builder(weights="imagenet")

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = model.predict(img_array)

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        st.markdown("### Original Image")
        st.image(img_path)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap, ax=ax)

        st.markdown("### Pixel Heatmap")
        st.write(fig)

        st.markdown("### XAI")
        save_and_display_gradcam(img_path, heatmap)

        # Print what the two top predicted classes are
        preds = model.predict(img_array)
        st.write(f"Predicted:  {decode_predictions(preds, top=1)[0][0][1]}")
        st.write("Source: https://keras.io/examples/vision/grad_cam/")

if __name__ == "__main__":
    app("GRAD CAM Example")