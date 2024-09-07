from flask import Flask, render_template, request, redirect, url_for
from tf_keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the pre-trained InceptionV3 model
model = tf.keras.models.load_model('inceptionv3_oil_spill_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam(img_path, model, last_conv_layer_name='mixed10', pred_index=None):
    img_array = preprocess_image(img_path)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Extract the area with the highest intensity in the heatmap
    max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_loc = (max_loc[1], max_loc[0])  # Convert to (x, y) format

    # Draw a pointer (e.g., a circle) on the superimposed image
    cv2.circle(superimposed_img, max_loc, 10, (0, 0, 255), 3)
    cv2.putText(superimposed_img, 'Oil Spill', (max_loc[0] + 15, max_loc[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return superimposed_img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Preprocess the image
            processed_img = preprocess_image(file_path)
            
            # Predict
            prediction = model.predict(processed_img)
            class_names = ['No Oil Spill', 'Oil Spill', 'Look alike']
            prediction_class = class_names[np.argmax(prediction)]
            
            # Generate Grad-CAM and save the image
            gradcam_image = apply_gradcam(file_path, model)
            gradcam_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + file.filename)
            cv2.imwrite(gradcam_image_path, gradcam_image)
            
            return render_template('result.html', prediction_text=f"Prediction: {prediction_class}", image_path=gradcam_image_path)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
