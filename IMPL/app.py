import os
from flask import Flask, render_template, request, send_from_directory
from keras.src.saving import load_model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
from utils import *

model = load_model('unet.keras', custom_objects={'dice_coefficients_loss': dice_coefficients_loss,
                                                      'intersection_over_union': intersection_over_union,
                                                      'dice_coefficients': dice_coefficients})

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Corrected path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html',
 error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Prediction

            result_image_path, overlay_image_path = process_image(filepath, filename.rsplit('.', 1)[0])
            return render_template('index.html', input_image=filename, result_image=result_image_path, overlay_image=overlay_image_path)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def process_image(filepath, base_filename):
    input_image = Image.open(filepath)
    input_image = input_image.convert("RGB")
    input_image = np.array(input_image)
    input_image = cv2.resize(input_image, (256, 256))
    input_image_display = input_image.copy()
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    predicted_mask = model.predict(input_image)
    predicted_mask = (predicted_mask[0] > 0.5).astype(np.uint8) * 255

    # Convert mask to 3 channels for overlaying
    predicted_mask_3ch = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)

    # Create overlay image
    overlay = cv2.addWeighted(input_image_display, 0.7, predicted_mask_3ch, 0.3, 0.0)

    # Save images
    result_filename = f'scanned_{base_filename}.png'
    overlay_filename = f'overlay_{base_filename}.png'
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    overlay_image_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)

    cv2.imwrite(result_image_path, predicted_mask)
    cv2.imwrite(overlay_image_path, overlay)

    return result_filename, overlay_filename

if __name__ == '__main__':
    app.run(debug=True)
