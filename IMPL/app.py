import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif'}

# Load your trained model
model = tf.keras.models.load_model('unet.keras', compile=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Convert .tif to .png
            if filename.rsplit('.', 1)[1].lower() == 'tif':
                file_path = convert_to_png(file_path)
            prediction_path = predict(file_path)
            return render_template('index.html', original=file_path, prediction=prediction_path)
    return render_template('index.html')

def convert_to_png(tif_path):
    with Image.open(tif_path) as img:
        png_path = tif_path.rsplit('.', 1)[0] + '.png'
        img.save(png_path)
    return png_path

def predict(image_path):
    # Preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    prediction = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255

    # Save the prediction as an image
    prediction_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.png')
    prediction_image = Image.fromarray(prediction)
    prediction_image.save(prediction_image_path)

    return 'uploads/prediction.png'

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)