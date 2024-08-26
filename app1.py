from flask import Flask, render_template, request, send_file
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw
import io
import tensorflow_hub as hub

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit the size of the uploaded file to 16MB

# Ensure the upload and processed folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load DeepLab v3 model from TensorFlow Hub
model_url = 'https://tfhub.dev/google/deeplabv3/1'
model = hub.load(model_url)

def process_image(image_path, effect, coordinates=None):
    # Open the image
    image = Image.open(image_path).convert('RGB')
    input_array = np.array(image)

    # Resize image for model input
    input_array = tf.image.resize(input_array, (512, 512))
    input_array = np.expand_dims(input_array, 0) / 255.0
    
    # Perform inference with DeepLab v3 model
    predictions = model(input_array)
    mask = np.argmax(predictions['default'][0].numpy(), axis=-1)
    mask = Image.fromarray(mask.astype(np.uint8))

    if effect == 'greyscale':
        # Convert the original image to greyscale
        image = ImageOps.grayscale(image)
        image = ImageOps.colorize(image, 'gray', 'gray')  # To ensure image is in RGB mode

    elif effect == 'highlight':
        # Convert mask to a binary image for highlighting
        mask = mask.convert('L').point(lambda x: 255 if x > 128 else 0)
        mask = mask.convert('RGBA')  # Ensure mask has an alpha channel
        
        # Create a new image with the mask
        highlighted_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        highlighted_image.paste(mask, (0, 0), mask)
        image = Image.alpha_composite(image.convert('RGBA'), highlighted_image)
        image = image.convert('RGB')  # Convert back to RGB
        
        if coordinates:
            # Highlight specific spots on the image
            draw = ImageDraw.Draw(image)
            for coord in coordinates:
                x, y = map(int, coord.split(','))
                draw.ellipse([(x - 10, y - 10), (x + 10, y + 10)], outline='red', width=5)

    # Save processed image
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_image.png')
    image.save(processed_path)
    return processed_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            effect = request.form.get('effect')
            coordinates = request.form.get('coordinates')
            coordinates = coordinates.split(',') if coordinates else None
            
            if effect == 'greyscale':
                processed_image_path = process_image(filename, 'greyscale')
            elif effect == 'highlight':
                processed_image_path = process_image(filename, 'highlight', coordinates)
            else:
                processed_image_path = process_image(filename, 'predict')  # Default action if no effect is specified
            
            return send_file(processed_image_path, as_attachment=True)

    return render_template('disease.html')

if __name__ == '__main__':
    app.run(debug=True)
