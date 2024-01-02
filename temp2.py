
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/upload', methods=['POST'])
def upload():
    if 'screenshot' in request.files:
        # Retrieve the screenshot file
        screenshot_file = request.files['screenshot']

        # Read the content of the file into a bytes buffer
        screenshot_bytes = screenshot_file.read()

        # Open the image using PIL
        image = Image.open(io.BytesIO(screenshot_bytes))

        # Retrieve the text data
        text_data = request.form.get('string_data', '')

        # Process the image or text data as needed...
        # For example, save the text data, process the image, etc.

        return jsonify({'message': 'File and text received successfully!'})
    else:
        return jsonify({'error': 'Screenshot file not provided.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)


