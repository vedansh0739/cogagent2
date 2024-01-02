
from flask import Flask, request, jsonify
from PIL import Image
import io
import logging
logger = logging.getLogger(__name__)




import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from pathlib import Path
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel
import tempfile
import argparse
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel










app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/initiate')
def initiate():
    print('dope')
    return("dope")

    
@app.route('/infer')
def infer():
    if 'screenshot' in request.files:
        # Retrieve the screenshot file
        screenshot_file = request.files['screenshot']
        screenshot_file.save('utils/utils/a.jpg')
        imagepath='utils/utils/a.jpg'
        string_data = request.form.get('string_data', 'Default String if Not Provided')


        
        return jsonify({'cmd': string_data})
    else:
        return jsonify({'error': 'Screenshot file not provided.'}), 400

if __name__ == '__main__':
    app.run()


