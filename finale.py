
from flask import Flask, request, jsonify
from PIL import Image
import io



from cog import BasePredictor, Path, Input, BaseModel, File
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

import cog

class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.max_length = 2048  # Default max_length
        self.top_p = 0.4  # Default top_p for nucleus sampling
        self.top_k = 1  # Default top_k for top k sampling
        self.temperature = 0.2  # Default temperature for sampling
        self.chinese = False  # Default for Chinese interface
        self.version = "chat"  # Default version of language process
        self.quant = None  # Default quantization bits
        self.from_pretrained = "cogagent-chat"  # Default pretrained checkpoint
        self.local_tokenizer = "lmsys/vicuna-7b-v1.5"  # Default tokenizer path
        self.fp16 = False  # Default fp16 setting
        self.bf16 = True  # Default bf16 setting
        self.stream_chat = True  # Default stream_chat setting

        
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))


        self.args=argparse.Namespace(
            max_length = 2048,  # Default max_length
            top_p = 0.4,  # Default top_p for nucleus sampling
            top_k = 1,  # Default top_k for top k sampling
            temperature = 0.2,  # Default temperature for sampling
            chinese = False,  # Default for Chinese interface
            version = "chat",  # Default version of language process
            quant = None,  # Default quantization bits
            from_pretrained = "cogagent-chat",  # Default pretrained checkpoint
            local_tokenizer = "lmsys/vicuna-7b-v1.5",  # Default tokenizer path
            fp16 = False,  # Default fp16 setting
            bf16 = True,  # Default bf16 setting
            stream_chat = True  # Default stream_chat setting
        )
                # load model
        self.model, self.model_args = AutoModel.from_pretrained(
            self.from_pretrained,
            args=argparse.Namespace(
                deepspeed=None,
                local_rank=self.rank,
                rank=self.rank,
                world_size=self.world_size,
                model_parallel_size=self.world_size,
                mode='inference',
                skip_init=True,
                use_gpu_initialization=True if (torch.cuda.is_available() and self.quant is None) else False,
                device='cpu' if self.quant else 'cuda',
                max_length = 2048,  # Default max_length
                top_p = 0.4,  # Default top_p for nucleus sampling
                top_k = 1,  # Default top_k for top k sampling
                temperature = 0.2,  # Default temperature for sampling
                chinese = False,  # Default for Chinese interface
                version = "chat",  # Default version of language process
                quant = None,  # Default quantization bits
                from_pretrained = "cogagent-chat",  # Default pretrained checkpoint
                local_tokenizer = "lmsys/vicuna-7b-v1.5",  # Default tokenizer path
                fp16 = False,  # Default fp16 setting
                bf16 = True,  # Default bf16 setting
                stream_chat = True,  # Default stream_chat setting
            ), overwrite_args={'model_parallel_size': self.world_size} if self.world_size != 1 else {}
        )
        self.model = self.model.eval()
        from sat.mpu import get_model_parallel_world_size
        assert self.world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

        self.language_processor_version = self.model_args.text_processor_version if 'text_processor_version' in self.model_args else self.version
        print("[Language processor version]:", self.language_processor_version)
        self.tokenizer = llama2_tokenizer(self.local_tokenizer, signal_type=self.language_processor_version)
        self.image_processor = get_image_processor(self.model_args.eva_args["image_size"][0])
        self.cross_image_processor = get_image_processor(self.model_args.cross_image_pix) if "cross_image_pix" in self.model_args else None
        
        if self.quant:
            quantize(self.model, self.quant)
            if torch.cuda.is_available():
                self.model = self.model.cuda()


        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

        self.text_processor_infer = llama2_text_processor_inference(self.tokenizer, self.max_length, self.model.image_length)

    
    def predict(
        self,
        query,
        imagepath
    ):
        with torch.no_grad():
            self.history = None
            self.cache_image = None
            self.response, self.history, self.cache_image = chat(
                        image_path,
                        self.model,
                        self.text_processor_infer,
                        self.image_processor,
                        query,
                        history=self.history,
                        cross_img_processor=self.cross_image_processor,
                        image=image,
                        max_length=self.max_length,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        invalid_slices=self.text_processor_infer.invalid_slices,
                        args=self.args
                        )
            return{'cmd':self.response, 'img':self.cache_image}
             


app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/initiate',methods=['POST'])
def initiate():
    predictor=Predictor()


    
@app.route('/infer', methods=['POST'])
def infer():
    if 'screenshot' in request.FILES:
        # Retrieve the screenshot file
        screenshot_file = request.FILES['screenshot']
        text_data=cmd = request.POST.get('string_data', '')
        screenshot_file.save('scr.jpg')
        imagepath='scr.jpg'

        answerdict=predictor.predict(text_data,imagepath)
        return jsonify({'cmd': answerdict['cmd']})
    else:
        return jsonify({'error': 'Screenshot file not provided.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)


