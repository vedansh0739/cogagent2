# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Path, Input
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel

import argparse
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        max_length = 2048  # Default max_length
        top_p = 0.4  # Default top_p for nucleus sampling
        top_k = 1  # Default top_k for top k sampling
        temperature = 0.2  # Default temperature for sampling
        chinese = False  # Default for Chinese interface
        version = "chat"  # Default version of language process
        quant = None  # Default quantization bits
        from_pretrained = "cogagent-chat"  # Default pretrained checkpoint
        local_tokenizer = "lmsys/vicuna-7b-v1.5"  # Default tokenizer path
        fp16 = False  # Default fp16 setting
        bf16 = True  # Default bf16 setting
        stream_chat = True  # Default stream_chat setting

        
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))


        args=argparse.Namespace(
             max_length = 2048  # Default max_length
        top_p = 0.4  # Default top_p for nucleus sampling
        top_k = 1  # Default top_k for top k sampling
        temperature = 0.2  # Default temperature for sampling
        chinese = False  # Default for Chinese interface
        version = "chat"  # Default version of language process
        quant = None  # Default quantization bits
        from_pretrained = "cogagent-chat"  # Default pretrained checkpoint
        local_tokenizer = "lmsys/vicuna-7b-v1.5"  # Default tokenizer path
        fp16 = False  # Default fp16 setting
        bf16 = True  # Default bf16 setting
        stream_chat = True  # Default stream_chat setting


                )
                # load model
        model, model_args = AutoModel.from_pretrained(
            from_pretrained,
            args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if (torch.cuda.is_available() and quant is None) else False,
            device='cpu' if quant else 'cuda',
            max_length = 2048  # Default max_length
            top_p = 0.4  # Default top_p for nucleus sampling
            top_k = 1  # Default top_k for top k sampling
            temperature = 0.2  # Default temperature for sampling
            chinese = False  # Default for Chinese interface
            version = "chat"  # Default version of language process
            quant = None  # Default quantization bits
            from_pretrained = "cogagent-chat"  # Default pretrained checkpoint
            local_tokenizer = "lmsys/vicuna-7b-v1.5"  # Default tokenizer path
            fp16 = False  # Default fp16 setting
            bf16 = True  # Default bf16 setting
            stream_chat = True  # Default stream_chat setting
        ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
        model = model.eval()
        from sat.mpu import get_model_parallel_world_size
        assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

        language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else version
        print("[Language processor version]:", language_processor_version)
        tokenizer = llama2_tokenizer(local_tokenizer, signal_type=language_processor_version)
        image_processor = get_image_processor(model_args.eva_args["image_size"][0])
        cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
        
        if quant:
            quantize(model, quant)
            if torch.cuda.is_available():
                model = model.cuda()


        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

        text_processor_infer = llama2_text_processor_inference(tokenizer, max_length, model.image_length)
             
    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        with torch.no_grad():
            history = None
            cache_image = None
            image_path = image_path[0]


            query = query[0]
            
            response, history, cache_image = chat(
                        image_path,
                        model,
                        text_processor_infer,
                        image_processor,
                        query,
                        history=history,
                        cross_img_processor=cross_image_processor,
                        image=cache_image,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature,
                        top_k=top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        args=args
                        )
         
             
