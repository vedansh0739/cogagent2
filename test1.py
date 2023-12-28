from sat.model.mixins import CachedAutoregressiveMixin
import inspect
print(inspect.getfile(CachedAutoregressiveMixin))
from sat.quantization.kernels import quantize
print(inspect.getfile(quantize))
from sat.model import AutoModel
print(inspect.getfile(AutoModel))

import sys
print("||||||")
print("Python sys.path:")
for path in sys.path:
    print(path)
print("||||||")