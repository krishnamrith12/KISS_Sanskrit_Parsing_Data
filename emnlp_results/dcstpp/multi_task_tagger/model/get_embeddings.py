import torch
import torch.autograd as autograd
import numpy as np
import itertools
import sys
from tqdm import tqdm

from model.crf import CRFDecode_vb
from model.utils import *



class forwardPass:

    """ Do a forward pass of the model and save contextualized embeddings (hidden states)
    """

    def __init__(self, if_cuda):
        self.if_cuda = if_cuda



    def
