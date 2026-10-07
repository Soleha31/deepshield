import flask
import whisper
import torch
import sklearn
import pandas
import numpy
import pydub
import librosa
import matplotlib

print("================================")
print("DeepShield environment is ready!")
print("================================")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())